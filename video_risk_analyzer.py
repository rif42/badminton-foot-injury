"""Offline video risk analyzer for badminton lower-body injury risk.

This module reads a video file, runs MediaPipe Pose on each frame, extracts
lower-body landmarks, and writes a per-frame CSV report. Optionally it can also
produce an annotated output video and/or show a live preview window.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Protocol, runtime_checkable

import cv2

# Suppress verbose MediaPipe / TensorFlow Lite runtime logging.
# These must be set before importing mediapipe to take effect.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import absl.logging  # isort: split

absl.logging.set_verbosity(absl.logging.ERROR)

import mediapipe as mp  # noqa: E402

from baseline_risk import LowerBodyPose, Point, core_risk_score  # noqa: E402
from motion_gate import MotionGate  # noqa: E402


@runtime_checkable
class PoseDetector(Protocol):
    """Minimal protocol for a pose detector compatible with MediaPipe Pose.

    Callers may inject any object that provides ``process`` and ``close`` with
    these signatures, which keeps tests free of the MediaPipe import.
    """

    def process(self, image: Any) -> Any:
        """Process an RGB image and return a result with ``pose_landmarks``."""
        ...

    def close(self) -> None:
        """Release detector resources."""
        ...


LOWER_BODY_INDICES: dict[str, int] = {
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

_STATUS_ACCEPTABLE = "acceptable"
_STATUS_CAUTION = "caution"
_STATUS_RISKY = "risky"

_CORE_RISK_CAUTION_THRESHOLD = 0.35
_CORE_RISK_RISKY_THRESHOLD = 0.65

_DEFAULT_MODEL_COMPLEXITY = 0
_DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
_DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5

_MOTION_GATE_WINDOW_SECONDS = 0.3

_COLOR_ACCEPTABLE = (0, 255, 0)
_COLOR_CAUTION = (0, 255, 255)
_COLOR_RISKY = (0, 0, 255)

_LABEL_ORIGIN = (10, 30)
_LABEL_FONT_SCALE = 0.7
_LABEL_FONT_THICKNESS = 2

DEFAULT_OUTPUT_CSV = "risk_report.csv"


def _create_pose_detector() -> PoseDetector:
    """Create and return a default MediaPipe Pose detector.

    This helper is separated from ``analyze_video`` so that tests and other
    callers can inject a mock or alternative detector without importing
    MediaPipe directly.

    Returns:
        A MediaPipe ``Pose`` instance configured for video streams.
    """
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=_DEFAULT_MODEL_COMPLEXITY,
        min_detection_confidence=_DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=_DEFAULT_MIN_TRACKING_CONFIDENCE,
    )


def _landmark_to_point(landmarks: Any, index: int, image_shape: tuple[int, ...]) -> Point:
    """Convert a normalized MediaPipe landmark to an image-space ``Point``.

    Args:
        landmarks: A MediaPipe ``NormalizedLandmarkList``.
        index: The landmark index to convert.
        image_shape: The frame shape as returned by ``numpy.ndarray.shape``.

    Returns:
        A 3-D point ``(x, y, z)`` in pixel coordinates.
    """
    lm = landmarks.landmark[index]
    return (
        lm.x * image_shape[1],
        lm.y * image_shape[0],
        lm.z * image_shape[1],
    )


def _extract_pose(landmarks: Any, image_shape: tuple[int, ...]) -> LowerBodyPose | None:
    """Build a ``LowerBodyPose`` from MediaPipe landmarks if all are present.

    Args:
        landmarks: A MediaPipe ``NormalizedLandmarkList``, or ``None``.
        image_shape: The frame shape as returned by ``numpy.ndarray.shape``.

    Returns:
        A ``LowerBodyPose`` in pixel coordinates, or ``None`` if landmarks are
        missing or incomplete.
    """
    if landmarks is None:
        return None
    try:
        kwargs = {
            name: _landmark_to_point(landmarks, idx, image_shape)
            for name, idx in LOWER_BODY_INDICES.items()
        }
        return LowerBodyPose(**kwargs)
    except (IndexError, KeyError, AttributeError):
        return None


def _status(core_risk: float) -> str:
    """Map a core risk score to a categorical status label.

    Args:
        core_risk: Core risk score in ``[0.0, 1.0]``.

    Returns:
        ``"acceptable"``, ``"caution"``, or ``"risky"``.
    """
    if core_risk < _CORE_RISK_CAUTION_THRESHOLD:
        return _STATUS_ACCEPTABLE
    if core_risk < _CORE_RISK_RISKY_THRESHOLD:
        return _STATUS_CAUTION
    return _STATUS_RISKY


def build_csv_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert analysis result dictionaries to CSV-ready string rows.

    Args:
        results: List of per-frame result dictionaries.

    Returns:
        List of dictionaries with string values and a fixed set of keys.
        Returns an empty list if ``results`` is empty.
    """
    fieldnames = [
        "frame",
        "time_sec",
        "is_moving",
        "knee_stiffness_risk",
        "ankle_foot_alignment_risk",
        "hip_displacement_proxy",
        "landing_asymmetry_score",
        "core_risk",
        "status",
    ]
    return [
        {key: str(row.get(key, "")) for key in fieldnames}
        for row in results
    ]


def analyze_video(
    input_path: str,
    output_csv: str,
    output_video: str | None,
    show_preview: bool = False,
    pose_detector: PoseDetector | None = None,
    **kwargs: Any,
) -> None:
    """Analyze a video and write a per-frame risk CSV report.

    Args:
        input_path: Path to the input video file.
        output_csv: Path where the CSV report will be written.
        output_video: Optional path where an annotated output video will be
            written. If ``None``, no video is produced.
        show_preview: If ``True``, display a live preview window. Press ``q`` to
            quit early.
        pose_detector: Optional pose detector to use. The object must satisfy
            the ``PoseDetector`` protocol (``process`` / ``close``). If
            ``None``, a default MediaPipe Pose detector is created via
            ``_create_pose_detector()``.

    Raises:
        RuntimeError: If the input video or output video writer cannot be
            opened.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: cv2.VideoWriter | None = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if writer is not None and not writer.isOpened():
            writer.release()
            cap.release()
            raise RuntimeError(f"Could not open video writer: {output_video}")

    pose = pose_detector if pose_detector is not None else _create_pose_detector()

    gate = MotionGate(window_seconds=_MOTION_GATE_WINDOW_SECONDS, fps=fps)
    results: list[dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_result = pose.process(rgb)
            pose_obj = _extract_pose(mp_result.pose_landmarks, frame.shape)

            row: dict[str, Any] = {
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 3),
                "is_moving": False,
                "knee_stiffness_risk": 0.0,
                "ankle_foot_alignment_risk": 0.0,
                "hip_displacement_proxy": 0.0,
                "landing_asymmetry_score": 0.0,
                "core_risk": 0.0,
                "status": _STATUS_ACCEPTABLE,
            }

            if pose_obj is not None:
                hip_center = (
                    (pose_obj.left_hip[0] + pose_obj.right_hip[0]) / 2.0,
                    (pose_obj.left_hip[1] + pose_obj.right_hip[1]) / 2.0,
                    (pose_obj.left_hip[2] + pose_obj.right_hip[2]) / 2.0,
                )
                leg_length = (
                    abs(pose_obj.left_hip[1] - pose_obj.left_ankle[1])
                    + abs(pose_obj.right_hip[1] - pose_obj.right_ankle[1])
                ) / 2.0
                state = gate.update(hip_center, leg_length)
                row["is_moving"] = state == "moving"

                if state == "moving":
                    score = core_risk_score(pose_obj)
                    row.update(score)
                    row["status"] = _status(score["core_risk"])

            results.append(row)

            if writer is not None or show_preview:
                display = frame.copy()
                if row["status"] == _STATUS_ACCEPTABLE:
                    color = _COLOR_ACCEPTABLE
                elif row["status"] == _STATUS_CAUTION:
                    color = _COLOR_CAUTION
                else:
                    color = _COLOR_RISKY
                label = f"{row['status'].upper()} {row['core_risk']:.2f}"
                cv2.putText(
                    display,
                    label,
                    _LABEL_ORIGIN,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    _LABEL_FONT_SCALE,
                    color,
                    _LABEL_FONT_THICKNESS,
                )
                if writer is not None:
                    writer.write(display)
                if show_preview:
                    cv2.imshow("Risk Analyzer", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        pose.close()
        if show_preview:
            cv2.destroyAllWindows()

    rows = build_csv_rows(results)
    if not rows:
        return

    with open(output_csv, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the offline video risk analyzer.

    Args:
        argv: Optional argument list. If ``None``, ``sys.argv`` is used.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Offline and live webcam badminton lower-body risk analyzer."
    )
    parser.add_argument(
        "input_video",
        nargs="?",
        default=None,
        help="Path to input video (optional if --webcam is used).",
    )
    parser.add_argument(
        "--webcam",
        nargs="?",
        const=0,
        type=int,
        default=None,
        help="Use live webcam input (optional index, default 0).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Path to optional annotated output video.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show live preview window.",
    )
    args = parser.parse_args(argv)

    if args.webcam is None and args.input_video is None:
        parser.error("Either input_video or --webcam is required.")
    if args.webcam is not None and args.input_video is not None:
        parser.error("Cannot specify both input_video and --webcam.")

    output_csv = args.output_csv if args.output_csv is not None else (DEFAULT_OUTPUT_CSV if args.webcam is None else None)

    analyze_video(
        args.input_video,
        output_csv,
        args.output_video,
        args.show_preview,
        webcam_index=args.webcam,
    )
    if output_csv:
        print(f"Wrote CSV report to {output_csv}")
    if args.output_video:
        print(f"Wrote annotated video to {args.output_video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
