"""Offline video risk analyzer for badminton lower-body injury risk.

This module reads a video file, runs MediaPipe Pose on each frame, extracts
lower-body landmarks, and writes a per-frame CSV report. Optionally it can also
produce an annotated output video and/or show a live preview window.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from .baseline_risk import LowerBodyPose, Point, core_risk_score  # noqa: E402
from .injury_descriptions import describe_critical_risks  # noqa: E402
from .motion_gate import MotionGate  # noqa: E402


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

_SKELETON_LINE_THICKNESS = 3
_SKELETON_CIRCLE_RADIUS = 5
_SKELETON_CIRCLE_THICKNESS = -1  # Filled circle

# Lower-body skeleton connections drawn on the overlay.
_SKELETON_CONNECTIONS: list[tuple[str, str]] = [
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
    ("right_heel", "right_foot_index"),
    ("left_hip", "right_hip"),
]

# Popup shown on the top-right when a critical risk is detected.
_POPUP_FONT_SCALE = 0.5
_POPUP_FONT_THICKNESS = 1
_POPUP_PADDING = 8
_POPUP_LINE_SPACING = 6
_POPUP_MARGIN = 10
_POPUP_DISPLAY_SECONDS = 3.0

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


def _status_color(status: str) -> tuple[int, int, int]:
    """Return the BGR color associated with a risk status.

    Args:
        status: One of ``"acceptable"``, ``"caution"``, or ``"risky"``.

    Returns:
        An OpenCV BGR color tuple.
    """
    if status == _STATUS_ACCEPTABLE:
        return _COLOR_ACCEPTABLE
    if status == _STATUS_CAUTION:
        return _COLOR_CAUTION
    return _COLOR_RISKY


def draw_pose_overlay(
    image: Any,
    pose: LowerBodyPose,
    status: str,
    core_risk: float,
) -> None:
    """Draw a color-coded lower-body skeleton overlay on ``image``.

    Joints and bones are colored by the overall risk status:
    green for acceptable, yellow for caution, red for risky. A status
    badge is drawn in the top-left corner.

    Args:
        image: The image to draw on (modified in place). Expected to be a
            BGR OpenCV image.
        pose: A ``LowerBodyPose`` in pixel coordinates.
        status: One of ``"acceptable"``, ``"caution"``, or ``"risky"``.
        core_risk: The core risk score in ``[0.0, 1.0]``.
    """
    color = _status_color(status)
    landmarks: dict[str, Point] = {
        "left_hip": pose.left_hip,
        "right_hip": pose.right_hip,
        "left_knee": pose.left_knee,
        "right_knee": pose.right_knee,
        "left_ankle": pose.left_ankle,
        "right_ankle": pose.right_ankle,
        "left_heel": pose.left_heel,
        "right_heel": pose.right_heel,
        "left_foot_index": pose.left_foot_index,
        "right_foot_index": pose.right_foot_index,
    }

    for start_name, end_name in _SKELETON_CONNECTIONS:
        start = landmarks[start_name]
        end = landmarks[end_name]
        cv2.line(
            image,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            color,
            _SKELETON_LINE_THICKNESS,
        )

    for point in landmarks.values():
        cv2.circle(
            image,
            (int(round(point[0])), int(round(point[1]))),
            _SKELETON_CIRCLE_RADIUS,
            color,
            _SKELETON_CIRCLE_THICKNESS,
        )

    label = f"{status.upper()} {core_risk:.2f}"
    cv2.putText(
        image,
        label,
        _LABEL_ORIGIN,
        cv2.FONT_HERSHEY_SIMPLEX,
        _LABEL_FONT_SCALE,
        color,
        _LABEL_FONT_THICKNESS,
    )


def _draw_popup(
    image: Any,
    injuries: list[dict[str, object]],
    color: tuple[int, int, int],
) -> None:
    """Draw a top-right popup listing the primary injury risks.

    Args:
        image: BGR OpenCV image to draw on (modified in place).
        injuries: List of injury dictionaries from ``describe_critical_risks``.
        color: BGR color tuple for the popup text.
    """
    if not injuries:
        return

    lines = ["CRITICAL RISK"]
    for injury in injuries:
        lines.append(str(injury["name"]))
        lines.append(f"  {injury['short_description']}")

    height, width = image.shape[:2]
    max_text_width = 0
    line_height = 0
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            _POPUP_FONT_SCALE,
            _POPUP_FONT_THICKNESS,
        )
        max_text_width = max(max_text_width, text_width)
        line_height = max(line_height, text_height)

    total_text_height = line_height * len(lines) + _POPUP_LINE_SPACING * (
        len(lines) - 1
    )
    box_width = max_text_width + 2 * _POPUP_PADDING
    box_height = total_text_height + 2 * _POPUP_PADDING

    top_left = (width - box_width - _POPUP_MARGIN, _POPUP_MARGIN)
    bottom_right = (width - _POPUP_MARGIN, _POPUP_MARGIN + box_height)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)

    y = top_left[1] + _POPUP_PADDING + line_height
    for line in lines:
        cv2.putText(
            image,
            line,
            (top_left[0] + _POPUP_PADDING, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            _POPUP_FONT_SCALE,
            color,
            _POPUP_FONT_THICKNESS,
        )
        y += line_height + _POPUP_LINE_SPACING


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


def _try_video_writer(
    path: str,
    fourcc_code: str,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter | None:
    """Try to open a ``cv2.VideoWriter`` with the requested codec.

    Returns the writer if it opens successfully, otherwise ``None`` (after
    releasing the failed writer). This lets callers try H.264 first and fall
    back to a more widely available codec without raising an error.
    """
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    if writer.isOpened():
        return writer
    writer.release()
    return None


def analyze_video(
    input_path: str | None,
    output_csv: str | None,
    output_video: str | None,
    show_preview: bool = False,
    pose_detector: PoseDetector | None = None,
    webcam_index: int | None = None,
    output_log: str | None = None,
) -> None:
    """Analyze a video and write a per-frame risk CSV report.

    Args:
        input_path: Path to the input video file. Optional when using a
            webcam via ``webcam_index``.
        output_csv: Path where the CSV report will be written. Optional in
            webcam mode; if ``None`` no CSV is produced.
        output_video: Optional path where an annotated output video will be
            written. If ``None``, no video is produced.
        show_preview: If ``True``, display a live preview window for file
            input. Press ``q`` to quit early.
        pose_detector: Optional pose detector to use. The object must satisfy
            the ``PoseDetector`` protocol (``process`` / ``close``). If
            ``None``, a default MediaPipe Pose detector is created via
            ``_create_pose_detector()``.
        webcam_index: If provided, open the webcam at this index instead of
            ``input_path``. Webcam mode always shows a preview window.
        output_log: Optional path to a JSON log of critical risk detections.
            If ``None``, no log is produced.

    Raises:
        RuntimeError: If the input video/webcam or output video writer cannot
            be opened.
    """
    if webcam_index is not None:
        cap = cv2.VideoCapture(webcam_index)
        source_name = f"webcam at index {webcam_index}"
        is_webcam = True
    else:
        cap = cv2.VideoCapture(input_path)
        source_name = f"video: {input_path}"
        is_webcam = False

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source_name}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_webcam:
        if width <= 0 or height <= 0:
            width, height = 640, 480
        display_window = True
    else:
        display_window = show_preview

    writer: cv2.VideoWriter | None = None
    if output_video:
        writer = _try_video_writer(output_video, "avc1", fps, (width, height))
        if writer is None:
            print(
                f"H.264 codec not available for {output_video}; "
                "falling back to mp4v.",
                file=sys.stderr,
            )
            writer = _try_video_writer(output_video, "mp4v", fps, (width, height))
        if writer is None:
            cap.release()
            raise RuntimeError(f"Could not open video writer: {output_video}")

    pose = pose_detector if pose_detector is not None else _create_pose_detector()

    gate = MotionGate(window_seconds=_MOTION_GATE_WINDOW_SECONDS, fps=fps)
    results: list[dict[str, Any]] | None = [] if output_csv is not None else None
    frame_idx = 0

    in_critical_segment = False
    popup_frames_remaining = 0
    popup_injuries: list[dict[str, object]] = []
    critical_events: list[dict[str, Any]] = []

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
                    if row["status"] == _STATUS_RISKY:
                        if not in_critical_segment:
                            in_critical_segment = True
                            popup_injuries = describe_critical_risks(score)
                            popup_frames_remaining = int(
                                round(fps * _POPUP_DISPLAY_SECONDS)
                            )
                            critical_events.append(
                                {
                                    "frame": frame_idx,
                                    "time_sec": row["time_sec"],
                                    "injuries": popup_injuries,
                                }
                            )
                    else:
                        in_critical_segment = False
                else:
                    in_critical_segment = False
            else:
                # No pose detected: reset any active critical segment.
                in_critical_segment = False

            if results is not None:
                results.append(row)

            if writer is not None or display_window:
                display = frame.copy()
                if pose_obj is not None:
                    draw_pose_overlay(
                        display,
                        pose_obj,
                        row["status"],
                        row["core_risk"],
                    )
                    if popup_frames_remaining > 0:
                        _draw_popup(display, popup_injuries, _COLOR_RISKY)
                        popup_frames_remaining -= 1
                        if popup_frames_remaining == 0:
                            popup_injuries = []
                else:
                    # No pose detected: show a neutral status badge only.
                    cv2.putText(
                        display,
                        "NO POSE",
                        _LABEL_ORIGIN,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        _LABEL_FONT_SCALE,
                        (128, 128, 128),
                        _LABEL_FONT_THICKNESS,
                    )
                if writer is not None:
                    writer.write(display)
                if display_window:
                    cv2.imshow("Risk Analyzer", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        pose.close()
        if display_window:
            cv2.destroyAllWindows()

    if output_csv is not None:
        rows = build_csv_rows(results)
        if rows:
            with open(output_csv, "w", newline="") as f:
                writer_csv = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer_csv.writeheader()
                writer_csv.writerows(rows)

    if output_log is not None:
        with open(output_log, "w", encoding="utf-8") as f:
            json.dump(critical_events, f, indent=2)


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
    parser.add_argument(
        "--output-log",
        default=None,
        help="Path to a JSON log of critical risk detections.",
    )
    args = parser.parse_args(argv)

    if args.webcam is None and args.input_video is None:
        parser.error("Either input_video or --webcam is required.")
    if args.webcam is not None and args.input_video is not None:
        parser.error("Cannot specify both input_video and --webcam.")

    if args.output_csv is not None:
        output_csv = args.output_csv
    elif args.webcam is None:
        output_csv = DEFAULT_OUTPUT_CSV
    else:
        output_csv = None

    analyze_video(
        args.input_video,
        output_csv,
        args.output_video,
        args.show_preview,
        webcam_index=args.webcam,
        output_log=args.output_log,
    )
    if output_csv:
        print(f"Wrote CSV report to {output_csv}")
    if args.output_video:
        print(f"Wrote annotated video to {args.output_video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
