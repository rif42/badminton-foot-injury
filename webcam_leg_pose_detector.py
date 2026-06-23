"""Live webcam lower-body pose detector using MediaPipe Pose and OpenCV.

This module provides a reusable PoseDetector class that wraps MediaPipe Pose
initialization and landmark drawing, plus a main() function that drives the
webcam capture loop. It intentionally contains no injury-risk scoring logic;
landmark extraction is exposed so follow-up modules can consume it directly.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe's results.pose_landmarks is a NormalizedLandmarkList protobuf.
# Expose a stable alias so callers do not depend on internal module paths.
LandmarkList = Any

__all__ = [
    "LOWER_BODY_CONNECTIONS",
    "LOWER_BODY_LANDMARKS",
    "LandmarkList",
    "PoseDetector",
    "PoseDetectorConfig",
    "main",
]


# MediaPipe Pose landmark indices for the lower body.
LOWER_BODY_LANDMARKS: Dict[str, int] = {
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

# Bilateral lower-body connections drawn as the skeleton overlay.
LOWER_BODY_CONNECTIONS: List[Tuple[int, int]] = [
    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
]

# Default drawing colors (BGR).
LANDMARK_COLOR: Tuple[int, int, int] = (0, 255, 0)
CONNECTION_COLOR: Tuple[int, int, int] = (0, 200, 0)
LANDMARK_RADIUS: int = 5
CONNECTION_THICKNESS: int = 2
VISIBILITY_THRESHOLD: float = 0.5

MAX_FRAME_RETRIES: int = 3
FRAME_RETRY_DELAY_SECONDS: float = 0.05


@dataclass
class PoseDetectorConfig:
    """Optional runtime configuration for PoseDetector."""

    camera_index: int = 0
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    model_complexity: int = 0


class PoseDetector:
    """Wraps MediaPipe Pose initialization, frame processing, and lower-body drawing."""

    def __init__(self, config: Optional[PoseDetectorConfig] = None) -> None:
        self.config = config if config is not None else PoseDetectorConfig()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.detection_confidence,
            min_tracking_confidence=self.config.tracking_confidence,
        )

    def process(
        self, frame: np.ndarray
    ) -> Optional[LandmarkList]:
        """Run pose detection on a BGR frame and return raw landmarks if found."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results.pose_landmarks if results.pose_landmarks else None

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: LandmarkList,
        landmark_color: Tuple[int, int, int] = LANDMARK_COLOR,
        connection_color: Tuple[int, int, int] = CONNECTION_COLOR,
    ) -> np.ndarray:
        """Overlay lower-body landmarks and connections onto the frame."""
        height, width, _ = frame.shape

        def _point(index: int) -> Optional[Tuple[int, int]]:
            lm = landmarks.landmark[index]
            if lm.visibility < VISIBILITY_THRESHOLD:
                return None
            return int(lm.x * width), int(lm.y * height)

        for start_idx, end_idx in LOWER_BODY_CONNECTIONS:
            start_point = _point(start_idx)
            end_point = _point(end_idx)
            if start_point is not None and end_point is not None:
                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    connection_color,
                    CONNECTION_THICKNESS,
                )

        for idx in LOWER_BODY_LANDMARKS.values():
            point = _point(idx)
            if point is not None:
                cv2.circle(frame, point, LANDMARK_RADIUS, landmark_color, -1)

        return frame

    @staticmethod
    def get_lower_body_landmarks(
        landmarks: LandmarkList,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """Return a dictionary of {joint_name: (x, y, z, visibility)} for the lower body.

        Note:
            If the frame was horizontally flipped for mirror display before
            detection (as ``main()`` does), the ``left_*`` / ``right_*`` keys
            are relative to the mirrored image (screen left/right), not the
            physical body sides of the person in front of the camera.
        """
        result: Dict[str, Tuple[float, float, float, float]] = {}
        for name, idx in LOWER_BODY_LANDMARKS.items():
            lm = landmarks.landmark[idx]
            result[name] = (lm.x, lm.y, lm.z, lm.visibility)
        return result

    def close(self) -> None:
        """Release MediaPipe Pose resources."""
        self.pose.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()


def main(config: Optional[PoseDetectorConfig] = None) -> int:
    """Open the webcam, run lower-body pose detection, and display the annotated feed.

    Args:
        config: Optional PoseDetectorConfig to override defaults. The camera
            index is read from this config.
    """
    camera_index = config.camera_index if config is not None else 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}.")
        cap.release()
        cv2.destroyAllWindows()
        return 1

    detector = PoseDetector(config)
    consecutive_frame_failures = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_frame_failures += 1
                print(
                    f"Warning: Failed to read frame "
                    f"({consecutive_frame_failures}/{MAX_FRAME_RETRIES})."
                )
                if consecutive_frame_failures >= MAX_FRAME_RETRIES:
                    print("Error: Exceeded maximum frame read retries; exiting.")
                    return 1
                time.sleep(FRAME_RETRY_DELAY_SECONDS)
                continue

            consecutive_frame_failures = 0
            # Mirror the frame for a more natural selfie-style display.
            # Detection therefore runs on the mirrored image, so left/right
            # landmark keys are screen-relative.
            frame = cv2.flip(frame, 1)
            landmarks = detector.process(frame)

            if landmarks is not None:
                detector.draw_landmarks(frame, landmarks)

            cv2.imshow("Lower-Body Pose Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
