"""Unit tests for webcam_leg_pose_detector.py.

Tests avoid using a real webcam or MediaPipe model by mocking OpenCV and
MediaPipe objects. They verify initialization, frame processing, landmark
drawing, extraction helpers, and error handling.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import webcam_leg_pose_detector as detector_module
from webcam_leg_pose_detector import (
    LOWER_BODY_CONNECTIONS,
    LOWER_BODY_LANDMARKS,
    PoseDetector,
    PoseDetectorConfig,
    main,
)


class TestPoseDetector(unittest.TestCase):
    """Tests for the PoseDetector class."""

    def _make_mock_landmarks(self):
        """Build a mock MediaPipe landmarks object with 33 landmarks."""
        landmarks = MagicMock()
        # MediaPipe Pose returns 33 landmarks indexed 0-32.
        landmarks.landmark = {}
        for idx in range(33):
            lm = MagicMock()
            lm.x = 0.0
            lm.y = 0.0
            lm.z = 0.0
            lm.visibility = 1.0
            landmarks.landmark[idx] = lm
        for idx in LOWER_BODY_LANDMARKS.values():
            lm = landmarks.landmark[idx]
            lm.x = idx / 100.0
            lm.y = idx / 100.0
        return landmarks

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    def test_default_initialization(self, mock_pose_cls):
        """PoseDetector initializes with sensible defaults."""
        detector = PoseDetector()

        self.assertEqual(detector.config.camera_index, 0)
        self.assertEqual(detector.config.detection_confidence, 0.5)
        self.assertEqual(detector.config.tracking_confidence, 0.5)
        self.assertEqual(detector.config.model_complexity, 0)
        mock_pose_cls.assert_called_once_with(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    def test_custom_initialization(self, mock_pose_cls):
        """PoseDetector accepts optional configuration."""
        config = PoseDetectorConfig(
            camera_index=1,
            detection_confidence=0.7,
            tracking_confidence=0.7,
            model_complexity=1,
        )
        detector = PoseDetector(config)

        self.assertEqual(detector.config.camera_index, 1)
        self.assertEqual(detector.config.detection_confidence, 0.7)
        self.assertEqual(detector.config.tracking_confidence, 0.7)
        self.assertEqual(detector.config.model_complexity, 1)
        mock_pose_cls.assert_called_once_with(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    @patch.object(detector_module.cv2, "cvtColor")
    def test_process_converts_bgr_to_rgb_and_returns_landmarks(
        self, mock_cvtColor, mock_pose_cls
    ):
        """process() converts the frame to RGB and returns detected landmarks."""
        mock_results = MagicMock()
        mock_landmarks = self._make_mock_landmarks()
        mock_results.pose_landmarks = mock_landmarks

        mock_pose_instance = MagicMock()
        mock_pose_instance.process.return_value = mock_results
        mock_pose_cls.return_value = mock_pose_instance

        detector = PoseDetector()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = detector.process(fake_frame)

        mock_cvtColor.assert_called_once_with(
            fake_frame, detector_module.cv2.COLOR_BGR2RGB
        )
        mock_pose_instance.process.assert_called_once_with(mock_cvtColor.return_value)
        self.assertEqual(output, mock_landmarks)

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    @patch.object(detector_module.cv2, "cvtColor")
    def test_process_returns_none_when_no_landmarks(
        self, mock_cvtColor, mock_pose_cls
    ):
        """process() returns None when pose detection finds no landmarks."""
        mock_results = MagicMock()
        mock_results.pose_landmarks = None

        mock_pose_instance = MagicMock()
        mock_pose_instance.process.return_value = mock_results
        mock_pose_cls.return_value = mock_pose_instance

        detector = PoseDetector()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = detector.process(fake_frame)

        self.assertIsNone(output)

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    @patch.object(detector_module.cv2, "circle")
    @patch.object(detector_module.cv2, "line")
    def test_draw_landmarks_draws_lower_body_points_and_connections(
        self, mock_line, mock_circle, mock_pose_cls
    ):
        """draw_landmarks() draws each lower-body landmark and connection."""
        detector = PoseDetector()
        landmarks = self._make_mock_landmarks()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        output = detector.draw_landmarks(fake_frame, landmarks)

        self.assertIs(output, fake_frame)
        self.assertEqual(
            mock_circle.call_count, len(LOWER_BODY_LANDMARKS)
        )
        self.assertEqual(
            mock_line.call_count, len(LOWER_BODY_CONNECTIONS)
        )

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    @patch.object(detector_module.cv2, "circle")
    @patch.object(detector_module.cv2, "line")
    def test_draw_landmarks_skips_low_visibility_points(
        self, mock_line, mock_circle, mock_pose_cls
    ):
        """draw_landmarks() does not draw landmarks below the visibility threshold."""
        detector = PoseDetector()
        landmarks = self._make_mock_landmarks()
        # Hide every landmark.
        for lm in landmarks.landmark.values():
            lm.visibility = 0.0

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.draw_landmarks(fake_frame, landmarks)

        mock_circle.assert_not_called()
        mock_line.assert_not_called()

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    @patch.object(detector_module.cv2, "circle")
    @patch.object(detector_module.cv2, "line")
    def test_draw_landmarks_skips_missing_indices(
        self, mock_line, mock_circle, mock_pose_cls
    ):
        """draw_landmarks() skips missing landmark indices without raising."""
        detector = PoseDetector()
        landmarks = self._make_mock_landmarks()
        # Remove a few landmarks to simulate a partial result.
        del landmarks.landmark[23]
        del landmarks.landmark[25]
        del landmarks.landmark[27]

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.draw_landmarks(fake_frame, landmarks)

        # No lines or circles should be drawn for the missing left leg,
        # but the right leg landmarks are still drawn (plus 29-31 on the left).
        expected_circles = len(LOWER_BODY_LANDMARKS) - 3
        self.assertEqual(mock_circle.call_count, expected_circles)
        self.assertEqual(mock_line.call_count, 6)

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    def test_get_lower_body_landmarks_format(self, mock_pose_cls):
        """get_lower_body_landmarks() returns the expected dictionary format."""
        detector = PoseDetector()
        landmarks = self._make_mock_landmarks()

        result = detector.get_lower_body_landmarks(landmarks)

        self.assertEqual(set(result.keys()), set(LOWER_BODY_LANDMARKS.keys()))
        for name, idx in LOWER_BODY_LANDMARKS.items():
            self.assertEqual(
                result[name], (idx / 100.0, idx / 100.0, 0.0, 1.0)
            )

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    def test_context_manager_closes_pose(self, mock_pose_cls):
        """PoseDetector can be used as a context manager and closes on exit."""
        mock_instance = MagicMock()
        mock_pose_cls.return_value = mock_instance

        with PoseDetector() as detector:
            self.assertIsInstance(detector, PoseDetector)

        mock_instance.close.assert_called_once()


class TestMain(unittest.TestCase):
    """Tests for the main() capture loop."""

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    @patch.object(detector_module.cv2, "destroyAllWindows")
    @patch.object(detector_module.cv2, "VideoCapture")
    def test_main_exits_when_camera_fails_to_open(
        self, mock_video_capture, mock_destroy, mock_pose_cls
    ):
        """main() prints an error and exits immediately if the camera cannot open."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        with patch("builtins.print") as mock_print:
            exit_code = main()

        self.assertEqual(exit_code, 1)
        mock_cap.release.assert_called_once()
        mock_destroy.assert_not_called()
        mock_pose_cls.assert_not_called()
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Could not open camera", printed)

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    @patch.object(detector_module.cv2, "destroyAllWindows")
    @patch.object(detector_module.cv2, "waitKey", return_value=ord("q"))
    @patch.object(detector_module.cv2, "imshow")
    @patch.object(detector_module.cv2, "flip", side_effect=lambda f, _: f)
    @patch.object(detector_module.cv2, "VideoCapture")
    def test_main_uses_config_camera_index(
        self, mock_video_capture, mock_flip, mock_imshow, mock_waitKey, mock_destroy, mock_pose_cls
    ):
        """main() uses the camera index from an optional PoseDetectorConfig."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_video_capture.return_value = mock_cap

        config = PoseDetectorConfig(camera_index=2)
        with patch.object(PoseDetector, "process", return_value=None):
            exit_code = main(config)

        self.assertEqual(exit_code, 0)
        mock_video_capture.assert_called_once_with(2)

    @patch.object(detector_module.cv2, "destroyAllWindows")
    @patch.object(detector_module.cv2, "waitKey", return_value=ord("q"))
    @patch.object(detector_module.cv2, "imshow")
    @patch.object(detector_module.cv2, "flip", side_effect=lambda f, _: f)
    @patch.object(detector_module.cv2, "VideoCapture")
    def test_main_successful_frame_loop(
        self, mock_video_capture, mock_flip, mock_imshow, mock_waitKey, mock_destroy
    ):
        """main() reads frames, flips them, detects pose, and displays the result."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_video_capture.return_value = mock_cap

        mock_landmarks = MagicMock()
        mock_results = MagicMock()
        mock_results.pose_landmarks = mock_landmarks

        mock_pose_instance = MagicMock()
        mock_pose_instance.process.return_value = mock_results

        with patch.object(
            detector_module.mp.solutions.pose, "Pose", return_value=mock_pose_instance
        ):
            with patch.object(
                PoseDetector, "draw_landmarks", return_value=fake_frame
            ) as mock_draw:
                exit_code = main()

        self.assertEqual(exit_code, 0)
        mock_flip.assert_called_once_with(fake_frame, 1)
        mock_imshow.assert_called_once()
        mock_draw.assert_called_once()
        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()

    @patch.object(detector_module.mp.solutions.pose, "Pose")
    @patch.object(detector_module.cv2, "destroyAllWindows")
    @patch.object(detector_module.cv2, "waitKey", return_value=ord("q"))
    @patch.object(detector_module.cv2, "imshow")
    @patch.object(detector_module.cv2, "flip", side_effect=lambda f, _: f)
    @patch.object(detector_module.cv2, "VideoCapture")
    def test_main_shows_original_frame_when_no_pose(
        self, mock_video_capture, mock_flip, mock_imshow, mock_waitKey, mock_destroy, mock_pose_cls
    ):
        """main() shows the original flipped frame when no pose is detected."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_video_capture.return_value = mock_cap

        with patch.object(PoseDetector, "process", return_value=None) as mock_process:
            with patch.object(PoseDetector, "draw_landmarks") as mock_draw:
                exit_code = main()

        self.assertEqual(exit_code, 0)
        mock_process.assert_called_once_with(fake_frame)
        mock_draw.assert_not_called()
        mock_flip.assert_called_once_with(fake_frame, 1)
        mock_imshow.assert_called_once_with("Lower-Body Pose Detector", fake_frame)
        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()

    @patch.object(detector_module.cv2, "destroyAllWindows")
    @patch.object(detector_module.cv2, "waitKey", return_value=ord("q"))
    @patch.object(detector_module.cv2, "imshow")
    @patch.object(detector_module.cv2, "VideoCapture")
    @patch("time.sleep")
    def test_main_retries_frame_read_failures(
        self, mock_sleep, mock_video_capture, mock_imshow, mock_waitKey, mock_destroy
    ):
        """main() retries a small number of frame read failures before exiting."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Fail one fewer than the retry limit, then succeed and quit.
        mock_cap.read.side_effect = [
            (False, None),
            (False, None),
            (True, fake_frame),
        ]
        mock_video_capture.return_value = mock_cap

        with patch.object(detector_module.mp.solutions.pose, "Pose"):
            with patch.object(PoseDetector, "process", return_value=None):
                with patch("builtins.print") as mock_print:
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_cap.read.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Failed to read frame", printed)

    @patch.object(detector_module.cv2, "destroyAllWindows")
    @patch.object(detector_module.cv2, "waitKey", return_value=ord("q"))
    @patch.object(detector_module.cv2, "imshow")
    @patch.object(detector_module.cv2, "VideoCapture")
    @patch("time.sleep")
    def test_main_exits_after_max_frame_retries(
        self, mock_sleep, mock_video_capture, mock_imshow, mock_waitKey, mock_destroy
    ):
        """main() exits when frame read failures exceed the retry limit."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        with patch.object(detector_module.mp.solutions.pose, "Pose"):
            with patch.object(PoseDetector, "process", return_value=None):
                with patch("builtins.print") as mock_print:
                    exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertEqual(mock_cap.read.call_count, detector_module.MAX_FRAME_RETRIES)
        self.assertEqual(mock_sleep.call_count, detector_module.MAX_FRAME_RETRIES - 1)
        mock_imshow.assert_not_called()
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Exceeded maximum frame read retries", printed)


if __name__ == "__main__":
    unittest.main()
