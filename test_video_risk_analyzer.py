"""Unit tests for the offline video risk analyzer."""

from __future__ import annotations

import csv
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from baseline_risk import LowerBodyPose
from video_risk_analyzer import (
    _extract_pose,
    _landmark_to_point,
    _status,
    analyze_video,
    build_csv_rows,
    main,
)


def _make_landmark(x: float, y: float, z: float) -> MagicMock:
    """Return a mock landmark with normalized x/y/z coordinates."""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _make_landmarks(coords: dict[int, tuple[float, float, float]]) -> MagicMock:
    """Return a mock landmarks container with the given index -> coordinate map."""
    landmarks = MagicMock()
    landmarks.landmark = {
        idx: _make_landmark(x, y, z) for idx, (x, y, z) in coords.items()
    }
    return landmarks


def _symmetric_pose_landmarks() -> dict[int, tuple[float, float, float]]:
    """Return a plausible symmetric lower-body pose in normalized coordinates."""
    return {
        # Left side
        23: (0.45, 0.30, 0.0),  # left_hip
        25: (0.45, 0.50, 0.0),  # left_knee
        27: (0.45, 0.75, 0.0),  # left_ankle
        29: (0.43, 0.82, 0.0),  # left_heel
        31: (0.47, 0.82, 0.05),  # left_foot_index
        # Right side
        24: (0.55, 0.30, 0.0),  # right_hip
        26: (0.55, 0.50, 0.0),  # right_knee
        28: (0.55, 0.75, 0.0),  # right_ankle
        30: (0.53, 0.82, 0.0),  # right_heel
        32: (0.57, 0.82, 0.05),  # right_foot_index
    }


def test_build_csv_rows():
    results = [
        {
            "frame": 0,
            "time_sec": 0.0,
            "is_moving": False,
            "knee_stiffness_risk": 0.0,
            "ankle_foot_alignment_risk": 0.0,
            "hip_displacement_proxy": 0.0,
            "landing_asymmetry_score": 0.0,
            "core_risk": 0.0,
            "status": "acceptable",
        }
    ]
    rows = build_csv_rows(results)
    assert rows[0]["frame"] == "0"
    assert rows[0]["status"] == "acceptable"


def test_status_acceptable():
    assert _status(0.0) == "acceptable"
    assert _status(0.34) == "acceptable"


def test_status_caution():
    assert _status(0.35) == "caution"
    assert _status(0.64) == "caution"


def test_status_risky():
    assert _status(0.65) == "risky"
    assert _status(1.0) == "risky"


def test_landmark_to_point():
    landmarks = _make_landmarks({0: (0.5, 0.5, 0.1)})
    point = _landmark_to_point(landmarks, 0, (480, 640, 3))
    assert point == (320.0, 240.0, 64.0)


def test_extract_pose_returns_lower_body_pose():
    landmarks = _make_landmarks(_symmetric_pose_landmarks())
    pose = _extract_pose(landmarks, (480, 640, 3))
    assert isinstance(pose, LowerBodyPose)
    assert pose.left_hip == (0.45 * 640, 0.30 * 480, 0.0 * 640)


def test_extract_pose_returns_none_when_landmarks_missing():
    assert _extract_pose(None, (480, 640, 3)) is None


def test_extract_pose_returns_none_when_index_missing():
    landmarks = _make_landmarks({23: (0.5, 0.5, 0.0)})
    assert _extract_pose(landmarks, (480, 640, 3)) is None


def test_analyze_video_outputs_csv(tmp_path):
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output_csv = tmp_path / "out.csv"

    mock_pose = MagicMock()
    mock_pose.process.return_value.pose_landmarks = None
    mock_pose.close = MagicMock()

    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [
            (True, fake_frame),
            (True, fake_frame),
            (False, None),
        ]
        mock_cap_instance.get.side_effect = lambda key: (
            30.0 if key == cv2.CAP_PROP_FPS else 640.0
        )
        mock_cap.return_value = mock_cap_instance

        analyze_video(
            str(tmp_path / "fake.mp4"),
            str(output_csv),
            None,
            show_preview=False,
            pose_detector=mock_pose,
        )

    assert output_csv.exists()
    mock_pose.close.assert_called_once()


def test_analyze_video_real_pose_path_writes_risk_csv(tmp_path):
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output_csv = tmp_path / "out.csv"

    def _make_moving_landmarks(frame_idx: int):
        """Return landmarks whose hip center moves horizontally each frame."""
        coords = _symmetric_pose_landmarks()
        # Shift both hips to the right by a few pixels per frame.
        offset = frame_idx * 0.01
        for idx in (23, 24):
            x, y, z = coords[idx]
            coords[idx] = (x + offset, y, z)
        result = MagicMock()
        result.pose_landmarks = _make_landmarks(coords)
        return result

    mock_pose = MagicMock()
    mock_pose.process.side_effect = [_make_moving_landmarks(i) for i in range(15)]
    mock_pose.close = MagicMock()

    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap_instance = MagicMock()
        # Provide enough frames for the motion gate window to classify as moving.
        frames = [(True, fake_frame) for _ in range(15)] + [(False, None)]
        mock_cap_instance.read.side_effect = frames
        mock_cap_instance.get.side_effect = lambda key: (
            30.0 if key == cv2.CAP_PROP_FPS else 640.0
        )
        mock_cap.return_value = mock_cap_instance

        analyze_video(
            str(tmp_path / "fake.mp4"),
            str(output_csv),
            None,
            show_preview=False,
            pose_detector=mock_pose,
        )

    assert output_csv.exists()
    with open(output_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 15
    # The motion gate needs a few frames before it reports "moving".
    moving_rows = [r for r in rows if r["is_moving"] == "True"]
    assert len(moving_rows) > 0
    # At least one moving frame should have been scored and marked acceptable.
    scored = [r for r in moving_rows if float(r["core_risk"]) > 0]
    assert len(scored) > 0
    assert all(r["status"] in ("acceptable", "caution", "risky") for r in rows)


def test_analyze_video_raises_when_input_video_cannot_be_opened(tmp_path):
    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_cap.return_value = mock_cap_instance

        with pytest.raises(RuntimeError, match="Could not open video"):
            analyze_video(
                str(tmp_path / "missing.mp4"), str(tmp_path / "out.csv"), None
            )


def test_analyze_video_raises_when_video_writer_cannot_be_opened(tmp_path):
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_pose = MagicMock()
    mock_pose.process.return_value.pose_landmarks = None
    mock_pose.close = MagicMock()

    with patch("cv2.VideoCapture") as mock_cap, patch("cv2.VideoWriter") as mock_writer:
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, fake_frame), (False, None)]
        mock_cap_instance.get.side_effect = lambda key: (
            30.0 if key == cv2.CAP_PROP_FPS else 640.0
        )
        mock_cap.return_value = mock_cap_instance

        mock_writer_instance = MagicMock()
        mock_writer_instance.isOpened.return_value = False
        mock_writer.return_value = mock_writer_instance

        with pytest.raises(RuntimeError, match="Could not open video writer"):
            analyze_video(
                str(tmp_path / "fake.mp4"),
                str(tmp_path / "out.csv"),
                str(tmp_path / "out.mp4"),
                pose_detector=mock_pose,
            )


def test_analyze_video_webcam_preview_writes_no_files(tmp_path, monkeypatch):
    """Webcam mode with no output flags shows preview and writes nothing."""
    monkeypatch.chdir(tmp_path)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_pose = MagicMock()
    mock_pose.process.return_value.pose_landmarks = None
    mock_pose.close = MagicMock()

    with (
        patch("cv2.VideoCapture") as mock_cap,
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyAllWindows") as mock_destroy,
    ):
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, fake_frame), (False, None)]
        mock_cap_instance.get.side_effect = lambda key: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        }[key]
        mock_cap.return_value = mock_cap_instance

        analyze_video(
            None,
            None,
            None,
            show_preview=False,
            pose_detector=mock_pose,
            webcam_index=0,
        )

    mock_cap.assert_called_once_with(0)
    mock_imshow.assert_called_once()
    mock_destroy.assert_called_once()
    assert not (tmp_path / "risk_report.csv").exists()


def test_analyze_video_webcam_writes_csv_when_requested(tmp_path, monkeypatch):
    """Webcam mode records CSV when --output-csv is explicitly provided."""
    monkeypatch.chdir(tmp_path)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_pose = MagicMock()
    mock_pose.process.return_value.pose_landmarks = None
    mock_pose.close = MagicMock()

    output_csv = tmp_path / "webcam_report.csv"

    with (
        patch("cv2.VideoCapture") as mock_cap,
        patch("cv2.imshow"),
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyAllWindows"),
    ):
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, fake_frame), (False, None)]
        mock_cap_instance.get.side_effect = lambda key: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        }[key]
        mock_cap.return_value = mock_cap_instance

        analyze_video(
            None,
            str(output_csv),
            None,
            show_preview=False,
            pose_detector=mock_pose,
            webcam_index=0,
        )

    assert output_csv.exists()


def test_main_prints_expected_output(capsys):
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_pose = MagicMock()
    mock_pose.process.return_value.pose_landmarks = None
    mock_pose.close = MagicMock()

    with (
        patch("cv2.VideoCapture") as mock_cap,
        patch("video_risk_analyzer._create_pose_detector", return_value=mock_pose),
    ):
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, fake_frame), (False, None)]
        mock_cap_instance.get.side_effect = lambda key: (
            30.0 if key == cv2.CAP_PROP_FPS else 640.0
        )
        mock_cap.return_value = mock_cap_instance

        rc = main(
            [
                "fake.mp4",
                "--output-csv",
                "report.csv",
            ]
        )

    assert rc == 0
    captured = capsys.readouterr()
    assert "Wrote CSV report to report.csv" in captured.out


def test_main_runs_webcam_mode(capsys, tmp_path, monkeypatch):
    """main(['--webcam']) opens camera 0 and shows preview without writing files."""
    monkeypatch.chdir(tmp_path)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_pose = MagicMock()
    mock_pose.process.return_value.pose_landmarks = None
    mock_pose.close = MagicMock()

    with (
        patch("cv2.VideoCapture") as mock_cap,
        patch("video_risk_analyzer._create_pose_detector", return_value=mock_pose),
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyAllWindows"),
    ):
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, fake_frame), (False, None)]
        mock_cap_instance.get.side_effect = lambda key: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        }[key]
        mock_cap.return_value = mock_cap_instance

        rc = main(["--webcam"])

    assert rc == 0
    mock_cap.assert_called_once_with(0)
    captured = capsys.readouterr()
    assert "Wrote CSV" not in captured.out
    mock_imshow.assert_called_once()


def test_main_on_synthetic_video(tmp_path):
    video_path = tmp_path / "test.mp4"
    output_csv = tmp_path / "out.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    for _ in range(30):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
    writer.release()

    mock_pose = MagicMock()
    mock_pose.process.return_value.pose_landmarks = None
    mock_pose.close = MagicMock()

    with patch("video_risk_analyzer._create_pose_detector", return_value=mock_pose):
        main([str(video_path), "--output-csv", str(output_csv)])

    assert output_csv.exists()
    with open(output_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 30


def test_main_requires_input_or_webcam():
    """Calling main with no source raises a usage error."""
    with pytest.raises(SystemExit):
        main([])


def test_main_rejects_both_sources():
    """Cannot specify both a file and a webcam."""
    with pytest.raises(SystemExit):
        main(["fake.mp4", "--webcam"])
