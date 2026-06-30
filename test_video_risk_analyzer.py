"""Unit tests for the offline video risk analyzer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from video_risk_analyzer import analyze_video, build_csv_rows


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
        mock_cap_instance.get.side_effect = lambda key: 30.0 if key == 5 else 3.0
        mock_cap.return_value = mock_cap_instance

        analyze_video(
            str(tmp_path / "fake.mp4"),
            str(output_csv),
            None,
            show_preview=False,
            pose_detector=mock_pose,
        )

    assert output_csv.exists()
