"""Unit tests for risk_overlay.py."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from injury_risk import RiskResult
from risk_overlay import RiskOverlay


class TestRiskOverlay(unittest.TestCase):
    @patch("risk_overlay.cv2")
    def test_draw_renders_status_and_profile(self, mock_cv2):
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = RiskResult(
            total_risk=35.0,
            base_score=30.0,
            interaction_score=5.0,
            normalized={},
            alerts=[],
            status="Caution",
            profile_label="Balanced",
        )

        overlay.draw(frame, result)

        mock_cv2.rectangle.assert_called()
        mock_cv2.putText.assert_called()
        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        self.assertIn("Caution", joined)
        self.assertIn("35.0", joined)
        self.assertIn("Balanced", joined)

    @patch("risk_overlay.cv2")
    def test_draw_renders_idle_status(self, mock_cv2):
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = RiskResult(
            total_risk=0.0,
            base_score=0.0,
            interaction_score=0.0,
            normalized={},
            alerts=[],
            status="Idle",
            profile_label="Balanced",
        )

        overlay.draw(frame, result)

        mock_cv2.rectangle.assert_called()
        mock_cv2.putText.assert_called()
        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        self.assertIn("Idle", joined)
        self.assertIn("0.0", joined)
        self.assertIn("Balanced", joined)

    @patch("risk_overlay.cv2")
    def test_draw_insufficient_data(self, mock_cv2):
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay.draw(frame, None)

        mock_cv2.rectangle.assert_called()
        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        self.assertIn("Insufficient data", joined)

    @patch("risk_overlay.cv2")
    def test_draw_insufficient_data_renders_profile_label(self, mock_cv2):
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay.draw(frame, None, profile_label="Conservative")

        mock_cv2.rectangle.assert_called()
        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        self.assertIn("Insufficient data", joined)
        self.assertIn("Profile:", joined)
        self.assertIn("Conservative", joined)

        # Dedicated profile panel is drawn at the top-right, away from the left edge.
        profile_panel_drawn = any(
            call.args[1][0] > 0 and call.args[1][1] == 0
            for call in mock_cv2.rectangle.call_args_list
        )
        self.assertTrue(profile_panel_drawn)

    @patch("risk_overlay.cv2")
    def test_draw_idle_result_from_model_shape(self, mock_cv2):
        """Idle result produced by RiskModel renders correctly."""
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = RiskResult(
            total_risk=0.0,
            base_score=0.0,
            interaction_score=0.0,
            normalized={
                "hip_trajectory_deviation": 0.0,
                "knee_flexion": 0.0,
                "foot_alignment": 0.0,
                "landing_pitch": 0.0,
            },
            alerts=[
                {"label": "Hip Control", "txt": "Low", "cls": "ok"},
                {"label": "Knee Overload", "txt": "Low", "cls": "ok"},
                {"label": "Foot Alignment", "txt": "Low", "cls": "ok"},
                {"label": "Kinetic Shock", "txt": "Low", "cls": "ok"},
            ],
            status="Idle",
            profile_label="Balanced",
        )

        overlay.draw(frame, result)

        mock_cv2.rectangle.assert_called()
        mock_cv2.putText.assert_called()
        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        self.assertIn("Idle", joined)
        self.assertIn("Balanced", joined)

    @patch("risk_overlay.cv2")
    def test_draw_renders_profile_in_dedicated_top_right_panel(self, mock_cv2):
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = RiskResult(
            total_risk=35.0,
            base_score=30.0,
            interaction_score=5.0,
            normalized={},
            alerts=[],
            status="Caution",
            profile_label="Balanced",
        )

        overlay.draw(frame, result)

        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        self.assertIn("Profile:", joined)
        self.assertIn("Balanced", joined)

        # Dedicated profile panel is drawn at the top-right, away from the left edge.
        profile_panel_drawn = any(
            call.args[1][0] > 0 and call.args[1][1] == 0
            for call in mock_cv2.rectangle.call_args_list
        )
        self.assertTrue(profile_panel_drawn)
