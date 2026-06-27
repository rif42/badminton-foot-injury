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
    def test_draw_insufficient_data(self, mock_cv2):
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay.draw(frame, None)

        mock_cv2.rectangle.assert_called()
        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        self.assertIn("Insufficient data", joined)
