"""Unit tests for injury_risk.py."""

import pytest

from injury_risk import RiskProfile, RiskResult, PROFILE_PRESETS


class TestDataStructures:
    def test_profile_has_required_fields(self):
        profile = RiskProfile.from_preset("balanced")
        assert profile.key == "balanced"
        assert profile.label == "Balanced"
        assert set(profile.curves) == {
            "hip_trajectory_deviation",
            "knee_flexion",
            "foot_alignment",
            "landing_pitch",
        }
        assert profile.weights == {
            "hip_trajectory_deviation": 0.15,
            "knee_flexion": 0.35,
            "foot_alignment": 0.25,
            "landing_pitch": 0.25,
        }
        assert set(profile.interactions) == {
            "knee_x_hip",
            "foot_x_pitch",
            "knee_x_pitch",
            "hip_x_foot",
        }
        assert profile.bands == {"green_max": 29, "yellow_max": 59}

    def test_all_presets_exist(self):
        assert set(PROFILE_PRESETS) == {"conservative", "balanced", "aggressive"}

    def test_risk_result_is_frozen_dataclass(self):
        result = RiskResult(
            total_risk=12.5,
            base_score=10.0,
            interaction_score=2.5,
            normalized={},
            alerts=[],
            status="Optimal",
            profile_label="Balanced",
        )
        assert result.total_risk == 12.5
        with pytest.raises(AttributeError):
            result.total_risk = 0.0
