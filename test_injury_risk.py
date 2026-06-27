"""Unit tests for injury_risk.py."""

import pytest

from injury_risk import (
    PROFILE_PRESETS,
    REQUIRED_JOINTS,
    RiskProfile,
    RiskResult,
    compute_base_score,
    compute_foot_alignment,
    compute_interaction_score,
    compute_knee_flexion,
    compute_landing_pitch,
    compute_total_risk,
    eval_piecewise_by_max,
    eval_piecewise_by_min,
    front_leg_side,
    has_required_landmarks,
    risk_level_from_normalized,
)


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


class TestCurveEvaluation:
    def test_eval_piecewise_by_max(self):
        curve = [
            {"max": 10, "r": 0.05},
            {"max": 20, "r": 0.55},
            {"max": float("inf"), "r": 0.9},
        ]
        assert eval_piecewise_by_max(5, curve) == pytest.approx(0.05)
        assert eval_piecewise_by_max(15, curve) == pytest.approx(0.55)
        assert eval_piecewise_by_max(25, curve) == pytest.approx(0.9)

    def test_eval_piecewise_by_min(self):
        curve = [
            {"min": 0.001, "r": 0.05},
            {"min": -5, "r": 0.35},
            {"min": float("-inf"), "r": 0.9},
        ]
        assert eval_piecewise_by_min(2, curve) == pytest.approx(0.05)
        assert eval_piecewise_by_min(-3, curve) == pytest.approx(0.35)
        assert eval_piecewise_by_min(-10, curve) == pytest.approx(0.9)

    def test_risk_level_from_normalized(self):
        assert risk_level_from_normalized(0.2)["txt"] == "Low"
        assert risk_level_from_normalized(0.5)["txt"] == "Moderate"
        assert risk_level_from_normalized(0.8)["txt"] == "High"

    def test_compute_total_risk_matches_reference(self):
        profile = RiskProfile.from_preset("balanced")
        normalized = {
            "hip_trajectory_deviation": 0.1,
            "knee_flexion": 0.9,
            "foot_alignment": 0.3,
            "landing_pitch": 0.2,
        }
        base = compute_base_score(normalized, profile.weights)
        interaction = compute_interaction_score(normalized, profile.interactions)
        total = compute_total_risk(base, interaction)

        expected_base = 100.0 * (
            0.15 * 0.1 + 0.35 * 0.9 + 0.25 * 0.3 + 0.25 * 0.2
        )
        expected_interaction = (
            12 * 0.9 * 0.1
            + 12 * 0.3 * 0.2
            + 8 * 0.9 * 0.2
            + 6 * 0.1 * 0.3
        )
        assert base == pytest.approx(expected_base)
        assert interaction == pytest.approx(expected_interaction)
        assert total == pytest.approx(min(expected_base + expected_interaction, 100.0))


def _landmark(x, y, z=0.0, visibility=1.0):
    return (x, y, z, visibility)


class TestParameterExtraction:
    def test_required_joints_set(self):
        assert REQUIRED_JOINTS == {
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        }

    def test_has_required_landmarks_false_when_low_visibility(self):
        landmarks = {
            "left_hip": _landmark(0.5, 0.5, visibility=0.2),
        }
        assert has_required_landmarks(landmarks) is False

    def test_front_leg_side_picks_lower_ankle(self):
        landmarks = {
            "left_ankle": _landmark(0.4, 0.7),
            "right_ankle": _landmark(0.6, 0.8),
        }
        assert front_leg_side(landmarks) == "right"

    def test_knee_flexion_is_internal_angle(self):
        # Straight leg: hip (0.5, 0.4), knee (0.5, 0.6), ankle (0.5, 0.8)
        # All y values; internal angle ~180.
        frame_shape = (480, 640)
        landmarks = {
            "left_hip": _landmark(0.5, 0.4),
            "left_knee": _landmark(0.5, 0.6),
            "left_ankle": _landmark(0.5, 0.8),
        }
        assert compute_knee_flexion(landmarks, "left", frame_shape) == pytest.approx(
            180.0, abs=1.0
        )

    def test_landing_pitch_negative_when_heel_lower(self):
        # Heel below toe in image -> negative pitch.
        frame_shape = (480, 640)
        landmarks = {
            "left_heel": _landmark(0.5, 0.8),
            "left_foot_index": _landmark(0.55, 0.75),
        }
        pitch = compute_landing_pitch(landmarks, "left", frame_shape)
        assert pitch < 0
