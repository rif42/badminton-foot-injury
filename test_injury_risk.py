"""Unit tests for injury_risk.py."""

import pytest

from injury_risk import (
    PROFILE_PRESETS,
    REQUIRED_JOINTS,
    RiskModel,
    RiskProfile,
    RiskResult,
    compute_base_score,
    compute_body_scale,
    compute_foot_alignment,
    compute_interaction_score,
    compute_knee_flexion,
    compute_landing_pitch,
    compute_normalized_hip_movement,
    compute_total_risk,
    eval_piecewise_by_max,
    eval_piecewise_by_min,
    front_leg_side,
    has_required_landmarks,
    is_idle,
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


def _make_safe_landmarks(hip_shift=(0.0, 0.0)):
    def lm(x, y, v=1.0):
        return (x, y, 0.0, v)

    return {
        "left_hip": lm(0.45 + hip_shift[0], 0.35 + hip_shift[1]),
        "right_hip": lm(0.55 + hip_shift[0], 0.35 + hip_shift[1]),
        "left_knee": lm(0.45, 0.55),
        "right_knee": lm(0.55, 0.55),
        "left_ankle": lm(0.45, 0.8),
        "right_ankle": lm(0.55, 0.8),
        "left_heel": lm(0.43, 0.82),
        "right_heel": lm(0.53, 0.82),
        "left_foot_index": lm(0.47, 0.81),
        "right_foot_index": lm(0.57, 0.81),
    }


class TestBodyScaleAndIdleGate:
    def test_compute_body_scale_average_leg_length(self):
        frame_shape = (480, 640)
        landmarks = {
            "left_hip": _landmark(0.5, 0.3),
            "left_ankle": _landmark(0.5, 0.7),
            "right_hip": _landmark(0.5, 0.3),
            "right_ankle": _landmark(0.5, 0.7),
        }
        # Each leg is 0.4 * 480 = 192 px; average is the same.
        scale = compute_body_scale(landmarks, frame_shape)
        assert scale == pytest.approx(192.0)

    def test_compute_body_scale_returns_none_when_missing(self):
        frame_shape = (480, 640)
        assert compute_body_scale({}, frame_shape) is None

    def test_compute_body_scale_uses_visible_side(self):
        frame_shape = (480, 640)
        landmarks = {
            "left_hip": _landmark(0.5, 0.3),
            "left_ankle": _landmark(0.5, 0.7),
        }
        assert compute_body_scale(landmarks, frame_shape) == pytest.approx(192.0)

    def test_compute_normalized_hip_movement(self):
        history = [(100.0, 100.0), (103.0, 104.0)]  # 5 px displacement
        body_scale = 100.0
        movement = compute_normalized_hip_movement(history, body_scale)
        assert movement == pytest.approx(0.05)

    def test_compute_normalized_hip_movement_returns_none_for_short_history(self):
        assert compute_normalized_hip_movement([(100.0, 100.0)], 100.0) is None

    def test_is_idle_true_when_movement_below_threshold(self):
        frame_shape = (480, 640)
        # Body scale ~192 px; movement ~1.4 px -> ~0.007 < 0.015.
        landmarks = {
            "left_hip": _landmark(0.5, 0.3),
            "left_ankle": _landmark(0.5, 0.7),
            "right_hip": _landmark(0.5, 0.3),
            "right_ankle": _landmark(0.5, 0.7),
        }
        history = [(320.0, 144.0), (321.0, 145.0)]
        assert is_idle(landmarks, frame_shape, history) is True

    def test_is_idle_false_when_movement_above_threshold(self):
        frame_shape = (480, 640)
        # Body scale ~192 px; movement 20 px -> ~0.104 > 0.015.
        landmarks = {
            "left_hip": _landmark(0.5, 0.3),
            "left_ankle": _landmark(0.5, 0.7),
            "right_hip": _landmark(0.5, 0.3),
            "right_ankle": _landmark(0.5, 0.7),
        }
        history = [(320.0, 144.0), (340.0, 144.0)]
        assert is_idle(landmarks, frame_shape, history) is False


class TestRiskModel:
    def test_update_returns_none_when_landmarks_missing(self):
        model = RiskModel()
        assert model.update({"left_hip": (0.5, 0.5, 0.0, 1.0)}, (480, 640)) is None

    def test_update_returns_risk_result_after_history_fills(self):
        model = RiskModel()
        frame_shape = (480, 640)
        result = None
        for i in range(5):
            result = model.update(_make_safe_landmarks(hip_shift=(i * 0.02, 0)), frame_shape)
        assert isinstance(result, RiskResult)
        assert 0.0 <= result.total_risk <= 100.0
        assert result.status in {"Optimal", "Caution", "High Risk"}
        assert result.profile_label == "Balanced"

    def test_profile_switching(self):
        model = RiskModel()
        assert model.profile.key == "balanced"
        model.set_profile("conservative")
        assert model.profile.key == "conservative"
        model.cycle_profile()
        assert model.profile.key == "aggressive"
        model.cycle_profile()
        assert model.profile.key == "balanced"

    def test_smoothing_averages_over_window(self):
        model = RiskModel(smoothing_window=3)
        frame_shape = (480, 640)
        first = None
        for i in range(5):
            lm = _make_safe_landmarks(hip_shift=(i * 0.02, 0))
            # Move ankles down each frame to create changing geometry.
            lm["left_ankle"] = (lm["left_ankle"][0], 0.75 + i * 0.01, 0.0, 1.0)
            lm["right_ankle"] = (lm["right_ankle"][0], 0.75 + i * 0.01, 0.0, 1.0)
            first = model.update(lm, frame_shape)
        assert first is not None

    def test_update_returns_idle_when_hip_is_stationary(self):
        model = RiskModel()
        frame_shape = (480, 640)
        for _ in range(5):
            result = model.update(_make_safe_landmarks(), frame_shape)
        assert result is not None
        assert result.status == "Idle"
        assert result.total_risk == pytest.approx(0.0)
        assert result.base_score == pytest.approx(0.0)
        assert result.interaction_score == pytest.approx(0.0)
        assert all(v == 0.0 for v in result.normalized.values())
        assert all(alert["txt"] == "Low" for alert in result.alerts)

    def test_idle_to_dynamic_resets_smoothing(self):
        model = RiskModel(smoothing_window=5)
        frame_shape = (480, 640)

        # Run idle for several frames.
        for _ in range(5):
            idle_result = model.update(_make_safe_landmarks(), frame_shape)
        assert idle_result is not None
        assert idle_result.status == "Idle"
        assert idle_result.total_risk == pytest.approx(0.0)

        # Transition to dynamic. Hip shift of 0.02/frame -> ~12.8 px/frame.
        result = None
        for i in range(1, 6):
            result = model.update(
                _make_safe_landmarks(hip_shift=(i * 0.02, 0)), frame_shape
            )
        assert result is not None
        assert result.status != "Idle"
        # Smoothing should not average in the previous idle zeros.
        assert result.total_risk > 0.0

    def test_dynamic_to_idle_resets_smoothing(self):
        model = RiskModel(smoothing_window=5)
        frame_shape = (480, 640)

        # Run dynamic for several frames.
        for i in range(5):
            model.update(_make_safe_landmarks(hip_shift=(i * 0.02, 0)), frame_shape)

        # Transition to idle; flush the hip-history window so the idle gate fires.
        idle_result = None
        for _ in range(5):
            idle_result = model.update(_make_safe_landmarks(), frame_shape)
        assert idle_result is not None
        assert idle_result.status == "Idle"
        assert idle_result.total_risk == pytest.approx(0.0)

