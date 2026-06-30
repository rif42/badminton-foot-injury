"""Unit tests for the 3-D geometry helpers in ``baseline_risk.py``."""

import math

import pytest

from baseline_risk import (
    angle_at,
    ankle_foot_alignment_risk,
    clamp,
    distance,
    hip_displacement_proxy,
    knee_flexion_angle,
    knee_stiffness_risk,
    landing_asymmetry_score,
    midpoint,
)


def test_clamp_inside_range():
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_below_range():
    assert clamp(-0.1, 0.0, 1.0) == 0.0


def test_clamp_above_range():
    assert clamp(1.2, 0.0, 1.0) == 1.0


def test_clamp_equal_bounds():
    assert clamp(7.5, 2.0, 2.0) == 2.0


def test_distance_3d():
    a = (0.0, 0.0, 0.0)
    b = (3.0, 4.0, 0.0)
    assert distance(a, b) == 5.0


def test_distance_along_z_axis():
    a = (0.0, 0.0, 0.0)
    b = (0.0, 0.0, 9.0)
    assert distance(a, b) == 9.0


def test_distance_negative_coordinates():
    a = (-1.0, -2.0, -3.0)
    b = (-4.0, -6.0, -3.0)
    assert distance(a, b) == 5.0


def test_midpoint():
    a = (0.0, 0.0, 0.0)
    b = (2.0, 4.0, 6.0)
    assert midpoint(a, b) == (1.0, 2.0, 3.0)


def test_angle_at_right_angle():
    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (0.0, 1.0, 0.0)
    assert math.isclose(angle_at(a, b, c), 90.0, abs_tol=1e-6)


def test_angle_at_collinear_zero_degrees():
    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (2.0, 0.0, 0.0)
    assert math.isclose(angle_at(a, b, c), 0.0, abs_tol=1e-6)


def test_angle_at_collinear_180_degrees():
    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (-1.0, 0.0, 0.0)
    assert math.isclose(angle_at(a, b, c), 180.0, abs_tol=1e-6)


def test_angle_at_collapsed_vertex_returns_zero():
    vertex = (0.0, 0.0, 0.0)
    other = (1.0, 0.0, 0.0)
    assert angle_at(other, vertex, vertex) == 0.0
    assert angle_at(vertex, vertex, other) == 0.0


def test_distance_rejects_non_3d_point():
    with pytest.raises(AssertionError):
        distance((0.0, 0.0), (1.0, 0.0, 0.0))


def test_midpoint_rejects_non_3d_point():
    with pytest.raises(AssertionError):
        midpoint((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_knee_flexion_right_angle():
    hip = (0.0, 1.0, 0.0)
    knee = (0.0, 0.0, 0.0)
    ankle = (1.0, 0.0, 0.0)
    assert math.isclose(knee_flexion_angle(hip, knee, ankle), 90.0, abs_tol=1e-6)


def test_knee_stiffness_risk_low():
    assert knee_stiffness_risk(120.0) == 0.0


def test_knee_stiffness_risk_at_threshold():
    assert knee_stiffness_risk(145.0) == 0.0


def test_knee_stiffness_risk_high():
    assert knee_stiffness_risk(180.0) == 1.0


def test_knee_stiffness_risk_mid():
    assert knee_stiffness_risk(160.0) == pytest.approx(0.429, abs=1e-3)


def test_knee_flexion_rejects_non_3d_points():
    with pytest.raises(AssertionError):
        knee_flexion_angle((0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(AssertionError):
        knee_flexion_angle((0.0, 1.0, 0.0), (0.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(AssertionError):
        knee_flexion_angle((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0))


def test_knee_stiffness_risk_custom_bounds_above_max():
    assert knee_stiffness_risk(100.0, min_safe=60.0, max_safe=90.0) == 1.0


def test_knee_stiffness_risk_custom_bounds_below_min():
    assert knee_stiffness_risk(30.0, min_safe=60.0, max_safe=90.0) == 0.0


def test_knee_stiffness_risk_equal_bounds_triggers_guard():
    with pytest.raises(AssertionError):
        knee_stiffness_risk(120.0, min_safe=120.0, max_safe=120.0)


def test_ankle_foot_alignment_perfect():
    knee = (0.0, 0.8, 0.5)
    ankle = (0.0, 0.1, 0.5)
    heel = (0.0, 0.0, 0.3)
    foot_index = (0.0, 0.0, 0.7)
    leg_length = 1.4
    assert ankle_foot_alignment_risk(knee, ankle, heel, foot_index, leg_length) == pytest.approx(0.0, abs=1e-3)


def test_ankle_foot_alignment_toe_in():
    # 45° toe-in with the knee centered over the foot -> risk ≈ 0.30.
    knee = (0.1, 0.8, 0.5)
    ankle = (0.1, 0.1, 0.5)
    heel = (0.0, 0.0, 0.3)
    foot_index = (0.2, 0.0, 0.5)
    leg_length = 1.4
    risk = ankle_foot_alignment_risk(knee, ankle, heel, foot_index, leg_length)
    assert risk == pytest.approx(0.30, abs=1e-3)


def test_ankle_foot_alignment_toe_in_with_knee_deviation():
    # 90° toe-in plus lateral knee deviation -> combined risk ≈ 0.527.
    knee = (0.0, 0.8, 0.5)
    ankle = (0.0, 0.1, 0.5)
    heel = (0.0, 0.0, 0.3)
    foot_index = (0.2, 0.0, 0.3)
    leg_length = 1.4
    risk = ankle_foot_alignment_risk(knee, ankle, heel, foot_index, leg_length)
    assert risk == pytest.approx(0.5273, abs=1e-3)


def test_ankle_foot_alignment_knee_deviation_at_threshold():
    # Foot pointing straight; knee deviation is exactly 22% of leg length.
    knee = (0.22, 0.8, 0.5)
    ankle = (0.0, 0.1, 0.5)
    heel = (0.0, 0.0, 0.3)
    foot_index = (0.0, 0.0, 0.7)
    leg_length = 1.0
    risk = ankle_foot_alignment_risk(knee, ankle, heel, foot_index, leg_length)
    assert risk == pytest.approx(0.70, abs=1e-3)


def test_ankle_foot_alignment_foot_angle_45():
    # Knee centered over foot; foot turned 45°.
    knee = (0.5, 0.8, 0.5)
    ankle = (0.5, 0.1, 0.5)
    heel = (0.0, 0.0, 0.0)
    foot_index = (1.0, 0.0, 1.0)
    leg_length = 1.0
    risk = ankle_foot_alignment_risk(knee, ankle, heel, foot_index, leg_length)
    assert risk == pytest.approx(0.30, abs=1e-3)


def test_ankle_foot_alignment_maximum_risk():
    knee = (0.5, 0.8, 0.5)
    ankle = (0.0, 0.1, 0.5)
    heel = (0.0, 0.0, 0.3)
    foot_index = (0.2, 0.0, 0.3)
    leg_length = 1.0
    risk = ankle_foot_alignment_risk(knee, ankle, heel, foot_index, leg_length)
    assert risk == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_ankle_foot_alignment_rejects_non_3d_points(idx):
    points = [
        (0.0, 0.8, 0.5),
        (0.0, 0.1, 0.5),
        (0.0, 0.0, 0.3),
        (0.0, 0.0, 0.7),
    ]
    invalid = (0.0, 0.0)
    args = [points[i] if i != idx else invalid for i in range(4)] + [1.4]
    with pytest.raises(AssertionError):
        ankle_foot_alignment_risk(*args)


@pytest.mark.parametrize("bad_length", [0.0, -1.0])
def test_ankle_foot_alignment_rejects_non_positive_leg_length(bad_length):
    knee = (0.0, 0.8, 0.5)
    ankle = (0.0, 0.1, 0.5)
    heel = (0.0, 0.0, 0.3)
    foot_index = (0.0, 0.0, 0.7)
    with pytest.raises(AssertionError):
        ankle_foot_alignment_risk(knee, ankle, heel, foot_index, bad_length)


def test_hip_displacement_low():
    left_hip = (-0.1, 1.0, 0.0)
    right_hip = (0.1, 1.0, 0.0)
    heel = (-0.1, 0.0, 0.0)
    foot_index = (0.1, 0.0, 0.0)
    leg_length = 1.0
    assert hip_displacement_proxy(left_hip, right_hip, heel, foot_index, leg_length) == pytest.approx(0.0, abs=1e-3)


def test_hip_displacement_high():
    left_hip = (-0.1, 1.0, -0.5)
    right_hip = (0.1, 1.0, -0.5)
    heel = (-0.1, 0.0, 0.5)
    foot_index = (0.1, 0.0, 0.5)
    leg_length = 1.0
    proxy = hip_displacement_proxy(left_hip, right_hip, heel, foot_index, leg_length)
    assert proxy == 1.0


def test_hip_displacement_mid():
    left_hip = (-0.1, 1.0, -0.25)
    right_hip = (0.1, 1.0, -0.25)
    heel = (-0.1, 0.0, 0.25)
    foot_index = (0.1, 0.0, 0.25)
    leg_length = 1.0
    proxy = hip_displacement_proxy(left_hip, right_hip, heel, foot_index, leg_length)
    assert proxy == pytest.approx(0.5, abs=1e-3)


def test_hip_displacement_clamps_above_one():
    left_hip = (-0.1, 1.0, -1.0)
    right_hip = (0.1, 1.0, -1.0)
    heel = (-0.1, 0.0, 1.0)
    foot_index = (0.1, 0.0, 1.0)
    leg_length = 1.0
    proxy = hip_displacement_proxy(left_hip, right_hip, heel, foot_index, leg_length)
    assert proxy == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_hip_displacement_rejects_non_3d_points(idx):
    points = [
        (-0.1, 1.0, 0.0),
        (0.1, 1.0, 0.0),
        (-0.1, 0.0, 0.0),
        (0.1, 0.0, 0.0),
    ]
    invalid = (0.0, 0.0)
    args = [points[i] if i != idx else invalid for i in range(4)] + [1.0]
    with pytest.raises(AssertionError):
        hip_displacement_proxy(*args)


@pytest.mark.parametrize("bad_length", [0.0, -1.0])
def test_hip_displacement_rejects_non_positive_leg_length(bad_length):
    left_hip = (-0.1, 1.0, 0.0)
    right_hip = (0.1, 1.0, 0.0)
    heel = (-0.1, 0.0, 0.0)
    foot_index = (0.1, 0.0, 0.0)
    with pytest.raises(AssertionError):
        hip_displacement_proxy(left_hip, right_hip, heel, foot_index, bad_length)


def test_landing_asymmetry_symmetric():
    left_hip = (-0.1, 1.0, 0.0)
    right_hip = (0.1, 1.0, 0.0)
    left_knee = (-0.1, 0.5, 0.0)
    right_knee = (0.1, 0.5, 0.0)
    left_ankle = (-0.1, 0.0, 0.0)
    right_ankle = (0.1, 0.0, 0.0)
    assert landing_asymmetry_score(
        left_hip, right_hip,
        left_knee, right_knee,
        left_ankle, right_ankle,
    ) == pytest.approx(0.0, abs=1e-3)


def test_landing_asymmetry_hip_drop():
    left_hip = (-0.1, 0.8, 0.0)
    right_hip = (0.1, 1.0, 0.0)
    left_knee = (-0.1, 0.5, 0.0)
    right_knee = (0.1, 0.5, 0.0)
    left_ankle = (-0.1, 0.0, 0.0)
    right_ankle = (0.1, 0.0, 0.0)
    score = landing_asymmetry_score(
        left_hip, right_hip,
        left_knee, right_knee,
        left_ankle, right_ankle,
    )
    assert score > 0.1


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4, 5])
def test_landing_asymmetry_rejects_non_3d_points(idx):
    points = [
        (-0.1, 1.0, 0.0),
        (0.1, 1.0, 0.0),
        (-0.1, 0.5, 0.0),
        (0.1, 0.5, 0.0),
        (-0.1, 0.0, 0.0),
        (0.1, 0.0, 0.0),
    ]
    invalid = (0.0, 0.0)
    args = [points[i] if i != idx else invalid for i in range(6)]
    with pytest.raises(AssertionError):
        landing_asymmetry_score(*args)
