"""Unit tests for the 3-D geometry helpers in ``baseline_risk.py``."""

import math

import pytest

from baseline_risk import (
    LowerBodyPose,
    _CORE_WEIGHT_ANKLE_FOOT_ALIGNMENT,
    _CORE_WEIGHT_HIP_DISPLACEMENT,
    _CORE_WEIGHT_KNEE_STIFFNESS,
    _CORE_WEIGHT_LANDING_ASYMMETRY,
    angle_at,
    ankle_foot_alignment_risk,
    clamp,
    core_risk_score,
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
    assert score == pytest.approx(0.221, abs=1e-3)


def test_landing_asymmetry_knee_angle_only():
    left_hip = (-0.1, 1.0, 0.0)
    right_hip = (0.1, 1.0, 0.0)
    # Left leg straight, right leg flexed to 120°.
    left_knee = (-0.1, 0.5, 0.0)
    right_knee = (0.1, 0.5, math.sqrt(1.0 / 12.0))
    left_ankle = (-0.1, 0.0, 0.0)
    right_ankle = (0.1, 0.0, 0.0)
    score = landing_asymmetry_score(
        left_hip, right_hip,
        left_knee, right_knee,
        left_ankle, right_ankle,
    )
    assert score == pytest.approx(0.35, abs=1e-3)


def test_landing_asymmetry_ankle_height_only():
    left_hip = (-0.1, 1.0, 0.0)
    right_hip = (0.1, 1.0, 0.0)
    left_knee = (-0.1, 0.5, 0.0)
    right_knee = (0.1, 0.5, 0.0)
    left_ankle = (-0.1, 0.0, 0.0)
    right_ankle = (0.1, 0.2, 0.0)
    score = landing_asymmetry_score(
        left_hip, right_hip,
        left_knee, right_knee,
        left_ankle, right_ankle,
    )
    assert score == pytest.approx(0.20, abs=1e-3)


def test_landing_asymmetry_saturates_at_max():
    # Configure each component to hit its individual maximum so the weighted
    # total reaches the simplified-model ceiling of 0.8.
    left_hip = (-0.1, 0.0, 0.0)
    right_hip = (0.1, 1.0, 0.0)
    left_knee = (-0.1, -0.5, 0.0)
    right_knee = (0.1, 0.5, math.sqrt(1.0 / 12.0))
    left_ankle = (-0.1, -1.0, 0.0)
    right_ankle = (0.1, 0.0, 0.0)
    score = landing_asymmetry_score(
        left_hip, right_hip,
        left_knee, right_knee,
        left_ankle, right_ankle,
    )
    assert score == pytest.approx(0.8, abs=1e-3)


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


def test_core_risk_perfect_pose():
    pose = LowerBodyPose(
        left_hip=(-0.1, 1.0, 0.0),
        right_hip=(0.1, 1.0, 0.0),
        left_knee=(-0.1, 0.5, 0.1),
        right_knee=(0.1, 0.5, 0.1),
        left_ankle=(-0.1, 0.0, 0.0),
        right_ankle=(0.1, 0.0, 0.0),
        left_heel=(-0.1, 0.0, -0.1),
        right_heel=(0.1, 0.0, -0.1),
        left_foot_index=(-0.1, 0.0, 0.1),
        right_foot_index=(0.1, 0.0, 0.1),
    )
    result = core_risk_score(pose)
    assert result["core_risk"] < 0.2


def test_core_risk_stiff_knees():
    pose = LowerBodyPose(
        left_hip=(-0.1, 1.4, 0.0),
        right_hip=(0.1, 1.4, 0.0),
        left_knee=(-0.1, 0.9, 0.0),
        right_knee=(0.1, 0.9, 0.0),
        left_ankle=(-0.1, 0.0, 0.0),
        right_ankle=(0.1, 0.0, 0.0),
        left_heel=(-0.1, 0.0, -0.1),
        right_heel=(0.1, 0.0, -0.1),
        left_foot_index=(-0.1, 0.0, 0.1),
        right_foot_index=(0.1, 0.0, 0.1),
    )
    result = core_risk_score(pose)
    assert result["core_risk"] > 0.2
    assert result["knee_stiffness_risk"] > 0.5


def test_core_risk_score_returns_expected_keys():
    pose = LowerBodyPose(
        left_hip=(0.0, 1.0, 0.0),
        right_hip=(0.0, 1.0, 0.0),
        left_knee=(0.0, 0.5, 0.1),
        right_knee=(0.0, 0.5, 0.1),
        left_ankle=(0.0, 0.0, 0.0),
        right_ankle=(0.0, 0.0, 0.0),
        left_heel=(0.0, 0.0, -0.1),
        right_heel=(0.0, 0.0, -0.1),
        left_foot_index=(0.0, 0.0, 0.1),
        right_foot_index=(0.0, 0.0, 0.1),
    )
    result = core_risk_score(pose)
    assert set(result.keys()) == {
        "knee_stiffness_risk",
        "ankle_foot_alignment_risk",
        "hip_displacement_proxy",
        "landing_asymmetry_score",
        "core_risk",
    }


def test_core_risk_weighted_knee_stiffness_only():
    # Straight, symmetric, centered legs drive knee stiffness to 1.0 while
    # all other sub-scores remain 0.0.
    pose = LowerBodyPose(
        left_hip=(0.0, 1.0, 0.0),
        right_hip=(0.0, 1.0, 0.0),
        left_knee=(0.0, 0.5, 0.0),
        right_knee=(0.0, 0.5, 0.0),
        left_ankle=(0.0, 0.0, 0.0),
        right_ankle=(0.0, 0.0, 0.0),
        left_heel=(0.0, 0.0, -0.1),
        right_heel=(0.0, 0.0, -0.1),
        left_foot_index=(0.0, 0.0, 0.1),
        right_foot_index=(0.0, 0.0, 0.1),
    )
    result = core_risk_score(pose)
    assert result["knee_stiffness_risk"] == pytest.approx(1.0, abs=1e-6)
    assert result["ankle_foot_alignment_risk"] == pytest.approx(0.0, abs=1e-6)
    assert result["hip_displacement_proxy"] == pytest.approx(0.0, abs=1e-6)
    assert result["landing_asymmetry_score"] == pytest.approx(0.0, abs=1e-6)
    assert result["core_risk"] == pytest.approx(_CORE_WEIGHT_KNEE_STIFFNESS, abs=1e-6)


def test_core_risk_weighted_ankle_alignment_only():
    # A flexed, symmetric pose with the knee deviated laterally and the foot
    # turned 45° drives ankle-foot alignment risk to 1.0 while keeping the
    # other sub-scores at 0.0.
    z_flex = 0.2
    knee_x = 0.264
    foot_half = 0.264
    pose = LowerBodyPose(
        left_hip=(0.0, 1.0, 0.0),
        right_hip=(0.0, 1.0, 0.0),
        left_knee=(knee_x, 0.5, z_flex),
        right_knee=(knee_x, 0.5, z_flex),
        left_ankle=(0.0, 0.0, 0.0),
        right_ankle=(0.0, 0.0, 0.0),
        left_heel=(-foot_half, 0.0, -foot_half),
        right_heel=(-foot_half, 0.0, -foot_half),
        left_foot_index=(foot_half, 0.0, foot_half),
        right_foot_index=(foot_half, 0.0, foot_half),
    )
    result = core_risk_score(pose)
    assert result["knee_stiffness_risk"] == pytest.approx(0.0, abs=1e-3)
    assert result["ankle_foot_alignment_risk"] == pytest.approx(1.0, abs=1e-3)
    assert result["hip_displacement_proxy"] == pytest.approx(0.0, abs=1e-3)
    assert result["landing_asymmetry_score"] == pytest.approx(0.0, abs=1e-3)
    assert result["core_risk"] == pytest.approx(_CORE_WEIGHT_ANKLE_FOOT_ALIGNMENT, abs=1e-3)


def test_core_risk_weighted_hip_displacement_only():
    # Flexed knees with the feet far in front of the pelvis drive the hip
    # displacement proxy to 1.0 while the other sub-scores stay at 0.0.
    pose = LowerBodyPose(
        left_hip=(0.0, 1.0, 0.0),
        right_hip=(0.0, 1.0, 0.0),
        left_knee=(0.0, 0.5, 0.5),
        right_knee=(0.0, 0.5, 0.5),
        left_ankle=(0.0, 0.0, 0.0),
        right_ankle=(0.0, 0.0, 0.0),
        left_heel=(0.0, 0.0, 9.9),
        right_heel=(0.0, 0.0, 9.9),
        left_foot_index=(0.0, 0.0, 10.1),
        right_foot_index=(0.0, 0.0, 10.1),
    )
    result = core_risk_score(pose)
    assert result["knee_stiffness_risk"] == pytest.approx(0.0, abs=1e-3)
    assert result["ankle_foot_alignment_risk"] == pytest.approx(0.0, abs=1e-3)
    assert result["hip_displacement_proxy"] == pytest.approx(1.0, abs=1e-6)
    assert result["landing_asymmetry_score"] == pytest.approx(0.0, abs=1e-3)
    assert result["core_risk"] == pytest.approx(_CORE_WEIGHT_HIP_DISPLACEMENT, abs=1e-6)


def test_core_risk_weighted_landing_asymmetry_only():
    # Asymmetric knee angles, a small hip drop, and asymmetric ankle heights
    # drive landing asymmetry to its maximum of 0.8 while other sub-scores
    # remain 0.0.
    z_145 = 0.5 / math.tan(math.radians(72.5))
    z_85 = 0.5 / math.tan(math.radians(42.5))
    ankle_diff = 0.2528
    pose = LowerBodyPose(
        left_hip=(0.0, 1.0, 0.0),
        right_hip=(0.0, 1.0001, 0.0),
        left_knee=(0.0, 0.5, z_145),
        right_knee=(0.0, 0.5, z_85),
        left_ankle=(0.0, 0.0, 0.0),
        right_ankle=(0.0, ankle_diff, 0.0),
        left_heel=(0.0, 0.0, -0.1),
        right_heel=(0.0, 0.0, -0.1),
        left_foot_index=(0.0, 0.0, 0.1),
        right_foot_index=(0.0, 0.0, 0.1),
    )
    result = core_risk_score(pose)
    assert result["knee_stiffness_risk"] == pytest.approx(0.0, abs=1e-3)
    assert result["ankle_foot_alignment_risk"] == pytest.approx(0.0, abs=1e-3)
    assert result["hip_displacement_proxy"] == pytest.approx(0.0, abs=1e-3)
    assert result["landing_asymmetry_score"] == pytest.approx(0.8, abs=1e-3)
    assert result["core_risk"] == pytest.approx(
        _CORE_WEIGHT_LANDING_ASYMMETRY * 0.8, abs=1e-3
    )
