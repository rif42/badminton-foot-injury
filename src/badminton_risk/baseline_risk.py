"""Small 3-D geometry helpers used by the baseline badminton injury-risk calculator.

All functions in this module assume points are 3-D coordinates represented as
``(x, y, z)`` tuples. Callers are responsible for providing correctly-shaped
inputs; ``distance`` and ``midpoint`` validate this at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

Point = tuple[float, float, float]

# Thresholds and weights for ankle-foot alignment risk.
# Knee-over-foot deviation is expressed as a fraction of leg length; a deviation
# equal to 22 % of leg length maps to the maximum knee-deviation score.
_ANKLE_FOOT_KNEE_DEVIATION_THRESHOLD = 0.22
# Foot progression (toe-in/toe-out) angle in degrees; 45° maps to the max angle score.
_ANKLE_FOOT_FOOT_ANGLE_THRESHOLD = 45.0
# Relative contribution of knee deviation vs. foot progression angle.
_ANKLE_FOOT_KNEE_DEVIATION_WEIGHT = 0.70
_ANKLE_FOOT_FOOT_ANGLE_WEIGHT = 0.30

# Threshold for hip displacement proxy; a displacement equal to one full leg length
# maps to the maximum proxy value.
_HIP_DISPLACEMENT_THRESHOLD = 1.0

# Thresholds and weights for landing/lunge asymmetry risk.
# Knee-angle difference in degrees; 60° maps to the maximum asymmetry score.
_LANDING_KNEE_ANGLE_THRESHOLD = 60.0
# Hip-drop score is the absolute height difference normalized by this fraction
# of pelvis width; 80 % maps to the maximum hip-drop score.
_LANDING_HIP_HEIGHT_RELATIVE_THRESHOLD = 0.8
# Ankle-height score is the absolute height difference normalized by this fraction
# of average leg length; 20 % maps to the maximum ankle-height score.
_LANDING_ANKLE_HEIGHT_RELATIVE_THRESHOLD = 0.2
# Relative contribution of knee angle, hip height, and ankle height asymmetries.
# The three weights intentionally sum to 0.8; the remaining 0.20 is reserved for
# a trunk/pelvis wobble component that is omitted in this simplified version.
_LANDING_KNEE_ASYMMETRY_WEIGHT = 0.35
_LANDING_HIP_HEIGHT_WEIGHT = 0.25
_LANDING_ANKLE_HEIGHT_WEIGHT = 0.20

# Core risk score composition weights. They intentionally sum to 1.0.
_CORE_WEIGHT_KNEE_STIFFNESS = 0.20
_CORE_WEIGHT_ANKLE_FOOT_ALIGNMENT = 0.30
_CORE_WEIGHT_HIP_DISPLACEMENT = 0.15
_CORE_WEIGHT_LANDING_ASYMMETRY = 0.35


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the inclusive range ``[low, high]``.

    Args:
        value: The value to clamp.
        low: The lower bound of the range.
        high: The upper bound of the range.

    Returns:
        The clamped value.
    """
    return max(low, min(high, value))


def distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two 3-D points.

    Args:
        a: A 3-D point ``(x, y, z)``.
        b: A 3-D point ``(x, y, z)``.

    Returns:
        The Euclidean distance between ``a`` and ``b``.

    Raises:
        AssertionError: If either point is not a 3-tuple.
    """
    assert len(a) == 3, "point a must be a 3-D coordinate"
    assert len(b) == 3, "point b must be a 3-D coordinate"
    return math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b)))


def midpoint(a: Point, b: Point) -> Point:
    """Return the midpoint of two 3-D points.

    Args:
        a: A 3-D point ``(x, y, z)``.
        b: A 3-D point ``(x, y, z)``.

    Returns:
        The 3-D midpoint ``((a_x+b_x)/2, (a_y+b_y)/2, (a_z+b_z)/2)``.

    Raises:
        AssertionError: If either point is not a 3-tuple.
    """
    assert len(a) == 3, "point a must be a 3-D coordinate"
    assert len(b) == 3, "point b must be a 3-D coordinate"
    return cast(Point, tuple((a_i + b_i) / 2.0 for a_i, b_i in zip(a, b)))


def angle_at(a: Point, b: Point, c: Point) -> float:
    """Return the internal angle (in degrees) formed at point ``b`` by ``a-b-c``.

    The angle is computed between vectors ``BA`` and ``BC``. If either vector is
    degenerate (zero length), the function returns ``0.0`` to avoid division by
    zero.

    Args:
        a: A 3-D point; the first arm of the angle.
        b: A 3-D point; the vertex of the angle.
        c: A 3-D point; the second arm of the angle.

    Returns:
        The angle in degrees, clamped to ``[0.0, 180.0]``.

    Raises:
        AssertionError: If any point is not a 3-tuple.
    """
    assert len(a) == 3, "point a must be a 3-D coordinate"
    assert len(b) == 3, "point b must be a 3-D coordinate"
    assert len(c) == 3, "point c must be a 3-D coordinate"
    ba = tuple(a_i - b_i for a_i, b_i in zip(a, b))
    bc = tuple(c_i - b_i for c_i, b_i in zip(c, b))
    ba_len = math.sqrt(sum(v ** 2 for v in ba))
    bc_len = math.sqrt(sum(v ** 2 for v in bc))
    denom = ba_len * bc_len
    # Tolerance avoids division by zero and numerical instability for near-zero vectors.
    if denom < 1e-8:
        return 0.0
    cos = clamp(sum(ba_i * bc_i for ba_i, bc_i in zip(ba, bc)) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def knee_flexion_angle(hip: Point, knee: Point, ankle: Point) -> float:
    """Return the knee flexion angle in degrees.

    The flexion angle is the internal angle at the knee formed by the hip,
    knee, and ankle landmarks.

    Args:
        hip: A 3-D point ``(x, y, z)`` representing the hip landmark.
        knee: A 3-D point ``(x, y, z)`` representing the knee landmark.
        ankle: A 3-D point ``(x, y, z)`` representing the ankle landmark.

    Returns:
        The knee flexion angle in degrees, clamped to ``[0.0, 180.0]``.

    Raises:
        AssertionError: If any landmark is not a 3-tuple.
    """
    return angle_at(hip, knee, ankle)


def knee_stiffness_risk(knee_angle: float, min_safe: float = 145.0, max_safe: float = 180.0) -> float:
    """Return 0–1 risk that the knee is too straight during loading.

    Angles at or below ``min_safe`` yield ``0.0`` (flexed enough). Angles at or
    above ``max_safe`` yield ``1.0`` (too straight). Values in between are
    linearly interpolated.

    Args:
        knee_angle: The knee flexion angle in degrees.
        min_safe: The angle below which risk is zero.
        max_safe: The angle above which risk is one.

    Returns:
        A normalized risk value in ``[0.0, 1.0]``.

    Raises:
        AssertionError: If ``max_safe`` is not greater than ``min_safe``.
    """
    assert max_safe > min_safe, "max_safe must be greater than min_safe"
    return clamp((knee_angle - min_safe) / (max_safe - min_safe), 0.0, 1.0)


def ankle_foot_alignment_risk(
    knee: Point,
    ankle: Point,
    heel: Point,
    foot_index: Point,
    leg_length: float,
) -> float:
    """Return normalized ankle-foot alignment risk in ``[0.0, 1.0]``.

    Risk combines two cues:

    1. **Knee-over-foot deviation** — the absolute medial-lateral (x-axis)
       distance between the knee and the center of the foot, expressed as a
       fraction of ``leg_length``.
    2. **Foot progression angle** — the absolute angle of the foot vector
       (``foot_index - heel``) relative to the anterior-posterior (z-axis),
       representing toe-in or toe-out.

    The x-axis is assumed to be the medial-lateral axis, the y-axis is the
    vertical axis, and the z-axis is the anterior-posterior / foot-progression
    axis. Angles are in degrees.

    Args:
        knee: A 3-D point ``(x, y, z)`` representing the knee landmark.
        ankle: A 3-D point ``(x, y, z)`` representing the ankle landmark.
        heel: A 3-D point ``(x, y, z)`` representing the heel landmark.
        foot_index: A 3-D point ``(x, y, z)`` representing the forefoot /
            metatarsal landmark.
        leg_length: Positive leg length, in the same units as the point
            coordinates.

    Returns:
        A normalized risk value in ``[0.0, 1.0]``.

    Raises:
        AssertionError: If any landmark is not a 3-tuple or if ``leg_length``
            is not positive.
    """
    assert len(knee) == 3, "knee must be a 3-D coordinate"
    assert len(ankle) == 3, "ankle must be a 3-D coordinate"
    assert len(heel) == 3, "heel must be a 3-D coordinate"
    assert len(foot_index) == 3, "foot_index must be a 3-D coordinate"
    assert leg_length > 0, "leg_length must be positive"

    foot_center = midpoint(heel, foot_index)
    knee_over_foot_deviation = abs(knee[0] - foot_center[0]) / leg_length
    knee_dev_score = clamp(
        knee_over_foot_deviation / _ANKLE_FOOT_KNEE_DEVIATION_THRESHOLD,
        0.0,
        1.0,
    )

    dx = foot_index[0] - heel[0]
    dz = foot_index[2] - heel[2]
    foot_angle = abs(math.degrees(math.atan2(dx, dz)))
    angle_score = clamp(
        foot_angle / _ANKLE_FOOT_FOOT_ANGLE_THRESHOLD,
        0.0,
        1.0,
    )

    return (
        _ANKLE_FOOT_KNEE_DEVIATION_WEIGHT * knee_dev_score
        + _ANKLE_FOOT_FOOT_ANGLE_WEIGHT * angle_score
    )


def hip_displacement_proxy(
    left_hip: Point,
    right_hip: Point,
    heel: Point,
    foot_index: Point,
    leg_length: float,
) -> float:
    """Return normalized pelvis displacement proxy in ``[0.0, 1.0]``.

    The proxy measures the horizontal distance between the pelvis center
    (midpoint of the two hips) and the foot base (midpoint of heel and forefoot)
    in the transverse plane (x-z), expressed as a fraction of ``leg_length``. A
    value of ``0.0`` indicates the pelvis is centered over the foot, while ``1.0``
    indicates the pelvis is displaced by at least one full leg length.

    Args:
        left_hip: A 3-D point ``(x, y, z)`` representing the left hip landmark.
        right_hip: A 3-D point ``(x, y, z)`` representing the right hip landmark.
        heel: A 3-D point ``(x, y, z)`` representing the heel landmark.
        foot_index: A 3-D point ``(x, y, z)`` representing the forefoot /
            metatarsal landmark.
        leg_length: Positive leg length, in the same units as the point
            coordinates.

    Returns:
        A normalized displacement proxy in ``[0.0, 1.0]``.

    Raises:
        AssertionError: If any landmark is not a 3-tuple or if ``leg_length``
            is not positive.
    """
    assert len(left_hip) == 3, "left_hip must be a 3-D coordinate"
    assert len(right_hip) == 3, "right_hip must be a 3-D coordinate"
    assert len(heel) == 3, "heel must be a 3-D coordinate"
    assert len(foot_index) == 3, "foot_index must be a 3-D coordinate"
    assert leg_length > 0, "leg_length must be positive"

    hip_center = midpoint(left_hip, right_hip)
    foot_center = midpoint(heel, foot_index)
    horizontal = math.sqrt(
        (hip_center[0] - foot_center[0]) ** 2 + (hip_center[2] - foot_center[2]) ** 2
    )
    return clamp(horizontal / leg_length / _HIP_DISPLACEMENT_THRESHOLD, 0.0, 1.0)


def landing_asymmetry_score(
    left_hip: Point,
    right_hip: Point,
    left_knee: Point,
    right_knee: Point,
    left_ankle: Point,
    right_ankle: Point,
) -> float:
    """Return a score for left-right imbalance during landing or a lunge.

    The score combines three cues:

    1. **Knee flexion asymmetry** — the absolute difference between left and
       right knee flexion angles, expressed as a fraction of
       ``_LANDING_KNEE_ANGLE_THRESHOLD`` degrees.
    2. **Pelvis obliquity** — the absolute vertical difference between the left
       and right hips, expressed as a fraction of
       ``_LANDING_HIP_HEIGHT_RELATIVE_THRESHOLD`` times the pelvis width.
    3. **Ankle height asymmetry** — the absolute vertical difference between the
       left and right ankles, expressed as a fraction of
       ``_LANDING_ANKLE_HEIGHT_RELATIVE_THRESHOLD`` times the average leg length.

    The three component weights sum to ``0.8`` because a fourth trunk/pelvis
    wobble component (weight ``0.20``) is omitted in this simplified version.
    Consequently, the returned score saturates at ``0.8`` when all three cues
    are at their individual maxima.

    Args:
        left_hip: A 3-D point ``(x, y, z)`` representing the left hip landmark.
        right_hip: A 3-D point ``(x, y, z)`` representing the right hip landmark.
        left_knee: A 3-D point ``(x, y, z)`` representing the left knee landmark.
        right_knee: A 3-D point ``(x, y, z)`` representing the right knee landmark.
        left_ankle: A 3-D point ``(x, y, z)`` representing the left ankle landmark.
        right_ankle: A 3-D point ``(x, y, z)`` representing the right ankle landmark.

    Returns:
        An asymmetry score in ``[0.0, 0.8]``.

    Raises:
        AssertionError: If any landmark is not a 3-tuple.
    """
    assert len(left_hip) == 3, "left_hip must be a 3-D coordinate"
    assert len(right_hip) == 3, "right_hip must be a 3-D coordinate"
    assert len(left_knee) == 3, "left_knee must be a 3-D coordinate"
    assert len(right_knee) == 3, "right_knee must be a 3-D coordinate"
    assert len(left_ankle) == 3, "left_ankle must be a 3-D coordinate"
    assert len(right_ankle) == 3, "right_ankle must be a 3-D coordinate"

    left_knee_angle = knee_flexion_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = knee_flexion_angle(right_hip, right_knee, right_ankle)
    knee_asym = clamp(
        abs(left_knee_angle - right_knee_angle) / _LANDING_KNEE_ANGLE_THRESHOLD,
        0.0,
        1.0,
    )

    pelvis_width = distance(left_hip, right_hip)
    hip_height_score = clamp(
        abs(left_hip[1] - right_hip[1]) / max(pelvis_width * _LANDING_HIP_HEIGHT_RELATIVE_THRESHOLD, 1e-6),
        0.0,
        1.0,
    )

    avg_leg_length = (
        distance(left_hip, left_knee)
        + distance(left_knee, left_ankle)
        + distance(right_hip, right_knee)
        + distance(right_knee, right_ankle)
    ) / 2.0
    ankle_height_score = clamp(
        abs(left_ankle[1] - right_ankle[1])
        / max(avg_leg_length * _LANDING_ANKLE_HEIGHT_RELATIVE_THRESHOLD, 1e-6),
        0.0,
        1.0,
    )

    return (
        _LANDING_KNEE_ASYMMETRY_WEIGHT * knee_asym
        + _LANDING_HIP_HEIGHT_WEIGHT * hip_height_score
        + _LANDING_ANKLE_HEIGHT_WEIGHT * ankle_height_score
    )


@dataclass(frozen=True)
class LowerBodyPose:
    """Bilateral lower-body landmarks in 3-D space.

    Points are ``(x, y, z)`` tuples where ``x`` is the medial-lateral axis,
    ``y`` is the vertical axis, and ``z`` is the anterior-posterior axis.
    Each side provides hip, knee, ankle, heel, and forefoot (foot_index)
    landmarks.
    """

    left_hip: Point
    right_hip: Point
    left_knee: Point
    right_knee: Point
    left_ankle: Point
    right_ankle: Point
    left_heel: Point
    right_heel: Point
    left_foot_index: Point
    right_foot_index: Point


def _leg_length(hip: Point, knee: Point, ankle: Point) -> float:
    """Return the hip-knee-ankle chain length for one leg."""
    return distance(hip, knee) + distance(knee, ankle)


def core_risk_score(pose: LowerBodyPose) -> dict:
    """Return the composed core injury-risk score and its sub-score breakdown.

    The function averages bilateral cues and combines four normalized
    sub-scores: knee stiffness, ankle-foot alignment, hip displacement, and
    landing asymmetry. The composition weights are module-level constants
    (``_CORE_WEIGHT_*``) and intentionally sum to ``1.0``.

    Args:
        pose: A ``LowerBodyPose`` containing 3-D landmarks for both legs.

    Returns:
        A dictionary with the following keys:

        - ``knee_stiffness_risk``: average bilateral knee-stiffness risk in
          ``[0.0, 1.0]``.
        - ``ankle_foot_alignment_risk``: average bilateral ankle-foot
          alignment risk in ``[0.0, 1.0]``.
        - ``hip_displacement_proxy``: average bilateral pelvis displacement
          proxy in ``[0.0, 1.0]``.
        - ``landing_asymmetry_score``: left-right landing asymmetry score in
          ``[0.0, 0.8]``.
        - ``core_risk``: weighted composite score clamped to ``[0.0, 1.0]``.
    """
    left_leg = _leg_length(pose.left_hip, pose.left_knee, pose.left_ankle)
    right_leg = _leg_length(pose.right_hip, pose.right_knee, pose.right_ankle)
    avg_leg = (left_leg + right_leg) / 2.0

    left_knee_angle = knee_flexion_angle(pose.left_hip, pose.left_knee, pose.left_ankle)
    right_knee_angle = knee_flexion_angle(pose.right_hip, pose.right_knee, pose.right_ankle)
    knee_stiffness = (knee_stiffness_risk(left_knee_angle) + knee_stiffness_risk(right_knee_angle)) / 2.0

    left_alignment = ankle_foot_alignment_risk(
        pose.left_knee, pose.left_ankle, pose.left_heel, pose.left_foot_index, avg_leg
    )
    right_alignment = ankle_foot_alignment_risk(
        pose.right_knee, pose.right_ankle, pose.right_heel, pose.right_foot_index, avg_leg
    )
    ankle_alignment = (left_alignment + right_alignment) / 2.0

    left_hip_disp = hip_displacement_proxy(
        pose.left_hip, pose.right_hip, pose.left_heel, pose.left_foot_index, avg_leg
    )
    right_hip_disp = hip_displacement_proxy(
        pose.left_hip, pose.right_hip, pose.right_heel, pose.right_foot_index, avg_leg
    )
    hip_disp = (left_hip_disp + right_hip_disp) / 2.0

    asym = landing_asymmetry_score(
        pose.left_hip, pose.right_hip,
        pose.left_knee, pose.right_knee,
        pose.left_ankle, pose.right_ankle,
    )

    core = (
        _CORE_WEIGHT_KNEE_STIFFNESS * knee_stiffness
        + _CORE_WEIGHT_ANKLE_FOOT_ALIGNMENT * ankle_alignment
        + _CORE_WEIGHT_HIP_DISPLACEMENT * hip_disp
        + _CORE_WEIGHT_LANDING_ASYMMETRY * asym
    )

    return {
        "knee_stiffness_risk": knee_stiffness,
        "ankle_foot_alignment_risk": ankle_alignment,
        "hip_displacement_proxy": hip_disp,
        "landing_asymmetry_score": asym,
        "core_risk": clamp(core, 0.0, 1.0),
    }
