"""Small 3-D geometry helpers used by the baseline badminton injury-risk calculator.

All functions in this module assume points are 3-D coordinates represented as
``(x, y, z)`` tuples. Callers are responsible for providing correctly-shaped
inputs; ``distance`` and ``midpoint`` validate this at runtime.
"""

from __future__ import annotations

import math
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
