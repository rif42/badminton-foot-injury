"""Small 3-D geometry helpers used by the baseline badminton injury-risk calculator.

All functions in this module assume points are 3-D coordinates represented as
``(x, y, z)`` tuples. Callers are responsible for providing correctly-shaped
inputs; ``distance`` and ``midpoint`` validate this at runtime.
"""

from __future__ import annotations

import math
from typing import cast

Point = tuple[float, float, float]


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
    """
    ba = tuple(a_i - b_i for a_i, b_i in zip(a, b))
    bc = tuple(c_i - b_i for c_i, b_i in zip(c, b))
    ba_len = math.sqrt(sum(v ** 2 for v in ba))
    bc_len = math.sqrt(sum(v ** 2 for v in bc))
    denom = ba_len * bc_len
    if denom < 1e-8:
        return 0.0
    cos = clamp(sum(ba_i * bc_i for ba_i, bc_i in zip(ba, bc)) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def knee_flexion_angle(hip: Point, knee: Point, ankle: Point) -> float:
    """Return the knee flexion angle in degrees."""
    return angle_at(hip, knee, ankle)


def knee_stiffness_risk(knee_angle: float, min_safe: float = 145.0, max_safe: float = 180.0) -> float:
    """Return 0–1 risk that the knee is too straight during loading."""
    return clamp((knee_angle - min_safe) / (max_safe - min_safe), 0.0, 1.0)
