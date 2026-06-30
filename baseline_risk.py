import math
from typing import Tuple

Point = Tuple[float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance(a: Point, b: Point) -> float:
    return math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b)))


def midpoint(a: Point, b: Point) -> Point:
    return tuple((a_i + b_i) / 2.0 for a_i, b_i in zip(a, b))


def angle_at(a: Point, b: Point, c: Point) -> float:
    ba = tuple(a_i - b_i for a_i, b_i in zip(a, b))
    bc = tuple(c_i - b_i for c_i, b_i in zip(c, b))
    ba_len = math.sqrt(sum(v ** 2 for v in ba))
    bc_len = math.sqrt(sum(v ** 2 for v in bc))
    denom = ba_len * bc_len
    if denom < 1e-8:
        return 0.0
    cos = clamp(sum(ba_i * bc_i for ba_i, bc_i in zip(ba, bc)) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cos))
