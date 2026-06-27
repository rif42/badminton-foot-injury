"""Injury-risk calculation for live lower-body pose landmarks.

This module converts MediaPipe Pose lower-body landmarks into four
biomechanical parameters, runs a weighted profile-based risk model, and
smooths the output over time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RiskProfile:
    """Configuration for one risk profile."""

    key: str
    label: str
    curves: Dict[str, List[Dict[str, float]]]
    weights: Dict[str, float]
    interactions: Dict[str, float]
    bands: Dict[str, int]

    @classmethod
    def from_preset(cls, key: str) -> "RiskProfile":
        """Load a built-in profile by key."""
        data = PROFILE_PRESETS[key]
        return cls(key=key, weights=WEIGHTS, **data)


@dataclass(frozen=True)
class RiskResult:
    """Output of one risk evaluation."""

    total_risk: float
    base_score: float
    interaction_score: float
    normalized: Dict[str, float]
    alerts: List[Dict[str, Any]]
    status: str
    profile_label: str


WEIGHTS = {
    "hip_trajectory_deviation": 0.15,
    "knee_flexion": 0.35,
    "foot_alignment": 0.25,
    "landing_pitch": 0.25,
}

PROFILE_PRESETS = {
    "conservative": {
        "label": "Conservative",
        "curves": {
            "hip_trajectory_deviation": [
                {"max": 8, "r": 0.05},
                {"max": 18, "r": 0.55},
                {"max": 30, "r": 0.8},
                {"max": float("inf"), "r": 0.95},
            ],
            "knee_flexion": [
                {"max": 140, "r": 0.05},
                {"max": 150, "r": 0.25},
                {"max": 160, "r": 0.6},
                {"max": float("inf"), "r": 0.9},
            ],
            "foot_alignment": [
                {"max": 8, "r": 0.05},
                {"max": 16, "r": 0.3},
                {"max": 24, "r": 0.65},
                {"max": float("inf"), "r": 0.9},
            ],
            "landing_pitch": [
                {"min": 0.001, "r": 0.05},
                {"min": -2, "r": 0.35},
                {"min": -6, "r": 0.7},
                {"min": float("-inf"), "r": 0.9},
            ],
        },
        "interactions": {
            "knee_x_hip": 14,
            "foot_x_pitch": 14,
            "knee_x_pitch": 10,
            "hip_x_foot": 8,
        },
        "bands": {"green_max": 24, "yellow_max": 49},
    },
    "balanced": {
        "label": "Balanced",
        "curves": {
            "hip_trajectory_deviation": [
                {"max": 10, "r": 0.05},
                {"max": 22, "r": 0.55},
                {"max": 35, "r": 0.8},
                {"max": float("inf"), "r": 0.95},
            ],
            "knee_flexion": [
                {"max": 145, "r": 0.05},
                {"max": 155, "r": 0.25},
                {"max": 165, "r": 0.6},
                {"max": float("inf"), "r": 0.9},
            ],
            "foot_alignment": [
                {"max": 10, "r": 0.05},
                {"max": 20, "r": 0.3},
                {"max": 30, "r": 0.65},
                {"max": float("inf"), "r": 0.9},
            ],
            "landing_pitch": [
                {"min": 0.001, "r": 0.05},
                {"min": -3, "r": 0.35},
                {"min": -8, "r": 0.7},
                {"min": float("-inf"), "r": 0.9},
            ],
        },
        "interactions": {
            "knee_x_hip": 12,
            "foot_x_pitch": 12,
            "knee_x_pitch": 8,
            "hip_x_foot": 6,
        },
        "bands": {"green_max": 29, "yellow_max": 59},
    },
    "aggressive": {
        "label": "Aggressive",
        "curves": {
            "hip_trajectory_deviation": [
                {"max": 12, "r": 0.05},
                {"max": 26, "r": 0.55},
                {"max": 40, "r": 0.8},
                {"max": float("inf"), "r": 0.95},
            ],
            "knee_flexion": [
                {"max": 150, "r": 0.05},
                {"max": 160, "r": 0.25},
                {"max": 170, "r": 0.6},
                {"max": float("inf"), "r": 0.9},
            ],
            "foot_alignment": [
                {"max": 12, "r": 0.05},
                {"max": 24, "r": 0.3},
                {"max": 34, "r": 0.65},
                {"max": float("inf"), "r": 0.9},
            ],
            "landing_pitch": [
                {"min": 0.001, "r": 0.05},
                {"min": -5, "r": 0.35},
                {"min": -10, "r": 0.7},
                {"min": float("-inf"), "r": 0.9},
            ],
        },
        "interactions": {
            "knee_x_hip": 10,
            "foot_x_pitch": 10,
            "knee_x_pitch": 6,
            "hip_x_foot": 4,
        },
        "bands": {"green_max": 34, "yellow_max": 69},
    },
}


def eval_piecewise_by_max(value: float, curve: list[dict[str, float]]) -> float:
    """Return the first risk value whose `max` threshold is >= ``value``."""
    for point in curve:
        if value <= point["max"]:
            return point["r"]
    return curve[-1]["r"]


def eval_piecewise_by_min(value: float, curve: list[dict[str, float]]) -> float:
    """Return the first risk value whose `min` threshold is <= ``value``."""
    for point in curve:
        if value >= point["min"]:
            return point["r"]
    return curve[-1]["r"]


def risk_level_from_normalized(r: float) -> dict[str, str]:
    """Map a normalized risk value to a human label and CSS-style class."""
    if r >= 0.7:
        return {"txt": "High", "cls": "bad"}
    if r >= 0.35:
        return {"txt": "Moderate", "cls": "warn"}
    return {"txt": "Low", "cls": "ok"}


def compute_base_score(
    normalized: dict[str, float], weights: dict[str, float]
) -> float:
    """Weighted sum of normalized parameter risks, scaled to 0-100."""
    return 100.0 * sum(weights[name] * normalized[name] for name in weights)


def compute_interaction_score(
    normalized: dict[str, float], interactions: dict[str, float]
) -> float:
    """Add synergy bonuses for dangerous parameter combinations."""
    return (
        interactions["knee_x_hip"]
        * normalized["knee_flexion"]
        * normalized["hip_trajectory_deviation"]
        + interactions["foot_x_pitch"]
        * normalized["foot_alignment"]
        * normalized["landing_pitch"]
        + interactions["knee_x_pitch"]
        * normalized["knee_flexion"]
        * normalized["landing_pitch"]
        + interactions["hip_x_foot"]
        * normalized["hip_trajectory_deviation"]
        * normalized["foot_alignment"]
    )


def compute_total_risk(base_score: float, interaction_score: float) -> float:
    """Clamp total risk to the 0-100 range."""
    return min(base_score + interaction_score, 100.0)


REQUIRED_JOINTS = {
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

VISIBILITY_THRESHOLD = 0.5


def has_required_landmarks(
    landmarks: dict[str, tuple[float, float, float, float]]
) -> bool:
    """Return True when all required joints are present and visible enough."""
    for joint in REQUIRED_JOINTS:
        if joint not in landmarks:
            return False
        if landmarks[joint][3] < VISIBILITY_THRESHOLD:
            return False
    return True


def _to_px(
    point: tuple[float, float, float, float], frame_shape: tuple[int, int]
) -> tuple[float, float]:
    """Convert a normalized landmark to pixel coordinates."""
    height, width = frame_shape[:2]
    return point[0] * width, point[1] * height


def _angle_between(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    """Internal angle ABC in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _vector_angle(v: tuple[float, float]) -> float:
    """Angle of vector ``v`` relative to the positive x-axis, in degrees."""
    return math.degrees(math.atan2(v[1], v[0]))


def front_leg_side(landmarks: dict[str, tuple[float, float, float, float]]) -> str:
    """Return 'left' or 'right' for the side whose ankle is lower in the frame."""
    left_y = landmarks["left_ankle"][1]
    right_y = landmarks["right_ankle"][1]
    return "right" if right_y >= left_y else "left"


def compute_knee_flexion(
    landmarks: dict[str, tuple[float, float, float, float]],
    side: str,
    frame_shape: tuple[int, int],
) -> float:
    """Internal angle at the knee (hip-knee-ankle) in degrees."""
    hip = _to_px(landmarks[f"{side}_hip"], frame_shape)
    knee = _to_px(landmarks[f"{side}_knee"], frame_shape)
    ankle = _to_px(landmarks[f"{side}_ankle"], frame_shape)
    return _angle_between(hip, knee, ankle)


def compute_landing_pitch(
    landmarks: dict[str, tuple[float, float, float, float]],
    side: str,
    frame_shape: tuple[int, int],
) -> float:
    """Angle of the foot vector relative to horizontal.

    Negative means the heel is lower than the toe (heel-first landing in
    image coordinates). Positive means toe-first.
    """
    heel = _to_px(landmarks[f"{side}_heel"], frame_shape)
    toe = _to_px(landmarks[f"{side}_foot_index"], frame_shape)
    return _vector_angle((toe[0] - heel[0], toe[1] - heel[1]))


def compute_foot_alignment(
    landmarks: dict[str, tuple[float, float, float, float]],
    side: str,
    frame_shape: tuple[int, int],
    hip_velocity: tuple[float, float],
) -> float:
    """Absolute angular difference between the foot vector and hip velocity."""
    heel = _to_px(landmarks[f"{side}_heel"], frame_shape)
    toe = _to_px(landmarks[f"{side}_foot_index"], frame_shape)
    foot_angle = _vector_angle((toe[0] - heel[0], toe[1] - heel[1]))
    velocity_angle = _vector_angle(hip_velocity)
    diff = abs((foot_angle - velocity_angle + 180) % 360 - 180)
    return diff


def compute_hip_trajectory_deviation(
    hip_history: list[tuple[float, float]]
) -> Optional[float]:
    """Absolute angle between hip velocity and vertical, in degrees."""
    if len(hip_history) < 2:
        return None
    start = hip_history[0]
    end = hip_history[-1]
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    if vx == 0 and vy == 0:
        return 0.0
    velocity_angle = _vector_angle((vx, vy))
    # In image coordinates, straight up is -90 degrees.
    vertical_angle = -90.0
    diff = abs((velocity_angle - vertical_angle + 180) % 360 - 180)
    return diff
