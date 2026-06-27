"""Injury-risk calculation for live lower-body pose landmarks.

This module converts MediaPipe Pose lower-body landmarks into four
biomechanical parameters, runs a weighted profile-based risk model, and
smooths the output over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


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
