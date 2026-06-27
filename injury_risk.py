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
