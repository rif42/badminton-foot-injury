"""Injury-risk calculation for live lower-body pose landmarks.

This module converts MediaPipe Pose lower-body landmarks into four
biomechanical parameters, runs a weighted profile-based risk model, and
smooths the output over time.
"""

from __future__ import annotations

import math
from collections import deque
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

# Body-scale estimation uses the average hip-to-ankle distance (pixels).
BODY_SCALE_JOINTS: Tuple[Tuple[str, str], ...] = (
    ("left_hip", "left_ankle"),
    ("right_hip", "right_ankle"),
)

# Idle gate: normalized hip movement below this fraction of body scale is
# treated as no meaningful athletic movement.
IDLE_MOVEMENT_THRESHOLD = 0.015


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


def compute_body_scale(
    landmarks: dict[str, tuple[float, float, float, float]],
    frame_shape: tuple[int, int],
) -> Optional[float]:
    """Estimate body scale as the average hip-to-ankle distance in pixels.

    Falls back to whichever side is visible so that the helper is usable
    even when one leg is partially out of frame.
    """
    lengths: list[float] = []
    for hip_name, ankle_name in BODY_SCALE_JOINTS:
        if hip_name not in landmarks or ankle_name not in landmarks:
            continue
        hip = _to_px(landmarks[hip_name], frame_shape)
        ankle = _to_px(landmarks[ankle_name], frame_shape)
        lengths.append(math.hypot(hip[0] - ankle[0], hip[1] - ankle[1]))
    if not lengths:
        return None
    return sum(lengths) / len(lengths)


def compute_normalized_hip_movement(
    hip_history: list[tuple[float, float]],
    body_scale: float,
) -> Optional[float]:
    """Hip displacement over the history window normalized by body scale.

    Returns None when there is not enough history or the scale is invalid.
    """
    if len(hip_history) < 2 or body_scale <= 0:
        return None
    start = hip_history[0]
    end = hip_history[-1]
    displacement = math.hypot(end[0] - start[0], end[1] - start[1])
    return displacement / body_scale


def is_idle(
    landmarks: dict[str, tuple[float, float, float, float]],
    frame_shape: tuple[int, int],
    hip_history: list[tuple[float, float]],
    threshold: float = IDLE_MOVEMENT_THRESHOLD,
) -> bool:
    """Return True when the athlete is essentially still.

    Movement is normalized by body scale so the gate is resolution and
    distance invariant.
    """
    body_scale = compute_body_scale(landmarks, frame_shape)
    if body_scale is None:
        return False
    movement = compute_normalized_hip_movement(hip_history, body_scale)
    if movement is None:
        return False
    return movement < threshold


class RiskModel:
    """Maintains pose history and produces smoothed injury-risk scores."""

    def __init__(
        self,
        profile_key: str = "balanced",
        *,
        hip_history_size: int = 5,
        smoothing_window: int = 3,
    ) -> None:
        self.profile = RiskProfile.from_preset(profile_key)
        self._hip_history_size = hip_history_size
        self._smoothing_window = smoothing_window
        self._hip_history: deque[tuple[float, float]] = deque(maxlen=hip_history_size)
        self._result_history: deque[RiskResult] = deque(maxlen=smoothing_window)
        self._was_idle: bool = False

    def set_profile(self, profile_key: str) -> None:
        """Switch to a built-in profile."""
        if profile_key not in PROFILE_PRESETS:
            raise KeyError(f"Unknown profile: {profile_key}")
        self.profile = RiskProfile.from_preset(profile_key)

    def cycle_profile(self) -> None:
        """Cycle through balanced -> conservative -> aggressive -> balanced."""
        order = ["balanced", "conservative", "aggressive"]
        idx = order.index(self.profile.key)
        next_idx = (idx + 1) % len(order)
        self.set_profile(order[next_idx])

    def _idle_result(self) -> RiskResult:
        """A first-class result emitted when the athlete is not moving."""
        zero_norm = {name: 0.0 for name in self.profile.weights}
        alerts = [
            {"label": "Hip Control", "txt": "Low", "cls": "ok"},
            {"label": "Knee Overload", "txt": "Low", "cls": "ok"},
            {"label": "Foot Alignment", "txt": "Low", "cls": "ok"},
            {"label": "Kinetic Shock", "txt": "Low", "cls": "ok"},
        ]
        return RiskResult(
            total_risk=0.0,
            base_score=0.0,
            interaction_score=0.0,
            normalized=zero_norm,
            alerts=alerts,
            status="Idle",
            profile_label=self.profile.label,
        )

    def _reset_smoothing(self) -> None:
        """Clear the smoothing window so state transitions do not mix values."""
        self._result_history.clear()

    def update(
        self,
        landmarks: dict[str, tuple[float, float, float, float]],
        frame_shape: tuple[int, int],
    ) -> Optional[RiskResult]:
        """Evaluate risk for the current frame.

        Returns ``None`` when required landmarks are missing or the hip
        history has not yet accumulated enough samples.
        """
        if not has_required_landmarks(landmarks):
            return None

        hip_center = self._hip_center_px(landmarks, frame_shape)
        self._hip_history.append(hip_center)

        if len(self._hip_history) < 2:
            return None

        currently_idle = is_idle(landmarks, frame_shape, list(self._hip_history))

        if currently_idle:
            if not self._was_idle:
                self._reset_smoothing()
            result = self._idle_result()
            self._result_history.append(result)
            self._was_idle = True
            return self._smooth_result()

        # Dynamic branch
        if self._was_idle:
            self._reset_smoothing()

        hip_velocity = self._compute_hip_velocity()
        if hip_velocity is None:
            return None

        params = self._compute_parameters(landmarks, frame_shape, hip_velocity)
        result = self._compute_risk(params)
        self._result_history.append(result)
        self._was_idle = False
        return self._smooth_result()

    def _hip_center_px(
        self,
        landmarks: dict[str, tuple[float, float, float, float]],
        frame_shape: tuple[int, int],
    ) -> tuple[float, float]:
        left = _to_px(landmarks["left_hip"], frame_shape)
        right = _to_px(landmarks["right_hip"], frame_shape)
        return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)

    def _compute_hip_velocity(self) -> Optional[tuple[float, float]]:
        if len(self._hip_history) < 2:
            return None
        start = self._hip_history[0]
        end = self._hip_history[-1]
        return (end[0] - start[0], end[1] - start[1])

    def _compute_parameters(
        self,
        landmarks: dict[str, tuple[float, float, float, float]],
        frame_shape: tuple[int, int],
        hip_velocity: tuple[float, float],
    ) -> dict[str, float]:
        side = front_leg_side(landmarks)
        hip_deviation = compute_hip_trajectory_deviation(list(self._hip_history))
        if hip_deviation is None:
            hip_deviation = 0.0
        return {
            "hip_trajectory_deviation": hip_deviation,
            "knee_flexion": compute_knee_flexion(landmarks, side, frame_shape),
            "foot_alignment": compute_foot_alignment(
                landmarks, side, frame_shape, hip_velocity
            ),
            "landing_pitch": compute_landing_pitch(landmarks, side, frame_shape),
        }

    def _compute_risk(self, params: dict[str, float]) -> RiskResult:
        profile = self.profile
        normalized: dict[str, float] = {}
        for name, value in params.items():
            curve = profile.curves[name]
            if name == "landing_pitch":
                normalized[name] = eval_piecewise_by_min(value, curve)
            else:
                normalized[name] = eval_piecewise_by_max(value, curve)

        base_score = compute_base_score(normalized, profile.weights)
        interaction_score = compute_interaction_score(normalized, profile.interactions)
        total = compute_total_risk(base_score, interaction_score)

        bands = profile.bands
        if total <= bands["green_max"]:
            status = "Optimal"
        elif total <= bands["yellow_max"]:
            status = "Caution"
        else:
            status = "High Risk"

        alerts = [
            {
                "label": "Hip Control",
                **risk_level_from_normalized(normalized["hip_trajectory_deviation"]),
            },
            {
                "label": "Knee Overload",
                **risk_level_from_normalized(normalized["knee_flexion"]),
            },
            {
                "label": "Foot Alignment",
                **risk_level_from_normalized(normalized["foot_alignment"]),
            },
            {
                "label": "Kinetic Shock",
                **risk_level_from_normalized(normalized["landing_pitch"]),
            },
        ]

        return RiskResult(
            total_risk=total,
            base_score=base_score,
            interaction_score=interaction_score,
            normalized=normalized,
            alerts=alerts,
            status=status,
            profile_label=profile.label,
        )

    def _smooth_result(self) -> RiskResult:
        if len(self._result_history) == 1:
            return self._result_history[0]

        n = len(self._result_history)
        total = sum(r.total_risk for r in self._result_history) / n
        base = sum(r.base_score for r in self._result_history) / n
        interaction = sum(r.interaction_score for r in self._result_history) / n
        latest = self._result_history[-1]

        return RiskResult(
            total_risk=total,
            base_score=base,
            interaction_score=interaction,
            normalized=latest.normalized,
            alerts=latest.alerts,
            status=latest.status,
            profile_label=latest.profile_label,
        )
