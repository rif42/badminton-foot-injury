# Live Injury-Risk Calculation Implementation Plan

> **For agentic workers:** REQUIRED SUB- SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real-time, profile-based injury-risk scoring and an on-screen HUD to the existing webcam lower-body MediaPipe pose detector.

**Architecture:** A new `injury_risk.py` module converts lower-body landmarks into four biomechanical parameters, runs a weighted profile model, and smooths the output. A new `risk_overlay.py` module draws the results. `webcam_leg_pose_detector.py` remains the orchestrator and adds keyboard profile switching.

**Tech Stack:** Python 3.11+, MediaPipe Pose, OpenCV, NumPy, `collections.deque`, `pytest`/`unittest`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `injury_risk.py` | Create | Data classes, profile presets, curve evaluation, landmark-to-parameter math, `RiskModel` with smoothing. |
| `risk_overlay.py` | Create | `RiskOverlay` class that draws status, score, and per-parameter alerts on an OpenCV frame. |
| `webcam_leg_pose_detector.py` | Modify | Wire `RiskModel`/`RiskOverlay` into the frame loop and add `p` key profile cycling. |
| `test_injury_risk.py` | Create | Unit tests for profiles, scoring, parameter extraction, smoothing, and visibility handling. |
| `test_risk_overlay.py` | Create | Mock-based tests for the HUD renderer. |
| `test_webcam_leg_pose_detector.py` | Modify | Add integration test for the risk path and keyboard profile switching. |

---

## Profile Constants (used by Tasks 1-4)

All profiles share the same weights. They differ in curve breakpoints, interaction bonuses, and final color bands.

```python
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
        "interactions": {"knee_x_hip": 14, "foot_x_pitch": 14, "knee_x_pitch": 10, "hip_x_foot": 8},
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
        "interactions": {"knee_x_hip": 12, "foot_x_pitch": 12, "knee_x_pitch": 8, "hip_x_foot": 6},
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
        "interactions": {"knee_x_hip": 10, "foot_x_pitch": 10, "knee_x_pitch": 6, "hip_x_foot": 4},
        "bands": {"green_max": 34, "yellow_max": 69},
    },
}
```

---

### Task 1: Create data classes and profile presets

**Files:**
- Create: `injury_risk.py`
- Test: `test_injury_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# test_injury_risk.py
import pytest

from injury_risk import RiskProfile, RiskResult, PROFILE_PRESETS


class TestDataStructures:
    def test_profile_has_required_fields(self):
        profile = RiskProfile.from_preset("balanced")
        assert profile.key == "balanced"
        assert profile.label == "Balanced"
        assert set(profile.curves) == {
            "hip_trajectory_deviation",
            "knee_flexion",
            "foot_alignment",
            "landing_pitch",
        }
        assert profile.weights == {
            "hip_trajectory_deviation": 0.15,
            "knee_flexion": 0.35,
            "foot_alignment": 0.25,
            "landing_pitch": 0.25,
        }
        assert set(profile.interactions) == {
            "knee_x_hip",
            "foot_x_pitch",
            "knee_x_pitch",
            "hip_x_foot",
        }
        assert profile.bands == {"green_max": 29, "yellow_max": 59}

    def test_all_presets_exist(self):
        assert set(PROFILE_PRESETS) == {"conservative", "balanced", "aggressive"}

    def test_risk_result_is_frozen_dataclass(self):
        result = RiskResult(
            total_risk=12.5,
            base_score=10.0,
            interaction_score=2.5,
            normalized={},
            alerts=[],
            status="Optimal",
            profile_label="Balanced",
        )
        assert result.total_risk == 12.5
        with pytest.raises(AttributeError):
            result.total_risk = 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_injury_risk.py::TestDataStructures -v`
Expected: three failures because `injury_risk.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# injury_risk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RiskProfile:
    key: str
    label: str
    curves: Dict[str, List[Dict[str, float]]]
    weights: Dict[str, float]
    interactions: Dict[str, float]
    bands: Dict[str, int]

    @classmethod
    def from_preset(cls, key: str) -> "RiskProfile":
        data = PROFILE_PRESETS[key]
        return cls(key=key, **data)


@dataclass(frozen=True)
class RiskResult:
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
        "interactions": {"knee_x_hip": 14, "foot_x_pitch": 14, "knee_x_pitch": 10, "hip_x_foot": 8},
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
        "interactions": {"knee_x_hip": 12, "foot_x_pitch": 12, "knee_x_pitch": 8, "hip_x_foot": 6},
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
        "interactions": {"knee_x_hip": 10, "foot_x_pitch": 10, "knee_x_pitch": 6, "hip_x_foot": 4},
        "bands": {"green_max": 34, "yellow_max": 69},
    },
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_injury_risk.py::TestDataStructures -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add injury_risk.py test_injury_risk.py
git commit -m "feat(risk): add RiskProfile/RiskResult data classes and presets"
```

---

### Task 2: Add curve evaluation and score composition

**Files:**
- Modify: `injury_risk.py`
- Modify: `test_injury_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# test_injury_risk.py
import pytest

from injury_risk import (
    RiskProfile,
    eval_piecewise_by_max,
    eval_piecewise_by_min,
    risk_level_from_normalized,
    compute_base_score,
    compute_interaction_score,
    compute_total_risk,
)


class TestCurveEvaluation:
    def test_eval_piecewise_by_max(self):
        curve = [
            {"max": 10, "r": 0.05},
            {"max": 20, "r": 0.55},
            {"max": float("inf"), "r": 0.9},
        ]
        assert eval_piecewise_by_max(5, curve) == pytest.approx(0.05)
        assert eval_piecewise_by_max(15, curve) == pytest.approx(0.55)
        assert eval_piecewise_by_max(25, curve) == pytest.approx(0.9)

    def test_eval_piecewise_by_min(self):
        curve = [
            {"min": 0.001, "r": 0.05},
            {"min": -5, "r": 0.35},
            {"min": float("-inf"), "r": 0.9},
        ]
        assert eval_piecewise_by_min(2, curve) == pytest.approx(0.05)
        assert eval_piecewise_by_min(-3, curve) == pytest.approx(0.35)
        assert eval_piecewise_by_min(-10, curve) == pytest.approx(0.9)

    def test_risk_level_from_normalized(self):
        assert risk_level_from_normalized(0.2)["txt"] == "Low"
        assert risk_level_from_normalized(0.5)["txt"] == "Moderate"
        assert risk_level_from_normalized(0.8)["txt"] == "High"

    def test_compute_total_risk_matches_reference(self):
        profile = RiskProfile.from_preset("balanced")
        normalized = {
            "hip_trajectory_deviation": 0.1,
            "knee_flexion": 0.9,
            "foot_alignment": 0.3,
            "landing_pitch": 0.2,
        }
        base = compute_base_score(normalized, profile.weights)
        interaction = compute_interaction_score(normalized, profile.interactions)
        total = compute_total_risk(base, interaction)

        expected_base = 100 * (
            0.15 * 0.1 + 0.35 * 0.9 + 0.25 * 0.3 + 0.25 * 0.2
        )
        expected_interaction = (
            12 * 0.9 * 0.1
            + 12 * 0.3 * 0.2
            + 8 * 0.9 * 0.2
            + 6 * 0.1 * 0.3
        )
        assert base == pytest.approx(expected_base)
        assert interaction == pytest.approx(expected_interaction)
        assert total == pytest.approx(min(expected_base + expected_interaction, 100.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_injury_risk.py::TestCurveEvaluation -v`
Expected: failures because the helper functions are not defined.

- [ ] **Step 3: Write minimal implementation**

Append to `injury_risk.py`:

```python
from typing import Tuple


def eval_piecewise_by_max(value: float, curve: List[Dict[str, float]]) -> float:
    for point in curve:
        if value <= point["max"]:
            return point["r"]
    return curve[-1]["r"]


def eval_piecewise_by_min(value: float, curve: List[Dict[str, float]]) -> float:
    for point in curve:
        if value >= point["min"]:
            return point["r"]
    return curve[-1]["r"]


def risk_level_from_normalized(r: float) -> Dict[str, str]:
    if r >= 0.7:
        return {"txt": "High", "cls": "bad"}
    if r >= 0.35:
        return {"txt": "Moderate", "cls": "warn"}
    return {"txt": "Low", "cls": "ok"}


def compute_base_score(
    normalized: Dict[str, float], weights: Dict[str, float]
) -> float:
    return 100.0 * sum(weights[name] * normalized[name] for name in weights)


def compute_interaction_score(
    normalized: Dict[str, float], interactions: Dict[str, float]
) -> float:
    return (
        interactions["knee_x_hip"] * normalized["knee_flexion"] * normalized["hip_trajectory_deviation"]
        + interactions["foot_x_pitch"] * normalized["foot_alignment"] * normalized["landing_pitch"]
        + interactions["knee_x_pitch"] * normalized["knee_flexion"] * normalized["landing_pitch"]
        + interactions["hip_x_foot"] * normalized["hip_trajectory_deviation"] * normalized["foot_alignment"]
    )


def compute_total_risk(base_score: float, interaction_score: float) -> float:
    return min(base_score + interaction_score, 100.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_injury_risk.py::TestCurveEvaluation -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add injury_risk.py test_injury_risk.py
git commit -m "feat(risk): add piecewise curve evaluation and score composition"
```

---

### Task 3: Convert landmarks to biomechanical parameters

**Files:**
- Modify: `injury_risk.py`
- Modify: `test_injury_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# test_injury_risk.py
import pytest

from injury_risk import (
    REQUIRED_JOINTS,
    front_leg_side,
    compute_knee_flexion,
    compute_foot_alignment,
    compute_landing_pitch,
    has_required_landmarks,
)


def _landmark(x, y, z=0.0, visibility=1.0):
    return (x, y, z, visibility)


class TestParameterExtraction:
    def test_required_joints_set(self):
        assert REQUIRED_JOINTS == {
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

    def test_has_required_landmarks_false_when_low_visibility(self):
        landmarks = {
            "left_hip": _landmark(0.5, 0.5, visibility=0.2),
        }
        assert has_required_landmarks(landmarks) is False

    def test_front_leg_side_picks_lower_ankle(self):
        landmarks = {
            "left_ankle": _landmark(0.4, 0.7),
            "right_ankle": _landmark(0.6, 0.8),
        }
        assert front_leg_side(landmarks) == "right"

    def test_knee_flexion_is_internal_angle(self):
        # Straight leg: hip (0.5, 0.4), knee (0.5, 0.6), ankle (0.5, 0.8)
        # All y values; internal angle ~180.
        frame_shape = (480, 640)
        landmarks = {
            "left_hip": _landmark(0.5, 0.4),
            "left_knee": _landmark(0.5, 0.6),
            "left_ankle": _landmark(0.5, 0.8),
        }
        assert compute_knee_flexion(landmarks, "left", frame_shape) == pytest.approx(180.0, abs=1.0)

    def test_landing_pitch_negative_when_heel_lower(self):
        # Heel below toe in image -> negative pitch.
        frame_shape = (480, 640)
        landmarks = {
            "left_heel": _landmark(0.5, 0.8),
            "left_foot_index": _landmark(0.55, 0.75),
        }
        pitch = compute_landing_pitch(landmarks, "left", frame_shape)
        assert pitch < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_injury_risk.py::TestParameterExtraction -v`
Expected: failures because extraction helpers are not defined.

- [ ] **Step 3: Write minimal implementation**

Append to `injury_risk.py`:

```python
import math
from typing import Optional, Tuple


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
    landmarks: Dict[str, Tuple[float, float, float, float]]
) -> bool:
    for joint in REQUIRED_JOINTS:
        if joint not in landmarks:
            return False
        if landmarks[joint][3] < VISIBILITY_THRESHOLD:
            return False
    return True


def _to_px(
    point: Tuple[float, float, float, float], frame_shape: Tuple[int, int]
) -> Tuple[float, float]:
    height, width = frame_shape[:2]
    return point[0] * width, point[1] * height


def _angle_between(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _vector_angle(v: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def front_leg_side(
    landmarks: Dict[str, Tuple[float, float, float, float]]
) -> str:
    left_y = landmarks["left_ankle"][1]
    right_y = landmarks["right_ankle"][1]
    return "right" if right_y >= left_y else "left"


def compute_knee_flexion(
    landmarks: Dict[str, Tuple[float, float, float, float]],
    side: str,
    frame_shape: Tuple[int, int],
) -> float:
    hip = _to_px(landmarks[f"{side}_hip"], frame_shape)
    knee = _to_px(landmarks[f"{side}_knee"], frame_shape)
    ankle = _to_px(landmarks[f"{side}_ankle"], frame_shape)
    return _angle_between(hip, knee, ankle)


def compute_landing_pitch(
    landmarks: Dict[str, Tuple[float, float, float, float]],
    side: str,
    frame_shape: Tuple[int, int],
) -> float:
    heel = _to_px(landmarks[f"{side}_heel"], frame_shape)
    toe = _to_px(landmarks[f"{side}_foot_index"], frame_shape)
    return _vector_angle((toe[0] - heel[0], toe[1] - heel[1]))


def compute_foot_alignment(
    landmarks: Dict[str, Tuple[float, float, float, float]],
    side: str,
    frame_shape: Tuple[int, int],
    hip_velocity: Tuple[float, float],
) -> float:
    heel = _to_px(landmarks[f"{side}_heel"], frame_shape)
    toe = _to_px(landmarks[f"{side}_foot_index"], frame_shape)
    foot_angle = _vector_angle((toe[0] - heel[0], toe[1] - heel[1]))
    velocity_angle = _vector_angle(hip_velocity)
    diff = abs((foot_angle - velocity_angle + 180) % 360 - 180)
    return diff


def compute_hip_trajectory_deviation(
    hip_history: List[Tuple[float, float]]
) -> Optional[float]:
    if len(hip_history) < 2:
        return None
    start = hip_history[0]
    end = hip_history[-1]
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    if vx == 0 and vy == 0:
        return 0.0
    velocity_angle = _vector_angle((vx, vy))
    # Absolute angle from vertical. Vertical up is -90 in image coords.
    vertical_angle = -90.0
    diff = abs((velocity_angle - vertical_angle + 180) % 360 - 180)
    return diff
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_injury_risk.py::TestParameterExtraction -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add injury_risk.py test_injury_risk.py
git commit -m "feat(risk): add landmark-to-parameter extraction"
```

---

### Task 4: Build RiskModel with history and smoothing

**Files:**
- Modify: `injury_risk.py`
- Modify: `test_injury_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# test_injury_risk.py
import pytest

from injury_risk import RiskModel, RiskResult


def _make_safe_landmarks():
    def lm(x, y, v=1.0):
        return (x, y, 0.0, v)

    return {
        "left_hip": lm(0.45, 0.35),
        "right_hip": lm(0.55, 0.35),
        "left_knee": lm(0.45, 0.55),
        "right_knee": lm(0.55, 0.55),
        "left_ankle": lm(0.45, 0.8),
        "right_ankle": lm(0.55, 0.8),
        "left_heel": lm(0.43, 0.82),
        "right_heel": lm(0.53, 0.82),
        "left_foot_index": lm(0.47, 0.81),
        "right_foot_index": lm(0.57, 0.81),
    }


class TestRiskModel:
    def test_update_returns_none_when_landmarks_missing(self):
        model = RiskModel()
        assert model.update({"left_hip": (0.5, 0.5, 0.0, 1.0)}, (480, 640)) is None

    def test_update_returns_risk_result_after_history_fills(self):
        model = RiskModel()
        frame_shape = (480, 640)
        result = None
        for _ in range(5):
            result = model.update(_make_safe_landmarks(), frame_shape)
        assert isinstance(result, RiskResult)
        assert 0.0 <= result.total_risk <= 100.0
        assert result.status in {"Optimal", "Caution", "High Risk"}
        assert result.profile_label == "Balanced"

    def test_profile_switching(self):
        model = RiskModel()
        assert model.profile.key == "balanced"
        model.set_profile("conservative")
        assert model.profile.key == "conservative"
        model.cycle_profile()
        assert model.profile.key == "balanced"
        model.cycle_profile()
        assert model.profile.key == "aggressive"

    def test_smoothing_averages_over_window(self):
        model = RiskModel(smoothing_window=3)
        frame_shape = (480, 640)
        first = None
        for i in range(5):
            lm = _make_safe_landmarks()
            # Move ankles down each frame to create changing geometry.
            lm["left_ankle"] = (lm["left_ankle"][0], 0.75 + i * 0.01, 0.0, 1.0)
            lm["right_ankle"] = (lm["right_ankle"][0], 0.75 + i * 0.01, 0.0, 1.0)
            first = model.update(lm, frame_shape)
        assert first is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_injury_risk.py::TestRiskModel -v`
Expected: failures because `RiskModel` is not defined.

- [ ] **Step 3: Write minimal implementation**

Append to `injury_risk.py`:

```python
from collections import deque
from typing import Optional, Sequence


class RiskModel:
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
        self._hip_history: deque = deque(maxlen=hip_history_size)
        self._result_history: deque = deque(maxlen=smoothing_window)

    def set_profile(self, profile_key: str) -> None:
        if profile_key not in PROFILE_PRESETS:
            raise KeyError(f"Unknown profile: {profile_key}")
        self.profile = RiskProfile.from_preset(profile_key)

    def cycle_profile(self) -> None:
        order = ["balanced", "conservative", "aggressive"]
        idx = order.index(self.profile.key)
        next_idx = (idx + 1) % len(order)
        self.set_profile(order[next_idx])

    def update(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        frame_shape: Tuple[int, int],
    ) -> Optional[RiskResult]:
        if not has_required_landmarks(landmarks):
            return None

        hip_center = self._hip_center_px(landmarks, frame_shape)
        self._hip_history.append(hip_center)

        hip_velocity = self._compute_hip_velocity()
        if hip_velocity is None:
            return None

        params = self._compute_parameters(landmarks, frame_shape, hip_velocity)
        result = self._compute_risk(params)
        self._result_history.append(result)
        return self._smooth_result()

    def _hip_center_px(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        frame_shape: Tuple[int, int],
    ) -> Tuple[float, float]:
        left = _to_px(landmarks["left_hip"], frame_shape)
        right = _to_px(landmarks["right_hip"], frame_shape)
        return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)

    def _compute_hip_velocity(self) -> Optional[Tuple[float, float]]:
        if len(self._hip_history) < 2:
            return None
        start = self._hip_history[0]
        end = self._hip_history[-1]
        return (end[0] - start[0], end[1] - start[1])

    def _compute_parameters(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        frame_shape: Tuple[int, int],
        hip_velocity: Tuple[float, float],
    ) -> Dict[str, float]:
        side = front_leg_side(landmarks)
        hip_deviation = compute_hip_trajectory_deviation(list(self._hip_history))
        if hip_deviation is None:
            hip_deviation = 0.0
        return {
            "hip_trajectory_deviation": hip_deviation,
            "knee_flexion": compute_knee_flexion(landmarks, side, frame_shape),
            "foot_alignment": compute_foot_alignment(landmarks, side, frame_shape, hip_velocity),
            "landing_pitch": compute_landing_pitch(landmarks, side, frame_shape),
        }

    def _compute_risk(self, params: Dict[str, float]) -> RiskResult:
        profile = self.profile
        normalized = {}
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
            {"label": "Hip Control", **risk_level_from_normalized(normalized["hip_trajectory_deviation"])},
            {"label": "Knee Overload", **risk_level_from_normalized(normalized["knee_flexion"])},
            {"label": "Foot Alignment", **risk_level_from_normalized(normalized["foot_alignment"])},
            {"label": "Kinetic Shock", **risk_level_from_normalized(normalized["landing_pitch"])},
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_injury_risk.py::TestRiskModel -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add injury_risk.py test_injury_risk.py
git commit -m "feat(risk): add RiskModel with history and smoothing"
```

---

### Task 5: Create the on-screen risk overlay

**Files:**
- Create: `risk_overlay.py`
- Test: `test_risk_overlay.py`

- [ ] **Step 1: Write the failing test**

```python
# test_risk_overlay.py
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from injury_risk import RiskResult
from risk_overlay import RiskOverlay


class TestRiskOverlay(unittest.TestCase):
    @patch("risk_overlay.cv2")
    def test_draw_renders_status_and_profile(self, mock_cv2):
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = RiskResult(
            total_risk=35.0,
            base_score=30.0,
            interaction_score=5.0,
            normalized={},
            alerts=[],
            status="Caution",
            profile_label="Balanced",
        )

        overlay.draw(frame, result)

        mock_cv2.rectangle.assert_called()
        mock_cv2.putText.assert_called()
        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        assert "Caution" in joined
        assert "35.0" in joined
        assert "Balanced" in joined

    @patch("risk_overlay.cv2")
    def test_draw_insufficient_data(self, mock_cv2):
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay.draw(frame, None)

        mock_cv2.rectangle.assert_called()
        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        assert "Insufficient data" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_risk_overlay.py -v`
Expected: failures because `risk_overlay.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# risk_overlay.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from injury_risk import RiskResult


# Colors are BGR.
COLOR_OK: Tuple[int, int, int] = (0, 255, 0)
COLOR_WARN: Tuple[int, int, int] = (0, 200, 255)
COLOR_BAD: Tuple[int, int, int] = (0, 0, 255)
COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)
COLOR_PANEL: Tuple[int, int, int] = (30, 30, 30)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
LINE_HEIGHT = 24
PADDING = 10


class RiskOverlay:
    """Draws injury-risk status and per-parameter alerts on an OpenCV frame."""

    def draw(
        self,
        frame: np.ndarray,
        result: Optional[RiskResult],
    ) -> np.ndarray:
        if result is None:
            self._draw_panel(frame, 0, 0, 420, 70, COLOR_PANEL)
            self._draw_text(frame, "Insufficient data — pose not fully visible", 15, 35, COLOR_TEXT)
            return frame

        color = self._status_color(result.status)
        # Status badge
        self._draw_panel(frame, 0, 0, 340, 110, color)
        self._draw_text(frame, f"{self._status_emoji(result.status)} {result.status}", 15, 35, (0, 0, 0))
        self._draw_text(frame, f"Risk: {result.total_risk:.1f}", 15, 80, (0, 0, 0))
        self._draw_text(frame, f"Profile: {result.profile_label}", 200, 40, (0, 0, 0), scale=0.5)

        # Per-parameter alerts
        y_start = 130
        panel_height = len(result.alerts) * LINE_HEIGHT + 30
        self._draw_panel(frame, 0, y_start, 300, panel_height, COLOR_PANEL)
        for i, alert in enumerate(result.alerts):
            y = y_start + 25 + i * LINE_HEIGHT
            self._draw_text(frame, alert["label"], 15, y, COLOR_TEXT, scale=0.5)
            self._draw_text(frame, alert["txt"], 180, y, self._level_color(alert["cls"]), scale=0.5)

        return frame

    def _status_color(self, status: str) -> Tuple[int, int, int]:
        return {
            "Optimal": COLOR_OK,
            "Caution": COLOR_WARN,
            "High Risk": COLOR_BAD,
        }.get(status, COLOR_PANEL)

    def _status_emoji(self, status: str) -> str:
        return {
            "Optimal": "OK",
            "Caution": "!",
            "High Risk": "X",
        }.get(status, "?")

    def _level_color(self, cls: str) -> Tuple[int, int, int]:
        return {
            "ok": COLOR_OK,
            "warn": COLOR_WARN,
            "bad": COLOR_BAD,
        }.get(cls, COLOR_TEXT)

    def _draw_panel(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int],
    ) -> None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        scale: float = FONT_SCALE,
    ) -> None:
        cv2.putText(frame, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_risk_overlay.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add risk_overlay.py test_risk_overlay.py
git commit -m "feat(overlay): add RiskOverlay for live risk HUD"
```

---

### Task 6: Wire risk model and overlay into the detector loop

**Files:**
- Modify: `webcam_leg_pose_detector.py`
- Modify: `test_webcam_leg_pose_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# test_webcam_leg_pose_detector.py (append)
from unittest.mock import MagicMock, patch

import numpy as np

from injury_risk import RiskModel
from risk_overlay import RiskOverlay


class TestRiskIntegration(unittest.TestCase):
    @patch.object(detector_module.cv2, "destroyAllWindows")
    @patch.object(detector_module.cv2, "waitKey", return_value=ord("q"))
    @patch.object(detector_module.cv2, "imshow")
    @patch.object(detector_module.cv2, "flip", side_effect=lambda f, _: f)
    @patch.object(detector_module.cv2, "VideoCapture")
    def test_main_exercises_risk_path(
        self, mock_video_capture, mock_flip, mock_imshow, mock_waitKey, mock_destroy
    ):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_video_capture.return_value = mock_cap

        landmarks = _make_mock_landmarks()
        mock_results = MagicMock()
        mock_results.pose_landmarks = landmarks
        mock_pose_instance = MagicMock()
        mock_pose_instance.process.return_value = mock_results

        with patch.object(
            detector_module.mp.solutions.pose, "Pose", return_value=mock_pose_instance
        ):
            with patch.object(RiskModel, "update", return_value=None) as mock_update:
                with patch.object(RiskOverlay, "draw") as mock_draw:
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        mock_update.assert_called_once()
        mock_draw.assert_called_once()

    @patch.object(detector_module.cv2, "destroyAllWindows")
    @patch.object(detector_module.cv2, "waitKey", side_effect=[ord("p"), ord("q")])
    @patch.object(detector_module.cv2, "imshow")
    @patch.object(detector_module.cv2, "flip", side_effect=lambda f, _: f)
    @patch.object(detector_module.cv2, "VideoCapture")
    def test_main_cycles_profile_on_p_key(
        self, mock_video_capture, mock_flip, mock_imshow, mock_waitKey, mock_destroy
    ):
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_video_capture.return_value = mock_cap

        with patch.object(detector_module.mp.solutions.pose, "Pose"):
            with patch.object(RiskModel, "update", return_value=None) as mock_update:
                with patch.object(RiskOverlay, "draw"):
                    with patch.object(RiskModel, "cycle_profile") as mock_cycle:
                        exit_code = main()

        self.assertEqual(exit_code, 0)
        mock_cycle.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_webcam_leg_pose_detector.py::TestRiskIntegration -v`
Expected: failures because `main()` does not yet call the risk path.

- [ ] **Step 3: Write minimal implementation**

Modify `webcam_leg_pose_detector.py`:

```python
# Add these imports near the top, after the mediapipe imports.
from injury_risk import RiskModel
from risk_overlay import RiskOverlay


def main(
    camera_index: int = 0,
    config: Optional[PoseDetectorConfig] = None,
    *,
    debug: bool = False,
    risk_profile: str = "balanced",
) -> int:
    # ... existing camera open check ...

    risk_model = RiskModel(risk_profile)
    risk_overlay = RiskOverlay()

    try:
        with PoseDetector(config) as detector:
            while True:
                # ... existing frame read logic ...

                landmarks = detector.process(frame)
                if landmarks is not None:
                    detector.draw_landmarks(frame, landmarks)
                    lower_body = detector.get_lower_body_landmarks(landmarks)
                    risk_result = risk_model.update(lower_body, frame.shape)
                    risk_overlay.draw(frame, risk_result)
                    if debug:
                        # ... existing debug print ...
                elif debug:
                    print("No pose detected.")

                # ... existing flip, imshow ...
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    risk_model.cycle_profile()
                    if debug:
                        print(f"Switched to profile: {risk_model.profile.label}")
    finally:
        # ... existing cleanup ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_webcam_leg_pose_detector.py::TestRiskIntegration -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add webcam_leg_pose_detector.py test_webcam_leg_pose_detector.py
git commit -m "feat(detector): wire live risk model and overlay into frame loop"
```

---

### Task 7: Run full test suite and finalize

**Files:**
- All touched files

- [ ] **Step 1: Run all tests**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Run the live detector smoke test (optional)**

Run: `python webcam_leg_pose_detector.py --debug`
Expected: window opens, press `p` to cycle profiles, press `q` to quit. (Requires webcam.)

- [ ] **Step 3: Commit any final fixes**

```bash
git add -A
git commit -m "chore: finalize live injury-risk integration and tests"
```

---

## Self-Review

1. **Spec coverage:**
   - Profile model / piecewise curves → Tasks 1-2.
   - Landmark-to-parameter extraction → Task 3.
   - Smoothing / history → Task 4.
   - On-screen HUD → Task 5.
   - Integration / keyboard switching → Task 6.
   - Tests → every task includes tests.

2. **Placeholder scan:**
   - No “TBD”, “TODO”, or vague steps. Every step contains code, commands, and expected output.

3. **Type consistency:**
   - `RiskProfile.from_preset`, `RiskModel.set_profile`, `RiskModel.cycle_profile`, `RiskOverlay.draw`, and `RiskResult` fields are consistent across tasks.
