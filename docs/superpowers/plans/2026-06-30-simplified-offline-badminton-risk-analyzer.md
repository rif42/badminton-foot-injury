# Simplified Offline Badminton Injury-Risk Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline command-line tool that analyzes a recorded badminton clip, gates on movement, and scores lower-body injury risk using only the four core parameters from the updated baseline.

**Architecture:** Separate the problem into small modules: geometry helpers, four parameter calculators, a motion gate, a core risk composer, and a thin CLI that wires MediaPipe extraction to CSV/video output.

**Tech Stack:** Python 3.11+, pytest, MediaPipe, OpenCV, NumPy.

---

## File Structure

| File | Responsibility |
|---|---|
| `baseline_risk.py` | Geometry helpers and four core risk-parameter calculators |
| `motion_gate.py` | Moving/standing classification from hip-center displacement |
| `video_risk_analyzer.py` | CLI: read video, extract landmarks, gate, score, write CSV/video |
| `test_baseline_risk.py` | Unit tests for all parameter and score functions |
| `test_motion_gate.py` | Unit tests for the motion gate |
| `test_video_risk_analyzer.py` | Integration tests for the CLI |

---

### Task 1: Geometry Helpers

**Files:**
- Create: `baseline_risk.py`
- Test: `test_baseline_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# test_baseline_risk.py
import math

import numpy as np

from baseline_risk import angle_at, clamp, distance, midpoint


def test_clamp_inside_range():
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_below_range():
    assert clamp(-0.1, 0.0, 1.0) == 0.0


def test_clamp_above_range():
    assert clamp(1.2, 0.0, 1.0) == 1.0


def test_distance_3d():
    a = (0.0, 0.0, 0.0)
    b = (3.0, 4.0, 0.0)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_baseline_risk.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'baseline_risk'`

- [ ] **Step 3: Write minimal implementation**

```python
# baseline_risk.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_baseline_risk.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add baseline_risk.py test_baseline_risk.py
git commit -m "feat(risk): add geometry helpers for baseline risk calculator"
```

---

### Task 2: Knee Flexion and Stiffness Risk

**Files:**
- Modify: `baseline_risk.py`
- Test: `test_baseline_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_baseline_risk.py
from baseline_risk import knee_flexion_angle, knee_stiffness_risk


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_baseline_risk.py::test_knee_flexion_right_angle -v`

Expected: FAIL with `ImportError: cannot import name 'knee_flexion_angle'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to baseline_risk.py

def knee_flexion_angle(hip: Point, knee: Point, ankle: Point) -> float:
    return angle_at(hip, knee, ankle)


def knee_stiffness_risk(knee_angle: float, min_safe: float = 145.0, max_safe: float = 180.0) -> float:
    return clamp((knee_angle - min_safe) / (max_safe - min_safe), 0.0, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_baseline_risk.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add baseline_risk.py test_baseline_risk.py
git commit -m "feat(risk): add knee flexion and stiffness risk"
```

---

### Task 3: Ankle-Foot Alignment Risk

**Files:**
- Modify: `baseline_risk.py`
- Test: `test_baseline_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_baseline_risk.py
from baseline_risk import ankle_foot_alignment_risk


def test_ankle_foot_alignment_perfect():
    knee = (0.0, 0.8, 0.5)
    ankle = (0.0, 0.1, 0.5)
    heel = (0.0, 0.0, 0.3)
    foot_index = (0.0, 0.0, 0.7)
    leg_length = 1.4
    assert ankle_foot_alignment_risk(knee, ankle, heel, foot_index, leg_length) == pytest.approx(0.0, abs=1e-3)


def test_ankle_foot_alignment_toe_in():
    knee = (0.0, 0.8, 0.5)
    ankle = (0.0, 0.1, 0.5)
    heel = (0.0, 0.0, 0.3)
    foot_index = (0.2, 0.0, 0.3)  # toe-out 45 degrees relative to +Z
    leg_length = 1.4
    risk = ankle_foot_alignment_risk(knee, ankle, heel, foot_index, leg_length)
    assert risk > 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_baseline_risk.py::test_ankle_foot_alignment_perfect -v`

Expected: FAIL with `ImportError: cannot import name 'ankle_foot_alignment_risk'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to baseline_risk.py
import math


def ankle_foot_alignment_risk(
    knee: Point,
    ankle: Point,
    heel: Point,
    foot_index: Point,
    leg_length: float,
) -> float:
    foot_center = midpoint(heel, foot_index)
    knee_over_foot_deviation = abs(knee[0] - foot_center[0]) / max(leg_length, 1e-6)
    knee_dev_score = clamp(knee_over_foot_deviation / 0.22, 0.0, 1.0)

    dx = foot_index[0] - heel[0]
    dz = foot_index[2] - heel[2]
    foot_angle = abs(math.degrees(math.atan2(dx, dz)))
    angle_score = clamp(foot_angle / 45.0, 0.0, 1.0)

    return 0.70 * knee_dev_score + 0.30 * angle_score
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_baseline_risk.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add baseline_risk.py test_baseline_risk.py
git commit -m "feat(risk): add ankle-foot alignment risk"
```

---

### Task 4: Hip Displacement Proxy

**Files:**
- Modify: `baseline_risk.py`
- Test: `test_baseline_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_baseline_risk.py
from baseline_risk import hip_displacement_proxy


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
    assert proxy > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_baseline_risk.py::test_hip_displacement_low -v`

Expected: FAIL with `ImportError: cannot import name 'hip_displacement_proxy'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to baseline_risk.py

def hip_displacement_proxy(
    left_hip: Point,
    right_hip: Point,
    heel: Point,
    foot_index: Point,
    leg_length: float,
) -> float:
    hip_center = midpoint(left_hip, right_hip)
    foot_center = midpoint(heel, foot_index)
    horizontal = math.sqrt(
        (hip_center[0] - foot_center[0]) ** 2 + (hip_center[2] - foot_center[2]) ** 2
    )
    return clamp(horizontal / max(leg_length, 1e-6), 0.0, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_baseline_risk.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add baseline_risk.py test_baseline_risk.py
git commit -m "feat(risk): add hip displacement proxy"
```

---

### Task 5: Landing Asymmetry Score

**Files:**
- Modify: `baseline_risk.py`
- Test: `test_baseline_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_baseline_risk.py
from baseline_risk import landing_asymmetry_score


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
    assert score > 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_baseline_risk.py::test_landing_asymmetry_symmetric -v`

Expected: FAIL with `ImportError: cannot import name 'landing_asymmetry_score'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to baseline_risk.py

def landing_asymmetry_score(
    left_hip: Point,
    right_hip: Point,
    left_knee: Point,
    right_knee: Point,
    left_ankle: Point,
    right_ankle: Point,
) -> float:
    left_knee_angle = knee_flexion_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = knee_flexion_angle(right_hip, right_knee, right_ankle)
    knee_asym = clamp(abs(left_knee_angle - right_knee_angle) / 60.0, 0.0, 1.0)

    pelvis_width = distance(left_hip, right_hip)
    hip_height_score = clamp(
        abs(left_hip[1] - right_hip[1]) / max(pelvis_width * 0.8, 1e-6), 0.0, 1.0
    )

    avg_leg_length = (
        distance(left_hip, left_knee) + distance(left_knee, left_ankle) +
        distance(right_hip, right_knee) + distance(right_knee, right_ankle)
    ) / 2.0
    ankle_height_score = clamp(
        abs(left_ankle[1] - right_ankle[1]) / max(avg_leg_length * 0.2, 1e-6), 0.0, 1.0
    )

    return 0.35 * knee_asym + 0.25 * hip_height_score + 0.20 * ankle_height_score
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_baseline_risk.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add baseline_risk.py test_baseline_risk.py
git commit -m "feat(risk): add landing asymmetry score"
```

---

### Task 6: Core Risk Score Composer

**Files:**
- Modify: `baseline_risk.py`
- Test: `test_baseline_risk.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_baseline_risk.py
from baseline_risk import core_risk_score, LowerBodyPose


def test_core_risk_perfect_pose():
    pose = LowerBodyPose(
        left_hip=(-0.1, 1.0, 0.0),
        right_hip=(0.1, 1.0, 0.0),
        left_knee=(-0.1, 0.5, 0.0),
        right_knee=(0.1, 0.5, 0.0),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_baseline_risk.py::test_core_risk_perfect_pose -v`

Expected: FAIL with `ImportError: cannot import name 'core_risk_score'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to baseline_risk.py
from dataclasses import dataclass


@dataclass(frozen=True)
class LowerBodyPose:
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
    return distance(hip, knee) + distance(knee, ankle)


def core_risk_score(pose: LowerBodyPose) -> dict:
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

    core = 0.20 * knee_stiffness + 0.30 * ankle_alignment + 0.15 * hip_disp + 0.35 * asym

    return {
        "knee_stiffness_risk": knee_stiffness,
        "ankle_foot_alignment_risk": ankle_alignment,
        "hip_displacement_proxy": hip_disp,
        "landing_asymmetry_score": asym,
        "core_risk": clamp(core, 0.0, 1.0),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_baseline_risk.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add baseline_risk.py test_baseline_risk.py
git commit -m "feat(risk): add core risk score composer"
```

---

### Task 7: Motion Gate

**Files:**
- Create: `motion_gate.py`
- Test: `test_motion_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# test_motion_gate.py
from motion_gate import MotionGate, classify_frame


def test_classify_frame_standing():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    hip_centers = [(0.0, 1.0, 0.0)] * 20
    assert classify_frame(gate, hip_centers) == "standing"


def test_classify_frame_moving():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    hip_centers = [(0.0, 1.0, 0.0)] * 5 + [(0.3, 1.0, 0.0)] * 15
    assert classify_frame(gate, hip_centers) == "moving"


def test_classify_frame_not_enough_history():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    hip_centers = [(0.0, 1.0, 0.0), (0.3, 1.0, 0.0)]
    assert classify_frame(gate, hip_centers) == "standing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_motion_gate.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'motion_gate'`

- [ ] **Step 3: Write minimal implementation**

```python
# motion_gate.py
import math
from dataclasses import dataclass, field
from typing import List, Tuple

Point = Tuple[float, float, float]


@dataclass
class MotionGate:
    window_seconds: float = 0.3
    fps: float = 30.0
    threshold_ratio: float = 0.05
    history: List[Point] = field(default_factory=list)

    @property
    def window_frames(self) -> int:
        return max(1, int(round(self.window_seconds * self.fps)))

    def update(self, hip_center: Point, leg_length: float) -> str:
        self.history.append(hip_center)
        if len(self.history) > self.window_frames + 1:
            self.history.pop(0)
        return self.classify(leg_length)

    def classify(self, leg_length: float) -> str:
        if len(self.history) < 2:
            return "standing"
        window = self.history[-self.window_frames:]
        start = window[0]
        end = window[-1]
        displacement = math.sqrt((end[0] - start[0]) ** 2 + (end[2] - start[2]) ** 2)
        threshold = max(leg_length, 1.0) * self.threshold_ratio
        return "moving" if displacement > threshold else "standing"


def classify_frame(gate: MotionGate, hip_centers: List[Point], leg_length: float = 1.0) -> str:
    result = "standing"
    for hc in hip_centers:
        result = gate.update(hc, leg_length)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_motion_gate.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add motion_gate.py test_motion_gate.py
git commit -m "feat(gate): add simple moving/standing motion gate"
```

---

### Task 8: Video Risk Analyzer CLI

**Files:**
- Create: `video_risk_analyzer.py`
- Test: `test_video_risk_analyzer.py`

- [ ] **Step 1: Write the failing test**

```python
# test_video_risk_analyzer.py
import csv
import os
from unittest.mock import MagicMock, patch

import numpy as np

from video_risk_analyzer import analyze_video, build_csv_rows


def test_build_csv_rows():
    results = [
        {
            "frame": 0,
            "time_sec": 0.0,
            "is_moving": False,
            "knee_stiffness_risk": 0.0,
            "ankle_foot_alignment_risk": 0.0,
            "hip_displacement_proxy": 0.0,
            "landing_asymmetry_score": 0.0,
            "core_risk": 0.0,
            "status": "acceptable",
        }
    ]
    rows = build_csv_rows(results)
    assert rows[0]["frame"] == "0"
    assert rows[0]["status"] == "acceptable"


def test_analyze_video_outputs_csv(tmp_path):
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output_csv = tmp_path / "out.csv"

    with patch("cv2.VideoCapture") as mock_cap, \
         patch("mediapipe.solutions.pose.Pose") as mock_pose_class:
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, fake_frame), (True, fake_frame), (False, None)]
        mock_cap_instance.get.side_effect = lambda key: 30.0 if key == 5 else 3.0
        mock_cap.return_value = mock_cap_instance

        mock_pose = MagicMock()
        mock_pose_class.return_value = mock_pose
        mock_result = MagicMock()
        mock_result.pose_landmarks = None
        mock_pose.process.return_value = mock_result

        analyze_video(str(tmp_path / "fake.mp4"), str(output_csv), None, show_preview=False)

    assert output_csv.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_video_risk_analyzer.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'video_risk_analyzer'`

- [ ] **Step 3: Write minimal implementation**

```python
# video_risk_analyzer.py
import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import mediapipe as mp

from baseline_risk import LowerBodyPose, Point, core_risk_score
from motion_gate import MotionGate


LOWER_BODY_INDICES = {
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def _landmark_to_point(landmarks, index: int, image_shape: tuple) -> Point:
    lm = landmarks.landmark[index]
    return (
        lm.x * image_shape[1],
        lm.y * image_shape[0],
        lm.z * image_shape[1],
    )


def _extract_pose(landmarks, image_shape: tuple) -> Optional[LowerBodyPose]:
    if landmarks is None:
        return None
    try:
        kwargs = {
            name: _landmark_to_point(landmarks, idx, image_shape)
            for name, idx in LOWER_BODY_INDICES.items()
        }
        return LowerBodyPose(**kwargs)
    except (IndexError, AttributeError):
        return None


def _status(core_risk: float) -> str:
    if core_risk < 0.35:
        return "acceptable"
    if core_risk < 0.65:
        return "caution"
    return "risky"


def build_csv_rows(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    fieldnames = [
        "frame", "time_sec", "is_moving",
        "knee_stiffness_risk", "ankle_foot_alignment_risk",
        "hip_displacement_proxy", "landing_asymmetry_score",
        "core_risk", "status",
    ]
    return [
        {key: str(row.get(key, "")) for key in fieldnames}
        for row in results
    ]


def analyze_video(
    input_path: str,
    output_csv: str,
    output_video: Optional[str],
    show_preview: bool = False,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    gate = MotionGate(window_seconds=0.3, fps=fps)
    results: List[Dict[str, Any]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = pose.process(rgb)
        pose_obj = _extract_pose(mp_result.pose_landmarks, frame.shape)

        row: Dict[str, Any] = {
            "frame": frame_idx,
            "time_sec": round(frame_idx / fps, 3),
            "is_moving": False,
            "knee_stiffness_risk": 0.0,
            "ankle_foot_alignment_risk": 0.0,
            "hip_displacement_proxy": 0.0,
            "landing_asymmetry_score": 0.0,
            "core_risk": 0.0,
            "status": "acceptable",
        }

        if pose_obj:
            hip_center = (
                (pose_obj.left_hip[0] + pose_obj.right_hip[0]) / 2.0,
                (pose_obj.left_hip[1] + pose_obj.right_hip[1]) / 2.0,
                (pose_obj.left_hip[2] + pose_obj.right_hip[2]) / 2.0,
            )
            leg_length = (
                ((pose_obj.left_hip[1] - pose_obj.left_ankle[1]) +
                 (pose_obj.right_hip[1] - pose_obj.right_ankle[1])) / 2.0
            )
            state = gate.update(hip_center, leg_length)
            row["is_moving"] = state == "moving"

            if state == "moving":
                score = core_risk_score(pose_obj)
                row.update(score)
                row["status"] = _status(score["core_risk"])

        results.append(row)

        if writer is not None or show_preview:
            display = frame.copy()
            color = (0, 255, 0) if row["status"] == "acceptable" else (0, 255, 255) if row["status"] == "caution" else (0, 0, 255)
            label = f"{row['status'].upper()} {row['core_risk']:.2f}"
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if writer is not None:
                writer.write(display)
            if show_preview:
                cv2.imshow("Risk Analyzer", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    pose.close()
    cv2.destroyAllWindows()

    rows = build_csv_rows(results)
    with open(output_csv, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline badminton lower-body risk analyzer.")
    parser.add_argument("input_video", help="Path to input video.")
    parser.add_argument("--output-csv", default="risk_report.csv", help="Path to output CSV.")
    parser.add_argument("--output-video", default=None, help="Path to optional annotated output video.")
    parser.add_argument("--show-preview", action="store_true", help="Show live preview window.")
    args = parser.parse_args(argv)

    analyze_video(args.input_video, args.output_csv, args.output_video, args.show_preview)
    print(f"Wrote CSV report to {args.output_csv}")
    if args.output_video:
        print(f"Wrote annotated video to {args.output_video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_video_risk_analyzer.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add video_risk_analyzer.py test_video_risk_analyzer.py
git commit -m "feat(cli): add offline video risk analyzer"
```

---

### Task 9: Integration Test With Synthetic Video

**Files:**
- Modify: `test_video_risk_analyzer.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_video_risk_analyzer.py
import tempfile

import cv2
import numpy as np

from video_risk_analyzer import main


def test_main_on_synthetic_video(tmp_path):
    video_path = tmp_path / "test.mp4"
    output_csv = tmp_path / "out.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    for _ in range(30):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
    writer.release()

    with patch("mediapipe.solutions.pose.Pose") as mock_pose_class:
        mock_pose = MagicMock()
        mock_pose_class.return_value = mock_pose
        mock_result = MagicMock()
        mock_result.pose_landmarks = None
        mock_pose.process.return_value = mock_result

        main([str(video_path), "--output-csv", str(output_csv)])

    assert output_csv.exists()
    with open(output_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_video_risk_analyzer.py::test_main_on_synthetic_video -v`

Expected: FAIL with `NameError: name 'tempfile' is not defined` or similar import error

- [ ] **Step 3: Write minimal implementation**

Add imports at the top of `test_video_risk_analyzer.py`:

```python
import csv
import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import pytest
from video_risk_analyzer import analyze_video, build_csv_rows, main
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_video_risk_analyzer.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test_video_risk_analyzer.py
git commit -m "test(cli): add integration test with synthetic video"
```

---

### Task 10: README / Usage Documentation

**Files:**
- Modify: `README.md` (or create `docs/superpowers/usage/offline-analyzer.md` if no README exists)

- [ ] **Step 1: Check if README.md exists**

Run:

```bash
ls README.md
```

Expected: either file path or "No such file or directory"

- [ ] **Step 2: Create or update usage doc**

If `README.md` exists, append a new section. Otherwise create `docs/superpowers/usage/offline-analyzer.md`.

Example content:

```markdown
## Offline Video Risk Analyzer

Analyze a recorded badminton clip for lower-body injury-risk patterns.

### Run

```bash
python video_risk_analyzer.py input.mp4 --output-csv report.csv
```

### Optional annotated video

```bash
python video_risk_analyzer.py input.mp4 --output-csv report.csv --output-video annotated.mp4
```

### Output CSV columns

| Column | Description |
|---|---|
| frame | Frame index |
| time_sec | Timestamp in seconds |
| is_moving | True if motion gate detected movement |
| knee_stiffness_risk | 0–1 stiffness risk |
| ankle_foot_alignment_risk | 0–1 alignment risk |
| hip_displacement_proxy | 0–1 hip displacement |
| landing_asymmetry_score | 0–1 asymmetry |
| core_risk | Combined 0–1 risk |
| status | acceptable / caution / risky |
```

- [ ] **Step 3: Commit**

```bash
git add README.md  # or docs/superpowers/usage/offline-analyzer.md
git commit -m "docs: add offline analyzer usage instructions"
```

---

## Self-Review

### Spec coverage

- ✅ Geometry helpers → Task 1
- ✅ Knee flexion/stiffness → Task 2
- ✅ Ankle-foot alignment → Task 3
- ✅ Hip displacement → Task 4
- ✅ Landing asymmetry → Task 5
- ✅ Core risk score → Task 6
- ✅ Motion gate → Task 7
- ✅ Video CLI → Task 8
- ✅ Integration test → Task 9
- ✅ Documentation → Task 10

### Placeholder scan

No TBD, TODO, or vague instructions. Each task includes actual code and exact commands.

### Type consistency

- `Point` is consistently `Tuple[float, float, float]`.
- `LowerBodyPose` fields match across `baseline_risk.py` and `video_risk_analyzer.py`.
- CSV column names match between `build_csv_rows` and tests.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-30-simplified-offline-badminton-risk-analyzer.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
