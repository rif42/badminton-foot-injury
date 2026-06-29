# Body-Scale-Normalized Idle vs Dynamic Detection — First Slice

**Goal:** Add a body-scale-aware idle gate so the risk model emits a first-class `Idle` result (with smoothing reset and overlay rendering) while leaving existing pixel-based risk curves untouched.

**Architecture:** Introduce a small body-scale estimator, normalize hip displacement by that scale, and use a named normalized threshold to decide when the athlete is effectively still. When the state flips between idle and dynamic, the smoothing window is cleared so stale values do not leak across states. The overlay learns to render `Idle` as a neutral status. Existing landing-pitch curves and pixel thresholds are intentionally not changed.

**Design source:** User scope (no design document on disk).

---

## Dependency Graph

```
Batch 1 (parallel): 1.1, 1.2 [foundation - no deps]
Batch 2 (parallel): 2.1, 2.2 [core integration - depends on batch 1]
```

---

## Batch 1: Foundation (parallel — 2 implementers)

All tasks in this batch have no dependencies and can run simultaneously.

---

### Task 1.1: Body-scale helper and normalized idle gate

**File:** `injury_risk.py`
**Test:** `test_injury_risk.py`
**Depends:** none

#### Implementation

Add the following constants immediately after `VISIBILITY_THRESHOLD = 0.5`:

```python
# Body-scale estimation uses the average hip-to-ankle distance (pixels).
BODY_SCALE_JOINTS: Tuple[Tuple[str, str], ...] = (
    ("left_hip", "left_ankle"),
    ("right_hip", "right_ankle"),
)

# Idle gate: normalized hip movement below this fraction of body scale is
# treated as no meaningful athletic movement.
IDLE_MOVEMENT_THRESHOLD = 0.015
```

Add the following functions near the other `compute_*` helpers (after `compute_hip_trajectory_deviation` is a good place):

```python
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
```

#### Test additions

Add these imports to the top of `test_injury_risk.py`:

```python
compute_body_scale,
compute_normalized_hip_movement,
is_idle,
```

Add the following test class before `TestRiskModel`:

```python
class TestBodyScaleAndIdleGate:
    def test_compute_body_scale_average_leg_length(self):
        frame_shape = (480, 640)
        landmarks = {
            "left_hip": _landmark(0.5, 0.3),
            "left_ankle": _landmark(0.5, 0.7),
            "right_hip": _landmark(0.5, 0.3),
            "right_ankle": _landmark(0.5, 0.7),
        }
        # Each leg is 0.4 * 480 = 192 px; average is the same.
        scale = compute_body_scale(landmarks, frame_shape)
        assert scale == pytest.approx(192.0)

    def test_compute_body_scale_returns_none_when_missing(self):
        frame_shape = (480, 640)
        assert compute_body_scale({}, frame_shape) is None

    def test_compute_body_scale_uses_visible_side(self):
        frame_shape = (480, 640)
        landmarks = {
            "left_hip": _landmark(0.5, 0.3),
            "left_ankle": _landmark(0.5, 0.7),
        }
        assert compute_body_scale(landmarks, frame_shape) == pytest.approx(192.0)

    def test_compute_normalized_hip_movement(self):
        history = [(100.0, 100.0), (103.0, 104.0)]  # 5 px displacement
        body_scale = 100.0
        movement = compute_normalized_hip_movement(history, body_scale)
        assert movement == pytest.approx(0.05)

    def test_compute_normalized_hip_movement_returns_none_for_short_history(self):
        assert compute_normalized_hip_movement([(100.0, 100.0)], 100.0) is None

    def test_is_idle_true_when_movement_below_threshold(self):
        frame_shape = (480, 640)
        # Body scale ~192 px; movement ~1.4 px -> ~0.007 < 0.015.
        landmarks = {
            "left_hip": _landmark(0.5, 0.3),
            "left_ankle": _landmark(0.5, 0.7),
            "right_hip": _landmark(0.5, 0.3),
            "right_ankle": _landmark(0.5, 0.7),
        }
        history = [(320.0, 144.0), (321.0, 145.0)]
        assert is_idle(landmarks, frame_shape, history) is True

    def test_is_idle_false_when_movement_above_threshold(self):
        frame_shape = (480, 640)
        # Body scale ~192 px; movement 20 px -> ~0.104 > 0.015.
        landmarks = {
            "left_hip": _landmark(0.5, 0.3),
            "left_ankle": _landmark(0.5, 0.7),
            "right_hip": _landmark(0.5, 0.3),
            "right_ankle": _landmark(0.5, 0.7),
        }
        history = [(320.0, 144.0), (340.0, 144.0)]
        assert is_idle(landmarks, frame_shape, history) is False
```

**Verify:**

```bash
python -m pytest test_injury_risk.py::TestBodyScaleAndIdleGate -v
```

**Commit:** `feat(idle): add body-scale helper and normalized idle gate`

---

### Task 1.2: Overlay idle status rendering

**File:** `risk_overlay.py`
**Test:** `test_risk_overlay.py`
**Depends:** none

#### Implementation

Add the idle color constant near the other color constants:

```python
COLOR_IDLE: Tuple[int, int, int] = (128, 128, 128)
```

Update `_status_color` to include `Idle`:

```python
    def _status_color(self, status: str) -> Tuple[int, int, int]:
        return {
            "Optimal": COLOR_OK,
            "Caution": COLOR_WARN,
            "High Risk": COLOR_BAD,
            "Idle": COLOR_IDLE,
        }.get(status, COLOR_PANEL)
```

Update `_status_prefix` to include `Idle`:

```python
    def _status_prefix(self, status: str) -> str:
        return {
            "Optimal": "OK",
            "Caution": "!",
            "High Risk": "X",
            "Idle": "·",
        }.get(status, "?")
```

No changes are required in `draw()` itself because it already uses these helpers to render any status.

#### Test additions

Add this test method to `TestRiskOverlay`:

```python
    @patch("risk_overlay.cv2")
    def test_draw_renders_idle_status(self, mock_cv2):
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = RiskResult(
            total_risk=0.0,
            base_score=0.0,
            interaction_score=0.0,
            normalized={},
            alerts=[],
            status="Idle",
            profile_label="Balanced",
        )

        overlay.draw(frame, result)

        mock_cv2.rectangle.assert_called()
        mock_cv2.putText.assert_called()
        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        self.assertIn("Idle", joined)
        self.assertIn("0.0", joined)
        self.assertIn("Balanced", joined)
```

**Verify:**

```bash
python -m pytest test_risk_overlay.py -v
```

**Commit:** `feat(idle): render idle status in risk overlay`

---

## Batch 2: Core Integration (parallel — 2 implementers)

All tasks in this batch depend on Batch 1 completing first.

---

### Task 2.1: RiskModel idle detection and smoothing reset

**File:** `injury_risk.py`
**Test:** `test_injury_risk.py`
**Depends:** 1.1

#### Implementation

Add `self._was_idle = False` to `RiskModel.__init__`:

```python
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
```

Add two helper methods to `RiskModel`:

```python
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
```

Replace `RiskModel.update` with the version below. The key change is the idle branch inserted after the hip-history has at least two samples:

```python
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
```

#### Test additions and updates

Update the `_make_safe_landmarks` helper to accept an optional hip shift so dynamic tests can produce enough motion to escape the idle gate:

```python
def _make_safe_landmarks(hip_shift=(0.0, 0.0)):
    def lm(x, y, v=1.0):
        return (x, y, 0.0, v)

    return {
        "left_hip": lm(0.45 + hip_shift[0], 0.35 + hip_shift[1]),
        "right_hip": lm(0.55 + hip_shift[0], 0.35 + hip_shift[1]),
        "left_knee": lm(0.45, 0.55),
        "right_knee": lm(0.55, 0.55),
        "left_ankle": lm(0.45, 0.8),
        "right_ankle": lm(0.55, 0.8),
        "left_heel": lm(0.43, 0.82),
        "right_heel": lm(0.53, 0.82),
        "left_foot_index": lm(0.47, 0.81),
        "right_foot_index": lm(0.57, 0.81),
    }
```

Update `test_update_returns_risk_result_after_history_fills` so the frames are dynamic:

```python
    def test_update_returns_risk_result_after_history_fills(self):
        model = RiskModel()
        frame_shape = (480, 640)
        result = None
        for i in range(5):
            result = model.update(_make_safe_landmarks(hip_shift=(i * 0.02, 0)), frame_shape)
        assert isinstance(result, RiskResult)
        assert 0.0 <= result.total_risk <= 100.0
        assert result.status in {"Optimal", "Caution", "High Risk"}
        assert result.profile_label == "Balanced"
```

Update `test_smoothing_averages_over_window` so the frames are dynamic:

```python
    def test_smoothing_averages_over_window(self):
        model = RiskModel(smoothing_window=3)
        frame_shape = (480, 640)
        first = None
        for i in range(5):
            lm = _make_safe_landmarks(hip_shift=(i * 0.02, 0))
            # Move ankles down each frame to create changing geometry.
            lm["left_ankle"] = (lm["left_ankle"][0], 0.75 + i * 0.01, 0.0, 1.0)
            lm["right_ankle"] = (lm["right_ankle"][0], 0.75 + i * 0.01, 0.0, 1.0)
            first = model.update(lm, frame_shape)
        assert first is not None
```

Add these new tests to `TestRiskModel`:

```python
    def test_update_returns_idle_when_hip_is_stationary(self):
        model = RiskModel()
        frame_shape = (480, 640)
        for _ in range(5):
            result = model.update(_make_safe_landmarks(), frame_shape)
        assert result is not None
        assert result.status == "Idle"
        assert result.total_risk == pytest.approx(0.0)
        assert result.base_score == pytest.approx(0.0)
        assert result.interaction_score == pytest.approx(0.0)
        assert all(v == 0.0 for v in result.normalized.values())
        assert all(alert["txt"] == "Low" for alert in result.alerts)

    def test_idle_to_dynamic_resets_smoothing(self):
        model = RiskModel(smoothing_window=5)
        frame_shape = (480, 640)

        # Run idle for several frames.
        for _ in range(5):
            idle_result = model.update(_make_safe_landmarks(), frame_shape)
        assert idle_result is not None
        assert idle_result.status == "Idle"
        assert idle_result.total_risk == pytest.approx(0.0)

        # Transition to dynamic. Hip shift of 0.02/frame -> ~12.8 px/frame.
        result = None
        for i in range(1, 6):
            result = model.update(
                _make_safe_landmarks(hip_shift=(i * 0.02, 0)), frame_shape
            )
        assert result is not None
        assert result.status != "Idle"
        # Smoothing should not average in the previous idle zeros.
        assert result.total_risk > 0.0

    def test_dynamic_to_idle_resets_smoothing(self):
        model = RiskModel(smoothing_window=5)
        frame_shape = (480, 640)

        # Run dynamic for several frames.
        for i in range(5):
            model.update(_make_safe_landmarks(hip_shift=(i * 0.02, 0)), frame_shape)

        # Transition to idle.
        idle_result = model.update(_make_safe_landmarks(), frame_shape)
        assert idle_result is not None
        assert idle_result.status == "Idle"
        assert idle_result.total_risk == pytest.approx(0.0)
```

**Verify:**

```bash
python -m pytest test_injury_risk.py -v
```

**Commit:** `feat(idle): integrate idle detection and smoothing reset into RiskModel`

---

### Task 2.2: End-to-end overlay idle integration test

**File:** `test_risk_overlay.py`
**Test:** `test_risk_overlay.py`
**Depends:** 1.2, 2.1

#### Implementation

No production code changes are required in this task; `risk_overlay.py` already supports `Idle` after Task 1.2, and `RiskModel` now emits `Idle` results after Task 2.1. This task only adds an integration-style test that constructs a real `RiskResult` with `Idle` status (the same shape `RiskModel` now produces) and verifies the overlay renders it without error.

#### Test addition

Add this test method to `TestRiskOverlay`:

```python
    @patch("risk_overlay.cv2")
    def test_draw_idle_result_from_model_shape(self, mock_cv2):
        """Idle result produced by RiskModel renders correctly."""
        overlay = RiskOverlay()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = RiskResult(
            total_risk=0.0,
            base_score=0.0,
            interaction_score=0.0,
            normalized={
                "hip_trajectory_deviation": 0.0,
                "knee_flexion": 0.0,
                "foot_alignment": 0.0,
                "landing_pitch": 0.0,
            },
            alerts=[
                {"label": "Hip Control", "txt": "Low", "cls": "ok"},
                {"label": "Knee Overload", "txt": "Low", "cls": "ok"},
                {"label": "Foot Alignment", "txt": "Low", "cls": "ok"},
                {"label": "Kinetic Shock", "txt": "Low", "cls": "ok"},
            ],
            status="Idle",
            profile_label="Balanced",
        )

        overlay.draw(frame, result)

        mock_cv2.rectangle.assert_called()
        mock_cv2.putText.assert_called()
        text_calls = [str(c) for c in mock_cv2.putText.call_args_list]
        joined = " ".join(text_calls)
        self.assertIn("Idle", joined)
        self.assertIn("Balanced", joined)
```

**Verify:**

```bash
python -m pytest test_risk_overlay.py -v
```

**Commit:** `test(idle): add overlay integration test for idle result`

---

## Final Verification

After all batches land:

```bash
python -m pytest test_injury_risk.py test_risk_overlay.py -v
```

All tests should pass. The existing landing-pitch curves and pixel thresholds must remain unchanged; only the new named normalized constant `IDLE_MOVEMENT_THRESHOLD` is introduced.

---

## Out of Scope

- Landing-pitch curve correction.
- Tuning existing hardcoded pixel thresholds in risk curves.
- Body-scale normalization of any risk parameter other than the idle gate itself.
