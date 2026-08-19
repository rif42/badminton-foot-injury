query: read this github https://github.com/rif42/badminton-foot-injury i have a problem with this project. i want to assess the risk of injury of someone doing sport live by taking video of his lower abdomen, detec it using mediapipe pose, and calculate the risk profile using the mediapipe vertex detection. the problem is that the current calculation seems to be extremely inaccurate, even standing still poses a significant risk profile. what causes this and how to fix it?

answer:
The cause is **not one bug only**. The current model is treating a normal standing frame as a **landing / cutting / impact event**. Because of that, standing still can become “High Risk.”

### What is causing the false high risk

In `injury_risk.py`, the model uses 4 parameters:

```python
hip_trajectory_deviation
knee_flexion
foot_alignment
landing_pitch
```

and combines them with these weights:

```python
knee_flexion: 0.35
foot_alignment: 0.25
landing_pitch: 0.25
hip_trajectory_deviation: 0.15
```

So knee and foot variables dominate the score. ([GitHub][1])

The biggest problem is here:

```python
"knee_flexion": [
    {"max": 145, "r": 0.05},
    {"max": 155, "r": 0.25},
    {"max": 165, "r": 0.6},
    {"max": inf, "r": 0.9},
]
```

A straight leg is about **180°**, and your own test confirms that a straight hip-knee-ankle line returns around 180°. ([GitHub][2])
That means a person standing normally with straight knees is automatically scored as **high knee risk**. This is logical only during **landing/cutting**, because a stiff knee during impact is risky. But while standing still, it should not be risky.

Second problem: `landing_pitch` treats negative foot angle as dangerous:

```python
{"min": 0.001, "r": 0.05},
{"min": -3, "r": 0.35},
{"min": -8, "r": 0.7},
{"min": -inf, "r": 0.9},
```

So if the toe/heel line appears slightly negative in the camera image, the model can score “kinetic shock” as high even when the person is just standing. ([GitHub][1])

Third problem: `foot_alignment` compares foot angle against **hip velocity**:

```python
velocity_angle = _vector_angle(hip_velocity)
diff = abs((foot_angle - velocity_angle + 180) % 360 - 180)
```

When standing still, hip velocity is zero or tiny. With tiny MediaPipe jitter, the velocity direction becomes almost random, so foot alignment can jump into high-risk values. ([GitHub][1])

Fourth problem: `hip_trajectory_deviation` also uses hip movement direction over only 5 frames. If the person is standing still, small landmark jitter can create fake movement direction. The code only returns zero when `vx == 0 and vy == 0`, but real MediaPipe output almost never stays exactly zero because landmarks jitter frame-to-frame. ([GitHub][1])

Also, the test does not catch this. The test only checks that the result is between `0` and `100` and status is one of `"Optimal", "Caution", "High Risk"`; it does **not** assert that a standing pose must be low risk. ([GitHub][2])

---

## Example: why standing becomes high risk

Using your test’s “safe landmarks” standing pose:

```python
right_hip  = (0.55, 0.35)
right_knee = (0.55, 0.55)
right_ankle = (0.55, 0.8)
right_heel = (0.53, 0.82)
right_foot_index = (0.57, 0.81)
```

The model roughly sees:

```text
knee_flexion ≈ 180°     -> high risk 0.9
landing_pitch ≈ -10.6°  -> high risk 0.9
foot_alignment ≈ 10.6°  -> moderate risk 0.3
hip_deviation = 0°      -> low risk 0.05
```

Weighted score becomes around:

```text
base score ≈ 62
interaction bonus ≈ 10
total ≈ 72
```

So standing still becomes **High Risk**, not because MediaPipe is wrong, but because the model’s interpretation is wrong.

---

# How to fix it properly

## 1. Add an “activity/event gate”

Do not calculate injury risk unless the person is actually moving, landing, lunging, or cutting.

Add something like this before scoring:

```python
def movement_magnitude(v: tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])

def body_scale_px(landmarks, frame_shape):
    left_hip = _to_px(landmarks["left_hip"], frame_shape)
    right_hip = _to_px(landmarks["right_hip"], frame_shape)
    left_ankle = _to_px(landmarks["left_ankle"], frame_shape)
    right_ankle = _to_px(landmarks["right_ankle"], frame_shape)

    pelvis_width = math.hypot(
        right_hip[0] - left_hip[0],
        right_hip[1] - left_hip[1],
    )

    leg_height = abs(
        ((left_ankle[1] + right_ankle[1]) / 2)
        - ((left_hip[1] + right_hip[1]) / 2)
    )

    return max(pelvis_width, leg_height, 1.0)
```

Then in `update()`:

```python
hip_velocity = self._compute_hip_velocity()
if hip_velocity is None:
    return None

scale = body_scale_px(landmarks, frame_shape)
speed = math.hypot(*hip_velocity) / scale

if speed < 0.03:
    return RiskResult(
        total_risk=0.0,
        base_score=0.0,
        interaction_score=0.0,
        normalized={
            "hip_trajectory_deviation": 0.0,
            "knee_flexion": 0.0,
            "foot_alignment": 0.0,
            "landing_pitch": 0.0,
        },
        alerts=[],
        status="Idle / No Impact",
        profile_label=self.profile.label,
    )
```

This alone will stop standing still from being marked risky.

---

## 2. Separate “posture risk” from “impact risk”

Your current model is an **impact/landing risk model**, not a general standing posture model.

A straight knee is dangerous during landing, but normal during standing. So knee flexion should be evaluated only during:

```text
landing
lunge
deceleration
direction change
jump landing
rapid step
```

Not during idle frames.

Better logic:

```python
if not is_dynamic_event:
    return idle_or_low_risk
else:
    calculate knee_flexion, foot_alignment, landing_pitch
```

This matches your research notes too: the findings talk about landing, initial contact, cutting, stiff landing, knee valgus, foot strike, and impact loading — not static standing posture. ([GitHub][3])

---

## 3. Do not use hip velocity direction when velocity is tiny

Fix `compute_foot_alignment()`:

```python
def compute_foot_alignment(
    landmarks,
    side,
    frame_shape,
    hip_velocity,
    min_velocity_px=8.0,
) -> float:
    if math.hypot(*hip_velocity) < min_velocity_px:
        return 0.0

    heel = _to_px(landmarks[f"{side}_heel"], frame_shape)
    toe = _to_px(landmarks[f"{side}_foot_index"], frame_shape)

    foot_angle = _vector_angle((toe[0] - heel[0], toe[1] - heel[1]))
    velocity_angle = _vector_angle(hip_velocity)

    return abs((foot_angle - velocity_angle + 180) % 360 - 180)
```

Without this, standing jitter creates random “movement direction.”

---

## 4. Add deadband to hip trajectory

Fix this:

```python
if vx == 0 and vy == 0:
    return 0.0
```

to this:

```python
if math.hypot(vx, vy) < 8.0:
    return 0.0
```

Even better, scale it by body size:

```python
if math.hypot(vx, vy) < 0.03 * body_scale:
    return 0.0
```

MediaPipe Pose outputs landmarks in image/video coordinates and is intended for pose/movement analysis, but per-frame landmark jitter is normal in live video systems; you should not treat tiny frame-to-frame changes as real biomechanics. Google’s Pose Landmarker documentation also notes that the task outputs pose landmarks in image coordinates and world coordinates for images/video/live feeds, so you can use those outputs, but you still need temporal filtering and event logic for sports analytics. ([Google for Developers][4])

---

## 5. Use world landmarks or calibrated camera if possible

Right now you are using 2D screen coordinates:

```python
x * width, y * height
```

That is fragile for foot pitch and alignment because camera angle changes the apparent toe/heel position. A foot can look “heel lower” simply because of perspective.

For better accuracy:

```text
Better: MediaPipe world landmarks / 3D coordinates
Acceptable: 2D + fixed camera angle + calibration
Weak: arbitrary webcam angle + 2D heel/toe vector
```

At minimum, lock the camera setup:

```text
camera fixed
same height
same distance
whole lower body visible
minimal lens distortion
court floor visible
no mirror flip confusion
```

---

## 6. Fix the test suite

Add this test:

```python
def test_standing_still_is_not_high_risk():
    model = RiskModel()
    frame_shape = (480, 640)

    result = None
    for _ in range(10):
        result = model.update(_make_safe_landmarks(), frame_shape)

    assert result is not None
    assert result.total_risk <= 10
    assert result.status in {"Optimal", "Idle / No Impact"}
```

Right now your test allows a standing pose to be High Risk, so the bug passes CI. ([GitHub][2])

---

# Recommended model design

Use this pipeline:

```text
1. Detect pose landmarks
2. Smooth landmarks
3. Estimate motion state:
   - idle
   - walking / preparation
   - lunge
   - landing
   - cutting / direction change
4. Only if dynamic event:
   - calculate knee flexion
   - calculate foot alignment
   - calculate landing pitch
   - calculate hip/trunk control
5. Score risk
6. Smooth risk output
```

So the output becomes:

```text
Standing still        -> Idle / No Impact
Walking slowly        -> Low / Monitoring
Bad lunge landing     -> Caution or High Risk
Hard landing + stiff knee + bad foot angle -> High Risk
```

---

## Most important immediate fix

Add this first:

```python
speed = math.hypot(*hip_velocity) / body_scale_px(landmarks, frame_shape)

if speed < 0.03:
    return idle_result()
```

Then fix knee/foot/pitch scoring so they are only active during dynamic movement. That will solve the “standing still = risky” problem most directly.

[1]: https://github.com/rif42/badminton-foot-injury/blob/master/injury_risk.py "badminton-foot-injury/injury_risk.py at master · rif42/badminton-foot-injury · GitHub"
[2]: https://github.com/rif42/badminton-foot-injury/blob/master/test_injury_risk.py "badminton-foot-injury/test_injury_risk.py at master · rif42/badminton-foot-injury · GitHub"
[3]: https://github.com/rif42/badminton-foot-injury/blob/master/research_badminton_injury_params/findings_knee_flexion_landing_pitch.md "badminton-foot-injury/research_badminton_injury_params/findings_knee_flexion_landing_pitch.md at master · rif42/badminton-foot-injury · GitHub"
[4]: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker?utm_source=chatgpt.com "Pose landmark detection guide | Google AI Edge"
