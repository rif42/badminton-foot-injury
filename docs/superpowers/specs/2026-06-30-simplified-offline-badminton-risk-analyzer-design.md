# Simplified Offline Badminton Injury-Risk Analyzer

## Goal

Build a one-person, offline video analyzer that takes a recorded badminton clip, detects when the player is moving (not standing still), and scores lower-body injury-risk using only the four core pose parameters from the updated baseline.

No live webcam, no event classifier, no public datasets, and no fatigue tracking in this first version.

---

## In Scope

- Command-line tool that reads a video file.
- MediaPipe Pose lower-body landmark extraction.
- Simple moving/standing gate based on hip-center motion.
- Risk scoring using the four core parameters from `../../../.mindmodel/plans/badminton_lower_body_injury_baseline_updated.md`:
  - `Knee_Flexion_Angle`
  - `Ankle_Foot_Alignment_Risk`
  - `Hip_Displacement_Proxy`
  - `Landing_Asymmetry_Score`
- Per-frame CSV output with risk score and component breakdown.
- Optional annotated output video with a simple risk badge.

## Out of Scope

- Live webcam loop.
- Multi-class event classification (`lunge`, `jump_landing`, `direction_change`).
- Public dataset gathering or training.
- `Impact_Load_Proxy`, `Postural_Instability_Score`, and `Fatigue_Modifier`.
- Support-side detection.
- Mobile or web deployment.

---

## Architecture

```text
input video (.mp4 / .avi / .mov)
    │
    ▼
MediaPipe Pose ──► 10 lower-body landmarks per frame
    │
    ▼
Moving/Standing Gate
    │
    ▼
Core Risk Calculator (4 parameters)
    │
    ▼
CSV report + optional annotated video
```

---

## Components

### 1. `video_risk_analyzer.py`

The main CLI entry point. Responsibilities:

- Read a video file with OpenCV.
- Run MediaPipe Pose on each frame.
- Extract the 10 lower-body landmarks.
- Apply the moving/standing gate.
- Compute risk for moving frames.
- Write CSV report and annotated video.

### 2. `baseline_risk.py` (new)

A minimal risk calculator implementing only the four core parameters from the updated baseline.

Responsibilities:

- `knee_flexion_angle(hip, knee, ankle)`
- `ankle_foot_alignment_risk(knee, ankle, heel, foot_index, ...)`
- `hip_displacement_proxy(...)`
- `landing_asymmetry_score(...)`
- `core_risk_score(...)` combining the four with weights `0.20`, `0.30`, `0.15`, `0.35`

### 3. `motion_gate.py` (new)

Simple gate based on hip-center displacement over a short window.

Responsibilities:

- Track hip center across frames.
- Return `moving` if displacement exceeds a threshold over N frames.
- Return `standing` otherwise.

### 4. `test_video_risk_analyzer.py` (new)

Unit tests for the analyzer using synthetic landmark data.

---

## Moving/Standing Gate

Use hip-center horizontal displacement over a 0.3-second window.

```text
hip_center = midpoint(left_hip, right_hip)

displacement = horizontal_distance(hip_center[t], hip_center[t - N])

if displacement > threshold:
    state = moving
else:
    state = standing
```

Suggested defaults:

```text
window = 0.3 seconds (≈ 9 frames at 30 fps)
threshold = 5% of average leg length
```

Risk is only calculated when `state == moving`. Standing frames get `risk_score = 0`.

---

## Risk Scoring

Use the updated baseline's four core parameters and weights.

### Parameters

| Parameter | Calculation | Weight |
|---|---|---|
| `Knee_Flexion_Angle` | `angle_at(hip, knee, ankle)` for each side | 0.20 |
| `Ankle_Foot_Alignment_Risk` | `0.70 * knee_over_foot_score + 0.30 * foot_progression_score` | 0.30 |
| `Hip_Displacement_Proxy` | `horizontal_distance(hip_center, support_foot_center) / leg_length` | 0.15 |
| `Landing_Asymmetry_Score` | `0.35 * knee_asym + 0.25 * hip_height_asym + 0.20 * ankle_height_asym` | 0.35 |

Notes:

- For `Hip_Displacement_Proxy`, use the side with the lower foot (or the more loaded side) as a simple support-side proxy. If unclear, average both sides.
- For `Landing_Asymmetry_Score`, skip the wobble component in this version.

### Core risk score

```text
core_risk =
  0.20 * knee_stiffness_risk
  + 0.30 * ankle_foot_alignment_risk
  + 0.15 * hip_displacement_proxy
  + 0.35 * landing_asymmetry_score
```

Where:

```text
knee_stiffness_risk = clamp((knee_angle - 145) / 35, 0, 1)
```

### Thresholds

```text
0.00–0.34 → acceptable
0.35–0.64 → caution
0.65–1.00 → risky
```

---

## CLI Interface

```bash
python video_risk_analyzer.py input.mp4 \
  --output-csv risk_report.csv \
  --output-video risk_annotated.mp4 \
  --show-preview
```

Arguments:

| Argument | Default | Description |
|---|---|---|
| `input_video` | required | Path to input video |
| `--output-csv` | `risk_report.csv` | Per-frame CSV output |
| `--output-video` | None | Optional annotated output video |
| `--show-preview` | False | Show live preview while processing |

---

## CSV Output Format

```csv
frame,time_sec,is_moving,knee_flexion_left,knee_flexion_right,ankle_foot_alignment_risk,hip_displacement_proxy,landing_asymmetry_score,core_risk,status
```

---

## Testing

- Unit tests with synthetic landmarks for:
  - Standing pose → `is_moving = False`, `core_risk = 0`
  - Good lunge → low risk
  - Knee valgus lunge → high ankle/foot alignment risk
  - Stiff landing → high knee stiffness risk
- End-to-end test on a short pre-recorded clip if available.

---

## Future Extensions (not in this version)

- Live webcam support.
- Multi-class event classification.
- `Impact_Load_Proxy`, `Postural_Instability_Score`, and `Fatigue_Modifier`.
- Support-side detection.
- Public dataset integration.

---

## Spec Self-Review

- **Placeholder scan:** No TBD or TODO items.
- **Internal consistency:** The four core parameters, weights, and thresholds match the updated baseline document.
- **Scope check:** This is a single, focused offline analyzer. It does not need further decomposition.
- **Ambiguity check:** The motion gate defaults and risk formulas are explicit.
