# Ankle Roll Detection — Implementation Plan

Detect ankle roll (inversion/eversion — the foot tilting sideways around the tibia,
the classic ankle-sprain mechanism) in the badminton injury-risk analyzer, and flag
severe roll events.

## Context

- **What exists today:** the analyzer has two risk models. Offline (`video_risk_analyzer.py`
  → `baseline_risk.core_risk_score`) computes 4 components per frame (knee stiffness,
  ankle-foot alignment, hip displacement, landing asymmetry) into a CSV. Live webcam
  (`injury_risk.py` → `RiskResult` + `risk_overlay.py`) uses a profile-based model
  (hip trajectory, knee flexion, foot alignment, landing pitch). **Neither measures
  ankle roll.** `ankle_foot_alignment_risk` covers knee-over-foot deviation and toe-in/out
  (foot progression) only.
- **Planned-but-unimplemented:** `docs/constraints/biomechanics.md` parameter #4
  "Trailing Foot Roll Angle" is exactly this feature.
- **Landmark availability:** MediaPipe Pose gives only knee/ankle/heel/foot_index per
  leg — no toe or outer-malleolus landmarks, so roll must be estimated geometrically.

## Algorithm (from literature review)

**Signed 3D roll of the foot plane relative to the shank axis** — the closest proxy
for subtalar inversion/eversion with the available landmarks:

```
A = ankle,  K = knee,  H = heel,  F = foot_index          (3D coords)
shank axis   t = K − A
foot plane   n = (H − A) × (F − A)     (plantar-surface surrogate)
foot long    a = F − H                 (roll axis)
θ = 90° − angle(n̂, t̂)                  (0° neutral, 90° fully rolled)
signed via projected decomposition about axis a; sign flipped per side (left/right foot).
```

**Thresholds (cited):** normal dynamic frontal-plane excursion ≈ ±5–25°; **≥25–30°
inversion = injury flag** (Siegler 1990 / IEEE 8692518); sprain peaks ~**48°** (Fong
2009); ~90° is beyond physiological ROM → treat as a **binary severe-roll event**, not
a graded score.

**Design-around caveats:** MediaPipe foot landmarks are noisy and the ankle landmark
slides during loading response (myogait R²=0.93 correction); guard the degenerate
triangle (`|n|` near zero); evaluate only during ground contact (dangling-foot frames
are meaningless); per-subject neutral-stance baseline calibration; light smoothing so
the ~50 ms injury peak is not washed out.

## Decisions (user-approved)

- **Scope:** full integration into the offline analyzer (geometry + CSV + core risk + tests).
- **Severity model:** graded 0–1 risk score **plus** a binary severe-roll event flag.
- **Modes:** offline CSV pipeline **and** live webcam HUD.

## Assumptions

- Injury flag at **≥25–30°** roll deviation from neutral baseline; severe event at
  **≥45–50°**; ~90° reading → severe event, with collapsed-triangle guard to reject
  landmark-tracking artifacts.
- Neutral baseline = median roll of the initial standing window (per subject).
- Core-risk reweight: split `ankle_foot_alignment` weight (0.30) into alignment (0.20)
  + roll (0.10); all `_CORE_WEIGHT_*` still sum to 1.0.

---

## 1. Add ankle-roll geometry to `baseline_risk.py`

- Add `ankle_roll_angle()`: foot-plane normal `n = (heel−ankle)×(foot_index−ankle)`,
  shank axis `t = knee−ankle`; `θ = 90° − angle(n̂, t̂)` (0° neutral, 90° fully rolled),
  signed per side using projected-component decomposition about the foot-long axis,
  sign flipped for left vs right foot.
- Add `ankle_roll_risk()`: graded 0–1 (0 at neutral ± deadband, 1.0 at ≥45°)
  plus a `severe_roll` boolean at ≥45–50°.
- Add degenerate-triangle guard (skip when `|n|` near zero / foot landmarks collapsed).
- Extend `core_risk_score()`: bilateral averaged roll; add `ankle_roll_risk`,
  `ankle_roll_angle_deg`, `ankle_roll_event` to the result dict; reweight
  `_CORE_WEIGHT_*` as stated in Assumptions.

## 2. Wire into the offline CSV pipeline (`video_risk_analyzer.py` + `streamlit_app.py`)

- Per frame: compute roll only when the foot is planted (reuse `MotionGate` state +
  heel-low/foot-contact check); otherwise emit neutral/NaN so dangling-foot frames
  don't false-alarm.
- Calibrate neutral baseline from the median roll of the initial standing window.
- Add CSV columns `ankle_roll_risk`, `ankle_roll_angle_deg`, `ankle_roll_event`;
  escalate `status` to risky on a severe event.
- Add the new columns to `streamlit_app.py` display/numeric lists and the injury
  descriptions.

## 3. Add to the live webcam model (`injury_risk.py` + `risk_overlay.py`)

- Add an `ankle_roll` parameter to `RiskProfile`: curves in all 3 `PROFILE_PRESETS`
  (conservative / balanced / aggressive) with the 25–30° / 45°+ thresholds, weight
  rebalance, alert band.
- Feed the same geometry function into the live loop; include roll in
  `RiskResult.normalized`.
- Render the roll angle + a flashing "ANKLE ROLL" alert in `risk_overlay.py`.

## 4. Tests + docs

- Unit tests mirroring existing patterns: math correctness (neutral → 0, rolled → large,
  per-side sign, degenerate guard), graded/severity thresholds, CSV column presence +
  status escalation in the offline analyzer, live profile curves.
- Update `docs/constraints/biomechanics.md` (mark parameter #4 as implemented) and
  `AGENTS.md` (new CSV columns).
- Verify: `pytest` on new + existing tests (note: `tests/test_webcam_leg_pose_detector.py`
  has a pre-existing collection error — `mediapipe` has no `solutions`; leave it
  untouched), plus a CLI smoke run on `data/examples/27.mp4` confirming the new columns.
