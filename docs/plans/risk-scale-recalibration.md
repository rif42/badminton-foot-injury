# Plan: Recalibrate the risk scale — "yellow = active load, red = injury risk"

Status: **approved by user on 2026-08-14** (scope = sandbox + Python pipeline; stiff
landing stays yellow — only geometric risky presets define red).

## Problem

The gap between the `risky` and `catastrophic` presets in the sandbox is far too large:
the red band starts at 65, so **every** geometric risky preset lands in yellow while the
catastrophic presets sit at 71–79. Verified with a Python mirror of the sandbox math
(current `final` score / band):

| Preset | Score | Band |
|---|---|---|
| good_jump_landing_soft | 26 | green |
| good_right_lunge | 41 | yellow |
| **risky_right_knee_valgus_lunge (knee inward)** | **51** | **yellow** |
| **risky_toe_in_landing** | **49** | **yellow** |
| **risky_overreach_lunge** | **62** | **yellow** |
| **ankle_roll_critical (60° roll)** | **48** | **yellow** (red only via roll-event override) |
| risky_stiff_jump_landing | 39 | yellow (context-driven, stays yellow) |
| catastrophic_knee_collapse_right_lunge | 71 | red |
| catastrophic_extreme_overreach_lunge | 79 | red |

## Target semantics

- **green** = "Acceptable / Low Load" — under 25; essentially only no-event / very low.
- **yellow** = "Active Load" — 25–45; the expected state during normal play, not alarming.
- **red** = "Injury Risk" — ≥ 45; the geometric risky presets (knee inward, 60° roll,
  toe-in, overreach). Severe ankle roll (≥ 45° deviation) still forces red.
- catastrophic presets sit deep red (≥ 65) at the top of the scale.

New band edges: **green < 25 · yellow 25–45 · red ≥ 45** (was 35 / 65).

---

## 1. Recalibrate the sandbox scale — `web/badminton_injury_sandbox_v2.html`

- Change band edges in `updateMetrics()` from 35/65 to **25/45** (risk-bar fill color and
  status selection; roll-event ≥ 45° still forces red).
- Relabel statuses:
  - green: "Acceptable / Low Load" (was "Acceptable Pattern")
  - yellow: **"Active Load"** (was "Caution Pattern")
  - red: **"Injury Risk"** (was "Risky Repeated Pattern")
  - roll-event red text: "Injury Risk — severe ankle roll" (unchanged text style)
- Add a one-line status legend in the side panel
  ("Green = low load · Yellow = active load · Red = injury risk").
- Verify: `node --check` on the extracted `<script type="module">` body.

## 2. Mirror into the Python pipeline

- `src/badminton_risk/video_risk_analyzer.py` — lower the `_CORE_RISK_*` status
  thresholds and clear bands in `_status_with_hysteresis`:
  - caution rise 0.35 → **0.25**; risky rise 0.60 → **0.45**
  - clear bands 0.30 → **0.20** / 0.55 → **0.40** (keeps ~0.05 hysteresis width)
- `src/badminton_risk/injury_risk.py` — `balanced` profile bands
  `green_max 29 / yellow_max 59` → **25 / 45** (live/webcam path; `conservative` and
  `aggressive` keep their relative sensitivities; hysteresis fraction untouched).
- Update tests:
  - `tests/test_video_risk_analyzer.py` — status/hysteresis assertions
    (rise 0.25/0.45, clears 0.20/0.40)
  - `tests/test_injury_risk.py` — bands assertion `{"green_max": 29, "yellow_max": 59}`
    → `{"green_max": 25, "yellow_max": 45}`
- Update `AGENTS.md` documented hysteresis numbers (0.35/0.60 rise, 0.30/0.55 clear
  → 0.25/0.45, 0.20/0.40).

## 3. Verify end-to-end

- Re-run the Python mirror of the final sandbox math across all ~20 presets; assert:
  - the 4 named risky presets ≥ 45 (red)
  - all good presets < 45 (yellow/green)
  - catastrophic presets ≥ 65 (deep red)
  - stiff landing in 25–45 (yellow)
  - print the full per-preset table for review
- `node --check` the sandbox script; run `pytest` (must stay green after test updates).
- Run the offline analyzer on one `data/dataset/` video and eyeball the CSV status
  spread (acceptable/caution/risky all present; risky no longer requires ≥ 0.60).
