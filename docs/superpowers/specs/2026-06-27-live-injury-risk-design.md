# Live Injury-Risk Calculation for MediaPipe Pose

**Goal:** Add real-time injury-risk scoring to the existing webcam lower-body pose detector, using the weighted profile model from `injury-sim.html` and the biomechanical parameters documented in `injury.md`.

**Architecture:** Keep `PoseDetector` responsible only for MediaPipe landmark extraction. Introduce a dedicated `injury_risk.py` module that converts landmarks into biomechanical parameters, runs the profile-based risk model, and smooths the output. Introduce `risk_overlay.py` to render the score and alerts on the OpenCV feed. The existing `webcam_leg_pose_detector.py` orchestrates the frame loop and forwards landmarks to the risk/overlay modules.

**Tech Stack:** Python 3.11+, MediaPipe Pose, OpenCV, NumPy, `collections.deque` for rolling buffers, `unittest`/`pytest` for testing.

---

## Context

- `webcam_leg_pose_detector.py` already exposes live MediaPipe Pose landmarks for the lower body (hips, knees, ankles, heels, foot indices) and draws the skeleton.
- `injury.md` defines the biomechanical parameters and safe/dangerous thresholds.
- `injury-sim.html` implements a weighted risk model with three profiles (`conservative`, `balanced`, `aggressive`), piecewise curves per parameter, interaction bonuses, and a 0-100 total risk score.
- `research_badminton_injury_params/` provides evidence supporting the four parameters and their interactions.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `webcam_leg_pose_detector.py` | Modify | Wire `RiskModel` and `RiskOverlay` into the frame loop; handle profile-switching keypress. |
| `injury_risk.py` | Create | Landmark-to-parameter math, profile configs, `RiskModel` class with smoothing buffers, `RiskResult` dataclass. |
| `risk_overlay.py` | Create | Draw status badge, total risk, per-parameter alerts, and current profile on the OpenCV frame. |
| `test_injury_risk.py` | Create | Unit tests for parameter extraction, profile scoring, smoothing, and low-visibility handling. |
| `test_risk_overlay.py` | Create | Mock-based tests verifying the HUD is drawn. |

---

## Data Flow

1. `PoseDetector.process(frame)` returns raw MediaPipe landmarks.
2. `PoseDetector.get_lower_body_landmarks(landmarks)` returns a dictionary `{joint_name: (x, y, z, visibility)}`.
3. `RiskModel.update(landmarks_dict, frame_shape)`:
   - Returns `None` if any required landmark has `visibility < 0.5`.
   - Identifies the **front leg** as the side whose ankle has the larger normalized `y` (lower in the frame).
   - Maintains a 5-frame rolling history of the hip center in pixel coordinates to compute a velocity vector.
   - Computes four live parameters in pixel space:
     - `hip_trajectory_deviation`: absolute angle between the hip-center velocity vector and the vertical axis.
     - `knee_flexion`: internal angle at the front knee (hip-knee-ankle). Larger values mean a stiffer, more extended landing.
     - `foot_alignment`: absolute angle between the front foot vector (heel → foot index) and the hip-center velocity vector.
     - `landing_pitch`: angle of the front foot vector relative to the horizontal. Negative = heel-first, positive = toe-first.
   - Evaluates the piecewise risk curves for the active profile, computes the weighted base score, adds interaction bonuses, and clamps the total to `[0, 100]`.
   - Averages the result over the last 3 frames for display stability.
4. `RiskOverlay.draw(frame, result)` renders the HUD.
5. `main()` polls the keyboard each frame; pressing `p` cycles through the profiles (`balanced` → `conservative` → `aggressive` → `balanced`).

---

## Parameter Mapping Notes

The four live parameters are mapped from landmarks to match the intent of `injury.md` and the structure of the `injury-sim.html` model. They are first approximations and can be refined with real-world testing:

- `hip_trajectory_deviation` replaces the simulation’s `hipTrajectory` slider with a real CoM-trajectory deviation measurement.
- `knee_flexion` is the internal knee angle; a straighter leg gives a larger angle and higher risk, consistent with the research findings on stiff landings.
- `foot_alignment` is the foot-to-trajectory alignment described in `injury.md`.
- `landing_pitch` matches the foot-strike pitch angle in `injury.md`.

---

## Error Handling

- **Low visibility:** If any landmark required for a parameter is missing or below threshold, `RiskModel.update()` returns `None`. The overlay displays “Insufficient data — pose not fully visible.”
- **No pose detected:** The existing detector already handles this; the overlay is skipped.
- **Invalid geometry:** Division by zero or degenerate angles are guarded with epsilon checks; the model returns the safest fallback for that parameter.

---

## Testing Strategy

- **Pure unit tests for `injury_risk.py`:** Build synthetic landmark dictionaries that represent known safe and dangerous poses and assert the expected risk level/profile behavior.
- **Smoothing tests:** Feed a sequence of synthetic frames and verify the displayed score is a moving average.
- **Profile tests:** Verify each profile produces different thresholds and that `p`/setter cycling works.
- **Overlay tests:** Mock `cv2` drawing functions and assert the status badge, risk text, and per-parameter alerts are drawn.
- **Integration test:** Patch `VideoCapture`, `Pose`, and `waitKey` to simulate a few frames with landmarks and verify the risk path is exercised and profile switching responds to a keypress.

---

## UI Sketch

- Top-left status badge: green “✅ Form Optimal”, yellow “⚠️ Caution Zone”, or red “⛔ High Injury Risk” with the total risk score.
- Below the badge: current profile name.
- Right side or lower-left: per-parameter alert list (Hip Control, Knee Overload, Foot Alignment, Kinetic Shock) with Low/Moderate/High labels.

---

## Non-Goals

- No video recording or CSV logging in this version.
- No calibration to real-world units; all angles are computed from the 2D image plane.
- No upper-body or arm landmarks.
