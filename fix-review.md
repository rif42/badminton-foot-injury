# Review Result

I reviewed `fix.md` against the project context. My recommendation is: **accept the direction, but do not apply it as-is**.

The core diagnosis is right: the scorer is treating **standing still** as a risky movement because the current model applies dynamic injury-risk thresholds even when there’s no meaningful motion.

But the proposed fix misses a few important issues.

## What `fix.md` Gets Right

The document correctly identifies that:

- **Straight-leg standing** is being scored as high knee-risk.
- **Foot alignment** becomes unstable when hip velocity is near zero.
- **Hip trajectory deviation** is noisy during standing because MediaPipe jitter creates fake motion.
- The tests currently don’t prove that a standing pose should be low risk.
- A **motion/activity gate** before scoring is the right high-level fix.

I agree with the main idea: **don’t run landing/cutting/lunge injury scoring unless the body is actually moving enough to make those signals meaningful.**

## Problems With The Proposed Fix

The fix should be changed before implementation.

### 1. Landing Pitch Is Also Wrong

`fix.md` frames landing pitch mostly as another standing-still false positive, but there’s a deeper issue: the current scoring appears inverted relative to the project’s injury spec.

The research/spec says:

- **Heel-first / negative pitch** should be safer.
- **Flat or toe-first / positive pitch** should be riskier.

The current scoring seems to treat that backwards. So we should fix the **landing-pitch risk curve**, not just suppress it during idle frames.

### 2. New Status Would Break The Overlay

The proposed `"Idle / No Impact"` status does not match the project’s existing status model.

Current statuses are effectively:

- **Optimal**
- **Caution**
- **High Risk**

Adding a new string without updating the overlay would likely render it with fallback styling. I’d either:

- Use **Optimal** for idle zero-risk frames, or
- Add a clean **Idle** status and update overlay/tests together.

I prefer **Idle** if we want the UI to be explicit.

### 3. Hard-Coded Pixel Thresholds Are Fragile

The proposed fix uses fixed pixel thresholds for motion deadbands. That will behave differently depending on:

- Camera resolution
- Distance from camera
- Body size in frame
- Webcam crop/scale

The better design is to use **body-scale-normalized motion thresholds** wherever possible.

### 4. Smoothing Needs A Policy

If idle results are averaged into the existing smoothing history, the UI may stay “High Risk” briefly after the person stops moving.

We need a clear rule:

- When entering idle, **clear risk smoothing history**, or
- Keep idle outside the smoothing buffer, or
- Smooth status transitions separately from risk values.

I’d choose: **clear active-risk history when entering idle**, because idle means “no evaluable injury event,” not “low-risk version of the previous event.”

## Recommended Fix Shape

I’d revise the fix around this design:

### Core Model

- Add an **activity gate** before biomechanical scoring.
- Use **body-scale-normalized hip movement** to decide whether the frame represents a dynamic event.
- If motion is below threshold, return an **idle/zero-risk result** or a distinct no-score state.
- Add velocity deadbands to alignment and trajectory calculations so jitter doesn’t produce fake directionality.
- Correct the **landing-pitch risk curve** to match the research/spec.

### UI

- If we introduce **Idle**, update the overlay to support it explicitly.
- If we want minimal UI change, map idle to **Optimal**, but that’s less informative.

I’m leaning toward **Idle** because it tells the user: “we’re not seeing a risky movement,” instead of pretending a static pose was actively evaluated as biomechanically optimal.

### Tests

Add tests proving:

- Standing still does **not** produce high risk.
- Tiny jitter does **not** trigger risky foot-alignment or trajectory scores.
- Real movement still produces evaluable risk output.
- Landing pitch follows the documented safe/risky direction.
- Overlay handles the idle state correctly if we add it.

## Final Recommendation

**Do not apply `fix.md` as written.**

Use it as the basis for the fix, but require these changes:

- Add the motion gate.
- Fix the inverted landing-pitch scoring.
- Avoid fixed pixel-only thresholds.
- Define idle behavior cleanly.
- Update overlay and tests together.
- Keep heuristic jitter thresholds separate from research-backed injury thresholds.
