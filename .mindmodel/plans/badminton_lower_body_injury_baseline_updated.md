# Badminton Lower-Body Injury-Risk Baseline

## Purpose

This document defines a practical baseline for a beginner-focused badminton footwork analysis system using **lower-body pose data only**.

The goal is **not** to medically predict injury. The goal is to detect repeated lower-body movement patterns that may become risky if repeated during badminton training or play.

Recommended feedback language:

> “This movement shows a risky landing/lunge pattern. Try correcting your form.”

Avoid overclaiming:

> “You will get injured.”

---

# 1. Detection Scope

## Supported pose landmarks

The baseline assumes only hip-and-below pose landmarks:

```text
left_hip
right_hip
left_knee
right_knee
left_ankle
right_ankle
left_heel
right_heel
left_foot_index
right_foot_index
```

No shoulder, arm, hand, racket, or upper-torso points are required.

## Primary badminton actions

Only score risk during dynamic lower-body events:

```text
lunge
jump_landing
direction_change
```

Do **not** score standing still, idle pose, or slow walking as risky.

## Recommended output classes

```text
no_event
acceptable_landing_or_lunge
caution
risky_landing_or_lunge
uncertain
```

---

# 2. Important Renaming for Accuracy

The original baseline used these four parameters:

```text
Knee_Flexion_Angle
Ankle_Valgus_Angle
Trunk_Lean_Angle
Landing_Asymmetry_Score
```

However, with **lower-body-only MediaPipe landmarks**, two of those names are too strong:

| Original Name | Updated Lower-Body-Only Name | Reason |
|---|---|---|
| `Knee_Flexion_Angle` | `Knee_Flexion_Angle` | Can be calculated directly from hip-knee-ankle. |
| `Ankle_Valgus_Angle` | `Ankle_Foot_Alignment_Risk` | True clinical ankle valgus cannot be measured accurately from MediaPipe heel/toe points only. |
| `Trunk_Lean_Angle` | `Hip_Displacement_Proxy` / `Pelvis_Control_Proxy` | True trunk lean requires shoulders or upper torso landmarks. |
| `Landing_Asymmetry_Score` | `Landing_Asymmetry_Score` | Can be estimated from left-right lower-body differences. |

Final lower-body-only baseline parameters:

```text
1. Knee_Flexion_Angle
2. Ankle_Foot_Alignment_Risk
3. Hip_Displacement_Proxy
4. Landing_Asymmetry_Score
5. Impact_Load_Proxy
6. Postural_Instability_Score
7. Fatigue_Modifier
```

---

# 3. Shared Geometry Helpers

The same helper formulas are used for most parameters.

## Point and vector

For two 3D points `a` and `b`:

```text
vector(a, b) = b - a
```

## Distance

```text
distance(a, b) = ||b - a||
```

## Angle at a joint

For three points `a`, `b`, and `c`, the angle at point `b` is:

```text
angle_at(a, b, c)
```

This is used for:

```text
hip-knee-ankle angle
knee-ankle-toe angle
```

## Hip center

```text
hip_center = midpoint(left_hip, right_hip)
```

## Foot center

For each side:

```text
foot_center = midpoint(heel, foot_index)
```

## Leg length normalization

For each side:

```text
leg_length = distance(hip, knee) + distance(knee, ankle)
```

Most distance-based values should be normalized by leg length so they are less affected by player height or camera scale.

---

# 4. Parameter Definitions and Calculations

---

## 4.1 Knee_Flexion_Angle

### Meaning

Measures how bent or straight the knee is.

A straight leg is close to:

```text
180 degrees
```

A bent knee has a smaller angle.

### Required landmarks

For each side:

```text
hip
knee
ankle
```

### Calculation

```text
Knee_Flexion_Angle = angle_at(hip, knee, ankle)
```

### Interpretation

```text
180°  = nearly straight
145°  = moderate bend
120°  = deep bend
```

### When to use

Only evaluate it during:

```text
lunge
jump_landing
direction_change
```

Do not treat a straight knee during standing as risky.

### Risk pattern

A risky pattern may occur when:

```text
knee stays too straight during landing/loading
knee does not bend enough to absorb impact
knee angle is poor during lunge or direction change
```

### Simple risk normalization

For dynamic events:

```text
knee_stiffness_risk = clamp((knee_angle - 145) / 35, 0, 1)
```

Where:

```text
0 = low stiffness concern
1 = high stiffness concern
```

---

## 4.2 Ankle_Foot_Alignment_Risk

### Meaning

A practical proxy for ankle/foot misalignment.

This replaces `Ankle_Valgus_Angle` because true ankle valgus cannot be measured reliably from lower-body pose landmarks alone.

### Required landmarks

For each side:

```text
knee
ankle
heel
foot_index
```

### Component A: Knee-over-foot deviation

This checks whether the knee is aligned over the support foot.

```text
foot_center = midpoint(heel, foot_index)

knee_over_foot_deviation =
abs(knee.x - foot_center.x) / leg_length
```

This is especially useful in a front or diagonal camera view.

### Component B: Foot progression angle

This checks whether the foot points too far inward or outward.

Using the heel-to-toe vector on the ground plane:

```text
foot_vector = foot_index - heel

foot_progression_angle =
atan2(foot_vector.x, foot_vector.z)
```

A larger absolute angle means more toe-in or toe-out relative to the expected forward direction.

### Combined calculation

```text
knee_dev_score = clamp(knee_over_foot_deviation / 0.22, 0, 1)

foot_angle_score = clamp(abs(foot_progression_angle) / 45°, 0, 1)

Ankle_Foot_Alignment_Risk =
0.70 * knee_dev_score
+ 0.30 * foot_angle_score
```

### Interpretation

```text
0.0 = good alignment
1.0 = large misalignment
```

### Risk pattern

A risky pattern may occur when:

```text
knee collapses inward relative to foot
foot points too far inward
knee and foot are not aligned during landing/lunge
support foot looks unstable
```

---

## 4.3 Hip_Displacement_Proxy

### Meaning

A lower-body-only replacement for `Trunk_Lean_Angle`.

True trunk lean requires shoulder or upper-torso landmarks. With hips and below only, the best alternative is to estimate whether the pelvis/hip center is too far from the support foot.

### Required landmarks

```text
left_hip
right_hip
support_side_heel
support_side_foot_index
support_side_hip
support_side_knee
support_side_ankle
```

### Calculation

```text
hip_center = midpoint(left_hip, right_hip)

support_foot_center = midpoint(support_heel, support_foot_index)

Hip_Displacement_Proxy =
horizontal_distance(hip_center, support_foot_center) / support_leg_length
```

Use the horizontal plane:

```text
horizontal_distance = sqrt((hip_center.x - foot_center.x)^2 + (hip_center.z - foot_center.z)^2)
```

### Interpretation

```text
low value  = pelvis is controlled near the support base
high value = pelvis/body mass is far from the support foot
```

### Risk pattern

A risky pattern may occur when:

```text
player overreaches during lunge
pelvis shifts too far outside support base
balance is poor during direction change
recovery becomes unstable
```

---

## 4.4 Landing_Asymmetry_Score

### Meaning

Measures imbalance between left and right sides during landing, lunge, or recovery.

### Required landmarks

Both sides:

```text
hips
knees
ankles
heels
foot_index points
```

### Component A: Knee angle asymmetry

```text
left_knee_angle = angle_at(left_hip, left_knee, left_ankle)

right_knee_angle = angle_at(right_hip, right_knee, right_ankle)

knee_asymmetry_score =
clamp(abs(left_knee_angle - right_knee_angle) / 60°, 0, 1)
```

### Component B: Hip height asymmetry

```text
pelvis_width = distance(left_hip, right_hip)

hip_height_score =
clamp(abs(left_hip.y - right_hip.y) / (pelvis_width * 0.8), 0, 1)
```

### Component C: Ankle height asymmetry

```text
ankle_height_score =
clamp(abs(left_ankle.y - right_ankle.y) / (average_leg_length * 0.20), 0, 1)
```

### Component D: Wobble score

If using time-series data, measure hip-center spread after contact:

```text
wobble_score =
normalized spread of hip_center over post-contact frames
```

If the simulation is static, this can be represented by a manual slider.

### Combined calculation

```text
Landing_Asymmetry_Score =
0.35 * knee_asymmetry_score
+ 0.25 * hip_height_score
+ 0.20 * ankle_height_score
+ 0.20 * wobble_score
```

### Interpretation

```text
0.0 = balanced
1.0 = highly asymmetric or unstable
```

### Risk pattern

A risky pattern may occur when:

```text
one side absorbs much more load
one hip drops significantly
one knee bends much more than the other
one foot contacts or stabilizes much later
player wobbles after landing
```

---

## 4.5 Impact_Load_Proxy

### Meaning

Estimated landing/loading intensity.

This is **not** true ground reaction force. It is a camera-based proxy.

### Required landmarks

At minimum:

```text
left_hip
right_hip
knees
ankles
```

### Required data type

This parameter needs movement over time, not a single frame.

### Component A: Hip vertical speed

Track hip center across frames:

```text
hip_vertical_speed =
vertical velocity of hip_center before or during contact
```

### Component B: Hip vertical deceleration

```text
hip_vertical_deceleration =
change in hip vertical velocity after contact
```

### Component C: Knee stiffness during impact

```text
stiffness_score =
clamp((support_knee_angle - 150°) / 30°, 0, 1)
```

A straighter support knee during landing usually increases stiffness concern.

### Combined calculation

```text
Impact_Load_Proxy =
0.35 * impact_speed_score
+ 0.40 * deceleration_score
+ 0.25 * stiffness_score
```

### Interpretation

```text
0.0 = soft or controlled loading
1.0 = hard landing/loading proxy
```

### Risk pattern

A risky pattern may occur when:

```text
landing is fast
body stops suddenly
knee remains stiff during impact
hard landing combines with poor alignment
```

---

## 4.6 Postural_Instability_Score

### Meaning

Measures instability after landing, lunging, or changing direction.

### Required landmarks

```text
hips
knees
ankles
heels
foot_index points
```

### Required data type

Best calculated from a short post-contact window:

```text
0.3 to 0.8 seconds after foot contact
```

### Component A: Hip center wobble

```text
hip_wobble_score =
normalized spread of hip_center after contact
```

### Component B: Knee alignment jitter

Track knee-over-foot deviation over several frames:

```text
knee_alignment_jitter =
standard deviation of knee_over_foot_deviation
```

### Component C: Recovery time

Measure time until the player stabilizes:

```text
recovery_time =
frames until hip speed, ankle movement, and knee angle changes become low
```

### Simplified combined calculation

```text
Postural_Instability_Score =
0.60 * hip_wobble_score
+ 0.40 * knee_alignment_jitter_score
```

In the 3D sandbox, this is represented by:

```text
manual wobble slider
+ alignment risk
+ hip displacement proxy
```

### Interpretation

```text
0.0 = stable landing/recovery
1.0 = unstable landing/recovery
```

### Risk pattern

A risky pattern may occur when:

```text
player wobbles after landing
knee/ankle jitters after contact
player needs extra correction steps
recovery to ready stance is slow
```

---

## 4.7 Fatigue_Modifier

### Meaning

Adjusts risk based on signs of form degradation over repeated actions.

Fatigue should not be calculated from one pose. It needs session history.

### Required data

```text
event count
movement speed over time
recovery time over time
risk score trend over time
landing asymmetry trend over time
postural instability trend over time
```

### Component A: Repetition count

```text
repetition_fatigue_score =
clamp(event_count / max_expected_events, 0, 1)
```

Example:

```text
max_expected_events = 80
```

### Component B: Movement speed drop

```text
speed_drop_score =
clamp((early_average_speed - recent_average_speed) / early_average_speed / 0.30, 0, 1)
```

A 30% speed drop can be treated as a strong fatigue signal.

### Component C: Recovery time increase

```text
recovery_time_increase_score =
clamp((recent_recovery_time - early_recovery_time) / early_recovery_time / 0.50, 0, 1)
```

A 50% recovery-time increase can be treated as a strong fatigue signal.

### Component D: Form degradation

```text
form_degradation_score =
clamp((recent_risk_score - early_risk_score) / 0.30, 0, 1)
```

### Combined calculation

```text
Fatigue_Modifier =
0.25 * repetition_fatigue_score
+ 0.25 * speed_drop_score
+ 0.25 * recovery_time_increase_score
+ 0.25 * form_degradation_score
```

### Important rule

Fatigue should not produce a high-risk alert by itself.

Recommended usage:

```text
final_score = core_pose_score + 0.10 to 0.15 * Fatigue_Modifier
```

---

# 5. Event-Gated Algorithm

The most important rule is:

```text
Do not score risk when there is no dynamic event.
```

## Stage 1: Detect event type

Classify the current movement window as one of:

```text
no_event
lunge
jump_landing
direction_change
uncertain
```

Simple event examples:

```text
lunge:
one foot moves far forward/sideways and support knee loads

jump_landing:
both feet or one foot contact after vertical drop

direction_change:
foot plants and hip direction changes

no_event:
standing, idle, ready stance, slow walking
```

## Stage 2: If no event, stop

```text
if event_type == no_event:
    output = no_event
    risk_score = 0
```

This prevents standing still from being treated as risky.

## Stage 3: Determine support side

For single-leg or lunge-like actions:

```text
support_side = side with larger loading / lower foot / larger knee bend / forward planted foot
```

For jump landing:

```text
support_side can be dominant side or both sides
```

## Stage 4: Calculate core pose parameters

```text
Knee_Flexion_Angle
Ankle_Foot_Alignment_Risk
Hip_Displacement_Proxy
Landing_Asymmetry_Score
```

## Stage 5: Calculate optional temporal modifiers

```text
Impact_Load_Proxy
Postural_Instability_Score
Fatigue_Modifier
```

## Stage 6: Score

Recommended initial core weights:

```text
Knee_Flexion_Angle risk        0.20
Ankle_Foot_Alignment_Risk     0.30
Hip_Displacement_Proxy        0.15
Landing_Asymmetry_Score       0.35
```

Core score:

```text
core_pose_score =
0.20 * knee_stiffness_risk
+ 0.30 * ankle_foot_alignment_risk
+ 0.15 * hip_displacement_proxy
+ 0.35 * landing_asymmetry_score
```

Final score:

```text
final_score =
core_pose_score
+ 0.12 * Impact_Load_Proxy
+ 0.13 * Postural_Instability_Score
+ 0.10 * Fatigue_Modifier
```

Clamp the final score:

```text
final_score = clamp(final_score, 0, 1)
```

Convert to 0–100:

```text
final_risk_score = round(final_score * 100)
```

---

# 6. Recommended Output Thresholds

```text
0–34:
acceptable_landing_or_lunge

35–64:
caution

65–100:
risky_landing_or_lunge
```

If pose quality is poor, landmarks are missing, or camera angle is unsuitable:

```text
output = uncertain
```

---

# 7. Feedback Mapping

## Knee stiffness

Condition:

```text
high knee_stiffness_risk
```

Feedback:

> “Your knee is too straight during landing/loading. Try bending the knee more to absorb impact.”

## Knee/foot misalignment

Condition:

```text
high Ankle_Foot_Alignment_Risk
```

Feedback:

> “Your knee and foot are not aligned. Try keeping your knee tracking in the same direction as your toes.”

## Hip displacement / overreach

Condition:

```text
high Hip_Displacement_Proxy
```

Feedback:

> “Your body is too far from your support foot. Try shortening the step and keeping your balance over the landing foot.”

## Asymmetry

Condition:

```text
high Landing_Asymmetry_Score
```

Feedback:

> “Your landing looks uneven between left and right sides. Try landing more balanced and controlled.”

## Hard landing

Condition:

```text
high Impact_Load_Proxy
```

Feedback:

> “Your landing appears hard or abrupt. Try landing more softly with better knee bend.”

## Instability

Condition:

```text
high Postural_Instability_Score
```

Feedback:

> “You appear unstable after landing. Try stabilizing before your next movement.”

## Fatigue degradation

Condition:

```text
high Fatigue_Modifier and recent risk scores are increasing
```

Feedback:

> “Your form appears to be degrading over repeated movements. Consider resting or reducing intensity.”

---

# 8. 3D Sandbox Preset Groups

The 3D sandbox uses movable/configurable lower-body points to test how the scoring responds.

## Good patterns

```text
Good split step
Good right lunge
Good left lunge
Good soft landing
Good direction change
```

Purpose:

```text
test whether technically acceptable geometry produces low score
```

## Risky patterns

```text
Risky right knee inward
Risky left knee inward
Risky toe-in landing
Risky overreach lunge
Risky pelvis shift
Risky stiff landing
Risky asymmetric landing
```

Purpose:

```text
test whether common beginner mistakes produce caution or risky scores
```

## Catastrophic stress tests

```text
Extreme knee collapse
Extreme crossed feet
Extreme overreach
Extreme ankle roll proxy
Extreme asymmetry drop
```

Purpose:

```text
stress-test whether the algorithm reacts strongly to obviously extreme geometry
```

Important:

```text
Catastrophic presets are exaggerated algorithm tests, not medical predictions.
```

---

# 9. Accuracy Expectations

## What this baseline can represent well

```text
joint-angle geometry
knee-over-foot alignment proxy
hip displacement from support base
left-right asymmetry
event-gated scoring logic
simple risk feedback
```

## What it can only approximate

```text
true ankle valgus
true trunk lean
impact force
ground reaction force
muscle load
fatigue physiology
joint torque
court friction
shoe behavior
```

## Practical accuracy level

| Use Case | Expected Usefulness |
|---|---:|
| Explaining algorithm behavior | High |
| Debugging formulas | High |
| Testing static pose scoring | High |
| Simulating MediaPipe landmark geometry | Medium |
| Estimating real badminton movement quality | Medium if validated |
| Predicting real injury | Low without clinical/sports-science validation |

---

# 10. Recommended MVP Implementation

## Minimum viable baseline

```text
event_type
support_side
Knee_Flexion_Angle
Ankle_Foot_Alignment_Risk
Hip_Displacement_Proxy
Landing_Asymmetry_Score
```

## Stronger baseline

```text
event_type
support_side
Knee_Flexion_Angle
Ankle_Foot_Alignment_Risk
Hip_Displacement_Proxy
Landing_Asymmetry_Score
Impact_Load_Proxy
Postural_Instability_Score
Fatigue_Modifier
```

## Suggested pipeline

```text
video input
→ MediaPipe lower-body landmarks
→ landmark smoothing
→ event detection
→ support-side detection
→ parameter calculation
→ event-gated risk score
→ feedback message
→ session-level fatigue tracking
```

---

# 11. Final Recommended Parameter List

```text
1. Knee_Flexion_Angle
2. Ankle_Foot_Alignment_Risk
3. Hip_Displacement_Proxy
4. Landing_Asymmetry_Score
5. Impact_Load_Proxy
6. Postural_Instability_Score
7. Fatigue_Modifier
```

This is the most honest and practical lower-body-only version of the badminton injury-risk baseline.
