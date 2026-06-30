# Public Data Sources for Badminton Live Movement Event Classification

## Purpose

This document summarizes publicly available datasets that can support a live badminton movement-event classifier.

The goal of the event classifier is to identify:

```text
“What movement is happening right now?”
```

before the app calculates injury-risk parameters.

The injury-risk scorer should run **after** event detection.

---

# 1. Target Event Classes

The recommended first version should classify only these movement events:

```text
idle
lunge
jump_landing
direction_change
transition_or_uncertain
```

## Why this is needed

The app should not calculate injury risk blindly on every frame.

Correct pipeline:

```text
Live video
→ MediaPipe pose detection
→ lower-body landmark smoothing
→ movement/event classification
→ event phase detection
→ support-side detection
→ event-specific injury-risk calculation
→ feedback
```

Example:

```text
if event_type == idle:
    risk_score = 0
    output = no_event

elif event_type == lunge:
    calculate lunge-specific injury-risk parameters

elif event_type == jump_landing:
    calculate landing-specific injury-risk parameters

elif event_type == direction_change:
    calculate cutting/planting injury-risk parameters

else:
    output = uncertain
```

---

# 2. Important Finding

There does not appear to be one public dataset that already provides exactly:

```text
badminton video
+ MediaPipe lower-body landmarks
+ event labels:
    idle
    lunge
    jump_landing
    direction_change
+ start/contact/max-load/end frame labels
+ support-side labels
```

So the practical approach is:

```text
combine public badminton/action/pose datasets
+ run MediaPipe yourself
+ manually or semi-automatically relabel a smaller subset
+ fine-tune with your own beginner badminton footage
```

---

# 3. Best Public Data Sources

---

## 3.1 Video-Based Frame-Level Badminton Player Movement Dataset

### Role

Best primary raw video source for movement-event relabeling.

### Why it is useful

This is one of the closest datasets to the event-classifier requirement because it focuses on frame-level badminton player movement.

It can support:

```text
idle
ready stance
lunge
directional change
stroke-related movement
transition / uncertain
```

### Use it for

```text
primary raw video source
manual event labeling
MediaPipe landmark extraction
badminton-specific movement windows
```

### Limitations

```text
You still need to run MediaPipe yourself.
You still need to label event_type and phases manually or semi-automatically.
It may not already contain your exact event labels.
```

### Best use

```text
Primary dataset for creating your own badminton event labels.
```

---

## 3.2 VideoBadminton

### Role

Badminton action-recognition pretraining dataset.

### Why it is useful

VideoBadminton is useful because it already contains badminton action clips and action labels.

It can help the model understand badminton stroke/action context such as:

```text
smash
clear
lift
drive
drop shot
push shot
serve
block
rush shot
```

### Use it for

```text
badminton action pretraining
stroke-context recognition
pose-sequence experiments
action-window localization
```

### Limitations

```text
It tells you what badminton action/stroke is happening.
It does not directly tell you:
- lunge
- jump_landing
- direction_change
- foot contact frame
- max-load frame
```

### Best use

```text
Pretrain the event classifier on badminton clips, then relabel selected clips into your simpler event labels.
```

---

## 3.3 BadmintonGRF

### Role

Specialized dataset for jump landing and impact/contact logic.

### Why it is useful

BadmintonGRF is useful for:

```text
jump_landing
impact event windows
landing/contact timing
pose-to-force relationship
Impact_Load_Proxy validation
```

It is especially useful because it includes synchronized video/pose/force-related data.

### Use it for

```text
jump_landing event support
impact/contact timing
validating Impact_Load_Proxy
understanding landing mechanics
```

### Limitations

```text
It is impact-centric.
It is not a general badminton event-classification dataset.
It may not cover idle, lunge, and directional change broadly.
```

### Best use

```text
Specialized support data for jump_landing and impact detection.
```

---

## 3.4 FineBadminton

### Role

High-level badminton rally and stroke-context dataset.

### Why it is useful

FineBadminton is useful for fine-grained badminton video understanding and action/stroke context.

### Use it for

```text
rally understanding
stroke context
hit-centric window selection
high-level badminton action localization
```

### Limitations

```text
It is not specifically a lower-body footwork event dataset.
It does not directly provide your target event labels.
```

### Best use

```text
Support data for locating important rally moments and stroke windows.
```

---

## 3.5 ShuttleSet

### Role

Stroke-level badminton singles dataset.

### Why it is useful

ShuttleSet can provide rally structure and shot-level context.

### Use it for

```text
rally structure
stroke timing
player position context
shot sequence context
movement forecasting support
```

### Limitations

```text
It is not a video-pose dataset for lunge/jump/directional-change classification.
It should not be used as the core event-type dataset.
```

### Best use

```text
Use as tactical or temporal context, not as the main event classifier dataset.
```

---

## 3.6 Kaggle: Motion Data for Badminton Shots

### Role

Badminton pose/joint-coordinate support dataset.

### Why it is useful

This dataset contains joint-coordinate-style motion data for badminton shots.

It may include useful lower-body joint points such as:

```text
hips
knees
feet
```

### Use it for

```text
pose-coordinate processing
joint-sequence experiments
lower-body feature extraction prototype
testing landmark-derived features
```

### Limitations

```text
It is shot-oriented.
It is not directly labeled as:
- idle
- lunge
- jump_landing
- direction_change
```

### Best use

```text
Good for pose-feature baseline experiments, but still requires event relabeling.
```

---

## 3.7 Kaggle: badminton_storke_video

### Role

Simple badminton stroke-action video dataset.

### Why it is useful

It contains common badminton stroke categories such as:

```text
forehand drive
forehand lift
forehand net shot
forehand clear
backhand drive
backhand net shot
```

### Use it for

```text
simple badminton video classification pretraining
stroke context recognition
basic action-classifier experiments
```

### Limitations

```text
It does not directly label lunge, jump landing, or directional change.
```

### Best use

```text
Secondary support dataset for badminton action context.
```

---

# 4. Similar-Sport Datasets

These datasets should not replace badminton data, but they can help pretrain sequence models.

---

## 4.1 Fencing Footwork Dataset

### Role

Useful similar-sport dataset for lunge and step dynamics.

### Why it is useful

Fencing and badminton both involve forward lunges, recovery steps, and rapid directional movement.

Useful action types include:

```text
step forward
step backward
rapid lunge
incremental speed lunge
lunge with waiting
jumping-sliding lunge
```

### Use it for

```text
lunge vs step recognition
dynamic footwork classification
sequence-model pretraining
support-side and lunge timing concepts
```

### Limitations

```text
Sport is fencing, not badminton.
Movement style is similar but not identical.
```

### Best use

```text
Pretrain lunge/step dynamics, then fine-tune on badminton videos.
```

---

## 4.2 THETIS Tennis Dataset

### Role

Racket-sport 3D skeleton/action dataset.

### Why it is useful

THETIS includes tennis action sequences with skeleton-style data.

### Use it for

```text
racket-sport action recognition
3D skeleton sequence modeling
beginner vs expert movement variation
```

### Limitations

```text
Tennis strokes and footwork differ from badminton.
It is not directly labeled as lunge/jump_landing/directional_change.
```

### Best use

```text
General racket-sport skeleton/action pretraining.
```

---

## 4.3 Tennis Player Actions Dataset on Kaggle

### Role

Static tennis action/posture support.

### Why it is useful

It contains tennis action images and pose/keypoint annotations.

### Use it for

```text
ready-position support
pose-estimation experiments
static posture recognition
```

### Limitations

```text
It is image-based, not video-sequence-based.
It is weak for live dynamic event classification.
```

### Best use

```text
Use only as secondary posture/pose support.
```

---

## 4.4 OpenTTGames / Extended OpenTTGames

### Role

Table-tennis event and posture support dataset.

### Why it is useful

Table tennis is another racket sport with fast reaction movements and short event windows.

### Use it for

```text
racket-sport event spotting
ready stance support
short reaction movement modeling
posture/action context
```

### Limitations

```text
Table-tennis footwork is smaller and less lunge-heavy than badminton.
```

### Best use

```text
General event-spotting support, not final badminton footwork training.
```

---

## 4.5 MPOSE2021

### Role

Generic pose-based human action-recognition pretraining dataset.

### Why it is useful

It provides short-time pose-based action-recognition data.

### Use it for

```text
generic real-time pose-sequence model pretraining
idle/walking/jumping-like separation
short-window action recognition architecture testing
```

### Limitations

```text
Not badminton-specific.
```

### Best use

```text
Use for architecture testing and generic skeleton-sequence pretraining.
```

---

# 5. Recommended Dataset Stack

## Tier 1: Main event data

```text
1. Video-Based Frame-Level Badminton Player Movement Dataset
2. VideoBadminton
3. Your own recorded beginner badminton clips
```

Reason:

```text
These are closest to actual badminton movement.
```

## Tier 2: Specialized badminton support

```text
4. BadmintonGRF
5. Motion Data for Badminton Shots
6. ShuttleSet / FineBadminton
```

Reason:

```text
BadmintonGRF helps jump_landing and impact.
Motion Data helps pose-coordinate processing.
ShuttleSet/FineBadminton help rally/stroke context.
```

## Tier 3: Similar-sport or generic pretraining

```text
7. Fencing Footwork Dataset
8. THETIS Tennis Dataset
9. OpenTTGames / Extended OpenTTGames
10. MPOSE2021
```

Reason:

```text
They help general skeleton-sequence learning, but they are not badminton-specific enough for final behavior.
```

---

# 6. Mapping Public Data to Target Event Labels

| Target Event | Best Public Sources | Notes |
|---|---|---|
| `idle` | Frame-level badminton movement videos, tennis ready-position images, MPOSE2021 | Usually still needs manual labeling. |
| `lunge` | Frame-level badminton videos, Fencing Footwork Dataset, VideoBadminton clips | Fencing data helps pretrain lunge dynamics; badminton data needed for final model. |
| `jump_landing` | BadmintonGRF, VideoBadminton smash clips, badminton movement videos | BadmintonGRF is strongest for contact/impact windows. |
| `direction_change` | Frame-level badminton videos, VideoBadminton, ShuttleSet player positions | Usually needs manual event relabeling. |
| `transition_or_uncertain` | Generated from event boundaries, occlusions, low-confidence pose windows | Usually must be created manually or semi-automatically. |

---

# 7. What Still Needs Manual Labeling

Even with public datasets, you still need labels for:

```text
event_type:
  idle
  lunge
  jump_landing
  direction_change
  transition_or_uncertain

phase:
  start_frame
  contact_frame
  max_load_frame
  end_frame

support_side:
  left
  right
  bilateral
  none
```

Public datasets reduce workload, but they do not fully replace your own labeling.

---

# 8. Recommended Label Format

Use a CSV like this:

```csv
clip_id,athlete_id,event_type,support_side,start_frame,contact_frame,max_load_frame,end_frame,camera_view,pose_quality
p001_lunge_right_001,p001,lunge,right,12,35,47,80,front_45deg,good
p001_idle_001,p001,idle,none,0,,,60,front_45deg,good
p002_jump_landing_001,p002,jump_landing,bilateral,20,44,51,78,side,good
p003_cut_left_001,p003,direction_change,left,14,32,43,68,front_45deg,good
```

Recommended JSON label example:

```json
{
  "clip_id": "p001_lunge_right_001",
  "athlete_id": "p001",
  "event_type": "lunge",
  "support_side": "right",
  "start_frame": 12,
  "contact_frame": 35,
  "max_load_frame": 47,
  "end_frame": 80,
  "camera_view": "front_45deg",
  "pose_quality": "good"
}
```

---

# 9. Recommended Training Sample Format

For the event classifier, use short movement windows.

Example:

```text
30–60 frame window
→ event_type label
```

Input shape example:

```text
30 frames × 10 lower-body landmarks × 3 coordinates
```

Or:

```text
60 frames × 10 lower-body landmarks × 3 coordinates
```

Recommended lower-body landmarks:

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

Optional derived features:

```text
hip center velocity
ankle velocity
knee angle over time
foot displacement
left-right ankle distance
hip height change
support foot stability
```

---

# 10. Practical Build Plan

## Step 1: Collect public video/action data

Start with:

```text
Video-Based Frame-Level Badminton Player Movement Dataset
VideoBadminton
BadmintonGRF
Fencing Footwork Dataset
```

## Step 2: Extract pose landmarks

Run MediaPipe Pose or Pose Landmarker on each video.

Save:

```text
raw video
landmarks per frame
landmark visibility/confidence
fps
camera view
```

## Step 3: Create a small manually labeled subset

Start with:

```text
300–500 labeled clips total
```

Balanced example:

```text
idle: 100
lunge: 100
jump_landing: 100
direction_change: 100
transition_or_uncertain: 100
```

## Step 4: Train a simple event classifier

Input:

```text
sliding 1-second landmark window
```

Output:

```text
idle
lunge
jump_landing
direction_change
transition_or_uncertain
```

Good first model choices:

```text
Random Forest / XGBoost / LightGBM with engineered features
small LSTM / GRU
Temporal CNN
```

## Step 5: Fine-tune with your own beginner data

Public data is mostly from skilled players, match footage, or non-badminton sports.

Your app is for beginners, so you eventually need beginner badminton footage.

Minimum:

```text
20–30 beginner athletes
10–20 clips per event type
```

Better MVP:

```text
50+ beginner athletes
200–500 clips per event type
```

---

# 11. Final Recommendation

The best public-data combination is:

```text
Primary event data:
- Video-Based Frame-Level Badminton Player Movement Dataset
- VideoBadminton

Specialized landing/impact data:
- BadmintonGRF

Pose/joint support:
- Motion Data for Badminton Shots

Lunge pretraining:
- Fencing Footwork Dataset

Generic real-time skeleton pretraining:
- MPOSE2021 or THETIS
```

But the final app will still need a small labeled beginner badminton dataset because the exact labels needed are not already available in one clean public dataset.

Final target data should provide:

```text
short badminton movement clips
+ MediaPipe lower-body landmark sequences
+ event_type labels
+ start/contact/max-load/end frame labels
+ support-side labels
+ many idle/no-event examples
```
