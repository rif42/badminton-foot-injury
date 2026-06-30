# Badminton Live Event-Type Dataset Evaluation & Selection Plan

> **For agentic workers:** This is a research/data-planning task, not an implementation plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate publicly available datasets and determine the most compatible, accurate source(s) for training and validating a live badminton lower-body event classifier (`idle`, `lunge`, `jump_landing`, `direction_change`, `transition_or_uncertain`).

**Architecture:** Combine a small set of public badminton video/pose datasets with manually relabeled clips, then score each candidate against a fixed compatibility rubric to decide the primary training stack.

**Tech Stack:** MediaPipe Pose Landmarker, Python (data scripts), CSV/JSON label format, optional FFmpeg for video handling.

---

## 1. Baseline Verification Summary

I read the three reference files:

- `badminton_injury_sandbox_v2.html`
- `badminton_lower_body_injury_baseline_updated.md`
- `public_datasets_for_badminton_event_classifier.md`

### 1.1 Parameters checked in the sandbox

The sandbox implements exactly the seven lower-body-only parameters listed in the updated baseline:

| Parameter | Sandbox Function | Baseline Section | Match |
|---|---|---|---|
| `Knee_Flexion_Angle` | `kneeFlexion(side)` | 4.1 | ✅ |
| `Ankle_Foot_Alignment_Risk` | `ankleFootAlignmentRisk(side)` | 4.2 | ✅ |
| `Hip_Displacement_Proxy` | `hipDisplacementProxy(supportSide)` | 4.3 | ✅ |
| `Landing_Asymmetry_Score` | `landingAsymmetryScore()` | 4.4 | ✅ |
| `Impact_Load_Proxy` | `impactLoadProxy(side)` | 4.5 | ✅ |
| `Postural_Instability_Score` | `posturalInstabilityScore(side)` | 4.6 | ✅ |
| `Fatigue_Modifier` | `fatigueModifier()` | 4.7 | ✅ |

### 1.2 Algorithm weights and thresholds

The sandbox scoring matches the baseline algorithm exactly:

```text
core_pose_score =
  0.20 * knee_stiffness_risk
  + 0.30 * ankle_foot_alignment_risk
  + 0.15 * hip_displacement_proxy
  + 0.35 * landing_asymmetry_score

final_score =
  core_pose_score
  + 0.12 * impact_load_proxy
  + 0.13 * postural_instability_score
  + 0.10 * fatigue_modifier
```

Thresholds:

```text
0–34   → acceptable_landing_or_lunge
35–64  → caution
65–100 → risky_landing_or_lunge
```

### 1.3 Event gating

Both documents agree: `no_event` / standing / idle should produce `risk_score = 0` and should not be scored. The sandbox returns zero for all risk components when `eventType === "no_event"`.

### 1.4 Minor notes

- **Postural instability formula difference:** The baseline text proposes `0.60 * hip_wobble + 0.40 * knee_alignment_jitter` for time-series data, while the sandbox uses `0.55 * wobble + 0.25 * alignment + 0.20 * hip_displacement`. The baseline already acknowledges that the sandbox version is a simplified static representation, so this is not a bug.
- **Supported landmarks:** Both limit input to the 10 lower-body landmarks. Good.
- **Renaming:** The renaming table (`Ankle_Valgus_Angle` → `Ankle_Foot_Alignment_Risk`, `Trunk_Lean_Angle` → `Hip_Displacement_Proxy`) is applied consistently.

**Verification result:** The baseline parameters and algorithm are internally consistent between the markdown spec and the HTML sandbox. No corrections needed before moving to dataset planning.

---

## 2. What the Event Classifier Needs

The event detector must run **before** the injury-risk scorer. For live badminton, the dataset should ideally provide:

### 2.1 Required labels

```text
event_type:
  idle
  lunge
  jump_landing
  direction_change
  transition_or_uncertain

phase (optional but valuable):
  start_frame
  contact_frame
  max_load_frame
  end_frame

support_side:
  left
  right
  bilateral
  none

metadata:
  camera_view
  pose_quality
  fps
```

### 2.2 Required pose input

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

### 2.3 Data characteristics for live use

| Characteristic | Ideal | Minimum |
|---|---|---|
| Video source | Badminton match / training footage | Similar racket sport or motion-capture data |
| Frame rate | ≥ 30 fps | ≥ 25 fps |
| Camera view | Side or front-diagonal | Any fixed view |
| Athlete level | Beginners (target user) | Mixed levels acceptable |
| Event balance | Roughly equal per class | At least 50 clips per class |
| Temporal window | 30–60 frames per clip | 20+ frames |

---

## 3. Candidate Dataset Inventory

All datasets come from `public_datasets_for_badminton_event_classifier.md`. They are grouped into tiers based on likely usefulness.

### 3.1 Tier 1 — Primary badminton event data

| Dataset | Expected Role | Why Consider |
|---|---|---|
| Video-Based Frame-Level Badminton Player Movement Dataset | Primary raw video + relabeling | Frame-level badminton movement; closest to target events |
| VideoBadminton | Pretraining + context | Badminton action clips; can be relabeled into event windows |
| Own recorded beginner badminton clips | Final fine-tuning | Only source guaranteed to match target user population |

### 3.2 Tier 2 — Specialized badminton support

| Dataset | Expected Role | Why Consider |
|---|---|---|
| BadmintonGRF | Jump landing + impact | Synchronized video/pose/force; validates `Impact_Load_Proxy` |
| Kaggle: Motion Data for Badminton Shots | Pose-coordinate experiments | Joint-coordinate data for lower-body feature testing |
| ShuttleSet / FineBadminton | Rally/stroke context | Helps locate important temporal windows |

### 3.3 Tier 3 — Similar-sport or generic pretraining

| Dataset | Expected Role | Why Consider |
|---|---|---|
| Fencing Footwork Dataset | Lunge/step pretraining | Forward lunge dynamics are similar to badminton |
| THETIS Tennis Dataset | Racket-sport skeleton pretraining | 3D skeleton sequences, action recognition |
| Tennis Player Actions Dataset (Kaggle) | Static posture support | Image-based pose annotations |
| OpenTTGames / Extended OpenTTGames | Event-spotting support | Fast reaction movements in racket sport |
| MPOSE2021 | Generic pose-action pretraining | Short-window pose-based action recognition |

---

## 4. Evaluation Rubric

Each dataset will be scored 0–3 on each criterion. The dataset with the highest weighted total is the primary recommendation.

### 4.1 Scoring criteria

| Criterion | Weight | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| **Badminton specificity** | 25% | Not badminton or similar sport | Similar racket/footwork sport | Badminton-adjacent | Direct badminton footage |
| **Event label compatibility** | 25% | No usable labels | Requires heavy relabeling | Partial labels mappable | Close to target labels |
| **Lower-body pose availability** | 20% | No pose data | Upper-body only | Full-body pose | Lower-body or easily extractable |
| **Temporal resolution** | 10% | Static images only | Low fps / short clips | 25–30 fps video | ≥ 30 fps video |
| **Accessibility / license** | 10% | Restricted / unavailable | Requires request | Public with citation | Fully open |
| **Beginner relevance** | 10% | Elite/pro only | Mixed | Some beginners | Mostly beginners |

### 4.2 Compatibility score formula

```text
score = Σ (criterion_weight * criterion_score) / 3
```

Scores range from 0.0 to 1.0.

---

## 5. Step-by-Step Gathering & Evaluation Plan

### Task 1: Create a dataset registry sheet

**Files:**
- Create: `docs/superpowers/plans/event-dataset-registry.md`

- [ ] **Step 1: Initialize the registry**

Create a table with these columns:

```markdown
| Dataset | Source URL | License | Size | Format | Badminton? | Pose? | Labels? | FPS | Beginners? | Status |
```

- [ ] **Step 2: Populate known entries**

Add all datasets from Section 3 with whatever URL/license information is already known from `public_datasets_for_badminton_event_classifier.md`.

---

### Task 2: Locate and confirm access for each dataset

**Files:**
- Modify: `docs/superpowers/plans/event-dataset-registry.md`

- [ ] **Step 3: Search for official dataset pages**

For each dataset, find the canonical source (paper, GitHub, Kaggle, or project page). Record the URL and access instructions.

- [ ] **Step 4: Check license and usage terms**

Record whether the dataset is:

```text
- fully open
- open with citation
- request required
- restricted / unavailable
```

- [ ] **Step 5: Record estimated download size and format**

Note video resolution, number of clips/sequences, and file format (MP4, AVI, CSV, NPZ, etc.).

---

### Task 3: Download a representative subset of each accessible dataset

**Files:**
- Create directory: `data/event-dataset-samples/`
- Create script: `scripts/download_dataset_samples.py`

- [ ] **Step 6: Write a small download helper**

For Kaggle or direct-URL datasets, create a Python helper that downloads a small sample (e.g., 5–10 clips or one split) for inspection.

- [ ] **Step 7: Download Tier 1 samples**

Download samples from:

```text
- Video-Based Frame-Level Badminton Player Movement Dataset
- VideoBadminton
- BadmintonGRF
```

- [ ] **Step 8: Download Tier 2 samples**

Download samples from:

```text
- Motion Data for Badminton Shots
- ShuttleSet / FineBadminton (if video is available)
```

- [ ] **Step 9: Download Tier 3 samples**

Download at least one sample from:

```text
- Fencing Footwork Dataset
- THETIS Tennis Dataset
- MPOSE2021
```

---

### Task 4: Extract lower-body pose landmarks from sample videos

**Files:**
- Create script: `scripts/extract_mediapipe_lower_body.py`
- Output directory: `data/event-dataset-samples/pose/`

- [ ] **Step 10: Install or verify MediaPipe in the project virtual environment**

Run:

```bash
python -m pip install mediapipe
```

- [ ] **Step 11: Write MediaPipe extraction script**

The script should:

```text
1. Read each sample video.
2. Run MediaPipe Pose Landmarker.
3. Extract only the 10 lower-body landmarks.
4. Save per-frame landmark coordinates + visibility to CSV or NPZ.
5. Record fps and total frame count.
```

- [ ] **Step 12: Run extraction on all downloaded video samples**

Save outputs to:

```text
data/event-dataset-samples/pose/<dataset_name>/<clip_id>.csv
```

---

### Task 5: Inspect label mappability

**Files:**
- Create: `docs/superpowers/plans/event-label-mapping.md`
- Modify: `docs/superpowers/plans/event-dataset-registry.md`

- [ ] **Step 13: Map existing labels to target labels**

For each dataset, create a mapping table:

```markdown
| Dataset Label | Target Label | Effort to Convert |
|---|---|---|
```

Example:

```markdown
| VideoBadminton: smash | jump_landing / transition | Manual window extraction |
```

- [ ] **Step 14: Estimate relabeling effort**

Score each dataset 0–3 on event-label compatibility and record in the registry.

---

### Task 6: Score datasets against the rubric

**Files:**
- Create: `docs/superpowers/plans/dataset-compatibility-scores.csv`
- Create script: `scripts/score_datasets.py` (optional)

- [ ] **Step 15: Score each dataset**

Use the rubric from Section 4. Record scores in the CSV:

```csv
dataset,badminton_specificity,event_label_compat,lower_body_pose,temporal_resolution,accessibility,beginner_relevance,total
VideoBadminton,3,2,2,3,3,1,...
```

- [ ] **Step 16: Compute weighted totals**

Apply weights:

```text
badminton_specificity  0.25
event_label_compat     0.25
lower_body_pose        0.20
temporal_resolution    0.10
accessibility          0.10
beginner_relevance     0.10
```

---

### Task 7: Build a small annotated evaluation subset

**Files:**
- Create directory: `data/event-dataset-samples/labels/`
- Create: `data/event-dataset-samples/labels/evaluation_subset.csv`

- [ ] **Step 17: Select 5–10 representative clips per promising dataset**

Pick clips that contain clear examples of `idle`, `lunge`, `jump_landing`, or `direction_change`.

- [ ] **Step 18: Manually label event windows**

Use a simple label format:

```csv
clip_id,dataset,event_type,support_side,start_frame,contact_frame,max_load_frame,end_frame,notes
```

- [ ] **Step 19: Compute basic pose statistics on the labeled subset**

For each labeled window, compute:

```text
- average knee flexion
- ankle-foot alignment risk proxy
- hip displacement proxy
- landing asymmetry score
```

This confirms whether the dataset's pose quality supports the baseline algorithm.

---

### Task 8: Compare and select the primary dataset stack

**Files:**
- Create: `docs/superpowers/plans/dataset-recommendation.md`

- [ ] **Step 20: Rank datasets by total compatibility score**

List top 5 with justification.

- [ ] **Step 21: Recommend primary + secondary + pretraining stack**

Example structure:

```markdown
## Primary event data
1. Video-Based Frame-Level Badminton Player Movement Dataset
2. VideoBadminton

## Specialized support
3. BadmintonGRF

## Pretraining support
4. Fencing Footwork Dataset
5. MPOSE2021
```

- [ ] **Step 22: Document gaps that require own data**

Note any missing target labels, beginner footage, or camera views.

---

### Task 9: Document final data pipeline recommendation

**Files:**
- Modify: `docs/superpowers/plans/dataset-recommendation.md`

- [ ] **Step 23: Define the recommended data pipeline**

```text
1. Start with primary badminton video datasets.
2. Run MediaPipe to extract 10 lower-body landmarks.
3. Manually label 300–500 clips into target event classes.
4. Augment with BadmintonGRF for jump landing / impact windows.
5. Optionally pretrain on Fencing Footwork + MPOSE2021.
6. Fine-tune on own beginner badminton footage.
```

- [ ] **Step 24: Define the training sample format**

```text
Input:  30–60 frames × 10 landmarks × 3 coordinates
Output: event_type label
```

---

## 6. Decision Process

At the end of Task 8, the final decision should be made using:

1. **Quantitative score** from the rubric.
2. **Qualitative fit** for live badminton: camera view, athlete level, movement realism.
3. **Relabeling effort** required to reach 300–500 labeled clips.
4. **License feasibility** for commercial or research use.

If two datasets score similarly, prefer the one with:

- More direct badminton footage.
- Higher frame rate.
- Clearer path to target labels.
- Less restrictive license.

---

## 7. Risks & Assumptions

| Risk | Mitigation |
|---|---|
| No single public dataset has exact labels | Plan for 300–500 manual labels |
| Most public data is elite/match footage | Add own beginner recordings |
| Camera views differ from live app setup | Document views and test robustness |
| MediaPipe may fail on some videos | Record pose quality and skip low-quality clips |
| Licenses may restrict usage | Verify terms before downloading full datasets |

---

## 8. Spec Coverage Check

This plan addresses:

- ✅ Verification of the updated baseline parameters and algorithm
- ✅ Evaluation of publicly available datasets
- ✅ Compatibility scoring against live badminton event detection needs
- ✅ Selection of the most accurate/compatible dataset stack
- ✅ Gap identification where own data is still required

No Python files are modified in this plan. Implementation of extraction scripts and labeling tools is deferred to a later plan.
