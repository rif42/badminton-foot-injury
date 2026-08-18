# Badminton Lower-Body Injury-Risk Analyzer

> ⚠️ **Educational / demo tool only — not a medical diagnostic device.** Consult a qualified clinician for any injury concerns.

A Python pipeline that scores badminton lower-body injury risk from video pose landmarks. It runs **MediaPipe Pose** on each frame, derives biomechanical cues from the lower-body landmarks, and outputs a per-frame risk report (CSV), an annotated video, and optionally a JSON log of critical detections.

The same engine powers two front-ends:

- **CLI / offline analyzer** (`src/badminton_risk/video_risk_analyzer.py`) — process a recorded video file or a live webcam feed.
- **Streamlit web app** (`streamlit_app.py`) — upload a short clip in the browser, get the annotated video + report, and download both.

A standalone client-side three.js sandbox lives in `web/` (no Python needed).

## Quick algorithm breakdown

```
video / webcam frames
        │
        ▼
MediaPipe Pose (Tasks API) ── 33 body landmarks per frame
        │
        ▼
Landmark smoothing ── 3-tap median pre-filter + One-Euro low-pass
        │                (kills jitter/teleports; slow-mo friendly)
        ▼
Lower-body landmarks ── hips (23/24), knees (25/26), ankles (27/28),
        │                heels (29/30), foot indices (31/32)
        ▼
Motion gate (MotionGate) ── moving vs. standing, from hip-center
        │                horizontal displacement over a ~0.3 s rolling
        │                window (as a fraction of leg length), with
        │                hysteresis + 2-frame debounce
        ▼
Risk scoring (core_risk_score) ── per-frame 0–1 sub-scores, averaged
        │                bilaterally and combined with fixed weights:
        │                • knee stiffness        (0.20)
        │                • ankle-foot alignment  (0.20)
        │                • ankle roll            (0.10)
        │                • hip displacement      (0.15)
        │                • landing asymmetry     (0.35)
        │                Ankle roll = foot-plane vs. shank 3-D angle,
        │                gated to planted feet, baseline-calibrated
        │                during standing frames; ≥45° deviation fires
        │                an ankle_roll_event (needs 2 consecutive frames)
        ▼
Temporal smoothing ── EMA over score components + status hysteresis
        │                (acceptable / caution / risky)
        ▼
Outputs ── per-frame CSV, annotated MP4 (HUD overlay + 3 s critical
           popup), JSON log of critical detections
```

The webcam/live path (`injury_risk.py`) uses a slightly different profile-based model with 5 parameters (hip-trajectory deviation, knee flexion, foot alignment, landing pitch, ankle roll) mapped through piecewise risk curves with interaction bonuses and idle-state calibration.

**CSV output columns:** `frame`, `time_sec`, `status` (acceptable/caution/risky), `core_risk`, `knee_stiffness_risk`, `ankle_foot_alignment_risk`, `ankle_roll_risk`, `ankle_roll_angle_deg`, `ankle_roll_event`, `hip_displacement_proxy`, `landing_asymmetry_score`, plus `injury_names` / `injury_descriptions` / `injury_preventions`.

## Installation

Requires **Python 3.11** (`>=3.11,<3.12`).

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
#    Windows:    .venv\Scripts\activate
#    macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` installs `mediapipe`, `streamlit`, `pandas`, and `opencv-contrib-python-headless`.

The MediaPipe pose model (`pose_landmarker_full.task`) is **auto-downloaded on first run** to your system temp directory (`%TEMP%\badminton_risk_models\` on Windows) — you don't need to fetch it manually, but the first run requires an internet connection.

## Run the script on a single video file

The package lives under `src/`, so set `PYTHONPATH=src` (the `scripts/*.bat` wrappers do this automatically).

```bash
# Basic: writes risk_report.csv
PYTHONPATH=src .venv/Scripts/python.exe -m badminton_risk.video_risk_analyzer input.mp4

# Full: CSV report + annotated video
PYTHONPATH=src .venv/Scripts/python.exe -m badminton_risk.video_risk_analyzer input.mp4 \
    --output-csv report.csv \
    --output-video annotated.mp4
```

### Windows batch wrapper

```bat
scripts\run_analyzer.bat input.mp4 --output-csv report.csv --output-video annotated.mp4
```

### Options

| Flag | Description |
|---|---|
| `input_video` | Path to any video OpenCV can read (`.mp4`, `.avi`, `.mov`, …). Default CSV is `risk_report.csv`. |
| `--output-csv PATH` | Write the per-frame risk report to `PATH`. |
| `--output-video PATH` | Write an annotated MP4 (HUD overlay, risk label, critical popup). |
| `--output-log PATH` | Write a JSON log of every critical (`risky`) detection with injury name, description, and prevention tip. |
| `--show-preview` | Show a live preview window while processing (press `q` to quit early). |
| `--webcam [INDEX]` | Use a live webcam feed instead of a file (mutually exclusive with `input_video`). Preview is always on; `q` quits. |

Example with everything:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m badminton_risk.video_risk_analyzer match.mp4 \
    --output-csv match_report.csv \
    --output-video match_annotated.mp4 \
    --output-log match_critical.json
```

> Short clips are recommended: processing is per-frame and the first run also downloads the model.

## Host on Streamlit

The web UI (`streamlit_app.py`) wraps the same `analyze_video` pipeline: upload a clip, analyze it, and view/download the annotated video and CSV report.

### Run locally

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m streamlit run streamlit_app.py
```

Then open the printed URL (default `http://localhost:8501`).

### Deploy to Streamlit Community Cloud

1. **Push the repo to GitHub** (make sure `requirements.txt` and `streamlit_app.py` are in the root).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** (or the "Create app" flow in your Streamlit dashboard) and sign in with GitHub.
3. **Deploy a new app**, select your repository and branch, and set the main file to `streamlit_app.py`.
4. Streamlit reads `requirements.txt` from the root, installs the dependencies on its cloud runners, and launches the app.

Notes for cloud deployment:

- No build step or secrets are needed — the app is self-contained.
- The pose model is downloaded to a temp directory **on first analysis**, so the very first upload may take a little longer.
- Keep uploads short (`< 30 s`) since each analysis runs on a shared cloud instance; the UI already recommends this.
