# Project Overview & Dependencies

## What This Project Does
Real-time lower-body pose detection from a webcam using MediaPipe Pose + OpenCV, with research-backed biomechanical thresholds for classifying badminton footwork injury risk (safe vs. dangerous mechanics).

## Tech Stack (Constrained)
- **Python 3.x** — only stdlib modules required at runtime beyond the two main dependencies.
- **Dependencies**: `mediapipe==0.10.21`, `opencv-python>=4,<5` — pinned versions where applicable to ensure consistent MediaPipe landmark index mapping.
- **Optional (future)**: `three.js` via npm for 3D visualization (`injury-sim.html`).

## Package Layout
```
badminton-foot-injury/
├── src/badminton_risk/               # Python package
│   ├── baseline_risk.py              # Core geometry + risk scoring
│   ├── injury_risk.py                # Profile-based risk model
│   ├── motion_gate.py                # Moving/standing gate
│   ├── risk_overlay.py               # OpenCV HUD renderer
│   ├── video_risk_analyzer.py        # Offline/webcam analyzer CLI
│   └── webcam_leg_pose_detector.py   # Live webcam pose detector
├── tests/                            # Unit tests (mirror src modules)
├── scripts/                          # Batch runners
├── data/                             # Input videos, datasets, outputs
├── web/                              # HTML/JS sandbox
├── docs/                             # Docs and specs
├── .mindmodel/                       # Constraints and plans
└── .mindmodel/plans/injury.md        # Biomechanical thresholds (authoritative reference)
```

## Dependency Import Order
- `mediapipe` is imported **after** environment variable suppression (`TF_CPP_MIN_LOG_LEVEL`, `GLOG_minloglevel`).
- OpenCV is imported before MediaPipe (MediaPipe depends on it).
- All imports use explicit relative or absolute paths — no wildcard imports.
