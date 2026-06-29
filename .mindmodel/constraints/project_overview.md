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
├── webcam_leg_pose_detector.py      # Pose capture + landmark extraction
├── test_webcam_leg_pose_detector.py # Unit tests (mocked)
├── injury.md                        # Biomechanical thresholds (authoritative reference)
├── research_badminton_injury_params/  # Research findings & parameters
│   ├── research_plan.md             # Research plan
│   └── findings_*.md                # Per-parameter evidence
├── docs/superpowers/specs/          # Design specs
└── .mindmodel/constraints/         # This directory
```

## Dependency Import Order
- `mediapipe` is imported **after** environment variable suppression (`TF_CPP_MIN_LOG_LEVEL`, `GLOG_minloglevel`).
- OpenCV is imported before MediaPipe (MediaPipe depends on it).
- All imports use explicit relative or absolute paths — no wildcard imports.
