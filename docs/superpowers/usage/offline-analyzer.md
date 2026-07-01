## Offline Video Risk Analyzer

Analyze a recorded badminton clip for lower-body injury-risk patterns.

### Prerequisites

- Python 3
- MediaPipe
- OpenCV
- NumPy
- `baseline_risk.py` and `motion_gate.py` must be present in the same directory (they are imported by `video_risk_analyzer.py`)

### Input format

The analyzer accepts any video format that OpenCV `VideoCapture` can read, such as `.mp4`, `.avi`, or `.mov`.

### Run

```bash
python video_risk_analyzer.py input.mp4
```

The default output CSV filename is `risk_report.csv`. To specify a different file:

```bash
python video_risk_analyzer.py input.mp4 --output-csv report.csv
```

### Webcam mode

Analyze a live webcam feed instead of a file:

```bash
python video_risk_analyzer.py --webcam
```

Use a specific camera index:

```bash
python video_risk_analyzer.py --webcam 1
```

Webcam mode shows a live preview window and writes no files by default. Press `q` to quit. To record the session, provide the usual output flags:

```bash
python video_risk_analyzer.py --webcam --output-csv live_report.csv --output-video live_annotated.mp4
```

### Optional flags

- `--output-video annotated.mp4` — write an annotated side-by-side video.
- `--show-preview` — display a live preview window while processing a file (preview is always on in webcam mode).

### Output CSV columns

| Column | Description |
|---|---|
| frame | Frame index |
| time_sec | Timestamp in seconds |
| is_moving | `True`/`False` string indicating whether the motion gate detected movement |
| knee_stiffness_risk | 0–1 stiffness risk |
| ankle_foot_alignment_risk | 0–1 alignment risk |
| hip_displacement_proxy | 0–1 hip displacement |
| landing_asymmetry_score | 0–1 asymmetry |
| core_risk | Combined 0–1 risk |
| status | acceptable / caution / risky |

### Scope note

This is a simplified offline MVP. It does not classify lunge/jump/direction-change events, gather public datasets, or track fatigue.
