## Offline Video Risk Analyzer

Analyze a recorded badminton clip for lower-body injury-risk patterns.

### Run

```bash
python video_risk_analyzer.py input.mp4 --output-csv report.csv
```

### Optional annotated video

```bash
python video_risk_analyzer.py input.mp4 --output-csv report.csv --output-video annotated.mp4
```

### Output CSV columns

| Column | Description |
|---|---|
| frame | Frame index |
| time_sec | Timestamp in seconds |
| is_moving | True if motion gate detected movement |
| knee_stiffness_risk | 0–1 stiffness risk |
| ankle_foot_alignment_risk | 0–1 alignment risk |
| hip_displacement_proxy | 0–1 hip displacement |
| landing_asymmetry_score | 0–1 asymmetry |
| core_risk | Combined 0–1 risk |
| status | acceptable / caution / risky |

### Scope note

This is a simplified offline MVP. It does not classify lunge/jump/direction-change events, gather public datasets, or track fatigue.
