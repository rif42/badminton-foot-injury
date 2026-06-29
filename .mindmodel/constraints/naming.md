# Naming Conventions

## Files
- Python source modules: `snake_case.py` (e.g., `webcam_leg_pose_detector.py`)
- Test modules: mirror source name with `_test_` suffix or co-located in same directory
- Constants and configuration data files: snake_case with `.md` extension (e.g., `findings_foot_alignment.md`)

## Python Identifiers
- **Classes**: `PascalCase` — `PoseDetector`, `PoseDetectorConfig`
- **Functions/methods**: `snake_case` — `process()`, `draw_landmarks()`, `get_lower_body_landmarks()`
- **Constants and public module-level variables**: `UPPER_SNAKE_CASE` — `LOWER_BODY_LANDMARKS`, `LOWER_BODY_CONNECTIONS`, `LANDMARK_COLOR`, `VISIBILITY_THRESHOLD`, `MAX_FRAME_RETRIES`

## Variable Names
- Single letters for simple values only (`i`, `k`). Otherwise descriptive.
- Coordinate tuples use positional clarity: `(x, y, z, visibility)` — never reorder.
- Boolean flags are lowercase single words or explicit names, not `flag`/`is`.
