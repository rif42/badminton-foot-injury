# Webcam Support for `video_risk_analyzer.py`

## Goal

Allow the existing offline `video_risk_analyzer.py` to run on a live webcam feed, not just pre-recorded video files.

## Design

### CLI changes

- Add a `--webcam` argument that accepts an optional camera index:
  - `--webcam` alone uses camera index `0`.
  - `--webcam 1` uses camera index `1`.
- Make the positional `input_video` argument optional (`nargs="?"`).
- Enforce that exactly one of `input_video` or `--webcam` is provided; otherwise `argparse` raises a usage error.

### Source handling

Inside `analyze_video`, branch once based on whether a webcam index is provided:

- **File mode** (existing): open `cv2.VideoCapture(input_path)`, read `fps`, width, and height from the file metadata.
- **Webcam mode**: open `cv2.VideoCapture(camera_index)`. If the camera fails to open, raise a `RuntimeError`. Read width and height from the capture properties; fall back to `640x480` if they are `0`. Use the camera-reported FPS if available, otherwise default to `30.0`.

### Output behavior

- **Webcam mode defaults to live preview only**: the annotated feed is shown in an OpenCV window and runs until the user presses `q`.
- `--output-csv` and `--output-video` remain optional in webcam mode. If provided, the session is recorded exactly like a file analysis; if omitted, no files are written.
- `--show-preview` still works in file mode and is effectively always on in webcam mode.

### Loop changes

The existing per-frame loop is reused unchanged:

1. Read a frame.
2. Run MediaPipe Pose.
3. Extract lower-body landmarks.
4. Update the motion gate and compute risk when moving.
5. Draw the overlay.
6. Write to the optional video writer.
7. Show the preview window if needed.
8. Quit on `q`.

For webcam mode, frame data is only accumulated into the in-memory results list when `--output-csv` is provided, so long sessions do not grow memory unbounded.

### Error handling

- Camera open failure: raise `RuntimeError("Could not open webcam at index N")` and release the capture.
- Frame read failures in webcam mode: log a warning and continue; the existing `cap.read()` loop simply exits when no more frames are available.

### Testing

Update `test_video_risk_analyzer.py` to cover:

- CLI parsing: `--webcam`, `--webcam 2`, and missing input with no `--webcam` fail.
- `analyze_video` with a mocked `cv2.VideoCapture` in webcam mode opens the correct index and processes frames.
- Webcam mode with no output flags does not write CSV or video files.

## Scope

Only `video_risk_analyzer.py` and its unit tests are modified. No changes to `baseline_risk.py`, `injury_risk.py`, `motion_gate.py`, `risk_overlay.py`, or `webcam_leg_pose_detector.py`.
