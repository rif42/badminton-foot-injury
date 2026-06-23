# Webcam Leg Pose Detector — Design

## Objective
Create a Python script that reads the user's webcam, runs a MediaPipe pose-detection model, and displays a live window with the lower-body skeleton overlay (hips, thighs, knees, ankles, heels, toes). No injury-risk logic is included in this version; the structure should make it easy to add later.

## Context
This is part of the `badminton-foot-injury` project. The file `injury.md` already defines biomechanical thresholds that will eventually be computed from these landmarks. The webcam script is the first step toward capturing the required landmark data.

## Selected Approach
**Option B: Small `PoseDetector` class.**
- A reusable class wraps MediaPipe Pose initialization, frame processing, and landmark drawing.
- `main()` handles OpenCV camera I/O.
- This keeps the landmark-extraction logic isolated so injury-risk calculations from `injury.md` can be added later without rewriting the capture loop.

## Components

### `PoseDetector`
Responsibilities:
- Initialize MediaPipe Pose with sensible defaults (lower model complexity for real-time speed).
- Accept optional configuration for camera index, detection confidence, and tracking confidence.
- Convert an incoming OpenCV BGR frame to the RGB format MediaPipe expects.
- Run pose detection and return raw `results.pose_landmarks`.
- Draw selected lower-body landmarks and connections onto the frame.

### `main()`
Responsibilities:
- Open the default webcam (`cv2.VideoCapture`).
- Flip frames horizontally so the display acts like a mirror.
- Loop until the user presses `q`:
  1. Read a frame.
  2. Detect pose via `PoseDetector.process`.
  3. Draw landmarks via `PoseDetector.draw_landmarks`.
  4. Show the annotated frame.
- Release the camera and close OpenCV windows on exit.

## Data Flow
1. Webcam provides a BGR frame.
2. `PoseDetector.process` converts to RGB and runs MediaPipe Pose.
3. If landmarks are found, `PoseDetector.draw_landmarks` overlays the lower-body skeleton.
4. OpenCV displays the annotated frame.

## Lower-Body Landmarks of Interest
MediaPipe Pose indices used for the lower body:
- Hips: 23 (left), 24 (right)
- Knees: 25 (left), 26 (right)
- Ankles: 27 (left), 28 (right)
- Heels: 29 (left), 30 (right)
- Foot index / big toe: 31 (left), 32 (right)

The drawing routine will highlight these points and the connecting edges.

## Error Handling
- Camera fails to open: print an error message and exit immediately.
- Frame read fails: print a warning, allow a small number of retries, then exit.
- No pose detected in a frame: show the original frame unchanged.

## Dependencies
- `opencv-python`
- `mediapipe`

## Out of Scope
- Injury-risk scoring or threshold checks (to be added in a follow-up).
- Recording video or writing landmark data to disk.
- Multi-camera support.
- Calibration or world-coordinate conversion.

## Future Hook
`PoseDetector` will expose a helper such as `get_lower_body_landmarks(results)` that returns a dictionary of `{joint_name: (x, y, z, visibility)}` so the injury-risk module can consume it directly.
