# Architecture Constraints

## Module Organization
- Each functional area lives in its own top-level `.py` module (e.g., `webcam_leg_pose_detector.py`).
- Modules are **importable standalone** — no module should fail to import if external services (camera, MediaPipe model) are unavailable.
- A dedicated research/injury-scoring subdirectory (`research_badminton_injury_params/`) holds parameter definitions and findings, separate from runtime code.

## Component Boundaries
- **Pose capture layer**: Owns webcam I/O, frame conversion, MediaPipe initialization, and landmark drawing. Must not contain injury-risk scoring logic.
- **Injury-scoring layer** (future): Consumes extracted landmarks from the capture layer; owns biomechanical thresholds and classification logic.
- Boundary: The pose detector exposes `get_lower_body_landmarks(landmarks) -> Dict[str, Tuple[float,...]]` so consumers can inject their own scoring without modifying the capture code.

## Data Flow Rules
- Pipeline is **unidirectional**: raw frame → RGB conversion → MediaPipe Pose → landmark dict → injury metrics → classification.
- No circular dependencies between modules.
- Each stage returns structured data (dataclass, dict, or tuple) — never raw OpenCV frames to scoring logic.

## Resource Ownership
- Every resource that can leak has a `close()` method and is used within `with` blocks.
- Resources: MediaPipe Pose model, camera capture handle, OpenCV windows.
