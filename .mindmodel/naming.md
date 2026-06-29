# Naming Conventions

## Python Identifiers

| Element | Style | Example |
|---------|-------|---------|
| Variables, functions, classes | `snake_case` | `get_lower_body_landmarks`, `PoseDetector` |
| Module-level constants | `UPPER_SNAKE_CASE` | `LOWER_BODY_LANDMARKS`, `MAX_FRAME_RETRIES`, `LANDMARK_COLOR` |
| Type aliases | `PascalCase` | `NormalizedLandmarkList` (when referencing third-party types) |

## Landmark Naming Convention

### Physical Body Semantics
All landmark keys use **`left_*` / `right_*`** prefixes to refer to the physical side of the person's body, regardless of image orientation:

```python
"left_hip", "right_hip"
"left_knee",  "right_knee"
"left_ankle", "right_ankle"
"left_heel",  "right_heel"
"left_foot_index", "right_foot_index"
```

### MediaPipe Index Mapping
Landmark dictionaries map human-readable names to MediaPipe Pose landmark indices:

```python
LOWER_BODY_LANDMARKS: Dict[str, int] = {
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}
```

### Coordinate Convention
Extracted landmarks use the tuple format `(x, y, z, visibility)` where:
- `x`, `y` are normalized coordinates (0.0–1.0) from MediaPipe's image frame
- `z` is depth relative to camera (0.0 = close)
- `visibility` is confidence weight for this landmark

## Function Naming

### Action-Oriented Names
Functions describe what they do, not where:

```python
# Good — describes the action
def process(self, frame): ...
def draw_landmarks(frame, landmarks): ...
def get_lower_body_landmarks(landmarks): ...

# Avoid — location-dependent or vague
def run(self, frame): ...        # "run" is too generic
def update_frame(frame): ...     # unclear what's being updated
```

## Class Naming

### Noun Phrases with Purpose Suffixes
- `*Detector` — runs detection models
- `*Scorer` — computes scores/risks from data
- `*Config` — runtime configuration (always a dataclass)
- `*Result` — output container types

```python
PoseDetector        # wraps MediaPipe Pose model
InjuryScorer        # evaluates biomechanical thresholds
RiskAssessment      # contains classification result + metadata
```

## Constant Naming

### Descriptive UPPER_SNAKE_CASE
Constants are self-documenting and used at module level:

```python
# Good — immediately understandable
LOWER_BODY_LANDMARKS    # which landmarks to track
LOWER_BODY_CONNECTIONS  # which joints connect
LANDMARK_COLOR          # default drawing color (BGR tuple)
CONNECTION_THICKNESS    # line width for skeleton overlay
VISIBILITY_THRESHOLD    # cutoff below which a landmark is invisible

# Bad — abbreviations or unclear
LB_LANDMARKS            # too abbreviated
CONN_LIST               # vague, not self-documenting
```

## Threshold/Parameter Naming (injury.md)

Biomechanical parameters use descriptive names reflecting their physical meaning:

```python
# Parameter descriptions follow this pattern:
# <what it measures>_<context>, e.g.:
foot_to_trajectory_alignment_angle  # angular difference between CoM vector and foot vector
sagittal_knee_extension_angle       # internal Hip-Knee-Ankle angle
foot_strike_pitch_angle             # pitch of foot relative to ground at contact
trailing_foot_roll_angle            # rear foot plane angle during lunge
hip_rotation_opening_angle          # how much hip "opens" laterally (to be added)
```

---

## Documentation String Style

All public functions and classes use **Google-style docstrings**:

```python
def process(self, frame: np.ndarray) -> Optional[NormalizedLandmarkList]:
    """Run pose detection on a BGR frame and return raw landmarks if found.
    
    Args:
        frame: Input OpenCV BGR frame to analyze.
        
    Returns:
        NormalizedLandmarkList with detected lower-body keypoints, or None
        if no pose was detected in this frame.
    """
```

**Required sections**: `Args`, `Returns` (and `Raises` / `Notes` when applicable).

---

## File Organization

- Test files mirror their source: `test_{module_name}.py`
- Module imports use the module name directly, not file paths
- Third-party library names in docs/imports match PyPI/package conventions exactly
