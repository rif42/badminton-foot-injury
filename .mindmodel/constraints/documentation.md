# Documentation Constraints

## Docstring Style: Google Format
- All public modules, classes, and functions must have docstrings.
- Follow Google-style format with explicit sections: `Args:`, `Returns:`, `Raises:` (if applicable), `Note:`, `Example:`.

### Module-level example
```python
"""Module purpose and high-level description.

This module provides [what it does] for [who uses it]. It intentionally contains no [exclusion note]; landmark extraction is exposed so follow-up modules can consume it directly.
"""
```

### Class-level example
```python
class PoseDetector:
    """Wraps MediaPipe Pose initialization, frame processing, and lower-body drawing."""
```

### Function-level example
```python
def draw_landmarks(
    frame: np.ndarray,
    landmarks: NormalizedLandmarkList,
) -> np.ndarray:
    """Overlay lower-body landmarks and connections onto the frame.

    Note:
        This method mutates ``frame`` in place and returns the same object.
    """
```

## Exclusion Documentation
- Every module must explicitly state what it does **not** do (e.g., "Intentionally contains no injury-risk scoring logic").
- This prevents scope creep and documents the boundary for future feature additions.
