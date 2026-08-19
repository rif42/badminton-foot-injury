# Type System Constraints

## Annotations
- **All public functions and classes** must have full type annotations on signatures (parameters + return type).
- Use `from __future__ import annotations` at module top to avoid forward-reference issues.
- Import types explicitly (`Optional`, `Tuple`, `Dict`, `List`, `Type`) rather than relying on `typing`.

## Specific Type Patterns
```python
# Configuration objects are dataclasses with typed fields:
@dataclass
class PoseDetectorConfig:
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    model_complexity: int = 0

# Landmark extraction returns a structured dict of tuples, never raw numpy arrays to consumers:
Dict[str, Tuple[float, float, float, float]]  # {joint_name: (x, y, z, visibility)}

# Functions that may fail should return Optional or raise typed exceptions:
Optional[NormalizedLandmarkList]  # Returns None when no pose is detected
```

## Type Hierarchy
- Keep types minimal — prefer `Tuple[float, float, ...]` over custom dataclass unless the structure needs named access.
- Use `Any` only for runtime-type-polymorphic cases (e.g., workaround for MediaPipe internal module paths). Suppress with comments like `# pragma: no cover`.
