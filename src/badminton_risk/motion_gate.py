"""Simple moving/standing motion gate based on hip-center displacement.

The gate buffers the most recent ``window_frames`` hip-center observations and
classifies the subject as ``_LABEL_MOVING`` when the horizontal (x-z)
displacement between the oldest and newest observation in the window exceeds a
ratio of the leg length, or ``_LABEL_STANDING`` otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Point = tuple[float, float, float]

_DEFAULT_WINDOW_SECONDS = 0.3
_DEFAULT_FPS = 30.0
_DEFAULT_THRESHOLD_RATIO = 0.05
# When ``exit_ratio`` is not provided, standing is declared below this
# fraction of the entry threshold (hysteresis band).
_DEFAULT_EXIT_FRACTION = 0.7
_LABEL_MOVING = "moving"
_LABEL_STANDING = "standing"


@dataclass
class MotionGate:
    """Stateful gate that classifies motion from a stream of hip centers.

    Points are ``(x, y, z)`` tuples where ``x`` is the medial-lateral axis,
    ``y`` is the vertical axis, and ``z`` is the anterior-posterior axis.
    Classification uses only the horizontal (x-z) displacement between the
    oldest and newest observations in the rolling window.

    Args:
        window_seconds: Length of the rolling classification window in seconds.
            Must be positive.
        fps: Frames per second of the input stream. Must be positive.
        threshold_ratio: Displacement threshold expressed as a fraction of
            ``leg_length``. Must be non-negative.
        history: Public internal buffer of the most recent ``window_frames``
            hip-center observations. Usually left as the default empty list.
    """

    window_seconds: float = _DEFAULT_WINDOW_SECONDS
    fps: float = _DEFAULT_FPS
    threshold_ratio: float = _DEFAULT_THRESHOLD_RATIO
    # Lower threshold (fraction of leg length) for declaring standing once
    # moving. ``None`` falls back to ``_DEFAULT_EXIT_FRACTION`` of the entry
    # threshold, creating a hysteresis band so hovering displacements do not
    # flicker the label.
    exit_ratio: float | None = None
    # Minimum number of consecutive disagreeing frames before the label flips
    # (debounce). ``0`` switches immediately (legacy behavior).
    min_consecutive_frames: int = 0
    history: list[Point] = field(default_factory=list)
    _state: str = field(default=_LABEL_STANDING, init=False, repr=False)
    _consecutive: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Validate constructor arguments."""
        assert self.window_seconds > 0, "window_seconds must be positive"
        assert self.fps > 0, "fps must be positive"
        assert self.threshold_ratio >= 0, "threshold_ratio must be non-negative"
        assert self.min_consecutive_frames >= 0, "min_consecutive_frames must be non-negative"
        if self.exit_ratio is not None:
            assert self.exit_ratio >= 0, "exit_ratio must be non-negative"

    @property
    def window_frames(self) -> int:
        """Number of frames that span ``window_seconds`` at ``fps``."""
        return max(1, int(round(self.window_seconds * self.fps)))

    def update(self, hip_center: Point, leg_length: float) -> str:
        """Append a new hip center and return the current classification.

        The internal history buffer is trimmed to the most recent
        ``window_frames`` observations before classification.

        Args:
            hip_center: A 3-D point ``(x, y, z)`` representing the hip center.
            leg_length: Positive leg length, in the same units as the point
                coordinates.

        Returns:
            ``_LABEL_MOVING`` or ``_LABEL_STANDING``.

        Raises:
            AssertionError: If ``hip_center`` is not a 3-tuple or if
                ``leg_length`` is not positive.
        """
        assert len(hip_center) == 3, "hip_center must be a 3-D coordinate"
        assert leg_length > 0, "leg_length must be positive"

        self.history.append(hip_center)
        if len(self.history) > self.window_frames:
            self.history.pop(0)
        return self.classify(leg_length)

    def classify(self, leg_length: float) -> str:
        """Classify motion from the current rolling window.

        Motion is reported when the horizontal displacement between the oldest
        and newest observations in the trimmed buffer exceeds
        ``leg_length * threshold_ratio``.

        Args:
            leg_length: Positive leg length, in the same units as the point
                coordinates.

        Returns:
            ``_LABEL_MOVING`` if the buffered horizontal displacement exceeds
            the threshold; otherwise ``_LABEL_STANDING``. Returns
            ``_LABEL_STANDING`` when fewer than ``window_frames`` observations
            are available.

        Raises:
            AssertionError: If ``leg_length`` is not positive.
        """
        assert leg_length > 0, "leg_length must be positive"

        if len(self.history) < self.window_frames:
            return _LABEL_STANDING
        start = self.history[0]
        end = self.history[-1]
        displacement = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[2] - start[2]) ** 2
        )
        entry_threshold = leg_length * self.threshold_ratio
        exit_threshold = leg_length * (
            self.exit_ratio
            if self.exit_ratio is not None
            else self.threshold_ratio * _DEFAULT_EXIT_FRACTION
        )
        if self._state == _LABEL_MOVING:
            want = _LABEL_MOVING if displacement > exit_threshold else _LABEL_STANDING
        else:
            want = _LABEL_MOVING if displacement > entry_threshold else _LABEL_STANDING

        if want == self._state:
            self._consecutive = 0
            return self._state
        self._consecutive += 1
        if self._consecutive >= self.min_consecutive_frames:
            self._state = want
            self._consecutive = 0
        return self._state


def classify_frame(
    gate: MotionGate,
    hip_centers: list[Point],
    leg_length: float = 1.0,
) -> str:
    """Feed a sequence of hip centers through ``gate`` and return the last label.

    Args:
        gate: A ``MotionGate`` instance to update.
        hip_centers: A list of 3-D hip-center points.
        leg_length: Positive leg length, in the same units as the point
            coordinates. Defaults to ``1.0``.

    Returns:
        The classification produced for the final frame, or ``_LABEL_STANDING``
        if ``hip_centers`` is empty.

    Raises:
        AssertionError: If any point in ``hip_centers`` is not a 3-tuple or if
            ``leg_length`` is not positive.
    """
    result = _LABEL_STANDING
    for hc in hip_centers:
        result = gate.update(hc, leg_length)
    return result
