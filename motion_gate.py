"""Simple moving/standing motion gate based on hip-center displacement.

The gate buffers the most recent hip-center observations and classifies the
subject as ``"moving"`` when the horizontal (x-z) displacement over the
configured time window exceeds a ratio of the leg length, or ``"standing"``
otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

Point = Tuple[float, float, float]


@dataclass
class MotionGate:
    """Stateful gate that classifies motion from a stream of hip centers.

    Points are ``(x, y, z)`` tuples where ``x`` is the medial-lateral axis,
    ``y`` is the vertical axis, and ``z`` is the anterior-posterior axis.
    Classification uses only the horizontal (x-z) displacement between the
    oldest and newest buffered observations.

    Args:
        window_seconds: Minimum observation window in seconds. The gate needs
            at least ``window_frames`` observations before it can report
            ``"moving"``. Must be positive.
        fps: Frames per second of the input stream. Must be positive.
        threshold_ratio: Displacement threshold expressed as a fraction of
            ``leg_length``. Must be non-negative.
        history: Internal buffer of observed hip centers. Usually left as the
            default empty list.
    """

    window_seconds: float = 0.3
    fps: float = 30.0
    threshold_ratio: float = 0.05
    history: List[Point] = field(default_factory=list)

    def __post_init__(self):
        """Validate constructor arguments."""
        assert self.window_seconds > 0, "window_seconds must be positive"
        assert self.fps > 0, "fps must be positive"
        assert self.threshold_ratio >= 0, "threshold_ratio must be non-negative"

    @property
    def window_frames(self) -> int:
        """Number of frames that span ``window_seconds`` at ``fps``."""
        return max(1, int(round(self.window_seconds * self.fps)))

    def update(self, hip_center: Point, leg_length: float) -> str:
        """Append a new hip center and return the current classification.

        Args:
            hip_center: A 3-D point ``(x, y, z)`` representing the hip center.
            leg_length: Positive leg length, in the same units as the point
                coordinates.

        Returns:
            ``"moving"`` or ``"standing"``.

        Raises:
            AssertionError: If ``hip_center`` is not a 3-tuple or if
                ``leg_length`` is not positive.
        """
        assert len(hip_center) == 3, "hip_center must be a 3-D coordinate"
        assert leg_length > 0, "leg_length must be positive"

        self.history.append(hip_center)
        return self.classify(leg_length)

    def classify(self, leg_length: float) -> str:
        """Classify motion from the current history buffer.

        Motion is reported when the horizontal displacement between the oldest
        and newest buffered observations exceeds
        ``leg_length * threshold_ratio``.

        Args:
            leg_length: Positive leg length, in the same units as the point
                coordinates.

        Returns:
            ``"moving"`` if the buffered horizontal displacement exceeds the
            threshold; otherwise ``"standing"``. Returns ``"standing"`` when
            fewer than ``window_frames`` observations are available.

        Raises:
            AssertionError: If ``leg_length`` is not positive.
        """
        assert leg_length > 0, "leg_length must be positive"

        if len(self.history) < self.window_frames:
            return "standing"
        start = self.history[0]
        end = self.history[-1]
        displacement = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[2] - start[2]) ** 2
        )
        threshold = leg_length * self.threshold_ratio
        return "moving" if displacement > threshold else "standing"


def classify_frame(
    gate: MotionGate,
    hip_centers: List[Point],
    leg_length: float = 1.0,
) -> str:
    """Feed a sequence of hip centers through ``gate`` and return the last label.

    Args:
        gate: A ``MotionGate`` instance to update.
        hip_centers: A list of 3-D hip-center points.
        leg_length: Positive leg length, in the same units as the point
            coordinates. Defaults to ``1.0``.

    Returns:
        The classification produced for the final frame, or ``"standing"`` if
        ``hip_centers`` is empty.

    Raises:
        AssertionError: If any point in ``hip_centers`` is not a 3-tuple or if
            ``leg_length`` is not positive.
    """
    result = "standing"
    for hc in hip_centers:
        result = gate.update(hc, leg_length)
    return result
