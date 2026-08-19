"""Temporal landmark smoothing for jittery MediaPipe pose output.

The One-Euro filter (Casiez, Roussel & Vogel 2012) adapts its cutoff frequency
to the signal speed: slow, stable motion is heavily smoothed while fast motion
is tracked with minimal lag. This is ideal for slow-motion clips where the leg
moves slowly but per-frame landmark detection jumps around.

The ``LandmarkSmoother`` keeps one One-Euro filter per landmark per axis and
exposes:

- ``smooth_pose`` for the offline ``LowerBodyPose`` (pixel coordinates);
- ``smooth_normalized`` for the live ``(x, y, z, visibility)`` landmark dicts.

Call ``reset()`` whenever the pose is lost for a few frames so re-acquisition
starts fresh instead of lagging behind the real position.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .baseline_risk import LowerBodyPose, Point

# Names of the lower-body landmarks managed by the smoother (MediaPipe 23-32).
LOWER_BODY_NAMES: tuple[str, ...] = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

_MIN_DT = 1e-3


@dataclass(frozen=True)
class OneEuroParams:
    """Tuning parameters for the One-Euro filter.

    Attributes:
        min_cutoff: Cutoff frequency (Hz) when the signal is stationary.
            Lower values smooth more aggressively.
        beta: Speed coefficient. Higher values raise the cutoff faster during
            fast motion, reducing lag on real movement.
        d_cutoff: Cutoff frequency (Hz) applied to the derivative signal.
        median_window: Odd window size of the median pre-filter applied to
            the raw input before One-Euro (1 disables it). A 3-tap median
            removes single-frame landmark teleports (tracking glitches) while
            genuine sustained motion passes with ~1 frame of lag.
    """

    min_cutoff: float = 0.8
    beta: float = 0.02
    d_cutoff: float = 0.5
    median_window: int = 3


class _LowPassAxis:
    """Median pre-filter + One-Euro low-pass for a single scalar signal.

    A 3-tap median over the raw inputs removes single-frame landmark teleports
    (tracking glitches); the median output is then fed through the One-Euro
    filter which adaptively smooths the remaining noise while preserving
    genuine motion.
    """

    def __init__(
        self,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
        median_window: int,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.median_window = max(1, median_window | 1)  # odd, >= 1
        self._x: float | None = None
        self._dx: float = 0.0
        self._raw_history: deque[float] = deque(maxlen=self.median_window)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, dt: float) -> float:
        dt = max(dt, _MIN_DT)
        if self.median_window > 1:
            self._raw_history.append(x)
            if len(self._raw_history) == self.median_window:
                x = sorted(self._raw_history)[len(self._raw_history) // 2]
        if self._x is None:
            self._x = x
            self._dx = 0.0
            return x
        dx = (x - self._x) / dt
        dx_alpha = self._alpha(self.d_cutoff, dt)
        self._dx = dx_alpha * dx + (1.0 - dx_alpha) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        alpha = self._alpha(cutoff, dt)
        self._x = alpha * x + (1.0 - alpha) * self._x
        return self._x

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._raw_history.clear()


class LandmarkSmoother:
    """One-Euro smoothing for named 3-D landmarks."""

    def __init__(
        self,
        params: OneEuroParams | None = None,
        names: tuple[str, ...] | None = None,
    ) -> None:
        self.params = params if params is not None else OneEuroParams()
        self.names = tuple(names) if names is not None else LOWER_BODY_NAMES
        self._filters: dict[str, list[_LowPassAxis]] = {
            name: [self._make_axis() for _ in range(3)] for name in self.names
        }

    def _make_axis(self) -> _LowPassAxis:
        p = self.params
        return _LowPassAxis(
            p.min_cutoff,
            p.beta,
            p.d_cutoff,
            p.median_window,
        )

    def reset(self) -> None:
        """Clear all filter state (call after the pose is lost)."""
        for axes in self._filters.values():
            for axis in axes:
                axis.reset()

    def smooth_point(self, name: str, point: Point, dt: float) -> Point:
        """Return the smoothed ``(x, y, z)`` for one named landmark."""
        axes = self._filters[name]
        return (
            axes[0](point[0], dt),
            axes[1](point[1], dt),
            axes[2](point[2], dt),
        )

    def smooth_pose(self, pose: LowerBodyPose, dt: float) -> LowerBodyPose:
        """Return a new ``LowerBodyPose`` with smoothed landmarks.

        Args:
            pose: The raw pose in pixel coordinates.
            dt: Seconds elapsed since the previous frame.

        Returns:
            A ``LowerBodyPose`` whose landmarks are the smoothed values.
        """
        kwargs = {
            name: self.smooth_point(name, getattr(pose, name), dt)
            for name in LOWER_BODY_NAMES
        }
        return LowerBodyPose(**kwargs)

    def smooth_normalized(
        self,
        landmarks: dict[str, tuple[float, float, float, float]],
        dt: float,
    ) -> dict[str, tuple[float, float, float, float]]:
        """Return smoothed ``(x, y, z, visibility)`` dicts for the live path.

        Visibility is preserved unchanged; unknown names pass through.
        """
        out: dict[str, tuple[float, float, float, float]] = {}
        for name, (x, y, z, vis) in landmarks.items():
            if name in self._filters:
                sx, sy, sz = self.smooth_point(name, (x, y, z), dt)
                out[name] = (sx, sy, sz, vis)
            else:
                out[name] = (x, y, z, vis)
        return out
