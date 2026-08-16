"""Unit tests for the One-Euro landmark smoother (``smoothing.py``)."""

from __future__ import annotations

import random

import pytest

from badminton_risk.baseline_risk import LowerBodyPose
from badminton_risk.smoothing import LandmarkSmoother, OneEuroParams


def test_stationary_noise_is_filtered():
    random.seed(7)
    sm = LandmarkSmoother()
    raw = [100.0 + random.uniform(-10, 10) for _ in range(90)]
    out = [sm.smooth_point("left_ankle", (v, v, v), 1 / 30)[0] for v in raw]
    raw_jitter = sum(abs(raw[i] - raw[i - 1]) for i in range(1, len(raw))) / (
        len(raw) - 1
    )
    out_jitter = sum(abs(out[i] - out[i - 1]) for i in range(1, len(out))) / (
        len(out) - 1
    )
    assert out_jitter < raw_jitter / 1.5


def test_single_frame_teleport_is_rejected():
    # A single 300px tracking-glitch frame must not pass through the filter.
    sm = LandmarkSmoother()
    vals = [100.0] * 20 + [400.0] + [100.0] * 10
    out = [sm.smooth_point("left_ankle", (v, v, v), 1 / 30)[0] for v in vals]
    assert max(out) < 150.0


def test_genuine_sustained_step_is_tracked():
    # Real sustained motion (100 -> 400) must be tracked, not smoothed away.
    sm = LandmarkSmoother()
    vals = [100.0] * 30 + [400.0] * 30
    out = [sm.smooth_point("left_knee", (v, v, v), 1 / 30)[0] for v in vals]
    lag = next(i for i in range(30, 60) if out[i] > 100 + 0.9 * 300) - 30
    assert lag <= 3


def test_fast_roll_spike_survives_smoothing():
    # A genuine severe-roll deviation (0 -> 60 deg) must not be washed out:
    # the smoothed angle must cross the 45 deg event threshold within 3 frames.
    sm = LandmarkSmoother()
    vals = [0.0] * 20 + [60.0] * 10
    out = [sm.smooth_point("left_ankle", (v, v, v), 1 / 30)[0] for v in vals]
    cross = next(i for i in range(20, 30) if out[i] > 45.0) - 20
    assert cross <= 3


def test_median_window_can_be_disabled():
    sm = LandmarkSmoother(OneEuroParams(median_window=1))
    vals = [100.0] * 20 + [400.0] + [100.0] * 10
    out = [sm.smooth_point("left_ankle", (v, v, v), 1 / 30)[0] for v in vals]
    # Without the median pre-filter a big spike still moves the output.
    assert max(out) > 150.0


def test_smooth_pose_first_call_returns_raw():
    sm = LandmarkSmoother()
    pose = LowerBodyPose(
        left_hip=(0.0, 1.0, 0.0),
        right_hip=(0.1, 1.0, 0.0),
        left_knee=(0.0, 0.5, 0.0),
        right_knee=(0.1, 0.5, 0.0),
        left_ankle=(0.0, 0.0, 0.0),
        right_ankle=(0.1, 0.0, 0.0),
        left_heel=(0.0, 0.0, -0.1),
        right_heel=(0.1, 0.0, -0.1),
        left_foot_index=(0.0, 0.0, 0.1),
        right_foot_index=(0.1, 0.0, 0.1),
    )
    out = sm.smooth_pose(pose, 1 / 30)
    assert isinstance(out, LowerBodyPose)
    assert out.left_hip == pose.left_hip  # warm-up frame passes through


def test_reset_clears_state():
    sm = LandmarkSmoother()
    sm.smooth_point("left_ankle", (100.0, 100.0, 100.0), 1 / 30)
    sm.smooth_point("left_ankle", (200.0, 200.0, 200.0), 1 / 30)
    sm.reset()
    assert sm.smooth_point("left_ankle", (300.0, 300.0, 300.0), 1 / 30) == (
        300.0,
        300.0,
        300.0,
    )


def test_smooth_normalized_preserves_visibility():
    sm = LandmarkSmoother()
    landmarks = {
        "left_ankle": (0.5, 0.8, 0.0, 0.9),
        "unknown": (1.0, 1.0, 0.0, 1.0),
    }
    out = sm.smooth_normalized(landmarks, 1 / 30)
    assert out["left_ankle"][3] == pytest.approx(0.9)
    assert out["unknown"] == landmarks["unknown"]
