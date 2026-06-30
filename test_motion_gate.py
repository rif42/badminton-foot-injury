"""Unit tests for the motion gate in ``motion_gate.py``."""

import pytest

from motion_gate import MotionGate, classify_frame


def test_classify_frame_standing():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    hip_centers = [(0.0, 1.0, 0.0)] * 20
    assert classify_frame(gate, hip_centers) == "standing"


def test_classify_frame_moving():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    hip_centers = [(0.0, 1.0, 0.0)] * 5 + [(0.3, 1.0, 0.0)] * 15
    assert classify_frame(gate, hip_centers) == "moving"


def test_classify_frame_not_enough_history():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    hip_centers = [(0.0, 1.0, 0.0), (0.3, 1.0, 0.0)]
    assert classify_frame(gate, hip_centers) == "standing"


def test_window_frames_rounds_correctly():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    assert gate.window_frames == 9


def test_motion_gate_rejects_non_positive_window_seconds():
    with pytest.raises(AssertionError):
        MotionGate(window_seconds=0.0, fps=30.0)
    with pytest.raises(AssertionError):
        MotionGate(window_seconds=-0.1, fps=30.0)


def test_motion_gate_rejects_non_positive_fps():
    with pytest.raises(AssertionError):
        MotionGate(window_seconds=0.3, fps=0.0)
    with pytest.raises(AssertionError):
        MotionGate(window_seconds=0.3, fps=-30.0)


def test_update_rejects_non_3d_point():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    with pytest.raises(AssertionError):
        gate.update((0.0, 1.0), 1.0)


def test_update_rejects_non_positive_leg_length():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    with pytest.raises(AssertionError):
        gate.update((0.0, 1.0, 0.0), 0.0)
    with pytest.raises(AssertionError):
        gate.update((0.0, 1.0, 0.0), -1.0)


def test_classify_rejects_non_positive_leg_length():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    gate.history = [(0.0, 1.0, 0.0), (0.3, 1.0, 0.0)]
    with pytest.raises(AssertionError):
        gate.classify(0.0)


def test_classify_frame_rejects_non_3d_point():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    with pytest.raises(AssertionError):
        classify_frame(gate, [(0.0, 1.0)])
