"""Unit tests for the motion gate in ``motion_gate.py``."""

import pytest

from badminton_risk.motion_gate import MotionGate, classify_frame


def test_classify_frame_standing():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    hip_centers = [(0.0, 1.0, 0.0)] * 20
    assert classify_frame(gate, hip_centers) == "standing"


def test_classify_frame_moving():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    # Exactly window_frames (9) points split across two positions.
    hip_centers = [(0.0, 1.0, 0.0)] * 5 + [(0.3, 1.0, 0.0)] * 4
    assert classify_frame(gate, hip_centers) == "moving"


def test_classify_frame_not_enough_history():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    hip_centers = [(0.0, 1.0, 0.0), (0.3, 1.0, 0.0)]
    assert classify_frame(gate, hip_centers) == "standing"


def test_window_frames_rounds_correctly():
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    assert gate.window_frames == 9


def test_sliding_window_clears_after_movement_stops():
    gate = MotionGate(window_seconds=0.3, fps=30.0)

    # Fill the window with still frames.
    for _ in range(9):
        gate.update((0.0, 1.0, 0.0), 1.0)

    # Burst of movement; expect "moving" to be reported at some point.
    moving_results = []
    for _ in range(5):
        moving_results.append(gate.update((0.3, 1.0, 0.0), 1.0))
    assert "moving" in moving_results

    # Return to stillness; once the window slides past the movement the label
    # should return to "standing".
    still_results = []
    for _ in range(9):
        still_results.append(gate.update((0.3, 1.0, 0.0), 1.0))
    assert still_results[-1] == "standing"


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


def test_classify_frame_scales_threshold_with_leg_length():
    leg_length = 1.7
    threshold = leg_length * 0.05  # 0.085

    # Displacement just below the scaled threshold -> standing.
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    below = [(0.0, 1.0, 0.0)] * 9 + [(threshold - 0.001, 1.0, 0.0)]
    assert classify_frame(gate, below, leg_length=leg_length) == "standing"

    # Displacement just above the scaled threshold -> moving.
    gate = MotionGate(window_seconds=0.3, fps=30.0)
    above = [(0.0, 1.0, 0.0)] * 9 + [(threshold + 0.001, 1.0, 0.0)]
    assert classify_frame(gate, above, leg_length=leg_length) == "moving"
