import math

from baseline_risk import angle_at, clamp, distance, midpoint


def test_clamp_inside_range():
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_below_range():
    assert clamp(-0.1, 0.0, 1.0) == 0.0


def test_clamp_above_range():
    assert clamp(1.2, 0.0, 1.0) == 1.0


def test_distance_3d():
    a = (0.0, 0.0, 0.0)
    b = (3.0, 4.0, 0.0)
    assert distance(a, b) == 5.0


def test_midpoint():
    a = (0.0, 0.0, 0.0)
    b = (2.0, 4.0, 6.0)
    assert midpoint(a, b) == (1.0, 2.0, 3.0)


def test_angle_at_right_angle():
    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (0.0, 1.0, 0.0)
    assert math.isclose(angle_at(a, b, c), 90.0, abs_tol=1e-6)
