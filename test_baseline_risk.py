"""Unit tests for the 3-D geometry helpers in ``baseline_risk.py``."""

import math

import pytest

from baseline_risk import angle_at, clamp, distance, knee_flexion_angle, knee_stiffness_risk, midpoint


def test_clamp_inside_range():
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_below_range():
    assert clamp(-0.1, 0.0, 1.0) == 0.0


def test_clamp_above_range():
    assert clamp(1.2, 0.0, 1.0) == 1.0


def test_clamp_equal_bounds():
    assert clamp(7.5, 2.0, 2.0) == 2.0


def test_distance_3d():
    a = (0.0, 0.0, 0.0)
    b = (3.0, 4.0, 0.0)
    assert distance(a, b) == 5.0


def test_distance_along_z_axis():
    a = (0.0, 0.0, 0.0)
    b = (0.0, 0.0, 9.0)
    assert distance(a, b) == 9.0


def test_distance_negative_coordinates():
    a = (-1.0, -2.0, -3.0)
    b = (-4.0, -6.0, -3.0)
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


def test_angle_at_collinear_zero_degrees():
    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (2.0, 0.0, 0.0)
    assert math.isclose(angle_at(a, b, c), 0.0, abs_tol=1e-6)


def test_angle_at_collinear_180_degrees():
    a = (1.0, 0.0, 0.0)
    b = (0.0, 0.0, 0.0)
    c = (-1.0, 0.0, 0.0)
    assert math.isclose(angle_at(a, b, c), 180.0, abs_tol=1e-6)


def test_angle_at_collapsed_vertex_returns_zero():
    vertex = (0.0, 0.0, 0.0)
    other = (1.0, 0.0, 0.0)
    assert angle_at(other, vertex, vertex) == 0.0
    assert angle_at(vertex, vertex, other) == 0.0


def test_distance_rejects_non_3d_point():
    with pytest.raises(AssertionError):
        distance((0.0, 0.0), (1.0, 0.0, 0.0))


def test_midpoint_rejects_non_3d_point():
    with pytest.raises(AssertionError):
        midpoint((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_knee_flexion_right_angle():
    hip = (0.0, 1.0, 0.0)
    knee = (0.0, 0.0, 0.0)
    ankle = (1.0, 0.0, 0.0)
    assert math.isclose(knee_flexion_angle(hip, knee, ankle), 90.0, abs_tol=1e-6)


def test_knee_stiffness_risk_low():
    assert knee_stiffness_risk(120.0) == 0.0


def test_knee_stiffness_risk_at_threshold():
    assert knee_stiffness_risk(145.0) == 0.0


def test_knee_stiffness_risk_high():
    assert knee_stiffness_risk(180.0) == 1.0


def test_knee_stiffness_risk_mid():
    assert knee_stiffness_risk(160.0) == pytest.approx(0.429, abs=1e-3)
