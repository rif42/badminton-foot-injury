"""Unit tests for injury description helpers."""

from __future__ import annotations

import pytest

from badminton_risk.injury_descriptions import (
    INJURY_DESCRIPTIONS,
    describe_critical_risks,
)


def test_describe_critical_risks_returns_components_above_threshold():
    score = {
        "knee_stiffness_risk": 0.2,
        "ankle_foot_alignment_risk": 0.8,
        "hip_displacement_proxy": 0.3,
        "landing_asymmetry_score": 0.6,
        "core_risk": 0.55,
    }
    injuries = describe_critical_risks(score)
    assert len(injuries) == 2
    assert injuries[0]["component"] == "ankle_foot_alignment_risk"
    assert injuries[1]["component"] == "landing_asymmetry_score"
    assert injuries[0]["score"] == pytest.approx(0.8)


def test_describe_critical_risks_falls_back_to_top_component():
    score = {
        "knee_stiffness_risk": 0.1,
        "ankle_foot_alignment_risk": 0.2,
        "hip_displacement_proxy": 0.15,
        "landing_asymmetry_score": 0.05,
        "core_risk": 0.2,
    }
    injuries = describe_critical_risks(score)
    assert len(injuries) == 1
    assert injuries[0]["component"] == "ankle_foot_alignment_risk"
    assert "name" in injuries[0]
    assert "description" in injuries[0]
    assert "prevention" in injuries[0]


def test_describe_critical_risks_returns_empty_for_missing_components():
    score = {
        "knee_stiffness_risk": 0.8,
        "core_risk": 0.8,
    }
    # Only known component is high; missing components are treated as 0.0.
    injuries = describe_critical_risks(score)
    assert injuries[0]["component"] == "knee_stiffness_risk"


def test_all_injuries_have_required_fields():
    for injury in INJURY_DESCRIPTIONS:
        assert injury.component
        assert injury.name
        assert injury.short_description
        assert injury.description
        assert injury.prevention
