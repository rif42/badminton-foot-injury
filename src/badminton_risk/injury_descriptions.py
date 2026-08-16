"""Injury descriptions for the badminton lower-body risk analyzer.

Each core risk component maps to a specific injury pattern, a short plain-language
description, and a prevention cue. The analyzer uses these to annotate the video
and to build a critical-event log.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InjuryDescription:
    """Human-readable metadata for one risk component."""

    component: str
    name: str
    short_description: str
    description: str
    prevention: str


INJURY_DESCRIPTIONS: tuple[InjuryDescription, ...] = (
    InjuryDescription(
        component="knee_stiffness_risk",
        name="Patellar tendon overload",
        short_description="Knee too straight during loading",
        description=(
            "The knee is too straight during landing or a lunge, increasing "
            "mechanical leverage and strain on the patellar tendon and meniscus."
        ),
        prevention=(
            "Keep the knee flexed and aligned over the toe during loading; "
            "avoid hyper-extending the knee."
        ),
    ),
    InjuryDescription(
        component="ankle_foot_alignment_risk",
        name="Ankle sprain (ATFL tear)",
        short_description="Foot not aligned with momentum",
        description=(
            "The foot is not aligned with the body's momentum, creating lateral "
            "shearing forces on the ankle ligaments."
        ),
        prevention=(
            "Land with the foot pointing along the direction of movement and "
            "keep the knee centered over the foot."
        ),
    ),
    InjuryDescription(
        component="ankle_roll_risk",
        name="Ankle roll (inversion/eversion)",
        short_description="Foot rolled sideways at the ankle",
        description=(
            "The foot is rolled sideways at the ankle (inversion/eversion), "
            "the classic lateral ankle-sprain mechanism. A severe roll can "
            "injure the ATFL and other lateral ankle ligaments."
        ),
        prevention=(
            "Land with the foot flat and pointed along the direction of "
            "movement; strengthen the peroneal muscles and practice stable "
            "single-leg landings."
        ),
    ),
    InjuryDescription(
        component="hip_displacement_proxy",
        name="Hip instability",
        short_description="Pelvis displaced from base of support",
        description=(
            "The pelvis is displaced far from the base of support, transferring "
            "excessive stress to the knee and ankle."
        ),
        prevention=(
            "Keep the hips centered over the feet and strengthen hip stabilizers "
            "for better control."
        ),
    ),
    InjuryDescription(
        component="landing_asymmetry_score",
        name="Uneven lower-body loading",
        short_description="Left-right imbalance on landing",
        description=(
            "Left-right imbalance in knee flexion, pelvis height, or ankle "
            "position produces uneven shock absorption."
        ),
        prevention=(
            "Train balanced, symmetric landing mechanics and distribute weight "
            "equally through both legs."
        ),
    ),
)

_COMPONENT_KEYS = tuple(d.component for d in INJURY_DESCRIPTIONS)

DEFAULT_COMPONENT_THRESHOLD = 0.5


def describe_critical_risks(
    score_dict: dict[str, float],
    component_threshold: float = DEFAULT_COMPONENT_THRESHOLD,
) -> list[dict[str, object]]:
    """Return injury descriptions for the risk components that are elevated.

    Args:
        score_dict: Mapping from component keys to scores. Missing component keys
            are treated as ``0.0``.
        component_threshold: Only components with a score at or above this value
            are considered contributing. If no component reaches the threshold,
            the single highest component is returned so the caller always has a
            primary injury to report.

    Returns:
        A list of injury dictionaries, sorted by descending score.
    """
    contributing = [
        {
            "component": injury.component,
            "name": injury.name,
            "short_description": injury.short_description,
            "description": injury.description,
            "prevention": injury.prevention,
            "score": score_dict.get(injury.component, 0.0),
        }
        for injury in INJURY_DESCRIPTIONS
        if score_dict.get(injury.component, 0.0) >= component_threshold
    ]
    contributing.sort(key=lambda item: item["score"], reverse=True)

    if not contributing:
        top_component = max(
            _COMPONENT_KEYS, key=lambda key: score_dict.get(key, 0.0)
        )
        injury = next(d for d in INJURY_DESCRIPTIONS if d.component == top_component)
        contributing = [
            {
                "component": injury.component,
                "name": injury.name,
                "short_description": injury.short_description,
                "description": injury.description,
                "prevention": injury.prevention,
                "score": score_dict.get(injury.component, 0.0),
            }
        ]

    return contributing
