# Docs

Tool-agnostic documentation for `badminton-risk`. Historical plans/specs for
already-implemented features were removed on cleanup — recover them from git
history if needed.

## Layout

- `constraints/` — durable project knowledge (moved from `.mindmodel/constraints/`):
  architecture, biomechanics, project overview, testing, naming, types, error
  handling, documentation, research conventions.
- `usage/` — how to run the tools (offline/webcam analyzer).

## Notes

- Risk thresholds are implemented in `src/badminton_risk/`; the old
  `.mindmodel/plans/injury.md` reference doc was deleted as historical.
- The 3D sandbox at `web/badminton_injury_sandbox_v2.html` documents the risk
  model interactively; it is the descendant of the old `injury-sim.html`.
