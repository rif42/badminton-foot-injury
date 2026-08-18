# badminton-risk — Badminton Lower-Body Injury-Risk Analyzer

Educational/demo tool (not a medical device) that scores badminton lower-body injury risk
from pose landmarks. Python 3.11 pipeline (MediaPipe Pose + OpenCV) plus a Streamlit UI
and a client-side three.js demo.

## Project

- Stack: Python `>=3.11,<3.12`, MediaPipe Tasks Pose, OpenCV, pandas, Streamlit.
- Python package: `src/badminton_risk` (src-layout, package name `badminton-risk`).
- CLI entry point: `src/badminton_risk/video_risk_analyzer.py` (`main()` → `analyze_video()`).
- UIs: `streamlit_app.py` (browser upload/analyze, wraps `analyze_video`); `web/index.html` (client-side 2D analyzer demo) and `web/badminton_injury_sandbox_v2.html` (3D risk-model sandbox), independent of the Python pipeline.
- Docs: tool-agnostic tree under `docs/` — `docs/constraints/` (durable project knowledge), `docs/usage/` (how to run), `docs/README.md` (index). Historical plans/specs of implemented features were deleted; recover from git history.
- Model: auto-downloaded on first run to `%TEMP%\badminton_risk_models\pose_landmarker_full.task`.

## Commands

All commands use the venv `.venv\Scripts\python.exe`. The package lives under `src/`,
so imports need `PYTHONPATH=src` (the `scripts/*.bat` wrappers set it already).

- Run CLI on a video: `PYTHONPATH=src .venv/Scripts/python.exe -m badminton_risk.video_risk_analyzer <video.mp4> --output-csv out.csv --output-video out.mp4`
- Webcam mode: add `--webcam [index]` (mutually exclusive with input video); `--output-log out.json` writes a critical-detection JSON log; `--show-preview` opens a window.
- Streamlit UI: `PYTHONPATH=src .venv/Scripts/python.exe -m streamlit run streamlit_app.py`
- Tests: `.venv/Scripts/python.exe -m pytest` (config in `pyproject.toml`: `pythonpath=["src"]`, `testpaths=["tests"]`, `-q`). Verified green.
- Windows wrappers: `scripts/run_analyzer.bat` (single video), `scripts/run_analyzer_on_dataset.bat` (all `data/dataset/*.mp4` → `data/results/`).
- Lint: code carries `noqa` comments for ruff rules (E402, S310, BLE001) and a `.ruff_cache/` exists, but ruff is NOT installed in the venv.

## Architecture

- `video_risk_analyzer.py` — orchestration: CLI args, video read/write, MediaPipe Tasks setup, per-frame loop, CSV/video/log output. Has adapters (`_LegacyLandmarkList`, `_PoseResultAdapter`, `_PoseLandmarkerAdapter`) exposing the legacy `process()` API over the Tasks detector.
- `baseline_risk.py` — 3-D geometry helpers (`Point = tuple[float, float, float]`), ankle-foot alignment, `core_risk_score`.
- `injury_risk.py` — live pose → 4 biomechanical parameters → weighted `RiskProfile` model with temporal smoothing (`RiskResult`).
- `injury_descriptions.py` — per-component injury metadata (name, description, prevention cue); feeds CSV columns and critical-event log.
- `motion_gate.py` — moving/standing gate on hip-center displacement over a rolling window.
- `risk_overlay.py` — OpenCV HUD drawing (BGR color constants, panel).
- `webcam_leg_pose_detector.py` — reusable `PoseDetector` webcam wrapper; no risk scoring.

CSV output columns: `frame`, `time_sec`, `status` (safe/caution/risky), `core_risk`, `knee_stiffness_risk`, `ankle_foot_alignment_risk`, `ankle_roll_risk`, `ankle_roll_angle_deg`, `ankle_roll_event`, `hip_displacement_proxy`, `landing_asymmetry_score`, plus `injury_names` / `injury_descriptions` / `injury_preventions`. Ankle roll (inversion/eversion) is computed from the foot-plane-vs-shank angle, gated to planted feet, and baseline-calibrated during standing frames; `ankle_roll_event` fires at ≥45° roll deviation.

## Conventions

- Every module starts with `from __future__ import annotations` and a docstring.
- Env-var logging suppression (`TF_CPP_MIN_LOG_LEVEL`, `GLOG_minloglevel`) and `absl.logging` silencing MUST run before `import mediapipe` (hence `# noqa: E402`).
- Keep the legacy MediaPipe wrapper/adapter pattern when touching detector integration; downstream code only knows `process()` / `close()`.
- Type style is mixed in old modules (`typing.Dict/List/Optional/Tuple` in `injury_risk.py`, `webcam_leg_pose_detector.py`); new code uses `X | None` and builtin generics.
- Risk thresholds/weights live as module-level `_UPPER_CASE` constants with comment units (e.g. knee deviation as fraction of leg length).
- Educational disclaimer: never present output as medical diagnosis (Streamlit caption already does this).

## Notes

- `.slim/`, `.worktrees/`, caches (`.pytest_cache`, `.ruff_cache`) are tool state — gitignored, leave them out of commits.
- `data/results/` and generated `*_report.csv` / `*_annotated.mp4` outputs are gitignored; `scripts/run_analyzer_on_dataset.bat` regenerates `data/results/` on demand.
- Web demos `web/` (three.js, `bun.lock`) are standalone; no build step wired into the Python workflow.

## Anti-jitter smoothing (slow-mo friendly)

- `src/badminton_risk/smoothing.py`: `LandmarkSmoother` = 3-tap median pre-filter (kills single-frame landmark teleports) + One-Euro low-pass (adaptively filters noise while tracking real motion with ~2-frame lag). Defaults `OneEuroParams(0.8, 0.02, 0.5, median_window=3)`; tune these to trade smoothness vs responsiveness.
- Applied right after landmark extraction in both paths (offline `analyze_video`, live `webcam_leg_pose_detector`), reset on pose loss; the smoothed landmarks feed scoring AND the drawn skeleton.
- Score-level: offline EMA (`_SCORE_SMOOTHING_ALPHA=0.5`) over risk components; status with hysteresis (rise 0.25/0.45, clear 0.20/0.40 via `_status_with_hysteresis`); `ankle_roll_event` needs 2 consecutive frames; live `RiskModel` status has `_STATUS_HYSTERESIS_FRACTION=0.15` margins.
- Motion gate: `MotionGate(exit_ratio=0.035, min_consecutive_frames=2)` debounces `is_moving`; offline `_extract_pose` skips frames with landmark visibility < 0.5.
