# Webcam Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `video_risk_analyzer.py` so it can analyze a live webcam feed in addition to pre-recorded video files.

**Architecture:** Add a `--webcam` CLI flag that branches capture initialization once. The existing per-frame processing loop is reused. Webcam mode defaults to live preview only and writes no files unless `--output-csv` or `--output-video` are explicitly provided.

**Tech Stack:** Python 3, OpenCV, MediaPipe, pytest.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `video_risk_analyzer.py` | Main module: CLI, capture initialization, analysis loop. |
| `test_video_risk_analyzer.py` | Unit tests for CLI validation, file mode, and webcam mode. |
| `docs/superpowers/usage/offline-analyzer.md` | Usage documentation for both file and webcam modes. |

---

## Task 1: Update CLI to support `--webcam`

**Files:**
- Modify: `video_risk_analyzer.py:427-461`
- Test: `test_video_risk_analyzer.py`

- [ ] **Step 1: Write the failing CLI-validation tests**

```python
def test_main_requires_input_or_webcam():
    """Calling main with no source raises a usage error."""
    with pytest.raises(SystemExit):
        main([])


def test_main_rejects_both_sources():
    """Cannot specify both a file and a webcam."""
    with pytest.raises(SystemExit):
        main(["fake.mp4", "--webcam"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest test_video_risk_analyzer.py::test_main_requires_input_or_webcam test_video_risk_analyzer.py::test_main_rejects_both_sources -v
```

Expected: FAIL because the CLI still requires `input_video`.

- [ ] **Step 3: Update the argument parser and dispatch logic**

Replace the existing `main()` body with:

```python
def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the offline and webcam risk analyzer.

    Args:
        argv: Optional argument list. If ``None``, ``sys.argv`` is used.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Offline and live webcam badminton lower-body risk analyzer."
    )
    parser.add_argument(
        "input_video",
        nargs="?",
        default=None,
        help="Path to input video (not used with --webcam).",
    )
    parser.add_argument(
        "--webcam",
        nargs="?",
        const=0,
        type=int,
        default=None,
        help="Use a webcam. Optionally specify the camera index (default 0).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to output CSV. Defaults to risk_report.csv for file mode; omitted in webcam mode.",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Path to optional annotated output video.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show live preview window in file mode. Preview is always shown in webcam mode.",
    )
    args = parser.parse_args(argv)

    if args.webcam is None and args.input_video is None:
        parser.error("Either input_video or --webcam is required.")
    if args.webcam is not None and args.input_video is not None:
        parser.error("Cannot specify both input_video and --webcam.")

    output_csv = args.output_csv
    if output_csv is None and args.webcam is None:
        output_csv = DEFAULT_OUTPUT_CSV

    analyze_video(
        args.input_video,
        output_csv,
        args.output_video,
        args.show_preview,
        webcam_index=args.webcam,
    )
    if output_csv:
        print(f"Wrote CSV report to {output_csv}")
    if args.output_video:
        print(f"Wrote annotated video to {args.output_video}")
    return 0
```

- [ ] **Step 4: Run tests to verify CLI validation passes**

```bash
pytest test_video_risk_analyzer.py::test_main_requires_input_or_webcam test_video_risk_analyzer.py::test_main_rejects_both_sources -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add video_risk_analyzer.py test_video_risk_analyzer.py
git commit -m "feat(cli): add --webcam flag and source validation"
```

---

## Task 2: Branch capture initialization for webcam vs file

**Files:**
- Modify: `video_risk_analyzer.py:291-335`
- Test: `test_video_risk_analyzer.py`

- [ ] **Step 1: Write the failing webcam-capture test**

```python
def test_analyze_video_webcam_preview_writes_no_files(tmp_path, monkeypatch):
    """Webcam mode with no output flags shows preview and writes nothing."""
    monkeypatch.chdir(tmp_path)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_pose = MagicMock()
    mock_pose.process.return_value.pose_landmarks = None
    mock_pose.close = MagicMock()

    with (
        patch("cv2.VideoCapture") as mock_cap,
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=ord("q")),
    ):
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, fake_frame), (False, None)]
        mock_cap_instance.get.side_effect = lambda key: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        }[key]
        mock_cap.return_value = mock_cap_instance

        analyze_video(
            None,
            None,
            None,
            show_preview=False,
            pose_detector=mock_pose,
            webcam_index=0,
        )

    mock_cap.assert_called_once_with(0)
    mock_imshow.assert_called_once()
    assert not (tmp_path / "risk_report.csv").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest test_video_risk_analyzer.py::test_analyze_video_webcam_preview_writes_no_files -v
```

Expected: FAIL because `analyze_video` does not accept `webcam_index`.

- [ ] **Step 3: Update `analyze_video` signature and capture opening**

Change the function signature and the capture-opening block:

```python
def analyze_video(
    input_path: str | None,
    output_csv: str | None,
    output_video: str | None,
    show_preview: bool = False,
    pose_detector: PoseDetector | None = None,
    webcam_index: int | None = None,
) -> None:
    """Analyze a video file or webcam stream and write a per-frame risk report.

    Args:
        input_path: Path to the input video file. Required in file mode; must
            be ``None`` in webcam mode.
        output_csv: Path where the CSV report will be written. In file mode,
            defaults to ``risk_report.csv`` if omitted. In webcam mode, no CSV
            is written unless this argument is provided.
        output_video: Optional path where an annotated output video will be
            written. If ``None``, no video is produced.
        show_preview: If ``True``, display a live preview window. Always
            ``True`` in webcam mode.
        pose_detector: Optional pose detector to use. The object must satisfy
            the ``PoseDetector`` protocol (``process`` / ``close``). If
            ``None``, a default MediaPipe Pose detector is created.
        webcam_index: If not ``None``, open this camera index instead of a file.

    Raises:
        RuntimeError: If the input source or output video writer cannot be
            opened.
    """
    if webcam_index is not None:
        cap = cv2.VideoCapture(webcam_index)
        source_name = f"webcam at index {webcam_index}"
        is_webcam = True
    else:
        cap = cv2.VideoCapture(input_path)
        source_name = f"video: {input_path}"
        is_webcam = False

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source_name}")
```

- [ ] **Step 4: Update FPS/dimensions initialization for webcam**

Immediately after the open check, set:

```python
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_webcam:
        if width <= 0 or height <= 0:
            width, height = 640, 480
        display_window = True
    else:
        display_window = show_preview
```

- [ ] **Step 5: Run the capture test**

```bash
pytest test_video_risk_analyzer.py::test_analyze_video_webcam_preview_writes_no_files -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add video_risk_analyzer.py test_video_risk_analyzer.py
git commit -m "feat(analyzer): open webcam capture when --webcam is given"
```

---

## Task 3: Skip file outputs by default in webcam mode

**Files:**
- Modify: `video_risk_analyzer.py:335-424`
- Test: `test_video_risk_analyzer.py`

- [ ] **Step 1: Write the failing CSV-recording test**

```python
def test_analyze_video_webcam_writes_csv_when_requested(tmp_path, monkeypatch):
    """Webcam mode records CSV when --output-csv is explicitly provided."""
    monkeypatch.chdir(tmp_path)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_pose = MagicMock()
    mock_pose.process.return_value.pose_landmarks = None
    mock_pose.close = MagicMock()

    output_csv = tmp_path / "webcam_report.csv"

    with (
        patch("cv2.VideoCapture") as mock_cap,
        patch("cv2.imshow"),
        patch("cv2.waitKey", return_value=ord("q")),
    ):
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, fake_frame), (False, None)]
        mock_cap_instance.get.side_effect = lambda key: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        }[key]
        mock_cap.return_value = mock_cap_instance

        analyze_video(
            None,
            str(output_csv),
            None,
            show_preview=False,
            pose_detector=mock_pose,
            webcam_index=0,
        )

    assert output_csv.exists()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest test_video_risk_analyzer.py::test_analyze_video_webcam_writes_csv_when_requested -v
```

Expected: FAIL because the analyzer tries to write CSV even when `output_csv` is `None`.

- [ ] **Step 3: Make result collection and CSV writing conditional**

Change the results list initialization:

```python
    pose = pose_detector if pose_detector is not None else _create_pose_detector()

    gate = MotionGate(window_seconds=_MOTION_GATE_WINDOW_SECONDS, fps=fps)
    results: list[dict[str, Any]] = [] if output_csv is not None else None
    frame_idx = 0
```

In the loop, append only when recording:

```python
            if results is not None:
                results.append(row)
```

At the end of the function, write the CSV only when requested:

```python
    if output_csv is not None:
        rows = build_csv_rows(results)
        if rows:
            with open(output_csv, "w", newline="") as f:
                writer_csv = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer_csv.writeheader()
                writer_csv.writerows(rows)
```

- [ ] **Step 4: Run both webcam tests**

```bash
pytest test_video_risk_analyzer.py::test_analyze_video_webcam_preview_writes_no_files test_video_risk_analyzer.py::test_analyze_video_webcam_writes_csv_when_requested -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add video_risk_analyzer.py test_video_risk_analyzer.py
git commit -m "feat(analyzer): make CSV/video output optional in webcam mode"
```

---

## Task 4: Force preview window in webcam mode

**Files:**
- Modify: `video_risk_analyzer.py:381-416`

- [ ] **Step 1: Update the display condition**

In the loop, replace the existing `if writer is not None or show_preview:` block guard with:

```python
            if writer is not None or display_window:
```

And replace the preview check:

```python
                if display_window:
                    cv2.imshow("Risk Analyzer", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
```

- [ ] **Step 2: Add a test for the main webcam path**

```python
def test_main_runs_webcam_mode(capsys, tmp_path, monkeypatch):
    """main(['--webcam']) opens camera 0 and shows preview without writing files."""
    monkeypatch.chdir(tmp_path)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_pose = MagicMock()
    mock_pose.process.return_value.pose_landmarks = None
    mock_pose.close = MagicMock()

    with (
        patch("cv2.VideoCapture") as mock_cap,
        patch("video_risk_analyzer._create_pose_detector", return_value=mock_pose),
        patch("cv2.imshow"),
        patch("cv2.waitKey", return_value=ord("q")),
    ):
        mock_cap_instance = MagicMock()
        mock_cap_instance.read.side_effect = [(True, fake_frame), (False, None)]
        mock_cap_instance.get.side_effect = lambda key: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        }[key]
        mock_cap.return_value = mock_cap_instance

        rc = main(["--webcam"])

    assert rc == 0
    mock_cap.assert_called_once_with(0)
    captured = capsys.readouterr()
    assert "Wrote CSV" not in captured.out
```

- [ ] **Step 3: Run the test**

```bash
pytest test_video_risk_analyzer.py::test_main_runs_webcam_mode -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add video_risk_analyzer.py test_video_risk_analyzer.py
git commit -m "feat(analyzer): always show preview in webcam mode"
```

---

## Task 5: Update usage documentation

**Files:**
- Modify: `docs/superpowers/usage/offline-analyzer.md`

- [ ] **Step 1: Add a Webcam section after the Windows batch shortcut**

Insert before `### Optional flags`:

```markdown
### Webcam mode

Analyze a live webcam feed instead of a file:

```bash
python video_risk_analyzer.py --webcam
```

Use a specific camera index:

```bash
python video_risk_analyzer.py --webcam 1
```

Webcam mode shows a live preview window and writes no files by default. Press `q` to quit. To record the session, provide the usual output flags:

```bash
python video_risk_analyzer.py --webcam --output-csv live_report.csv --output-video live_annotated.mp4
```
```

- [ ] **Step 2: Update Optional flags description**

Change:

```markdown
- `--show-preview` — display a live preview window while processing.
```

to:

```markdown
- `--show-preview` — display a live preview window while processing a file (preview is always on in webcam mode).
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/usage/offline-analyzer.md
git commit -m "docs(usage): document --webcam mode"
```

---

## Task 6: Run the full test suite

**Files:**
- None (verification only)

- [ ] **Step 1: Run all tests**

```bash
pytest test_video_risk_analyzer.py -v
```

Expected: All tests pass, including the new webcam tests and the existing file-mode tests.

- [ ] **Step 2: Commit if any fixes are needed**

If any existing tests broke due to signature changes, fix them and commit.

---

## Self-Review

1. **Spec coverage:**
   - `--webcam` CLI flag with optional index → Task 1.
   - Source validation → Task 1.
   - Webcam capture opening with fallback dimensions → Task 2.
   - Default live preview, optional outputs → Tasks 3 and 4.
   - Tests for CLI, capture, and output behavior → Tasks 1–4.
   - Documentation → Task 5.
2. **Placeholder scan:** All code blocks contain concrete code; no TBD/TODO/fill-in-later patterns.
3. **Type consistency:** `webcam_index: int | None = None` is used consistently in `analyze_video` and passed from `main` via keyword argument.
