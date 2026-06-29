# Error Handling Constraints

## Resource Lifecycle
- **Context managers required** for any resource that needs cleanup: `PoseDetector.__enter__`/`__exit__`, camera capture release via `finally`.
- Never rely on `del` or garbage collection; explicit close/release is mandatory.

## Failure Modes (in priority order)
1. **Hard failure, exit immediately**: Camera cannot open — print error, return non-zero exit code, do not attempt detection loop.
2. **Retry with backoff**: Frame read fails transiently — retry up to `MAX_FRAME_RETRIES` times (default 3), wait `FRAME_RETRY_DELAY_SECONDS` between attempts. After exhausting retries, exit gracefully.
3. **Graceful degradation**: No pose detected in a frame — continue loop showing the raw flipped frame; do not abort the capture session.

## External Dependency Failures
- MediaPipe model loading errors must be caught and logged before any user-visible operation. Do not crash on import-time configuration issues.
- OpenCV display failures (`cv2.imshow`/`waitKey`) are non-fatal — log warning, continue loop.

## Logging Rules
- Runtime warnings (frame failure, no pose): `print()` to stdout is acceptable for a CLI tool; no need for structured logging at this stage.
- Import-time errors: use `os.environ` guards to silence verbose TF/MediaPipe logs before importing (`TF_CPP_MIN_LOG_LEVEL=2`).
