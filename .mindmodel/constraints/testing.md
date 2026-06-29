# Testing Constraints

## Test Framework
- Use Python's standard library `unittest` — no external test frameworks required.
- Each module gets a corresponding `_test_*.py` file co-located in the same directory.
- Run with: `python -m unittest <module>_test.py`.

## Test Strategy
- **Mock all external dependencies**: OpenCV (`cv2.VideoCapture`, `cv2.imshow`, `cv2.waitKey`, etc.) and MediaPipe (`mp.solutions.pose.Pose`) must be mocked — never use real camera or model in tests.
- Build a `_make_mock_landmarks()` helper that returns a mock object with 33 landmarks (matching MediaPipe Pose's output shape).

## Test Coverage Areas
1. **Initialization**: Default values and custom config applied correctly to internal state.
2. **Frame processing**: Correct BGR→RGB conversion, return value handling for both success and failure cases.
3. **Drawing pipeline**: All expected landmark points and connections are drawn; low-visibility landmarks are skipped silently; missing indices don't raise exceptions.
4. **Landmark extraction helper**: Returns correct dict format and omits absent keys without error.
5. **Context manager**: `close()` is called exactly once on exit, even if an exception occurs.
6. **Main loop scenarios**: Camera failure (immediate exit), custom camera index, successful frame loop, debug mode output, no-pose fallback display, retry behavior on consecutive failures, max-retry exhaustion.

## Test Structure Rules
- One assertion per test function — tests must be atomic and independently runnable.
- Mock setup uses `@patch.object` decorators for class-level mocking or `with patch(...)` blocks for instance-level.
- Helper functions (`_make_mock_landmarks()`) are module-private (leading underscore) but duplicated in test files as needed — this avoids coupling the production code's public API to internal structure.
