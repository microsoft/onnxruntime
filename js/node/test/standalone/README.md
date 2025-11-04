# Process Exit Tests

These tests verify that the ORT Node.js binding handles various process exit scenarios without crashing, specifically addressing the mutex crash issue reported in [#24579](https://github.com/microsoft/onnxruntime/issues/24579).

## What This Tests

- **Normal process exit** - Verifies clean shutdown without mutex crashes
- **`process.exit()` calls** - Tests the primary crash scenario that was fixed
- **Uncaught exceptions** - Ensures crashes don't occur during unexpected exits
- **Session cleanup** - Tests both explicit `session.release()` and automatic cleanup
- **Stability** - Multiple runs to ensure consistent behavior

## How It Works

Each test runs in a separate Node.js process to isolate the test environment. Tests use command-line flags to control behavior:

- `--process-exit`: Triggers `process.exit(0)`
- `--throw-exception`: Throws an uncaught exception
- `--release`: Calls `session.release()` before exit

## Expected Result

All tests should pass without `mutex lock failed` or `std::__1::system_error` messages in stderr.
