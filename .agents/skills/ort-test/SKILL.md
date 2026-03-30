---
name: ort-test
description: Run ONNX Runtime tests. Use this skill when asked to run tests, debug test failures, or find and execute specific test cases in ONNX Runtime.
---

# Running ONNX Runtime Tests

ONNX Runtime uses **Google Test** for C++ and **unittest** (preferred) / **pytest** for Python.

## C++ tests

### Run all tests via CTest

```bash
# Linux/macOS
cd build/Linux/Release && ctest

# Windows
cd build\Windows\Release && ctest
```

The platform subdirectory under `build/` matches the OS (e.g., `Linux`, `Windows`, `Darwin`).

### Run a specific C++ test binary

The main test binary is `onnxruntime_test_all`. Use `--gtest_filter` to select tests:

```bash
# Run a specific test by name pattern
./build/Linux/Release/onnxruntime_test_all --gtest_filter="*TestName*"

# Windows
.\build\Windows\Release\onnxruntime_test_all.exe --gtest_filter="*TestName*"
```

### Run via build script

```bash
# Run all tests through the build script (skips update and build)
./build.sh --config Release --test
.\build.bat --config Release --test
```

## Python tests

Use `pytest` as the test runner:

```bash
# Run an entire test file
pytest onnxruntime/test/python/test_specific.py

# Run a specific test class or method
pytest onnxruntime/test/python/test_specific.py::TestClass::test_method

# Run with verbose output
pytest -v onnxruntime/test/python/test_specific.py

# Run tests matching a keyword
pytest -k "test_keyword" onnxruntime/test/python/
```

## Test naming convention

Python tests follow this pattern:

```
test_<method>_<expected_behavior>_[when_<condition>]
```

Example: `test_method_x_raises_error_when_dims_is_not_a_sequence`

## Workflow

1. Determine what the user wants to test (all tests, specific C++ test, specific Python test, etc.).
2. Detect the platform to construct the correct paths.
3. For C++ tests, check that the build directory exists and a prior build has been completed.
4. Construct and run the appropriate test command.
5. If tests fail, analyze the output and help debug the failures.
