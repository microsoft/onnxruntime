---
name: ort-test
description: Run ONNX Runtime tests. Use this skill when asked to run tests, debug test failures, or find and execute specific test cases in ONNX Runtime.
---

# Running ONNX Runtime Tests

ONNX Runtime uses **Google Test** for C++ and **unittest** (preferred) / **pytest** for Python.

## Run tests via build script

```bash
# Run all tests through the build script (skips update and build)
./build.sh --config Release --test
.\build.bat --config Release --test
```

## C++ tests

### C++ test executables

There are multiple test executables, each covering different areas:

| Executable | What it tests |
|---|---|
| `onnxruntime_test_all` | Core framework, graph, optimizer, session tests |
| `onnxruntime_provider_test` | Operator/kernel tests (Conv, MatMul, etc.) across execution providers |

Use `--gtest_filter` to select specific tests:

```bash
./onnxruntime_provider_test --gtest_filter="*Conv3D*"
```

### Running test executables

**Always run test executables from the build output directory** (where the executable and test data/DLLs are located), not from the repo root. Tests may fail to find dependencies otherwise.

```bash
# Navigate to the build output directory first
cd build/Linux/Release
./onnxruntime_provider_test --gtest_filter="*TestName*"

# Windows
cd build\Windows\Release
.\onnxruntime_provider_test.exe --gtest_filter="*TestName*"
```

### Locating the build output directory

The build output path depends on the build directory provided to `build.py`.

If you're unsure where the test executable is, search for it:

```powershell
# Windows
Get-ChildItem -Path build -Recurse -Filter "onnxruntime_provider_test.exe" | Select-Object -ExpandProperty FullName

# Linux/macOS
find build -name "onnxruntime_provider_test" -type f
```

Common locations:
- Default: `build/<Platform>/<Config>/` (e.g., `build/Windows/Release/`)
- Specified with `build.sh` or `build.bat` `--build_dir` option

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

### Test naming convention

Python tests follow this pattern:

```
test_<method>_<expected_behavior>_[when_<condition>]
```

Example: `test_method_x_raises_error_when_dims_is_not_a_sequence`

## Workflow

1. **Activate a Python virtual environment** before running tests. See the "Python Environment" section in `AGENTS.md` for instructions.
2. Determine what the user wants to test (all tests, specific C++ test, specific Python test, etc.).
3. Detect the platform to construct the correct paths.
4. For C++ tests, check that the build directory exists and a prior build has been completed.
5. Construct and run the appropriate test command.
6. If tests fail, analyze the output and help debug the failures.

## Agent tips

- **Redirect test output to a file** (e.g., `> test_output.txt 2>&1`). Test output can be large and will overflow terminal buffers. Read the file afterward to check results.
- When running the full test suite takes too long, use `--gtest_filter` to run a targeted subset of tests relevant to the change being verified.
