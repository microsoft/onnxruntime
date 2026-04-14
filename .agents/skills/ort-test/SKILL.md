---
name: ort-test
description: Run ONNX Runtime tests. Use this skill when asked to run tests, debug test failures, or find and execute specific test cases in ONNX Runtime.
---

# Running ONNX Runtime Tests

ONNX Runtime uses **Google Test** for C++ and **unittest** (preferred) / **pytest** for Python.

## C++ tests

### Test executables

| Executable | What it tests |
|---|---|
| `onnxruntime_test_all` | Core framework, graph, optimizer, session tests |
| `onnxruntime_provider_test` | Operator/kernel tests (Conv, MatMul, etc.) across execution providers |

Use `--gtest_filter` to select specific tests:

```bash
./onnxruntime_provider_test --gtest_filter="*Conv3D*"
```

### Running tests

**Always run from the build output directory** — tests may fail to find dependencies otherwise.

```bash
# Linux
cd build/Linux/Release
./onnxruntime_provider_test --gtest_filter="*TestName*"

# macOS
cd build/MacOS/Release
./onnxruntime_provider_test --gtest_filter="*TestName*"

# Windows
cd build\Windows\Release
.\onnxruntime_provider_test.exe --gtest_filter="*TestName*"
```

You can also run all tests via the build script (assumes a prior successful build):

```bash
./build.sh --config Release --test
.\build.bat --config Release --test    # Windows
```

### Locating the build output directory

The default path follows the pattern `build/<Platform>/<Config>/` where Platform is `Linux`, `MacOS`, or `Windows`. With Visual Studio multi-config generators on Windows, the config may appear twice (e.g., `build/Windows/Release/Release/`). The path can also be customized via `--build_dir`.

If you can't find a test binary, search for it:

```powershell
# Windows
Get-ChildItem -Path build -Recurse -Filter "onnxruntime_provider_test.exe" | Select-Object -ExpandProperty FullName

# Linux/macOS
find build -name "onnxruntime_provider_test" -type f
```

## Python tests

Use `pytest` as the test runner:

```bash
pytest onnxruntime/test/python/test_specific.py                          # entire file
pytest onnxruntime/test/python/test_specific.py::TestClass::test_method  # specific test
pytest -k "test_keyword" onnxruntime/test/python/                        # by keyword
```

Python test naming convention: `test_<method>_<expected_behavior>_[when_<condition>]`

## Agent tips

- **Activate a Python virtual environment** before running tests. See "Python > Virtual environment" in `AGENTS.md`.
- **Redirect test output to a file** (e.g., `> test_output.txt 2>&1`) — output can be large.
- For C++ tests, verify the build directory exists and a prior build completed before running.
- Use `--gtest_filter` to run a targeted subset when the full suite takes too long.
