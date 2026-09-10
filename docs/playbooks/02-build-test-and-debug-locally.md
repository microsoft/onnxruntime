# Playbook 02: Build, Test, and Debug Locally

## Outcome

By the end of this playbook, you will be able to run a fast local edit-build-test-debug loop and scale up only when needed.

## Start Here

- [build.sh](../../build.sh)
- [build.bat](../../build.bat)
- [docs/Model_Test.md](../Model_Test.md)
- [onnxruntime/test/providers](../../onnxruntime/test/providers)

## Build Phases You Need to Know

The build driver supports three phases:

- update: generate or regenerate build files
- build: compile code
- test: run tests

First build in a new directory should include update. Incremental builds for existing source changes usually only need build.

## Recommended Local Loops

### Loop A: First local build

Linux or macOS:

```bash
./build.sh --config RelWithDebInfo --update --build --parallel --skip_tests
```

Windows:

```powershell
.\build.bat --config RelWithDebInfo --update --build --parallel --skip_tests
```

### Loop B: Incremental rebuild while iterating

Linux or macOS:

```bash
./build.sh --config RelWithDebInfo --build --parallel --skip_tests
```

Windows:

```powershell
.\build.bat --config RelWithDebInfo --build --parallel --skip_tests
```

### Loop C: Run tests only after a successful build

Linux or macOS:

```bash
./build.sh --config RelWithDebInfo --test
```

Windows:

```powershell
.\build.bat --config RelWithDebInfo --test
```

## Targeted C++ Test Execution

Run tests directly from the build output directory.

Linux example:

```bash
cd build/Linux/RelWithDebInfo
./onnxruntime_provider_test --gtest_filter="*Conv*"
./onnxruntime_test_all --gtest_filter="*Session*"
```

Windows example:

```powershell
cd build\Windows\RelWithDebInfo
.\onnxruntime_provider_test.exe --gtest_filter="*Conv*"
.\onnxruntime_test_all.exe --gtest_filter="*Session*"
```

## Model Tests

For ONNX model test data setup and onnx_test_runner options, use [docs/Model_Test.md](../Model_Test.md).

## Practical Debug Workflow

1. Build in RelWithDebInfo while iterating.
2. Run one focused test with a narrow gtest filter.
3. Reproduce with deterministic settings when possible.
4. Add or update a test before changing code if the bug is not already covered.
5. Re-run the smallest affected test set after each change.

## Common Failure Modes

- Running test binaries from the wrong working directory.
- Running full test suites too early instead of using targeted filters.
- Forgetting update phase after changing CMake files or adding new source files.
- Mixing unrelated fixes while debugging one issue.

## Exit Checklist

- [ ] You can run first build and incremental build loops.
- [ ] You can run a targeted provider test and a targeted core test.
- [ ] You can locate and use the model test runner documentation.
- [ ] You have a repeatable minimal test loop for your current task.