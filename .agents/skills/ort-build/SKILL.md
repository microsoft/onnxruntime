---
name: ort-build
description: Build ONNX Runtime from source. Use this skill when asked to build, compile, or generate CMake files for ONNX Runtime.
---

# Building ONNX Runtime

The main build scripts are `build.sh` (Linux/macOS) and `build.bat` (Windows). They delegate to `tools/ci_build/build.py`.

## Build phases

There are three phases, controlled by flags:

- `--update` — generate CMake build files
- `--build` — compile (add `--parallel` to speed this up)
- `--test` — run tests

For native builds, if none of `--update`, `--build`, or `--test` are specified (and `--skip_tests` is not passed), **all three run by default**.

For cross-compiled builds, the default is `--update` + `--build` only; `--test` must be specified explicitly.

## Platform detection

- On **Windows**, use `.\build.bat`
- On **Linux/macOS**, use `./build.sh`

Detect the platform and use the correct script automatically.

## Common build commands

```bash
# Full build (update + build + test)
./build.sh --config Release --parallel
# Windows equivalent
.\build.bat --config Release --parallel

# Just regenerate CMake files
./build.sh --config Release --update

# Just compile (skip CMake regeneration and tests)
./build.sh --config Release --build --parallel

# Just run tests (after a prior build)
./build.sh --config Release --test

# Build with CUDA execution provider
./build.sh --config Release --parallel --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda

# Build Python wheel
./build.sh --config Release --parallel --build_wheel

# Build only a specific CMake target (much faster than a full build)
./build.sh --config Release --build --parallel --target onnxruntime_common
```

## Key flags

| Flag | Description |
|------|-------------|
| `--config` | Build configuration: `Debug`, `MinSizeRel`, `Release`, `RelWithDebInfo` |
| `--parallel` | Enable parallel compilation (recommended) |
| `--skip_tests` | Skip running tests after build |
| `--build_wheel` | Build the Python wheel package |
| `--use_cuda` | Enable CUDA execution provider (requires `--cuda_home` and `--cudnn_home`) |
| `--use_tensorrt` | Enable TensorRT execution provider |
| `--use_dml` | Enable DirectML execution provider (Windows) |
| `--use_openvino` | Enable OpenVINO execution provider |
| `--target TARGET` | Build a specific CMake target (e.g., `onnxruntime_common`). Can be repeated. |
| `--targets T1 T2 ...` | Build one or more specific CMake targets in a single flag. |
| `--build_dir` | Specify build output directory (default: `build/`) |

## Build duration

A full ONNX Runtime build can take a **long time** (tens of minutes to over an hour depending on hardware and configuration). If you have other work to do while the build runs, consider redirecting output to a file and running the build in the background so you can continue with other tasks:

```bash
# Run build in the background, redirecting output to a log file
./build.sh --config Release --parallel > build.log 2>&1 &

# Windows equivalent (PowerShell)
Start-Process -NoNewWindow -FilePath .\build.bat -ArgumentList '--config','Release','--parallel' -RedirectStandardOutput build.log -RedirectStandardError build_err.log
```

When using the CLI agent's shell tools, prefer running the build with `mode="sync"` and a short `initial_wait` — the command will continue in the background and you'll be notified when it completes, freeing you to do other work in the meantime.

## Workflow

1. Ask the user what they want to build (config, execution providers, wheel, etc.) if not clear from their prompt.
2. Detect the OS and choose `build.bat` or `build.sh`.
3. Construct the build command with the appropriate flags.
4. Run the build. Use `--parallel` by default unless the user says otherwise.
5. Since the build may take a long time, continue with other tasks while waiting for it to complete.
6. If the build fails, examine the error output and suggest fixes.
