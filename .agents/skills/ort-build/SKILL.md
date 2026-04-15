---
name: ort-build
description: Build ONNX Runtime from source. Use this skill when asked to build, compile, or generate CMake files for ONNX Runtime.
---

# Building ONNX Runtime

The build scripts `build.sh` (Linux/macOS) and `build.bat` (Windows) delegate to `tools/ci_build/build.py`.

## Build phases

Three phases, controlled by flags:

- `--update` — generate CMake build files
- `--build` — compile (add `--parallel` to speed this up)
- `--test` — run tests

For native builds, if none are specified (and `--skip_tests` is not passed), **all three run by default**. For cross-compiled builds, the default is `--update` + `--build` only.

### When to use `--update`

You need `--update` when:
- First build in a new build directory
- New source files are added (some CMake targets use glob patterns, others use explicit file lists — re-run to pick up new files either way)
- CMake configuration changes (new flags, updated CMakeLists.txt)

You do **not** need `--update` when only modifying existing `.cc`/`.h` files in an already-configured build tree — just use `--build`. Skipping it saves time.

For major configuration changes (switching EP set, generator, toolchain, or architecture), prefer a **fresh `--build_dir`** rather than reconfiguring in-place, which can leave stale CMake cache entries.

## Examples

```bash
# Full build (update + build + test)
./build.sh --config Release --parallel
.\build.bat --config Release --parallel     # Windows

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

# Build a specific CMake target (much faster than a full build)
./build.sh --config Release --build --parallel --target onnxruntime_common

# Load flags from an option file (one flag per line)
./build.sh "@./custom_options.opt" --build --parallel

# Direct build.py invocation (must specify --build_dir)
python tools/ci_build/build.py --build_dir build/Linux --config Release --build --parallel
```

## Key flags

| Flag | Description |
|------|-------------|
| `--config` | `Debug`, `MinSizeRel`, `Release`, or `RelWithDebInfo` |
| `--parallel` | Enable parallel compilation (recommended) |
| `--skip_tests` | When no explicit phase flags are given, suppress the default test phase. Has no effect if `--test` is explicitly passed. |
| `--build_wheel` | Build the Python wheel package |
| `--use_cuda` | Enable CUDA EP. Requires `--cuda_home`/`--cudnn_home` or `CUDA_HOME`/`CUDNN_HOME` env vars. On Windows, only `cuda_home`/`CUDA_HOME` is validated. |
| `--target T` | Build a specific CMake target (requires `--build`; e.g., `onnxruntime_common`, `onnxruntime_test_all`) |
| `--build_dir` | Build output directory |

## Build output path

Default: `build/<Platform>/<Config>/` where Platform is `Linux`, `MacOS`, or `Windows`.

With Visual Studio multi-config generators, the config name appears twice (e.g., `build/Windows/Release/Release/`).

It may be customized with `--build_dir`.

## Agent tips

- **Activate a Python virtual environment** before building. See "Python > Virtual environment" in `AGENTS.md`.
- **Prefer `python tools/ci_build/build.py` directly** over `build.bat`/`build.sh` when redirecting output. The `.bat` wrapper spawns `cmd.exe`, which can cause unreliable output capture — `cmd.exe` and PowerShell handle stream redirection differently, leading to lost or garbled output.
- **When calling `build.py` directly, always pass `--build_dir`** (e.g., `--build_dir build/Windows`). The wrapper scripts inject this automatically, but `build.py` requires it.
- **Redirect output to a file** (e.g., `> build.log 2>&1`) — build output is large and will overflow terminal buffers. Put log files under the build directory (e.g., `<build_dir>/logs/build.log`) to avoid cluttering the repo root.
- **Builds are long-running** — a full build can take tens of minutes to over an hour. Run builds in the same session with a long timeout. If the timeout expires, the build continues running — poll infrequently until it finishes. Do not restart or re-launch the build.
- **After a build completes**, treat exit code 0 as success. If the exit code is nonzero or expected outputs are missing, inspect the log for errors. `"Build complete"` in the log is a confirming success signal for `build.py`. Expected outputs depend on the command: compiled binaries for `--build`, generated CMake files for `--update`, a `.whl` file for `--build_wheel`, etc.
- **Use `--parallel`** by default unless the user says otherwise.
- Ask the user what they want to build (config, execution providers, wheel, etc.) if not clear from their prompt.
