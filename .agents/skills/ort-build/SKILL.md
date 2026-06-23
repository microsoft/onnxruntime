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

You do **not** need `--update` when only modifying existing `.cc`/`.h` files — just use `--build`. Skipping it saves time.

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
```

## Key flags

| Flag | Description |
|------|-------------|
| `--config` | `Debug`, `MinSizeRel`, `Release`, or `RelWithDebInfo` |
| `--parallel` | Enable parallel compilation (recommended) |
| `--skip_tests` | Skip running tests after build |
| `--build_wheel` | Build the Python wheel package |
| `--use_cuda` | Enable CUDA EP. Requires `--cuda_home`/`--cudnn_home` or `CUDA_HOME`/`CUDNN_HOME` env vars. On Windows, only `cuda_home`/`CUDA_HOME` is validated. |
| `--target T` | Build a specific CMake target (requires `--build`; e.g., `onnxruntime_common`, `onnxruntime_test_all`) |
| `--use_webgpu` | Enable WebGPU EP. To run its tests locally on Linux without a GPU, see the `webgpu-local-testing` skill. |
| `--cmake_extra_defines onnxruntime_QUICK_BUILD=ON` | Faster CUDA build: instantiates a reduced kernel set. **Side effect:** Flash is compiled for head_dim 128 only, so most attention shapes fall back to **MEA** (changes which attention kernel is compiled/dispatched). Don't use it to characterize Flash-vs-arch behavior. |
| `--build_dir` | Build output directory |

## Build output path

Default: `build/<Platform>/<Config>/` where Platform is `Linux`, `MacOS`, or `Windows`.

With Visual Studio multi-config generators, the config name appears twice (e.g., `build/Windows/Release/Release/`).

It may be customized with `--build_dir`.

## Agent tips

- **Activate a Python virtual environment** before building. See "Python > Virtual environment" in `AGENTS.md`.
- **Build flags can silently reroute which kernel/code path executes.** A build option can
  change *which* kernel is compiled, and therefore which code path actually runs — so a CI
  failure can live in a different code path than your local build exercises. Before
  hypothesizing a hardware- or algorithm-specific cause (e.g. "this GPU arch miscomputes"),
  first identify **which kernel actually ran** for the failing configuration (see the
  `ort-test` skill → "Verify which path/kernel actually executed"). Concrete instance:
  `onnxruntime_QUICK_BUILD=ON` compiles FlashAttention for head_dim 128 only, so most
  attention shapes silently dispatch to Memory-Efficient Attention instead of Flash —
  details in the `cuda-attention-kernel-patterns` skill.
- **Prefer `python tools/ci_build/build.py` directly** over `build.bat`/`build.sh` when redirecting output. The `.bat` wrapper runs in `cmd.exe`, which breaks PowerShell redirection.
- **Redirect output to a file** (e.g., `> build_log.txt 2>&1`). Build output is large and will overflow terminal buffers.
- **Run builds in the background** — a full build can take tens of minutes to over an hour. Poll the log for `"Build complete"` or errors.
- **Use `--parallel`** by default unless the user says otherwise.
- Ask the user what they want to build (config, execution providers, wheel, etc.) if not clear from their prompt.
