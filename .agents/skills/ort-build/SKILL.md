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

### When to use `--update`

`--update` triggers CMake regeneration. You need it when:
- **First build** in a new build directory
- **New source files** are added (CMake glob patterns need to be re-evaluated)
- **CMake configuration changes** (new flags, updated CMakeLists.txt, etc.)

You do **not** need `--update` when:
- Only modifying existing `.cc`/`.h` files (just use `--build`)
- Re-running the same build after a prior successful build

Skipping `--update` when unnecessary saves time since CMake regeneration can be slow.

## Platform detection

- On **Windows**, use `.\build.bat`
- On **Linux/macOS**, use `./build.sh`

Detect the platform and use the correct script automatically.

## Option files

Build flags can be stored in option files and loaded with the `@` prefix:

```bash
# Load flags from an option file
./build.sh "@./custom_options.opt" --build --parallel

# Windows equivalent
.\build.bat "@./custom_options.opt" --build --parallel
```

Option files (e.g., `custom_options.opt`) contain one flag per line. Any additional flags on the command line are merged with the file's flags.

Option files can be used to avoid repeating common flags across build calls.

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
| `--targets T1 T2 ...` | Build one or more specific CMake targets in a single flag (e.g., `onnxruntime_common`, `onnxruntime_test_all`). |
| `--build_dir` | Specify build output directory (default: `build/`) |

## Build output path

With Visual Studio generators, the config name appears twice in the output path. For example, `--build_dir build/foo --config Debug` produces binaries in `build/foo/Debug/Debug/`.

## Build duration

A full ONNX Runtime build can take a **long time** (tens of minutes to over an hour depending on hardware and configuration).

## Agent tips

- **Prefer calling `python tools/ci_build/build.py` directly** over `build.bat`/`build.sh` when you need to redirect output. The `.bat` wrapper runs in `cmd.exe`, which breaks PowerShell piping and redirection.
- **Always redirect build output to a file** (e.g., `python tools/ci_build/build.py ... > build_log.txt 2>&1`). Build output is large and will overflow terminal buffers. You can then search the log with `grep` or `Select-String`, or read just the last few lines.
- **Verify build success** by checking the log for `"Build complete"` and confirming expected binaries exist on disk.
- **Run builds as background commands** since they take a long time. Poll the log file for completion rather than relying on terminal idle detection, which can trigger prematurely when MSBuild spawns parallel child processes. Wait patiently rather than polling rapidly.
- **Do not** run the build in a foreground terminal and then run another command in the same terminal — this will interrupt the build.

## Workflow

1. **Activate a Python virtual environment** before building. See the "Python Environment" section in `AGENTS.md` for instructions.
2. Ask the user what they want to build(config, execution providers, wheel, etc.) if not clear from their prompt.
3. Detect the OS and choose `build.bat` or `build.sh`.
4. Construct the build command with the appropriate flags.
5. Run the build in a background terminal. Use `--parallel` by default unless the user says otherwise.
6. Wait for the build to complete. Check the last few lines for `Build complete` or errors.
7. If the build fails, examine the error output and suggest fixes.
