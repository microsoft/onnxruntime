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

Option files (e.g., `custom_options.opt`) contain one flag per line. Any additional flags on the command line are merged with the file's flags. Option files can be used to avoid repeating common flags across build calls.

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
| `--target TARGET` | Build a specific CMake target (e.g., `onnxruntime_common`). Can be repeated. |
| `--targets T1 T2 ...` | Build one or more specific CMake targets in a single flag. |
| `--build_dir` | Specify build output directory (default: `build/`) |

## Build duration

A full ONNX Runtime build can take a **long time** (tens of minutes to over an hour depending on hardware and configuration). If you have other work to do while the build runs, consider redirecting output to a file and running the build in the background so you can continue with other tasks:

```bash
# Run build in the background, redirecting output to a log file
./build.sh --config Release --parallel > build.log 2>&1 &

# Windows — redirect output to a file in a background terminal
.\build.bat --config Release --parallel > build_output.txt 2>&1
```

To monitor progress, periodically check the output file:

```powershell
# Check last few lines of build output
Get-Content build_output.txt -Tail 20

# Search for errors
Select-String -Path build_output.txt -Pattern "error C|error LNK|FAILED|Build succeeded"
```

When using the CLI agent's shell tools, run the build command in a background terminal with output redirected to a file. Use a separate terminal to periodically check progress. Do **not** run the build in a foreground terminal and then run another command in the same terminal — this will interrupt the build with a `KeyboardInterrupt`.

## Workflow

1. Ask the user what they want to build (config, execution providers, wheel, etc.) if not clear from their prompt.
2. Detect the OS and choose `build.bat` or `build.sh`.
3. Construct the build command with the appropriate flags.
4. Run the build. Use `--parallel` by default unless the user says otherwise.
5. Since the build may take a long time, continue with other tasks while waiting for it to complete.
6. If the build fails, examine the error output and suggest fixes.
