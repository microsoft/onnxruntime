# Agent Instructions for ONNX Runtime

## Build, Test, and Lint

See the `/ort-build`, `/ort-test`, and `/ort-lint` skills (in `.agents/skills/`) for detailed instructions.

### Build prerequisites

- **CMake ≥ 3.28**, **C++20**, C99
- Python 3.10+ for core build scripts/tooling; Python 3.11+ for building/installing the Python wheel and running Python tests
- Platform toolchain (MSVC on Windows, GCC/Clang on Linux, Xcode on macOS)
- EP-specific SDKs as needed (CUDA Toolkit, TensorRT, etc.)

## Architecture Overview

ONNX Runtime is a cross-platform inference and training engine for ONNX models. The core pipeline is: **Load model → Build graph → Optimize graph → Partition across Execution Providers → Execute**.

### Key layers (`onnxruntime/core/`)

- **`graph/`** — ONNX model/graph IR. `Model` wraps a `Graph` of `Node`s. `GraphViewer` provides read-only traversal.
- **`optimizer/`** — Graph transformations (fusion, elimination, constant folding, layout transforms). Organized by optimization level (Level1–Level4).
- **`framework/`** — Execution machinery: `OpKernel`, `Tensor`, `KernelRegistry`, allocators, executors.
- **`session/`** — `InferenceSession`: `Load()` → `Initialize()` (optimize + assign kernels) → `Run()`.
- **`providers/`** — Execution Provider (EP) implementations. Each EP implements `IExecutionProvider`. CPU EP is the default fallback. 20+ EPs exist (CUDA, TensorRT, DirectML, CoreML, OpenVINO, WebGPU, QNN, etc.).
- **`common/`** — Utilities, status/error types, logging, threading.
- **`platform/`** — OS abstraction (file I/O, threading).

### Contrib ops (`onnxruntime/contrib_ops/`)

Custom operators not in the ONNX standard, organized by EP (`cpu/`, `cuda/`, `js/`, `webgpu/`). Each EP has its own contrib kernel registration file (e.g., `cpu_contrib_kernels.cc`, `cuda_contrib_kernels.cc`, `js_contrib_kernels.cc`, `webgpu_contrib_kernels.cc`).

### Training (`orttraining/`)

Training-specific code (gradient ops, loss functions, optimizers, `TrainingSession`) layered on top of the inference framework.

### Language bindings

`csharp/`, `java/`, `js/`, `objectivec/`, `rust/` — each wraps the C API (`include/onnxruntime/core/session/onnxruntime_c_api.h`).

### Adding operators / kernels

Kernel registration uses macros from `core/framework/op_kernel.h`:

- `ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, version, OpName)` — standard op
- `ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, version, type, OpName)` — typed variant
- `ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startVer, endVer, OpName)` — versioned range

To add a new contrib op: implement the kernel class, then register it in the EP's contrib kernel file (e.g., `contrib_ops/cpu/cpu_contrib_kernels.cc`). See `docs/OperatorKernels.md` and `docs/ContribOperators.md` for more details.

### Execution Providers

Each EP implements `IExecutionProvider` (`include/onnxruntime/core/framework/execution_provider.h`). Key overrides: `GetCapability()` (claim nodes), kernel registration, `GetDataTransfer()` (device memory copies). See `docs/execution_providers/` for EP-specific documentation.

### Test organization (`onnxruntime/test/`)

| Directory | Content |
|-----------|---------|
| `framework/` | Core framework tests (tensor, allocator, session) |
| `optimizer/` | Graph transformer unit tests |
| `providers/` | Per-EP operator kernel tests |
| `contrib_ops/` | Tests for contrib/custom operators |
| `python/` | Python API and integration tests |
| `shared_lib/` | C API / shared library tests |
| `quantization/` | Quantization tool tests |
| `wasm/`, `webgpu/` | Web platform tests |
| `testdata/` | ONNX models and data files used by tests |

## C++ Conventions

**Style**: Google C++ Style with modifications. Max line length 120 (aim for 80). See `docs/Coding_Conventions_and_Standards.md` for full details.

### Error handling

Functions that can fail return `onnxruntime::common::Status`. Key macros from `core/common/common.h`:

- `ORT_RETURN_IF_ERROR(expr)` — early-return if `expr` returns non-OK Status
- `ORT_THROW_IF_ERROR(expr)` — throw if `expr` returns non-OK Status
- `ORT_RETURN_IF(cond, ...)` / `ORT_RETURN_IF_NOT(cond, ...)` — conditional early-return with message
- `ORT_ENFORCE(cond, ...)` — assert-like; throws `OnnxRuntimeException` on failure
- `ORT_MAKE_STATUS(category, code, ...)` — construct a Status object

Exceptions may be disabled in a build, in which case, the throwing macros will call `abort()` instead.

At the C API boundary, use `API_IMPL_BEGIN` / `API_IMPL_END` to catch exceptions — C++ exceptions must never cross the C API boundary.

### Container types

Use these instead of `std::vector` / `std::unordered_map`:

- `InlinedVector<T>` — small-buffer-optimized vector (64 bytes inline)
- `InlinedHashSet<T>`, `InlinedHashMap<K,V>` — flat hash containers
- `NodeHashSet<T>`, `NodeHashMap<K,V>` — when pointer stability is needed
- `TensorShapeVector` — for shape dimensions

Use `reserve()` not `resize()`. Do not use `absl::` directly — use the ORT typedefs.

### Other conventions

- `#pragma once` for header guards
- `ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE` for new classes until copy/move is proven necessary
- Prefer `gsl::span<const T>` over `const std::vector<T>&` for input parameters
- Prefer `std::string_view` by value over `const std::string&`
- `SafeInt<size_t>` (from `core/common/safeint.h`) for memory size arithmetic
- Don't use `else` after `return`
- Avoid `long` (ambiguous width) — use `int64_t` for dimensions, `size_t` for counts
- `using namespace` allowed in limited scope but never at global scope in headers
- `std::make_unique()` for heap allocations; prefer `std::optional` over `unique_ptr` for optional/delayed construction

## Python

### Virtual environment

Build and test processes may install Python packages. Create and activate an isolated virtual environment first:

```bash
python -m venv .venv                  # one-time setup
source .venv/bin/activate             # Linux/macOS
.\.venv\Scripts\Activate.ps1          # Windows (PowerShell)
```

If a virtual environment already exists (e.g., `.venv/`), activate it rather than creating a new one.

### Conventions

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) (extension of PEP 8)
- Max line length: 120 characters
- Formatter: ruff (configured in `pyproject.toml`)
- Static type checking: pyright/pylance
- Test framework: `unittest` (preferred) with `pytest` as runner

## C API Conventions

The main public C API header is `include/onnxruntime/core/session/onnxruntime_c_api.h`. Other public headers are in `include/onnxruntime/core/session/` and `orttraining/orttraining/training_api/include/`.

- Functions that may fail return `OrtStatus*` (`nullptr` on success); release/cleanup functions return `void`
- Object lifecycle: `OrtCreateXxx` / `OrtReleaseXxx`
- All strings are UTF-8 encoded
- Use `int64_t` for dimensions, `size_t` for counts and memory sizes
- APIs requiring allocation take an `OrtAllocator*` parameter
- Failed calls must not modify out-parameters

## PR Guidelines

- Keep PRs small (aim for ≤10 files; separate cosmetic changes from functional ones)
- All changes must have unit tests, unless documentation-only or already adequately covered
- Build and test locally on at least one platform before submitting
- PR author is responsible for merging after approval

## Tool Configuration

| Tool | Purpose | Config file |
|------|---------|-------------|
| ruff | Python lint + format | `pyproject.toml` |
| pyright | Python static type checking | `pyproject.toml` |
| clang-format | C++ formatting | `.clang-format` |
| clang-tidy | C++ static analysis | `.clang-tidy` |
| lintrunner | Unified lint runner (wraps ruff, clang-format) | `.lintrunner.toml` |

Use `lintrunner -a` to auto-fix changed files. See the `/ort-lint` skill for detailed commands.

## Key Documentation

| Topic | Path |
|-------|------|
| C++ coding conventions | `docs/Coding_Conventions_and_Standards.md` |
| C API design guidelines | `docs/C_API_Guidelines.md` |
| CMake guidelines | `docs/cmake_guideline.md` |
| Operator kernels | `docs/OperatorKernels.md` |
| Contrib operators | `docs/ContribOperators.md` |
| PR guidelines | `docs/PR_Guidelines.md` |
| Execution providers | `docs/execution_providers/` |
| Graph partitioning | `docs/annotated_partitioning/` |
| Training (ORTModule) | `docs/ORTModule_Training_Guidelines.md` |
| Threading model | `docs/NotesOnThreading.md` |
| Release process | `docs/ReleaseManagement.md`, `docs/Versioning.md` |
