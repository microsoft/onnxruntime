# Agent Instructions for ONNX Runtime

## Build, Test, and Lint

See the `/ort-build`, `/ort-test`, and `/ort-lint` skills (in `.agents/skills/`) for detailed build, test, and lint instructions.

## Architecture Overview

ONNX Runtime is a cross-platform inference and training engine for ONNX models. The core pipeline is: **Load model → Build graph → Optimize graph → Partition across Execution Providers → Execute**.

### Key layers (`onnxruntime/core/`)

- **`graph/`** — ONNX model/graph IR. `Model` wraps a `Graph` of `Node`s connected by edges. `GraphViewer` provides read-only traversal.
- **`optimizer/`** — Graph transformation passes (fusion, elimination, constant folding, layout transformation). Transformers implement `GraphTransformer::ApplyImpl()` and are organized by optimization level (Level1–Level4).
- **`framework/`** — Execution machinery: `OpKernel` (operator implementations), `Tensor`, `KernelRegistry`, allocators, executors.
- **`session/`** — `InferenceSession` is the main runtime class. Flow: `Load()` → `Initialize()` (optimize + assign kernels) → `Run()`.
- **`providers/`** — Execution Provider (EP) implementations. Each EP implements `IExecutionProvider` to declare which ops it can run and how to allocate device memory. CPU EP is the default fallback. 20+ EPs exist (CUDA, TensorRT, DirectML, CoreML, OpenVINO, WebGPU, QNN, etc.).
- **`common/`** — Utilities, status/error types, logging, threading.
- **`platform/`** — OS abstraction (file I/O, threading).

### Contrib ops (`onnxruntime/contrib_ops/`)

Custom operators not in the ONNX standard, organized by EP (`cpu/`, `cuda/`, `js/`, `webgpu/`). Registration is in `cpu_contrib_kernels.cc` / `cuda_contrib_kernels.cc`.

### Training (`orttraining/`)

Training-specific code (gradient ops, loss functions, optimizers, `TrainingSession`) layered on top of the inference framework.

### Language bindings

`csharp/`, `java/`, `js/`, `objectivec/`, `rust/` — each wraps the C API (`include/onnxruntime/core/session/onnxruntime_c_api.h`).

## C++ Conventions

**Style**: Google C++ Style with modifications. Max line length 120. Configured in `.clang-format` and `.clang-tidy`.

### Error handling

Functions that can fail return `onnxruntime::common::Status`. Use these macros from `core/common/common.h`:

- `ORT_RETURN_IF_ERROR(expr)` — early-return if `expr` returns non-OK Status
- `ORT_THROW_IF_ERROR(expr)` — throw if `expr` returns non-OK Status
- `ORT_RETURN_IF(condition, ...)` / `ORT_RETURN_IF_NOT(condition, ...)` — conditional early-return with message
- `ORT_ENFORCE(condition, ...)` — assert-like; throws `OnnxRuntimeException` on failure
- `ORT_MAKE_STATUS(category, code, ...)` — construct a Status object

Exceptions may be disabled in a build, in which case, the throwing macros will call `abort()` instead.

In the C API boundary, use `API_IMPL_BEGIN` / `API_IMPL_END` to catch exceptions—C++ exceptions must never cross the C API boundary.

### Container types (minimize allocations)

Required over `std::vector` / `std::unordered_map`:

- `InlinedVector<T>` — small-buffer-optimized vector (64 bytes inline). From `core/common/inlined_containers_fwd.h`.
- `InlinedHashSet<T>`, `InlinedHashMap<K,V>` — flat hash containers. From `core/common/inlined_containers.h`.
- `NodeHashSet<T>`, `NodeHashMap<K,V>` — when pointer stability is needed.
- `TensorShapeVector` — for shape dimensions. From `core/framework/tensor_shape.h`.

Use `reserve()` not `resize()`. Do not use `absl::` directly—use the ORT typedefs.

### Other key conventions

- Use `#pragma once` for header guards.
- Use `ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE` for new classes until copy/move is proven necessary.
- Prefer `gsl::span<const T>` over `const std::vector<T>&` for input parameters.
- Prefer `std::string_view` by value over `const std::string&`.
- Use `SafeInt<size_t>` (from `core/common/safeint.h`) for memory size arithmetic to prevent overflow.
- Don't use `else` after `return`.
- Avoid the `long` type (ambiguous width). Use `int64_t` for dimensions, `size_t` for counts.
- `using namespace` is allowed in limited scope but never at global scope in headers.
- Use `std::make_unique()` for heap allocations; prefer `std::optional` over `unique_ptr` for optional/delayed construction.

## Python Environment

The build and test processes may install Python dependencies. To avoid modifying the system or user Python environment, create and activate an isolated virtual environment before building or testing:

```bash
# Create a virtual environment (one-time setup)
python -m venv .venv

# Activate it
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

If a virtual environment already exists (e.g., `.venv/`), activate it rather than creating a new one.

## Python Conventions

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) (extension of PEP 8).
- Max line length: 120 characters.
- Formatter: ruff (configured in `pyproject.toml`).
- Static type checking: pyright/pylance.
- Test framework: `unittest` (preferred) with `pytest` as runner.

## C API Conventions

The main public C API header is in `include/onnxruntime/core/session/onnxruntime_c_api.h`:

- Functions that may fail return `OrtStatus*` (`nullptr` on success); release/cleanup functions (e.g., `OrtReleaseXxx`) return `void`.
- Object lifecycle: `OrtCreateXxx` / `OrtReleaseXxx`.
- All strings are UTF-8 encoded.
- Use `int64_t` for dimensions, `size_t` for counts and memory sizes.
- APIs requiring allocation take an `OrtAllocator*` parameter.
- Failed calls must not modify out-parameters.

Other public C/C++ API headers are named like `onnxruntime_*.h` and located in one of these directories:
- `include/onnxruntime/core/session/`
- `orttraining/orttraining/training_api/include/`

## PR Guidelines

- Keep PRs small (aim for ≤10 files; separate cosmetic changes from functional ones).
- All changes must have unit tests, unless they are documentation-only or already adequately covered by existing unit tests.
- Build and test locally on at least one platform before submitting.
- PR author is responsible for merging after approval.
