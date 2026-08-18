# Agent Instructions for ONNX Runtime

## Build, Test, and Lint

See the `/ort-build`, `/ort-test`, and `/ort-lint` skills (in `.agents/skills/`) for detailed instructions.

## CI

See the `/ort-ci` skill (in `.agents/skills/`) for triggering, re-running, and unblocking CI checks on a pull request (GitHub Actions, Azure Pipelines, `Python format`, and `license/cla`).

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
- **Signed vs unsigned on negative-capable differences.** Any expression of the form `a - b`
  that can be negative (an offset or remaining-budget computed from counts, e.g.
  `num_keys - num_queries`) must be stored and compared using a **signed** type
  (`int32_t`/`int64_t`), and any unsigned operand must be `static_cast` to signed *before*
  the subtraction/comparison. An unsigned result silently wraps to a huge value (`~4.29e9`
  for `uint32_t`), which can permanently satisfy or skip a relational guard with **no crash
  and no warning** — a correct-looking-but-wrong result. Concrete ORT instance + the exact
  fix sites: CUTLASS FMHA `causal_diagonal_offset`, see the `cuda-attention-kernel-patterns`
  skill §12.
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
