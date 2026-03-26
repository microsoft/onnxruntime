# Copilot Instructions for ONNX Runtime

## Build

All build scripts delegate to `tools/ci_build/build.py`, which has three main phases:

- `--update` — generate CMake build files
- `--build` — compile (add `--parallel` to speed this up)
- `--test` — run tests

For native builds, if none of `--update`, `--build`, or `--test` are specified and you do not pass `--skip_tests`, **all three run by default**. For cross-compiled builds, the default is `--update` + `--build` only, and you must specify `--test` explicitly if you want to run tests.

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

# Build with specific execution provider
./build.sh --config Release --parallel --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda
# Build Python wheel
./build.sh --config Release --parallel --build_wheel
```

Key flags: `--config` (Debug|MinSizeRel|Release|RelWithDebInfo), `--parallel`, `--skip_tests`, `--build_wheel`, `--use_cuda`, `--use_tensorrt`, `--use_dml`, `--use_openvino`, `--enable_training`.

## Test

C++ tests use Google Test. Python tests use `unittest` (preferred) and `pytest`.

```bash
# Run all C++ tests after build
cd build/<platform>/Release && ctest

# Run a single C++ test binary
./build/Linux/Release/onnxruntime_test_all --gtest_filter="*TestName*"

# Run Python tests
pytest onnxruntime/test/python/test_specific.py
pytest onnxruntime/test/python/test_specific.py::TestClass::test_method
```

Python test naming convention: `test_<method>_<expected_behavior>_[when_<condition>]` (e.g., `test_method_x_raises_error_when_dims_is_not_a_sequence`).

## Lint

Uses [lintrunner](https://github.com/suo/lintrunner) for both C++ (clang-format) and Python (ruff).

```bash
pip install -r requirements-lintrunner.txt
lintrunner init

# Format changed files
lintrunner -a
# Format all files
lintrunner -a --all-files
# Format Python files only
lintrunner f --all-files
```

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

**Style**: Google C++ Style with modifications. Max line length 120 (aim for 80). Configured in `.clang-format` and `.clang-tidy`.

### Error handling

Functions that can fail return `onnxruntime::common::Status`. Use these macros from `core/common/common.h`:

- `ORT_RETURN_IF_ERROR(expr)` — early-return if `expr` returns non-OK Status
- `ORT_THROW_IF_ERROR(expr)` — throw if `expr` returns non-OK Status
- `ORT_RETURN_IF(condition, ...)` / `ORT_RETURN_IF_NOT(condition, ...)` — conditional early-return with message
- `ORT_ENFORCE(condition, ...)` — assert-like; throws `OnnxRuntimeException` on failure
- `ORT_MAKE_STATUS(category, code, ...)` — construct a Status object

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

### Operator kernel registration pattern

Kernels are declared with macros and registered in a `BuildKernelCreateInfo` list:

```cpp
// Forward declaration
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Relu);

// Registration in RegisterCPUKernels() / RegisterCpuContribKernels()
BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Relu)>,

// Kernel implementation
ONNX_OPERATOR_KERNEL_EX(OpName, domain, opset, provider, kernel_def, KernelClass);
```

For CUDA ops: host code goes in `.cc`, device kernels in `.cu`/`.cuh`. Use `ToCudaType<T>::MappedType` for type mapping. CUDA kernel classes inherit `CudaKernel` and override `ComputeInternal`.

## Python Conventions

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) (extension of PEP 8).
- Max line length: 120 characters.
- Formatter: ruff (configured in `pyproject.toml`).
- Static type checking: pyright/pylance.
- Test framework: `unittest` (preferred) with `pytest` as runner.

## C API Conventions

The public C API is in `include/onnxruntime/core/session/onnxruntime_c_api.h`:

- Functions that may fail return `OrtStatus*` (`nullptr` on success); release/cleanup functions (e.g., `OrtReleaseXxx`) return `void`.
- Object lifecycle: `OrtCreateXxx` / `OrtReleaseXxx`.
- All strings are UTF-8 encoded.
- Use `int64_t` for dimensions, `size_t` for counts and memory sizes.
- APIs requiring allocation take an `OrtAllocator*` parameter.
- Failed calls must not modify out-parameters.

## PR Guidelines

- Keep PRs small (aim for ≤10 files; separate cosmetic changes from functional ones).
- All changes must have unit tests, unless they are documentation-only or already adequately covered by existing unit tests.
- Build and test locally on at least one platform before submitting.
- PR author is responsible for merging after approval.
