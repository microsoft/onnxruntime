# Compile API: Saving a Fully-Optimized ONNX Model

## Overview

`OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT` (bit 2 of `OrtCompileApiFlags`) enables the Compile API
(`OrtCompileApi` / `ModelCompilationOptions` / `CompileModel`) to save a fully-optimized plain ONNX model
instead of an EPContext model. All existing Compile API output targets are supported: file,
user-provided allocator+buffer, and write-function.

When this flag is set:

- No EPContext nodes are generated.
- The graph optimization level defaults to `TransformerLevel::MaxLevel` (ORT_ENABLE_ALL) if the caller
  has not set an explicit level via `SetGraphOptimizationLevel`. An explicit user value always wins.
- `kOrtSessionOptionsDisableModelCompile=1` is set, so any registered compiling EP fails fast with
  `MODEL_REQUIRES_COMPILATION` rather than producing unserializable compiled function nodes.
- The output is saved after the Level 2/3/4 optimization loop completes (not during partitioning).

## Modified files

### C API / public headers

- **[include/onnxruntime/core/session/onnxruntime_c_api.h](../include/onnxruntime/core/session/onnxruntime_c_api.h)**
  — `OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT = 1 << 2` added to `OrtCompileApiFlags`.

### Core implementation

- **[onnxruntime/core/framework/ep_context_options.h](../onnxruntime/core/framework/ep_context_options.h)**
  — `OutputModelType` enum (`kEpContext`, `kOptimizedOnnx`) added to `ModelGenOptions`.

- **[onnxruntime/core/framework/compile_utils.h](../onnxruntime/core/framework/compile_utils.h)** /
  **[compile_utils.cc](../onnxruntime/core/framework/compile_utils.cc)**
  (formerly `ep_context_utils.*`) — `namespace epctx` contains all model-generation helpers:
  - `GetValidatedEpContextPath` — validates and resolves the output path.
  - `EpContextModelToProto` — serializes a `Model` to `ModelProto` (EPContext and optimized paths share this).
  - `SaveModelProtoToLocation` — writes a `ModelProto` to buffer / write-func / file.
  - `CreateEpContextModel` — builds and saves the EPContext model (moved here from `graph_partitioner.cc`).
  - `BuildAndSaveOptimizedModel` — new: builds and saves the optimized plain ONNX model (called from
    `InferenceSession::Initialize` after the L2–L4 optimization loop).

- **[onnxruntime/core/session/model_compilation_options.cc](../onnxruntime/core/session/model_compilation_options.cc)**
  — `SetFlags` handles `OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT`: sets
  `output_model_type = kOptimizedOnnx`, defaults optimization level to MaxLevel if not user-set, sets
  `kOrtSessionOptionsDisableModelCompile=1`.

- **[onnxruntime/core/session/inference_session.cc](../onnxruntime/core/session/inference_session.cc)**
  — After the L2–L4 loop, a `saving_via_compile_api_non_compiling` block detects
  `enable && output_model_type == kOptimizedOnnx` and calls `epctx::BuildAndSaveOptimizedModel`.
  Setting both `optimized_model_filepath` and this mode returns an error.

- **[onnxruntime/core/framework/graph_partitioner.cc](../onnxruntime/core/framework/graph_partitioner.cc)**
  — `CreateEpContextModel` call is guarded to `output_model_type == kEpContext` only. The local
  `GetValidatedEpContextPath` wrapper and `CreateEpContextModel` implementation were removed (now in
  `compile_utils.cc`).

### Language bindings

- **Python** — `OrtCompileApiFlags.OPTIMIZED_ONNX_OUTPUT` exposed via pybind in
  `onnxruntime_pybind_state.cc`.
- **C#** — `OrtCompileApiFlags.OPTIMIZED_ONNX_OUTPUT = 1 << 2` added to
  `csharp/src/Microsoft.ML.OnnxRuntime/CompileModel.shared.cs`.
- **Java** — `OrtCompileApiFlags.OPTIMIZED_ONNX_OUTPUT(1 << 2)` added to
  `java/src/main/java/ai/onnxruntime/OrtModelCompilationOptions.java`.

## Tests

- **C++**: `onnxruntime/test/compile_api/test_optimized_onnx_output.cc` (added to `onnxruntime_test_all`
  via `cmake/onnxruntime_unittests.cmake`). Four tests:
  - `ToFile` — compile to file, verify no EPContext nodes, reload with ORT_DISABLE_ALL.
  - `ToBuffer` — compile to buffer, verify no EPContext nodes, reload from bytes.
  - `ToWriteFunc` — compile via write-function, verify bytes written and no EPContext nodes.
  - `AppliesMaxLevelOptimizations` *(guarded by `!DISABLE_CONTRIB_OPS`)* — uses
    `testdata/transform/fusion/conv_relu.onnx`; verifies that `Relu` is absent and `FusedConv`
    (Level 2, `ConvActivationFusion`) is present in the output.

- **Python**: four `test_optimized_onnx_output_*` methods in
  `onnxruntime/test/python/onnxruntime_test_python_compile_api.py` covering the same four cases.
  The Level 2 optimization test uses `transform/fusion/conv_relu.onnx`.

## Edge cases and constraints

- **Compiling EPs.** `disable_model_compile=1` makes `GraphPartitioner::Partition` fail fast with
  `MODEL_REQUIRES_COMPILATION` if an EP needs to compile a node. The `NumFuncs() > 0` guard in
  `Initialize` is a defensive backstop.
- **`error_if_output_file_exists`.** `GetValidatedEpContextPath` is called before optimizations run
  to fail fast on a pre-existing output file.
- **Contrib ops in output.** Level 2+ may produce `com.microsoft` nodes (e.g. `FusedConv`). The saved
  model requires ORT (or a compatible EP) to load — same constraint as `optimized_model_filepath`.
- **`Memcpy*` nodes.** The save runs after `MemcpyTransformer`. For CPU-only / single-device
  non-compiling configurations there are none.
- **Mutual exclusivity.** Setting both `optimized_model_filepath` and `OPTIMIZED_ONNX_OUTPUT` returns
  an error.
