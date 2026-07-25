# Plan: Use the Compile API to save a fully-optimized (non-compiling) model

## Goal

Allow the Compile API (`OrtCompileApi` / `ModelCompilationOptions` / `CompileModel`) to save a model that does
**not** produce EPContext nodes, reusing the Compile API's output mechanisms (user-provided allocator+buffer,
write-function, file, external-initializer handling).

In this mode:

- No EPContext nodes are generated.
- The model is saved after partitioning **and** Level 2+ graph optimizations have been applied.
- The mode is explicitly selected. Selecting it implies no EPContext nodes are created, and no warnings are
  emitted about absent EPContext nodes.

## Background

Compilation runs inside `InferenceSession::Initialize()` -> `TransformGraph()`. The ordering in
[`TransformGraph`](../onnxruntime/core/session/inference_session.cc#L1510) is:

1. AOT function inlining
2. `EnsureUniqueDQForNodeUnit` (required)
3. Level 0 + Level 1 optimizations
4. `partitioner.Partition(...)` — EP `GetCapability` / `Compile` / fusion, and the EPContext model save
5. Level 2 / Level 3 / cast / Level 4 optimization loop
6. `MemcpyTransformer` (copy-node insertion)

There are two existing model-save mechanisms:

1. **EPContext save (Compile API).** Runs inside
   [`GraphPartitioner::Partition`](../onnxruntime/core/framework/graph_partitioner.cc#L1593) via
   [`CreateEpContextModel`](../onnxruntime/core/framework/graph_partitioner.cc#L1057). It builds a separate
   `Model`, substitutes fused nodes -> EPContext nodes, resolves, converts to `ModelProto`
   ([`EpContextModelToProto`](../onnxruntime/core/framework/ep_context_utils.cc)), then writes to buffer /
   write-func / file based on [`epctx::ModelGenOptions`](../onnxruntime/core/framework/ep_context_options.h#L58).
   Because it runs at step 4, Level 2+ never touches the saved model.

2. **Optimized-model save (`optimized_model_filepath`).** Runs in `Initialize` at
   [inference_session.cc](../onnxruntime/core/session/inference_session.cc#L2742), after `TransformGraph`
   completes (so Level 2+ is applied). It guards against compiled nodes
   (`session_state_->GetFuncMgr().NumFuncs() > 0` -> error), warns about hardware-specific L3
   (`NchwcTransformer`), and saves via `Model::Save` or `Model::SaveWithExternalInitializers`. It only supports
   file output.

Relevant facts:

- Output-location and initializer-externalization options already live in
  [`epctx::ModelGenOptions`](../onnxruntime/core/framework/ep_context_options.h#L58) (buffer/allocator,
  write-func, file, external-initializer file info).
- The Compile API currently forces
  [`graph_optimization_level = TransformerLevel::Default` (L0, required transformers only)](../onnxruntime/core/session/model_compilation_options.cc#L38)
  and sets `enable=1`, `disable_model_compile=0`, `compile_only=1`.
- [`kOrtSessionOptionsDisableModelCompile=1`](../onnxruntime/core/framework/graph_partitioner.cc#L784) makes a
  compiling EP fail cleanly with `MODEL_REQUIRES_COMPILATION`.
- `MemcpyTransformer` (step 6) inserts ORT-internal `Memcpy*` nodes. The optimized-model save runs after it.

## Approach

Add a new output-model type to the Compile API. When selected, the model is saved from the post-optimization
save location in `Initialize` using the Compile API's output targets (buffer / write-func / file). The existing
EPContext save is unchanged. The only shared logic — serializing a `ModelProto` to buffer / file / write-func
with external-initializer handling — is extracted into one helper.

Optimization level:

- **EPContext (compiling) mode** continues to force `graph_optimization_level = TransformerLevel::Default` (L0),
  since compiling EPs do not want optimizations to run before compilation.
- **Non-compiling mode** defaults `graph_optimization_level` to the ORT full level so Level 2+ applies. The user
  can still cap it via the existing
  [`ModelCompilationOptions_SetGraphOptimizationLevel`](../onnxruntime/core/session/compile_api.cc#L293) API
  (`ORT_DISABLE_ALL`..`ORT_ENABLE_ALL`); an explicit user value always wins.

## Implementation steps

### 1. Add an output-model-type selector to `epctx::ModelGenOptions`

In [ep_context_options.h](../onnxruntime/core/framework/ep_context_options.h#L58) add:

```cpp
enum class OutputModelType {
  kEpContext,     // default: generate EPContext nodes (current behavior)
  kOptimizedOnnx, // no EPContext nodes; save fully-optimized ONNX model
};
OutputModelType output_model_type = OutputModelType::kEpContext;
```

The default of `kEpContext` preserves existing behavior with no caller change required. When `kOptimizedOnnx`,
no EPContext nodes are generated and `action_if_no_compiled_nodes` does not apply.

### 2. Add a dedicated setter on `ModelCompilationOptions`

Add `ModelCompilationOptions_SetOutputModelType(...)` to the C API (`OrtCompileApi`) plus the C++/C#/Java
wrappers. In [model_compilation_options.cc](../onnxruntime/core/session/model_compilation_options.cc#L18), when
`kOptimizedOnnx` is selected:

- set `ep_context_gen_options.output_model_type = kOptimizedOnnx`,
- default `graph_optimization_level` to the ORT full level (do not force L0),
- set `kOrtSessionOptionsDisableModelCompile=1` so any registered compiling EP fails fast with
  `MODEL_REQUIRES_COMPILATION`,
- keep `enable=1` and `compile_only=1`.

The optimization-level default must not clobber an explicit user choice regardless of call order. Track whether
the user called
[`SetGraphOptimizationLevel`](../onnxruntime/core/session/model_compilation_options.cc#L226) (a flag set in that
setter), and resolve the effective level when compiling: use the user's value if set, otherwise L0 for
`kEpContext` and the full level for `kOptimizedOnnx`. This lets a user explicitly set a maximum (e.g. cap at
`ORT_ENABLE_EXTENDED`) in the non-compiling mode.

### 3. Extract a reusable serialization helper

Pull the tail of `CreateEpContextModel` (from `EpContextModelToProto(...)` through the buffer / write-func /
file writing, lines ~1213–1290 of
[graph_partitioner.cc](../onnxruntime/core/framework/graph_partitioner.cc#L1213)) into a shared function in
`ep_context_utils.*`:

```cpp
Status SaveModelProtoToLocation(ONNX_NAMESPACE::ModelProto& model_proto,
                                const epctx::ModelGenOptions& gen_options,
                                const std::filesystem::path& valid_output_model_path,
                                const logging::Logger& logger);
```

Both the EPContext save and the non-compiling save call it. This centralizes the buffer/allocator, write-func,
and file writing paths.

### 4. Skip `CreateEpContextModel` for the non-compiling mode

In [GraphPartitioner::Partition](../onnxruntime/core/framework/graph_partitioner.cc#L1596), call
`CreateEpContextModel(...)` only when `output_model_type == kEpContext`. For `kOptimizedOnnx`, `Partition`
performs normal node assignment only, which also suppresses the "no compiled nodes" warnings.

### 5. Save the optimized model in `Initialize`

Extend the `saving_model` block in
[inference_session.cc](../onnxruntime/core/session/inference_session.cc#L2742):

- Treat the session as saving a model when either `optimized_model_filepath` is set **or**
  `ep_context_gen_options.enable && output_model_type == kOptimizedOnnx`.
- If `optimized_model_filepath` is set *and* the non-compiling Compile-API mode is active, return an error — the
  two output mechanisms are mutually exclusive.
- Reuse the existing compiled-node guard (`NumFuncs() > 0`) and the `NchwcTransformer`/L3 warning.
- For the Compile-API mode, build the `ModelProto` from `model_` (using the same external-initializer path as
  `EpContextModelToProto` / `Model::SaveWithExternalInitializers`) and call `SaveModelProtoToLocation(...)` with
  `ep_context_gen_options` so output goes to the buffer / write-func / file target chosen through the Compile
  API.
- Ensure initializers are available at serialization time. The `optimized_model_filepath` path passes
  `keep_initializers = !saving_model` to `FinalizeSessionState`; in compile-only mode `FinalizeSessionState` is
  skipped entirely, so the graph retains its initializers.

The save runs inside the `saving_model` block, before the `compile_only_session` early-return at
[inference_session.cc](../onnxruntime/core/session/inference_session.cc#L2782), so keeping `compile_only=1`
works: the model is saved, then the session returns early.

## Edge cases

- **In-memory initializers.** `ConvertInitializersIntoOrtValues()` runs before partitioning and rewrites
  `TensorProto`s with a native-memory-address tag. Reuse the existing model-proto conversion path
  (`EpContextModelToProto` / `Model::ToGraphProtoWithExternalInitializers`) so these serialize correctly.
  Keep the conversion rather than skipping it in compile-only mode: the existing EPContext save already relies
  on this convert-then-serialize path, so reusing it is consistent and proven. The conversion also performs
  external-data path validation and guards against malicious in-memory references, and provides OrtValue-format
  initializers to plugin EPs during `GetCapability` (still invoked in compile-only mode). Skipping it would only
  save an inference-oriented round-trip while requiring those checks to be re-added and risking a behavioral
  divergence between compile and inference partitioning; treat "skip conversion in compile-only mode" as a later,
  separately-validated optimization if profiling shows the round-trip is costly.
- **Contrib ops in output.** Level 2+ produces `com.microsoft` nodes; the saved model requires ORT (or an EP
  with those kernels) to load, the same as `optimized_model_filepath`.
- **Fused / compiled nodes.** There is currently no way to serialize function-based fused nodes, so the
  non-compiling mode must never produce one. Two guards ensure this:
  - `disable_model_compile=1` (set by this mode) makes partitioning fail fast: in
    [graph_partitioner.cc](../onnxruntime/core/framework/graph_partitioner.cc#L783), immediately before an EP's
    `Compile()` is called, if the fused subgraph still contains a node the EP would need to compile, `Partition`
    returns `MODEL_REQUIRES_COMPILATION`. Because the check runs *before* `Compile()`, no compiled function nodes
    (and no `FuncManager` entries) are ever created. This is what "fails fast on a compiling EP" means — it is
    triggered when an EP actually needs to compile a node, not by EP type.
  - The `NumFuncs() > 0` guard at save time is a defensive backstop that errors if any compiled function
    somehow exists.

  This mode is therefore intended for non-compiling EP configurations (e.g. CPU EP).
- **`error_if_output_file_exists`.** For file output, perform the `GetValidatedEpContextPath` validation early
  to fail fast before running optimizations.
- **`Memcpy*` nodes.** Same limitation as `optimized_model_filepath`: the save runs after `MemcpyTransformer`.
  For CPU-only / single-device non-compiling models there are none.

## Testing

- Compile-API tests (e.g.
  [qnn_ep_context_test.cc](../onnxruntime/test/providers/qnn/qnn_ep_context_test.cc), Python
  [onnxruntime_test_python_compile_api.py](../onnxruntime/test/python/onnxruntime_test_python_compile_api.py))
  with CPU-EP-only cases: input model -> non-compiling save to (a) file, (b) buffer, (c) write-func. Assert the
  output has no EPContext nodes, contains expected Level 2+ fusions (a contrib op that only appears post-L2),
  and re-loads/runs correctly.
- Selecting the new mode with a compiling EP registered fails with `MODEL_REQUIRES_COMPILATION`.
- No warning is emitted about missing EPContext nodes in the new mode.
- Setting both `optimized_model_filepath` and the new mode returns an error.
- Regression: existing EPContext compile path and `optimized_model_filepath` path are unchanged.
