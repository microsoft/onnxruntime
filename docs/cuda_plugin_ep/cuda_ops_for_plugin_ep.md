# CUDA Kernel Changes for Plugin EP Compatibility

## Overview

The CUDA Plugin EP builds CUDA operator kernels into a separate shared library
(`onnxruntime_providers_cuda_plugin.so`) that communicates with the ORT core
through the ORT EP API. This architecture requires that kernel source files
**not** depend on framework-internal types that are unavailable across the
shared-library boundary.

The plugin build uses two key mechanisms to achieve compatibility with minimal
(or zero) changes to existing kernel `.cc` files:

1. **Force-included adapter headers** — The CMake build injects
   `adapters.h` and `cuda_kernel_adapter.h` via `-include` compiler flags.
   These headers redefine macros (`ONNX_OPERATOR_*_KERNEL_EX`), provide a
   plugin-compatible `CudaKernel` base class, and supply shims for
   `OpKernelContext`, `OpKernelInfo`, etc.

2. **`BUILD_CUDA_EP_AS_PLUGIN` preprocessor guard** — For cases where the
   adapter headers alone are insufficient, kernel headers can use
   `#ifdef BUILD_CUDA_EP_AS_PLUGIN` to select an alternative code path
   (e.g., a self-contained class instead of inheriting from a CPU base class).

## Common Incompatibility Patterns

| Pattern | Description | Typical Fix |
|---------|-------------|-------------|
| **`GetComputeStream()` returning `onnxruntime::Stream*`** | The adapter `OpKernelContext` exposes `GetComputeStream()` that returns the adapter `Stream*` with `GetHandle()`. Most kernels call `GetScratchBuffer<T>(n, ctx->GetComputeStream())` which already works through the adapter. Kernels that `dynamic_cast<CudaStream*>` or call `CudaStream`-specific methods break. | Use `static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())` instead of `CudaStream*` methods. The adapter `CudaKernel::GetCublasHandle(cudaStream_t)` and `GetCudnnHandle(cudaStream_t)` are available. |
| **Inheritance from CPU base class** | Kernels like `Resize : Upsample`, `SpaceToDepth : SpaceDepthBase`, `NonMaxSuppression : NonMaxSuppressionBase` inherit from CPU provider classes that are not linked into the plugin. | Add a `#ifdef BUILD_CUDA_EP_AS_PLUGIN` block in the header with a self-contained class that inlines the needed logic (see `constant_of_shape.h` for an example). |
| **`TensorSeq` (incomplete type)** | `TensorSeq` is not available in the plugin build. `identity_op.cc` and `sequence_op.cc` operate on sequence types. | These ops should remain excluded or need `TensorSeq` to be exposed through the EP API. |
| **`CudaTuningContext`** | Kernels that call `GetTuningContext()` and use `CudaTuningContext` methods directly. The adapter provides a stub `GetTuningContext()` but full tuning infra is unavailable. | Guard tuning-specific calls with `#ifndef BUILD_CUDA_EP_AS_PLUGIN` or use the adapter's stub which returns `nullptr` (callers should null-check). |
| **`PhiloxGenerator` / RNG state** | Dropout-family ops use `PhiloxGenerator` from the `CudaStream` object. This requires `CudaStream*` access. | Needs a `PhiloxGenerator` accessor in the adapter or exclusion. |
| **`QkvToContext` taking `Stream*`** | Attention ops pass `context->GetComputeStream()` (an `onnxruntime::Stream*`) to `QkvToContext`. This function dereferences `Stream*` internally. | Either change `QkvToContext` signature to accept `cudaStream_t` + handles, or provide a `PluginStreamShim` wrapper (already in the adapter). |
| **Pure CPU ops** | `Shape`, `Size` — these register CPU-side `OpKernel` classes whose `Compute()` is in the CPU provider library. | Permanently excluded; handled by `GetCpuPreferredNodes()`. |
| **`cuda_execution_provider.h` include** | Files that directly include the real `CUDAExecutionProvider` class definition conflict with the adapter's shim class. | Use the adapter's `CUDAExecutionProvider` shim (automatically provided by `cuda_kernel_adapter.h`). |
| **KernelInfoGetAttributeArray\_string** | RNN ops call `GetAttrs<std::string>(...)` which maps to a C API function not yet available. | Wait for C API extension, or inline attribute parsing. |
| **Registration tables** | `cuda_nhwc_kernels.cc` and `cuda_contrib_kernels.cc` contain centralized `BuildKernelCreateInfo<>` tables that reference all kernel classes, including excluded ones. | Not needed — the plugin uses `PluginKernelCollector` for self-registration via macro overrides. |

## How to Bring an Excluded Kernel to Plugin EP

### Step 1: Identify the Dependency

Check why the kernel is excluded by looking at:
- The comment in `cmake/onnxruntime_providers_cuda_plugin.cmake`
- The kernel `.cc`/`.h` files for the patterns listed above

### Step 2: Apply the Minimal Fix

The preferred approach (in order of preference):

1. **No source change needed** — If the only issue was `GetComputeStream()`
   usage with `GetScratchBuffer()`, the adapter already handles this. Just
   remove the exclusion from the cmake file and test.

2. **Use `BUILD_CUDA_EP_AS_PLUGIN` guard in the header** — For CPU base
   class dependencies, add an alternative class definition:
   ```cpp
   #ifdef BUILD_CUDA_EP_AS_PLUGIN
   class MyOp final : public CudaKernel {
     // Self-contained implementation that inlines base class logic
   };
   #else
   class MyOp final : public CpuBaseClass, public CudaKernel {
     // Original implementation
   };
   #endif
   ```

3. **Modify calling convention** — For functions that take
   `onnxruntime::Stream*` or `CudaStream*`, change to accept
   `cudaStream_t` + explicit handles:
   ```cpp
   // Before:
   SomeHelper(context->GetComputeStream(), ...);
   // After:
   SomeHelper(static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle()), ...);
   ```

4. **Add a shim in `cuda_kernel_adapter.h`** — For utility functions from
   CPU providers (e.g., `ValidateInputs`, `PrepareCompute`), inline the
   logic in the adapter header so it's available in the plugin build.

5. **Inline CPU helper to header** — Move the helper implementation
   from the CPU `.cc` file to the `.h` header, wrapped in
   `#ifdef SHARED_PROVIDER` (declaration only) / `#else` (inline body).
   The `SHARED_PROVIDER` build retains the existing `ProviderHostCPU`
   bridge path. See `padbase.h`, `slice.h`, `scatter_nd.h` for examples.

6. **Templatize on info/context type** — For base class constructors
   that call `GetAttr<T>()`, templatize on `KernelInfoType` with
   `info.template GetAttr<T>(...)`. For methods that take
   `OpKernelContext&`, templatize on `KernelContextType`.
   See `roialign.h`, `unsqueeze.h`, `attention_base.h` for examples.

7. **Move CUDA type helpers to shared header** — For utility functions
   that only depend on CUDA types (not framework types), move from
   `.cc` to a header so the plugin build can consume them directly.
   See `cuda_common_type_helpers.h`.

### Step 3: Remove the CMake Exclusion

In `cmake/onnxruntime_providers_cuda_plugin.cmake`, comment out the exclusion
line with a note about what was done:
```cmake
# myop.cc: <brief description of fix>.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/myop\\.cc$")  # REMOVED in Stage N
```

### Step 4: Build and Test

```bash
# Build with plugin EP enabled
./build.sh --config Release --use_cuda --build_cuda_ep_as_plugin
# Run parity tests
python tools/ci_build/cuda_plugin_parity_report.py
```

---

## Excluded Operators Table

### Infrastructure Files (Not Operator Kernels)

These are excluded because they define the real EP infrastructure, which is
replaced by the plugin's own implementations in `plugin/`.

| File | Reason | Resolution |
|------|--------|------------|
| `cuda_execution_provider.cc` | Defines the real `CUDAExecutionProvider` class; conflicts with adapter shim. | Permanently excluded; replaced by `plugin/cuda_ep.cc`. |
| `cuda_provider_factory.cc` | Creates the real CUDA EP via `ProviderFactory`; not used in plugin architecture. | Permanently excluded; replaced by `plugin/cuda_ep_factory.cc`. |
| `cuda_provider_interface.cc` | Shared-library provider interface for the old (non-plugin) shared-library model. | Permanently excluded; not applicable to plugin EP. |
| `cuda_stream_handle.cc` | Defines `CudaStream` class; replaced by plugin stream adapter. | Permanently excluded; replaced by `plugin/cuda_stream_plugin.cc`. |
| `cuda_execution_provider_info.cc` | EP configuration parsing tied to the real EP. | Permanently excluded; replaced by `plugin/cuda_ep.cc` config. |
| `cuda_graph.cc` | CUDA graph capture tied to real EP stream management. | Permanently excluded; replaced by `plugin/cuda_graph_plugin.cc`. |
| `cuda_mempool_arena.cc` | Memory arena tied to real EP allocator infrastructure. | Permanently excluded; replaced by `plugin/cuda_allocator_plugin.cc`. |
| `cuda_common.cc` | `HalfGemmOptions` definitions conflict with adapter's inline shim. | Permanently excluded; shims provided in `cuda_kernel_adapter.h`. |
| `cuda_nhwc_kernels.cc` | Centralized kernel registration table; references all NHWC kernel classes. | Permanently excluded; `PluginKernelCollector` auto-registers. |
| `cuda_contrib_kernels.cc` | Centralized kernel registration table; references all contrib kernel classes. | Permanently excluded; `PluginKernelCollector` auto-registers. |

### Standard ONNX Operator Kernels — Currently Excluded

| File | Exclusion Reason | Change Needed to Include |
|------|-----------------|--------------------------|
| `math/einsum.cc` | Inherits from `onnxruntime::Einsum` (CPU provider); calls `Einsum::Compute()` which chains to `DeviceCompute()` through the CPU base class vtable. Also depends on `einsum_utils/` which calls `ReduceCompute`. | Add `#ifdef BUILD_CUDA_EP_AS_PLUGIN` path that directly implements `ComputeInternal()` without the CPU base class. Substantial effort — einsum is complex. |
| `math/einsum_utils/*` | `einsum_auxiliary_ops.cc` calls `ReductionOps::ReduceCompute` which is a framework-only function. | Must inline or rewrite reduction logic for plugin build. Coupled with `einsum.cc`. |
| `controlflow/*` (If, Loop, Scan) | Inherits from CPU base classes (`If`, `Loop`, `Scan` from `core/providers/cpu/controlflow/`). These ops call into the ORT session to execute subgraphs. | Plugin has custom wrappers in `plugin/cuda_controlflow_plugin.cc` that delegate to `OrtEpApi`. Permanently excluded from standard source; plugin equivalents exist. |
| `tunable/*` | Depends on `CudaTuningContext` and the real `CUDAExecutionProvider` for tuning infrastructure. | Needs full tuning API exposure through plugin interface. Low priority — tuning is optional. |
| `rnn/*` (RNN, GRU, LSTM) | Kernel constructors call `GetAttrs<std::string>("activations", ...)` which maps to `KernelInfoGetAttributeArray_string` — a C API function that does not yet exist. Also uses `CudnnRnnBase` which manages cuDNN RNN descriptors. | Extend the ORT C API with `KernelInfoGetAttributeArray_string`. After that, the dual-build signatures (already in place) should work. |
| `tensor/identity_op.cc` | Uses `TensorSeq` (incomplete type in plugin build) for sequence pass-through in `IdentityOp`. | Expose `TensorSeq` through the EP API adapter, or split the sequence codepath into a separate file with `#ifdef`. |
| `tensor/sequence_op.cc` | All ops (`SequenceAt`, `SequenceConstruct`, `SequenceInsert`, etc.) heavily use `TensorSeq`. | Same as `identity_op.cc` — requires `TensorSeq` support in the adapter. |
| `tensor/size.cc` | Pure CPU op — registers `onnxruntime::Size` whose `Compute()` is in the CPU provider. | **Permanently excluded.** Handled by `GetCpuPreferredNodes()`. |
| `tensor/shape_op.cc` | Pure CPU op — inherits from `onnxruntime::OpKernel` (framework class, not adapter `OpKernel`). Output is on CPU. | **Permanently excluded.** Handled by `GetCpuPreferredNodes()`. |
| `tensor/space_depth_ops.cc` | Inherits from `SpaceDepthBase` (CPU provider, `core/providers/cpu/tensor/space_depth_ops.h`). | `SpaceDepthBase` constructor templatized on `KernelInfoType` (#27628). Remaining: inline `SpaceDepthCompute` validation logic or add adapter-compatible path. Reduced effort. |
| `tensor/upsample.cc` | Inherits from `UpsampleBase` (CPU provider). `UpsampleBase` uses `InputDefs()` and complex attribute/input parsing in its constructor. | `UpsampleBase::AdjustOutputSizeAsPolicy` moved to header (#27628). Remaining blockers: `InputDefs()` and `OpKernelInfo::GetAllocator()` not available in adapter. Moderate effort. |
| `tensor/resize.cc` | Inherits from `Upsample<T>` which inherits from `UpsampleBase`. | Blocked on `upsample.cc` — must fix `Upsample` first, then `Resize` follows. |
| `generator/constant_of_shape.cc` | Inherits from `ConstantOfShapeBase` (CPU provider) which uses `TensorProto`/`UnpackTensor`. | **Already has `#ifdef BUILD_CUDA_EP_AS_PLUGIN` path** in the header with a self-contained class. Currently excluded because the `.cc` file's `#else` path still compiles `ConstantOfShapeBase` version. Need to verify the `#ifdef` path compiles and remove the exclusion. |
| `object_detection/*` (NonMaxSuppression, RoiAlign) | `NonMaxSuppression` inherits from `NonMaxSuppressionBase`; `RoiAlign` inherits from `RoiAlignBase`. Both CPU base classes. `NonMaxSuppression` also uses CPU helper `PrepareCompute`. | `NonMaxSuppressionBase` refactored to `NonMaxSuppressionBaseImpl` template (#27617). `RoiAlignBase` constructor templatized, `CheckROIAlignValidInput` inlined (#27628). Remaining: integration verification and residual `GetComputeStream()` issues. |
| `llm/*` | Attention kernels that call `QkvToContext` with `onnxruntime::Stream*`. Deep dependency on attention implementation internals. | Change `QkvToContext` to accept `cudaStream_t` + explicit handles, or use `PluginStreamShim`. Large surface area. |

### Contrib Operator Kernels — Currently Excluded

| File | Exclusion Reason | Change Needed to Include |
|------|-----------------|--------------------------|
| **aten_ops/\*** | PyTorch ATen operator bindings; requires `libtorch`. Not relevant for plugin EP. | **Permanently excluded.** |
| **collective/\*** | NCCL/MPI collective ops; requires distributed runtime. | **Permanently excluded** (or separate plugin). |
| **contrib llm/\*** | Same as standard `llm/` — deep `Stream*` and attention infra dependencies. | Same fix as standard `llm/`. |
| **transformers/\*** (beam_search, greedy_search, sampling) | Directly includes `cuda_execution_provider.h`. Uses session-level APIs to run subgraphs (encoder/decoder). Heavy framework dependency. | Would need significant refactoring to route subgraph execution through `OrtEpApi`. Very high effort. |
| **bert/attention.cc** | Calls `GetScratchBuffer` with `context->GetComputeStream()` (works via adapter). Main issue: calls `QkvToContext` passing `context->GetComputeStream()` (`Stream*`), and uses `IAllocator::MakeUniquePtr` with stream. | `AttentionBase::CheckInputs`/`CheckMask`/`GetPresent` moved to header (#27628). Remaining blocker: `QkvToContext` takes `Stream*`. Moderate-high effort. |
| **bert/decoder_attention.cc** | Same pattern as `attention.cc` — `QkvToContext` with `Stream*`. | Same fix as `attention.cc`. |
| **bert/decoder_masked_self_attention.cc** | Uses `GetComputeStream()` for scratch buffers and stream handle extraction. | Replace `GetComputeStream()` → adapter-compatible calls. Moderate effort. |
| **bert/embed_layer_norm.cc** | `embed_layer_norm_helper::CheckInputs` templatized and moved to header (#27617). CPU base class dependency resolved. | Verify compilation with exclusion removed — helper refactoring complete. **Very low effort.** |
| **bert/fast_gelu.cc** | Was excluded due to `bias_gelu_helper` CPU base class dependency. `bias_gelu_helper::CheckInputs` now templatized and inlined (#27617). | Verify compilation with exclusion removed — helper refactoring complete. **Very low effort.** |
| **bert/group_query_attention.cc** | Heavy use of `GetComputeStream()` (scratch buffers, stream handle extraction, `CudaStream*` cast). Complex attention pipeline with flash attention, XQA loader. | Same approach as `attention.cc`. High effort due to many code paths. |
| **bert/longformer_attention.cc** | Uses `GetScratchBuffer` with `GetComputeStream()`, workspace allocation. `LongformerAttentionBase::CheckInputs` moved to header (#27628). | Remaining blocker: `GetComputeStream()` / `Stream*` usage. Moderate effort. |
| **bert/multihead_attention.cc** | Same pattern as `attention.cc` — `QkvToContext` with `Stream*`. | Same fix as `attention.cc`. |
| **bert/packed_attention.cc** | Same attention pipeline dependency. | Same fix as `attention.cc`. |
| **bert/packed_multihead_attention.cc** | Same attention pipeline dependency. | Same fix as `attention.cc`. |
| **bert/paged_attention.cc** | Uses `GetComputeStream()` for scratch buffers and paged KV-cache management. | Replace stream access pattern. Moderate effort. |
| **bert/relative_attn_bias.cc** | Uses `GetScratchBuffer` with `GetComputeStream()`. | Simple `GetComputeStream()` pattern — may work with adapter. **Low effort to try.** |
| **bert/remove_padding.cc** | Uses `GetScratchBuffer` with `GetComputeStream()`. | Simple `GetComputeStream()` pattern — may work with adapter. **Low effort to try.** |
| **diffusion/group_norm.cc** | Uses `CudaTuningContext*` and `Stream*` in the `DispatchGroupNorm` helper. | Guard tuning path with `#ifndef BUILD_CUDA_EP_AS_PLUGIN`, change stream parameter. Moderate effort. |
| **fused_conv.cc** | Uses `GetComputeStream()` for cuDNN workspace allocation. | Replace stream access with adapter-compatible calls. Moderate effort. |
| **inverse.cc** | Uses `GetScratchBuffer` with `GetComputeStream()`. cuBLAS batched operations. | Simple pattern — likely works with adapter. **Low effort to try.** |
| **math/bias_dropout.cc** | Uses `PhiloxGenerator` from `CudaStream` for RNG state. Also `GetComputeStream()`. | Needs `PhiloxGenerator` accessor in adapter. Blocked on RNG infrastructure. |
| **math/fft_ops.cc** | Uses `onnxruntime::Stream*` directly. cuFFT plan management. | Change stream access to adapter pattern. Moderate effort. |
| **math/gemm_float8.cc/.cu** | `ComputeInternal` is in `.cu` file which uses `GetComputeStream()`. `.cu` files don't receive the force-include adapter header. | Move `GetComputeStream()` usage to `.cc` file, or pass stream as parameter to `.cu` function. Moderate effort. |
| **moe/moe.cc** | Uses `GetComputeStream()`. MoE routing + expert computation. | Replace `context->GetComputeStream()` with adapter-compatible calls. Moderate effort. |
| **sparse/sparse_attention.cc** | Uses `onnxruntime::Stream*`. Sparse attention kernel dispatch. | Same stream pattern fix. Moderate effort. |
| **tensor/shrunken_gather.cc** | Training op — includes `provider_api.h` in header. `ENABLE_TRAINING_OPS` guard. | **Permanently excluded** (training op, not needed for inference plugin). |
| **tensor/crop.cc** | `CropBase` constructor templatized on `KernelInfoType` (#27628). No `GetComputeStream()` usage. | Verify compilation with exclusion removed — constructor refactoring complete. **Very low effort.** |
| **tensor/dynamic_time_warping.cc** | Uses `GetScratchBuffer` with `GetComputeStream()`. | Simple pattern — likely works with adapter. **Low effort to try.** |
| **tensor/dynamicslice.cc** | Uses `onnxruntime::Stream*` via `GetComputeStream()`. | Simple pattern — likely works with adapter. **Low effort to try.** |
| **quantization/attention_quantization.cc** | Uses `GetScratchBuffer` with `GetComputeStream()`, calls `QkvToContext`. | Same fix as `attention.cc`. Moderate-high effort. |
| **quantization/matmul_bnb4.cc** | Uses `GetScratchBuffer` and `GetComputeStream()->GetHandle()`. | Adapter should handle this pattern. **Low effort to try.** |
| **quantization/matmul_nbits.cc** | Uses `GetScratchBuffer` with `GetComputeStream()` and `GetHandle()`. | Adapter should handle this pattern. **Low effort to try.** |
| **quantization/moe_quantization.cc** | Uses `GetComputeStream()`. Quantized MoE pipeline. | Same as `moe.cc`. Moderate effort. |
| **quantization/qordered_ops/\*** | Ordered quantization ops with framework dependencies. | Needs investigation. Low priority. |

### Operators Successfully Brought to Plugin EP (Reference)

These were previously excluded and are now included thanks to adapter
compatibility. Listed here as examples of the fix patterns applied.

| File | Fix Applied | Stage |
|------|-------------|-------|
| `tensor/reshape.cc` | `CopyTensor` replaced with explicit `cudaMemcpyAsync` on kernel stream (#27719). | 5A |
| `tensor/concat.cc` | `InputArgCount`/`GetComputeStream` usage works through adapter `OpKernelContext`. | 5A |
| `tensor/split.cc` | `GetComputeStream` usage works via `CudaKernel::GetComputeStream`. | 5A |
| `tensor/gather.cc` | Switched to `GatherBase::PrepareForComputeImpl` compatible with adapter context. | 5B |
| `tensor/gather_nd.cc` | `PrepareCompute` signature changed to `void*`/`cudaStream_t`. | 5 |
| `tensor/unsqueeze.cc` | Plugin-local `PrepareCompute` path added for adapter context. | 5B |
| `tensor/tile.cc` | Plugin-local `IsTileMemcpy` helper added. | 5B |
| `math/cumsum.cc` | Axis parsing helper inlined for plugin build. | 5B |
| `tensor/scatter_nd.cc` | `ValidateShapes` inlined for plugin; `GetComputeStream` fixed. | 5 |
| `tensor/pad.cc` | Plugin-local wrappers for `PadBase` static helpers. | 5C.2 |
| `tensor/slice.cc` | Plugin-local wrappers for `SliceBase::PrepareForCompute`/`FlattenOutputDims`. | 5C.3 |
| `math/variadic_elementwise_ops.cc` | Adapter `InputCount`/`RequiredInput`/`RequiredOutput` supported. | 5C |
| `math/matmul.cc` | `GetComputeStream` fixed; `GetTuningContext` guarded. | 5 |
| `math/matmul_integer.cc` | `GetComputeStream` fixed; `GemmInt8` signature updated. | 5 |
| `math/integer_gemm.cc` | `dynamic_cast<CudaStream*>` replaced with stream-based `GetCublasHandle()` overload (#27719). | 5 |
| `contrib/math/fused_matmul.cc` | Included after `matmul.cc` was fixed. | 5 |

## Priority Recommendations

### High Priority (Common ops, likely low effort)

These excluded ops use simple `GetComputeStream()` patterns that the adapter
already supports. They should be tried first. Ops marked with (✓) have had
their CPU helper dependencies fully refactored and are ready for build
verification:

- `contrib/bert/embed_layer_norm.cc` (✓ helper refactored #27617)
- `contrib/bert/fast_gelu.cc` (✓ helper refactored #27617)
- `contrib/bert/relative_attn_bias.cc`
- `contrib/bert/remove_padding.cc`
- `contrib/tensor/crop.cc` (✓ constructor templatized #27628)
- `contrib/tensor/dynamic_time_warping.cc`
- `contrib/tensor/dynamicslice.cc`
- `contrib/inverse.cc`
- `contrib/quantization/matmul_bnb4.cc`
- `contrib/quantization/matmul_nbits.cc`
- `generator/constant_of_shape.cc` (already has `#ifdef` path)

### Medium Priority (Moderate refactoring needed)

- `tensor/space_depth_ops.cc` — constructor templatized; remaining validation to inline
- `contrib/diffusion/group_norm.cc` — guard tuning context
- `contrib/moe/moe.cc` — fix stream access
- `contrib/fused_conv.cc` — fix stream access
- `contrib/math/fft_ops.cc` — fix stream access
- `contrib/math/gemm_float8.cc/.cu` — move stream access to `.cc`

### Low Priority (Significant effort or niche)

- `tensor/upsample.cc` + `tensor/resize.cc` — `AdjustOutputSizeAsPolicy` moved to header; `InputDefs()`/`GetAllocator()` still needed
- `rnn/*` — blocked on C API string array extension
- `llm/*` + `bert/attention*.cc` family — deep attention pipeline changes
- `math/einsum.cc` — complex CPU base class
- `object_detection/*` — base classes partially refactored; integration verification needed
- `transformers/*` — subgraph execution, very high effort

### Permanently Excluded

- `tensor/size.cc`, `tensor/shape_op.cc` — pure CPU ops
- `aten_ops/*` — PyTorch dependency
- `collective/*` — distributed runtime
- `tensor/shrunken_gather.cc` — training only
- Infrastructure files — replaced by `plugin/` equivalents
