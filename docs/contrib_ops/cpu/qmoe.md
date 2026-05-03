# QMoE CPU Implementation Notes

This document describes the current CPU implementation of the `com.microsoft.QMoE` operator in ONNX Runtime.

## Scope

The CPU QMoE kernel is implemented in:

- `onnxruntime/contrib_ops/cpu/moe/moe_quantization_cpu.h`
- `onnxruntime/contrib_ops/cpu/moe/moe_quantization_cpu.cc`
- `onnxruntime/contrib_ops/cpu/moe/moe_helper.h`

The operator schema itself is defined in:

- `onnxruntime/core/graph/contrib_ops/contrib_defs.cc`

This document focuses on runtime behavior on the CPU Execution Provider, not on the general QMoE schema.

## High-Level Execution Flow

At a high level, the CPU kernel executes QMoE in five stages:

1. Validate input shapes and attributes.
2. Compute top-k expert routing from `router_probs` or `router_weights`.
3. Group routed tokens by expert.
4. Run expert-local FC1 -> activation -> FC2.
5. Scatter and accumulate weighted expert outputs back to the final output tensor.

The implementation keeps routing and accumulation shared across bit-widths. The main bit-width-specific differences are in how expert weights are prepared and how the FC1/FC2 GEMMs are executed.

## Supported Data Types and Weight Bit-Widths

### Activations and scales

- Activations/output (`T`): `float` or `MLFloat16`
- Scales (`T2`): `float` or `MLFloat16`
- Quantized weights (`T1`): `uint8`

For CPU, the kernel currently accepts:

- `expert_weight_bits = 2`
- `expert_weight_bits = 4`
- `expert_weight_bits = 8`

Internally, most CPU compute is performed in `float`. `MLFloat16` inputs such as activations, router values, scales, and biases are converted to `float` scratch buffers when needed.

## Supported Quantization Layouts

The CPU implementation supports both row-wise and block-wise quantization.

### Row-wise quantization

- `block_size = 0`
- One scale per output row
- Optional zero points are packed along the row dimension

Weight tensor shapes:

- FC1: `(num_experts, fc1_out_features, hidden_size / pack_size)`
- FC2: `(num_experts, hidden_size, inter_size / pack_size)`

### Block-wise quantization

- `block_size > 0`
- Quantization is along the K dimension of each GEMM
- One scale per output row and per block

Weight tensor shapes (same as row-wise — block-wise only changes scales):

- FC1: `(num_experts, fc1_out_features, hidden_size / pack_size)`
- FC2: `(num_experts, hidden_size, inter_size / pack_size)`

Scale tensor shapes:

- FC1: `(num_experts, fc1_out_features, hidden_size / block_size)`
- FC2: `(num_experts, hidden_size, inter_size / block_size)`

For packed weights:

- `pack_size = 8 / expert_weight_bits`
- 2-bit stores 4 values per byte
- 4-bit stores 2 values per byte
- 8-bit stores 1 value per byte

The CPU implementation validates that packed dimensions are compatible with `pack_size`, and also validates that `hidden_size` and inferred `inter_size` divide cleanly where required.

## Routing

Routing is shared for all bit-widths.

For each input token:

1. The kernel reads one row of `router_probs`.
2. It selects the top-`k` experts.
3. It computes aggregation weights:
   - from softmax of `router_probs`, or
   - directly from `router_weights` if that optional input is provided
4. It stores `(token, expert, weight)` assignments into per-expert route lists.

To reduce contention, the routing stage first builds thread-local expert-token maps and merges them afterward.

## Expert Execution

After routing, the kernel processes experts independently:

1. Gather the routed token activations for one expert into `A1`
2. Run FC1
3. Apply activation
4. Run FC2
5. Scatter-add weighted results into a thread-local output buffer

The final output tensor is produced by reducing all thread-local output buffers.

## FC1/FC2 Execution Paths

The CPU implementation uses multiple execution paths depending on bit-width, quantization style, and MLAS support.

### 2-bit path

The 2-bit path has a dedicated fast path for block-wise quantization using MLAS LUT GEMM.

#### 2-bit LUT GEMM fast path

This is used when:

- `expert_weight_bits == 2`
- quantization is block-wise
- the weight shape satisfies MLAS LUT GEMM constraints

As of the current MLAS implementation, this effectively requires:

- non-zero `block_size`
- `block_size` compatible with LUT GEMM
- K divisible by 32
- N divisible by 128 for the 2-bit kernel

The CPU kernel:

1. Detects whether FC1/FC2 can use LUT GEMM.
2. Uses prepacked LUT weights when available.
3. Otherwise packs the current expert's quantized weights, scales, and optional zero points into a thread-local LUT buffer.
4. Calls `MlasLutGemm` for FC1 and/or FC2.

The LUT-specific helper logic is intentionally isolated from the shared routing and accumulation flow.

#### 2-bit fallback path

If LUT GEMM is not available for a particular shape or layout, the kernel falls back to:

1. dequantize packed weights into `float`
2. run standard `MlasGemm`

This fallback supports both row-wise and block-wise quantization.

### 4-bit path

The 4-bit path supports several modes:

#### Direct MLAS Q4 GEMM fast path

When the configuration is compatible, the kernel can use MLAS Q4 GEMM directly.

This path is used only for symmetric 4-bit weights where the MLAS Q4 layout is supported.

#### Prepacked transposed fallback

If direct Q4 GEMM is not used, the kernel can reuse prepacked transposed/unpacked buffers and then run:

1. dequantize from the prepacked layout
2. `MlasGemm`

#### General fallback

Otherwise, the kernel dequantizes directly from the packed ONNX input tensors and runs `MlasGemm`.

### 8-bit path

The 8-bit path currently uses the general dequantize-plus-GEMM flow. It shares the same routing, activation, and output accumulation logic as the other paths.

## PrePack Behavior

The CPU kernel implements `PrePack()` and `UseSharedPrePackedBuffers()` to reduce repeated setup cost.

### 2-bit prepack

When FC1/FC2 weights and scales, plus any zero points, are constant initializers and the block-wise shape is
supported by MLAS LUT GEMM, the kernel pre-packs the weights into MLAS LUT GEMM packed buffers. These packed
buffers are cached and can be reused across sessions through shared prepacked buffers. If scales or zero points are
runtime inputs, execution falls back to packing per expert during `Compute()`.

### 4-bit prepack

The kernel pre-packs 4-bit weights into a transposed/unpacked format used by the fallback path. When possible, it also creates a direct MLAS Q4 packed cache for faster execution.

### 8-bit prepack

There is no special MLAS packed path today analogous to the 2-bit LUT or 4-bit Q4 paths.

## Activation Handling

The CPU QMoE kernel currently supports:

- `identity`
- `gelu`
- `relu`
- `silu`
- `swiglu`

For CPU, `SwiGLU` requires `swiglu_fusion = 1`, meaning FC1 output is interpreted as interleaved gate/value data and activation writes directly into the FC2 input buffer.

## Bias Handling

Biases are optional for FC1 and FC2.

- If the fast GEMM path does not apply bias internally, the kernel converts bias to `float` and applies it afterward.
- FC2 bias is added during the final scatter-add stage when it was not already fused into the GEMM path.

## Zero Points

Optional zero-point tensors are supported for FC1 and FC2.

- 2-bit and 4-bit zero points are packed
- block-wise zero points are packed per row and per K-block
- row-wise zero points are packed per row

The direct 4-bit MLAS Q4 path only supports symmetric weights, so it is not used when zero points are present.

The 2-bit LUT path supports both symmetric and asymmetric packed inputs as long as the MLAS LUT requirements are satisfied.

## Threading Model

The CPU implementation uses the ORT thread pool in several phases:

- routing
- per-expert token gather
- block dequantization
- activation
- final accumulation

It avoids global write contention by accumulating expert outputs into thread-local output buffers and reducing them at the end.

## Current CPU Limitations

The current CPU QMoE implementation has a few important limitations:

- FC3 gating path is not implemented on CPU for QMoE
- the 2-bit fast path is limited to MLAS LUT-supported block-wise shapes
- the direct 4-bit MLAS path only applies to supported symmetric layouts
- some paths still dequantize into `float` scratch buffers before GEMM, which is simpler but slower than a fully native low-bit kernel

## Code Structure Summary

- `moe_helper::CheckInputs(...)`
  - central input validation and shape inference
- `QMoECPU<T>::PrePack(...)`
  - builds reusable weight caches
- `QMoECPU<T>::UseSharedPrePackedBuffers(...)`
  - restores shared prepacked buffers
- `QMoECPU<T>::Compute(...)`
  - shared validation and dispatch into the common execution flow
- `TryRunLutGemm(...)`
  - isolates the 2-bit LUT pack-and-run fast path

## Tests

Relevant CPU-side tests live in:

- `onnxruntime/test/contrib_ops/moe_test.cc`
- `onnxruntime/test/python/transformers/test_qmoe_cpu.py`

These cover:

- 2-bit and 4-bit CPU execution
- row-wise and block-wise quantization
- validation failures
- non-trivial numeric correctness cases
- the 2-bit LUT-eligible block-wise identity case

## Notes for Future Work

Likely future improvements include:

- more aggressive native low-bit CPU GEMM support beyond the current LUT/Q4 fast paths
- broader prepacking support for additional 2-bit and 8-bit cases
- reducing temporary dequantization buffers on fallback paths
- extending CPU support for FC3 gating if needed
