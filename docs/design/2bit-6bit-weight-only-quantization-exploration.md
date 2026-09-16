# 2-Bit and 6-Bit Weight-Only Quantization Exploration

## Executive Summary

ONNX Runtime already has a substantial 2-bit foundation: the `MatMulNBits` model format, Python quantization tooling, CPU kernels, and correctness tests support 2-bit weights. The largest practical gap is CUDA `MatMulNBits`, whose execution and prepacking paths currently support only 4-bit and 8-bit weights.

The recommended first implementation target is therefore 2-bit CUDA `MatMulNBits`. A correctness-first dequantization plus cuBLAS fallback would establish end-to-end coverage before investing in fused GEMV and small-M kernels.

Six-bit support should begin as a format and value-validation spike. It is not a small extension of the existing implementation: current packing code assumes that the bit width divides eight, while 6-bit values cross byte boundaries. There is also no native INT6 tensor type or NVIDIA Tensor Core instruction. A portable 6-bit representation and execution-provider prepacking contract should be agreed upon before optimized kernels are implemented.

## Scope

This document focuses on weight-only block quantization for `MatMulNBits` and related embedding gather operations. It covers:

- Existing ONNX Runtime support and gaps.
- Relevant implementations in other inference and quantization projects.
- Model-format and packing choices.
- A staged implementation and evaluation plan.

Activation quantization, KV-cache quantization, and floating-point formats such as FP4 and FP6 are outside the primary scope.

## Current ONNX Runtime Support

### Support Matrix

| Component | 2-bit | 6-bit | Notes |
| --- | --- | --- | --- |
| `MatMulNBits` contrib schema | Yes | No | The schema attribute lists 2, 4, and 8 as supported values. |
| Python `MatMulNBits` quantizer | Yes | No | Dedicated 2-bit, 4-bit, and 8-bit native packers are exported. |
| CPU `MatMulNBits` | Yes | No | Includes fallback, AVX512/VNNI, and Arm64 work. |
| CUDA `MatMulNBits` | No | No | Kernel dispatch, prepacking, and memory estimation accept only 4 and 8 bits. |
| WebGPU `MatMulNBits` | No | No | Accepts only 4 and 8 bits. |
| CPU `GatherBlockQuantized` | Yes | No | Supports packed 2-bit, 4-bit, and 8-bit integer data. |
| CUDA `GatherBlockQuantized` | Yes | No | The generic packed-`uint8` path supports 2-bit extraction. |
| QDQ quantization helper | Limited | No | The dedicated native QDQ packing helper is currently 4-bit-oriented. |

### Existing 2-Bit Assets

The following pieces significantly reduce the cost and risk of a 2-bit CUDA implementation:

- `onnxruntime/core/graph/contrib_ops/contrib_defs.cc` defines the portable `MatMulNBits` inputs and block layout.
- `onnxruntime/python/onnxruntime_pybind_quant.cc` exports `quantize_matmul_2bits`.
- `onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py` creates 2-bit `MatMulNBits` models.
- `onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc` provides the CPU execution path.
- `onnxruntime/core/mlas/` contains optimized and fallback 2-bit implementations for supported CPU architectures.
- `onnxruntime/test/python/quantization/test_op_matmul_2bits.py` exercises model conversion and inference correctness.
- `onnxruntime/test/python/quantization/test_quantizeblockwise_2bits.py` verifies blockwise packing.
- `onnxruntime/test/mlas/unittest/test_sqnbitgemm_2bit_gemm.cpp` covers optimized MLAS behavior.

The CPU history shows that 2-bit support has moved beyond a prototype. It includes AVX512/VNNI kernels, Arm64 kernels, optimized dequantization, float zero-point handling, and fallback paths.

### CUDA Gap

CUDA `MatMulNBits` currently assumes either 4-bit or 8-bit weights in several places:

- Operator construction and dispatch.
- Blockwise dequantization specialization.
- Fused M=1 and small-M kernels.
- CUTLASS `fpA_intB` eligibility and weight conversion.
- Offline and runtime prepacked layouts.
- Workspace and persistent-memory estimates.

This means enabling 2-bit requires more than relaxing one validation check. However, the generic dequantize-to-floating-point plus cuBLAS path provides a narrow route to initial correctness without extending CUTLASS prepacked formats.

## External Landscape

### llama.cpp

llama.cpp has broad execution support for Q2_K, IQ2 variants, and Q6_K across CPU, CUDA, Metal, and additional backends. Its formats are not equivalent to the ONNX Runtime `MatMulNBits` format.

Q6_K stores each 6-bit code as two planes:

- Lower four bits: `QK_K / 2` bytes.
- Upper two bits: `QK_K / 4` bytes.
- Quantized sub-block scales plus a super-block scale.

This layout is friendly to SIMD and GPU extraction, but its hierarchical scaling and metadata produce an effective size greater than exactly six bits per weight. Current llama.cpp documentation reports approximately 6.56 bits per weight for Q6_K in representative models.

The main lesson for ONNX Runtime is that an optimized 6-bit implementation benefits from a deliberately designed physical layout. Q6_K should not be treated as a drop-in representation for uniform `MatMulNBits` quantization.

### AutoGPTQ

AutoGPTQ accepts 2-bit, 3-bit, 4-bit, and 8-bit GPTQ formats and contains legacy CUDA 2-bit matrix-vector kernels. Its most optimized modern paths, such as Marlin, remain 4-bit-specific. Mainline GPTQ packing does not provide a comparable production 6-bit path.

AutoGPTQ is useful as a reference for 2-bit bit extraction and small-batch execution, but its shape restrictions, layouts, and project status make direct reuse less attractive than extending the existing ONNX Runtime 4-bit/8-bit CUDA structure.

### TorchAO

TorchAO contains generic x-bit packing and CPU/MPS low-bit operators. Its MPS implementation explicitly packs four 6-bit weights into three bytes. It also demonstrates an important architectural pattern: portable or generic quantized values can be converted into opaque, backend-specific packed formats selected by the runtime.

Backend coverage and maturity vary by bit width. The most broadly optimized CUDA weight-only paths still center on 4-bit and 8-bit execution.

### TensorRT-LLM

TensorRT-LLM weight-only CUDA kernels and tests focus on INT4, INT8, and NVFP4. Its kernel type registry does not expose equivalent INT2 or INT6 weight-only paths. This is consistent with current NVIDIA hardware acceleration, which does not provide a native INT6 Tensor Core operation.

### ONNX Data Types

ONNX standardizes `INT2` and `UINT2`. Four consecutive 2-bit values are packed into one byte from least significant bits to most significant bits. This is compatible with the logical ordering already used by ONNX Runtime's packed 2-bit weights.

ONNX currently has floating-point 6-bit types, but no standard `INT6` or `UINT6` tensor type. `MatMulNBits` can still carry 6-bit codes in an opaque `uint8` blob, but the contrib operator must define the packing contract precisely.

## Design Considerations

### 2-Bit Packing

Two-bit packing naturally fits the current representation:

```text
packed = (x0 & 0x03)
       | ((x1 & 0x03) << 2)
       | ((x2 & 0x03) << 4)
       | ((x3 & 0x03) << 6)
```

No value crosses a byte boundary, and block sizes supported by `MatMulNBits` are multiples of four. The primary work is therefore execution-provider implementation and performance tuning rather than format design.

### 6-Bit Packing Options

#### Option A: Canonical Contiguous Bitstream

Pack four values into three bytes, with each value occupying six consecutive bits. This gives exactly six payload bits per weight and a portable representation.

Advantages:

- Minimal model size.
- A simple canonical definition independent of an execution provider.
- Similar to generic x-bit packing in other frameworks.

Disadvantages:

- Some values cross byte boundaries.
- Existing `8 / bits` pack and unpack logic cannot be reused.
- Direct kernel extraction requires additional shifts and merged loads.

#### Option B: Lower-4/Upper-2 Split Planes

Store low four bits and high two bits in separate regions, following the basic physical idea used by Q6_K.

Advantages:

- Efficient aligned loads and extraction.
- Easier reuse of 4-bit and 2-bit unpacking primitives.
- Better fit for SIMD, DP4A, and backend-specific prepacking.

Disadvantages:

- Requires a new explicitly documented layout.
- Is less naturally represented as one generic bitstream.
- Can complicate interoperability with generic quantization tools.

#### Recommended Contract

Use a canonical contiguous bitstream in the portable model and allow each execution provider to prepack it into an opaque optimized representation, such as lower-4/upper-2 planes. Do not change the interpretation of existing models based only on `bits=6`. The exact canonical ordering, padding, zero-point packing, and versioning behavior must be specified first.

## Proposed Roadmap

### Phase 0: Define Targets and Baselines

Before implementation, agree on:

- Target execution providers and GPU architectures.
- Target models and model sizes.
- Decode, small-batch, and prefill workloads.
- Required symmetric and asymmetric quantization support.
- Required block sizes, bias, `g_idx`, and data types.
- Accuracy and performance acceptance criteria.

### Phase 1: 2-Bit CUDA Correctness

1. Add a CUDA 2-bit blockwise dequantization kernel.
2. Route 2-bit `MatMulNBits` through dequantization plus cuBLAS.
3. Keep CUTLASS and offline-prepacked paths disabled for 2-bit initially.
4. Update workspace and memory estimation for the fallback path.
5. Add CUDA tests for symmetric and asymmetric zero points, bias, tails, and supported block sizes.
6. Add model-level quantization and inference tests using the existing Python packer.

This phase provides complete functionality and a stable reference for optimized kernels.

### Phase 2: 2-Bit CUDA Performance

1. Implement a fused M=1 GEMV path for token decode.
2. Implement or adapt a small-M batched kernel.
3. Evaluate an INT8 activation plus INT2 weight dot-product strategy.
4. Add runtime prepacking if profiling shows that the canonical layout limits load efficiency.
5. Tune dispatch thresholds against the dequantization plus cuBLAS fallback.

Pure 2-bit round-to-nearest quantization may not meet model-quality targets. GPTQ, HQQ, K-quant-inspired optimization, importance-aware quantization, and mixed 2-bit/4-bit layer assignment should be evaluated alongside kernel work.

### Phase 3: 6-Bit Format and Value Spike

1. Write a precise portable packing specification.
2. Implement Python pack and unpack reference functions.
3. Add CPU reference dequantization and correctness tests.
4. Quantize representative models with 4-bit, 6-bit, and 8-bit configurations.
5. Measure quality, model size, load time, and dequantization overhead.
6. Prototype contiguous and split-plane CUDA extraction with a microbenchmark.

The output of this phase should be a go/no-go decision for optimized MLAS and CUDA work.

### Phase 4: 6-Bit Optimized Execution

Proceed only if Phase 3 demonstrates a meaningful quality, memory, or latency niche that is not covered by mixed 4-bit/8-bit quantization.

Potential work includes:

- CPU SIMD dequantization and GEMM integration.
- CUDA fused decode GEMV.
- Small-M CUDA execution.
- Execution-provider-specific prepacking.
- Model conversion and compatibility tests.

## Evaluation Plan

### Correctness

- Compare operator output against explicit dequantization plus floating-point MatMul.
- Cover FP32, FP16, and BF16 where supported.
- Cover symmetric, asymmetric, omitted, packed, and floating-point zero points as applicable.
- Cover K and N tails, empty dimensions, multiple leading dimensions, and block sizes 16 through 256.
- Verify model serialization and execution-provider fallback behavior.

### Accuracy

At minimum, compare:

- 2-bit RTN, GPTQ, HQQ, and K-quant-inspired quantization.
- Uniform 2-bit versus mixed 2-bit/4-bit.
- Uniform 4-bit, 6-bit, and 8-bit.
- Perplexity plus representative downstream tasks used by the ONNX Runtime model-validation pipeline.

### Performance

Measure operator latency and end-to-end model performance for:

- M=1 decode.
- Small M values such as 2, 4, 8, 16, and 32.
- Prefill M values of 128 and above.
- Representative transformer K and N dimensions.
- Each target GPU architecture.

Report:

- Latency and throughput.
- Effective memory bandwidth.
- Peak and persistent memory.
- Runtime prepacking cost.
- Model load time.
- End-to-end tokens per second.

## Risks

- Two-bit quality may require mixed precision or more expensive calibration algorithms.
- A 2-bit CUDA kernel can become unpacking-bound and fail to outperform a mature 4-bit kernel.
- Six-bit saves only 25 percent of weight payload relative to 8-bit while requiring non-native unpacking.
- A poorly specified 6-bit format could create incompatible models across execution providers.
- Extending CUTLASS prepacked formats too early would increase scope before value is established.
- Metadata overhead can materially change the effective bits per weight, especially for small blocks.

## Recommended Initial Deliverables

### 2-Bit Track

- CUDA correctness fallback for `MatMulNBits`.
- Focused CUDA and Python tests.
- Decode and prefill benchmark baseline against 4-bit and CPU 2-bit.
- Accuracy report for uniform and mixed 2-bit models.

### 6-Bit Track

- Packing-format proposal.
- Python reference packer and CPU reference implementation.
- 4-bit/6-bit/8-bit quality and size comparison.
- CUDA unpack microbenchmark for contiguous and split-plane layouts.
- Go/no-go recommendation for optimized kernels.

## Recommendation to Management

Start implementation with 2-bit CUDA `MatMulNBits` because ONNX Runtime already has the portable format, quantizer, CPU implementation, and test foundation. Treat 6-bit as a separate format-and-value spike because it requires a new cross-provider packing contract and has no native Tensor Core path.

## References

- ONNX 2-bit integer types: https://onnx.ai/onnx/technical/int2.html
- ONNX Runtime `MatMulNBits` CUDA documentation: ../contrib_ops/cuda/matmul_nbits.md
- llama.cpp quantization documentation: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- llama.cpp Q6_K block definition: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
- TorchAO: https://github.com/pytorch/ao
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM