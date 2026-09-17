# 2-Bit and 6-Bit Weight-Only Quantization Exploration

## Executive Summary

ONNX Runtime already has a substantial 2-bit foundation: the `MatMulNBits` model format, Python quantization tooling, CPU kernels, and correctness tests support 2-bit weights. This makes INT2 the shortest path to extending CUDA weight-only execution without introducing a new portable format.

The recommended first implementation target is therefore 2-bit CUDA `MatMulNBits`. The initial CUDA work should compare direct packed-INT2 execution with a GPU-native LUT approach, then integrate the best path for both M=1 decode and representative large-M prefill. This is required for a GitHub Copilot-style workload: M=1 decode affects token-generation speed, while long-context prefill affects time to first token (TTFT).

This choice prioritizes implementation readiness and maximum memory-bandwidth reduction while accepting material model-quality risk. The first week must establish uniform and mixed-precision INT2 quality baselines against INT4, INT8, and BF16. Mixed INT2/INT4/INT8 quantization may be required for sensitive layers. INT6 remains a follow-up option if INT2 cannot meet coding and tool-calling quality targets or if a less aggressive quality/size tradeoff is needed.

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
| WebGPU `MatMulNBits` | Yes | No | Supports 2-bit unpacking with symmetric and asymmetric test coverage. |
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

This means enabling 2-bit requires more than relaxing one validation check. The implementation should preserve the portable model layout while allowing a CUDA-specific runtime prepack. It should execute packed weights directly rather than materializing a floating-point weight matrix. The CPU implementation remains the semantic reference, but its AVX/VNNI and LUT layouts should not be copied literally because CUDA requires different memory coalescing, register, shared-memory, and occupancy tradeoffs.

### Existing M-Dependent CUDA Dispatch

The existing INT4 and INT8 `MatMulNBits` CUDA implementations already select different execution paths according to the runtime row count M, although their fused-kernel coverage is not identical:

| Weight type | M=1 | Small M | Larger M |
| --- | --- | --- | --- |
| INT4 | Dedicated fused GEMV | Fused batched/small-M kernels for M=2-16, subject to data type, shape, block-size, and shared-memory constraints | The FP16/BF16 `fpA_intB` path can select a CUTLASS weight-only GEMM; otherwise execution falls back to dequantization followed by cuBLAS |
| INT8 | Dedicated fused GEMV | Fused batched kernels for M=2-5, subject to shape and block-size constraints | The FP16/BF16 `fpA_intB` path can select a CUTLASS weight-only GEMM; otherwise execution falls back to dequantization followed by cuBLAS |

For eligible prepacked FP16/BF16 configurations, the `fpA_intB` profiler selects between its CUDA GEMV and CUTLASS GEMM tactics using the actual M bucket rather than a single fixed threshold. Consequently, the table describes the current specialized direct-kernel coverage, not a universal three-way dispatcher that applies to every data type and shape.

The proposed INT6 implementation should preserve this M-sensitive architecture: a fused GEMV for M=1 decode, a tiled fused GEMM for representative large-M prefill, and eventually a separate small-M path. It cannot simply reuse the existing INT4/INT8 CUTLASS kernels because NVIDIA hardware and the current CUTLASS integration do not expose a native INT6 weight-only operation. INT6 therefore also requires a new lower-4/upper-2 prepack, extraction logic, and fused compute kernels. Unsupported INT6 configurations should retain a correctness fallback, but that fallback is not a performance milestone.

## External Landscape

### llama.cpp

llama.cpp has broad execution support for Q2_K, IQ2 variants, and Q6_K across CPU, CUDA, Metal, and additional backends. Its formats are not equivalent to the ONNX Runtime `MatMulNBits` format.

#### Qwen3.8-Flash-Next Published 2-Bit Baseline

The published Unsloth `UD-Q2_K_XL` GGUF for Qwen3.8-Flash-Next is approximately 78.9 GB, but it is neither uniformly Q2_K nor a stock llama.cpp quantization preset. Its metadata labels the model `MOSTLY_Q2_K` while assigning formats per tensor using an importance matrix with 926 entries derived from 45 calibration chunks. The inspected assignments include:

- Expert gate/up tensors: `IQ2_XS` in 47 layers and `IQ3_XXS` in layer 2.
- Expert down tensors: `IQ4_NL`.
- Attention projections: primarily `Q5_K` and `Q6_K`.
- Token and output embeddings: `Q5_K` and `Q4_K`.
- N-gram embedding: `IQ4_NL`.
- Hyper-connection matrices: `Q8_0`.
- Sparse-attention indexer Q/K tensors: BF16.
- Norms and small control tensors: primarily F32.

`IQ2_XS` is a nominal 2-bit importance-quantized format with approximately 2.31 effective bits per weight after its scales and indexing metadata. It is not the same numerical format as blockwise affine `MatMulNBits(bits=2)`. Consequently, "2-bit" describes the lowest and dominant expert tier of this GGUF, not a two-bit average across every parameter.

The first ONNX Runtime model-level experiment should nevertheless target the same tensor placement: use `MatMulNBits(bits=2)` for the 47 expert gate/up tensor groups, retain layer 2 and the remaining sensitive tensor classes at supported higher precision, and then measure quality and effective model size. This tests whether expert gate/up INT2 captures most of the useful compression and bandwidth reduction without claiming bit-exact equivalence to `IQ2_XS`. Comparisons with `UD-Q2_K_XL` must report tensor-type distribution and effective bits per parameter rather than comparing quantization names alone.

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

### Phase 1: 2-Bit Quality and Kernel Gate

1. Freeze representative Qwen shapes, workloads, quality metrics, and performance baselines.
2. Validate the existing portable INT2 packing and CPU implementation as the CUDA semantic reference.
3. Measure uniform and mixed INT2/INT4/INT8 coding, tool-calling, KL-divergence, and effective model size, starting with `MatMulNBits(bits=2)` on the 47 expert gate/up tensor groups and higher precision elsewhere.
4. Prototype direct packed-INT2 and GPU-native LUT extraction for M=1 decode.
5. Prototype a tiled direct or LUT-based path for representative large-M prefill.
6. Select the CUDA execution and runtime-prepacking strategy using measured quality and performance data.

The first week is a quality and workload gate, not a stop condition for all INT2 engineering. If uniform INT2 misses model-quality thresholds, the implementation should proceed with a mixed-precision recipe that preserves sensitive layers at INT4, INT8, or BF16. The CUDA kernel decision must be based on end-to-end value rather than unpack throughput alone.

#### Required Deliverables

- Uniform and mixed-precision INT2 quality and effective-size results on the agreed Qwen coding-model workload.
- An expert gate/up INT2 model variant compared with the approximately 78.9 GB Unsloth `UD-Q2_K_XL` baseline, including tensor-type distribution and effective bits per parameter.
- Direct-unpack versus GPU-native LUT microbenchmarks for representative M=1 decode and large-M prefill shapes.
- A selected CUDA runtime-prepacking and kernel strategy, including memory overhead and architecture constraints.
- A written assessment of whether INT2 delivers useful end-to-end decode, TTFT, and memory improvements over INT4.

#### Go/No-Go Criteria

- **Quality:** A uniform or mixed-precision INT2 recipe meets agreed coding and tool-calling thresholds.
- **Size:** Effective model size, including higher-precision layers and metadata, remains materially below INT4.
- **CUDA value:** Fused M=1 and large-M prototypes show credible decode and prefill gains over INT4 without becoming dominated by unpacking, LUT, or occupancy costs.
- **Complexity:** Runtime prepacking, kernel coverage, and maintenance cost are justified by end-to-end model improvements.

### Phase 2: Scoped 2-Bit CUDA Delivery

Integrate the selected approach into `MatMulNBits(bits=2)` with:

- Native fused M=1 decode GEMV and representative large-M prefill GEMM for FP16 activations, symmetric weights, and one selected block size.
- Runtime prepacking where it provides a measured benefit while preserving the existing portable INT2 model layout.
- Correctness checks against the existing CPU implementation and explicit dequantization.
- Focused Qwen-shape performance tests and end-to-end decode throughput and TTFT measurements.

### Phase 3: 2-Bit Production Expansion

After the scoped delivery, expand to BF16, additional block sizes, asymmetric zero points, bias, tails, intermediate/small-M execution, offline prepacking, and broader GPU tuning as justified by measured demand.

### Phase 4: 6-Bit Follow-Up Gate

Evaluate INT6 if INT2 cannot achieve the required quality/size tradeoff or product requirements call for a less aggressive quantization option. Reuse the contiguous portable format and lower-4/upper-2 prepacking analysis in this document, but require a separate format review and measured advantage over mixed INT4/INT8 before implementation.

## Schedule Estimate

These estimates assume one engineer working full time with Copilot assistance, timely access to representative Ampere, Ada, and Hopper GPUs, and reusable ONNX exports for the target Qwen models. They include implementation, profiling, tests, documentation, and normal review fixes, but not unpredictable CI queue or external model-conversion blockers.

### 2-Bit CUDA `MatMulNBits`

| Work item | Estimate |
| --- | ---: |
| Qwen shape inventory, quality baselines, benchmark harness, and CPU/reference validation | 1 week |
| Direct and GPU-native LUT M=1 INT2 GEMV prototypes and profiling | 1-2 weeks |
| Direct and LUT-based large-M INT2 GEMM prototypes and profiling | 2 weeks |
| ORT integration for the primary symmetric FP16 configuration and selected block size | 1 week |
| Focused correctness tests, end-to-end measurements, tuning, and PR cleanup | 2-3 weeks |

Allow **7-9 engineering weeks** for the scoped M=1 decode and large-M prefill vertical slice under the primary configuration. Broader data types, block sizes, asymmetric quantization, small-M kernels, and multi-architecture tuning would extend the work beyond this initial delivery. Copilot reduces coding and test-authoring time, but it does not remove hardware profiling, kernel tuning, model-quality evaluation, or code-review time.

### 6-Bit `MatMulNBits`

| Work item | Estimate |
| --- | ---: |
| Portable format specification and schema/tooling design | 1-2 weeks |
| Python pack/unpack, CPU reference, tests, and 4/6/8-bit quality study | 2-3 weeks |
| CUDA contiguous-versus-split-plane extraction and fused M=1/large-M prototypes | 3-4 weeks |
| Go/no-go analysis and design review | 1 week |

Allow **6-8 engineering weeks** for the complete 6-bit format-and-value study and an evidence-based final go/no-go decision when both decode and prefill prototypes are required. The format, quality experiments, reference implementation, and scoped CUDA work can overlap to target an earlier vertical slice. A production-ready CUDA 6-bit track with broad operator coverage remains approximately **11-17 engineering weeks total**. Adding optimized CPU/MLAS support or another execution provider would require separate estimates.

## November 15 Delivery Plan

There are approximately 8.5 calendar weeks from September 16 to November 15, 2026. With one engineer, the committed delivery should be a scoped INT2 CUDA vertical slice built on the existing portable format, Python tooling, and CPU reference implementation. The target is reviewable native fused CUDA execution for M=1 decode and representative large-M prefill under one primary symmetric FP16 configuration. This is narrower than complete production INT2 support. Upstream merge by November 15 cannot be guaranteed because review and CI timing are outside the implementation owner's control.

### Committed INT2 Scope

- Uniform and mixed INT2/INT4/INT8 quality and effective-size results for representative Qwen3.8 coding-model workloads.
- Direct packed-INT2 versus GPU-native LUT prototype results and a selected CUDA strategy.
- CUDA runtime prepacking if justified by profiling, plus fused M=1 GEMV and fused large-M GEMM for FP16 activations, symmetric weights, and one selected block size.
- Correctness and performance results for M=1 decode and representative prefill M values, such as 128, 512, and 2048, on Qwen3.8-27B and Qwen3.8-Flash-Next matrix shapes.
- End-to-end decode throughput and TTFT measurements for a Copilot-style long-context workload.
- ORT integration, focused tests, documentation, and a reviewable pull request or draft pull request, depending on review readiness.

### Schedule

| Dates | Milestone |
| --- | --- |
| September 16-20 | Freeze quality thresholds, Qwen workloads, candidate block sizes, and INT4/INT8/BF16 baselines. |
| September 21-27 | Run uniform and mixed-precision INT2 quality/size experiments; validate the existing portable format and CPU reference. |
| September 28-October 11 | Prototype and compare direct packed-INT2 and GPU-native LUT paths for M=1 decode and large-M prefill. |
| October 12-25 | Select the kernel/prepack strategy; implement fused M=1 GEMV and large-M GEMM for the primary FP16 configuration. |
| October 26-November 1 | Integrate both paths into CUDA `MatMulNBits(bits=2)` and validate representative Qwen decode and prefill shapes. |
| November 2-8 | Add focused correctness tests, serialization, memory estimates, build integration, and performance measurements. |
| November 9-15 | Regression testing, documentation, final quality/performance report, PR cleanup, and review buffer. |

### Stretch Scope

The following items should not put the November 15 commitment at risk:

- Asymmetric zero points.
- BF16 activation support.
- Additional block sizes beyond the selected primary configuration.
- Bias and tail combinations not already covered by the selected kernel path.
- Optimized intermediate-M execution for M values between the committed decode and prefill ranges.
- Offline CUDA-specific prepacking.
- Broad multi-GPU tuning.
- INT6 format, tooling, or kernel implementation.

### INT6 Scheduling Impact

With the same engineer, INT6 implementation should not run concurrently if it threatens the INT2 deadline. Before November 15, INT6 work should be limited to preserving the format analysis and collecting quality data that directly informs the INT2 comparison. A second engineer could run the INT6 quality and format gate independently.

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

### 2-Bit Priority Track

- Accuracy and effective-size report for uniform and mixed 2-bit/4-bit/8-bit/BF16 coding-model recipes.
- Direct-unpack and GPU-native LUT comparison on Qwen3.8 decode and prefill shapes.
- CUDA M=1 GEMV and representative large-M GEMM vertical slice for the primary symmetric FP16 configuration.
- Qwen-shape decode, prefill, and TTFT correctness/performance report.
- Go/no-go recommendation for broader production investment.

### 6-Bit Follow-Up Track

- Preserve the portable contiguous and lower-4/upper-2 prepacking design analysis.
- Run a 4-bit/6-bit/8-bit quality and effective-size comparison if INT2 quality is insufficient.
- Require a separate format and value gate before native CUDA implementation.

## Recommendation to Management

Prioritize INT2 because ONNX Runtime already has a portable model format, quantization tooling, CPU kernels, and correctness coverage, and because INT2 offers the largest potential weight-memory and bandwidth reduction. Use the first week to freeze quality thresholds and identify a viable uniform or mixed-precision recipe, then target a scoped CUDA `MatMulNBits(bits=2)` vertical slice by November 15: direct-versus-LUT evidence, fused M=1 decode, and fused representative large-M prefill for the primary symmetric FP16 configuration. Both execution paths are required for a GitHub Copilot-style workload because decode determines generation speed and prefill determines TTFT for long repository context.

This November scope is not complete production INT2 support. Broader data types, block sizes, asymmetric quantization, small-M execution, offline prepacking, and multi-architecture tuning remain follow-up work. Keep INT6 as the next quality-oriented option if INT2 cannot meet the agreed coding and tool-calling targets at a meaningful effective-size advantage over INT4.

## References

- ONNX 2-bit integer types: https://onnx.ai/onnx/technical/int2.html
- ONNX Runtime `MatMulNBits` CUDA documentation: ../contrib_ops/cuda/matmul_nbits.md
- llama.cpp quantization documentation: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- llama.cpp Q6_K block definition: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
- TorchAO: https://github.com/pytorch/ao
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM