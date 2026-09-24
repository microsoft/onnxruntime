# 2-, 3-, 5-, and 6-Bit Weight-Only Quantization Exploration

## Executive Summary

ONNX Runtime already has a substantial 2-bit foundation: the `MatMulNBits` model format, Python quantization tooling, CPU kernels, and correctness tests support 2-bit weights. This makes INT2 the shortest path to extending CUDA weight-only execution without introducing a new portable format.

The current product priority is mixed-width INT2 QMoE for expert-heavy Qwen models: INT2 expert gate/up projections with INT4 expert down projections. The end-to-end contract, export, runtime, validation, and staffing plan is defined in [INT2 QMoE End-to-End Delivery Plan](int2-qmoe-end-to-end-delivery-plan.md). Dense CUDA `MatMulNBits(bits=2)` remains valuable as a source of reusable packing, validation, and kernel primitives, but complete dense GEMV/GEMM delivery should not block the QMoE path.

This choice prioritizes implementation readiness and maximum memory-bandwidth reduction while accepting material model-quality risk. The first week must establish uniform and mixed-precision INT2 quality baselines against INT4, INT8, and BF16. Mixed INT2/INT4/INT8 quantization may be required for sensitive layers. INT3, INT5, and INT6 remain follow-up quality/size options if INT2 cannot meet coding and tool-calling targets or if a less aggressive quantization point is needed.

## Scope

This document focuses on weight-only block quantization for `MatMulNBits` and related embedding gather operations. It covers:

- Existing ONNX Runtime support and gaps.
- Relevant implementations in other inference and quantization projects.
- Model-format and packing choices.
- A staged implementation and evaluation plan.

The dense CUDA vertical slice described in this exploration remains a supporting implementation option for independently exported projections. The prioritized fused QMoE work and its packed expert-weight contract are scoped separately in the end-to-end QMoE delivery plan.

Activation quantization, KV-cache quantization, floating-point formats such as FP4 and FP6, and fused MoE/QMoE execution are outside the primary scope.

## Current ONNX Runtime Support

### Support Matrix

| Component | 2-bit | 3-bit | 5-bit | 6-bit | Notes |
| --- | --- | --- | --- | --- | --- |
| `MatMulNBits` contrib schema | Yes | No | No | No | The schema attribute lists 2, 4, and 8 as supported values. |
| Python `MatMulNBits` quantizer | Yes | No | No | No | Dedicated 2-bit, 4-bit, and 8-bit native packers are exported. |
| CPU `MatMulNBits` | Yes | No | No | No | Includes fallback, AVX512/VNNI, and Arm64 work for 2-bit. |
| CUDA `MatMulNBits` | No | No | No | No | Kernel dispatch, prepacking, and memory estimation accept only 4 and 8 bits. |
| WebGPU `MatMulNBits` | Yes | No | No | No | Supports 2-bit unpacking with symmetric and asymmetric test coverage. |
| CPU `GatherBlockQuantized` | Yes | No | No | No | Supports packed 2-bit, 4-bit, and 8-bit integer data. |
| CUDA `GatherBlockQuantized` | Yes | No | No | No | The generic packed-`uint8` path supports 2-bit extraction. |
| QDQ quantization helper | Limited | No | No | No | The dedicated native QDQ packing helper is currently 4-bit-oriented. |

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

Any selected INT3, INT5, or INT6 implementation should preserve this M-sensitive architecture: a fused GEMV for M=1 decode, a tiled fused GEMM for representative large-M prefill, and eventually a separate small-M path. These widths cannot simply reuse the existing INT4/INT8 CUTLASS kernels because NVIDIA hardware and the current CUTLASS integration do not expose corresponding native weight-only operations. They therefore require width-aware runtime prepacking, extraction logic, and fused compute kernels. Unsupported configurations should retain a correctness fallback, but that fallback is not a performance milestone.

### INT2 QMoE Product Assessment

INT2 QMoE is a meaningful follow-up because expert weights dominate the storage and memory traffic of large MoE models, and a fused operator preserves top-k routing instead of expanding every expert into independently scheduled dense operations. CPU QMoE already accepts blockwise INT2 and includes an MLAS LUT GEMM path, so it can provide a semantic reference. However, the current cross-provider implementation is not production-ready:

| Area | Current state | Required work |
| --- | --- | --- |
| QMoE schema | Independent FC1, FC2, and FC3 widths merged in [#32697](https://github.com/microsoft/onnxruntime/pull/32697) | Maintain backward compatibility and add model-level conformance coverage |
| CPU QMoE | Accepts blockwise INT2 and has an optimized LUT path | Add mixed-width semantics and model-level conformance coverage |
| CUDA QMoE | Bounded INT2/mixed-width correctness fallback merged in [#32743](https://github.com/microsoft/onnxruntime/pull/32743); packed decode is in review in [#32761](https://github.com/microsoft/onnxruntime/pull/32761) | Complete review, benchmark packed decode, and add a bounded or native prefill path |
| WebGPU QMoE | Rejects INT2 and uses a 4/8-bit-specific pack-size calculation | Use `8 / bits`, complete reachable INT2 shader support, and add QMoE tests |
| Model production | Dense mixed-bit export is the initial Olive/Mobius target | Define and qualify a distinct fused QMoE graph and weight-binding contract |

The mixed-width model contract is no longer the primary blocker. QMoE now supports independent FC-specific width attributes with inheritance from `expert_weight_bits`, including the target `fc1_expert_weight_bits=2` and `fc2_expert_weight_bits=4` recipe. The remaining product blockers are fused Olive/Mobius model production, CPU and CUDA model-level parity, packed-decode performance evidence, bounded or native prefill, and quality qualification on the target Qwen model.

CUDA dequantization to persistent FP16/BF16 expert weights is useful only as a correctness oracle because it expands INT2 payloads by approximately 8x and removes the deployment memory benefit. The first performance-relevant QMoE target should be packed INT2 fused decode for small expanded-row counts. Long-context prefill ultimately requires a native or equivalently bounded W2A16 grouped GEMM; full expert dequantization is not a production milestone.

Shared MatMulNBits INT2 packing, CPU-reference, validation, and CUDA load/dequantization primitives remain early enabling work because they reduce QMoE implementation risk. QMoE contract and export work should proceed in parallel and must not wait for complete dense MatMulNBits coverage across every data type, block size, or M regime.

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

The first ONNX Runtime model-level experiment should nevertheless target the same tensor placement: use `MatMulNBits(bits=2)` for the 47 expert gate/up tensor groups when they are exported as independent dense projections, retain layer 2 and the remaining sensitive tensor classes at supported higher precision, and then measure quality and effective model size. This tests whether expert gate/up INT2 captures most of the useful compression and bandwidth reduction without claiming bit-exact equivalence to `IQ2_XS`. Comparisons with `UD-Q2_K_XL` must report tensor-type distribution and effective bits per parameter rather than comparing quantization names alone.

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

TensorRT-LLM weight-only CUDA kernels and tests focus on INT4, INT8, and NVFP4. Its kernel type registry does not expose equivalent INT2, INT3, INT5, or INT6 weight-only paths. This is consistent with current NVIDIA hardware acceleration, which does not provide native INT3, INT5, or INT6 Tensor Core operations.

### ONNX Data Types

ONNX standardizes `INT2` and `UINT2`. Four consecutive 2-bit values are packed into one byte from least significant bits to most significant bits. This is compatible with the logical ordering already used by ONNX Runtime's packed 2-bit weights.

ONNX does not currently provide standard `INT3`, `UINT3`, `INT5`, `UINT5`, `INT6`, or `UINT6` integer tensor types. `MatMulNBits` can still carry these codes in an opaque `uint8` blob, but the contrib operator must define the packing contract precisely.

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

### 3-, 5-, and 6-Bit Packing Options

All three widths cross byte boundaries under a compact representation. A common packing quantum of eight logical values gives:

| Width | Values per quantum | Payload bytes | Candidate runtime bit planes |
| ---: | ---: | ---: | --- |
| 3 | 8 | 3 | Lower 2 bits plus upper 1 bit |
| 5 | 8 | 5 | Lower 4 bits plus upper 1 bit |
| 6 | 8 | 6 | Lower 4 bits plus upper 2 bits |

The portable format should not pad INT3 to four bits or INT5/INT6 to eight bits because that would erase much of the intended model-size benefit.

#### Option A: Canonical Contiguous Bitstream

Pack each value into a little-endian, least-significant-bit-first contiguous bitstream. Logical value `i` begins at bit offset `i * bits`; any unused tail bits in the final byte are zero. This gives exactly three, five, or six payload bits per weight before block metadata and padding.

Advantages:

- Minimal model size.
- A simple canonical definition independent of an execution provider.
- Similar to generic x-bit packing in other frameworks.

Disadvantages:

- Some values cross byte boundaries.
- Existing `8 / bits` pack and unpack logic cannot be reused.
- Direct kernel extraction requires additional shifts and merged loads.

#### Option B: Backend-Specific Split Planes

Prepack the portable bitstream into aligned planes selected by the execution provider: lower-2/upper-1 for INT3, lower-4/upper-1 for INT5, and lower-4/upper-2 for INT6. The INT6 layout follows the basic physical idea used by Q6_K without adopting its hierarchical quantization format.

Advantages:

- Efficient aligned loads and extraction.
- Easier reuse of 1-bit, 2-bit, and 4-bit unpacking primitives.
- Better fit for SIMD, DP4A, and backend-specific prepacking.

Disadvantages:

- Requires a new explicitly documented layout.
- Is less naturally represented as one generic bitstream.
- Can complicate interoperability with generic quantization tools.

#### Recommended Contract

Use one canonical contiguous-bitstream rule for `bits=3`, `bits=5`, and `bits=6`, and allow each execution provider to prepack it into an opaque optimized representation. Do not change the interpretation of existing models based only on a new `bits` value. The exact ordering, row and block boundaries, tail padding, zero-point packing, and versioning behavior must be specified before enabling any new width.

## Proposed Roadmap

The dense `MatMulNBits` phases below describe supporting technology and independent projection coverage. The active QMoE product roadmap, milestones, and acceptance criteria are maintained in the separate end-to-end QMoE delivery plan.

### Phase 0: Define Targets and Baselines

Before implementation, agree on:

- Target execution providers and GPU architectures.
- Target models and model sizes.
- Decode, small-batch, and prefill workloads.
- Required symmetric and asymmetric quantization support.
- Required block sizes, bias, `g_idx`, and data types.
- Accuracy and performance acceptance criteria.

### Initial Model-Production Contract

The required model-production workflow for the initial CUDA vertical slice is:

```text
HF/PyTorch model
       -> Olive RTN, GPTQ, or SelectiveMixedPrecision INT2/mixed checkpoint
       -> Mobius export
       -> dense MatMulNBits(bits=2/4/8) ONNX model
       -> ONNX Runtime CUDA execution
```

This path best matches the GPTQ, selective mixed-precision, and Qwen quality goals. The vertical slice must qualify graph conversion, initializer binding, tensor-wise bit-width selection, and numerical parity from the Olive checkpoint through Mobius export and ONNX Runtime execution.

The ONNX-native workflow, `FP ONNX -> Olive OnnxBlockWiseRtnQuantization -> MatMulNBits(bits=2)`, remains required for broader product support but follows the initial PyTorch/Mobius milestone. Complete Olive ONNX RTN and built-in quantized-linear INT2 export support must be tracked with the owning Olive work rather than assumed to exist because the ONNX Runtime kernel is available.

The initial contract produces dense `MatMulNBits` nodes only. Exporting expert tensors into a fused MoE/QMoE operator would require a separate Olive/Mobius/runtime schema, packing, weight-binding, kernel, and parity contract and is not part of this CUDA delivery.

### Phase 1: 2-Bit Quality and Kernel Gate

1. Freeze representative Qwen shapes, workloads, quality metrics, and performance baselines.
2. Validate the existing portable INT2 packing and CPU implementation as the CUDA semantic reference.
3. Measure uniform and mixed INT2/INT4/INT8 coding, tool-calling, KL-divergence, and effective model size, starting with `MatMulNBits(bits=2)` on independently exported expert gate/up projections and higher precision elsewhere.
4. Qualify the Olive checkpoint-to-Mobius-to-ONNX path for graph structure, initializer binding, tensor-wise bit widths, and numerical parity.
5. Prototype direct packed-INT2 and GPU-native LUT extraction for M=1 decode.
6. Prototype a tiled direct or LUT-based path for representative large-M prefill.
7. Select the CUDA execution and runtime-prepacking strategy using measured quality and performance data.

The first week is a quality and workload gate, not a stop condition for all INT2 engineering. If uniform INT2 misses model-quality thresholds, the implementation should proceed with a mixed-precision recipe that preserves sensitive layers at INT4, INT8, or BF16. The CUDA kernel decision must be based on end-to-end value rather than unpack throughput alone.

#### Required Deliverables

- Uniform and mixed-precision INT2 quality and effective-size results on the agreed Qwen coding-model workload.
- An expert gate/up INT2 model variant compared with the approximately 78.9 GB Unsloth `UD-Q2_K_XL` baseline, including tensor-type distribution and effective bits per parameter.
- An end-to-end mixed INT2/INT4/INT8 model exported through Olive and Mobius with graph, weight-binding, and numerical-parity coverage.
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
- A deployable dense `MatMulNBits` model produced through the qualified PyTorch/Olive-to-Mobius workflow.

### Phase 3: 2-Bit Production Expansion

After the scoped delivery, expand to BF16, additional block sizes, asymmetric zero points, bias, tails, intermediate/small-M execution, offline prepacking, and broader GPU tuning as justified by measured demand.

### Phase 4: Mixed-Width INT2 QMoE Product Track

Execute the prioritized QMoE work according to the standalone end-to-end delivery plan. Reuse MatMulNBits INT2 packing, validation, and CUDA extraction primitives as they stabilize, but do not gate the QMoE schema or Olive/Mobius export work on complete dense kernel coverage:

1. Define independent FC1 and FC2 bit-width semantics and preserve compatibility with the existing single-width QMoE contract.
2. Add CPU correctness and model-level tests for FC1 INT2 with FC2 INT4.
3. Qualify Olive/Mobius fused QMoE export, graph binding, and numerical parity separately from dense `MatMulNBits` export.
4. Implement CUDA packed INT2 QMoE decode by reusing validated INT2 extraction and dequantization primitives.
5. Implement or evaluate a bounded native W2A16 grouped-GEMM path for prefill.

WebGPU QMoE enablement is an independent follow-up and should not block the CUDA model-level milestone. A cross-provider correctness fallback may be useful for conformance, but persistent full expert dequantization does not satisfy the production acceptance criteria.

### Phase 5: 3-, 5-, and 6-Bit MatMulNBits Gate

Evaluate INT3, INT5, and INT6 if INT2 cannot achieve the required quality/size tradeoff or product requirements call for less aggressive quantization. Use one generic contiguous pack/unpack reference and compare all three widths against INT2, INT4, and INT8 on quality, effective model size, and extraction cost. INT3 is the first candidate for optimized implementation because it directly fills the gap between INT2 and INT4, but measured results must select the width.

Do not build three independent production kernel families in parallel. After the format and quality study, select at most one width for CUDA M=1 and large-M prototypes. A production commitment requires that the selected width materially outperform neighboring supported formats or mixed-precision recipes.

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

### 3-, 5-, and 6-Bit `MatMulNBits` Study

| Work item | Estimate |
| --- | ---: |
| Generic contiguous format specification and schema/tooling design | 1-2 weeks |
| Python pack/unpack, CPU reference, tests, and 2/3/4/5/6/8-bit quality study | 2-3 weeks |
| CUDA contiguous-versus-bit-plane extraction for all candidates | 1-2 weeks |
| Fused M=1 and large-M prototypes for one selected width | 3-4 weeks |
| Go/no-go analysis and design review | 1 week |

Allow **7-10 engineering weeks** for the generic format-and-value study plus decode and prefill prototypes for one selected width. This estimate does not include production kernels for all three widths. A production-ready CUDA track for the selected width with broad operator coverage remains approximately **11-17 engineering weeks total** after selection. Each additional optimized width requires a separate estimate. Optimized CPU/MLAS support or another execution provider also requires separate estimates.

## Supporting Dense MatMulNBits Delivery Estimate

If staffed as a separate supporting track, approximately 8.5 calendar weeks would allow one engineer to target a scoped dense INT2 CUDA vertical slice built on the existing portable format, Python tooling, and CPU reference implementation. The candidate target is reviewable native fused CUDA execution for M=1 decode and representative large-M prefill under one primary symmetric FP16 configuration. This estimate is not the current QMoE product commitment, and upstream merge timing cannot be guaranteed because review and CI are outside the implementation owner's control.

### Candidate Dense INT2 Scope

- Uniform and mixed INT2/INT4/INT8 quality and effective-size results for representative Qwen3.8 coding-model workloads.
- A qualified PyTorch/Olive-to-Mobius export path producing dense mixed-bit `MatMulNBits` nodes, with graph, initializer-binding, and numerical-parity tests.
- Direct packed-INT2 versus GPU-native LUT prototype results and a selected CUDA strategy.
- CUDA runtime prepacking if justified by profiling, plus fused M=1 GEMV and fused large-M GEMM for FP16 activations, symmetric weights, and one selected block size.
- Correctness and performance results for M=1 decode and representative prefill M values, such as 128, 512, and 2048, on Qwen3.8-27B and Qwen3.8-Flash-Next matrix shapes.
- End-to-end decode throughput and TTFT measurements for a Copilot-style long-context workload.
- ORT integration, focused tests, documentation, and a reviewable pull request or draft pull request, depending on review readiness.

### Schedule

| Dates | Milestone |
| --- | --- |
| September 16-20 | Freeze quality thresholds, Qwen workloads, candidate block sizes, and INT4/INT8/BF16 baselines. |
| September 21-27 | Run uniform and mixed-precision INT2 quality/size experiments; validate the portable format and qualify Olive checkpoint-to-Mobius graph and weight export. |
| September 28-October 11 | Complete export numerical-parity coverage; prototype and compare direct packed-INT2 and GPU-native LUT paths for M=1 decode and large-M prefill. |
| October 12-25 | Select the kernel/prepack strategy; implement fused M=1 GEMV and large-M GEMM for the primary FP16 configuration. |
| October 26-November 1 | Integrate both paths into CUDA `MatMulNBits(bits=2)` and validate representative Qwen decode and prefill shapes. |
| November 2-8 | Add focused correctness tests, serialization, memory estimates, build integration, and performance measurements. |
| November 9-15 | Regression testing, documentation, final quality/performance report, PR cleanup, and review buffer. |

### Stretch Scope

The following items are excluded from the supporting dense estimate:

- Asymmetric zero points.
- BF16 activation support.
- Additional block sizes beyond the selected primary configuration.
- Bias and tail combinations not already covered by the selected kernel path.
- Optimized intermediate-M execution for M values between the committed decode and prefill ranges.
- Offline CUDA-specific prepacking.
- Broad multi-GPU tuning.
- INT3, INT5, or INT6 format, tooling, or kernel implementation.
- ONNX-native Olive INT2 RTN and built-in quantized-linear export completion.
- Fused MoE/QMoE export, packing contracts, and runtime kernels, which are tracked as the separate product-priority plan.

### INT3/INT5/INT6 Scheduling Impact

With the same engineer, INT3/INT5/INT6 implementation should not run concurrently if it threatens the prioritized INT2 QMoE delivery. Before November 15, work should be limited to preserving the generic format analysis and collecting quality data that informs width selection. A second engineer could run the generic reference and quality gate independently.

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
- Uniform 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit, plus selected mixed-precision recipes.
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
- INT3, INT5, and INT6 require non-native unpacking; extraction overhead may erase their bandwidth advantage over neighboring formats.
- INT5 and INT6 may not save enough effective model size over higher-quality neighboring formats after scales, padding, and metadata.
- A poorly specified non-byte-aligned format could create incompatible models across execution providers.
- Supporting every candidate width would multiply kernel, prepacking, test, and maintenance cost without guaranteeing model-level value.
- Extending CUTLASS prepacked formats too early would increase scope before value is established.
- Metadata overhead can materially change the effective bits per weight, especially for small blocks.

## Recommended Initial Deliverables

### 2-Bit Priority Track

- Accuracy and effective-size report for uniform and mixed 2-bit/4-bit/8-bit/BF16 coding-model recipes.
- Direct-unpack and GPU-native LUT comparison on Qwen3.8 decode and prefill shapes.
- CUDA M=1 GEMV and representative large-M GEMM vertical slice for the primary symmetric FP16 configuration.
- Qwen-shape decode, prefill, and TTFT correctness/performance report.
- Go/no-go recommendation for broader production investment.

### 3-/5-/6-Bit Follow-Up Track

- Freeze one generic contiguous packing rule and width-specific 2+1, 4+1, and 4+2 runtime-prepack candidates.
- Run a 2/3/4/5/6/8-bit quality and effective-size comparison if INT2 quality is insufficient.
- Select at most one candidate width for initial CUDA decode and prefill prototypes.
- Require a separate value gate before production CUDA implementation.

## Recommendation to Management

Prioritize the mixed-width INT2 QMoE delivery defined in the standalone plan because expert weights dominate the target model's storage and memory traffic. Develop shared dense MatMulNBits packing, validation, and CUDA extraction primitives where they directly reduce QMoE risk, but do not require complete dense GEMV/GEMM coverage before QMoE schema, export, CPU reference, and packed decode work proceeds.

Keep INT3, INT5, and INT6 MatMulNBits as a parallel measured quality/size study rather than a near-term production commitment. INT3 should receive the first optimization consideration because it fills the INT2-to-INT4 gap, but only after the common reference implementation and model-quality evidence select a width with a defensible advantage. Do not commit to production kernels for all candidate widths.

## References

- ONNX 2-bit integer types: https://onnx.ai/onnx/technical/int2.html
- ONNX Runtime `MatMulNBits` CUDA documentation: ../contrib_ops/cuda/matmul_nbits.md
- llama.cpp quantization documentation: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- llama.cpp Q6_K block definition: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
- TorchAO: https://github.com/pytorch/ao
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM