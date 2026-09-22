# INT2 QMoE End-to-End Delivery Plan

## Executive Summary

The priority is an end-to-end deployable INT2 QMoE path for Qwen3.8-Flash-Next-class models. The target recipe uses INT2 for expert gate/up projections and INT4 for expert down projections, while sensitive non-expert tensors remain at higher precision. This follows the tensor placement of the published Unsloth `UD-Q2_K_XL` model without claiming numerical compatibility with its `IQ2_XS` and `IQ4_NL` formats.

The delivery is complete only when a supported source checkpoint can be quantized, exported to ONNX, loaded by ONNX Runtime, and executed by CUDA with demonstrated model quality, memory reduction, decode throughput, and prefill TTFT. Operator-only correctness is necessary but is not an end-to-end deliverable.

Dense INT2 `MatMulNBits` optimization is no longer the primary product milestone. QMoE should reuse provider-neutral INT2 packing, dequantization, validation, and CUDA load primitives where practical, but completion of dense fused GEMV/GEMM coverage must not block the QMoE contract, export, or packed expert-kernel work.

The critical path is:

```text
Mixed-width QMoE contract
  -> CPU reference and deterministic tests
  -> Olive quantization and Mobius export
  -> CUDA correctness path
  -> packed CUDA decode kernel
  -> bounded or native CUDA prefill path
  -> Qwen model validation and release evidence
```

## Product Target

### Initial Quantization Recipe

| Tensor class | Initial ONNX target | Rationale |
| --- | --- | --- |
| Expert gate/up, QMoE FC1 | Blockwise affine INT2 | Dominant low-bit tier and primary storage/bandwidth target |
| Expert down, QMoE FC2 | Blockwise affine INT4 | Lower quality risk than INT2 for the down projection |
| Attention projections | INT4, INT8, or BF16 according to measured quality | Outside the first QMoE kernel contract |
| Embeddings, indexers, norms, and control tensors | Existing supported higher precision | Preserve quality and avoid unrelated kernel work |

The first supported configuration is:

- CUDA execution provider.
- FP16 activations with FP32 accumulation where required for accuracy.
- Symmetric blockwise integer quantization.
- FC1 INT2 and FC2 INT4.
- One selected block size, chosen from 32, 64, or 128 after quality and kernel profiling.
- Interleaved fused SwiGLU (`swiglu_fusion=1`).
- Top-k routing with multiple tokens and experts.
- Raw portable model weights plus execution-provider-specific runtime prepacking.

BF16, asymmetric zero points, additional block sizes, WebGPU, and alternative SwiGLU layouts are follow-up coverage.

### Format Boundary

ORT blockwise INT2 uses four affine integer codes per byte:

```text
packed = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
dequantized = (q - zero_point) * scale
```

This is not `IQ2_XS`. The published Qwen GGUF is a comparison point for tensor placement, quality, and effective model size, not a bit-compatible input format. Conversion must quantize from the source floating-point checkpoint or another representation that preserves sufficient information; it must not relabel IQ2_XS bytes as `MatMulNBits` or QMoE INT2 bytes.

## Required Operator Contract

### Mixed FC1 and FC2 Widths

QMoE originally had one `expert_weight_bits` attribute shared by FC1 and FC2, which could not represent FC1 INT2 with FC2 INT4. The mixed-width contract merged in [#32697](https://github.com/microsoft/onnxruntime/pull/32697) adds independent bit widths while preserving existing models.

The accepted contract is:

```text
expert_weight_bits      # existing default for backward compatibility
fc1_expert_weight_bits  # optional override
fc2_expert_weight_bits  # optional override
fc3_expert_weight_bits  # optional override when FC3 is present
```

The merged semantics are:

1. An omitted FC-specific value inherits `expert_weight_bits`.
2. Each effective width is restricted to a supported value.
3. Each weight input uses its own `pack_size = 8 / effective_bits`.
4. FC-specific scales, zero points, shape checks, strides, and prepacked layouts use the corresponding effective width.
5. Existing single-width INT4 and INT8 models remain byte-for-byte compatible.
6. Prepacked layouts encode their bit width and layout version; INT2 must never be interpreted as INT4.

### Initial Shapes

For $E$ experts, hidden size $H$, and intermediate size $I$:

```text
FC1 logical: [E, 2 * I, H]
FC1 INT2:    [E, 2 * I, H / 4]

FC2 logical: [E, H, I]
FC2 INT4:    [E, H, I / 2]
```

Scale shapes remain per output row and K-axis block. Validation must use independent FC1 and FC2 bit widths and pack sizes.

## End-to-End Model Production

### Required Workflow

```text
HF/PyTorch Qwen checkpoint
  -> Olive calibration and tensor-wise mixed quantization
  -> mixed FC1 INT2 / FC2 INT4 checkpoint
  -> Mobius fused QMoE export
  -> ONNX model with portable raw expert weights
  -> ONNX Runtime CUDA runtime prepack
  -> QMoE inference
```

The Olive/Mobius path must provide:

- Stable identification of expert gate/up and down tensors.
- Per-tensor quantization decisions and a persisted recipe manifest.
- Deterministic INT2 and INT4 packing.
- Correct FC1 interleaving for fused SwiGLU.
- Correct scales, optional zero points, attributes, and initializer bindings.
- External-data support for large models.
- Graph validation that no selected expert tensor silently remains unquantized.
- Numerical parity between the exported graph and the quantized PyTorch reference.

The recipe manifest should report tensor names, logical shapes, selected widths, block sizes, scale types, packed bytes, and fallback precision. Effective bits per parameter must include scales, padding, metadata, and higher-precision tensors.

## Implementation Workstreams

### Workstream 1: Contract and Shared Validation

Status: Complete. The contract and validation changes merged in [#32697](https://github.com/microsoft/onnxruntime/pull/32697).

Deliverables:

- Maintain the approved mixed-width QMoE schema design.
- Centralize bit-width, pack-size, default-zero-point, and packed-shape calculations.
- Update FC1, FC2, and FC3 validation to use independent effective widths.
- Define raw and provider-prepacked layout behavior.
- Add schema inference, invalid-shape, inheritance, and backward-compatibility tests.

Exit gate: one raw mixed FC1 INT2 / FC2 INT4 model validates identically on all registered QMoE providers, even where execution returns a clear unsupported-status error.

### Workstream 2: CPU Reference

CPU is the semantic oracle for model production and CUDA validation.

Deliverables:

- Extend CPU QMoE from one shared width to independent FC widths.
- Reuse the existing INT2 MLAS LUT path for FC1 where eligible.
- Preserve the existing INT4 path for FC2.
- Cover routing, fused SwiGLU, bias, empty experts, multiple tokens, and top-k greater than one.
- Compare against a scalar explicit-dequantization reference using all INT2 codes 0, 1, 2, and 3.

Exit gate: deterministic mixed-width CPU QMoE tests pass and the exported model matches the quantized PyTorch reference within agreed tolerances.

### Workstream 3: Olive and Mobius

Deliverables:

- Add or qualify selective mixed-precision QMoE quantization.
- Export the approved FC-specific bit-width contract.
- Emit portable raw `[E, N, K / pack_size]` weights.
- Validate FC1 gate/up interleaving and FC2 orientation.
- Add graph, initializer-binding, external-data, and numerical-parity tests.
- Produce a small checked-in synthetic model and a reproducible Qwen conversion command.

Exit gate: a clean environment can convert the selected checkpoint and run the exported model on the CPU reference path without manual graph edits.

### Workstream 4: CUDA Correctness

The correctness path must not reuse an INT4 type or layout for INT2.

Deliverables:

- Accept the approved FC1 INT2 / FC2 INT4 contract in CUDA QMoE.
- Explicitly exclude INT2 from existing INT4/INT8 CUTLASS preprocessing and tactic selection.
- Implement bit-exact INT2 dequantization with block scales and correct row-local packed zero-point addressing.
- Execute FC1 through a bounded dequantization plus existing dense MoE runner for correctness.
- Keep FC2 on its existing INT4 path where possible.
- Add CPU/CUDA parity tests.

Persistent full-model INT2-to-FP16 dequantization expands the affected payload by approximately 8x. It is acceptable only for small tests and as an oracle. The test implementation must have an explicit memory bound and must not be presented as production support.

Exit gate: CUDA executes synthetic and reduced Qwen mixed-width models correctly without interpreting INT2 as INT4 or allocating unbounded scratch.

### Workstream 5: CUDA Packed Decode

The first performance milestone is fused packed execution for decode and low expanded-row counts, where expanded rows are approximately `num_tokens * top_k`.

Deliverables:

- Define a versioned CUDA INT2 runtime-prepacked layout.
- Implement vectorized packed INT2 loads, extraction, scale application, and FP32 accumulation.
- Integrate packed FC1 INT2 with routing and interleaved SwiGLU.
- Reuse or preserve the optimized FC2 INT4 path.
- Support the selected block size and representative Qwen dimensions.
- Profile register pressure, occupancy, effective bandwidth, and prepack cost.
- Add fused-versus-reference correctness tests and M=1 end-to-end benchmarks.

Exit gate: packed QMoE improves decode latency or throughput over the agreed INT4 baseline while preserving the accepted model quality.

### Workstream 6: CUDA Prefill

Long-context prefill must not dequantize every expert into persistent FP16 storage.

Evaluate in this order:

1. Bounded selected-expert or row-tiled dequantization for functional integration.
2. Chunked execution with a documented scratch cap.
3. Native W2A16 grouped GEMM for the production performance target.

A native grouped path requires packed INT2 iterators, converters, block-scale loading, grouped expert pointers and strides, tactic selection, and architecture-specific tuning.

Exit gate: representative prefill values such as M=128, 512, and 2048 complete within the memory budget and improve TTFT or provide an explicitly accepted intermediate baseline.

### Workstream 7: Model Quality and Performance

Quality comparisons must include:

- BF16 baseline.
- Existing INT4 baseline.
- Uniform expert INT2, for diagnosis only.
- Mixed FC1 INT2 / FC2 INT4.
- The published approximately 78.9 GB `UD-Q2_K_XL` result when reproducible metrics are available.

Measure coding, tool-calling, long-generation stability, perplexity or KL divergence, and task-specific acceptance metrics. Performance reporting must include model size, peak and persistent memory, load/prepack time, M=1 latency, tokens per second, prefill TTFT, and effective memory bandwidth.

## Delivery Sequence

### PR 1: Mixed-Width Contract - Merged

Merged as [#32697](https://github.com/microsoft/onnxruntime/pull/32697). Mixed-width execution remains intentionally disabled until the projection-aware CPU and provider paths in the following PRs are implemented.

- Schema and inheritance semantics.
- Shared shape and packing helpers.
- Backward-compatibility and validation tests.

### PR 2: CPU Mixed QMoE

- FC1 INT2 and FC2 INT4 execution.
- Scalar and MLAS parity tests.
- Routing and SwiGLU coverage.

### PR 3: Olive/Mobius Export

- Selective recipe and manifest.
- Fused QMoE graph export.
- Weight-binding and numerical-parity tests.

PR 2 and PR 3 can proceed in parallel on the merged contract.

### PR 4: CUDA Correctness

- INT2 validation and dequantization.
- Bounded fallback.
- CPU/CUDA parity tests.

### PR 5: CUDA Packed Decode

- Runtime prepack.
- Fused FC1 INT2 decode.
- FC2 INT4 integration.
- Correctness and decode benchmarks.

### PR 6: CUDA Prefill

- Bounded integration path.
- Native grouped GEMM when required by the performance gate.
- TTFT and memory benchmarks.

### PR 7: End-to-End Qualification

- Full conversion and inference automation.
- Quality and performance report.
- Documentation and support matrix updates.

## Schedule and Staffing

With one engineer, a production-quality mixed-width QMoE path spanning schema, tooling, CPU, CUDA decode, CUDA prefill, and model qualification is not a credible single eight-week task. A practical schedule uses parallel owners:

| Weeks | Contract/CPU owner | Olive/Mobius owner | CUDA owner | Model-validation owner |
| --- | --- | --- | --- | --- |
| 1-2 | Freeze contract and add validation | Prototype recipe/export against draft contract | Prototype INT2 extraction and layouts | Freeze models, metrics, and baselines |
| 3-4 | Complete CPU mixed-width reference | Complete graph and parity tests | Complete bounded correctness path | Run first quality comparison |
| 5-6 | Support integration fixes | Produce full external-data model | Implement and tune packed decode | Validate decode quality and memory |
| 7-8 | Regression and compatibility tests | Reproducible conversion package | Integrate bounded prefill; prototype grouped GEMM | End-to-end decode and TTFT report |
| 9-12 | Follow-up coverage | Export hardening | Native grouped-GEMM tuning and architecture coverage | Release qualification |

An eight-week milestone should commit to a reviewable mixed-width contract, reproducible export, CPU reference, CUDA correctness, and packed decode on one GPU architecture. Production prefill performance and broad architecture coverage are follow-up commitments unless additional CUDA staffing is assigned.

## Acceptance Criteria

### Functional Completion

- A documented command converts the selected Qwen checkpoint without manual graph edits.
- The ONNX graph contains FC1 INT2 and FC2 INT4 QMoE weights with correct shapes and metadata.
- CPU and CUDA outputs match the quantized PyTorch reference within frozen tolerances.
- Routing, top-k, fused SwiGLU, multiple tokens, empty experts, and bias behave correctly.
- Existing single-width INT4 and INT8 QMoE models do not regress.
- Failures for unsupported configurations are explicit and occur before unsafe preprocessing.

### Product Completion

- The mixed recipe meets frozen coding and tool-calling quality thresholds.
- Effective model size is materially below the INT4 baseline after all metadata and higher-precision tensors are counted.
- CUDA executes packed INT2 FC1 without persistent full expert dequantization.
- Decode demonstrates an accepted improvement over INT4 on representative Qwen workloads.
- Prefill meets the agreed TTFT and peak-memory targets using a bounded or native path.
- Conversion, model loading, prepacking, inference, and evaluation are automated in CI or a reproducible qualification pipeline.

## Risks and Stop Conditions

| Risk | Mitigation or stop condition |
| --- | --- |
| Affine INT2 quality is materially below IQ2_XS | Improve calibration or mixed precision; stop kernel expansion if no viable recipe exists |
| Mixed-width schema creates excessive compatibility cost | Evaluate separate FC attributes versus a versioned quantization descriptor before implementation |
| INT2 unpacking removes decode benefit | Compare direct extraction and LUT/prepacked layouts before committing to one kernel family |
| Prefill activates most experts | Prioritize native grouped GEMM; do not rely on selected-expert dequantization as the final solution |
| Runtime prepack increases load time or memory | Report persistent bytes and prepack latency; support offline versioned packing only after the layout stabilizes |
| Export and runtime contracts diverge | Use one deterministic golden model across Olive, Mobius, CPU, and CUDA tests |

Stop production kernel expansion if no mixed affine INT2 recipe meets the frozen quality and effective-size gates. A correctness implementation alone does not justify shipping large-model QMoE INT2 support.

## Immediate Decisions Needed

1. Select the first block size and CUDA architecture.
2. Assign owners for ORT CPU execution, Olive/Mobius, CUDA, and model validation.
3. Freeze quality, memory, decode, and TTFT thresholds before collecting candidate results.
4. Decide whether native grouped prefill is part of the first committed date or a follow-up performance milestone.