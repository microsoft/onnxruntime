# INT2 QMoE End-to-End Delivery Plan

## Executive Summary

The priority is an end-to-end deployable INT2 QMoE path for Qwen3.8-Flash-Next-class models. The target recipe uses INT2 for expert gate/up projections and INT4 for expert down projections, while sensitive non-expert tensors remain at higher precision. This follows the tensor placement of the published Unsloth `UD-Q2_K_XL` model without claiming numerical compatibility with its `IQ2_XS` and `IQ4_NL` formats.

The delivery is complete only when a supported source checkpoint can be quantized, exported to ONNX, loaded by ONNX Runtime, and executed by CUDA with demonstrated model quality, memory reduction, decode throughput, and prefill TTFT. Operator-only correctness is necessary but is not an end-to-end deliverable.

Dense INT2 `MatMulNBits` optimization is no longer the primary product milestone. QMoE should reuse provider-neutral INT2 packing, dequantization, validation, and CUDA load primitives where practical, but completion of dense fused GEMV/GEMM coverage must not block the QMoE contract, export, or packed expert-kernel work.

The critical path is:

```text
Mixed-width QMoE contract
  -> deterministic packed test vectors and scalar reference
  -> CUDA correctness path
  -> packed CUDA decode kernel
  -> CPU reference and cross-provider parity
  -> Olive quantization and Mobius export
  -> bounded or native CUDA prefill path
  -> Qwen model validation and release evidence
```

## Product Target

### Delivery Status (October 6, 2026)

- Contract, bounded CUDA correctness and packed decode merged in #32697, #32743 and #32761, respectively.
- PR 4a [#32963](https://github.com/microsoft/onnxruntime/pull/32963) merged on October 1 as `a4d55e2b6f`: SM80 FP16/BF16 INT2 grouped GEMM foundation.
- PR 4a2 [#33045](https://github.com/microsoft/onnxruntime/pull/33045) merged on October 1 as `78c834918d`: grouped-kernel test coverage, plugin test build footprint and diagnostic follow-up, including QUICK_BUILD-compatible baseline tactics.
- PR 4b [#33005](https://github.com/microsoft/onnxruntime/pull/33005) merged on October 2 as `2cdff3ca8e`: default-enabled SM80+ FP16/BF16 packed INT2 FC1 / INT4 FC2 prefill, block size 64, symmetric weights and interleaved fused SwiGLU. Eligible prefill uses packed row tiling within a 256 MiB estimated temporary-scratch budget rather than switching long prompts to dense weight dequantization.
- PR 4c [#33092](https://github.com/microsoft/onnxruntime/pull/33092) merged on October 5 as `0c67e77bb7cb`: packed prefill supports block sizes 32, 64 and 128 and `(FC1, FC2)` widths `(2,4)`, `(2,2)` and `(4,2)`, dispatching each projection to the INT2 or INT4 grouped GEMM. Final-head local A100 CUDA validation passed the expanded integration matrix and direct grouped-GEMM tests; see PR 4c below for scope and limitations.
- Implementation merge is not end-to-end qualification. Remaining work includes reproducible fused Olive/Mobius export, CPU/CUDA model parity, target-model quality, controlled benchmarks, exact memory attribution and transition/concurrency/capture qualification. SM80+ dispatch eligibility does not establish execution coverage on every supported GPU architecture.

### Initial Quantization Recipe

| Tensor class | Initial ONNX target | Rationale |
| --- | --- | --- |
| Expert gate/up, QMoE FC1 | Blockwise affine INT2 | Dominant low-bit tier and primary storage/bandwidth target |
| Expert down, QMoE FC2 | Blockwise affine INT4 | Lower quality risk than INT2 for the down projection |
| Attention projections | INT4, INT8, or BF16 according to measured quality | Outside the first QMoE kernel contract |
| Embeddings, indexers, norms, and control tensors | Existing supported higher precision | Preserve quality and avoid unrelated kernel work |

The first supported configuration is:

- CUDA execution provider.
- FP16 or BF16 activations with FP32 accumulation.
- Symmetric blockwise integer quantization.
- `(FC1, FC2)` widths `(2,4)`, `(2,2)` and `(4,2)` are merged, including the extensions in #33092.
- Block sizes 32, 64 and 128 for packed prefill are merged and locally tested on A100. Block size 32 has no GEMV path; broader deployment-GPU qualification remains open.
- Interleaved fused SwiGLU (`swiglu_fusion=1`).
- Top-k routing with multiple tokens and experts.
- Raw portable model weights plus execution-provider-specific runtime prepacking.

Asymmetric zero points, additional block sizes, WebGPU, and alternative SwiGLU layouts are follow-up coverage.

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

### Workstream 2: CUDA Correctness

Status: Complete. The bounded FP16/BF16 dense dequantization fallback for INT2 and mixed-width QMoE merged in [#32743](https://github.com/microsoft/onnxruntime/pull/32743). It includes canonical raw-weight validation, blockwise scales and zero points, a configurable scratch-memory limit, and CUDA correctness coverage for uniform INT2 and mixed FC widths.

The correctness path must not reuse an INT4 type or layout for INT2.

Deliverables:

- Accept the approved FC1 INT2 / FC2 INT4 contract in CUDA QMoE.
- Explicitly exclude INT2 from existing INT4/INT8 CUTLASS preprocessing and tactic selection.
- Implement bit-exact INT2 dequantization with block scales and correct row-local packed zero-point addressing.
- Execute FC1 through a bounded dequantization plus existing dense MoE runner for correctness.
- Keep FC2 on its existing INT4 path where possible.
- Compare against deterministic scalar explicit-dequantization and quantized PyTorch references using all INT2 codes 0, 1, 2, and 3.

Persistent full-model INT2-to-FP16 dequantization expands the affected payload by approximately 8x. It is acceptable only for small tests and as an oracle. The test implementation must have an explicit memory bound and must not be presented as production support.

Exit gate: CUDA executes synthetic and reduced Qwen mixed-width models correctly without interpreting INT2 as INT4 or allocating unbounded scratch.

### Workstream 3: CUDA Packed Decode

Status: Merged. [#32761](https://github.com/microsoft/onnxruntime/pull/32761) merged as `a11b4e5931` and adds direct SM80 packed CUDA decode for uniform INT2 and mixed INT2/INT4 widths, with a bounded dense fallback for packed-ineligible configurations. Model-level performance and quality gates remain open.

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

### Workstream 4: CUDA Prefill

Status: Native packed implementation merged through PR 4a, PR 4a2 and PR 4b. Product-level quality, performance, memory and supported-platform qualification remain open.

Prefill is enabled by default for eligible configurations; `ORT_ENABLE_QMOE_INT2_PREFILL=0` opts out. Packed decode retains priority, and `ORT_DISABLE_MOE_GEMV=1` disables eligible GEMV dispatch for both INT2 and existing INT4/INT8 paths. Set these environment variables before process startup. Ineligible configurations retain the bounded dense fallback.

The packed scratch planner includes packed workspace, tile-local routing metadata and uncached scale transposes. It honors `ep.cuda.qmoe_row_tile_size` as an upper bound, sizes both full and final partial tiles, and reduces tile rows until the estimate fits 256 MiB. If even one row cannot fit, it reports an error rather than allocating full dense expert weights. This bound is not a whole-process GPU memory cap.

Initial A100 GPT-OSS-20B prefill/decode results are recorded in #33005. They predate the final default-dispatch and packed-tiling changes and are not validation of those revisions or of newer GPU architectures. The following alternatives and engineering estimates record the original design decision, not remaining implementation work.

Long-context prefill must not dequantize every expert into persistent FP16 storage.

Full-model GPT-OSS-20B validation demonstrates why a separate prefill path is required. Its `top_k=4` routing produces 76 expanded rows for a 19-token prompt, outside the packed GEMV gate of eight expanded rows. The correctness fallback then requests 1,592,524,800 bytes to materialize all FC1 and FC2 expert weights, exceeding the default 1 GiB scratch limit. Raising the limit is useful only for bounded diagnosis and is not a production solution.

The estimates below assume one engineer familiar with ORT CUDA and QMoE, one initial architecture (SM80/A100), symmetric FP16/BF16 mixed FC1 INT2 / FC2 INT4 with block size 64, and include implementation, focused correctness tests, and profiling. They exclude external review and CI queue time. Asymmetric zero points, additional block sizes, and broad architecture tuning require follow-up estimates.

#### Option A: Full Temporary Dequantization

Dequantize every expert into temporary FP16/BF16 FC1 and FC2 buffers, then reuse the existing dense grouped-MoE runner. This is the current correctness fallback.

- **Implementation estimate: 3-5 engineering days** to harden error reporting, memory accounting, and reduced-model coverage because the execution path already exists.
- **Qualification estimate: 2-3 additional engineering days** for model-level parity and bounded-memory failure tests.
- **Total: approximately 1-1.5 engineering weeks.**
- **Use:** correctness oracle, reduced models, and explicitly bounded diagnostics.
- **Limitation:** GPT-OSS-20B already exceeds the default scratch cap; this option does not satisfy the production memory or TTFT gate.

#### Option B: Selected-Expert or Chunked Dequantization

Use routing results to dequantize only active experts, or process expert/row tiles through reusable bounded scratch before invoking the existing dense grouped-MoE runner.

- **Prototype estimate: 1.5-2 engineering weeks** for routing compaction, scratch planning, and a functional FC1/FC2 pipeline.
- **Hardening estimate: 1.5-2 engineering weeks** for stream ordering, buffer lifetime, empty/duplicate expert handling, multi-token/top-k coverage, and parity tests.
- **Performance qualification: 1 engineering week** for M=128/512/2048 profiling, TTFT, peak-memory measurement, and threshold tuning.
- **Total: approximately 4-5 engineering weeks.**
- **Use:** the preferred bounded intermediate path and the fastest credible route to functional long-prompt inference.
- **Limitation:** large prefill batches may activate most experts, reducing memory and bandwidth savings; this should not be assumed to be the final production-performance solution.

#### Option C: Native Packed W2A16 Grouped GEMM

Consume packed INT2/INT4 expert weights directly in a grouped GEMM without materializing complete A16 expert weights in global memory. Tile-local unpacking and conversion to FP16 or BF16 for floating-point Tensor Core computation are allowed; this does not require a hardware INT2-by-A16 instruction. This is the preferred production path.

- **Kernel foundation: 2-3 engineering weeks** for packed INT2 iterators, conversion, block-scale loading, mixed FC widths, and explicit instantiations.
- **QMoE integration: 2-3 engineering weeks** for grouped expert descriptors, routing, SwiGLU/finalization epilogues, workspace planning, and dispatch.
- **Correctness and tuning: 2-3 engineering weeks** for tactic profiling, M=128/512/2048 parity and performance, and regression coverage on SM80.
- **Total: approximately 6-9 engineering weeks for one architecture and the primary configuration.**
- **Follow-up: 2-4 engineering weeks** for additional block sizes, asymmetric zero points where required, and SM90/SM100/SM120 tuning.
- **Use:** production TTFT and memory target.
- **Risk:** largest implementation and review surface; architecture-specific profiling may require separate dispatch thresholds or kernels.

##### Option C Delivery Plan: Two Core PRs

The preferred review boundary is a separately testable grouped-GEMM kernel followed by QMoE integration. These are PR 4a and PR 4b in the overall delivery sequence, not two separate implementations of prefill.

The initial implementation and profiling targeted SM80/A100; the merged integration accepts SM80+ while retaining the SM80 CUTLASS template and packed layout. Target GPUs require suitable native code or compatible virtual PTX; an SM80 cubin alone is not cross-major compatible. The primary configuration uses FP16/BF16 activations, symmetric FC1 gate/up INT2 and FC2 down INT4, block size 64, and existing interleaved SwiGLU semantics. The portable model contract (`weights_prepacked=0`) and runtime prepacking remain unchanged. Additional block sizes and bit-width combinations, asymmetric zero points, architecture-specific tuning and hardware qualification are follow-ups. The packed-GEMV gate is not expanded merely to serve prefill.

**PR 4a: SM80 packed W2A16 grouped GEMM foundation**

- Reuse the existing INT2 numeric converters and layout traits where compatible. Prove compatibility with grouped-GEMM iterators, fragments, and block-scale indexing rather than assuming a new `uint2b_t` instantiation is sufficient.
- Start with a fixed-tactic prototype for one expert, then multiple experts. Complete packed INT2 loads, tile-local conversion, block-64 scale handling, FP32 accumulation, shape/alignment eligibility, and explicit FP16/BF16 instantiations. Verify that the FC2 W4A16 path supports the required block-64 configuration.
- Add independently callable FP16/BF16 kernel tests covering uneven expert row counts, empty experts, distinct scales across experts/output columns/K blocks, and supported alignment boundaries. Include kernel benchmarks and evidence that complete A16 expert weights are not materialized.
- Leave QMoE's default execution path unchanged. Merge only when the kernel is directly exercised by tests and has basic performance evidence, not as an untested collection of unused type definitions.

**PR 4b: Mixed INT2/INT4 QMoE prefill integration**

- Depend on PR 4a. Evaluate separate W2 FC1 and W4 FC2 runner instances sharing routing/workspace management before broadening the existing runner abstraction. Reuse routing, token permutation, bias/SwiGLU, and routing-weighted finalization; independent activation/finalization kernels are acceptable initially.
- Integrate prepacking, grouped expert descriptors, workspace sizing, dispatch, and necessary tactic selection using the current profiling/KernelPilot mechanisms. Select FC1 and FC2 tactics independently where needed. Validate whether GEMV and GEMM can share packed buffers; account for every persistent copy if they cannot.
- Preserve packed decode and the bounded fallback for configurations outside the new path. Retain raw weights when a later invocation can require fallback. Validate buffer lifetimes, stream ordering, and prefill-to-decode transitions within one session.
- Deliver end-to-end correctness, TTFT, and peak-memory evidence before enabling the new path by default for qualified configurations. Necessary performance qualification belongs in this PR, not entirely in a later tuning PR.

**Shared acceptance and measurement plan**

- Compare against a reference built from the same quantized weights and scales, with frozen numerical tolerances. Test bias, SwiGLU, different top-k values, inactive/hot experts, skewed routing, and prefill -> decode -> prefill in one session.
- Use input token counts `T=128,512,2048`, plus `T=1,2,3` for the GPT-OSS `top_k=4` dispatch boundary. Record `T * top_k` and the distribution of per-expert row counts; total input rows are not the same as the GEMM row count for each expert.
- Set `ep.cuda.qmoe_int_dequant_max_scratch_bytes` to one byte in supported-path tests and confirm successful execution with kernel traces. This proves the dense expert-dequantization fallback is bypassed, not that total workspace is one byte.
- Report QMoE latency, full-model prefill latency, TTFT, prepack time, persistent packed/raw weight bytes, activation/routing workspace, and peak device memory. Check packed-decode and existing INT4/INT8 regressions.
- Use the dense fallback as a controlled reduced-model correctness/performance baseline and uniform INT4 as a model-level product baseline. Different bit widths prevent the latter from being interpreted as a pure kernel comparison. Freeze the TTFT/memory acceptance targets before collecting candidate results.
- Qualify GPT-OSS-20B mixed-width multi-token prefill followed by generation without raising the default dense-dequantization scratch cap. No full-expert A16 allocation is allowed on the new path.

The first approximately one-week kernel prototype is included in the 2-3-week foundation estimate; splitting review does not reduce the total 6-9 engineering weeks or add a second foundation budget. Follow-up PRs expand dtype, quantization, architecture coverage, and optional fusion/tuning. If the prototype shows that splitting requires substantial temporary interfaces or duplicated infrastructure, use one PR with clearly layered commits instead of forcing an artificial boundary.

Option B remains the preferred bounded intermediate path when an earlier functional milestone is required, but it is not a prerequisite for PR 4a/4b. A dedicated Option C effort can proceed directly with Option A as the correctness oracle. Option A must not be represented as product prefill support.

Exit gate: representative prefill values such as M=128, 512, and 2048 complete within the memory budget and improve TTFT or provide an explicitly accepted intermediate baseline.

### Workstream 5: Olive and Mobius

Downstream ownership is tracked in [Olive#2638](https://github.com/microsoft/Olive/issues/2638) for the fused-expert recipe and checkpoint qualification, and [Mobius#735](https://github.com/onnxruntime/mobius/issues/735) for checkpoint ingestion, projection-specific packing, graph emission, initializer binding, and ORT parity. The dense `MatMulNBits` work in [Olive#2671](https://github.com/microsoft/Olive/pull/2671) and [Mobius#740](https://github.com/onnxruntime/mobius/pull/740) does not imply fused mixed-width QMoE support.

Native Mobius QMoE construction is sufficient for the first exporter milestone. Dense-graph-to-QMoE fusion is a separate follow-up; it must not block the first deterministic fused-QMoE fixture and export path.

Deliverables:

- Add or qualify selective mixed-precision QMoE quantization.
- Export the approved FC-specific bit-width contract.
- Emit portable raw `[E, N, K / pack_size]` weights.
- Validate FC1 gate/up interleaving and FC2 orientation.
- Add graph, initializer-binding, external-data, and numerical-parity tests.
- Produce a small checked-in synthetic model and a reproducible Qwen conversion command.

Exit gate: a clean environment can convert the selected checkpoint and run the exported model on CUDA, then reproduce the result on the CPU reference path without manual graph edits.

### Workstream 6: CPU Reference and Cross-Provider Parity

CPU provides the portable regression oracle after the CUDA execution contract is proven.

Deliverables:

- Extend CPU QMoE from one shared width to independent FC widths.
- Reuse the existing INT2 MLAS LUT path for FC1 where eligible.
- Preserve the existing INT4 path for FC2.
- Cover routing, fused SwiGLU, bias, empty experts, multiple tokens, and top-k greater than one.
- Compare CPU and CUDA against the same scalar explicit-dequantization reference.

Exit gate: deterministic mixed-width CPU QMoE tests pass, CPU and CUDA agree within frozen tolerances, and both match the quantized PyTorch reference.

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

Merged as [#32697](https://github.com/microsoft/onnxruntime/pull/32697). CUDA mixed-width execution was subsequently enabled by the correctness, decode and prefill PRs below; CPU mixed-width execution remains a separate workstream.

- Schema and inheritance semantics.
- Shared shape and packing helpers.
- Backward-compatibility and validation tests.

### PR 2: CUDA Correctness - Merged

Merged as [#32743](https://github.com/microsoft/onnxruntime/pull/32743).

- INT2 validation and dequantization.
- Bounded fallback.
- Scalar-reference and quantized-PyTorch parity tests.

### PR 3: CUDA Packed Decode - Merged

Merged as [#32761](https://github.com/microsoft/onnxruntime/pull/32761) in `a11b4e5931`.

- Runtime prepack.
- Fused FC1 INT2 decode.
- FC2 INT4 integration.
- Correctness and decode benchmarks.

### PR 4a: CUDA Packed Prefill Kernel Foundation - Merged

Merged as [#32963](https://github.com/microsoft/onnxruntime/pull/32963) on October 1 in `a4d55e2b6f`.

- Option C SM80 FP16/BF16 W2A16 grouped GEMM with block size 64, independent tests, and kernel benchmarks.
- Verify W4A16 block-64 support for FC2; keep QMoE default dispatch unchanged.
- Original estimate: approximately 2-3 engineering weeks, including the initial prototype.

### PR 4a2: Grouped GEMM Test and Plugin Follow-Up - Merged

Merged as [#33045](https://github.com/microsoft/onnxruntime/pull/33045) on October 1 in `78c834918d`.

- SM8x test coverage, plugin test build footprint, diagnostic options and file headers.
- QUICK_BUILD-compatible baseline tactics; the previously reported stages=3 CI failure is no longer an outstanding delivery task.

### PR 4b: CUDA Mixed-Width Prefill Integration - Merged

Merged as [#33005](https://github.com/microsoft/onnxruntime/pull/33005) on October 2 in `2cdff3ca8e`.

- Depends on PR 4a: FC1 INT2 / FC2 INT4 routing, activation/finalization, workspace, prepack lifetime, and dispatch.
- Default-enabled SM80+ FP16/BF16 packed prefill, bounded packed row tiling and shared GEMV controls; packed decode and unsupported-configuration fallback are preserved.
- Original estimate: approximately 2-3 engineering weeks for integration plus 2-3 for correctness and tuning, totaling 6-9 weeks with PR 4a.
- Initial GPT-OSS-20B benchmarks are recorded in #33005. Controlled TTFT/peak-memory measurements and broader model/platform qualification remain follow-up work, not evidence implied by merge.
- Option B is an optional separate intermediate PR (approximately 4-5 engineering weeks), not a dependency. Option A remains the correctness oracle only.

See Workstream 4 for the support boundary, merge gates, and the condition under which PR 4a/4b should remain one layered PR. Additional dtypes, architectures, and optional fusion are follow-ups. Block sizes and bit-width combinations are addressed by PR 4c.

### PR 4c: Block Sizes and Width Combinations - Merged

Merged as [#33092](https://github.com/microsoft/onnxruntime/pull/33092) on October 5, 2026 (`0c67e77bb7cb`), extending the PR 4a/4b packed prefill path.

- Block sizes 32, 64 and 128 for packed prefill. The INT2 grouped GEMM support check also requires the reduction size to be divisible by the block size, and the QMoE integration already rejects hidden or intermediate sizes not divisible by it. Block size 32 adds no GEMV path, so small-batch decode at block size 32 uses grouped GEMM instead of the dense fallback.
- `(FC1, FC2)` widths `(2,4)`, `(2,2)` and `(4,2)`. Each projection is dispatched by its own width; `(4,4)` stays on the existing path, and INT8 combinations, zero points and non-fused or non-interleaved SwiGLU keep the bounded fallback.
- Defaults are unchanged: omitted FC-specific widths inherit `expert_weight_bits`, which defaults to 4, so existing INT4 models are unaffected.
- Final-head validation at `db8a7625b8` passed locally on A100-SXM4-80GB in Docker `jiafa-dev`, using a non-plugin Release CUDA 12.8 build with BF16 and internal tests enabled: 15 outer tests (14 QMoE tests plus the internal-test wrapper), zero failures/skips; 26 internal tests (22 grouped-GEMM and four validation tests), zero failures/skips, with three benchmarks disabled. The 216-invocation FP16/BF16 integration matrix, direct block-size/row-tile cases, cached-scale decode-to-prefill transition, packed-decode fallback and block32 dense fallback with prefill disabled passed. The [validation report](https://github.com/microsoft/onnxruntime/pull/33092#issuecomment-5988509632) records the commands and scope. This is local A100 correctness evidence, not plugin integration, every-GPU coverage or a performance-improvement claim.

### PR 5: Olive/Mobius Export

- Selective recipe and manifest.
- Fused QMoE graph export.
- Weight-binding and numerical-parity tests.

### PR 6: CPU Mixed QMoE and Parity

- FC1 INT2 and FC2 INT4 execution.
- Scalar and MLAS parity tests.
- Routing and SwiGLU coverage.
- CPU/CUDA cross-provider parity tests.

PR 5 and PR 6 can proceed in parallel with the PR 4a/4b prefill effort after the CUDA execution contract is established; their numbering does not make them dependent on native prefill.

### PR 7: End-to-End Qualification

- Full conversion and inference automation.
- Quality and performance report.
- Documentation and support matrix updates.

## Schedule and Staffing

The schedule below is the original planning baseline. As of October 2, the CUDA contract, correctness, packed decode and native packed prefill implementations have merged. Do not interpret their historical estimates as remaining work; export, CPU parity and product qualification still require tracked owners and evidence.

With one engineer, a production-quality mixed-width QMoE path spanning schema, tooling, CPU, CUDA decode, CUDA prefill, and model qualification is not a credible single eight-week task. A practical schedule uses parallel owners:

The schedule below retains the Option B-first intermediate milestone. If Option C is selected directly, replace the CUDA prefill sequence with PR 4a/4b's 6-9 engineering-week budget; do not treat Option B as mandatory or collapse native-prefill qualification into the kernel-foundation milestone.

| Weeks | Contract/CPU owner | Olive/Mobius owner | CUDA owner | Model-validation owner |
| --- | --- | --- | --- | --- |
| 1-2 | Maintain merged contract and deterministic test vectors | Prototype recipe/export against merged contract | Complete bounded correctness path | Freeze models, metrics, and baselines |
| 3-4 | Support CUDA reference validation | Continue graph and initializer work | Implement and tune packed decode | Run first CUDA quality comparison |
| 5-6 | Complete CPU mixed-width reference and parity | Produce full external-data model | Design Option B scratch/routing plan; prototype bounded prefill | Validate decode quality and memory |
| 7-9 | Regression and compatibility tests | Reproducible conversion package | Implement and harden Option B; profile prefill memory and TTFT | End-to-end decode and bounded-prefill report |
| 10-15 | Follow-up coverage | Export hardening | Implement and tune Option C native packed grouped GEMM on SM80 | Native-prefill qualification |
| 16-19 | Additional provider coverage | Export regression coverage | BF16, additional quantization configurations, and architecture tuning | Release qualification |

An eight-week milestone should commit to the merged mixed-width contract, CUDA correctness and packed decode on one GPU architecture, a CPU parity path, reproducible export, and at most an Option B bounded-prefill baseline. Option C production prefill and broad architecture coverage are follow-up commitments unless an additional CUDA engineer is assigned.

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

1. Qualify the merged block-64 SM80+ configuration on the intended deployment GPUs and prioritize additional configurations.
2. Assign owners for ORT CPU execution, Olive/Mobius, CUDA, and model validation.
3. Freeze quality, memory, decode, and TTFT thresholds before collecting candidate results.
4. Freeze the end-to-end release scope and qualification date now that native grouped prefill is merged.
