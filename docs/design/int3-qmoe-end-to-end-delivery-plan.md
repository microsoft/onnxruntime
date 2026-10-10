# INT3 QMoE End-to-End Delivery Plan

## Summary and Document Status

Date: 2026-10-08. This plan follows the stage-gate approach in [ORT roadmap PR #32657](https://github.com/microsoft/onnxruntime/pull/32657) and the [INT2 end-to-end delivery plan](int2-qmoe-end-to-end-delivery-plan.md). It respects the canonical-bitstream / provider-prepack boundary in the [low-bit-width exploration document](2bit-6bit-weight-only-quantization-exploration.md) and defines a uniform INT3 QMoE baseline, subsequent performance work, and a portable-format draft for discussion with the ONNX committee.

**This is a design proposal, not an approved ONNX standard or a claim that existing QMoE supports INT3.** The current `com.microsoft::QMoE` is an ORT contrib operator. Additional widths, standard-domain operators, and tensor datatypes require their respective schema review, versioning, and ONNX approval processes. This plan does not assign an ONNX INT3 datatype number or standard opset.

First deliver a trustworthy baseline that can be quantized, exported, loaded, and executed for both decode and prefill; then decide where to invest in optimization. The baseline does not promise to outperform INT2/INT4. Production performance qualification requires separate acceptance. INT3's initial purpose is to establish whether it offers a useful trade-off among quality, effective model size, and execution cost.

```text
INT3 portable contract and reference byte vectors
  -> independent scalar oracle and shared validation
  -> bounded CUDA correctness for decode + prefill
  -> packed CUDA decode
  -> bounded selected-expert prefill / native packed prefill
  -> CPU reference and cross-provider parity
  -> Olive quantization + Mobius fused QMoE export
  -> Qwen model quality, capacity, and performance qualification
  -> ONNX committee proposal package and interoperability review
```

Export, CPU reference, and CUDA work can proceed in parallel after the contract is frozen. Existing INT2 experience is reusable, but its types, divisibility formulas, and runtime-prepack bytes cannot be treated as an INT3 implementation. The overview is not a strict dependency chain; the stage table and Delivery Sequence below define the dependencies, including the conditional native-prefill branch.

## 1. Product Goals and Initial Scope

The initial target is a Qwen3-30B-A3B-class model with all three baseline `(FC1, FC2)` combinations: `(3,4)`, `(3,3)`, and `(4,3)`. FC1 contains gate/up and FC2 contains down; sensitive non-expert parameters retain existing higher-precision recipes. Start with small deterministic fused fixtures for every combination before a full model; byte-level debugging must not begin only at full-model scale. A `(3,4)` prototype may be the first development checkpoint, but it does not complete the baseline.

| Area | Baseline requirement | Follow-up research, not a baseline blocker |
| --- | --- | --- |
| Width combinations | `(FC1, FC2)=(3,4)`, `(3,3)`, and `(4,3)`, with fused FC3 at the FC1 width | Other mixed strategies beyond these three required combinations |
| Activations | FP16/BF16 with FP32 accumulation, validated separately | INT8 activation, DP4A, INT8 Tensor Core |
| Weight quantization | Uniform, symmetric, grouped along K; block64 for the first model | Asymmetric quantization, codebooks, additional block sizes |
| Logical block size | Format profile defines 32/64/128; the first optimized kernel may support only 64 | Extend optimized dispatch based on evidence |
| Fusion | `swiglu_fusion=1`, with explicit gate/up row interleaving | Non-fused FC3 and concatenated layouts require separate qualification |
| Routing | Multiple experts, top-k, single/multiple tokens, empty expert buckets | Production coverage for concurrency, capture, etc. still requires stage acceptance |
| Platforms | CUDA SM80/A100 as the first hardware target; CPU as a reference | SM86/89, H200, Spark, other providers |
| Prefill | Baseline must execute with bounded scratch | Native packed grouped GEMM belongs to the performance stage |

Do not change existing routing, normalization, activation-parameter, or FC2 semantics. Include bias coverage in correctness fixtures; do not misrepresent a no-bias optimization eligibility condition as an operator-format restriction. Do not import IQ3/Q3_K bytes or expand this plan into a full dense `MatMulNBits(bits=3)` product commitment.

## 2. Proposed ONNX-Facing Portable INT3 Format

The MUST/SHOULD statements below are normative requirements **within this proposal**. The committee has not approved them. Exporters must not describe the format as a stable general standard before the schema and shared vectors are frozen.

### 2.1 Mathematical Semantics and Codes

Each stored weight code is an unsigned integer `u in [0,7]`. The initial symmetric profile has implicit zero point `z=4` and logical signed value `q=u-4 in [-4,3]`; this is not three-bit two's-complement encoding.

For logical weight `W[e,n,k]`:

```text
b = floor(k / block_size)
W_dequant[e,n,k] = (u[e,n,k] - 4) * S[e,n,b]
```

`S` MUST be finite and nonnegative. `S=0` means that every decoded weight in the group is zero. The exporter SHOULD set the codes in such groups to 4 for deterministic zero groups. Decoding semantics do not depend on RTN, GPTQ, or the calibration method.

The independent RTN fixture uses the following default quantization rule:

```text
s = max(max(-W, 0) / 4, max(W, 0) / 3)  # Take the maximum over each actual K block.
q = clamp(round_to_nearest_even(W / s), -4, 3)
u = q + 4
```

An all-zero group directly produces `s=0,u=4`. The fixture first converts the scale to its final storage dtype, then generates codes using that rounded scale. If the scale underflows to zero, treat the group as a zero group. Record the error rather than silently changing dtype. Production quantizers may use other scale-optimization algorithms, but MUST record the method, rounding, clipping, and scale dtype. The quantization algorithm is not part of the packed-decoding standard.

### 2.2 Single Model Serialization Layout: Row-Local LSB-First Contiguous Bitstream

Portable models MUST use a `uint8` tensor container; no new ONNX INT3 tensor datatype is required. Pack each output row of each expert independently, contiguously along K.

The first ORT exporter MUST explicitly set `quant_type='int'`, `weights_prepacked=0`, `block_size`, and the effective FC widths; it must not rely on current provider-specific prepacked defaults. Derive `H` from the activation's hidden dimension and `I` from half the logical output-row count of fused FC1. Then validate the FC2 logical row count and packed K dimension; do not infer I from the packed byte count.

```text
logical shape: [E, N, K]
row_bytes R = ceil(3 * K / 8)
packed shape: [E, N, R]
row byte offset = (e * N + n) * R
value k starts at bit offset 3 * k within that row
```

Write each code's least significant bit first. The bitstream advances by byte address and, within each byte, from bit0 to bit7. Values may cross byte boundaries; padding every INT3 value to a nibble is prohibited. Rows and experts do not share bytes. Unused high bits of the final byte MUST be zero.

A block is a logical scale group, not a header or extra padding in the bitstream. Complete blocks in the block32/64/128 profile are naturally byte-aligned, but the decoder MUST follow the bit-offset definition above rather than assume that `8 / bits` integers fit exactly in a byte.

Model bytes are the sole source of truth and do not depend on CPU endianness. Reference extraction across byte boundaries is:

```text
bit = 3 * k
byte = floor(bit / 8)
shift = bit % 8
word = P[byte]
if byte + 1 < R: word |= P[byte + 1] << 8
u = (word >> shift) & 7
```

Reads MUST not cross the end of the current row. All size, stride, and offset calculations MUST check for overflow.

### 2.3 Logical Tails and Physical Padding Bits

The format permits `K` that is not a multiple of the block size or 8. The number of scale blocks is `B=ceil(K/block_size)`, and the final group contains only the remaining real elements. `K` MUST come from the logical shape/operator attributes, not be inferred from the packed byte count.

Unused high bits in the last byte are **physical padding bits**. Their zero value does not represent an additional `q=-4` weight. If the runtime pads K for vectorization, it MUST mask extra elements or use logical-zero code=4. Extra elements must not affect routing, scale grouping, or output semantics. Runtime padding must not be serialized back as the original K.

The first CUDA optimization may require alignment such as `K % block_size == 0`. These are execution eligibility conditions, not portable-format restrictions. A valid model that cannot execute must receive an explicit unsupported diagnostic or use a bounded reference fallback; silently misreading tails is prohibited.

The current QMoE schema requires H/I to be divisible by the block size and the last weight dimension to be byte-aligned. This proposal's ceil/tail semantics are not current schema capabilities; P1 must review that extension. The first ORT model retains the requirement that H/I be divisible by the shared block size. If the tail extension is deferred, portable-format fixtures may define tails first, but the ORT exporter must reject tail models not supported by the approved schema.

### 2.4 FC1/FC2 Ordering and Scale Shapes

For `E` experts, hidden size `H`, and intermediate size `I`, the initial fused SwiGLU profile defines the logical row order as follows:

```text
FC1 logical: [E, 2*I, H]
FC1 row 2*j     = gate[j,:]
FC1 row 2*j + 1 = up[j,:]
Y[j] = activation(gate[j]) * up[j]
FC1 INT3 bytes: [E, 2*I, ceil(3*H/8)]
FC1 INT4 bytes: [E, 2*I, ceil(4*H/8)]
FC1 scales:     [E, 2*I, ceil(H/block_size)]

FC2 logical: [E, H, I]  # down: output rows are hidden; K is intermediate.
FC2 INT3 bytes: [E, H, ceil(3*I/8)]
FC2 INT4 bytes: [E, H, ceil(4*I/8)]
FC2 scales:     [E, H, ceil(I/block_size)]
```

Select each FC's packed shape, decoder, zero-point semantics, and runtime prepack independently from its effective width. INT3 follows this proposal in either projection; INT4 retains its existing contract in either projection. The three required combinations share the logical ordering and scale shapes above; an INT3 FC2 must not use FC1's H-based K stride or fused gate/up layout.

Scale expert/output-row ordering MUST match the weights. The portable format permits only `float32/float16/bfloat16` scales; FP8 and integer scales are prohibited. The first ORT execution profile requires FC1/FC2 scales to share a dtype matching the FP16/BF16 activation. FP32 scales or independent FC scale-dtype combinations require subsequent schema/provider qualification review. Unsupported combinations must be explicitly rejected, not reinterpreted. The baseline profile does not add implicit alternative shapes for row-wise scales.

Export fixtures MUST verify gate/up ordering and the down-projection transpose, not just packed element counts. The first exporter constructs native fused QMoE; automatic detection and fusion from a dense graph is a later stage.

### 2.5 Zero Points, Mixed Widths, and Versioning

The initial profile accepts only omitted zero points or explicit INT3 zero points whose logical codes are all 4. When present, independently pack the B block codes for each `[e,n]` row LSB-first: `Z shape=[E,N,ceil(3*B/8)]`, with unused high bits in the final byte zero. This is **not** a single flattened bitstream for the whole zero-point tensor without row boundaries.

Each `[e,n]` is a contiguous bitstream of B codes: `Z_row_bytes=ceil(3*B/8)`, row byte offset `(e*N+n)*Z_row_bytes`, and starting bit `3*b` for block b. Reuse the cross-byte extraction rule in Section 2.2. With Z omitted, the decoder uses 4 for every block. With explicit Z, the validator must check that every logical code equals 4 and reject non-4 codes or nonzero tail bits. Both representations produce identical outputs. The default exporter omits Z to save space; the explicit form serves interoperability and future-extension validation.

The general asymmetric formula is `(u-z)*S`, but arbitrary `z` requires a later profile/schema review and must not be silently accepted in the initial profile. An INT4 projection, whether FC1 or FC2, retains its approved INT4 zero-point semantics; do not apply INT3's default value 4 to INT4. Apply the INT3 omitted/explicit zero-point rules independently to every INT3 projection, including FC2 in `(3,3)` and `(4,3)`.

FC widths retain the independent optional-override design: omitted `fc1/fc2/fc3_expert_weight_bits` inherit `expert_weight_bits`, whose existing default remains 4. Fused FC3 has the FC1 width because its data resides in the FC1 tensor. Effective width 3 is permitted only after a dedicated schema revision is reviewed. Adding INT3 must not reinterpret existing models.

Do not use `pack_size=8/bits` to calculate INT3 shapes/strides. The general rule is `ceil(K*bits/8)`; existing INT2/4/8 must retain byte and numerical compatibility. A format change that changes the meaning of model bytes MUST have an explicit schema/opset version or an approved format identifier. Exporter version or GPU architecture alone must not determine interpretation.

### 2.6 Golden Vectors and Verifiable Examples

```text
q:     [-4,-3,-2,-1,0,1,2,3]
u:     [ 0, 1, 2, 3,4,5,6,7]
bytes: [0x88, 0xc6, 0xfa]

K=3, q=[-4,0,3], u=[0,4,7]
bytes: [0xe0,0x01]  # Bits 1..7 of the second byte are zero padding.

All-zero logical weights for K=8, with every u equal to 4
bytes: [0x24,0x49,0x92]
```

Provide machine-readable fixtures containing logical shape, width, block size, codes, scales, zero points, expected raw bytes, explicit dequantized weights, and QMoE outputs. Cover every code, byte and block crossings, multiple rows/experts, tails, zero groups, truncation, and invalid padding. At least two independent pack/unpack implementations must agree byte-for-byte. Do not use the same potentially faulty helper to produce both expected and actual values.

## 3. CUDA Internal Layout: A 2+1 Bit-Plane Candidate

**Models still use the contiguous bitstream in Section 2.** CUDA may convert it during session initialization into a low-two-bit plane and a high-one-bit plane:

```text
lo2[k] = u[k] & 3
hi1[k] = (u[k] >> 2) & 1
u[k] = lo2[k] | (hi1[k] << 2)
q[k] = u[k] - 4
```

This borrows the physical low/high-bit separation idea from llama.cpp `Q3_K`, not its hierarchical scales, quantization algorithm, or GGUF ABI. The IQ3 codebook format is not uniform INT3 either. Each plane may have its own tiling, interleaving, alignment, and provider descriptor, but descriptors must distinguish width, layout version, architecture, logical shape, and padding. Cache keys include inputs and version information that affect semantics.

Choose tile shapes, byte alignment, plane offsets, and fused layouts after kernel-prototype measurements; these are not committee-format contracts. Do not export this internal cache as `weights_prepacked=0`, or promise portable offline prepack in the initial profile. GPU-memory accounting must include coexistence of raw weights and prepack caches plus extra padding, not just the three-bit payload.

Prioritize FP16/BF16 activation. Packed decode extracts codes in place, applies scales, and accumulates in FP32. DP4A/INT8 Tensor Core paths require separate activation quantization and accuracy qualification and must not become baseline INT3-format dependencies. A100/H200 do not provide the native packed INT3 operation needed here; hardware INT8 support does not eliminate decoding or imply automatic speedup.

## 4. Workstreams, PR Boundaries, and Acceptance Gates

All stages are currently **Planned**. This is a dependency-driven plan, not a merged implementation or a delivery-date commitment from other teams. PR labels below are proposed sequence labels, not actual GitHub PR numbers; none of these INT3 PRs is claimed to be opened or merged.

| Stage / proposed PR label | Main deliverables | Exit gate |
| --- | --- | --- |
| P0 / PR 0: Format and feasibility | Review Section 2; freeze fixtures; compare Olive's existing INT3 checkpoint format with this contract | Exporter/runtime owners agree on bytes and mathematical semantics; independent pack/unpack agrees for every code, tails, and zero groups; do not directly reuse unknown checkpoint packing |
| P1 / PR 1: QMoE contract and shared validation | Reviewed INT3 allowance in either FC; ceil-bit sizes/strides; shape inference; tail and zero-point validation | All `(3,4)`, `(3,3)`, and `(4,3)` models validate; old INT2/4/8 fixtures remain fully compatible; unsupported providers reject clearly |
| P2 / PR 2: Independent reference and CUDA correctness | Scalar FP32 oracle; FP16/BF16 numerical references; bounded expert/block dequantization; decode and prefill | Synthetic/reduced Qwen correctness; no unbounded full-expert dequantization; actual provider-target builds/tests pass |
| P3 / PR 3: Packed CUDA decode | Raw bitstream to versioned prepack; in-place INT3 load/decode for FC1 and FC2; independent INT3/INT4 dispatch | All three baseline combinations pass target routing/shape parity and memcheck; no large dequantized buffer; record prepack and repeated-call costs |
| P4a / PR 4a: Baseline prefill | Selected-expert or row/block-tiled dequantization + existing GEMM; explicit scratch budget | Long prompts, partial tiles, and decode/prefill transitions work; bounded memory; do not label this native packed performance |
| P4b / PR 4b + PR 4c: Conditional prefill performance | Kernel foundation followed by QMoE integration; packed grouped kernel or measured local-conversion strategy, block64 first | End-to-end cost beats baseline; scratch estimates/bounds regressions pass; stop native optimization without a benefit |
| Follow-up / PR 4d: Optional coverage | Additional optimized block sizes, combinations beyond the three baseline combinations, and provider qualification | Qualify each added configuration; no new mixed-width defaults or implied untested coverage; not a baseline dependency |
| P5 / PR 5: Olive/Mobius export | Expert classification, mixed checkpoint, packing converter, fused graph, external data, manifest | Checkpoint-to-ONNX-to-ORT parity; correct width binding and gate/up order; reproducible source revisions/tool versions |
| P6 / PR 6: CPU parity | CPU reference execution and invalid-model validation on the same raw bytes | CPU/CUDA and independent oracle agree within specified tolerances; no CPU production-throughput promise |
| P7 / PR 7: Model qualification | Matched-recipe INT2/INT3/INT4 quality, capacity, prefill, and decode measurements | Meet pre-agreed quality/capacity targets; repeatable performance conclusions; explicit fail/stop decision |
| P8 / PR 8: Committee package | CUDA-independent specification, reference, interoperability fixtures, schema-version proposal | ONNX review determines container/operator/version paths; ORT merge is not ONNX approval |

P3/PR 3 and P4a/PR 4a may proceed in parallel after P2/PR 2. P5/PR 5 and P6/PR 6 may proceed in parallel after P1 and its shared fixtures are frozen, coordinating executable parity gates with CUDA readiness. P4b/PR 4b and PR 4c are conditional performance work, not prerequisites for a basic model. PR 4d is optional follow-up. With one engineer, start with P0/P1/P2, complete one small Qwen export loop, then implement packed decode and baseline prefill. Do not introduce INT8 activation, INT3 codebooks, and multiple GPU specializations simultaneously.

### 4.1 Ownership and Scheduling

P0/P1 require joint approval from the ORT operator owner and Olive/Mobius exporter owners. The CUDA owner is responsible for P2/P3/P4; the CPU owner for P6. Model-quality and benchmark owners review P7, and the ONNX proposal sponsor coordinates P8. These are roles awaiting assignment, not named individuals or commitments from other teams. PR 4d requires owners for each added provider/configuration.

At P0 completion, estimate engineering effort separately for schema/reference, CUDA correctness, packed decode, baseline prefill, export, and model qualification. Assign calendar dates only after considering staffing and actual PR review throughput. Track native prefill and committee approval separately, outside the baseline's hard deadline. Weekly updates should report only closed gates, current blockers, and the next verifiable deliverable. ONNX standardization must not block ORT contrib design experiments, and ORT delivery must not be presented as completion of standardization.

### 4.2 Distinguish Baseline Completion from Production Qualification

Baseline completion requires all three combinations `(3,4)`, `(3,3)`, and `(4,3)` to support source-checkpoint quantization/export, fixture/CPU/CUDA parity, packed CUDA decode, and model prefill with bounded scratch. Record a complete quality and capacity/speed baseline for each combination on the initial target. Neither a working `(3,4)` model alone nor kernel unit tests alone are sufficient. A dequantization correctness fallback is not a performance delivery.

Production qualification means that packed paths and dispatch are demonstrated for actual deployment shapes; prefill/decode/memory have no unexplained regressions; repeated measurements and accuracy targets pass; and fallback, concurrency, capture, scale updates, and prepack lifetime are validated for the target deployment. SM80 code or dispatch conditions do not establish qualification on all SM80+ hardware. PR 4b/4c ship only if evidence supports a winning design; their absence does not prevent baseline completion, but production claims must stay within demonstrated paths and performance gates.

## Delivery Sequence

Every heading in this section uses a **proposed label**, not an actual PR number. All entries are planned, not opened or merged. PR 5 may comprise coordinated PRs across repositories. Numbering identifies scope, not a mandatory serial execution order; no implementation PR URLs are assigned here.

### PR 0: Portable Format and Golden Fixtures - Proposed (P0)

- Freeze the unsigned codes, offset-4 semantics, row-local LSB-first raw bytes, scale/zero-point rules, FC ordering, and portable tail definitions in Section 2 with exporter/runtime-owner review.
- Add machine-readable golden fixtures and two independent pack/unpack implementations covering every code, byte/block crossings, row/expert boundaries, tails, zero groups, and malformed inputs.
- Audit Olive's INT3 checkpoint representation and identify explicit canonical conversion requirements; unknown or Q3_K/IQ3 packing is not interchangeable with this format.
- Exit gate: byte-for-byte agreement and agreed mathematical semantics. This establishes a proposed portable contract, not current ORT tail support, approved ONNX datatypes, or runtime execution.

### PR 1: INT3 QMoE Contract and Shared Validation - Proposed (P1)

- Depends on PR 0's frozen contract/fixtures; review the schema revision permitting effective width 3 and its version/format-identifier requirements.
- Implement shared ceil-bit shape/stride calculations with overflow checks, shape inference, raw `uint8` validation, explicit INT3 zero-point checks, and clear unsupported-provider diagnostics.
- Preserve default `expert_weight_bits=4`, independent FC override inheritance, fused FC3/FC1 agreement, and byte/numerical compatibility for INT2/4/8.
- Gate tail allowance through schema review; keep the first ORT model's H/I divisibility constraints and reject unapproved tail models at export. Enforce the initial matching FP16/BF16 activation/FC-scale execution profile.
- Exit gate: valid `(3,4)`, `(3,3)`, and `(4,3)` models validate and invalid models reject before computation, with old-model regression tests. Cover width-specific FC1/FC2 shapes and zero points independently. Schema acceptance alone is not provider execution support.

### PR 2: CUDA Correctness and Bounded Fallback - Proposed (P2)

- Depends on PR 1 and shared fixtures; implement an independent scalar FP32 oracle and separate FP16/BF16 numerical references.
- Provide bounded expert/block dequantization and CUDA correctness execution for both decode and prefill for `(3,4)`, `(3,3)`, and `(4,3)`, preserving routing, gate/up interleaving, SwiGLU, bias, and FC2 orientation. Decode INT3 or INT4 independently in each projection.
- Test synthetic and reduced Qwen cases, invalid inputs, tails only where schema-approved, and fallback scratch bounds; prohibit unbounded full-expert dequantization.
- Exit gate: oracle parity within agreed tolerances and builds/tests for the actual provider targets. Independent harness results do not qualify legacy/plugin integration or untested providers.
- This is functional correctness, not a packed-performance claim, INT8-activation path, or all-GPU support claim.

### PR 3: Packed CUDA Decode - Proposed (P3)

- Depends on PR 2; convert canonical raw bytes to a versioned provider-private prepack, evaluating the internal 2+1 planes without changing model serialization.
- Implement in-place INT3 extraction, scale application, and FP32 accumulation for FP16/BF16 in both fused FC1 and down-projection FC2. Reuse existing INT4 paths where applicable and dispatch each projection by its effective width for `(3,4)`, `(3,3)`, and `(4,3)`.
- Cover routing/shapes, cached/runtime scales, cache identity and lifetime, repeated calls, memcheck, and bounded fallback; no large dequantized weight buffer.
- Exit gate: packed decode parity and memcheck for all three combinations, with measured prepack/repeated-call costs including raw/cache coexistence and padding. A correctness-only FC2 INT3 fallback does not complete this packed-decode gate. Architecture eligibility is not qualification on every device.
- Do not export provider caches as raw weights, promise portable offline prepack, or require INT8 activation.

### PR 4a: Bounded Prefill Baseline - Proposed (P4a)

- Depends on PR 2 and may proceed alongside PR 3; implement selected-expert or row/block-tiled dequantization with existing GEMM and an explicit scratch budget.
- Validate `(3,4)`, `(3,3)`, and `(4,3)` with long prompts, partial tiles, empty experts, FP16/BF16, and decode-to-prefill-to-decode transitions, coordinating cached-scale/prepack behavior with PR 3. INT3 FC2 execution is required here, not deferred to PR 4d.
- Exit gate: correct model prefill, enforceable scratch bounds, scratch-guard tests, and actual provider-target coverage.
- Required for the usable baseline, but not a native packed grouped-GEMM delivery or a promise to beat INT2/INT4. It remains the comparison/fallback path for conditional performance work.

### PR 4b: Native Packed Prefill Kernel Foundation - Proposed (Performance P4b)

- Conditional on PR 0/1's frozen contract, PR 2 correctness, and PR 4a measurements supporting a viable performance design; start with block64 FP16/BF16 on the initial SM80/A100 target.
- Prototype a packed grouped kernel and compare measured local-conversion alternatives. Decide tiles, plane layout, alignment, and bounded workspace from evidence, including INT3 use in both fused FC1 and FC2 down and compatibility with either projection's INT4 path.
- Deliver independent kernel unit tests, scalar-oracle parity, scratch-bound regressions, sanitizer checks, and kernel benchmarks with full conversion costs disclosed.
- Keep QMoE default dispatch unchanged. This PR is the kernel foundation only; routing, provider dispatch, lifetime integration, and model qualification belong to PR 4c and PR 7.
- Exit gate: a correct, maintainable candidate with evidence justifying integration. Stop native optimization if no design wins; this PR is not required for a basic model and must not imply end-to-end production speedup.

### PR 4c: QMoE Packed Prefill Integration - Proposed (Performance P4b)

- Conditional on an accepted PR 4b candidate and PR 3/4a readiness; integrate independent FC1/FC2 INT3/INT4 dispatch for `(3,4)`, `(3,3)`, and `(4,3)`, including routing, activation/finalization, workspace, and prepack lifetime. Qualify optimized eligibility per combination and preserve PR 4a for any combination outside the winning kernel's support boundary.
- Add actual legacy/plugin provider integration tests as applicable, including eligibility/fallback, cached/runtime scales, partial row tiles, and decode/prefill transitions. Kernel unit tests alone do not satisfy this gate.
- Preserve packed decode and bounded fallback for unsupported configurations; enable optimized dispatch only within the validated support boundary and after the performance decision.
- Exit gate: end-to-end cost beats PR 4a, with scratch estimates/bounds regressions passing and repeatable unprofiled measurements. Report conversion, raw/prepack memory, and host orchestration costs.
- Do not ship or enable a losing native design merely to complete the sequence. This is conditional production-performance work, not a baseline dependency or blanket hardware/provider claim.

### PR 4d: Optional Block Sizes/Width Combinations and Provider Coverage - Proposed (Follow-Up)

- Follow relevant completed contract/correctness gates and, when extending native packed prefill, PR 4b/4c. This optional work must not block the block64 baseline for any of `(3,4)`, `(3,3)`, and `(4,3)`.
- Separately qualify optimized block32/128 dispatch, combinations beyond the three required baseline combinations, additional architectures/providers, or separately reviewed fusion/dtype extensions; format definitions alone do not prove execution support. Required INT3 FC2 support belongs to PR 1/2/3/4a/5/6/7, not this optional PR.
- Preserve existing default 4 and FC override inheritance, fused FC3/FC1 agreement, approved zero-point semantics, and explicit unsupported diagnostics. Do not assume new mixed-schema defaults or relax divisibility without approval.
- Exit gate: correctness, actual provider builds/tests, measured performance, scratch/lifetime checks, and a support-matrix entry for each added configuration; distinguish untested plugin/device paths.
- No INT8-activation, asymmetric-profile, portable-prepack, or every-GPU commitment follows automatically from this PR label.

### PR 5: Olive/Mobius Export - Proposed (P5)

- Depends on PR 1's contract and frozen fixtures; may proceed in parallel with CUDA/CPU work, with final execution parity gated on the corresponding runtime readiness. Use coordinated repository PRs if needed.
- Olive scope: qualify native INT3 checkpoint capability, classify expert/non-expert tensors, provide recipes for `(3,4)`, `(3,3)`, and `(4,3)`, convert checkpoint packing explicitly to canonical raw bytes for either INT3 projection, and emit a reproducible manifest with quantization/rounding/clipping, scale dtype, widths, shapes, block size, and source/tool revisions.
- Mobius scope: construct native fused QMoE graphs, bind gate/up-interleaved FC1 and correctly oriented FC2 initializers, manage external data, and explicitly emit `quant_type='int'`, `weights_prepacked=0`, block size, and effective widths.
- Reject schema-unapproved tails/dtype combinations and validate graph/initializer/external-data bindings against shared fixtures; native INT3 linear support does not establish fused QMoE export support.
- Exit gate: checkpoint-to-ONNX-to-ORT numerical parity and reproducibility for all three required combinations. Automatic dense-graph fusion, unknown checkpoint-byte reuse, and native packed prefill are not dependencies or implied deliverables.

### PR 6: CPU Reference and Cross-Provider Parity - Proposed (P6)

- Depends on PR 1's frozen contract/fixtures and may proceed alongside PR 5 and CUDA work; coordinate CPU/CUDA comparisons with executable CUDA readiness.
- Execute the same canonical raw bytes through a CPU reference for `(3,4)`, `(3,3)`, and `(4,3)`, including independent projection decoding, routing, SwiGLU, bias, scale/zero-point rules, and schema-approved shapes.
- Add invalid-model checks, scalar-oracle comparisons, and CPU/CUDA parity with explicitly agreed FP16/BF16 tolerances and old-width regressions.
- Exit gate: reference and cross-provider results agree for every required combination, with tested provider scope recorded. CPU production throughput or optimized MLAS INT3 support is not promised by reference completion.

### PR 7: End-to-End Model Qualification - Proposed (P7)

- Depends on PR 3/4a/5/6 and their baseline gates; qualify PR 4c only if its conditional path ships. Optional PR 4d coverage is not required for initial qualification.
- Automate full conversion and inference for `(3,4)`, `(3,3)`, and `(4,3)` with fixed checkpoints/tokenizers, placement, block sizes, non-expert precision, and matched INT2/INT3/INT4 recipes plus a floating-point quality reference. Report per-combination quality, size, decode, prefill, and memory; selecting one preferred recipe does not waive execution/export qualification for the others.
- Freeze quality tolerances, effective-size benefits, and performance non-regression criteria before full-model acceptance; report tasks, layer/logit errors, load/prepack, TTFT, prefill, decode TPS, peak memory, and scratch.
- Exit gate: reproducible results meeting pre-agreed quality/capacity goals, explicit performance and fail/stop decisions, and documentation/support-matrix updates. Separate functional baseline completion from production qualification under Section 4.2.
- Do not infer quality from equal greedy tokens, performance from microbenchmarks, or deployment coverage from a dispatch condition; unqualified hardware, concurrency, capture, and lifetime cases remain explicitly uncovered.

### PR 8: ONNX Committee Proposal Package - Proposed (P8)

- Build on PR 0/1's reviewed format and versioning proposals, independent decoders and PR 5/6 interoperability; include actual PR 7 quality/capacity/performance evidence for submission. Drafting can proceed earlier.
- Assemble a CUDA-independent specification, mathematical semantics, golden bytes including tails/zero points, shape inference, invalid-model definitions, at least two independent decoders, standard-graph reference decomposition, and interoperability reports.
- Propose `uint8` packed-input container semantics with existing external data. Do not request native INT3/UINT3 datatype identifiers in this delivery sequence; Section 7.1 explains the deferral and reconsideration criteria. A packed operator and standardization of fused MoE remain separate review decisions.
- Exit gate: submit a reviewable package and record the committee's container/operator/version decisions; unresolved decisions remain open. ORT contrib merges do not approve an ONNX INT3 datatype, standard QMoE, or standard opset.
- Ownership and timing remain conditional on an assigned proposal sponsor and committee review; no approval deadline or outcome is promised.

## 5. Validation Matrix and Performance Method

- Format: codes 0..7, byte crossings, K tails, block tails, multiple experts/output rows, all-zero groups, scale dtypes, truncated external data, overflow, invalid shapes/zero points/padding.
- Operator: cross all `(3,4)`, `(3,3)`, and `(4,3)` combinations with FP16/BF16, top-k 1/2/8, single/multiple tokens, empty experts, bias, routing permutations, SwiGLU parameters, FC2 orientation, and old-model regressions; reject invalid inputs before computation. Prove each FC uses its own bit width, K stride, scale indexing, zero point, and prepack layout.
- Kernel: packed eligibility and fallback; actual decode-to-prefill-to-decode transitions, partial row tiles, cached/runtime scales, preprocessing reuse, scratch guards, and sanitizers. Independent harnesses do not replace actual legacy/plugin provider targets; explicitly mark untested plugin coverage.
- Tools: first verify Olive's native INT3 checkpoint capability, then qualify fused-expert coverage and Mobius graph/initializer/external-data binding. Native INT3 linear support is not QMoE export support under this contract.
- Models: fixed checkpoint/tokenizer, tensor placement, block size, and non-expert precision across INT2/INT3/INT4, plus a floating-point quality reference. Do not mask width effects with different blocks or quantized layers.
- Performance: same GPU/runtime, profiling disabled, forward/reverse or alternating A/B runs; separately measure load/prepack, TTFT, prefill, decode TPS, peak GPU memory, and scratch. Record host-orchestration scope. Use separate profiling to prove dispatch, not to report unprofiled TPS.
- Accuracy: fixed tasks/samples, separate calibration/evaluation splits, layer/logit errors, and full task metrics. Elementwise equality or equal greedy tokens is not a substitute for quality qualification.

Before full-model acceptance, freeze task-quality tolerances, effective-size benefits, and performance non-regression criteria with product and exporter/runtime owners. Do not move thresholds after seeing results. Report throughput changes smaller than measurement variation as inconclusive, not successful.

## 6. Effective Storage Cost and Stop Criteria

For each `[E,N,K]` INT3 tensor, the weight payload is `E*N*ceil(3*K/8)` bytes and scales occupy `E*N*ceil(K/block_size)*sizeof(scale)`. Account separately for explicit zero points, padding, manifests, unquantized parameters, and external-data alignment.

For example, with aligned K, FP16 scales, no explicit zero point, and block64, the effective weight-plus-scale rate is `3 + 16/64 = 3.25 bit/weight`, not exactly 3. The corresponding INT4 rate under the same scale policy is 4.25 bit/weight. Do not present the nominal 25% payload saving as whole-model or GPU-memory saving.

Stop or adjust if quality offers no meaningful value over INT2; effective capacity offers insufficient benefit over INT4; prepack/raw caches or temporary expansion consume the capacity benefit; packed extraction offsets bandwidth gains; microbenchmarks improve while full models regress; or exporter/runtime byte interpretations cannot interoperate. Without measured benefit, retain baseline correctness and the results rather than add unmaintainable optimization special cases just to showcase a new width.

## 7. ONNX Committee Submission Package and Open Decisions

The package should include an independent packed-format specification and mathematical semantics, golden bytes including tails/zero points, shape inference and invalid-model definitions, at least two independent decoders, exporter/runtime interoperability reports, a standard-graph reference decomposition, and real performance/quality/capacity evidence.

Prefer a `uint8` container with explicit operator semantics for opaque packed inputs and existing ONNX external data. Adding native INT3/UINT3 datatypes is a separate decision. A three-bit bitstream cannot be passed directly to existing `DequantizeLinear` as ordinary uint8 elements; it requires an explicitly decoding reference decomposition or an approved packed operator. Whether to standardize the full fused MoE operator or first a general packed quantization representation/operation is also a separate discussion.

### 7.1 Why Native INT3/UINT3 Datatype Standardization Is Deferred

**Decision: do not propose new native ONNX INT3/UINT3 tensor datatypes as part of the current delivery plan.** The committee package still proposes portable packed-weight semantics and interoperability evidence; deferring a datatype proposal does not mean abandoning format review or treating a contrib contract as an ONNX standard.

- The immediate use case is operator-consumed packed weights, not general-purpose three-bit graph tensors. All three baseline combinations `(3,4)`, `(3,3)`, and `(4,3)` can use `uint8` containers with explicit logical dimensions, effective widths, packing, scales, and zero-point semantics. Native datatype registration is not required to implement or export them.
- A native datatype would not reduce the existing three-bit payload, provide native packed INT3 CUDA arithmetic, or remove extraction, scaling, prepacking, and dispatch work. These costs and benefits must be established by the actual runtime implementation; a datatype enum alone cannot deliver them.
- The scope is substantially larger than adding enum entries. A proposal must define numerical encoding, logical tensor shapes, raw and typed-field serialization, padding, and IR compatibility, then coordinate checker, shape/type inference, tensor utilities, exporter/runtime support, and relevant operator type constraints. Datatype acceptance does not automatically provide standard quantization operators or kernels.
- The format and product evidence are not yet mature. First establish useful quality/effective-size/runtime trade-offs and independent exporter/decoder interoperability. Premature datatype standardization risks fixing semantics before representative use cases and implementation costs are understood.
- Our codes use offset-binary `q=u-4`, not signed two's-complement. A future native signed INT3 type must define its encoding independently. Our row-local packed shape and padding rules are also operator-format choices, not automatically a general tensor serialization rule. Existing model bytes must not be relabeled as native INT3 without an explicitly reviewed conversion and versioning contract.

ONNX already defines `UINT2=25` and `INT2=26` in IR version 13; native INT2 uses two's-complement encoding. See the [ONNX protobuf definition](https://github.com/onnx/onnx/blob/main/onnx/onnx.in.proto). This is distinct from the current QMoE integer-weight input, which uses a packed `tensor(uint8)` container. The existence of native INT2 is therefore not a requirement to follow the same datatype route for QMoE INT3, nor evidence that every runtime/operator supports native INT2. ONNX does not currently define native INT3/UINT3.

Reconsider a separate native datatype proposal when there is demonstrated demand for three-bit tensors across multiple independent frameworks/runtimes or standard operators beyond this QMoE packed-input use case; reproducible model evidence supporting adoption; and agreement on numerical encoding, general tensor serialization, migration, and an operator/tooling support plan. These are triggers for review, not claims of current adoption or a guarantee of committee approval.

Until then, PR 0/1/5 establish the portable packed contract and its ORT/export integration, PR 7 supplies measured product evidence, and PR 8 submits the format/operator discussion package without making native datatype approval a baseline dependency. Any future INT3/UINT3 datatype proposal should be tracked separately and must not retroactively reinterpret existing packed models.

### 7.2 Open Decisions for the Current Proposal

Open approval questions include whether the schema upgrade needs a new contrib version; acceptance of the initial symmetric-only profile, block32/64/128 set, and tail rules; explicit zero-point extension strategy; scale-dtype constraints; specification name/identifier; reference decomposition and shape inference; and diagnostics for older exporters and unsupported providers. Do not claim an ONNX standard is frozen before review resolves these questions.

## 8. Initial Execution Checklist

1. Review and freeze the portable raw bitstream, code/scale/zero-point semantics, and FC1/FC2 golden fixtures.
2. Audit existing Olive INT3 checkpoints; add explicit canonical-bitstream conversion and shared-fixture qualification tests.
3. After independent-reference and schema validation pass, implement bounded CUDA correctness and complete small decode/prefill export loops for `(3,4)`, `(3,3)`, and `(4,3)`; `(3,4)` may be the first checkpoint, not the baseline exit gate.
4. Implement packed decode and baseline prefill for every required combination with FP16/BF16 activation, then measure each Qwen recipe's quality, capacity, and speed against matched INT2/INT4 references.
5. Use the results to choose native prefill, other GPUs, or integer-activation research. The baseline need not wait for these experiments.

This is a design document; no new INT3 kernel, model, or CI runs were performed for this plan. Each subsequent implementation PR must record the actual source revision, build configuration, provider, hardware, test results, and uncovered scope. INT2 or local-prototype results cannot serve as INT3 qualification evidence.