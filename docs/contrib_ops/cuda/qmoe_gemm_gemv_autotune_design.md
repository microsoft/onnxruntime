# QMoE GEMM/GEMV Autotuning Design

This document sketches a QMoE operator-level autotuner for choosing between
CUTLASS grouped GEMM and the custom MoE GEMV fast path for known QMoE input
shapes. The goal is not to pick a global default. The goal is to profile valid
execution routes for the current operator configuration and cache the best route
for the given shape.

Related evidence is recorded in
[qmoe_gemv_experiments.md](qmoe_gemv_experiments.md).

## Goals

- Include grouped GEMM and GEMV as first-class tuning candidates.
- Tune inside the QMoE CUDA operator, where block size, quantization mode,
  activation dtype, weight layout, and input shape are known.
- Preserve existing behavior when tuning is disabled, unsupported, or fails.
- Reuse the existing grouped-GEMM tactic profiler where possible.
- Cache tuning results per shape/configuration so profiling runs once per key.
- Keep the first implementation focused on `quant_type="int"`, INT4/INT8,
  FP16/BF16 activations, SM80+, and single-EP decode/prefill shapes.

## Non-Goals

- Do not choose arbitrary quantization block sizes inside the operator unless
  multiple packed weight layouts are available. Today `block_size` is part of the
  model's quantized weight format and QMoE only receives one layout.
- Do not replace the existing CUTLASS GEMM tactic profiler. The route autotuner
  should sit above it.
- Do not make GEMV the only candidate. Some shapes and quantization modes can be
  faster on grouped GEMM.
- Avoid profiling the most latency-sensitive single-row decode path by default.
  Route tuning is enabled by default for `rows > 1`; single-row tuning remains
  available through an explicit override.

## Current State

QMoE already profiles grouped-GEMM tactics in
`onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.*`:

- `MoeGemmProfiler` caches the best CUTLASS `CutlassGemmConfig` by
  `(MoeGemmId, M bucket)`.
- `QMoE::ComputeInternal` profiles and retrieves separate GEMM tactics for FC1
  and FC2, then calls `m_moe_runner->runMoe(...)`.
- `CutlassMoeFCRunner::gemm1` and `gemm2` try the custom GEMV path before
  falling back to grouped GEMM when the GEMV gate rejects the shape.
- `ORT_DISABLE_MOE_GEMV=1` forces the fallback route, but it is process-global
  and cached after first use. It is useful for tests, not for in-operator tuning.

The missing layer is a route tuner that can compare:

- grouped GEMM for both FCs;
- GEMV for supported FCs;
- mixed routes, such as GEMV for FC1 and GEMM for FC2, or the reverse;
- future GEMV kernel variants, if multiple compiled GEMV configurations are
  added later.

## Tuning Key

The route cache key should include only values that affect route validity or
performance. A first version can use:

| Field | Reason |
|-------|--------|
| `sm` | GEMM tactics and GEMV support differ by architecture. |
| activation dtype | FP16 and BF16 have different kernels and latency. |
| quant type | Initial scope is `int`; future FP4/FP8 paths differ. |
| expert weight bits | INT4 and INT8 have different GEMV tile and packing rules. |
| block size | Fixed by the model; affects scale layout and GEMV support. |
| has zero points | Asymmetric block-wise INT currently forces grouped GEMM. |
| num rows | Runtime input shape. |
| expanded rows | `num_rows * top_k`; key GEMV gating dimension. |
| hidden size | FC1 K and FC2 N. |
| intermediate size | FC1 logical inter size and FC2 K. |
| top-k | Affects expanded rows and routing cost. |
| num experts per node | Affects grouped GEMM/routing shape. |
| activation type | SwiGLU fusion changes FC1 route and output shape. |
| `swiglu_fusion` | Interleaved SwiGLU can use fused GEMV. |
| EP/TP parallelism | Token routing/all-to-all can invalidate GEMV assumptions. |
| bias/zero-point state | Affects fused route validity and grouped GEMM work. |

`MoeGemmProfiler::bucketM()` already buckets the row count for GEMM tactics. The
route tuner should reuse the same power-of-two bucketing for the row dimension in
its key rather than keying on exact `num_rows`/`expanded_rows`. Route preference
is very unlikely to flip inside a single power-of-two bucket, and exact keying
would re-profile on every new prefill length, defeating the cache. Key on:

- `bucketM(num_rows)` for the GEMM-side row dimension;
- `bucketM(expanded_num_rows)` for the GEMV gating dimension.

Keep the exact `expanded_num_rows` in the side-effect-free GEMV validation, but
use the bucketed value in the cache key so a decode bucket and a prefill bucket
each get one tuning pass. The old `(0, 8]` decode limit is a legacy heuristic for
the default `Auto` route, not a hard validity boundary for explicit autotune
GEMV candidates.

### Key Struct Sketch

```cpp
struct QMoERouteTuningKey {
  int sm;
  nvinfer::DataType act_dtype;     // kHALF / kBF16
  int expert_weight_bits;          // 4 or 8
  int64_t block_size;              // -1, 0, 32, 64, 128
  bool has_zero_points;            // asymmetric block-wise INT
  int row_bucket;                  // bucketM(num_rows)
  int expanded_row_bucket;         // bucketM(expanded_num_rows)
  int64_t hidden_size;
  int64_t inter_size;
  int top_k;
  int num_experts_per_node;
  ActivationType activation_type;
  int swiglu_fusion;
  int ep_size;                     // parallelism, gates GEMV today

  bool operator==(QMoERouteTuningKey const&) const = default;
};

struct QMoERouteTuningKeyHash {
  size_t operator()(QMoERouteTuningKey const& k) const;  // combine fields
};
```

`quant_type` does not need to be a separate field in the first version because it
is already implied by `act_dtype` + `expert_weight_bits` + the int-only scope;
add it explicitly when non-int modes become candidates.

## Candidate Model

Define route candidates at the FC level, then compose an MoE route from FC1 and
FC2 choices. The single route enum used everywhere in this design is:

```cpp
enum class MoeFcRoute {
  kAuto,                  // current default dispatch (heuristic GEMV-or-GEMM)
  kGroupedGemm,           // force grouped GEMM, skip GEMV attempts
  kGemv,                  // force plain GEMV (FC1 non-fused, FC2)
  kGemvInterleavedSwiGlu, // force fused FC1 SwiGLU GEMV (FC1 only)
};
```

A concrete candidate composes two FC routes plus the GEMM tactics needed for any
grouped-GEMM portion. Candidates never use `kAuto` for an FC; `kAuto` exists only
as the default execution policy when tuning is disabled.

```cpp
struct MoeRouteCandidate {
  MoeFcRoute fc1_route;   // not kAuto
  MoeFcRoute fc2_route;   // not kAuto
  std::optional<CutlassGemmConfig> fc1_gemm_config;
  std::optional<CutlassGemmConfig> fc2_gemm_config;
};
```

Initial candidates:

| Candidate | FC1 | FC2 | Notes |
|-----------|-----|-----|-------|
| `Auto` | legacy heuristic dispatch | legacy heuristic dispatch | Baseline matching the default route. |
| `GemmOnly` | grouped GEMM | grouped GEMM | Always valid if runner/tactics exist. |
| `GemvWhereSupported` | fused GEMV or plain GEMV when kernel-supported | GEMV when kernel-supported | Bypasses legacy profiling thresholds. |
| `Fc1GemvFc2Gemm` | GEMV when valid | grouped GEMM | Useful if FC2 GEMV loses for a shape. |
| `Fc1GemmFc2Gemv` | grouped GEMM | GEMV when valid | Useful if FC1 fused GEMV loses. |

Future candidates can add multiple GEMV variants, for example different
`CtaN`/thread tile variants, but the route tuner should not require those in the
first implementation.

## Validity Rules

Candidate validation must happen before profiling.

Grouped GEMM is valid when:

- the runner has at least one tactic for the dtype/weight type;
- the existing `MoeGemmProfiler` can return valid FC1/FC2 configs;
- the current quantization mode is supported by the runner.

GEMV is valid when the kernel-support gate is valid and the route is explicitly
allowed by the candidate:

- activation/output dtype is FP16 or BF16;
- weight type is INT4 or INT8;
- scales/biases use the activation dtype;
- block-wise INT has no zero-point compensation;
- group size is `0`, `32`, `64`, or `128` and dimensions satisfy the existing
  alignment rules;
- expanded rows are positive and within the launch/indexing range;
- EP parallelism and TMA/activation-fusion states do not invalidate the route;
- FC1 fused interleaved SwiGLU is used only when `swiglu_fusion == 1`, activation
  is SwiGLU/SwiGLU-bias, and bias handling is compatible.

The existing `is_moe_gemv_supported(...)` should remain the low-level kernel
support gate. Legacy profile thresholds should be applied only by the default
`Auto` route; explicit autotune candidates should be profiled whenever the kernel
support gate accepts the shape.

## Profiling Approach

Use CUDA event timing. Avoid host wall-clock timing inside the operator.

### Scope Short-Circuit

Before doing any profiling, the tuner must check whether more than one candidate
is even possible. If the configuration has no valid GEMV candidate, there is
nothing to compare and the tuner must return `GemmOnly` immediately without
allocating profiling buffers or running candidates. This short-circuit applies
to:

- `quant_type` other than `int` (fp4/fp8/wfp4afp8 have no GEMV path today);
- asymmetric block-wise INT (zero points present);
- `expanded_num_rows` or problem dimensions outside the GEMV launch/indexing range;
- multi-EP / token-dropping configurations that disable GEMV;
- any shape where `is_moe_gemv_supported(...)` rejects both FCs.

The short-circuit result is still cached so the check runs once per key.

### Candidate Loop

For each uncached tuning key with at least two valid candidates:

1. Build or retrieve the best FC1/FC2 grouped-GEMM tactics with
   `MoeGemmProfiler`.
2. Enumerate valid route candidates.
3. Allocate scratch buffers large enough for the largest candidate.
4. Run a small warmup for each candidate.
5. Run a small timed loop for each candidate with CUDA events.
6. Select the lowest median or minimum average time.
7. Cache the selected route and any required GEMM tactics.

Suggested first-implementation settings:

| Setting | Value |
|---------|-------|
| Warmup iterations | 2 |
| Profile iterations | 5 |
| Metric | average CUDA event time, then later median of rounds |
| Tie band | prefer existing/default route within 3% |
| Failure behavior | skip failed candidate; if all fail, use default route |

The first implementation can profile the full `runMoe` route rather than timing
only FC kernels. Full-route timing includes routing, activation, finalization,
and workspace effects, which are exactly what the operator must optimize. Later,
if profiling overhead is too high, split out FC-only timing helpers.

### Profiling Buffer Safety

The production path calls `runMoe(..., output->MutableDataRaw())`, so naively
profiling candidates would write the user output buffer multiple times and race
with the selected run. The tuner must:

- run every candidate into a **scratch output buffer**, not the user output;
- reuse one scratch output buffer across candidates (sized for the user output);
- keep input, router probs, weights, scales, and quant params read-only;
- run the **selected** candidate once into the real user output after tuning, or
  cache the route and let the normal `runMoe` call apply it on this same forward.

Profiling reuses the real routing/permutation for the current input, so candidate
timings reflect the true per-expert token distribution rather than the synthetic
random routing the existing `GemmProfilerBackend` uses for GEMM tactic selection.
This is the main accuracy advantage of tuning inside the operator.

### Overhead and Concurrency

Profiling N candidates runs the full MoE pipeline N times on the first forward
for a new key. To keep this acceptable:

- profile once per key and cache aggressively (bucketed key dimensions);
- keep N small in the first version (`Auto`, `GemmOnly`, `GemvWhereSupported`,
  and mixed routes);
- hold the profiler mutex only while profiling and selecting, the same way
  `QMoE::ComputeInternal` already brackets `setTactic`/`runMoe`. Holding it during
  candidate runs serializes concurrent QMoE inferences for that one-time tuning
  pass, which is acceptable because it happens once per key.

The per-forward GEMM tactic profiling in `MoeGemmProfiler` already caches by
`(GemmId, M bucket)` and returns immediately after the first pass; the route
tuner adds one more cached layer on top and must not re-run GEMM tactic profiling
per candidate.

## Runner Changes

Today GEMV is controlled by heuristic checks embedded in `gemm1` and `gemm2`.
The route tuner needs explicit route control. Add a route policy (using the
`MoeFcRoute` enum defined in [Candidate Model](#candidate-model)) passed into
`runMoe`, then down to `gemm1`/`gemm2`:

```cpp
struct MoeRoutePolicy {
  MoeFcRoute fc1_route = MoeFcRoute::kAuto;
  MoeFcRoute fc2_route = MoeFcRoute::kAuto;
};
```

Behavior:

- `kAuto`: current behavior, preserving existing default dispatch.
- `kGroupedGemm`: skip all GEMV attempts for that FC.
- `kGemv`: attempt plain GEMV; if validation fails, the candidate is invalid
  during tuning. In normal execution, fallback to GEMM defensively.
- `kGemvInterleavedSwiGlu`: attempt fused FC1 SwiGLU GEMV only. Invalid for FC2.

This avoids using `ORT_DISABLE_MOE_GEMV` for tuning and makes GEMM/GEMV
selection local to one QMoE invocation.

## QMoE Integration

Add a new cache owned by `QMoE`:

```cpp
mutable QMoERouteTuner qmoe_route_tuner_;
```

The tuner can live near `MoeGemmProfiler` or in a new pair of files:

- `onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_route_tuner.h`
- `onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_route_tuner.cc`

Suggested QMoE flow:

```text
ComputeInternal
  validate inputs and prepare quant params
  profile/retrieve GEMM tactics as today
  build QMoERouteTuningKey
  if tuning enabled:
    selected_route = qmoe_route_tuner_.GetOrProfile(...)
  else:
    selected_route = Auto/current behavior
  allocate workspace based on selected route requirements
  set GEMM tactics
  runMoe(..., selected_route.policy)
```

The route tuner should use the same mutex as the GEMM profiler at first, because
`m_moe_runner->setTactic(...)` mutates runner state. A later refinement can
separate locks if the runner state becomes immutable during profiling.

## Workspace Sizing

The current workspace size comes from `m_moe_runner->getWorkspaceSize(...)` after
GEMM tactics are selected. For the first implementation:

- allocate the existing maximum workspace size for all candidates;
- reuse the same scratch layout used by normal `runMoe`;
- do not try to reduce workspace for GEMV-only candidates yet.

This keeps correctness simple. A later version can shrink workspace for a stable
GEMV route if memory pressure matters.

## Configuration Surface

Route tuning is enabled by default for multi-row QMoE inputs (`rows > 1`) and
skipped for single-row decode unless explicitly requested. Suggested controls:

| Control | Meaning |
|---------|---------|
| `ORT_QMOE_ROUTE_TUNING=0/1` | Master switch for GEMM/GEMV route tuning. Default on. |
| `ORT_QMOE_ROUTE_TUNING_ALL_ROWS=1` | Also tune single-row decode (`rows == 1`). Default off. |
| `ORT_QMOE_ROUTE_TUNING_LOG=1` | Log selected route, key, candidate times. |
| `ORT_QMOE_ROUTE_TUNING_FORCE=gemm/auto/gemv/fc1_gemv_fc2_gemm/fc1_gemm_fc2_gemv` | Debug override for route policy. |

Session config entries are preferable long-term, but env vars are faster for the
first profiling implementation and match the existing GEMV debug workflow.

`ORT_DISABLE_MOE_GEMV=1` should remain a stronger debug override. If it is set,
the route tuner should either be disabled or only profile grouped-GEMM
candidates.

## Cache Semantics

The route cache maps:

```text
QMoERouteTuningKey -> QMoERouteTuningResult
```

The result should include:

- selected `MoeRouteCandidate`;
- measured time for each valid candidate;
- rejection reason for each invalid candidate;
- GEMM tactic configs used by the selected result;
- a flag indicating whether tuning succeeded or fell back to default.

Cache lifetime can initially be per `QMoE` kernel instance. Cross-session
serialization is deferred.

## Correctness Requirements

- Tuning must not change the observable output.
- Candidate profiling must use temporary output buffers or tolerate overwriting
  only scratch outputs. The final user output should be written once by the
  selected candidate.
- Profiling must synchronize the profiling stream before reading CUDA event
  timings.
- Failed candidates must clear CUDA errors and must not poison the normal run.
- If profiling fails, use the current default `kAuto` route and current GEMM
  tactics.
- Deterministic mode should skip tuning and use the current deterministic tactic
  selection behavior.

## Measurement Policy

The route tuner should optimize end-to-end QMoE latency for the current operator
shape, not isolated FC kernel time. This is important because route choice can
change:

- FC1 fused SwiGLU behavior;
- whether a separate activation kernel runs;
- temporary workspace usage;
- finalization dependencies;
- routing/finalization overlap opportunities in future implementations.

Offline Nsight traces remain useful to explain a route choice, but the in-operator
autotuner should use CUDA event timing around the candidate execution.

## Implementation Milestones

1. **Route-control plumbing**
   - Add `MoeFcRoute`/`MoeRoutePolicy`.
   - Pass policy through `runMoe`, `gemm1`, and `gemm2`.
   - Preserve `kAuto` behavior exactly.
   - Add force-GEMM policy without using `ORT_DISABLE_MOE_GEMV`.

2. **Validation helpers**
   - Extract side-effect-free GEMV validation for FC1 and FC2.
   - Return structured rejection reasons for logging/tests.
   - Ensure asymmetric zero points and unsupported block sizes reject GEMV.

3. **Route tuner skeleton**
   - Add `QMoERouteTuningKey`, `QMoERouteTuningResult`, and cache.
   - Enumerate `GemmOnly`, `GemvWhereSupported`, and mixed candidates.
   - Integrate with the existing GEMM profiler configs.

4. **Candidate profiling**
   - Profile full `runMoe` candidates with CUDA events and temporary outputs.
   - Add tie-band handling and default-route fallback.
  - Add a master disable env var, an all-rows override, and optional logging.

5. **Tests and validation**
   - Unit-test key hashing/equality and candidate validation.
   - Add Python benchmark tests that force/tune routes for known shapes.
   - Verify tuned route agrees with offline sweep for representative shapes.
   - Capture Nsight traces for at least one case where GEMM is selected and one
     where GEMV is selected.

## Open Questions

- Should route selection be per full MoE op or per FC? Per-FC is more flexible,
  but full-op profiling should still choose the composed route.
- How many timed iterations are acceptable during first-run tuning? The answer
  may differ for decode and prefill.
- Do we need a persistent tuning artifact? In-memory cache is enough for the
  first version.
- Is the bucketed row keying coarse enough to ever pick a clearly wrong route at
  a bucket boundary (e.g. just above the GEMV gate)? The hard GEMV gate is still
  checked on the exact value, but a follow-up may want sub-bucket validation.

Resolved during this design (kept here for traceability):

- Cache key uses `bucketM`-bucketed row dimensions, not exact `num_rows`
  (see [Tuning Key](#tuning-key)).

## Expected First PR Scope

The first implementation PR should be intentionally narrow:

- add route policy plumbing;
- add force-GEMM and force-current-auto execution without env var dependence;
- add validation helpers and structured route rejection;
- add the tuning key/result structs behind the default multi-row tuning policy;
- profile `Auto`, `GemmOnly`, explicit `GemvWhereSupported`, and valid mixed
  `Fc1GemvFc2Gemm` / `Fc1GemmFc2Gemv` routes;
- leave multiple GEMV kernel variants and persistent tuning for later.

This first scope directly answers the most important question: for the known
QMoE shape, should the operator run grouped GEMM or the GEMV path?
