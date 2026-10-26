# Contrib Operator FLOP Estimation: Design and Rollout Plan

**Status:** Rollout proposal with an initial internal implementation; later phases remain proposed.

## 1. Summary

Introduce an opt-in diagnostic feature that estimates the mathematical work of a contrib operator invocation.
The minimum operator scope is `com.microsoft::MatMulNBits` and GroupQueryAttention (GQA). Start with the existing
MatMulNBits estimator, bias component, and legacy static-report preview; next connect estimates to runtime workload metadata,
then add GQA baseline estimates. Route-aware adjustments are optional, incremental refinements, not an exact
executed-work prerequisite for GQA. Keep dispatch, placement,
memory admission, and allocation behavior unchanged throughout the diagnostic rollout.

This plan builds on the
[ORT Memory Estimates Copilot discussion](https://github.com/copilot/tasks/2cffab6d-eafa-4f8a-8595-590c601ee41c),
including its follow-up rollout and design proposals. The discussion was recovered from session history; its
recommendations are proposals, not evidence that the suggested APIs already exist.

The key distinction is:

- **FLOPs:** a count of floating-point mathematical operations for a defined workload.
- **Dense-equivalent operations:** comparable dense arithmetic for a quantized operator; not literal floating-point
  instructions executed by that implementation.
- **FLOP/s or equivalent operations/s:** a count divided by a matched measured duration. A graph alone cannot
  establish this rate.

**Target deliverable:** runtime-derived workload estimates for MatMulNBits and GQA in a versioned developer report.
Initialization should configure runtime reporting, not generate the target workload report. The existing static
export is a legacy developer preview to replace in a later batch, not the desired reporting lifecycle.
Throughput and performance-policy changes are separate deliverables.

The initial core-only preview and the core-plus-bias increment remain independently reviewable and shippable.
Neither alone satisfies the minimum MatMulNBits-and-GQA deliverable. Prioritize useful, convention-based workload
estimates over exhaustive kernel auditing. Add implementation-dependent adjustments only when supported by
execution metadata and documented arithmetic models; an unknown route must not invalidate an available baseline
or be treated as zero extra work. Timing and performance-policy changes remain separate deliverables.

### Initial implementation

The internal [MatMulNBits compute estimator](../../onnxruntime/contrib_ops/cpu/quantization/matmul_nbits_compute_estimate.h)
provides a checked shape-based count and a static adapter for resolved in-tree graph nodes. Its result is either a
core dense-equivalent count under convention version 1 or an explicit unavailable reason. Rank-one inputs use
`M = 1`; unknown dimensions remain unavailable even alongside zero extents. The adapter reads canonical graph
shapes, not maximum-shape scenarios, and does not validate packed weights or replace normal operator validation.

The Phase-2 preview added a static JSON report via the existing session configuration API. That preview does not
collect runtime counts or change inference policy. The estimator sources are included by the
existing CPU contrib source glob, independently of CUDA.

Increment 2a adds a separate `bias` component: `MN` logical floating-point additions when the sixth schema input
is present, or zero when it is omitted. Known absence does not require concrete input shapes; present bias with
unknown required geometry is unavailable. The pure helper accepts unknown presence and returns
`missing_semantic_value`; a resolved graph identifies presence from its optional input slot. A zero reduction
dimension `K` makes the core count zero but does not make bias additions zero if the output is nonempty.
Core arithmetic and its convention remain unchanged. Runtime invocation counts and route-specific executed FLOPs
are not implemented by this increment.

### Structured runtime collector: review batches

The target report uses metadata from actual runtime invocations, not initialization-time workload assumptions.
The existing initialization-time export below remains a legacy preview until replacement is implemented.
Metadata transport must be typed and independent of logger severity or text parsing.

**Batch 1: standalone storage primitive (implemented, not connected to inference).**
The internal [RuntimeWorkloadCollector](../../onnxruntime/core/framework/runtime_workload_collector.h) owns a
run's submitted MatMulNBits records. Each record copies graph/provider identity, node index, actual shapes,
matrix geometry/batch count, bias presence, activation element type, quantization settings, and optional
route/effective compute mode. Session/run identifiers are supplied by the caller; accepted records receive a
unique collector-local invocation ID in commit order, not execution order. Missing route fields remain missing.

Limits are explicit constructor arguments, with no enabled-by-default collector. Record count and accounted
payload bytes are bounded. The byte budget includes record size, copied string bytes/terminators, and shape
elements; allocator capacity, bookkeeping, caller buffers, and snapshot copies are not a hard-memory guarantee.
Rejected submissions return a diagnostic Status and update saturating drop counters and collection completeness.
Allocation failures are converted to Status in exception-enabled builds; no-exception builds retain the existing
allocation-failure termination behavior. Thread-safe snapshots copy retained records and status without exposing
mutable collector state. Collection completeness describes submissions only, not run success or FLOP coverage.
Callers must keep the collector alive until all submitters finish.

This batch has no kernel/context/session plumbing, run-ID generation, semantic validation, counts, run-outcome
handling, logger integration, or JSON/file export. No production inference behavior or public API changes.
The full-build-only implementation does not depend on contrib kernels or MLAS. Tests cover ownership, repeated
invocations, missing routes, empty-output metadata, isolation, limits, and concurrent submissions/snapshots.

**Batch 2 (proposed): execution integration.** Pass an optional recorder through execution context, capture metadata
at actual CPU MatMulNBits dispatch/backend call sites, and commit only established successful invocations.
Define run failure/incompleteness behavior and ensure collection errors cannot fail inference. Test disabled
behavior, actual routes, changing shapes, and concurrent/repeated runs. Do not use logs as data transport.

**Batch 3 (proposed): runtime counts and export.** Apply existing checked core/bias formulas to invocation records,
export versioned runtime reports with partial/drop/run-status coverage, and replace initialization-time workload
export. Agree on configuration and export lifecycle before implementation; GQA extends this path afterward.

### Generating a review artifact

The following describes the still-available legacy static preview, not the target runtime report.

Set `session.compute_estimate_report_path` to a UTF-8 JSON file path before creating a session:

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_session_config_entry("session.compute_estimate_report_path", "compute_estimates.json")
session = ort.InferenceSession("model.onnx", sess_options=options, providers=["CPUExecutionProvider"])
```

The equivalent C++ configuration is:

```cpp
Ort::SessionOptions options;
options.AddConfigEntry("session.compute_estimate_report_path", "compute_estimates.json");
```

These examples require an ORT build containing this implementation, not an older installed wheel.
Session initialization generates the report from the final executable graph before removable attributes are
pruned. No `Run()` or profiling call is required. Relative paths use the process working directory; parents must
already exist. Existing files are overwritten, so choose a path distinct from model/optimized-model files and a
different path for each concurrent session. Reports contain node names and shape metadata; review them before sharing.

Unknown/unsupported estimates are normal report entries. Generation/open/write failures log a warning and do not
fail session initialization. Minimal builds and compile-only sessions warn that export is unsupported. Full builds
with contrib ops disabled can export coverage reports but label the MatMulNBits estimator unavailable.
The unset/empty option performs no report traversal or file I/O.

The developer-facing schema is version 2. It replaces the core-only version-1 preview, adding bias and
component-level subtotals and introducing partial node availability. `estimator_support` is
`matmul_nbits_core_and_bias` when enabled. `nodes` contains per-node components and positional input shapes
(unknown dimensions are `null`, omitted optional inputs have state `missing`). `graphs` contains separate summaries
for `main` and each nested graph. Nested identities use `main/<node_index>/<attribute_name>`, recursively;
attribute segments escape `~` as `~0` and `/` as `~1`. The top-level summary covers only direct nodes in `main`.
Counts never aggregate alternative branches or unknown loop iteration counts into a per-run total.

Node availability is `available` when both declared components are available, `partial` when only one is
available, and `unavailable` when neither is available (or the operator is unsupported).
The three node counts in each summary are disjoint: `estimated_node_count` means fully estimated within the
declared component scope, `partially_estimated_node_count` means partially estimated, and `unavailable_node_count`
means no component is available. Reason counts include partial nodes.
Each `component_summaries` entry reports its metric, convention, logical basis, per-invocation phase,
estimated/unavailable component counts, unavailable reasons, checked `known_operation_count`, and completeness.
An overflowing component subtotal omits its count and reports overflow. These subtotals cover declared components
of supported direct nodes, not unsupported nodes or complete operator work.
The existing `known_dense_equivalent_ops` and `subtotal_availability` remain core-only; bias is never added to that
subtotal. Overall declared-scope completeness also requires no partial nodes or component-subtotal overflow.
Core still lists bias among its exclusions; bias has its own `floating_point_ops` metric and `bias_add_v1`
convention. Neither component claims route-specific executed work or whole-operator completeness.

Run the targeted tests after rebuilding:

```text
onnxruntime_provider_test --gtest_filter="ComputeEstimateReportTest.*:MatMulNBitsComputeEstimate*:MatMulNBitsBiasEstimate*"
```

## 2. Motivation and boundaries

The feature should help users and ORT developers:

1. Quantify workload growth across batch sizes, input lengths, and model variants.
2. Identify expensive supported operators while making missing coverage visible.
3. Compare latency and effective throughput for equivalent workloads.
4. Build evidence for later route-selection and CPU/GPU-placement experiments.

It does **not** initially provide hardware instruction counts, predicted latency, total memory traffic,
universal contrib-op coverage, symbolic-expression algebra, training support, or automatic optimization decisions.
Do not interpret a high operation count as proof that an operator is compute-bound.

## 3. Relationship to existing memory estimation

The [constrained-environment roadmap](future_directions_constrained_env.md) and
[tracking issue #29775](https://github.com/microsoft/onnxruntime/issues/29775) provide useful architectural precedents:

| Existing surface | Reusable lesson | Boundary for compute estimation |
| --- | --- | --- |
| [Level1MemoryEstimate](../../include/onnxruntime/core/framework/level1_memory_estimate.h) | Distinguish quantities with different meanings and lifetimes | Keep compute counts in a separate result; do not add them to byte budgets |
| [MatMulNBits memory adapter](../../onnxruntime/contrib_ops/cuda/quantization/matmul_nbits_workspace_estimate.h) | Separate graph parsing from checked dimension arithmetic | Logical counts should not require CUDA device properties or CUDA build flags |
| [WorkspaceInputShape](../../include/onnxruntime/core/framework/workspace_input_shape.h) | Preserve positional inputs, missing optionals, partial shapes, and zero extents | It does not carry shape provenance; a maximum-shape hint is not a concrete invocation |
| [DeclareWorkspaceRequirements](../../include/onnxruntime/core/framework/op_kernel.h) | Use metadata without requiring live tensor allocations | Do not introduce a new kernel virtual method just to implement a reporting pilot |
| [Attention workspace roadmap](attention_workspace_estimation.md) | Validate workload geometry and distinguish runtime routes from AOT scenarios | Workspace bytes do not determine operation count, memory traffic, or latency |
| [Node profiling](../../onnxruntime/core/framework/sequential_executor.cc) | Existing events include node identity, provider, and shape information | An event named `kernel_time` is not automatically a device-execution interval |

The eventual planner would consume separate inputs:

```text
memory requirements + allocation lifetimes -> feasible candidates
measured/calibrated performance costs      -> preference among feasible candidates
```

The count/reporting pilot must not depend on completion of memory-policy or preallocation work.

## 4. Counting contract

### 4.1 Required semantics

- Count a multiply-add as **two operations**, including when implemented as one fused instruction.
- Version the counting convention independently from the report schema.
- Declare the metric and included components for every estimate.
- Interpret "exact" as exact under that convention and component scope, not exact hardware execution.
- Return unavailable with a machine-readable reason for unsupported operators, unknown required metadata,
  invalid metadata, or arithmetic overflow. Never substitute zero for an unavailable estimate.
- Accept a legitimate empty workload as zero only when the required metadata establishes its validity.
- Keep static concrete shapes, runtime invocation shapes, and declared workload scenarios distinguishable.
- Do not call maximum-shape hints proven upper bounds unless the estimator establishes that guarantee.
- Keep floating-point counts and dense-equivalent counts in separate subtotals.

Malformed model inputs remain subject to normal ORT validation. A diagnostic failure must not silently change
inference results or bypass model validation. Unsupported estimation is a reportable condition, not an inference
failure; allocation, serialization, and other operational errors must use standard error reporting.

### 4.2 Pilot formula: MatMulNBits

Use the operator's logical `K` and `N` attributes, not the packed weight storage dimensions. For a supported valid
input `A` with shape `[..., K]`:

```text
M = product of A's leading dimensions
core dense-equivalent operations = 2 * M * N * K
```

The [MatMulNBits schema](../../onnxruntime/core/graph/contrib_ops/contrib_defs.cc) defines packed weights,
dequantization, and an optional bias. The pilot counts **only core matrix multiplication**:

- Exclude dequantization, unpacking, prepacking, bias, conversions, padding, and implementation-specific extra work.
- Report those exclusions even when bias or other optional inputs are present.
- Do not scale the logical count down by the quantization bit width.
- Preserve the same core count across providers and kernel routes for the same logical invocation.
- Validate rank and agreement between the last input dimension and `K`; handle rank-one inputs according to the
  validated schema/kernel contract rather than assuming they are supported.
- Use checked integer multiplication; reject invalid negative dimensions before unsigned conversion.

Example: `A = [2, 3, 64]`, `K = 64`, `N = 32` gives `M = 6` and **24,576 dense-equivalent operations**.
This is not a complete count of all work performed by `MatMulNBits`.

### 4.3 Ancillary components and counting categories

Core versus ancillary describes an operation's role, not whether it is floating-point work. Floating-point bias
addition and dequantization arithmetic can contribute FLOPs; bitwise unpacking does not. Report named components
alongside the stable core metric rather than silently broadening the meaning of `2MNK`.

| Component | Initial counting approach | Required basis and limitations |
| --- | --- | --- |
| Matrix-product core | `2MNK` dense-equivalent operations | Preserve the existing convention; not literal executed floating-point instructions |
| Optional bias | `MN` logical floating-point additions when bias is present | Known output geometry and optional-input presence; zero for absent bias, unavailable when presence or geometry is unknown |
| Logical weight reconstruction | Under an explicit scalar convention, `KN` subtractions and `KN` multiplications for `(q - zero_point) * scale` | One reconstruction of logical weights only; not a claim about per-invocation execution |
| Executed dequantization | Count arithmetic over the elements actually processed, including repetition | Validated route, dtype, padding, reuse, and preparation frequency; otherwise unavailable |
| Unpacking | Route-specific bitwise/integer operations where justified | Keep separate from FLOPs; do not assume a universal instruction count per quantized weight |
| Conversions | Number of elements converted, with source and destination types | Separate category, not an arithmetic FLOP count |
| Layout transforms | Logical elements or bytes moved | Not measured DRAM traffic; caches and reuse are outside this count |
| Padding and recomputation | Additional route-specific arithmetic, classified by arithmetic type | Count only work beyond the corresponding baseline; avoid overlap with other components |
| Prepacking | Separately scoped initialization or first-use work | Never charge automatically to every steady-state invocation |

Logical reconstruction is an optional workload scenario, not an executed-work subtotal. A kernel may reconstruct
weights once and cache them, repeat reconstruction across tiles, or fold the arithmetic into another computation.
An integer subtraction or an optimized-away operation must not be reported as an executed floating-point operation.
Explicitly document the treatment of symmetric zero points, fused operations, and eliminated arithmetic.

For every added component, retain:

- metric/unit, included work, and counting-convention version;
- basis: logical semantics, declared scenario, or validated selected route;
- phase: initialization, first use, or per invocation;
- processed geometry and, where needed, repetition/reuse assumptions;
- available count or an explicit unavailable reason.

Keep dense-equivalent operations, floating-point arithmetic, integer/bitwise operations, conversions, and logical
bytes moved in separate categories. Only combine non-overlapping components with compatible units, conventions,
bases, and phases. In particular, do not present core dense-equivalent operations plus ancillary FLOPs as literal
total executed FLOPs. Known-component subtotals must retain exclusions and incomplete-coverage indicators.

Adding components improves coverage; validated route attribution improves fidelity to execution. Neither makes
the quantized core a hardware instruction count or predicts latency without measurement.

### 4.4 Baseline estimates and optional route-aware refinements

Provide two distinct results rather than progressively relabeling the existing core count:

- **Logical FLOP-equivalent workload:** the dense-equivalent core plus explicitly declared, non-overlapping
  logical floating-point components under a common workload convention. For the example above, core plus a
  present bias is `24576 + 192 = 24768` FLOP-equivalent operations. This scope excludes reconstruction unless
  a separate reconstruction scenario is declared; it is not a complete executed-work total.
- **Estimated executed FLOPs:** floating-point arithmetic attributed to a validated implementation route for a
  specified workload and lifecycle phase. Integer/bitwise work, conversions, and bytes moved remain separate.
  An FMA counts as two FLOPs; this is an arithmetic estimate, not a hardware instruction count.

The primary deliverable is the logical baseline for MatMulNBits and GQA, with explicit component scope and
assumptions. It does not require instruction-exact counts. A route-adjusted arithmetic estimate is a separate,
optional result: record its model/convention, evidence, processed geometry, and limitations. Call it estimated
executed FLOPs only when the model accounts for the floating-point arithmetic attributed to that route and phase.
Do not sum a baseline component with its adjusted replacement.

Route-family refinements in Increment 3a use focused implementation verification rather than requiring an
exhaustive audit of every kernel before baseline delivery:

1. Select a bounded route, initially a CPU path exercised by the pilot tests, and confirm the actual kernel
   selected. Record relevant algorithm, dtypes, quantization settings, and preparation state. Reuse actual route
   decisions rather than introducing a parallel dispatch predictor.
2. Verify count-affecting behavior, including library calls where necessary. Establish floating-point versus integer core arithmetic,
   retained/eliminated reconstruction, bias/epilogues, padded processing, repetitions, and additional reductions.
   Do not treat `2MNK` as executed FLOPs when the selected route performs integer dot products.
3. Establish initialization, first-use, and per-invocation lifetimes, including cached/shared preparation.
   Distinguish cold and warm invocations; do not charge shared or one-time work repeatedly.
4. Implement checked formulas from validated processed geometry and execution frequency. Count padding and
   repetition within the affected component, without adding them again as overlapping extra work.
5. Validate the arithmetic model with independent reference cases and route-boundary, padding, reuse, and lifecycle
   tests. Share formulas across variants only after confirming their assumptions. Use backend documentation and
   deeper inspection where necessary; hardware counters are optional supporting evidence, not a baseline gate.
   An opaque library call is not sufficient evidence for its internal arithmetic count.
6. Report logical and executed metrics side by side with route identity, phase, convention, precision, assumptions,
   and coverage. Runtime shapes alone do not establish an executed-FLOP count.

For one invocation, the executed estimate sums only validated floating-point core, reconstruction, epilogue,
and other arithmetic performed for that invocation. If any applicable component is unknown, expose a
known-component subtotal with explicit gaps, not a complete executed-FLOP total. Enable the executed metric only
for validated routes; adding ancillary logical estimates alone does not qualify a route as supported. If route
metadata or an adjustment model is unavailable, retain the logical baseline and mark the adjustment unavailable.
Fusion alone does not imply fewer FLOPs; it may reduce memory traffic without eliminating arithmetic.

## 5. Proposed architecture and report

```text
optimized executable graph OR runtime invocation
                |
                v
operator metadata adapter
                |
                v
validated, graph-free workload description
                |
                v
pure checked estimator
                |
                v
per-node result -> coverage-aware report -> optional timing correlation
```

### Implementation shape

1. Start with an internal lookup keyed by domain, operator name, and supported schema-version range.
2. Keep formulas independent of `Node`, `GraphViewer`, CUDA, allocation, and execution.
3. Have static and runtime adapters call the same formula.
4. Reuse validated shape/attribute helpers where their contracts fit; do not copy a CUDA-only helper into a second
   implementation or make core reporting depend on it. Extract only genuinely shared arithmetic when necessary.
5. Preserve input positions and distinguish an omitted optional input from a present input without shape metadata.
6. Cache only with all count-affecting dimensions, attributes, and semantic values in the key.
7. Defer public C/C++/Python APIs and plugin registration until the internal contract stabilizes.

The result type should make contradictory states impossible: an available estimate has a count, and an unavailable
estimate has a reason rather than a numeric placeholder. The following is an abbreviated version-2 report
for a node without bias;
the implementation also emits positional input metadata, estimator support, and per-graph summaries:

```json
{
  "schema_version": 2,
  "convention_version": "core_arithmetic_v1",
  "graph_scope": "optimized_executable",
  "nodes": [
    {
      "graph_path": "main",
      "node_index": 12,
      "node_name": "projection",
      "domain": "com.microsoft",
      "op_type": "MatMulNBits",
      "opset_version": 1,
      "provider": "CUDAExecutionProvider",
      "availability": "available",
      "whole_operator_work_complete": false,
      "components": [
        {
          "component": "matmul_core",
          "estimate_kind": "exact_under_convention",
          "input_basis": "static_concrete",
          "basis": "logical_semantics",
          "execution_phase": "per_invocation",
          "convention_version": "core_arithmetic_v1",
          "availability": "available",
          "metric": "dense_equivalent_ops",
          "scope": "matmul_core",
          "operation_count": "24576",
          "excluded_components": [
            "dequantization", "bias", "packing", "conversion", "padding", "route_specific_extra_work"
          ]
        },
        {
          "component": "bias",
          "scope": "bias",
          "metric": "floating_point_ops",
          "convention_version": "bias_add_v1",
          "estimate_kind": "exact_under_convention",
          "input_basis": "optional_input_absent",
          "basis": "logical_semantics",
          "execution_phase": "per_invocation",
          "availability": "available",
          "operation_count": "0",
          "excluded_components": [
            "matmul_core", "dequantization", "packing", "conversion", "padding", "route_specific_extra_work"
          ]
        }
      ]
    }
  ],
  "summary": {
    "reported_node_count": 1,
    "estimated_node_count": 1,
    "partially_estimated_node_count": 0,
    "unavailable_node_count": 0,
    "known_dense_equivalent_ops": "24576",
    "subtotal_availability": "available",
    "complete_for_declared_graph_and_component_scope": true,
    "whole_operator_work_complete": false,
    "count_scope": "direct_nodes_only",
    "per_run_total_available": false,
    "graph_path": "main"
  }
}
```

Serialize counts as decimal strings to preserve integer precision in JSON consumers. An unavailable component omits
`operation_count` and includes a reason such as `unsupported_version`, `unknown_shape`, `invalid_metadata`, `overflow`,
or `estimators_disabled`. Unsupported operators and opaque partitions have unavailable node entries with empty
components. Control-flow nodes use `runtime_execution_required`; their nested graphs are reported separately.
`missing_semantic_value` indicates unknown semantic information such as bias presence in the pure helper.
The static resolved-graph adapter can determine bias presence without runtime data.
If a graph subtotal overflows, omit it and report `subtotal_unavailable_reason: overflow`; never wrap or silently
drop a node's count. A zero known subtotal with unavailable nodes is not a zero-work model.

Report the optimized graph actually represented by ORT, with graph/subgraph identity and final node identity.
Estimate fused nodes directly or mark them unsupported; never count both a fusion and its removed constituents.
An EP-compiled opaque partition must be unavailable unless a valid attribution mechanism exists.

Known-operation subtotals are not complete model totals. Node coverage is not compute-weighted coverage.
Do not mix declared-scenario estimates with concrete counts in the same total. Branch and loop bodies cannot be
summed into an exact per-run count without execution information.

## 6. Phased rollout

Each phase should be delivered through reviewable PRs. The named owners below are roles to assign, not commitments
from particular contributors. Calendar estimates should follow contract review and a pilot implementation.

| Phase | Deliverables | Dependencies | Proposed owner | Exit gate |
| --- | --- | --- | --- | --- |
| 0. Contract | Metric conventions, exclusions, result states, report schema, supported build scope | None | Framework maintainer + operator owner | Resolve Q1-Q3 below; approve reference cases and unknown/error semantics |
| 1. Pure pilot | Internal result type and lookup; graph-free MatMulNBits estimator; unit tests | Phase 0 | Framework + quantization owners | Independent expected counts pass; no CUDA dependency; overflow and invalid metadata diagnosed |
| 2. Static reporting / MVP | Opt-in developer report over the optimized executable graph; coverage and provenance | Phase 1 | Framework + tooling owners | End-to-end report matches golden expectations; fusion/partial coverage are honest; inference unchanged |
| 3. Runtime workload metadata and counts | MatMulNBits invocation estimates, actual route/compute-type metadata where exposed, event identity and bounded caching; accommodate GQA metadata needs | Phase 2 and Increment 2a | Execution/profiling + quantization owners | Changing shapes and actual selected-route attribution pass; no extra device synchronization; agreed overhead gate passes |
| 4. Required GQA baseline / later Attention expansion | GQA core estimate, explicit masking convention, prefill/decode/chunked and active-length semantics; other Attention ops later | Phase 2 and approved GQA convention; Phase 3 integration for runtime-only metadata | Attention owners | Independent GQA references, missing-metadata handling, and static/runtime report integration pass for declared support scope |
| 5. Measured throughput | Matched CPU and eager-CUDA timing; units, scope, route provenance where available | Phase 3; Phase 4 for GQA throughput | Profiling + EP owners | Attribution, aggregation, invalid durations, and timing validation pass |
| 6. Cost-model experiments | Calibrated latency/transfer model; separate route/placement experiment | Phase 5 + applicable memory feasibility contracts | Performance + EP owners | Representative end-to-end benefit and fallback behavior proven before any policy adoption |

Phases 0-2 remain an independently shippable initial MVP, extended by the implemented bias increment.
The current priority is a MatMulNBits runtime slice followed by required GQA baseline coverage, not exhaustive
executed-work accounting. GQA formula/convention work can proceed without waiting for all runtime infrastructure.
Timing does not block either baseline; MatMulNBits timing need not wait for broader Attention coverage.

The ancillary increments retain their numbering; route-specific work does not block the minimum operator scope:

| Increment | Deliverables | Dependencies | Proposed owner | Exit gate |
| --- | --- | --- | --- | --- |
| 2a. Shape-based ancillary work | Separate optional-bias count; component-level coverage and exclusions | Phase 2 and approved component contract | Framework + quantization owners | Present/absent bias and dynamic/empty geometry tests pass; core counts remain unchanged |
| Optional logical reconstruction | Separately labeled one-reconstruction scenario, if consumers need it | Phase 2 and approved scalar reconstruction convention | Quantization + tooling owners | Hand-calculated cases pass; never labeled per-invocation executed work |
| 3a. Route-specific ancillary work | Incremental dequantization, unpacking, conversion, layout, padding, and prepacking estimates | Phase 3 plus validated route and preparation-frequency metadata for each supported path | Quantization + EP owners | Route, reuse, padding, repetition, phase, and non-overlap tests pass; unknown paths remain unavailable |

Increment 2a is implemented for review. Deliver Phase 3 first as a bounded MatMulNBits runtime slice, then
integrate GQA baseline estimates. Logical reconstruction and Increment 3a remain optional follow-ups.
Do not require complete ancillary coverage or exact executed counts before adding GQA.

### Current implementation order and minimum-scope acceptance

Here, "MatMul counting" means the existing `com.microsoft::MatMulNBits` target, not new coverage of standard
`MatMul`, `Gemm`, or every quantized matrix operator.

1. Review and commit the existing MatMulNBits core-plus-bias feature, tests, and report documentation.
2. Implement the next coding task: capture actual MatMulNBits invocation dimensions, bias presence, selected
   top-level route, and compute type where available; reuse the existing pure estimators and expose runtime results.
   Design metadata/reporting with GQA needs in mind without speculative universal kernel abstractions.
3. Add GQA baseline formulas and adapters using query heads, head dimensions, query lengths, active KV lengths,
   and an approved causal/windowed/ragged or rectangular convention. Support prefill, decode, and chunked workloads
   within a declared scope; report missing host-visible semantic values explicitly.
4. Validate both required operators with independent references, changing workloads, report artifacts, compatibility,
   and count-only overhead measurements. No route-specific refinement is required to pass baseline acceptance.
5. Add route-family adjustments incrementally using actual selected-route metadata and justified models of
   reconstruction, padding, repetitions, or reductions. Keep preparation lifetime and arithmetic categories explicit.

The minimum-scope acceptance gate requires:

- MatMulNBits core and bias and GQA core estimates match independent references under documented conventions.
- Static concrete and runtime invocation results retain distinct provenance; repeated/dynamic/concurrent cases
  have correct identity and no stale count reuse. Control-flow totals include only established executions.
- GQA uses query-head count and active KV lengths, not KV-head count or cache capacity as substitutes; the masking
  convention, ancillary exclusions, and unsupported cases are visible.
- Available baseline estimates survive missing route metadata. Unsupported adjustments have explicit reasons;
  neither unknown extra work nor missing baseline inputs are silently treated as zero.
- Reports separate logical baselines, adjusted replacements, categories, and phases, with no double counting or
  unsubstantiated executed-FLOP/whole-operator claims.
- Supported build/inference compatibility and the approved overhead gate pass; review artifacts cover both
  available and unavailable cases.

This gate covers useful estimates for both required operators, not exact instruction counts or universal EP
coverage. Route-specific ancillary work, other Attention operators, and timing are independently reviewable
follow-ups, not prerequisites for the required GQA baseline.

### Phase 2: static-report launch

- Use an explicit-path session config entry for the developer preview, with warning-only report failures.
- Restrict initial support to full inference builds. Builds excluding contrib ops must retain unchanged behavior
  and must not acquire a dependency on unavailable contrib implementations.
- Validate real optimized models, including cases where fusion removes all directly supported pilot nodes.
- Include feature availability, convention/schema versions, exclusions, and unavailable reasons in documentation.
- Publish reports without changing provider assignment, kernel selection, allocator behavior, or inference outputs.
- Establish component naming, units, provenance, execution phase, and completeness rules for later ancillary work.
  Keep this initial contract small; the core-only report does not require speculative route-estimation machinery.

### Increment 2a: shape-based ancillary work

Initial bias implementation is available for review in the version-2 report described above.

- Add optional bias as a separate `MN` logical floating-point-addition count, leaving `matmul_core` unchanged.
- Distinguish absent bias from unknown optional-input metadata; use checked arithmetic and existing unavailable states.
- Report component-level exclusions and coverage, not just whether a node has any estimate.
- If a logical reconstruction scenario is requested, report its subtraction and multiplication counts separately
  under an approved convention. Do not add `2KN` to every invocation or imply reconstruction actually ran.

### Phase 3: runtime refinement

- Start with a MatMulNBits runtime slice. Read dimensions and bias presence from validated inputs, route/compute
  type from actual implementation decisions, and preparation state where exposed. Reuse existing metadata;
  never infer actual execution from a maximum-shape hint or a provider name alone.
- For GQA, collect validated head geometry, query/active KV lengths, masking/window settings, and per-sequence
  lengths where host-visible. Missing device-only values remain unavailable; do not copy them solely for estimates.
- Add an opt-in runtime reporting hook: initialization-time static export cannot describe per-invocation routes.
  Baseline counting must remain independent of whether route-specific metadata is available.
- Recompute counts for actual invocation geometry, rather than presenting static hints as observed work.
- Distinguish concurrent runs, repeated invocations, subgraphs, and node identity within one session.
- Aggregate only executed control-flow paths once runtime support exists; otherwise retain explicit incompleteness.
- Do not copy GPU scalar values to the host or synchronize solely to compute a count.
- Use already validated host-visible metadata, or report `missing_semantic_value`.
- Bound cache growth and test changing shapes and semantic values, not just repeated fixed-shape runs.

### Increment 3a: route-specific ancillary work

Optional refinement, not a MatMulNBits/GQA baseline gate. For one arithmetic family at a time:

1. Reuse the kernel's validated geometry and actual route decision; do not duplicate its dispatch cascade.
2. Establish whether preparation runs during initialization, first use, or each invocation.
3. Determine processed elements, including padding, repeated tiles, split-K work, and any cached or shared preparation.
4. Classify floating-point arithmetic, integer/bitwise work, conversions, and logical data movement separately.
5. Add checked pure formulas fed by that metadata, with independent reference tests.
6. Report unavailable when route, reuse, or frequency is unknown; do not synchronize or copy device data solely for
   estimation. Include all count-affecting route and preparation state in cache keys or invalidate cached estimates.

Fusion does not imply that a component is absent or separately materialized. Verify which arithmetic is retained,
eliminated, or repeated before attributing it. Treat prepacking as its own lifetime-scoped component, and distinguish
first-use preparation from steady-state work. Do not attribute shared preparation in full to multiple consumers.

### Phase 4: Attention expansion

GQA is required baseline coverage, not an expansion blocked on MatMulNBits route audits. Implement it after the
first MatMulNBits runtime slice; convention and pure-formula work can begin earlier. Other Attention operators
follow separately. Core formulas must have exact tests under their declared workload convention, even though
their relationship to kernel execution is approximate.

First agree on useful unmasked work versus rectangular dense-equivalent work. For useful core work, define `P` as
the sum of valid query-key pairs across batches and query heads:

```text
QK-transpose + PV core operations = 2 * P * (D_qk + D_v)
```

For rectangular Attention with equal head dimensions `D`, this reduces to:

```text
4 * batch * query_heads * query_length * active_KV_length * D
```

Use query heads for grouped-query Attention, not KV-head count. Separate active KV length from allocated cache
capacity. Count valid pairs explicitly for causal, sliding-window, and ragged sequences, including any past-length
offset. Packed token count alone does not determine the sum of per-sequence query-key pairs.

State whether projections, softmax, bias, rotary transforms, and other auxiliary work are included. PackedAttention
and PackedMultiHeadAttention do not have identical projection work. Backend tiling, recomputation, and padding
belong to a separately named route-specific metric, not an undocumented change to the logical count.

### Phase 5: throughput

Timing is separate from the minimum count deliverable and does not require exhaustive route accounting.
Count-only overhead measurements remain part of runtime-count validation.

Only calculate a rate when the count and duration cover the same invocation and components:

```text
effective operations/s = operation_count / duration_seconds
effective TFLOP/s       = logical_flops / duration_seconds / 10^12
```

Label quantized results as dense-equivalent operations/s, not literal hardware TFLOP/s.

- Identify timing scope: CPU invocation, host submission, correlated device execution, or end-to-end run.
- A multi-kernel op needs a defined correlated interval; do not divide its full count by one child kernel's duration.
- Separate warmup, initialization/prepack, compilation/tactic search, and steady-state execution.
- Omit rates for missing, zero, negative, or uncorrelated durations and report why.
- Aggregate repeated matched invocations using `sum(counts) / sum(durations)`, not the mean of their rates.
- Do not sum overlapping device intervals and call the result end-to-end latency.
- Keep CUDA Graph replay and unsupported EP timing explicitly unavailable until attribution is verified.
- Record device, dtype, workload, provider, and route when observable; unknown route identity must remain unknown.
- Name the numerator's component scope. Core-only throughput can ship without ancillary coverage; adding a component
  must not silently redefine an existing rate or mix initialization work into a steady-state duration.

## 7. Validation and acceptance

Use independent expected values rather than comparing two implementations of the same formula.

| Area | Required cases | Acceptance |
| --- | --- | --- |
| Pilot arithmetic | Small hand-calculated cases, batched leading dimensions, dtype/bit-width variants | Exact equality under the convention; example above equals 24,576 |
| Metadata | Unknown rank/dimensions, mismatched K, missing/invalid attributes, omitted optional inputs | Explicit correct reason; no zero fallback or unsigned wrap |
| Boundaries | Valid zero extent, large products, uint64 overflow, zero with other unknown dimensions | Documented validity rules and checked arithmetic; unavailable when validity is unproven |
| Semantics | Bias present/absent, prepacked weights, CPU/CUDA routes | Same core count; exclusions preserved |
| Logical ancillary work | Bias present/absent, empty/dynamic outputs, overflow, hand-calculated reconstruction cases | Exact component counts under the stated convention; logical reconstruction never claimed as observed execution |
| Route-specific ancillary work | Cached versus per-invocation preparation, dtype, padding, repeated tiles, split-K, fusion, shared prepacking, route changes | Independently verified counts and phase attribution; no overlap, stale reuse, or extra synchronization; unknown routes unavailable |
| Static graph | Unsupported ops, fusion, opaque partitions, subgraphs, shape scenarios | No double counting; exact subtotals and coverage state |
| Runtime | Changing batch/sequence sizes, repeated/concurrent runs, count-affecting scalar changes | Invocation-specific counts; no stale cache reuse or extra synchronization |
| Attention | Hand-enumerated pairs for causal/past/windowed/ragged cases; unequal QK/V dimensions | Exact pair-based counts for supported cases; explicit missing-metadata result otherwise |
| Timing | Multiple kernels, asynchronous GPU work, invalid durations, unequal repeated workloads | Correct attribution, units, and weighted aggregation |
| Compatibility | Feature disabled, contrib-disabled build, unsupported build/report request | Existing inference behavior preserved; feature unavailability surfaced |

Proposed overhead gates, **subject to team approval before Phase 3**:

- Disabled: no estimator invocation, report allocation, device event, or new synchronization.
- Static-only: no per-run work; measure report/initialization cost separately.
- Runtime-count-only: at most 1% median steady-state latency overhead on an agreed CPU/CUDA matrix, with uncertainty
  reported. Measure tiny launch-bound workloads separately rather than hiding them in an average.
- Timing-enabled: publish overhead separately; it is not covered by the runtime-count-only budget.

Use existing framework/operator test targets and profiling test infrastructure. Run CPU tests first; CUDA tests are
required when CUDA runtime correlation is added. Maintain a golden report/schema test and an independent benchmark
comparing disabled, count-only, and timing-enabled modes. Record exact commands and results in implementation PRs.

## 8. Enablement, observability, and rollback

1. **Internal validation:** maintainers exercise the pilot and golden reports on small deterministic graphs.
2. **Developer preview:** expose opt-in static reporting with documented coverage; gather model/report feedback.
3. **Runtime preview:** opt in to MatMulNBits runtime counts and exposed route metadata; record overhead and gaps.
4. **Required operator coverage:** add GQA baseline estimates and validate both operators against minimum-scope
   acceptance. Do not block on executed-work completeness.
5. **Optional refinements and broader support:** add route adjustments, matched timing, and other operators with
   owner-reviewed conventions and reference cases. Stabilize a public
   API only after consumers validate the report and unknown-state semantics.

Track unavailable-reason frequencies, supported-node coverage, report generation cost, and count/timing overhead
in local reports and benchmark results. This proposal does not add automatic telemetry.

Rollback is to disable reporting/instrumentation, not to change inference policy. Keep count-only and timing modes
independently switchable. Any later placement/dispatch experiment needs its own flag, fallback, quality gates,
and rollback plan.

## 9. Open questions for the team

Phase-2 preview decisions: use `session.compute_estimate_report_path`, versioned component-oriented JSON, separate
nested graph scopes, and warning-only export failures. These choices are implemented for developer review; ownership,
additional build coverage, and later-phase semantics still require team agreement.

| ID | Decision to resolve | Recommendation | Needed by |
| --- | --- | --- | --- |
| Q1 | Core arithmetic only or complete operator work? | Core arithmetic first; explicit exclusions and separate auxiliary components later | Phase 0 |
| Q2 | Is quantized dense-equivalent work an acceptable first pilot? | Yes; never label it literal executed FLOPs. If literal floating-point work is required first, choose a floating-point pilot before implementation | Phase 0 |
| Q3 | Who owns conventions, schema, and the first reporting surface? | Framework owns the contract; operator owners approve formulas; internal developer report before public APIs | Phase 0 |
| Q4 | Which full/minimal/training/plugin builds participate? | Full inference first; no public/plugin ABI changes in the MVP | Phase 2 |
| Q5 | How should scenarios, valid zeros, and incomplete control-flow totals appear? | Distinct provenance and unavailable reasons; no claimed per-run total without execution evidence | Phase 2 |
| Q6 | Is the proposed count-only overhead budget acceptable, and on which workloads? | Approve a representative CPU/CUDA matrix and a tiny-op cohort before instrumentation | Phase 3 |
| Q7 | Useful unmasked or rectangular dense-equivalent Attention work? | One explicitly named primary convention; optional secondary metric only if needed | Phase 4 |
| Q8 | Which timing paths reliably correlate full op execution, including capture? | CPU and eager CUDA first; defer capture and other EPs until validated | Phase 5 |
| Q9 | Should external EPs/custom ops register estimators? | Revisit after multiple built-in operators demonstrate the needed contract | Public API design |
| Q10 | When may estimates affect placement or dispatch? | Only after calibrated end-to-end experiments, including bandwidth and transfer costs | Phase 6 |
| Q11 | Which ancillary components and reconstruction conventions should be supported first? | Bias after static reporting; logical reconstruction only if needed; executed-work estimates require validated route and frequency metadata | Phase 2 design; per-route approval before Increment 3a |
| Q12 | What is the minimum supported workload/build scope for MatMulNBits and GQA? | Agree on baseline components, prefill/decode/chunked and masking cases, runtime metadata availability, and explicit gaps; route refinements are optional | Phase 3 design and Phase 4 baseline acceptance |

## 10. Completion criteria

The **MVP is complete** when a developer can request a persistent, versioned static report for an optimized graph,
see the correct `MatMulNBits` core count for supported concrete shapes, and understand every exclusion and missing
estimate without any change to inference behavior.

The **minimum operator scope is complete** when MatMulNBits core-plus-bias and GQA core estimates satisfy the
Section 6 acceptance gate for the declared workload/build matrix, with documented conventions, runtime support,
and explicit gaps. Exact route-specific executed-FLOP counts and full ancillary coverage are not required.

The **diagnostic feature is mature** when supported runtime invocations and matched throughput can be reported with
validated attribution, honest coverage, and measured overhead. Automatic performance planning remains a separate
feature, not a prerequisite for delivering useful counts.
