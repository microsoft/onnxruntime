# Contrib Operator FLOP Estimation: Design and Rollout Plan

**Status:** Rollout proposal with an initial internal implementation; later phases remain proposed.

## 1. Summary

Introduce an opt-in diagnostic feature that estimates the mathematical work of a contrib operator invocation.
Start with a pure `com.microsoft::MatMulNBits` estimator and a coverage-aware static report. Then complete and
validate MatMulNBits counting, including bias, concrete runtime shapes, and supported route-specific ancillary
work, before expanding to Attention/GQA, measured throughput, or other operators. Keep dispatch, placement,
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

**Recommended first release:** counting conventions, `MatMulNBits` core dense-equivalent operations, and a versioned
developer report. Throughput and performance-policy changes are separate deliverables.

The core-only developer preview remains independently reviewable and shippable, but it is not the MatMulNBits
counting-completion milestone. Prioritize its ancillary and runtime-count follow-ups before starting broader
operator coverage or throughput work. Add implementation-dependent counts only when their route and frequency
can be established; otherwise retain explicit unavailable results rather than inventing a total.

### Initial implementation

The internal [MatMulNBits compute estimator](../../onnxruntime/contrib_ops/cpu/quantization/matmul_nbits_compute_estimate.h)
provides a checked shape-based count and a static adapter for resolved in-tree graph nodes. Its result is either a
core dense-equivalent count under convention version 1 or an explicit unavailable reason. Rank-one inputs use
`M = 1`; unknown dimensions remain unavailable even alongside zero extents. The adapter reads canonical graph
shapes, not maximum-shape scenarios, and does not validate packed weights or replace normal operator validation.

The Phase-2 preview adds a static JSON report via the existing session configuration API. There is no runtime
instrumentation, new public API function, or inference-policy change. The estimator sources are included by the
existing CPU contrib source glob, independently of CUDA.

### Generating a review artifact

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

The developer-facing schema is version 1. `nodes` contains per-node components and positional input shapes
(unknown dimensions are `null`, omitted optional inputs have state `missing`). `graphs` contains separate summaries
for `main` and each nested graph. Nested identities use `main/<node_index>/<attribute_name>`, recursively;
attribute segments escape `~` as `~0` and `/` as `~1`. The top-level summary covers only direct nodes in `main`.
Counts never aggregate alternative branches or unknown loop iteration counts into a per-run total.

Run the targeted tests after rebuilding:

```text
onnxruntime_provider_test --gtest_filter="ComputeEstimateReportTest.*:MatMulNBitsComputeEstimate*"
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
estimate has a reason rather than a numeric placeholder. The following is an abbreviated version-1 report;
the implementation also emits positional input metadata, estimator support, and per-graph summaries:

```json
{
  "schema_version": 1,
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
        }
      ]
    }
  ],
  "summary": {
    "reported_node_count": 1,
    "estimated_node_count": 1,
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
`missing_semantic_value` is reserved for future runtime-dependent components.
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
| 3. Runtime counts | Opt-in invocation estimates using concrete shapes; event identity and bounded caching | Phase 2 | Execution/profiling owners | Dynamic-shape cases pass; no extra device synchronization; agreed overhead gate passes |
| 4. Attention coverage | Incremental packed Attention and GQA support; phase and active-length semantics | MatMulNBits counting-completion gate below; Phase 3 for runtime-only metadata | Attention owners | Prefill/decode/chunked, causal/windowed, and ragged reference cases pass |
| 5. Measured throughput | Matched CPU and eager-CUDA timing; units, scope, route provenance where available | MatMulNBits counting-completion gate and Phase 3; Phase 4 only for Attention throughput | Profiling + EP owners | Attribution, aggregation, invalid durations, and timing validation pass |
| 6. Cost-model experiments | Calibrated latency/transfer model; separate route/placement experiment | Phase 5 + applicable memory feasibility contracts | Performance + EP owners | Representative end-to-end benefit and fallback behavior proven before any policy adoption |

Phases 0-2 remain the useful, independently shippable core-only MVP. The current priority after Phase 2 is
MatMulNBits counting completion, not Attention or timing. After that gate passes, Phases 4 and 5 can proceed
independently; MatMulNBits throughput need not wait for complete Attention coverage.

The ancillary increments below retain their numbering but now participate in the MatMulNBits-first milestone:

| Increment | Deliverables | Dependencies | Proposed owner | Exit gate |
| --- | --- | --- | --- | --- |
| 2a. Shape-based ancillary work | Separate optional-bias count; component-level coverage and exclusions | Phase 2 and approved component contract | Framework + quantization owners | Present/absent bias and dynamic/empty geometry tests pass; core counts remain unchanged |
| Optional logical reconstruction | Separately labeled one-reconstruction scenario, if consumers need it | Phase 2 and approved scalar reconstruction convention | Quantization + tooling owners | Hand-calculated cases pass; never labeled per-invocation executed work |
| 3a. Route-specific ancillary work | Incremental dequantization, unpacking, conversion, layout, padding, and prepacking estimates | Phase 3 plus validated route and preparation-frequency metadata for each supported path | Quantization + EP owners | Route, reuse, padding, repetition, phase, and non-overlap tests pass; unknown paths remain unavailable |

Agree on component semantics during Phase 2 design, but do not implement every component before shipping the MVP.
Deliver Increment 2a next, then Phase 3 for MatMulNBits and Increment 3a for the agreed supported routes.
Logical reconstruction remains optional and cannot substitute for evidence of executed ancillary work.
Do not start Attention/GQA, other operator estimators, throughput, or cost-model implementation until the
MatMulNBits counting-completion gate passes. Each route-specific component is enabled only with sufficient evidence.

### MatMulNBits-first order and completion gate

Here, "MatMul counting" means the existing `com.microsoft::MatMulNBits` target, not new coverage of standard
`MatMul`, `Gemm`, or every quantized matrix operator.

1. Finish Phase-2 report review and correctness validation, preserving the independent core-only preview.
2. Implement Increment 2a: separate bias counts and component-level coverage, including absent, unknown, empty,
   and overflow cases. Keep the `2MNK` core convention unchanged.
3. Implement Phase 3 for MatMulNBits: concrete invocation geometry, execution identity, repeated/dynamic workloads,
   and checked aggregation without stale cache reuse or extra device synchronization.
4. Implement Increment 3a for a bounded, owner-approved CPU/CUDA route matrix. Audit dequantization, unpacking,
   conversion, layout, padding/recomputation, and prepacking; identify retained/eliminated work, reuse, repetition,
   and initialization/first-use/per-invocation lifetimes. Record unvalidated routes and components as unavailable.
5. Review independent reference counts, end-to-end reports, inference compatibility, build coverage, and measured
   count-only overhead against the agreed workload matrix before opening the next operator/timing workstream.

The gate passes only when:

- Core and bias counts match independent references for the declared static and runtime workload matrix.
- Every applicable ancillary component on each claimed supported route has a validated count and attribution;
  components proven absent or eliminated are distinguished from components whose execution is unknown.
- Unsupported routes/components have explicit reasons and documented limitations. Reviewers approve the bounded
  coverage and any deferred gaps; marking all ancillary work unavailable is not completion.
- Reports keep arithmetic categories and lifetimes separate, avoid overlap/shared-preparation double counting,
  and do not claim literal total executed FLOPs or whole-operator completeness without supporting evidence.
- Dynamic/repeated/concurrent invocation tests, route/reuse changes, overflow, disabled behavior, and the
  approved overhead gate pass. Published review artifacts show both covered and unavailable cases.

This is completion of counting within a declared, validated support scope, not a promise to count every hardware
instruction or support every EP. Broader route coverage can follow, but silently treating unknown work as zero
or relaxing the gate to begin Attention is not the planned rollout.

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

- Add optional bias as a separate `MN` logical floating-point-addition count, leaving `matmul_core` unchanged.
- Distinguish absent bias from unknown optional-input metadata; use checked arithmetic and existing unavailable states.
- Report component-level exclusions and coverage, not just whether a node has any estimate.
- If a logical reconstruction scenario is requested, report its subtraction and multiplication counts separately
  under an approved convention. Do not add `2KN` to every invocation or imply reconstruction actually ran.

### Phase 3: runtime refinement

- Recompute counts for actual invocation geometry, rather than presenting static hints as observed work.
- Distinguish concurrent runs, repeated invocations, subgraphs, and node identity within one session.
- Aggregate only executed control-flow paths once runtime support exists; otherwise retain explicit incompleteness.
- Do not copy GPU scalar values to the host or synchronize solely to compute a count.
- Use already validated host-visible metadata, or report `missing_semantic_value`.
- Bound cache growth and test changing shapes and semantic values, not just repeated fixed-shape runs.

### Increment 3a: route-specific ancillary work

For one supported route at a time:

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

Begin implementation only after the MatMulNBits counting-completion gate passes.

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

Begin timing integration only after the MatMulNBits counting-completion gate passes. Count-only overhead
measurements needed to validate Phase 3 are not deferred by this restriction.

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
3. **Runtime preview:** separately opt in to counts, then timing; record overhead and attribution gaps.
4. **MatMulNBits counting completion:** validate bias, runtime counts, and supported ancillary routes against the
   gate above before starting Attention/GQA, other operators, or throughput implementation.
5. **Broader support:** add operators only with owner-reviewed conventions and reference cases. Stabilize a public
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
| Q12 | What is the bounded MatMulNBits counting-completion scope? | Agree on CPU/CUDA routes, workload matrix, applicable components, and explicit deferred gaps; require owner approval before broader work starts | Before Increment 3a; gate review before Phases 4-5 |

## 10. Completion criteria

The **MVP is complete** when a developer can request a persistent, versioned static report for an optimized graph,
see the correct `MatMulNBits` core count for supported concrete shapes, and understand every exclusion and missing
estimate without any change to inference behavior.

The **MatMulNBits counting milestone is complete** only when the MatMulNBits-first gate in Section 6 passes.
Shipping the core-only MVP does not satisfy that gate or authorize starting broader operator/timing work.

The **diagnostic feature is mature** when supported runtime invocations and matched throughput can be reported with
validated attribution, honest coverage, and measured overhead. Automatic performance planning remains a separate
feature, not a prerequisite for delivering useful counts.
