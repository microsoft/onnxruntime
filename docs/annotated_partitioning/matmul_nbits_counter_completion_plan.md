# MatMulNBits Counter: Validation, Hardware Coverage, and Timing Completion Plan

## 1. Objective and definition of done

Finish a developer-facing diagnostic that reports useful, auditable MatMulNBits workload counts and measured
throughput without changing inference results, dispatch, or memory policy.

This is the execution plan following Batches 1-3 of the
[contrib-op rollout](contrib_op_flop_estimation_rollout.md). It focuses on MatMulNBits; GQA remains a separate
required operator in the broader rollout, not a prerequisite for validating this counter. Exhaustive
hardware-instruction accounting must not delay either baseline.

The deliverable has three distinct levels:

| Level | Deliverable | Accuracy contract |
| --- | --- | --- |
| Logical workload | Runtime core and bias counts | Exact under the declared convention for supported captured invocations |
| Implementation work | Optional route-specific estimates | Approximate, with documented assumptions, evidence and supported configurations |
| Measured throughput | Count divided by matched execution duration | Measured rate with explicit numerator, timing scope and uncertainty |

**FLOPs are a count; FLOP/s is a rate.** Adding timing does not turn a logical count into literal hardware work.
For quantized MatMulNBits, the primary rate must be named **dense-equivalent operations/s** or
**dense-equivalent TOP/s**, not actual executed floating-point TFLOP/s.
An estimated floating-point-only numerator can have its own estimated FLOP/s metric when justified.
Do not combine integer/bitwise operations, floating-point operations, conversions and bytes into one FLOP count.

The feature is complete for a declared platform when:

- The user's model has passed count and output-equivalence validation on that platform.
- Supported executed MatMulNBits invocations are accounted for, with missing coverage explicitly identified.
- Reference arithmetic checks pass exactly; optional implementation models meet their own evidence-based gates.
- Rates use correlated counts and durations, with no silent fallback to host launch timing.
- Count-only, timing-enabled and export overhead have been measured separately.
- Unsupported routes/platforms/replay modes remain explicit rather than returning misleading zero counts.

## 2. Starting point: implemented versus unverified

Implemented by Batches 1-3:

- Bounded typed per-run collector, run identity/outcome, diagnostics and subgraph propagation.
- CPU MatMulNBits runtime shapes, bias presence, quantization metadata and selected route/effective-mode provenance.
- Checked core `2MNK` and bias `MN` formulas, with independent components.
- Opt-in schema-v3 runtime JSON export after canonical Run, with unique process/session/run filename suffixes.
- Configurable automatic limits: 10,000 records and 16 MiB accounted payload per run by default.
- Explicit overflow, unavailable estimates, drops and captured-scope completeness.

Not implemented or not qualified:

- CUDA capture, GPU architecture provenance, GPU completion-aware collection and device timing.
- Timing/rate metrics, route-adjusted work, complete model coverage or standard MatMul/Gemm instrumentation.
- Production overhead qualification, all platform/build combinations, and the user's own model validation.

Existing implementation anchors:

- [Collector contract](../../onnxruntime/core/framework/runtime_workload_collector.h).
- [Runtime report builders](../../onnxruntime/core/framework/compute_estimate_report.h).
- [Pure core/bias formulas](../../onnxruntime/contrib_ops/cpu/quantization/matmul_nbits_compute_estimate.cc).
- [CPU dispatch and recording](../../onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc).
- [CUDA implementation to extend](../../onnxruntime/contrib_ops/cuda/quantization/matmul_nbits.cc).

Prior CPU Debug results are baseline evidence, not proof for a new build: 88 estimator/report/transformer tests
passed, with 25 collector/execution passes and one capability-gated LUT skip. Re-run after implementation changes.

## 3. Reviewable implementation sequence

| Batch | Work | Prerequisite | Review evidence |
| --- | --- | --- | --- |
| 4 | Validate current CPU behavior on synthetic and user models | Existing Batches 1-3 | Reproducible runs, independent counts and coverage reconciliation |
| 5 | Fix demonstrated gaps and qualify CPU overhead | Batch 4 findings | Focused regressions and before/after measurements |
| 6 | Provider-neutral provenance and CUDA count capture | Stable logical contract; CPU fixes affecting shared transport | Cross-device count parity and explicit actual CUDA routing |
| 7 | CPU and CUDA timing with matched rates | Stable identity; CUDA capture for GPU timing | Correlated timelines, timing accuracy and overhead results |
| 8 | Optional implementation-work models and hardware qualification | Actual route evidence; timing where needed for profiling | Per-route formulas, held-out validation and supported-range table |
| 9 | Developer rollout and expansion to further EPs | Count/timing acceptance on each advertised platform | Usage recipe, known limitations and rollback instructions |

Keep each batch independently reviewable. Capture numerical baseline improvements before adding refinements.
CPU timing can proceed before CUDA timing if the shared identity contract is settled.
Do not add GPU-specific arithmetic multipliers just to make the plan appear hardware-aware.

## 4. Batch 4: test the existing work

### 4.1 Verify the actual runtime artifact

1. Build the branch containing Batch 3; record revision, build configuration and enabled EPs.
2. For Python, build/install that branch's wheel into an isolated virtual environment.
   Record `onnxruntime.__file__`, version and available providers, plus wheel/build provenance.
   A published wheel accepting a configuration key does not prove it contains the counter.
3. Keep the CPU Debug test build separate from the Python wheel: the existing
   `build\Windows\FlopEstimate` configuration has Python bindings and CUDA disabled.
4. Run C++ tests from the build output directory. Confirm a nonzero test count and that the changed targets rebuilt.

Existing targeted commands on Windows, after building the corresponding targets:

```powershell
Set-Location C:\Users\tommasoadani\repos\onnxruntime\build\Windows\FlopEstimate\Debug

.\onnxruntime_provider_test.exe --gtest_filter="ComputeEstimateReportTest.*:RuntimeComputeEstimateReportTest.*:MatMulNBitsComputeEstimate*:MatMulNBitsBiasEstimate*"
.\onnxruntime_test_all.exe --gtest_filter="RuntimeWorkloadCollectorTest.*:RuntimeWorkloadExecutionTest.*"
```

See the repository [build skill](../../.github/skills/ort-build/SKILL.md) and
[test skill](../../.github/skills/ort-test/SKILL.md) for wheel/build commands and test execution guidance.
Missing route capabilities must skip explicitly; skipped cases remain unvalidated on that machine.

### 4.2 Run a small hand-checkable model

Use the existing synthetic end-to-end report test first: A=[2,3,64], N=32 and bias.

```text
M = product(A dimensions except the last) = 6
core dense-equivalent operations = 2 * 6 * 32 * 64 = 24576
bias logical floating-point additions = 6 * 32 = 192
```

The local [MatMulNBits-plus-Add model](../../onnxruntime/test/testdata/transform/runtime_optimization/matmulnbits_add.onnx)
is another candidate. Its [generator](../../onnxruntime/test/testdata/transform/runtime_optimization/matmulnbits_add_gen.py)
defines A=[M,2], N=3, 4-bit weights and a separate Add. With optimizations disabled and M=4:
core=48, internal bias=0; the separate Add is outside this counter.
Execute and inspect the actual artifact before treating it as validated.
With optimizations enabled, inspect whether Add fused into MatMulNBits and update the component expectation.

### 4.3 Test the user's own model

1. Start with a CPU-compatible ONNX model containing `com.microsoft::MatMulNBits`.
   Inspect operator domains/types and the optimized executable graph, not just the filename or quantization label.
   A QDQ-plus-MatMul model or a fused opaque partition does not automatically qualify.
2. Save representative real input shapes/settings: batch size, prompt length, decode steps and cache usage as applicable.
   Keep model/input contents local; record only non-sensitive metadata needed to reproduce the experiment.
3. Create a fresh report directory. Enable `session.compute_estimate_report_path` with an output prefix,
   not a literal JSON filename. Leave explicit budgets large enough for the planned run.
4. Run without reporting and retain reference outputs. Run with reporting using identical inputs/settings.
   Apply an appropriate predeclared dtype/backend tolerance; use exact equality where deterministic and appropriate.
5. Verify initialization writes no report and each Run produces one distinct schema-v3 report.
6. Require successful run state, no drops/reuse rejection, available core/bias components and captured-scope completeness.
7. Reconcile executed invocation identities/counts with an independent trace or controlled graph execution.
   Static node counts alone do not account for branch selection, loop iterations, decoder subgraphs or optimization.
8. Recompute each recorded core/bias count independently and compare both per-invocation values and checked totals.
9. Repeat for dynamic batch/sequence sizes, multiple runs, concurrent callers and relevant subgraphs.
10. Repeat with zero record/payload budgets, disabled reporting, invalid output directory and intentional inference failure.
    Inference behavior must remain unchanged; diagnostics and partial results must reflect what happened.

**Do not accept an empty report as a successful count validation.**
`collection_complete` describes attempted submissions, not all-model coverage.
`whole_run_work_complete` and `whole_operator_work_complete` remain false for this scoped feature.

Deliverable: a local validation bundle with build provenance, input shapes, configuration, output checks, generated
reports, independent reference totals, invocation reconciliation and unresolved gaps. Do not commit model weights,
tensor contents or routine logs.

## 5. Counting strategy and accuracy evidence

### 5.1 Maintain three independent checks

| Check | Method | What it establishes |
| --- | --- | --- |
| Formula correctness | Hand examples and independent arbitrary-precision reference arithmetic | Exact counts under convention, including overflow behavior |
| Runtime coverage | Compare executed identities/multiplicity with independent execution evidence | Records are not missing, duplicated or attributed to the wrong run |
| Implementation work | Inspect actual selected backend path and corroborate with targeted profiling | Optional route estimates reflect the implementation within declared assumptions |

Reference arithmetic should use actual A dimensions and logical K/N, never packed-weight storage dimensions.
Do not call the production helper from the reference oracle or multiply by backend batch_count after using full A shape.
Use arbitrary-precision integers in the oracle, then check whether the result fits the production uint64 contract.

Cover aligned/unaligned K/N, vector inputs, batching, bias presence, supported bit widths/block sizes, empty outputs,
zero reduction where supported, repeated subgraphs and unsupported metadata.
Test helper-only semantic cases separately from configurations that an actual kernel accepts.
Property tests should check linear scaling of counts, route-independent baselines and absence of wraparound.

### 5.2 Count actual implementation work only where justified

Manually audit a bounded set of route families once, then automate formulas from runtime metadata:

- Padded matrix/tile work only if those lanes actually perform arithmetic; padding storage alone is not proof.
- Dequantization and scale/zero-point arithmetic, distinguishing cached preparation from per-invocation reconstruction.
- Repeated reconstruction within tiles, split-K reductions, conversions and separate/fused bias work.
- Packing/prepacking and tactic search as initialization/first-use work, not automatically charged to every steady-state run.

For each model, document formula, arithmetic categories, route/version applicability, lifetime, required metadata,
excluded work and evidence. Missing required details mean unavailable adjustment, not zero extra work.
Preserve the original logical count and convention; add separately named/versioned implementation components.

Use backend source, controlled microbenchmarks and profiler observations together.
Profiler hardware counters are corroboration, not universal ground truth: instruction-to-operation mapping,
tensor-core tile dimensions, predication, precision, replay and counter availability vary by architecture.
Record metric definitions/tool versions and isolate the intended operation. Do not infer FLOPs from utilization
percentages or treat matching latency as proof of a correct operation count.

### 5.3 Accuracy acceptance

- Logical counts: exact agreement with all supported reference cases; tolerance is zero.
- Coverage: every expected supported invocation reconciles in controlled tests with no unexplained duplicates or drops.
- Implementation models: publish error/limitations per route/configuration where independent measurements exist.
  If no defensible reference exists, label the result model-based/unvalidated rather than claiming a percentage.
- Performance prediction is optional and separate from measuring actual duration.
  Any predicted latency requires held-out shapes and device/backend-specific calibration.
- Do not advertise a universal accuracy percentage or arbitrary uncertainty band before collecting evidence.

### 5.4 Route order and the refinement loop

Use the following order rather than attempting to model every CPU implementation simultaneously:

1. **Unpacked floating-point route:** establish reconstruction frequency, logical GEMM work, actual padding behavior
   and bias placement. Start with one CPU configuration and hand-checkable cases, then validate held-out shapes
   and the user's model where that route actually executes.
2. **Packed QNBit:** choose one available effective compute mode, validate its arithmetic categories and reuse
   assumptions, then expand to other modes/CPU capabilities individually. Do not equate quantized dot products
   with floating-point GEMM just because the dense-equivalent count matches.
3. **LUT:** establish table construction, lookup/accumulation work and reuse frequency before publishing adjustments.
   Missing LUT capability is an explicit coverage gap, not evidence that the route model passed.
4. **CUDA routes:** apply the same loop using actual device/route evidence and the completion rules in Section 7.

Keep the baseline tests running for every route, even while only one route's refinement is under development.
Advance a refinement when its reference cases pass, actual route execution is proven, supported assumptions are
documented and held-out cases expose no unexplained discrepancies. Other routes may retain baseline-only support.

The repeatable loop is:

```text
Choose route and hypothesis
    -> generate controlled factor matrix
    -> run and prove actual route
    -> compare with independent arithmetic/coverage references
    -> investigate discrepancies using backend/profiler evidence
    -> revise formula or collected metadata
    -> add focused regression
    -> rerun reference and held-out cases
    -> validate on user's model
    -> accept documented scope or repeat
```

Changing the formula is not always the fix: a mismatch may be missing capture, wrong route attribution, duplicated
batches, a different preparation lifetime or an invalid test configuration. Record the cause and evidence.
Do not tune arithmetic coefficients merely to make elapsed time agree.

### 5.5 Planned automation: route-validation matrix runner

**Yes, this workflow can be scripted.** Add a Python CLI as a follow-up implementation task, with proposed location
`tools\python\matmul_nbits_counter_matrix.py`. This script does **not** exist as part of this documentation change.
It should orchestrate model generation/execution, independent baseline checks and result aggregation; it cannot
automatically discover correct implementation-work formulas from timing alone.

Implement a CPU unpacked pilot first, then extend the same runner to packed QNBit, LUT and CUDA.
Use existing model-generation/quantization helpers where suitable, but keep the arithmetic oracle independent
of the production estimator. Use an isolated environment with a wheel containing the feature.

**Configurable factors**

| Group | Factors to sweep |
| --- | --- |
| Geometry | M/N/K, leading batch dimensions, rank, aligned/irregular dimensions |
| Semantics | Bias absent/present, supported bit widths/block sizes, activation dtype |
| Route controls | Requested route/mode, constant versus dynamic weights where supported, relevant session settings |
| Lifecycle | Fresh session, first invocation, warmup and repeated steady-state invocations |
| Execution | EP/device, CPU thread count, serial/concurrent run cases |
| Diagnostics | Disabled, capture-only where accessible, export enabled; collection budgets |

Use a small one-factor-at-a-time cohort first, followed by selected interactions and a separate held-out cohort.
Do not blindly run the full Cartesian product: preflight configuration validity, budget total cases/runtime,
and report unsupported combinations explicitly. Keep failed attempts visible rather than silently filtering them out.
CPU-only shared settings and CUDA-only settings must not be applied interchangeably.

**Runner responsibilities**

- Assign a stable case ID, matrix/cohort version and deterministic input seed.
- Record build/wheel provenance, model identity, device/backend configuration and requested settings.
- Run controlled cases in isolated output directories; isolate processes when session/global backend settings require it.
- Assert the **observed** route/effective mode from structured records, not logs or requested configuration.
  Route controls are requests, not guarantees. A route mismatch must remain a visible non-pass outcome.
- Detect missing reports, failed runs, missing/duplicate invocations, drops and unavailable estimates.
- Compare outputs against a predeclared reference/tolerance and compute baseline counts using arbitrary-precision
  reference arithmetic. Preserve decimal counts without floating-point precision loss in exports.
- Include route-adjusted expected/measured components only when a validated reference and required metadata exist.
  Missing reference data is unavailable, not zero error.
- Separate arithmetic checks, coverage checks, numeric correctness and timing checks into distinct outcomes.
- Preserve requested repetitions, warmup policy and variability; initially leave unsupported timing fields unavailable.
  CPU Run wall time includes synchronous export when enabled and must not be labeled operator compute time.
- Return a failing exit status for required-case failures, route mismatches or unexpectedly missing coverage.
  Capability skips can be allowed by the declared matrix policy but must be counted and listed.

**Decision-friendly output**

Produce detailed machine-readable JSON, a flat CSV and a generated Markdown summary in the chosen results directory.
Keep per-invocation evidence available separately; aggregate by route/mode, geometry cohort and device.
Suggested columns:

```text
case_id, cohort, device, EP, dtype, bits, block_size, A_shape, N, bias,
requested_route, observed_route, effective_mode, lifecycle,
expected_invocations, captured_invocations, collection_complete,
expected_core_ops, reported_core_ops, core_delta,
expected_bias_ops, reported_bias_ops, bias_delta,
refinement_version, implementation_reference_status, implementation_error,
output_check, timing_scope, duration, duration_unit, timing_status, repeats, variability,
status, reason, evidence_path
```

Use explicit statuses such as pass, arithmetic mismatch, output mismatch, route mismatch, incomplete capture,
unsupported configuration, capability skip and execution failure. Summary tables should show how many requested
cases actually exercised each route, all non-pass reasons, worst discrepancies and unvalidated configurations.
Link failures to raw reports and profiler/reference evidence. A dashboard is optional; CSV and Markdown suffice first.

At each refinement iteration, compare matrices with the same case/cohort IDs and retain formula/build versions.
Review where counts or statuses changed and add minimal failing cases to the normal unit/regression suites.
Do not adjust models using the held-out cohort without replacing that cohort for subsequent independent validation.

## 6. Batch 5: refine only demonstrated gaps

Prioritize fixes from Batch 4 and retain a regression for each:

1. Correct arithmetic, identity, missing-subgraph or export behavior before adding new metrics.
2. Reconcile graph-path encoding and schema semantics where cross-tool consumers need stable identities.
3. Surface coverage exclusions more usefully; distinguish known excluded nodes from unknown dynamic execution counts.
4. Qualify failed/async runs, collector lifetime and report publication semantics.
   Consider atomic report publication if partial-file visibility causes consumer errors.
5. Measure collector storage, snapshot/serialization memory, export duration and growth in report files.
   Add retention/sampling controls only if usage evidence requires them; preserve explicit partial-coverage semantics.
6. Retire legacy graph adapters only after their remaining tests/consumers are understood; retain useful pure math.

Benchmark Release builds with reporting disabled, capture only, and capture plus export.
Include tiny operations, decode-like small M and larger prefill-like M, with repeated trials and variability.
Set an overhead budget with the team before production rollout; no numeric budget is claimed as achieved here.

## 7. Batch 6: expand to CUDA, then other EPs

### 7.1 Shared transport and schema

- Separate logical operator metadata from provider-specific route details.
  Current route/mode enums describe CPU paths; do not relabel CUDA routes as CPU LUT/QNBit modes.
- Make shared-provider/plugin boundaries explicit: use supported bridge mechanisms or a boundary-safe typed contract,
  not direct in-tree framework dependencies from a provider DLL.
- Reuse pure arithmetic across providers through an appropriate shared layer; avoid linking CUDA to CPU kernel code.
- Generalize the report's current CPU-only provider check and coverage label as part of CUDA enablement.
- Capture device identity/architecture and backend/build versions once per session/device.
  Correlate records to the actual device in multi-device sessions; GPU name alone is not a dispatch key.

### 7.2 CUDA recording and lifecycle

Audit actual dispatch branches in the CUDA implementation, including mixed-weight GEMM, architecture/layout-specific
paths, fallback dequantization/GEMM and tactic selection where applicable. These are inspection targets, not a claim
that every branch is available on every GPU or build.

- Record actual host-visible dimensions, bias, precision, quantization settings and selected path.
- Record tile/tactic/reduction information only where exposed and needed by a supported refinement.
- Do not transfer tensors to the CPU or synchronize a device solely to calculate baseline counts.
- Distinguish host submission, device completion and failure. Successful launch is not confirmed GPU completion.
- Define deferred completion/event ownership before closing snapshots or reporting complete GPU results.
  Keep pending/unverified records explicit when callers disable synchronization.
- Ensure profiling/autotuning trial launches do not become multiple logical invocations.
- Keep CUDA Graph capture/replay unavailable until replay-aware identities, counts and timing are validated.

### 7.3 Hardware and build matrix

Use available representatives of Ampere (for example A100), Hopper (H100) and Blackwell (B100/B200-class),
plus representative CPU configurations. Availability is not presumed: list untested hardware explicitly.
Record compute capability, driver/toolkit/library versions, build flags, dtype, prepacked layout and selected route.
Source support for one architecture or prepacked layout is not evidence that another supports it.

Validate:

- Same executed logical shapes produce identical baseline counts across hardware.
- EP-specific tests prove actual CUDA dispatch, not successful CPU fallback.
- Irregular dimensions, small/large M, bias, supported quantization settings and dynamic weights where allowed.
- Architecture-dependent route eligibility, fallback and resource/launch limits.
- Native/shared-provider, Windows/Linux and selected build guards for the advertised support matrix.
- Unsupported device/layout/build combinations give explicit diagnostics, not misleading count-only success.

Further EPs follow the same contract. Opaque fused partitions require their own attribution strategy;
do not fabricate per-node records from pre-partition graphs.

## 8. Batch 7: timing and measured throughput

### 8.1 Define timing scope before implementing rates

| Scope | Numerator | Duration | Label |
| --- | --- | --- | --- |
| CPU operator | Captured operator logical core count | CPU wall interval covering the operator computation, including its worker completion | Dense-equivalent operator throughput |
| CUDA operator | Captured operator logical core count | Correlated device interval covering all its constituent compute kernels | Dense-equivalent device throughput |
| End-to-end Run | Explicit supported-work subtotal | Completed run wall time, with transfer/export inclusion stated | Supported-work throughput over run latency, not whole-model FLOP/s |
| Optional floating-point implementation component | Validated estimated FP-only work | Matched component interval, or explicitly labeled enclosing-operator interval | Estimated FP work per matched/enclosing interval |

A core-only logical numerator over the full operator duration is a useful **effective** throughput metric,
including bias/dequantization costs in the elapsed interval; it is not isolated core execution throughput.
Name this convention explicitly. Never divide the full operator count by only its fastest child kernel.

### 8.2 Correlation and instrumentation

1. Reuse existing ORT profiling infrastructure where it provides the required scope.
2. Introduce stable run/node/invocation correlation shared by records and timing events.
   Current collector invocation IDs are assigned on successful submission; do not match events by list position,
   node name, or commit order. Repeated/concurrent subgraph calls need distinct identities.
3. CPU: use a monotonic wall clock around the defined compute region, not CPU thread time that misses worker work.
   State whether recorder overhead is included; keep JSON serialization/file I/O outside the compute interval.
4. CUDA: use correlated CUDA events on the actual stream or suitable provider profiling.
   An operator spanning multiple streams requires dependency-aware attribution; a single-stream event pair is insufficient.
5. Reclaim bounded event/profiling resources after completion; surface exhaustion or unsupported timing as unavailable.
   Count-only mode must not add event/synchronization overhead. Timing collection can be sampled explicitly.
6. Export finalized timing/rates only when completion is established.
   Define an explicit finalize/deferred-export path where required, rather than forcing synchronization in count-only mode.
7. Separate warmup, first-use packing, tactic search and steady-state samples.
8. Version the report when adding timing fields. Include duration units, clock/scope, correlation identity,
   completion state, availability reason and timing methodology.

### 8.3 Calculation and aggregation

```text
dense_equivalent_ops_per_second = matched_dense_equivalent_ops / duration_seconds
dense_equivalent_TOP_per_second = dense_equivalent_ops_per_second / 1e12
estimated_FP_ops_per_second = matched_estimated_FP_ops / duration_seconds
```

- Omit a rate for missing, nonpositive, invalid or uncorrelated duration, failed/pending device work or unavailable count.
- Never infer time from GPU peak TFLOPS and present it as a measurement.
- Aggregate matched invocation samples with sum(counts)/sum(durations), not the arithmetic mean of their rates.
- Keep per-operator accumulated time separate from run wall time: overlapping intervals are not additive latency.
- Exclude warmup from steady-state aggregates and report median/tail latency, sample count and variability.
- Preserve incomplete/dropped sample counts; no full-run throughput claim from an unknown partial numerator.
- Keep bias logical additions separate. Whole-operator time is not an isolated bias duration.

Validate event duration/correlation against independent timelines, with concurrent runs and multi-kernel routes.
Compare uninstrumented, count-only, timing-enabled and full-export runs to measure observer overhead.
Timing acceptance thresholds must account for timer resolution and very short operations and be approved before rollout.

## 9. Batch 8: optional hardware-aware refinement

Add a refinement only when it answers a consumer question that the baseline and measured duration cannot.
Start with the **CPU unpacked floating-point route**, then a bounded **packed QNBit** configuration, then **LUT**.
Validate one available CUDA route afterward, then broaden only after evidence supports each model.

Each refinement must include:

- A route/architecture/backend-version applicability key, not a GPU-model correction factor.
- Independent logical baseline, named modeled components and separate arithmetic/traffic categories.
- Source-based derivation plus controlled measurements and held-out workload validation where possible.
- Explicit unsupported/extrapolated cases and model/convention version.
- Preparation/reuse assumptions and whether work is amortized across invocations.

For hardware performance comparisons, use measured rates on the actual devices.
If prediction for an unavailable device is needed, treat that as a separate calibrated model with error reporting.
Keep architecture-specific instruction auditing bounded; useful ballpark estimates do not require exhaustive assembly counting.

## 10. Rollout gate and audit checklist

- [ ] CPU reference arithmetic and existing regression suites pass on the final build.
- [ ] User-model inputs, output equivalence, invocation coverage and per-component totals are reconciled.
- [ ] Dynamic/repeated/concurrent/subgraph, zero, overflow, failure, drop and disabled cases pass.
- [ ] Report migration, coverage exclusions, filesystem failures and resource budgets are documented.
- [ ] Release-build collection/export memory and latency overhead are measured against agreed budgets.
- [ ] Every advertised EP/device/build combination has actual execution evidence; skips/untested cases are listed.
- [ ] GPU submission/completion and graph replay semantics are honest.
- [ ] Timing events correlate to the same invocations as counts; async/multi-kernel cases are verified.
- [ ] Rates distinguish logical dense-equivalent throughput from estimated executed floating-point work.
- [ ] Optional route refinements meet their own evidence gates without modifying the portable baseline.
- [ ] Usage examples and generated reports are available for reviewer sign-off.

Ship count-only and timing as independently opt-in modes. Start with a limited developer preview on validated CPU
and CUDA configurations. Roll back by disabling diagnostics, not by changing inference dispatch.
Broader GQA support and other EPs can reuse the validated contract without waiting for exhaustive route refinement.
