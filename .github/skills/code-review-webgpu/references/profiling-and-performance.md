# Profiling and performance

**Owns:** GPU/CPU correlation, profiling lifetimes/effects, performance evidence,
and hardware workaround scope.
**Update when:** Timestamp/query APIs, scheduling, capture interaction, benchmark
configuration, or hardware behavior change.

- Separate GPU duration, queue delay, and end-to-end latency. Prove clock domains,
  units/origin, conversion/normalization, precision, and reset behavior before
  alignment; first-submit offsets do not prove hardware-clock correlation.
  Native calibration requires ownership of the installed Dawn proc table;
  omitted options on a later context do not establish it. External Dawn may
  require the host's clock API; inspect the pinned behavior.
- Match each dispatch/query pair/readback/event, including repeated dispatches
  before flush; retain metadata/buffers through asynchronous completion. Check
  session/run profiling, repeated runs, capture/replay, unsupported queries, and failures.
- Instrumentation must preserve algorithm, cache, synchronization, and capture
  eligibility. Keep controls when reproducing hangs; disabling profiling changes
  the experiment.
- Claims need before/after model/shapes/dtype, GPU/backend/driver, options/build,
  warmup/repetitions/method (`docs/PR_Guidelines.md`). Separate initialization/
  prepack/compile, steady runs, and replay; control partitions, CPU fallback,
  transfers, and cold/warm caches.
- Tie regression concerns to per-run compilation, readback/copies, lost
  vectorization, padded waves, or retained memory. Less theoretical work does
  not prove lower latency.
- Scope vendor workarounds to device/backend/feature evidence; retain correctness
  rationale/fallback and test unaffected paths. Old failures are not universal
  current restrictions.
- Trace consumers must retain cross-run events and scale to long generation.
  Compare baseline configurations structurally, ignoring JSON key order; reject
  materially incomparable runs. Test repeated/overlapping spans and replay encoding.

Inspect `webgpu_profiler.*`, `webgpu_context.*` timestamp/query/flush paths,
Dawn platform/PIX integration, and perf tests.
