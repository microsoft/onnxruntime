# Profiling and performance

**Owns:** GPU/CPU event correlation, profiling resource lifetime, instrumentation
effects, performance evidence, and scope of hardware workarounds.
**Update when:** Timestamp/profiler APIs, query scheduling, capture interaction,
benchmark configuration, or hardware-specific behavior change.

- Distinguish GPU execution duration, submit/queue delay, and end-to-end latency.
  Prove clock domains, units, origin, conversion/normalization, precision, and
  reset behavior before aligning GPU and CPU events. An offset based on first
  submit alone does not establish true hardware-clock correlation. External
  Dawn may require the owning host's clock API; inspect exact pinned behavior.
- Preserve per-dispatch metadata, including repeated kernel dispatches before
  a flush. Match each query pair/readback to its event and retain metadata and
  buffers across asynchronous completion. Check session-level versus run-level
  profiling, repeated runs, capture/replay, unsupported queries, and failures.
- Instrumentation must not silently change the algorithm, buffer cache,
  synchronization, or capture eligibility being measured. Preserve controls
  when reproducing hangs; disabling profiling changes the experiment.
- Performance claims need before/after model, shapes, dtype, GPU/backend/driver,
  options, build mode, warmup/repetitions, and method (`docs/PR_Guidelines.md`).
  Separate initialization/prepacking/compilation, steady runs, and replay.
  Control partitioning, CPU fallback, upload/readback, and cold/warm caches.
- Report a concrete mechanism for suspected regressions: per-run compilation,
  host readback, extra copies, lost vectorization, excessive padded waves, or
  retained memory. Less theoretical work is not a measured latency win.
- Scope vendor workarounds to supported device/backend/feature evidence, retain
  the correctness reason and fallback, and test unaffected paths. Historical
  failures do not establish a current universal vendor restriction.
- Trace/report consumers must preserve events crossing run boundaries and scale
  to long generation traces. Compare baseline configuration values structurally,
  independent of JSON key order; reject materially incomparable runs. Test
  repeated/overlapping spans and replay encoding as well as eager dispatches.

Source landmarks: `webgpu_profiler.*`, `webgpu_context.*` timestamp/query/flush
paths, Dawn platform integration, PIX support, and perf tests.
