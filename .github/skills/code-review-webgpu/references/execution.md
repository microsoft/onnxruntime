# Execution and memory

**Owns:** How work is submitted and resources are shared/retained: threads,
callbacks, buffers, transfers, capture/replay, program-cache storage, and profiling code.
**Update when:** Scheduling, synchronization, resource ownership, callbacks, or event collection changes.
Cache-key correctness belongs to [computation](computation.md#cache-identity);
host/API compatibility belongs to [platform support and release](platform.md#plugin-contracts).

## Thread-safety contract

Establish implementation, host version, mode, and sharing boundary using the live
[plugin contract](../../../../onnxruntime/core/providers/webgpu/ep/README.md#runtime-compatibility),
`ep/runtime_compatibility.*`, and native/plugin concurrent-Run callbacks.

- Independent concurrent sessions require plugin hosts enabled by the runtime
  predicate; built-in WebGPU retains serialized use. Both callbacks currently
  reject overlapping same-session `Run`, which ORT serializes. Explicit stream/
  I/O-binding operations still need serialization with that session's Run.
- Plugin serialized compatibility mode requires callers to serialize **all
  same-device operations**: session creation/destruction, allocation, binding,
  transfers, Run/replay, without callback reentry. This is neither an internal
  lock nor thread affinity. Same-session flags do not serialize other sessions/Env.
- Native devices, including supplied ones, require Dawn
  `ImplicitDeviceSynchronization`; WASM does not. Dawn synchronization does not
  protect ORT state. Check WASM threads/workers and static plugins under actual guards.
- Inventory process/context-shared, session, per-run, and callback state.
  Recordings have one caller; plugin streamless operations need independent
  recordings. Audit cache lookup/insertion, duplicate builds, published lifetime,
  context/factory creation, allocation/free, error/profiling state, and shutdown.
  Returned cache pointers need stable storage after unlock, including eviction.
- Trace lock order through submission/callbacks/logging/destruction; never wait
  holding a lock completion needs. Check lazy initialization and every shared
  access; `const`, one atomic, or a thread-safe device is insufficient.

## Callbacks and kernel state

- Dawn callbacks should be `noexcept`: no `ORT_ENFORCE`, `ORT_THROW`, or transitive
  throws. Store `Status` in callback context and return after waiting. Retain
  context and objects used until delivery, even after failed waits; inspect
  `program_manager.h` lifetime comments.
- `ComputeInternal` is `const`; concurrent plugin sessions share context/cache.
  Flag per-run mutable members, device objects/per-run data in local statics,
  and unsynchronized globals. Session-varying WGSL options need
  [cache identity](computation.md#cache-identity).
- Check delivery mode/thread, result publication, failed waits, cancellation,
  device loss, and teardown. Failed waits do not cancel delivery. Preserve
  resources without retaining dead sessions indefinitely; contain callback/ABI failures.

## Concurrency validation

Assess coverage of affected scenarios in
[`plugin/test_webgpu_concurrency.cc`](../../../../onnxruntime/test/providers/webgpu/plugin/test_webgpu_concurrency.cc)
plus allocator/callback tests: simultaneous cold-cache builds, warm execution,
different sessions, same-session Run serialization, allocation/upload/readback
during Run/replay, and teardown with outstanding resources. Assert values and
completion with bounded waits. Stress/race evidence supplements ownership proof.

Compatibility coverage must set `ORT_WEBGPU_EP_FORCE_LEGACY=1` before plugin
loading in a fresh process and cover sequential interleaving, cleanup, and
failures/capture cleanup. Concurrency is unsupported there. Record native/
plugin/WASM/device gaps via [validation](validation.md).

## Graph capture

- Keep GPU-data-dependent decisions on-device (`SetIndirectDispatchTensor`);
  `ComputeInternal` readback stalls the queue and breaks capture.
- Clears/copies/uploads outside `RunProgram` use capture-aware context APIs,
  not raw Dawn calls. Allocate temporaries through context for capture lifetime;
  buffer-manager/allocator/session-pool/transfer edits need cross-kernel tests.
- Capture is a whole-run transaction: kernel/flush/validation/replay failures
  must not publish partial graphs. Check cleanup, retry, graph IDs/destruction;
  `OnRunEnd` does not prove success.
- Retain buffers/bindings through pending submissions and captured use,
  including graph release and session teardown.

## Allocation, transfers, and external data

- Trace effective allocator/transfer wrapper, context, buffer manager, and
  recording. Streamless work cannot borrow another session's recording/capture;
  submit required clears/uploads/copies before returning. Preserve session-stream order.
- Prove shared-container/reuse-flag synchronization; relaxed atomics need an
  ownership/happens-before argument. Check coverage of public allocation/upload/completion/
  readback during another session/replay; allocation/free misses deferred clears.
- Distinguish WebGPU handles/context IDs from other providers' GPU pointers.
  Preserve transfer directions, zero-byte copies, mapped buffers, batch fallback,
  and source lifetime.
- Binding aliases must share native allocation/session ownership that survives
  explicit source disposal; rooting a JS object alone is insufficient. Preserve
  caller-owned fetches and CPU/string conversions. Require idempotent disposal
  and finalizer cleanup after partial output-conversion failure. Compare opaque
  buffer handles by identity, not address ranges; assess numeric GPU consumption
  after producer release/disposal, not disposal-only tests.
- Separate live/pending/captured/cached/staging bytes. Bound retention across
  buckets/generations, idle tails, and transient peaks. Check fresh-context identity
  reuse under changing shapes; counts/warmed caches can hide churn.
- External loaders need safe offset/length arithmetic, alignment, destination
  identity, upload completion, and failure ownership. Check ordinary/accelerated
  paths and platform guards; unsupported modes must fail clearly or fall back.

## Profiling and traces

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

- Trace consumers must retain cross-run events and scale to long generation.
  Compare baseline configurations structurally, ignoring JSON key order; reject
  materially incomparable runs. Test repeated/overlapping spans and replay encoding.

Inspect `webgpu_profiler.*`, `webgpu_context.*` timestamp/query/flush paths,
Dawn platform/PIX integration, and perf tests.
