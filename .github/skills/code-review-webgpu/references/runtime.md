# Runtime, thread safety, and plugin contracts

**Owns:** Thread/session/callback state, capture, memory/transfers, and plugin ABI.
**Update when:** Execution/concurrency, callbacks, recording, allocation/loading,
host negotiation, APIs/exports, memory identity, or deployment contracts change.
Device configuration lives in [devices and features](devices-and-features.md).

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
  [cache identity](shaders.md#cache-identity).
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

## Plugin contracts

- Distinguish release version, build SDK/API bounds, minimum host, and runtime
  negotiation. Verify API and patch-release predicates in `ep/runtime_compatibility.*`;
  one fixed release line does not cover intervening lines. Preserve conservative
  compatibility/overrides; mocks do not prove multi-runtime GPU support.
- Initialize APIs before wrappers or exception-to-status macros; failed
  negotiation still needs supported status reporting. Audit transitive
  allocation/Free/destruction under `noexcept` C callbacks; contain exceptions
  using boundary conventions and preserve cleanup.
- Compare native `GetCapability` and plugin `ep/ep.cc` predicates for affected
  operators/attributes/dtypes/features/layouts. Shared source does not prove
  acceptance parity. Check evidence of graph acceptance/execution, rejection, and
  supported CPU partition boundaries.
- Distinguish `OrtMemType`, `OrtDeviceMemoryType`, and memory-device identity.
  Verify vendor/device IDs, default/read-only allocators, session/Env ownership,
  streams/notifications, and release order against the host ABI.
- Shared plugins need exact exports/runtime dependencies; static plugins need
  prefixed entry points/API initialization. Check factory registration/removal
  and context/logger/library teardown.
- New internal dependencies must reach `plugin-ep-webgpu/paths.txt` and EP
  adapters. Inspect other adapter consumers (including CUDA) and affected native,
  shared/static plugin, and WASM integration via [build checks](build-and-dependencies.md).

Inspect `ep/{api,ep,factory,allocator,sync_stream}.cc`, public EP APIs/adapters,
and session `ep_factory_webgpu.*`.
