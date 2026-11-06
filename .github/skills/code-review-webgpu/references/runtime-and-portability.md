# Runtime and memory

**Owns:** Multi-thread support/thread safety, shared execution state, asynchronous
errors, captured work and resource lifetime, buffer reuse, transfers, and external-data loading.
**Update when:** Those ownership, feature, or deployment contracts change.
Keep kernel registration in registration and test selection in validation.

## Thread-safety contract

Establish the implementation, host version, execution mode, and sharing boundary
before judging a concurrency claim. Use the live
[plugin concurrency contract](../../../../onnxruntime/core/providers/webgpu/ep/README.md#runtime-compatibility),
`ep/runtime_compatibility.*`, and both native/plugin concurrent-Run callbacks.

- Concurrent independent sessions are supported by the plugin only on hosts
  enabled by the runtime-compatibility predicate. Built-in WebGPU retains its
  serialized-use contract. Do not equate independent sessions with overlapping
  `Run` on one session: both concurrent-Run callbacks currently return false,
  and ORT serializes those Runs. Explicit stream/I/O-binding operations still
  need serialization with their owning session's Run.
- In plugin serialized compatibility mode, callers must serialize all WebGPU
  operations on the same device, including session creation/destruction,
  allocation, binding, transfers, Run and replay, without callback reentry.
  This is a calling contract, not an internal lock or fixed-thread affinity.
  A same-session Run flag does not serialize other sessions or Env calls.
- Native devices require Dawn `ImplicitDeviceSynchronization`, including
  externally supplied devices; the requirement does not apply to WASM. Dawn
  synchronization does not protect ORT-owned containers or mutable state.
  Review WASM threads/worker and static-plugin paths under their actual build
  guards rather than assuming native threading guarantees transfer to them.
- Inventory process/context-shared, session-owned, per-run, and callback-owned
  state. A command recording has one caller at a time; plugin streamless
  operations use independent recordings. Verify cache lookup/insertion,
  duplicate pipeline builds, published artifact lifetime, context/factory
  creation, allocation/free, error/profiling state, and shutdown. A returned
  cache pointer needs stable storage after unlocking, including any new eviction.
- Trace lock order across submission, callbacks, logging, and destruction.
  A wait while holding a lock needed by completion deadlocks. Verify lazy
  initialization and every access to shared state; `const`, an atomic field,
  or a thread-safe Dawn device alone does not establish thread safety.

## Callbacks and kernel state

- Never throw from a Dawn callback. Callbacks should be `noexcept`; do not use
  `ORT_ENFORCE`, `ORT_THROW`, or calls that throw. Store a `Status` in callback
  context and return it after waiting. Callback result state must outlive
  delivery, including after failed waits; see `program_manager.h`'s lifetime
  comments. Retain the objects actually used by the callback.
- `ComputeInternal` is `const`. Supported concurrent plugin sessions share a
  `WebGpuContext` and its program cache. Flag per-run mutable members, device
  objects or per-run data in function-local statics, and unsynchronized globals.
  Per-session options affecting WGSL need [cache identity](cache-identity.md).
- Check the callback mode and delivery thread, publication of its result,
  failed waits, cancellation, device loss, and teardown. A failed wait does not
  cancel delivery. Keep context/resources alive as required without retaining
  dead sessions indefinitely; contain failures across callback/ABI boundaries.

## Concurrency validation

Select scenarios from
[`plugin/test_webgpu_concurrency.cc`](../../../../onnxruntime/test/providers/webgpu/plugin/test_webgpu_concurrency.cc)
and allocator/callback tests according to the changed contract. Cover
simultaneous cold-cache builds and warm-cache execution; different sessions;
same-session Run serialization; allocation/upload/readback during Run or replay;
and session teardown with outstanding resources. Check output values and
completion with bounded waits, not just absence of crashes. Repeated stress or
a race detector, when supported, supplements ownership/order evidence.

Test serialized compatibility separately by setting `ORT_WEBGPU_EP_FORCE_LEGACY=1`
before plugin loading in a fresh process. Exercise sequential interleaving and
cleanup; concurrent operations in that mode are unsupported. Also test relevant
failure paths and capture cleanup. Native/plugin, WASM, and device-specific
coverage gaps must remain explicit; the selected tests do not establish all of
them. Use [validation](validation.md) for execution evidence and fallback checks.

## Graph capture

- Do not read GPU data back in `ComputeInternal` for decisions: it stalls the
  queue and breaks capture. Keep data-dependent work on-device, for example
  using `SetIndirectDispatchTensor`.
- Clears, copies, and uploads outside `RunProgram` must use context APIs that
  record graph capture, not raw Dawn calls.
- Allocate temporaries through the context rather than raw `wgpu::Device`
  calls so their lifetime matches the captured graph. Buffer-manager,
  allocator, session-buffer-pool, and data-transfer changes affect every kernel
  and need targeted tests.
- Treat capture as a transaction across the whole run: kernel, flush, validation,
  or replay failure must not publish a replayable partial graph. Check cleanup,
  retry, graph IDs, and destruction; `OnRunEnd` running is not proof of success.
- Retain buffers and binding resources until pending submissions and captured
  commands no longer need them, including graph release and session teardown.

## Allocation, transfers, and external data

- Trace the effective allocator/transfer wrapper, owning context, buffer manager,
  and command recording. Streamless work must not borrow another session's
  recording or capture state. Submit required clears/uploads/copies before
  returning; session-stream operations must preserve that stream's order.
- Prove synchronization for each shared container and reuse flag. Relaxed
  atomics require an ownership/happens-before argument, not just atomic access.
  Exercise public allocate, upload, run completion, and readback concurrently
  with another session/replay; allocation/free alone misses deferred clears.
- Transfer eligibility must distinguish WebGPU handles from other providers'
  GPU pointers and incompatible context IDs. Preserve CPU/GPU directions,
  zero-byte copies, mapped buffers, batch fallback, and source lifetime.
- For buffer-cache policy, separate live, pending, captured, cached, and staging
  allocations. Bound retained bytes across buckets and generations, including
  idle tails and transient peaks. Test fresh contexts and identity reuse under
  changing shape distributions; counts or warmed caches can hide churn.
- External-data loaders must preserve offset/length arithmetic, alignment,
  destination memory identity, upload completion, and ownership on failure.
  Check ordinary and accelerated loaders with their actual platform guards;
  an unsupported build mode must fail clearly or take a supported fallback.

Device capability checks live in [devices and features](devices-and-features.md);
deployment checks in [builds and dependencies](build-and-dependencies.md).

## Regression anchors

- [#29625](https://github.com/microsoft/onnxruntime/pull/29625): exceptions from callbacks.
- [#32587](https://github.com/microsoft/onnxruntime/pull/32587): concurrent-session state.
- [#33509](https://github.com/microsoft/onnxruntime/pull/33509), clears, and
  [#33362](https://github.com/microsoft/onnxruntime/pull/33362), copies: work missing from capture.
- Subgroup-matrix configuration fix for WebAssembly: verify non-native table initialization.
- [#32874 streamless ordering](https://github.com/microsoft/onnxruntime/pull/32874#discussion_r4165616715)
  and [the relaxed-atomic invariant](https://github.com/microsoft/onnxruntime/pull/32874#discussion_r4141059183).
- [#32587 deferred native allocation issue](https://github.com/microsoft/onnxruntime/pull/32587#discussion_r4068605107):
  a resolved thread explicitly deferred the concern; verify the reviewed path.
