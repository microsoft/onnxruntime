# Runtime and memory

**Owns:** Shared execution state, asynchronous errors, captured work and resource
lifetime, buffer reuse, transfers, and external-data loading.
**Update when:** Those ownership, feature, or deployment contracts change.
Keep kernel registration in registration and test selection in validation.

## Callbacks and concurrent sessions

- Never throw from a Dawn callback. Callbacks should be `noexcept`; do not use
  `ORT_ENFORCE`, `ORT_THROW`, or calls that throw. Store a `Status` in callback
  context and return it after waiting. The context must outlive the future,
  including failed waits; see `program_manager.h`'s lifetime comments.
- `ComputeInternal` is `const`. Sessions can run concurrently sharing a
  `WebGpuContext` and its program cache. Flag per-run mutable members, device
  objects or per-run data in function-local statics, and unsynchronized globals.
  Per-session options affecting WGSL need [cache identity](cache-identity.md).

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
