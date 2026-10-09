## The `ep` folder

The current folder contains the implementation of EP ABI adapter for WebGPU.

The Session stream and notification bridge lives in `sync_stream.h` and `sync_stream.cc`.
The Session allocator ABI wrapper and plugin-only stream allocation implementation live in
`allocator.h` and `allocator.cc` in this folder.
Shared buffer management and data-transfer implementations remain in the parent directory.
Native builds exclude this folder.

### Design considerations

To ensure both static library and dynamic library builds work, we need to make as few changes to existing code as possible. A few design decisions are as below:

- No changes to static library build. It should still work as before.

- For dynamic library:

  - use and only use the EP ABI. (no support for `GetProvider`)

  - still depends on onnxruntime targets.

  - use a bridge to connect EP ABI and the internal classes

### Runtime compatibility

The plugin supports ORT 1.24.4 and later. Both execution modes use Session-owned recordings.
Concurrent independent Sessions are enabled on 1.28.x starting at 1.28.3, on 1.30.x starting
at 1.30.1, and on 1.31 and later. Other supported versions, including all 1.29.x hosts, use
single-thread compatibility mode, selected by `UseSingleThreadMode()`. There is no shared
legacy recording. Cached-buffer clearing remains enabled, and kernels still batch their
dispatches; there is no per-kernel submission and no traversal of other Sessions.

Single-thread callers must serialize **all WebGPU operations on the same device**, including Session
creation/destruction, Run, I/O binding, allocator use, and Env transfers. Use sequential graph
execution. Multiple Sessions may be used sequentially; overlapping operations are unsupported and
are not detected or serialized by the plugin. This is not a restriction to one fixed CPU thread,
but operations must not overlap or reenter from callbacks.
The plugin's same-Session Run concurrency flag alone cannot serialize separate Sessions or Env calls.

In single-thread mode, Session ordinary `Alloc` uses its owning Session's recording and a
`!IsRunActive()` submission-policy callback: submit cached-buffer clears outside Run, defer them
during Run. `AllocOnStream` uses the same policy after validating the stream's Session.
Kernel scratch uses ordinary Tensor allocation through that same allocator, without
`KernelContext_GetSyncStream`, which is unavailable on 1.24. Under the required serial,
non-reentrant calling contract, application allocations occur outside Run and submit before
returning. `OnRunEnd` cleanup resets the active flag on success and failure.
Env allocations and concurrent-mode ordinary allocations continue to submit independent clears,
including during Run. Submission does not wait for GPU completion. Other plugin allocators
without this explicit submission callback retain the independent-clear policy.

The context tracks a non-owning pointer to the active Session's recording during a single-thread
Run or replay. Framework copies with a stream use that stream's Session recording. If an old host
drops the stream, the copy first flushes the active Session's recording, then uses a local recording.
This preserves `clear -> upload -> compute -> readback` ordering without sharing command state.
Framework/Env copies and Env allocations use the default context buffer manager; graph execution
keeps its per-graph buffer manager. Capture/replay boundaries drain pending Session work.
BufferManager tracks deferred releases by recording, and Session teardown discards only that
Session's pending entries. Env allocations retain no Session recording and can survive Session teardown.
Run/replay callbacks clear the active pointer on their cleanup paths; Run cleanup also resets the
Session's graph-manager selection. There is no context-wide Run gate, so a host error that skips
`OnRunEnd` cannot permanently reject subsequent Runs. Such errors can still leave pre-existing
Run/capture state, and complete recovery from host-side or partially recorded replay failures
is not guaranteed.

The execution mode is selected once at plugin registration using the host's minor and patch
versions. Set `ORT_WEBGPU_EP_FORCE_LEGACY=1` **before loading the plugin** to exercise single-thread
mode on a host that supports concurrent Sessions. The existing override name is retained for
compatibility; it does not enable a legacy recording. The setting is process-wide and only forces
the safe compatibility direction; it cannot enable concurrency on an unsupported host.

### Session streams

Concurrent use of independent Sessions is supported only by the plugin WebGPU EP.
Built-in WebGPU does not support concurrent use. Each Session has one command recording
timeline without an internal mutex. ORT serializes Run calls on the same Session; callers
must serialize explicit stream and I/O binding operations with that Session's Run.
Streamless allocation, tensor release, and Env transfers can run concurrently on independent tensors.

The WebGPU plugin implements `OrtEp::CreateSyncStreamForDevice`. Each stream references its
owning EP's command state; kernels and stream-bearing data transfers use that same state.
Graph Memcpy kernels access their owning EP directly. The generic single-tensor transfer
wrapper also forwards explicit streams, keeping BindInput copies ordered with allocation clears.

Native devices must enable Dawn's `ImplicitDeviceSynchronization` feature. ORT requests it for
internally created devices; callers supplying an external device must include it in
`DeviceDescriptor.requiredFeatures` when creating that device. Initialization rejects native
devices without this feature, even for serial use. This requirement does not apply to WASM.

In concurrent mode, Session allocators expose the existing `OrtAllocator::AllocOnStream` callback and validate
that the stream belongs to the same Session. Allocations with a matching stream defer cached-buffer
clears. Plugin kernel scratch tensors created through `CreateGPUTensor` use the kernel's explicit
sync stream, so cached-buffer clears stay ordered with kernel work without submitting each scratch
allocation. Plain `Alloc` and null-stream allocations use a fresh encoder to submit cached-buffer
clears before returning, including during Run, without recording or flushing Session work.
This keeps CPU-produced outputs bound to GPU ordered with fallback uploads without
additional core stream creation. The policy depends on the allocation's stream, not `IsRunActive()`.
Uploads, readbacks, stream synchronization, dispatch batch limits, and Run/capture boundaries can
still submit work.

Env and Session allocators reuse the `GpuBufferAllocator` implementation. Session getters borrow
their owning EP for stream allocation and deferred buffer release. Env allocators retain a context
without a Session recording and track allocation counts atomically. They are created lazily by
`Factory::CreateAllocatorImpl` and use the streamless `adapter::Allocator` ABI wrapper without
exposing `OrtAllocator::AllocOnStream`. Independent allocation encoders and per-call transfer
recordings are specific to plugin builds; built-in WebGPU retains its existing recording policy.

Multiple threads may use an Env or Session allocator on independent tensors, including while the
Session runs. Keep a Session allocator and its owning Session alive until their tensors are released,
and synchronize writes before another Session consumes a shared tensor. BufferManager owns pending
releases for reusable buffers, grouped by recording and protected by its cache lock. Submission
returns only that recording's buffers to the pool; caches without reuse need no such tracking.
Pending lists are created only when buffers are released, with no buffer-manager registration per dispatch.
CommandRecordingState exposes an atomic unsubmitted-work flag so concurrent frees after submission can
return buffers directly to the cache. Submission and abandonment clear that flag under the cache lock,
preventing concurrent releases from missing cleanup. Context-shared program caches retain their locks.

Environment transfers with no stream use command state local to each CopyTensors call. GPU-to-GPU copies
without a stream submit before returning, but do not wait for GPU completion. A subsequent
Session Run uses a different recording, so the transfer must submit its copy first; the shared
device queue orders it before later Session work.

Stream flush and notification activation submit the owning Session's recording without a CPU
wait. Submission is still needed before streamless readbacks, such as node I/O dumps, use a
different recording. GPU consumers rely on the shared queue's submission order, and CPU
consumers synchronize through blocking readbacks; notification wait callbacks are no-ops.
These callbacks preserve WebGPU's existing submission-only behavior rather than providing a
general host-completion barrier. In particular, returning GPU-backed outputs from Run does
not guarantee that GPU execution has completed.

### Internal checked completion

`WebGpuContext::FlushAndWaitChecked(buffer_manager, recording)` is an opt-in,
Status-returning host barrier. It uses ordinary `Flush` to encode deferred work
and submit the owning recording, then waits for the same queue's
`OnSubmittedWorkDone` future through the context's existing Instance/WaitAny
machinery. Caller-provided instances must enable Dawn's `TimedWaitAny` feature
in `InstanceDescriptor.requiredFeatures`, as ORT does for its own instance.
The helper also waits when the recording is empty, covering work already
submitted to that queue. Work submitted after the completion future is
registered, or unsubmitted work on other recordings, is not covered.

Validation, out-of-memory and internal error scopes cover flushing and queue
submission independently of the configured validation mode. All three scopes
are resolved even on failure; caller-owned outer scopes are not consumed.
Scopes cannot retroactively capture errors from earlier resource creation or
encoding, and cannot enable validation on a device created with Dawn's
`skip_validation` toggle. ORT-created devices retain their first uncaptured
error for checked completion; ordinary Flush/Run behavior is unchanged.
For externally supplied devices, earlier errors captured by the caller's
scopes or callbacks remain the caller's responsibility.

The helper polls `Device::GetLostFuture` after waiting, rather than treating
a successful queue callback as proof of device health. This also detects loss
of an external device without replacing its callbacks. A failed wait or
work-done callback, scoped error, device loss, or retained uncaptured error
returns failure Status. One-shot callback results are heap-owned so a failed
WaitAny cannot leave a dangling stack pointer. Callers must keep the context,
recording and resources alive and serialize operations on the recording;
this is not a new stream or cross-provider API.

Focused context tests cover empty, pending and submitted work, deferred
dispatch, validation at Flush, scope cleanup after a failed wait,
validation/OOM injection and external device loss. Dawn's existing InjectError
API accepts validation and OOM only. Deterministic internal-error and
non-success work-done callback injection is not available here; those branches
are checked but lack injected end-to-end coverage. Device loss injection is
native-Dawn-only.

Deferred pipeline results are shared by the pending build and Dawn's one-shot
callback. A failed pipeline wait still returns failure and discards the deferred
dispatch window without encoding it; the callback retains its result state until
delivery or instance-shutdown cancellation. It does not retain the ORT context or
recording. Callback completion releases that ownership, so abandoned builds need
no permanent retention list. Native context tests force a failed timed wait with
an undelivered callback and verify cleanup after explicit delivery, including
delivery after ORT context destruction, as well as normal successful pipeline
completion. Native instance-shutdown cancellation is not injected: pending Dawn
events can retain the native instance, so the teardown test retains it explicitly
and delivers the callback after the ORT context is gone.

The implementation requires an ORT build with stream support. CPU I/O, graph-internal CPU/GPU
copies, mixed feed copies, CPU outputs bound to GPU, concurrent independent Sessions, serialized
same-Session Runs, same-Session and dedicated-Session allocator concurrency, and shared Env allocation/transfers
are covered by the tests in
`onnxruntime/test/providers/webgpu/plugin`, built into `onnxruntime_provider_test`.
The tests also cover initial graph capture and replay across independent Sessions, and concurrent
Run calls for different graph IDs within one Session, serialized by ORT. Initial-capture tests use
preallocated bindings; they do not cover external Session allocator calls overlapping capture.
The compatibility regressions additionally exercise single-input CPU BindInput with dirty-buffer
reuse, interleaved bindings across serialized Sessions, Env tensors surviving Session destruction,
and serial graph capture/replay with multiple graph IDs. A regression verifies that a rejected
device-mismatched stream override does not block subsequent serial Runs on either the failed
Session or another Session on the same device. Run serial tests with
`ORT_WEBGPU_EP_FORCE_LEGACY=1`; the concurrent-success tests apply only to modern mode.
The `onnxruntime_webgpu_legacy_test` CTest entry sets this environment variable before loading
the plugin and runs a single-thread-safe allowlist in normal PR CI. The existing CTest name is retained.
Public allocation tests verify dirty-buffer reuse and read the raw buffer with an independent
Dawn command encoder, so ORT's readback path cannot hide an unsubmitted clear. Submission counts
are also checked before that external readback, including after a cancelled Run.
Concurrent profiling, cross-device transfer, and successful arbitrary foreign stream overrides are not
established by these tests. Performance must be measured separately.

### Missing parts

This section describes what is missing.

- need a way to do WebGPU cleanup (`OrtEnv::~OrtEnv()` currently calls `webgpu::CleanupWebGpuContexts()` in static lib build)

- need a way to setup "default configurations" for WebGPU. (currently missing for both static lib and shared lib)
  - we want something like `SetCurrentGpuDeviceId` in ORT C-API, which set a global state and is directly available to user.
    - to make it general, it can be something like:
      ```c++
      ORT_API2_STATUS(SetEpDefaultConfig, _In_ const char* ep_name, _In_ const char* key, _In_ const char* value);
      ```
