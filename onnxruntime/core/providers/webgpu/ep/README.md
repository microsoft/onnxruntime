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

Session allocators expose the existing `OrtAllocator::AllocOnStream` callback and validate
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

The implementation requires an ORT build with stream support. CPU I/O, graph-internal CPU/GPU
copies, mixed feed copies, CPU outputs bound to GPU, concurrent independent Sessions, serialized
same-Session Runs, same-Session and dedicated-Session allocator concurrency, and shared Env allocation/transfers
are covered by the tests in
`onnxruntime/test/providers/webgpu/plugin`, built into `onnxruntime_provider_test`.
The tests also cover initial graph capture and replay across independent Sessions, and concurrent
Run calls for different graph IDs within one Session, serialized by ORT. Initial-capture tests use
preallocated bindings; they do not cover external Session allocator calls overlapping capture.
Concurrent profiling, cross-device transfer, and arbitrary foreign stream overrides are not
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
