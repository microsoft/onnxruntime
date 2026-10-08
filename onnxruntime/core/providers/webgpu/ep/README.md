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

The plugin supports ORT 1.24.4 and later. Session-owned recording is enabled on 1.28.x starting
at 1.28.3, on 1.30.x starting at 1.30.1, and on 1.31 and later. Other supported versions,
including all 1.29.x hosts, use a serial compatibility path:
Session kernels, allocators, and framework/Env copies share a context-owned recording. This
preserves `clear -> upload -> compute -> readback` ordering even when the host drops the stream
on a single-tensor copy. Cached-buffer clearing remains enabled, and kernels still batch their
dispatches; there is no per-kernel submission and no traversal of other Sessions.

Legacy callers must serialize **all WebGPU operations on the same device**, including Session
creation/destruction, Run, I/O binding, allocator use, and Env transfers. Use sequential graph
execution. Multiple Sessions may be used sequentially; overlapping Runs are rejected. This is not
a restriction to one fixed CPU thread, but operations must not overlap or reenter from callbacks.
The plugin's same-Session Run concurrency flag alone cannot serialize separate Sessions or Env calls.

Legacy Session ordinary `Alloc` submits cached-buffer clears outside Run and defers them while
`IsRunActive()` is true. The same allocator supplies kernel temp space, avoiding per-scratch
submissions without `KernelContext_GetSyncStream`, which is unavailable on 1.24. Env ordinary
`Alloc` always submits. Under the required serial, non-reentrant calling contract, application
allocations occur outside Run and therefore submit before returning. This is submission, not a wait
for GPU completion. A matching `AllocOnStream` continues to defer clears. Run-end cleanup resets the
active flag on success and failure; a copy or the existing dispatch/Run boundary submits pending work.
An omitted allocation submission policy defaults to no immediate submission, as in built-in WebGPU.
Session and Env allocators set explicit policies; the default does not change their behavior.
Framework/Env copies and Env allocations use the default context buffer manager. Graph execution
keeps its per-graph buffer manager, while sharing the same legacy recording with those copies.
Capture/replay boundaries drain pending shared work. Run-end flushing refreshes the graph and
default managers before capture ends, even if a framework copy already submitted the recording.
BufferManager tracks deferred releases by recording. Destroying one legacy Session must not
discard the context-owned recording's pending entries, which may still belong to other Sessions
or Env allocations. Modern Session-owned recordings are discarded on Session teardown.
Run/replay guards release the legacy Run gate; Run cleanup also resets the Session's graph-manager selection.
They do not abandon a partially recorded replay; recovery after replay failure is not guaranteed.

The recording mode is selected once at plugin registration using the host's minor and patch
versions. The Session-owned recording path is described below. Set `ORT_WEBGPU_EP_FORCE_LEGACY=1`
**before loading the plugin** to exercise the serial path on a host that supports Session-owned
recording. The setting is process-wide and only forces the safe compatibility direction; it
cannot enable the modern path on an unsupported host.

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
and serial graph capture/replay with multiple graph IDs. A legacy-only test verifies that a second
Run is rejected while another Session is executing. Run serial tests with
`ORT_WEBGPU_EP_FORCE_LEGACY=1`; the concurrent-success tests apply only to modern mode.
The `onnxruntime_webgpu_legacy_test` CTest entry sets this environment variable before loading
the plugin and runs a legacy-safe allowlist in normal PR CI.
Public allocation tests verify dirty-buffer reuse and read the raw buffer with an independent
Dawn command encoder, so ORT's readback path cannot hide an unsubmitted clear. Submission counts
are also checked before that external readback, including after a cancelled Run.
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
