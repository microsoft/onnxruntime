## The `ep` folder

The current folder contains the implementation of EP ABI adapter for WebGPU.

### Design considerations

To ensure both static library and dynamic library builds work, we need to make as few changes to existing code as possible. A few design decisions are as below:

- No changes to static library build. It should still work as before.

- For dynamic library:

  - use and only use the EP ABI. (no support for `GetProvider`)

  - still depends on onnxruntime targets.

  - use a bridge to connect EP ABI and the internal classes

### Session and environment isolation

With GPU profiling disabled, different Sessions can be created, initialized, and run concurrently,
including creation while another Session runs. Each Session owns a `CommandRecordingState` used by
its kernels and copies.
The device, buffer caches, and pipeline cache remain shared and synchronized. Released buffers
remain with their recording until its commands have been submitted.

Env and Session operations reuse `GpuBufferAllocator` and `DataTransferImpl`. Env allocators and
transfers share one context-owned recording, separate from every Session's recording. Cached-buffer
clears are submitted before Env allocation returns. Session allocations submit clears outside Run
and batch them during Run.
The plugin `CopyTensors` callback submits the associated recording and waits for copy completion,
including uploads and device copies, as required by the no-stream synchronous-copy contract.
Ordinary Run and graph replay submit commands without an additional completion wait;
`OrtEp::Sync` remains a no-op.
I/O Binding synchronization calls do not guarantee GPU completion; output downloads wait for readback.

There is no recording mutex. Applications must serialize same-Session allocator operations,
copies, tensor destruction, graph replay, and graph release with that Session's execution. ORT's
ordinary same-Session Run lock does not cover these external operations. All Env copies and allocator
operations on the same context, including Free/GetStats and tensor destruction, require caller
serialization. Explicit concurrent calls to Env or Session allocator APIs, concurrent Env copies,
and cross-device transfers are outside this PR's supported scope. Internal allocation and copies
during independent Session creation and Run are supported in both native and plugin builds.

Session transfers are created by the optional `OrtEp::CreateDataTransfer` callback and share the
EP's recording with kernel-internal transfers. Each copy uses the EP's current buffer manager,
including during graph capture. ORT owns and releases the ABI wrapper before releasing the EP.
Factory-created transfers remain independent and serve Env copies; no callback ordering or thread
affinity is assumed. Other plugins can leave the new callback null and retain the factory behavior.
This WebGPU plugin requires ORT 1.31.0 or newer, which supports the instance-level callback. The
minimum runtime version is enforced at library registration rather than silently using an unbound
Session transfer on an older runtime.

Dawn native advertises the software feature `ImplicitDeviceSynchronization` on all adapters, and ORT
requests it when creating a device. An externally supplied native device must also have enabled it
in `DeviceDescriptor.requiredFeatures`; otherwise context creation fails with an explicit error.
The pinned Dawn native implementation keeps an error-scope stack per calling thread
(`DeviceBase::GetErrorScopeStack`), so validation scopes in
different Session threads do not require a context-wide Run mutex. Each push/pop pair must stay on
the same thread.

GPU profiling state remains context-wide. Concurrent use of a shared context while profiling is
enabled is not supported by this change; applications must serialize those operations. Recording-local
profiling is deferred to a separate change.

### Missing parts

This section describes what is missing.

- need a way to do WebGPU cleanup (`OrtEnv::~OrtEnv()` currently calls `webgpu::CleanupWebGpuContexts()` in static lib build)

- need a way to setup "default configurations" for WebGPU. (currently missing for both static lib and shared lib)
  - we want something like `SetCurrentGpuDeviceId` in ORT C-API, which set a global state and is directly available to user.
    - to make it general, it can be something like:
      ```c++
      ORT_API2_STATUS(SetEpDefaultConfig, _In_ const char* ep_name, _In_ const char* key, _In_ const char* value);
      ```
