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

Different Sessions can be created, initialized, and run concurrently, including creation while
another Session runs. Each Session owns a `CommandRecordingState` used by its kernels and copies.
The device, buffer caches, and pipeline cache remain shared and synchronized. Released buffers
remain with their recording until its commands have been submitted.

The plugin binds Session transfers during creation without new EP APIs or stream routing. This
relies on the current ORT calling sequence, not an API guarantee: the first successful factory
`CreateDataTransfer` call is for the Env; later calls run on the thread that created the awaiting
EP. There may be only one unbound EP per factory/thread. Binding removes the pending entry, and
the Session can subsequently run on another thread. The factory lock does not cover creation or Run.

Env and Session operations reuse `GpuBufferAllocator` and `DataTransferImpl`, but Env allocation
owns separate recording and Env copies use local recording. Cached-buffer clears are submitted
before Env allocation returns. Session allocations submit clears outside Run and batch them during Run.
`CopyTensors` submits pending commands before returning; downloads wait for readback, but uploads and
device copies do not wait for GPU completion. The existing gap against the no-stream synchronous-copy
contract is left as a TODO for separate work. Ordinary Run and graph replay submit commands without
an additional completion wait; `OrtEp::Sync` remains a no-op.
I/O Binding synchronization calls do not guarantee GPU completion; output downloads wait for readback.

There is no recording mutex. Applications must serialize same-Session allocator operations,
copies, tensor destruction, graph replay, and graph release with that Session's execution. ORT's
ordinary same-Session Run lock does not cover these external operations. Shared Env allocator
operations, including Free/GetStats and tensor destruction, also require caller serialization.
Concurrent profiling and cross-device transfers are outside the supported scope.

`PluginEpWebGpuConcurrency` covers concurrent creation/Run, CPU/GPU I/O, and independent Session
capture/replay. `CpuPartitionBetweenGpuKernels` verifies both the CPU intermediate and final GPU
result across a forced GPU/CPU/GPU partition. The 12-thread shared-allocator/Run and 16-thread
same-Session allocator/Run tests remain disabled next-phase targets, along with the other tests
requiring concurrent access to a shared allocator or recording. Their interleavings are retained
and must not be enabled until the corresponding support is implemented.

### Missing parts

This section describes what is missing.

- need a way to do WebGPU cleanup (`OrtEnv::~OrtEnv()` currently calls `webgpu::CleanupWebGpuContexts()` in static lib build)

- need a way to setup "default configurations" for WebGPU. (currently missing for both static lib and shared lib)
  - we want something like `SetCurrentGpuDeviceId` in ORT C-API, which set a global state and is directly available to user.
    - to make it general, it can be something like:
      ```c++
      ORT_API2_STATUS(SetEpDefaultConfig, _In_ const char* ep_name, _In_ const char* key, _In_ const char* value);
      ```
