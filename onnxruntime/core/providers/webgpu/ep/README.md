## The `ep` folder

The current folder contains the implementation of EP ABI adapter for WebGPU.

### Design considerations

To ensure both static library and dynamic library builds work, we need to make as few changes to existing code as possible. A few design decisions are as below:

- No changes to static library build. It should still work as before.

- For dynamic library:

  - use and only use the EP ABI. (no support for `GetProvider`)

  - still depends on onnxruntime targets.

  - use a bridge to connect EP ABI and the internal classes

### Session streams

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
allocation. Plain `Alloc` and null-stream allocations submit clears before returning, including
during Run. This keeps CPU-produced outputs bound to GPU ordered with fallback uploads without
additional core stream creation. The policy depends on the allocation's stream, not `IsRunActive()`.
Uploads, readbacks, stream synchronization, dispatch batch limits, and Run/capture boundaries can
still submit work.

Multiple threads may use one Session allocator, including while that Session or other Sessions run,
provided they operate on independent tensors. A dedicated small Session can also allocate inputs
for concurrent inference Sessions on the same WebGPU device/context. Keep the allocator Session
and allocator alive until their tensors are released, and synchronize writes before consuming a
shared tensor. Plain allocations can flush pending Session work, so correctness isolation does
not guarantee freedom from contention.

Environment transfers with no stream use their private command state. GPU-to-GPU copies
without a stream submit and wait before returning. Stream notifications currently complete
producer work synchronously during activation; the wait callbacks consequently have no
remaining work. This conservative implementation prioritizes correctness over overlap.

The implementation requires an ORT build with stream support. CPU I/O, graph-internal CPU/GPU
copies, mixed feed copies, CPU outputs bound to GPU, concurrent Sessions, and same-Session and
dedicated-Session allocator concurrency are covered by AutoEP tests.
Concurrent graph capture, concurrent profiling, cross-device transfer, and arbitrary foreign
stream overrides are not established by these tests. Performance must be measured separately.

### Missing parts

This section describes what is missing.

- need a way to do WebGPU cleanup (`OrtEnv::~OrtEnv()` currently calls `webgpu::CleanupWebGpuContexts()` in static lib build)

- need a way to setup "default configurations" for WebGPU. (currently missing for both static lib and shared lib)
  - we want something like `SetCurrentGpuDeviceId` in ORT C-API, which set a global state and is directly available to user.
    - to make it general, it can be something like:
      ```c++
      ORT_API2_STATUS(SetEpDefaultConfig, _In_ const char* ep_name, _In_ const char* key, _In_ const char* value);
      ```
