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
Graph Memcpy kernels access their owning EP directly, avoiding the generic single-tensor
transfer wrapper that currently drops the stream argument.

Session allocators expose the existing `OrtAllocator::AllocOnStream` callback. Allocations
with a matching Session stream defer cached-buffer clears; plain `Alloc` submits those clears
before returning. No thread-local Session lookup is needed. External allocations can still
flush pending Session work, so this is correctness isolation, not a no-contention guarantee.

Environment transfers with no stream use their private command state. GPU-to-GPU copies
without a stream submit and wait before returning. Stream notifications currently complete
producer work synchronously during activation; the wait callbacks consequently have no
remaining work. This conservative implementation prioritizes correctness over overlap.

The implementation requires an ORT build with stream support. CPU I/O, graph-internal CPU/GPU
copies, concurrent Sessions, and same-Session allocator concurrency are covered by AutoEP tests.
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
