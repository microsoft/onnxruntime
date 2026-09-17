## The `ep` folder

The current folder contains the implementation of EP ABI adapter for WebGPU.

### Design considerations

To ensure both static library and dynamic library builds work, we need to make as few changes to existing code as possible. A few design decisions are as below:

- Static and dynamic builds share the provider's buffer initialization and recycling policy.

- For dynamic library:

  - use and only use the EP ABI. (no support for `GetProvider`)

  - still depends on onnxruntime targets.

  - use a bridge to connect EP ABI and the internal classes

### Session streams

The WebGPU plugin implements `OrtEp::CreateSyncStreamForDevice`. Each stream references its
owning EP's command state; kernels and stream-bearing data transfers use that same state.
Graph Memcpy kernels access their owning EP directly. On ORT cores without single-copy stream
forwarding, the generic single-tensor transfer wrapper drops explicit streams, so those copies
use the fallback recording. The core fix is tracked separately in
[#32666](https://github.com/microsoft/onnxruntime/pull/32666) /
[#32643](https://github.com/microsoft/onnxruntime/issues/32643).

ORT requests Dawn's `ImplicitDeviceSynchronization` feature for internally created native devices
and checks that it is enabled. An externally supplied native device without the feature is accepted
with a warning once during context initialization. ORT cannot enable the feature on an existing
device, does not replace that device, and does not guarantee thread-safe access without it.
The caller must serialize all Session and external WebGPU access to that device, or create a device
with `ImplicitDeviceSynchronization` in `DeviceDescriptor.requiredFeatures` before using it concurrently.
The concurrency support described below assumes this feature is enabled on native devices.
This Dawn native feature does not apply to WASM.

Session allocators expose the existing `OrtAllocator::AllocOnStream` callback and validate
that the stream belongs to the same Session. For ordinary storage caches (`bucket` and
`simple`), all frees go through the existing recording's pending-buffer list. Flush appends storage
clears after deferred computation, submits once, and then returns the buffers to the existing cache.
Allocation from that cache needs no additional clear. On a cache miss, a recording with only
pending frees can be flushed to reclaim them; a live compute batch is not flushed just to allocate.
This also handles frees outside Run and Flush calls that initially have no command encoder.

Capture caches retain allocation-time clearing and immediate submission: captured commands may
still depend on buffers after their CPU Tensor is released. Uniform and other non-storage caches
are unchanged. Explicit kernel `FillZero` stays at its original recording position without forcing
a separate submission. These initialization rules are shared by native and plugin builds;
allocators do not choose a separate clear or submission policy.

Submitted initialization precedes fallback uploads, but does not order later deferred computation
before a fallback output readback. Mixed GPU-to-CPU and CPU-to-GPU output-copy batches remain a
known correctness limitation without core stream forwarding, even for serialized Runs: they
can return incorrect results rather than being safely rejected.
Uploads, readbacks, stream synchronization, dispatch batch limits, and Run/capture boundaries can
still submit work.

Multiple threads may use one Session allocator, including while that Session or other Sessions run,
provided they operate on independent tensors. A dedicated small Session can also allocate inputs
for concurrent inference Sessions on the same WebGPU device/context. Keep the allocator Session
and allocator alive until their tensors are released, and synchronize writes before consuming a
shared tensor. Ordinary-cache misses without live computation can submit pending clears; capture-cache
allocations can still flush pending Session work. Neither guarantees freedom from contention.

Environment transfers with no stream use their private command state. GPU-to-GPU copies
without a stream submit and wait before returning. Stream notifications currently complete
producer work synchronously during activation; the wait callbacks consequently have no
remaining work. This conservative implementation prioritizes correctness over overlap.

The implementation requires an ORT build with stream support. CPU I/O, graph-internal CPU/GPU
copies, mixed feed copies, CPU-only graph outputs bound to GPU, concurrent Sessions, and
same-Session and dedicated-Session allocator concurrency have AutoEP test coverage, including
single-buffer and batch handle reuse without Run. Mixed-direction output-copy cases affected
by the missing core forwarding are excluded from this coverage, not fixed by their exclusion.
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
