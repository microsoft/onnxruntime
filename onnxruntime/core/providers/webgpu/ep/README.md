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

Different Sessions may run concurrently. Each Session owns its command recording state;
kernels and its data transfer use that same state without stream-based routing.
The context-level buffer and pipeline caches remain shared and synchronized. Buffers used by
unsubmitted commands remain associated with their recording until submission.

Sessions may be constructed concurrently, with one WebGPU EP per Session. This implementation
relies on the current ORT creation order and calling thread, not a public API guarantee:
the first successful `OrtEpFactory::CreateDataTransfer` call creates the environment transfer,
and subsequent calls bind to the EP awaiting its Session transfer on the current thread.
Pending EPs are tracked per factory and creation thread. Each thread may have at most one
unbound EP for that factory. The two creation callbacks must run on the same thread; once
bound, the transfer directly references its EP and the Session may run on other threads.
Entries are removed on successful binding or EP release, even if release uses another thread.
The factory lock only protects bookkeeping, not EP construction or Session execution.
Revisit these assumptions if ORT's factory calling sequence changes.

Environment allocators create and release Dawn buffers directly, without using the shared
buffer cache or Session recording state. Each environment `CopyTensors` call uses local
recording state and completes its copies before returning. Environment allocation and copies
on independent tensors may run concurrently with Session inference.

Session allocators still reuse cached buffers. The plugin's Session device allocator submits
cached-buffer clears before allocation returns, including during `Run`. This orders the clears
before subsequent environment copies, but also flushes pending Session commands and reduces
batching. This immediate-submit policy is experimental; performance has not been measured and
graph capture coverage is limited to the scenario below. Applications should continue to serialize external
Session allocator operations (`Alloc`, `Free`, and tensor destruction) with that Session's `Run`
until the advanced contract is established. Tensor data and lifetimes must be synchronized by
the caller regardless of allocator locking.

The plugin does not register synchronization streams or support stream overrides. Session
data transfers are synchronous; `OrtEp::Sync` and the default run-end synchronization submit
and wait for Session work. No ORT stream support is needed for this routing.

AutoEP tests cover CPU I/O, graph-internal CPU/GPU copies, serial and concurrent Session creation,
execution on other threads after creation, concurrent Sessions with environment copies, and
Run-external cached-buffer clearing. Graph capture/replay is covered for independent Sessions
with fixed GPU I/O. Concurrent profiling, graph capture combined with same-Session allocator/Run
interleaving, and cross-device transfers remain outside the tested contract.
Performance must be measured separately.

### Concurrency test gates

Both mixed-load tests use four threads per operation group, a common start barrier, and
ten iterations per worker. Run and copy workers verify the returned tensor data.

| Gate | Test in `PluginEpWebGpuConcurrency` | Concurrent operations |
| --- | --- | --- |
| Required baseline, 12 threads | `MixedSessionAndEnvironmentOperationsConcurrently12Threads` | Session creation/destruction, existing Session inference, environment allocation/copies |
| Advanced target, 16 threads | `DISABLED_MixedSessionAndAllocatorOperationsConcurrently16Threads` | The baseline plus allocator operations on the same Sessions that are running inference |

The 12-thread gate runs by default. The 16-thread test retains the experimental allocator/Run
interleaving as an advanced acceptance target and is disabled by default, not removed or
serialized. Enable it explicitly with Google Test's `--gtest_also_run_disabled_tests` and
`--gtest_filter=PluginEpWebGpuConcurrency.DISABLED_MixedSessionAndAllocatorOperationsConcurrently16Threads`.
A passing baseline does not establish support for the advanced target.

`DifferentSessionsGraphCaptureAndReplayConcurrently` enables graph capture for four Sessions
using `mul_1.onnx`, each with independent, fixed-address GPU inputs and outputs and the default
graph ID. A per-iteration barrier aligns their first capture runs and subsequent replay runs.
Each worker updates its input and verifies its output for twenty runs. Per-Session logging
callbacks require at least nineteen entries into ORT's graph replay fast path, so numerical
correctness alone cannot hide a fallback to ordinary execution. This test runs by default and
does not enable profiling or overlap external allocator operations with the same Session's Run.

### Missing parts

This section describes what is missing.

- need a way to do WebGPU cleanup (`OrtEnv::~OrtEnv()` currently calls `webgpu::CleanupWebGpuContexts()` in static lib build)

- need a way to setup "default configurations" for WebGPU. (currently missing for both static lib and shared lib)
  - we want something like `SetCurrentGpuDeviceId` in ORT C-API, which set a global state and is directly available to user.
    - to make it general, it can be something like:
      ```c++
      ORT_API2_STATUS(SetEpDefaultConfig, _In_ const char* ep_name, _In_ const char* key, _In_ const char* value);
      ```
