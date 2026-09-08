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
kernels and its data transfer use that same state without stream-based routing or a recording mutex.
Access to one Session's recording must be serialized by the caller. Ordinary same-Session Run
calls remain serialized by ORT, but that does not cover external allocation, copies, or graph replay.
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

Environment and Session operations reuse `GpuBufferAllocator`, `DataTransferImpl`, and the
context-level `BufferManager`. The Env allocator retains the context and its own recording;
cached-buffer clears are always submitted before Alloc returns. Env copies use local recording
for each CopyTensors call, while Session copies use the owning EP's recording. Both paths
submit and wait before returning from CopyTensors. There is no separate external allocator or
local-encoder copy implementation. Sharing implementation and caches does not share Session
encoders or graph capture state.

Env APIs remain available for sequential use. Applications must serialize operations on the
shared Env allocator, including Alloc, Free, GetStats, and tensor destruction. Env concurrency
is outside this revision's validation scope; no Env recording mutex is introduced.

Session allocators still reuse cached buffers. `IsRunActive()` controls clear submission:
outside Run, cached-buffer clears are submitted before Alloc returns; during Run, clears are
deferred in the Session recording to preserve batching. This is valid only when external
Session allocator operations do not overlap Run. Applications must serialize allocation, Free,
GetStats, tensor destruction, Session-bound copies, graph capture/replay, and graph release for
the same Session with its execution. Sharing a recording or Session allocator across threads
without this serialization is unsupported. Tensor data and lifetimes also remain the caller's
responsibility. The concurrency target is creating/initializing Sessions while other Sessions
run, along with independent Session creation and execution. Env operations are not part of
the concurrency acceptance gate.

The plugin does not register synchronization streams or support stream overrides. Session
data transfers are synchronous and use their owning Session recording. Ordinary Run and graph
replay still submit pending commands, but do not add a queue-completion wait at run end.
`OrtEp::Sync` is not implemented, restoring the earlier WebGPU behavior: I/O Binding's
`SynchronizeInputs` and `SynchronizeOutputs` use the default no-op Sync and do not guarantee
GPU completion. Subsequent work on the same queue is ordered by submission; an explicit
output download waits for CPU-readable results. No ORT stream support is needed for this routing.

The current validation is limited to `DifferentSessionsCreateAndRunConcurrently`. Other tests
are retained for future validation; results obtained before implementation sharing do not
establish their behavior for this revision. Concurrent profiling, same-Session allocator/Run
interleaving, and cross-device transfers remain unsupported. Performance must be measured separately.

### Concurrency test gates

The current gate uses four creation workers and four existing-Session Run workers. A barrier
aligns each of twenty iterations. Creation workers create, initialize, verify, and destroy
their own Session; Run workers execute independent pre-created Sessions. All runs use CPU
inputs and outputs with CPU fallback disabled and verify the model's results. No worker uses
an external allocator or Env copy concurrently with another worker.

| Gate | Test in `PluginEpWebGpuConcurrency` | Concurrent operations |
| --- | --- | --- |
| Current gate, 8 threads | `DifferentSessionsCreateAndRunConcurrently` | New Session creation/initialization and first Run alongside existing Session inference |
| Future Env concurrency target, 12 threads | `DISABLED_MixedSessionAndEnvironmentOperationsConcurrently12Threads` | Session creation/destruction, existing Session inference, environment allocation/copies |
| Advanced target, 16 threads | `DISABLED_MixedSessionAndAllocatorOperationsConcurrently16Threads` | The baseline plus allocator operations on the same Sessions that are running inference |

The 12/16-thread tests are retained as disabled future targets, not removed or serialized.
The next phase targets shared Env allocator operations concurrent with Session Run (12 threads),
then adds same-Session allocator/Run concurrency (16 threads). Enable each gate only when its
corresponding concurrency support is implemented and validated.
Their concurrent shared-allocator operations can race and must not be used to claim current
support. Earlier results do not apply to this implementation. Enable the advanced target only
when developing that future support, with Google Test's `--gtest_also_run_disabled_tests` and
`--gtest_filter=PluginEpWebGpuConcurrency.DISABLED_MixedSessionAndAllocatorOperationsConcurrently16Threads`.
A passing baseline does not establish support for the advanced target.

The built-in `WebGpuConcurrentContextTest.DISABLED_SessionAllocatorAndRunConcurrently` and
`WebGpuConcurrentContextTest.DISABLED_SharedDataTransferMultiThreadCopy` likewise retain
unsupported same-recording interleavings as disabled targets. Concurrent shared Env allocator
tests are also disabled, including `DISABLED_SharedAllocatorMultiThreadCreateTensor`,
`DISABLED_SharedAllocatorCreatesAndCopiesConcurrently`, and
`DISABLED_DifferentSessionsAndEnvironmentCopiesRunConcurrently`. Independent Session and
independent recording tests remain enabled, but are not all rerun for this revision.

`DifferentSessionsGraphCaptureAndReplayConcurrently` enables graph capture for four Sessions
using `mul_1.onnx`, each with independent, fixed-address GPU inputs and outputs and the default
graph ID. A per-iteration barrier aligns their first capture runs and subsequent replay runs.
Each worker updates its input and verifies its output for twenty runs. Per-Session logging
callbacks require at least nineteen entries into ORT's graph replay fast path, so numerical
correctness alone cannot hide a fallback to ordinary execution. This test runs by default and
does not enable profiling or overlap external allocator operations with the same Session's Run.
It is retained but is not included in this revision's create/Run-only validation.

### Missing parts

This section describes what is missing.

- need a way to do WebGPU cleanup (`OrtEnv::~OrtEnv()` currently calls `webgpu::CleanupWebGpuContexts()` in static lib build)

- need a way to setup "default configurations" for WebGPU. (currently missing for both static lib and shared lib)
  - we want something like `SetCurrentGpuDeviceId` in ORT C-API, which set a global state and is directly available to user.
    - to make it general, it can be something like:
      ```c++
      ORT_API2_STATUS(SetEpDefaultConfig, _In_ const char* ep_name, _In_ const char* key, _In_ const char* value);
      ```
