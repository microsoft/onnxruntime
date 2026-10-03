# Plugin EP shared allocators and data transfer

This document describes how plugin execution providers (EPs), ONNX Runtime (ORT), and applications cooperate to
allocate and copy device memory. It covers three related but separate facilities:

- allocator and data-transfer registration in an `OrtEnv`;
- application use of `GetSharedAllocator` and `CopyTensors`; and
- allocator selection inside an `OrtSession`.

The complete reference implementation is the
[`example_plugin_ep`](../../onnxruntime/test/autoep/library/example_plugin_ep/) test library. In particular,
[`ep_allocator.h`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_allocator.h),
[`ep_arena.h`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_arena.h), and
[`ep_arena.cc`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_arena.cc) are designed to be copied and
adapted by EP authors.

## Concepts and ownership

| Object | Created by | Owned by | Typical use |
|---|---|---|---|
| `OrtMemoryInfo` | EP factory or application | Creator | Identifies the device and memory represented by an allocator or tensor. |
| `OrtAllocator` returned by `OrtEpFactory::CreateAllocator` | EP factory | ORT until it calls `OrtEpFactory::ReleaseAllocator` | Allocates EP device or host-accessible memory. |
| Shared `OrtAllocator` returned by `OrtApi::GetSharedAllocator` | EP factory or application | `OrtEnv` releases EP-created allocators; the application retains ownership of a registered custom allocator; the returned pointer is unowned | Creates tensors for `CopyTensors` or other application-managed device data. |
| `OrtDataTransferImpl` | EP factory | EP-defined; ORT calls `Release` when its adapter is destroyed | Determines supported copy routes and performs copies. |
| `OrtSyncStream` | ORT around an EP-provided `OrtSyncStreamImpl` | Application or session, depending on who created it | Orders asynchronous copies and EP work. |

An EP may return a newly allocated object from every factory callback or return the same factory-owned object more
than once. If it returns the same allocator for environment and session requests, it must make the allocator
thread-safe and account for every matching `ReleaseAllocator` call.

An allocator's `OrtMemoryInfo` must remain valid for the allocator's lifetime. An `OrtMemoryInfo` passed to
`EpDevice_AddAllocatorInfo` is stored by pointer, so it must remain valid for the lifetime of the `OrtEpDevice`; it is
also the pointer ORT later passes to `CreateAllocator`. An application-provided allocator and
all memory allocated from it must also remain valid until the allocator has been unregistered and all users have
released their memory.

## Environment registration

Registering an EP library does more than make its factories discoverable. ORT also prepares the allocators and data
transfers that can be used without creating a session.

```text
RegisterExecutionProviderLibrary
  -> load library and obtain OrtEpFactory instances
  -> call GetSupportedDevices on each factory
     -> EP creates OrtEpDevice instances
     -> EP calls EpDevice_AddAllocatorInfo for each supported memory kind
  -> for each OrtEpDevice
     -> create a default shared allocator for DEFAULT memory, if provided
     -> create a default shared allocator for HOST_ACCESSIBLE memory, if provided
  -> call CreateDataTransfer on each factory
     -> register the returned transfer in the Environment DataTransferManager
```

### Publishing allocator information

In `OrtEpFactory::GetSupportedDevices`, the EP creates each `OrtEpDevice` and adds the allocator information it
supports:

```cpp
RETURN_IF_ERROR(ep_api.EpDevice_AddAllocatorInfo(ep_device, default_memory_info));
RETURN_IF_ERROR(ep_api.EpDevice_AddAllocatorInfo(ep_device, host_accessible_memory_info));  // optional
RETURN_IF_ERROR(ep_api.EpDevice_AddAllocatorInfo(ep_device, readonly_memory_info));         // optional
```

The valid combinations are:

- `OrtDeviceAllocator` with `OrtDeviceMemoryType_DEFAULT`;
- `OrtDeviceAllocator` with `OrtDeviceMemoryType_HOST_ACCESSIBLE`; and
- `OrtReadOnlyAllocator` with `OrtDeviceMemoryType_DEFAULT`, for initializers.

The default and host-accessible entries cause ORT to call `OrtEpFactory::CreateAllocator` during EP-library
registration, so the factory must implement it if either entry is added. The read-only entry is used when a session
creates its initializer allocator. If it is absent, ORT uses the default device allocator for initializers.

`CreateAllocator` may return a null allocator to indicate that the EP uses ORT's default CPU allocator. In that case
no shared allocator is registered for that memory info.

The initial environment-level `CreateAllocator` call has `allocator_options == nullptr`. An application can later
call `CreateSharedAllocator` with options to replace that default shared allocator.

### Automatic allocator creation and replacement

During EP registration, ORT creates an allocator only when a compatible shared allocator does not already exist.
This preserves a custom allocator that the application registered before loading the EP library.

Applications do not need to call `OrtApi::CreateSharedAllocator` for normal shared-allocator use. Call it only to
replace the automatically created allocator with different configuration options, such as enabling or configuring an
EP-owned arena. `allocator_options` is passed through to the EP factory. Destroy tensors and stop all work that uses
the old shared allocator before replacing it; replacement releases ORT's reference to the old allocator.

`OrtApi::GetSharedAllocator` returns an unowned pointer. The application must not call the allocator's release
callback or use it after the allocator, EP library, or environment has been released. A default CPU allocator is
returned for default CPU memory when no custom CPU allocator is registered.

EP-created shared allocators are released automatically when the EP library is unregistered or the environment is
released. `OrtApi::ReleaseSharedAllocator` is optional and only needed to free an allocator, and any memory it holds,
earlier.

### Application-provided custom allocators

An application can call `OrtApi::RegisterAllocator` or `Ort::Env::RegisterAllocator` to add its own `OrtAllocator` to
the environment. The allocator then participates in both shared-allocator lookup and the optional session allocator
path described below.

The required ordering is:

1. Create the `OrtEnv`.
2. Register the custom allocator.
3. Register the EP library.
4. Create sessions or use `GetSharedAllocator` and `CopyTensors`.
5. Destroy tensors and sessions that use the allocator.
6. Unregister the allocator.

The custom allocator must be registered before the EP library. EP registration then skips creating its own shared
allocator for that memory. Registering a custom allocator for memory that already has an EP shared allocator is not
supported. The allocator's memory
information must describe memory that the EP's data-transfer and execution implementations can actually use.

### Data-transfer registration

ORT calls `OrtEpFactory::CreateDataTransfer` once while registering the factory with the environment. The callback is
optional and may also return `nullptr` if the EP does not need a custom transfer. Otherwise, the returned
`OrtDataTransferImpl` is wrapped and registered in the environment's `DataTransferManager`.

Data-transfer capability is factory-level, not session-specific. ORT nevertheless registers it in two places:

- the environment's `DataTransferManager`, used by the public `CopyTensors` API; and
- each session's `DataTransferManager`, used for execution-time copies.

Consequently, ORT calls the factory callback during EP-library registration and again when the plugin EP is added to
a session. These managers have separate adapters, but the factory may return the same `OrtDataTransferImpl` pointer.
The example EP does this and makes `OrtDataTransferImpl::Release` a no-op. An EP that creates a transfer for each
callback should delete it in `Release`.

When a transfer is requested, ORT uses the first registered implementation whose `CanCopy` callback accepts the
source and destination `OrtMemoryDevice` pair. `CanCopy` should therefore be precise about device type, memory type,
vendor, and device ID. Do not claim a route that the implementation cannot copy correctly.

## Using shared allocators and `CopyTensors`

Shared allocators let an application allocate memory in an `OrtEnv` without first creating a session. The most common
use is to prepare device-resident inputs or destinations for `CopyTensors`.

### Synchronous copy

The following C++ API example copies one CPU tensor to an EP device. Error handling around EP discovery is omitted.

```cpp
Ort::ConstMemoryInfo device_memory_info = ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
Ort::UnownedAllocator device_allocator = env.GetSharedAllocator(device_memory_info);
if (!device_allocator) {
  throw std::runtime_error("No shared allocator for the EP device");
}

std::array<int64_t, 2> shape{2, 4};
Ort::Value device_tensor =
    Ort::Value::CreateTensor<float>(device_allocator, shape.data(), shape.size());

Ort::ThrowOnError(env.CopyTensor(cpu_tensor, device_tensor, nullptr));
```

With a null stream, the transfer implementation must complete the copy before returning. This is the simplest option
when the caller immediately reads the destination on the host or cannot arrange stream ordering.

### Asynchronous copy followed by `Run`

An EP that supports streams can perform the copy asynchronously. Use the same `OrtSyncStream` for the copy and the
session run so that the run is ordered after the copy without blocking the host:

```cpp
Ort::SyncStream stream = ep_device.CreateSyncStream();

Ort::ThrowOnError(env.CopyTensor(cpu_tensor, device_tensor, stream));

Ort::RunOptions run_options;
run_options.SetSyncStream(stream);
session.Run(run_options,
            input_names.data(), &device_tensor, 1,
            output_names.data(), &output, 1);
```

`RunOptionsSetSyncStream` requires the stream to remain alive for the duration of `Run`. The source and destination
tensors, including any application-owned backing storage, must remain alive until all asynchronous work that uses them
has completed.

The stream must be compatible with the device and transfer implementation. An EP typically obtains its native handle
inside `CopyTensors` with `OrtApi::SyncStream_GetHandle` and passes it to an API such as `cudaMemcpyAsync`.

### `CopyTensors` requirements

For one call to `CopyTensors`:

- `num_tensors` must be greater than zero and the source and destination arrays must have that many entries;
- every entry must be a non-null, allocated tensor `OrtValue`;
- all source tensors must have equal `OrtMemoryInfo` values;
- all destination tensors must have equal `OrtMemoryInfo` values;
- a registered transfer must accept the source and destination memory devices; and
- each destination must already have the correct type, shape, and capacity for its source.

`OrtMemoryInfo` equality compares the name, allocator type, memory type, and device. Tensors on the same device that
were created with differently named memory info cannot be combined in one call.

`CopyTensors` does not allocate destinations. The public entry point validates the tensor and memory-location
requirements, but it does not establish that each destination buffer is large enough. The EP transfer must copy the
correct number of bytes and return an `OrtStatus` on failure. If no transfer accepts the route, the public API returns
`ORT_NOT_IMPLEMENTED`.

The optional stream is forwarded with every source/destination pair. ORT always passes a `streams` array with one entry
per pair, and each entry may be null. If an `OrtDataTransferImpl` queues work asynchronously, it must use the supplied
stream and must not report completion on an unrelated stream.

## Shared allocators and sessions

Environment-level allocators are not used by a session by default. A plugin session normally asks the EP to create
its preferred allocators for default, host-accessible, and read-only memory:

1. ORT prefers `OrtEp::CreateAllocator` when the EP instance implements it.
2. Otherwise ORT calls `OrtEpFactory::CreateAllocator`.
3. ORT owns each returned reference and eventually calls `OrtEpFactory::ReleaseAllocator`, including for allocators
   created by `OrtEp::CreateAllocator`, so the factory must be able to release allocators that an `OrtEp` created.

This default gives the EP a place to apply session-specific settings and isolation. When the `OrtEp` allocator callback
is not implemented, the factory path receives the session configuration entries prefixed with `ep.<ep_name>.arena.`
(the EP name is lowercased) as `allocator_options`, with the `ep.<ep_name>.` prefix removed.

To make a session use allocators registered in its environment, set:

```cpp
Ort::SessionOptions session_options;
session_options.AddConfigEntry("session.use_env_allocators", "1");
```

During session initialization, ORT overlays the environment allocators onto the session allocator maps by device.
`OrtReadOnlyAllocator` entries go to the initializer map; other entries go to the general allocator map. All sessions
using one environment allocator may allocate concurrently, so the allocator must be thread-safe.

An EP may reject `session.use_env_allocators=1` for features that require session-owned allocators. For example, the
CUDA plugin EP does not allow it together with CUDA graph capture.

The following choices are all valid EP designs:

| Design | Consequence |
|---|---|
| Return a new allocator for every request | Environment and sessions have isolated pools and configuration. |
| Return the same factory allocator for every request | Shared and per-session paths use one pool even without `session.use_env_allocators`; the EP must reference-count releases. |
| Implement `OrtEp::CreateAllocator` for sessions and `OrtEpFactory::CreateAllocator` for shared use | Sessions can use session-specific state while the environment still supports `CopyTensors`. |
| Ask users to set `session.use_env_allocators=1` | Sessions explicitly share environment allocators, including application-registered custom allocators. |

Do not assume that an allocator obtained from `GetSharedAllocator` is the allocator a session is using. Set the option
or define the EP's allocator policy accordingly.

## Providing an arena in a plugin EP

An arena provided by a plugin EP lives inside the plugin library. ORT sees an ordinary `OrtAllocator` whose
`OrtMemoryInfo::alloc_type` is `OrtDeviceAllocator`. `OrtArenaAllocator` is reserved for ORT's internal arena and is
rejected when returned by a plugin factory.

An EP-owned arena can implement:

- `Alloc`, `Free`, and `Info`, which are required;
- `Reserve`, to keep long-lived initializer allocations from changing arena growth behavior;
- `GetStats`, for allocator diagnostics;
- `AllocOnStream`, for stream-aware chunk reuse; and
- `Shrink`, if the arena supports releasing unused regions and should participate in session arena shrinking.

`Shrink` is optional. If `version >= 25` and `Shrink` is non-null, ORT wraps the allocator as an internal `IArena` so
session-level shrink requests can reach it. The arena remains implemented and owned by the plugin.

### Example plugin EP reference implementation

The example plugin EP contains the complete arena implementation and factory integration and can be used as a reference.

| File | What to use it for |
|---|---|
| [`ep_allocator.h`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_allocator.h) | Raw allocator wrapper, allocator callback setup, lifetime, and statistics. |
| [`ep_arena.h`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_arena.h) | Arena configuration, the `OrtAllocator` arena wrapper, and stream-aware allocator interfaces. |
| [`ep_arena.cc`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_arena.cc) | Arena allocation, chunk management, coalescing, statistics, and stream bookkeeping. |
| [`ep_factory.h`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_factory.h) | Factory-owned arena state, synchronization, and callback declarations. |
| [`ep_factory.cc`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_factory.cc) | Allocator memory-info publication, `CreateAllocator`, `ReleaseAllocator`, arena options, and shared-instance lifetime management. |
| [`ep_stream_support.cc`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_stream_support.cc) | Releasing arena stream assignments when a run ends or a stream is destroyed. |
| [`test_allocators.cc`](../../onnxruntime/test/autoep/test_allocators.cc) | Expected shared-allocator behavior, arena configuration, replacement, and custom allocator registration. |

### Stream-aware arena integration

The example arena's `AllocOnStream(size, stream)` associates chunks with the framework `OrtSyncStream*`. The stream is
bookkeeping for safe chunk reuse; it does not choose the native stream on which a kernel executes.

When a session run ends, the example `OrtSyncStreamImpl::OnSessionRunEnd` calls
`ArenaAllocator::ResetChunksUsingStream`. This removes the stream assignment from chunks after ORT knows that run is
finished. Cross-stream reuse before run end is allowed only when the stream synchronization IDs show that the consumer
has waited on the producer.

An EP that implements stream-aware allocation must therefore keep these pieces consistent:

- return `true` from `OrtEpFactory::IsStreamAware`;
- implement `CreateSyncStreamForDevice`;
- pass the framework `OrtSyncStream*`, not only its native handle, to `AllocOnStream`;
- implement the stream notification/wait operations used by ORT; and
- release per-stream arena bookkeeping from `OnSessionRunEnd` and stream destruction.

If the allocator is not safe for overlapping work on different streams, leave `AllocOnStream` null or ensure the EP
does not advertise unsupported concurrent execution.

## EP data-transfer implementation checklist

The example implementation in
[`ep_data_transfer.h`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_data_transfer.h) and
[`ep_data_transfer.cc`](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_data_transfer.cc) can also be copied.
When adapting it to real device memory:

1. Set `ort_version_supported`, `CanCopy`, `CopyTensors`, and `Release`.
2. Make `CanCopy` accept only routes implemented by the EP.
3. In `CopyTensors`, query each value's memory device and data pointer through `OrtEpApi` and `OrtApi`.
4. Validate any EP-specific restrictions and determine the copy direction.
5. Use `streams[i]` for an asynchronous copy when present; otherwise complete synchronously.
6. Return an `OrtStatus` for native API failures.
7. Define whether each callback creates a transfer or returns a shared factory instance, and implement `Release`
   accordingly.

Host-accessible memory deserves special care. It can be addressable by the CPU while still participating in pending
device work. Synchronize or establish the required stream dependency before a CPU read or write.

## Source map

- Environment registration and allocator ownership:
  [`environment.cc`](../../onnxruntime/core/session/environment.cc)
- Public `CopyTensors` validation and dispatch:
  [`onnxruntime_c_api.cc`](../../onnxruntime/core/session/onnxruntime_c_api.cc)
- Plugin transfer adapter:
  [`plugin_data_transfer.cc`](../../onnxruntime/core/framework/plugin_data_transfer.cc)
- Session allocator opt-in:
  [`inference_session.cc`](../../onnxruntime/core/session/inference_session.cc)
- Plugin per-session allocator creation:
  [`ep_plugin_provider_interfaces.cc`](../../onnxruntime/core/session/plugin_ep/ep_plugin_provider_interfaces.cc)
- Public C API contracts:
  [`onnxruntime_c_api.h`](../../include/onnxruntime/core/session/onnxruntime_c_api.h) and
  [`onnxruntime_ep_c_api.h`](../../include/onnxruntime/core/session/onnxruntime_ep_c_api.h)
- Focused tests:
  [`test_allocators.cc`](../../onnxruntime/test/autoep/test_allocators.cc),
  [`test_data_transfer.cc`](../../onnxruntime/test/autoep/test_data_transfer.cc), and
  [`test_data_copy.cc`](../../onnxruntime/test/shared_lib/test_data_copy.cc)
