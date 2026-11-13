# Accelerate model loading

ONNX Runtime provides two independent loading optimizations for models with large weights:

| Target | Mechanism | Configuration | Default |
|---|---|---|---|
| CPU prepacking | Run eligible CPU kernel `PrePack()` calls concurrently | `session.prepack.enable_parallel` | Disabled (`"0"`) |
| CUDA external initializers | Read external data directly into CUDA memory with GPUDirect Storage (Linux) or Microsoft DirectStorage (Windows) | CUDA EP options `external_data_loader_use_gds` / `external_data_loader_use_directstorage` | Disabled (`"0"`) |
| CUDA external initializers | Read external data through reusable pinned buffers while copying to the GPU | `session.external_data_loader.threads` / CUDA EP `external_data_loader_reading_threads` | Auto from CPU cores (`2`..`8`) |
| CUDA external initializers | Total pinned host staging memory across the two reusable buffers | `session.external_data_loader.pinned_memory_bytes` / CUDA EP `external_data_loader_pinned_memory_bytes` | 32 MiB (`"33554432"`) |
| CUDA external initializers | Maximum per-transfer chunk size for parallel reads and async H2D copies | `session.external_data_loader.chunk_bytes` / CUDA EP `external_data_loader_chunk_bytes` | 2 MiB (`"2097152"`) |
| CUDA external initializers | Minimum total external initializer bytes on the device before allocating pinned buffers/streams | `session.external_data_loader.min_size_bytes` | 16 MiB (`"16777216"`) |

The `session.*` options are session configuration entries. The CUDA external-data loader options can also be passed as
execution provider options when the CUDA EP (built-in or plugin) is appended to `SessionOptions`. GPUDirect Storage and
Microsoft DirectStorage are execution provider options of the built-in CUDA EP only; there is no session configuration
entry for them and the CUDA plugin EP does not support them.

## Parallel CPU weight prepacking

Some CPU kernels transform constant weights into a layout that is faster to use during inference. This prepacking
happens once during session initialization and can dominate loading time for large INT4 models with many
`MatMulNBits` nodes.

When `session.prepack.enable_parallel` is set to `"1"`, ONNX Runtime dispatches eligible CPU `PrePack()` calls across
the session intra-op thread pool. This overlaps both the CPU work and page-ins for memory-mapped external weights.
`MatMulNBits` avoids nested parallelism while this outer parallelism is active.

The optimization applies when:

- more than one eligible CPU node needs prepacking;
- the intra-op thread pool has more than one thread; and
- cross-session prepacked-weight sharing is not in use. Sessions created with a shared prepacked-weights container
  retain the serialized path needed to protect that shared cache.

Enable the optimization and size the same intra-op thread pool in Python:

```python
import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 8
session_options.add_session_config_entry("session.prepack.enable_parallel", "1")

session = ort.InferenceSession(
    "model.onnx",
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)
```

The best thread count depends on available CPU cores, memory bandwidth, storage, and concurrent workloads. Setting
`intra_op_num_threads` to `1` keeps prepacking sequential even if the configuration entry is enabled.

## CUDA external initializer loading

Models saved with [external data](https://onnx.ai/onnx/repo-docs/ExternalData.html) normally load weights through
pageable CPU memory before copying them to the GPU. The CUDA execution provider can instead use NVIDIA GPUDirect
Storage (GDS) on Linux, Microsoft DirectStorage on Windows, or reusable pinned host buffers.

### GPUDirect Storage

GDS loads external initializers without staging file data in CPU memory. It is opt-in. If it cannot be initialized or
cannot read an external-data file, ONNX Runtime logs a warning and uses the configured pinned/pageable host-memory
fallback for the rest of the session.

With GDS enabled, ONNX Runtime opens each external-data file with `O_DIRECT` and uses `libcufile` to read 64 MiB
blocks into a reusable, cuFile-registered CUDA buffer. Each block is then copied device-to-device into the
initializer allocation owned by the CUDA arena:

```text
external-data file -> registered CUDA staging buffer -> CUDA arena initializer
                          cuFileRead                    device-to-device copy
```

The reusable staging buffer bounds additional GPU memory usage to 64 MiB per CUDA external-data loader. Each
device-to-device copy completes before that buffer is reused. String and Boolean initializers retain the existing
loading path because they require host-side conversion. Loaders share a process-wide cuFile driver. Its final release
and subsequent initialization are serialized, so a new loader cannot configure or reopen the driver until the
previous driver has finished closing.

GDS requires:

- Linux and `cufile.h` with `cuFileSetParameterBool`, `CUFILE_PARAM_USE_PCIP2PDMA`, and
  `CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE`;
- `libcufile.so` at runtime;
- either `nvidia-fs` or a recent open NVIDIA kernel module that supports PCI P2PDMA;
- a supported storage/filesystem and PCIe topology; and
- external weights stored in a file that can be opened with `O_DIRECT`.

ONNX Runtime loads `libcufile` dynamically, so enabling the option does not add a mandatory runtime dependency for
users who keep GDS disabled. It requests PCI P2PDMA, which can provide GDS without `nvidia-fs` on supported recent
kernels, GPUs, and storage devices. It also disables cuFile compatibility mode: if the storage stack cannot provide
a native GDS path, ONNX Runtime uses its configured host-memory fallback instead of cuFile's internal POSIX fallback.
GDS is attempted only for external-data ranges whose offset and length are both 4 KiB aligned. An unaligned
initializer uses the configured fallback without disabling GDS for later aligned initializers.

The build checks for the required cuFile configuration API. Older CUDA toolkits without it remain supported, but
enabling GDS in those builds logs a warning and uses the configured host-memory fallback.

### Microsoft DirectStorage (Windows)

The built-in CUDA execution provider also supports Microsoft's DirectStorage API through D3D12/CUDA
interoperability. Build with `--cmake_extra_defines onnxruntime_USE_CUDA_DIRECTSTORAGE=ON` in addition to the
usual CUDA build options. This option is not supported with the CUDA plugin EP
(`onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON`) or minimal builds (`onnxruntime_MINIMAL_BUILD`, `onnxruntime_CUDA_MINIMAL`);
CMake rejects those combinations rather than silently compiling an unavailable backend.
This opt-in build downloads the pinned DirectStorage SDK headers; it does not
introduce a link-time dependency on `dstorage.dll`. Deploy the SDK's matching x64 `dstorage.dll` and
`dstoragecore.dll` beside the application executable, following Microsoft's
[DirectStorage deployment guidance](https://github.com/microsoft/DirectStorage/blob/main/Docs/DeveloperGuidance.md#sdk-path).
For Python, the application executable is `python.exe`, not the CUDA provider DLL.
The application must make `dstorage.dll` discoverable through the Windows application/system/user DLL search
directories; the current working directory is not searched.

Set the CUDA EP option `external_data_loader_use_directstorage` to `"1"` to enable this path.
It requires a Windows D3D12-capable NVIDIA adapter that supports CUDA external memory and fence import.
The D3D12 adapter is selected by the CUDA device's LUID, not by assuming that both APIs enumerate GPUs
in the same order. Linked D3D12 adapters are not supported.

```text
external-data file -> DirectStorage -> shared D3D12 GPU buffer -> CUDA arena initializer
                     internal staging    CUDA external memory    device-to-device copy
```

The loader reuses a 32 MiB shared GPU buffer and a DirectStorage queue. A shared D3D12 fence establishes
completion and visibility to CUDA; request errors are checked before copying bytes into the initializer.
CUDA copies complete before the next DirectStorage write reuses the buffer. The additional GPU buffer
does not include DirectStorage's own internal staging allocations. This implementation loads uncompressed
ONNX external data, not GDeflate-compressed weights.

**DirectStorage is not NVIDIA GPUDirect Storage:** Microsoft's uncompressed data flow can use system-memory
and upload-heap staging. Successful DirectStorage loading does not establish zero-copy storage-to-VRAM DMA
or prove that Windows BypassIO was used. Compare actual timings rather than assuming the API is faster.
See Microsoft's [uncompressed data flow documentation](https://github.com/microsoft/DirectStorage/blob/main/Docs/DeveloperGuidance.md#uncompressed-data-flow).

DirectStorage accepts unaligned offsets and lengths. Because its API opens files by path, the loader compares
the opened file's identity with the handle already validated by ONNX Runtime before submitting reads.
Initialization or read failures produce a warning and disable DirectStorage for the remainder of that loader's
lifetime, using the configured host-memory fallback. Boolean tensors retain the host conversion path.
If both direct-storage options are enabled, DirectStorage is tried first, followed by GDS, then the host path;
normally enable only the option appropriate for the operating system. Unsupported builds/platforms report
unavailability and use the fallback instead of silently claiming DirectStorage support.

### Selecting the direct-storage paths

The following CUDA execution provider options enable the direct-storage paths:

| Option | Values | Default | Purpose |
|---|---|---:|---|
| `external_data_loader_use_gds` | `0` or `1` | `0` | Try GDS before the host-memory loading paths |
| `external_data_loader_use_directstorage` | `0` or `1` | `0` | Try Microsoft DirectStorage through D3D12/CUDA on Windows |

Initializers that a direct-storage path cannot load use the pinned-buffer or pageable host-memory path described below.
Keep `external_data_loader_reading_threads` greater than zero when enabling either option to retain the pinned-buffer
backup.

### Pinned-buffer host loading

For models whose total external initializer size on the target CUDA device is at least 16 MiB
(`session.external_data_loader.min_size_bytes`), the CUDA execution provider can load external initializers through two
reusable 16 MiB pinned host buffers (32 MiB total pinned memory by default):

```text
external-data file -> pinned buffer 0/1 -> CUDA initializer allocation
                         CPU read       cudaMemcpyAsync
```

The buffers are sliced across reader workers, each owning a pair of non-blocking CUDA streams and alternating pinned
staging slices so that multi-threaded file reads overlap host-to-device DMA transfers across chunks and initializers.
Each stream is synchronized before its slice is reused, and all in-flight transfers are finalized before session
initialization completes, bounding pinned host memory use at 32 MiB per loader by default. Both the built-in CUDA EP
and the CUDA plugin EP use this loader (not compiled in minimal builds).

The `session.external_data_loader.threads` session configuration entry (or `external_data_loader_reading_threads` CUDA
provider option) controls how each block is filled:

| Value | Behavior |
|---:|---|
| `0` | Disable the CUDA external-data loader and use the framework's pageable-memory path. With GDS or DirectStorage enabled, the loader is kept for direct reads and everything else uses pageable host buffers instead of pinned buffers |
| `1` | Use pinned buffers with synchronous reads on the calling thread |
| `2` to `64` | Use that many independent CPU read tasks |

When left unset, the default thread count is computed from the host physical CPU core count (`clamp(cores / 2, 2, 8)`).
Explicitly setting `session.external_data_loader.threads` or `external_data_loader_reading_threads` uses the exact
specified thread count.

Additional session configuration entries (and matching CUDA provider options) tune staging memory, transfer
granularity, and the minimum model-size threshold:

| Session Config Key | CUDA Provider Option | Valid Range | Default |
|---|---|---|---|
| `session.external_data_loader.threads` | `external_data_loader_reading_threads` | `0` to `64` | Auto from CPU cores (`2`..`8`) |
| `session.external_data_loader.pinned_memory_bytes` | `external_data_loader_pinned_memory_bytes` | `1048576` (1 MiB) to `1073741824` (1 GiB), multiple of `8192` | `33554432` (32 MiB total, two 16 MiB buffers) |
| `session.external_data_loader.chunk_bytes` | `external_data_loader_chunk_bytes` | `4096` (4 KiB) to `67108864` (64 MiB), multiple of `4096` | `2097152` (2 MiB) |
| `session.external_data_loader.min_size_bytes` | N/A | `0` to `4294967296` (4 GiB) | `16777216` (16 MiB) |

If the model's total external weights on the target device are smaller than `session.external_data_loader.min_size_bytes`,
or if pinned buffers or CUDA streams cannot be created, loading uses a pageable buffer without allocating pinned
memory or extra CUDA streams. Models with weights embedded in the ONNX file do not use this external-data path.

### Opened-file path validation requirements

For native models that use external data, ONNX Runtime validates the canonical path of the opened file handle before
reading or mapping it. This prevents a pathname or symlink from being swapped between validation and access.

- On Linux and Android, `/proc/self/fd` must be available. Containers, chroots, and sandboxes must mount procfs so
  ONNX Runtime can resolve the opened handle securely.
- Apple platforms use `F_GETPATH`, FreeBSD uses `F_KINFO`, and Windows uses
  `GetFinalPathNameByHandle`.
- Other native POSIX platforms without a secure handle-backed canonical-path API, including the current NetBSD and
  AIX implementations, fail closed when loading external-data models. Embedded-data models remain supported.
- WebAssembly does not provide native canonical-path validation. Its external-data loader uses the environment/path
  interface instead of the native opened-file path.

ONNX Runtime intentionally does not fall back to resolving and comparing a mutable pathname because that would
reintroduce a time-of-check/time-of-use race.

Configure the external-data loader in Python via `SessionOptions` or CUDA EP provider options:

```python
import sys

import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.add_session_config_entry("session.external_data_loader.threads", "8")
session_options.add_session_config_entry("session.external_data_loader.pinned_memory_bytes", str(128 * 1024 * 1024))
session_options.add_session_config_entry("session.external_data_loader.chunk_bytes", str(4 * 1024 * 1024))

providers = [
    (
        "CUDAExecutionProvider",
        {
            "external_data_loader_use_gds": "0" if sys.platform == "win32" else "1",
            "external_data_loader_use_directstorage": "1" if sys.platform == "win32" else "0",
        },
    ),
    "CPUExecutionProvider",
]

session = ort.InferenceSession(
    "model.onnx",
    sess_options=session_options,
    providers=providers,
)
```

For C++, both optimizations are configured through `Ort::SessionOptions`:

```cpp
Ort::SessionOptions session_options;
session_options.SetIntraOpNumThreads(8);
session_options.AddConfigEntry("session.prepack.enable_parallel", "1");
session_options.AddConfigEntry("session.external_data_loader.threads", "8");
session_options.AddConfigEntry("session.external_data_loader.pinned_memory_bytes", "134217728");
session_options.AddConfigEntry("session.external_data_loader.chunk_bytes", "4194304");

Ort::CUDAProviderOptions cuda_options;
cuda_options.Update({
#ifdef _WIN32
    {"external_data_loader_use_directstorage", "1"},
#else
    {"external_data_loader_use_gds", "1"},
#endif
});
session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model_loading");
Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
```

The CPU prepacking option only parallelizes CPU EP kernels. The CUDA provider option only changes how external
initializers assigned to CUDA memory are staged and copied; CPU and other execution providers retain their existing
loading paths. They can be enabled together for models partitioned between CPU and CUDA.

### Tests

`CudaExternalDataLoaderTest.*Gds*` exercises the real loader with GDS enabled, including aligned and unaligned ranges,
multiple buffers, repeated loads, and different host-memory fallback configurations. These tests require a CUDA GPU,
but use the configured fallback when native GDS is unavailable. A passing result alone does not prove native GDS usage
or performance. `CudaGdsDriverTest.*` separately checks shared-driver lifetime synchronization without GPU hardware.

`CApiTest.CUDAProviderOptions*Gds*` checks string-based configuration, invalid values, and serialization round trips.

`CudaExternalDataLoaderTest.*DirectStorage*` covers option validation, aligned and unaligned reads, repeated
loads, multiple buffers, and configured fallbacks. `NativeDirectStorageWithoutFallback` calls the DirectStorage
backend directly, verifies loaded bytes, rejects out-of-range and mismatched-file requests, and cannot succeed
by using the host fallback. It explicitly skips when D3D12/CUDA/DirectStorage initialization is unavailable.
`CApiTest.CUDAProviderOptionsDirectStorageRoundTrip` covers the string-based option.

Successful loads emit INFO records of the form `CUDA external data loader: path=<path> bytes=<bytes>`,
where `<path>` is `pageable`, `pinned`, `gds`, or `directstorage`. Enable the **default** logger's INFO severity
(`ort.set_default_logger_severity(1)` in Python) as well as the session logger when collecting these records.
They identify the path actually used, including fallbacks; requested provider options alone are not proof.

### Comparing all three loading paths

`onnxruntime/test/python/transformers/benchmark_cuda_model_loading.py` compares pageable CPU staging
(the loader-disabled baseline), pinned buffers, and the platform's direct-storage API, all targeting the
same CUDA device. Use the Python package from the build being evaluated, not an installed older wheel.
On Windows, enable the DirectStorage build option and deploy the SDK runtime as described above.

For example, generate identical aligned external weights totaling 1 GiB and collect five measurements per path:

```powershell
python onnxruntime\test\python\transformers\benchmark_cuda_model_loading.py `
  --generate-model .\cuda-loading-1gib --weight-count 16 --weight-dim 4096 `
  --threads 1 --repetitions 5 --output .\cuda-loading-1gib.json
```

Use `--weight-count 64` and a different output directory for 4 GiB of weights, provided enough GPU memory
is available. Generation refuses to overwrite an existing fixture. To benchmark a real model instead, supply
`--model model.onnx --inputs inputs.npz --expected-outputs expected.npz`; the NPZ keys must match tensor names,
and reference outputs must be computed independently.

Each sample uses a fresh process. The timed interval covers session construction, including graph initialization
and completed weight transfers, but not imports, fixture generation, or output verification. A blocking inference
and comparison against independent reference values happen afterward. Reported GiB/s is therefore **effective
end-to-end initialization throughput**, not raw disk bandwidth. The JSON report includes timing distributions,
GPU/configuration metadata, actual loaded bytes by path, and separately classified fallback or mixed-path samples.
A direct-storage fallback is never counted as a successful direct-storage measurement.

Caches remain OS-managed: fixture creation and warmups may warm the filesystem cache, and a fresh process does
not imply a cold disk. The optional POSIX per-file eviction hint is also not proof of cold-cache operation.
The script never performs privileged or global cache flushing. Record storage/filesystem details alongside results,
and do not compare Windows DirectStorage and Linux GDS numbers as if they came from an identical software stack.

#### Windows measurements (2026-09-23)

Measured on Windows 11 (build 26200), an NVIDIA RTX 4060 Laptop GPU (8 GiB, WDDM, driver 591.55),
and a local WD `SDCPNRZ-2T00-1124-WD` 2 TB SSD. The source build used MSVC 2022, CUDA 13.0.2,
DirectStorage 1.2.3, and CPython 3.13.14. It was a Release/quick build restricted to SM89 and
MatMul/Gather registrations, with contrib support enabled. The fixture used FP32 4096-by-4096 weights,
one intra-op thread, four pinned-buffer readers, disabled graph optimization/prepacking, and disabled TF32.

Each row summarizes five fresh-process samples after one warmup per path, with OS-managed (potentially warm)
caches. The complete weight-byte counts were confirmed from actual-path logs and every sample's outputs
matched independent references. DirectStorage rows used the DirectStorage API, not the pinned fallback.

| External weights | Loading path | Median initialization | Effective throughput |
|---|---|---:|---:|
| 1 GiB | CPU pageable | 1.517 s | 0.659 GiB/s |
| 1 GiB | Pinned buffers | 1.105 s | 0.905 GiB/s |
| 1 GiB | Microsoft DirectStorage | 1.853 s | 0.540 GiB/s |
| 4 GiB | CPU pageable | 6.201 s | 0.645 GiB/s |
| 4 GiB | Pinned buffers | 2.292 s | 1.745 GiB/s |
| 4 GiB | Microsoft DirectStorage | 7.870 s | 0.508 GiB/s |

On this machine and workload, pinned buffers outperform both pageable loading and the current uncompressed
DirectStorage implementation. These are complete session-initialization measurements, not isolated I/O timings,
and are not evidence of cold-cache disk bandwidth or Linux GDS performance.

## Zero-copy external initializer buffers (`session.use_external_initializer_file_buffers_directly`)

When an application supplies external initializer weights from memory-mapped files (for example `.onnx.data` or `.gguf`
weight files mapped via `mmap`) using `AddExternalInitializers` or `AddExternalInitializersFromFilesInMemory`,
ONNX Runtime by default copies non-owning CPU tensor buffers into a newly allocated internal CPU buffer during
`InferenceSession::Initialize()`.

Setting `session.use_external_initializer_file_buffers_directly` to `"1"` instructs ONNX Runtime to borrow naturally
aligned CPU buffers larger than 127 bytes on little-endian hosts instead of allocating and copying a duplicate CPU
buffer. Other buffers are copied. Tensors passed to `AddExternalInitializers` must be contiguous, non-owning views of
user-owned buffers and contain native-endian values; allocator-owned tensors are rejected.
Files passed to `AddExternalInitializersFromFilesInMemory` contain serialized little-endian values.

```cpp
Ort::SessionOptions session_options;
session_options.AddConfigEntry("session.use_external_initializer_file_buffers_directly", "1");
session_options.AddExternalInitializers(initializer_names, initializer_values);
```

Borrowing is opt-in because it changes the lifetime contract: the application must keep each borrowed CPU buffer
unchanged and alive until all sessions using it have been released. With the default setting, non-owning buffers can
be released after session creation.

Device tensors passed to `AddExternalInitializers` are always copied to owned CPU staging memory using the registered
execution provider's data-transfer implementation before graph optimization or shape inference. They are subsequently
copied to the device selected by the execution plan. This option does not enable borrowing device buffers, so caller
device buffers can be released after session creation regardless of the setting. Session creation fails if no suitable
data-transfer implementation is registered.
