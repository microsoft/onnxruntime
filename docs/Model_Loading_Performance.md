# Accelerate model loading

ONNX Runtime provides two independent loading optimizations for models with large weights:

| Target | Mechanism | Configuration | Default |
|---|---|---|---|
| CPU prepacking | Run eligible CPU kernel `PrePack()` calls concurrently | `session.prepack.enable_parallel` | Disabled (`"0"`) |
| CUDA external initializers | Read external data through reusable pinned buffers while copying to the GPU | `session.external_data_loader.threads` / CUDA EP `external_data_loader_reading_threads` | Auto from CPU cores (`2`..`8`) |
| CUDA external initializers | Total pinned host staging memory across the two reusable buffers | `session.external_data_loader.pinned_memory_bytes` / CUDA EP `external_data_loader_pinned_memory_bytes` | 32 MiB (`"33554432"`) |
| CUDA external initializers | Maximum per-transfer chunk size for parallel reads and async H2D copies | `session.external_data_loader.chunk_bytes` / CUDA EP `external_data_loader_chunk_bytes` | 2 MiB (`"2097152"`) |
| CUDA external initializers | Minimum total external initializer bytes on the device before allocating pinned buffers/streams | `session.external_data_loader.min_size_bytes` | 16 MiB (`"16777216"`) |

The `session.*` options are session configuration entries. The CUDA external-data loader options can also be passed as
execution provider options when the CUDA EP (built-in or plugin) is appended to `SessionOptions`.

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

## Pinned-buffer loading for CUDA external initializers

Models saved with [external data](https://onnx.ai/onnx/repo-docs/ExternalData.html) normally load weights through
pageable CPU memory before copying them to the GPU. For models whose total external initializer size on the target
CUDA device is at least 16 MiB (`session.external_data_loader.min_size_bytes`), the CUDA execution provider can instead
load external initializers through two reusable 16 MiB pinned host buffers (32 MiB total pinned memory by default):

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
| `0` | Disable the CUDA external-data loader and use the framework's pageable-memory path |
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
import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.add_session_config_entry("session.external_data_loader.threads", "8")
session_options.add_session_config_entry("session.external_data_loader.pinned_memory_bytes", str(128 * 1024 * 1024))
session_options.add_session_config_entry("session.external_data_loader.chunk_bytes", str(4 * 1024 * 1024))

session = ort.InferenceSession(
    "model.onnx",
    sess_options=session_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model_loading");
Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
```

The CPU prepacking option only parallelizes CPU EP kernels. The CUDA provider option only changes how external
initializers assigned to CUDA memory are staged and copied; CPU and other execution providers retain their existing
loading paths. They can be enabled together for models partitioned between CPU and CUDA.

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
