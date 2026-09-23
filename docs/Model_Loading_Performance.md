# Accelerate model loading

ONNX Runtime provides two independent loading optimizations for models with large weights:

| Target | Mechanism | Configuration | Default |
|---|---|---|---|
| CPU prepacking | Run eligible CPU kernel `PrePack()` calls concurrently | `session.prepack.enable_parallel` | Disabled (`"0"`) |
| CUDA external initializers | Read external data with GPUDirect Storage or reusable pinned buffers | CUDA EP options `external_data_loader_use_gds` and `external_data_loader_reading_threads` | GDS disabled; 4 readers |

The CPU option is a session configuration entry. The CUDA options are execution provider options passed when the CUDA
EP is appended to `SessionOptions`.

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
Storage (GDS) or reusable pinned host buffers.

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

### Pinned-buffer host loading

The CUDA execution provider can load external initializers through two reusable 64 MiB pinned host buffers:

```text
external-data file -> pinned buffer 0/1 -> CUDA initializer allocation
                         CPU read       cudaMemcpyAsync
```

The buffers alternate so that reading the next block can overlap the host-to-device transfer of the current block.
Each buffer is synchronized before reuse. Initializers are loaded one at a time through the shared staging resources,
which bounds pinned host memory use at 128 MiB per loader.

The following CUDA execution provider options control the primary and fallback paths:

| Option | Values | Default | Purpose |
|---|---|---:|---|
| `external_data_loader_use_gds` | `0` or `1` | `0` | Try GDS before another external-data loading path |
| `external_data_loader_reading_threads` | `0` to `64` | `4` | Configure the pinned-buffer fallback; `0` disables it |

Keep `external_data_loader_reading_threads` greater than zero when enabling GDS to retain the pinned-buffer backup.
`1` uses synchronous reads into pinned memory. Values from `2` through `64` use that many parallel read tasks per
64 MiB pinned buffer. The default is `4`, so the pinned-buffer loader is enabled without additional configuration.
Parallel reads are used for external tensors of at least 16 MiB; smaller tensors use one read. The optimal reader count
depends on the storage device and filesystem. If pinned buffers or CUDA streams cannot be created, loading falls back
to a pageable buffer. If the value is `0`, the pageable path is used when GDS is disabled or unavailable. Models with
weights embedded in the ONNX file do not use this external-data path.

Configure the CUDA EP in Python:

```python
import onnxruntime as ort

session_options = ort.SessionOptions()
providers = [
    (
        "CUDAExecutionProvider",
        {
            "external_data_loader_use_gds": "1",
            "external_data_loader_reading_threads": "4",
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

Ort::CUDAProviderOptions cuda_options;
cuda_options.Update({
    {"external_data_loader_use_gds", "1"},
    {"external_data_loader_reading_threads", "4"},
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
