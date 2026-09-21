# Accelerate model loading

ONNX Runtime provides two independent loading optimizations for models with large weights:

| Target | Mechanism | Configuration | Default |
|---|---|---|---|
| CPU prepacking | Run eligible CPU kernel `PrePack()` calls concurrently | `session.prepack.enable_parallel` | Disabled (`"0"`) |
| CUDA external initializers | Read external data through reusable pinned buffers while copying to the GPU | CUDA EP option `external_data_loader_reading_threads` | 4 readers |

The CPU option is a session configuration entry. The CUDA option is an execution provider option passed when the
CUDA EP is appended to `SessionOptions`.

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
pageable CPU memory before copying them to the GPU. The CUDA execution provider can instead load external
initializers through two reusable 64 MiB pinned host buffers:

```text
external-data file -> pinned buffer 0/1 -> CUDA initializer allocation
                         CPU read       cudaMemcpyAsync
```

The buffers alternate so that reading the next block can overlap the host-to-device transfer of the current block.
Each buffer is synchronized before reuse. Initializers are loaded one at a time through the shared staging resources,
which bounds pinned host memory use at 128 MiB per loader.

The `external_data_loader_reading_threads` CUDA provider option controls how each block is filled:

| Value | Behavior |
|---:|---|
| `0` | Disable the CUDA external-data loader and use the framework's pageable-memory path |
| `1` | Use pinned buffers with synchronous reads on the calling thread |
| `2` to `64` | Use that many independent CPU read tasks per block |

The default is `4`, so the pinned-buffer loader is enabled without additional configuration. Parallel reads are used
for external tensors of at least 16 MiB; smaller tensors use one read. The optimal reader count depends on the storage
device and filesystem. If pinned buffers or CUDA streams cannot be created, loading falls back to a pageable buffer.
Models with weights embedded in the ONNX file do not use this external-data path.

Configure the CUDA EP in Python:

```python
import onnxruntime as ort

session_options = ort.SessionOptions()
providers = [
    (
        "CUDAExecutionProvider",
        {"external_data_loader_reading_threads": "4"},
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
cuda_options.Update({{"external_data_loader_reading_threads", "4"}});
session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model_loading");
Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
```

The CPU prepacking option only parallelizes CPU EP kernels. The CUDA provider option only changes how external
initializers assigned to CUDA memory are staged and copied; CPU and other execution providers retain their existing
loading paths. They can be enabled together for models partitioned between CPU and CUDA.
