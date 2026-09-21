# CUDA external-data loading with GPUDirect Storage

The CUDA execution provider can use NVIDIA GPUDirect Storage (GDS) to load ONNX external initializers without
staging file data in CPU memory. GDS is opt-in. If it cannot be initialized or cannot read an external-data file,
ONNX Runtime logs a warning and uses the existing pinned-host-buffer loader for the rest of the session.

## Data path

With GDS enabled, ONNX Runtime opens each external-data file with `O_DIRECT` and uses `libcufile` to read 64 MiB
blocks into a reusable, cuFile-registered CUDA buffer. Each block is then copied device-to-device into the
initializer allocation owned by the CUDA arena:

```text
external-data file -> registered CUDA staging buffer -> CUDA arena initializer
                          cuFileRead                    device-to-device copy
```

The reusable staging buffer bounds additional GPU memory to 64 MiB per CUDA external-data loader. Each
device-to-device copy completes before that buffer is reused. String and Boolean initializers retain the existing
loading path because they require host-side conversion.

GDS requires:

- Linux and a CUDA toolkit that provides `cufile.h`;
- `libcufile.so` at runtime;
- a working GDS driver and supported storage/filesystem configuration; and
- external weights stored in a file that can be opened with `O_DIRECT`.

ONNX Runtime loads `libcufile` dynamically, so enabling the option does not add a mandatory runtime dependency for
users who keep GDS disabled.

## Configuration

The following CUDA execution provider options control the primary and fallback paths:

| Option | Values | Default | Purpose |
|---|---|---:|---|
| `external_data_loader_use_gds` | `0` or `1` | `0` | Try GDS before another external-data loading path |
| `external_data_loader_reading_threads` | `0` to `64` | `4` | Configure the pinned-buffer fallback; `0` disables it |

Keep `external_data_loader_reading_threads` greater than zero when enabling GDS to retain the pinned-buffer backup.
`1` uses synchronous reads into pinned memory. Values from `2` through `64` use that many parallel read tasks per
64 MiB pinned buffer. If the value is `0` and GDS is unavailable, ONNX Runtime falls back to pageable host memory.

Python:

```python
import onnxruntime as ort

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

session = ort.InferenceSession("model.onnx", providers=providers)
```

C++:

```cpp
Ort::SessionOptions session_options;
Ort::CUDAProviderOptions cuda_options;
cuda_options.Update({
    {"external_data_loader_use_gds", "1"},
    {"external_data_loader_reading_threads", "4"},
});
session_options.AppendExecutionProvider_CUDA_V2(cuda_options);

Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
```

The option only affects initializers stored as ONNX external data and assigned to CUDA memory. Embedded initializers
and initializers assigned to other execution providers retain their existing paths.
