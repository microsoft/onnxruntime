# Source-compatible BF16 CUDA operators

This optional, standalone custom-op library preserves the FlashInfer RMSNorm
and sink-enabled FlashAttention-3 arithmetic used by RecursiveSI Flint.
It uses the public ORT API, the ORT CUDA compute stream, and pinned upstream
CUDA/CUTLASS sources. It does not link PyTorch, LibTorch, or a publisher wheel.
It is not enabled in the regular ORT build.

## Build and package

Requires CMake 3.26+, CUDA 12.6+, a C++17 compiler, and a Hopper GPU.
The deployed ORT must support the API version in the headers used for the build.
The ORT CUDA provider's own CUDA/cuDNN dependencies must also be available.

```sh
cmake -S onnxruntime/contrib_ops/cuda/bert/source_compatibility \
  -B build/source-compat -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/source-compat --parallel 2
cmake --install build/source-compat --prefix /path/to/model
```

Installation produces `lib/libonnxruntime_source_compat_cuda.so` and upstream
license files under `share/source_compat_cuda`. Keep the licenses with
redistributed binaries. Dependency archives are pinned by revision and SHA256.
`ORT_HEADER_DIR` can select headers from a matching ORT installation.

Register the installed library using
`SessionOptions.register_custom_ops_library()` before constructing an ORT
session. For ORT GenAI, add the following decoder session option:

```json
{
  "custom_ops_libraries": ["lib/libonnxruntime_source_compat_cuda.so"]
}
```

The library path is relative to the model directory. Mobius Flint exports emit
this requirement; install the library into the export directory before loading.

## Operator contract

Domain: `com.microsoft.source_compat`, version 1. All floating inputs and outputs
are BF16; no implicit precision conversion is provided.

| Operator | Inputs | Outputs | Attributes |
| --- | --- | --- | --- |
| `RMSNorm` | `X[..., H]`, `scale[H]` | normalized `X` | positive, finite `epsilon` |
| `SkipRMSNorm` | `X[..., H]`, `residual[..., H]`, `scale[H]` | normalized sum, stored residual | positive, finite `epsilon` |
| `SinkAttention` | `Q[1,Q,Hq*128]`, `K[1,K,Hkv*128]`, `V[1,K,Hkv*128]`, `sinks[Hq]`, INT64 `mask[1,K]` | `[1,Q,Hq*128]` | `q_num_heads`, `kv_num_heads`, `window_left`, positive finite `scale` |

Norm width must be divisible by eight. `SkipRMSNorm` normalizes the unrounded
FP32 sum and separately stores its BF16 residual. It does not modify its inputs.

Attention requires one **unpadded** sequence, positive query length, `K >= Q`,
head dimension 128, and grouped heads with `Hq > Hkv` and `Hq % Hkv == 0`.
The mask must contain only ones; padding fails explicitly. Queries represent
the final `Q` positions in the supplied complete KV sequence. Full causal
attention uses `window_left=-1`; Flint's 513-token sliding window uses
`window_left=512`. Learned sinks have null values and retain their softmax mass.
Cache concatenation and logical position IDs are supplied by the graph.

Only Hopper is supported. Batching, padding, static/shared caches, beam search,
CUDA graph capture, non-BF16 types, and non-128 attention heads are not qualified.
CUDA scratch allocations are stream-ordered. Errors are returned through the
ORT V2 custom-op interface rather than terminating the host process.

## Focused execution tests

The shared provider source-list regression does not require CUDA:

```sh
python onnxruntime/contrib_ops/cuda/bert/source_compatibility/source_filters_test.py
```

With `onnxruntime-gpu`, `onnx-ir`, `onnxscript`, `numpy`, `ml_dtypes`, and `pytest`
available in an isolated environment:

```sh
ORT_SOURCE_COMPAT_CUDA_LIBRARY=/path/to/model/lib/libonnxruntime_source_compat_cuda.so \
  python -m pytest \
  onnxruntime/contrib_ops/cuda/bert/source_compatibility/source_compat_test.py -q
```

The tests exercise native CUDA placement, unrounded residual normalization,
prefill/decode, 512/513/514 boundaries, and invalid masks/scales/shapes.
They do not establish full-model numerical parity or generation quality.
