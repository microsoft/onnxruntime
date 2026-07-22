# MatMulBlockScaledFp4 - CUDA Experiments

This document records CUDA experiments for
**MatMulBlockScaledFp4** (`com.microsoft::MatMulBlockScaledFp4`) that are useful
for future performance work but are not part of the final dispatch chain.

Related documentation:

- [matmul_block_scaled_fp4.md](matmul_block_scaled_fp4.md) - operator behavior and current dispatch chain.
- [matmul_nbits_small_m_experiments.md](matmul_nbits_small_m_experiments.md) - similar small-M GEMV experiment notes for MatMulNBits.

---

## Table of Contents

1. [Native SM120 FP4 x FP4 GEMM](#1-native-sm120-fp4-x-fp4-gemm)
2. [Prepacking SM120 Swizzled B Scales](#2-prepacking-sm120-swizzled-b-scales)
3. [Rejected Experiment - Reuse Swizzled B Scales for Decode GEMV](#3-rejected-experiment---reuse-swizzled-b-scales-for-decode-gemv)
4. [Benchmark Commands](#4-benchmark-commands)
5. [Lessons](#5-lessons)

---

## 1. Native SM120 FP4 x FP4 GEMM

The default FP4 operator is weight-only: `A` remains FP16/BF16, `B` is
dequantized to the activation type, and cuBLAS computes the GEMM. On Blackwell
SM120, CUTLASS also supports native block-scaled NVFP4 x NVFP4 GEMM. The native
path was added behind:

```bash
ORT_MATMUL_BLOCK_SCALED_FP4_NATIVE_SM120=1
```

Native path conditions:

- SM120 device,
- `M > 8` because decode GEMV has priority,
- `block_size == 16`,
- `K % 32 == 0`,
- `N % 32 == 0`.

The path quantizes `A` to packed NVFP4 E2M1, creates per-16-block E4M3
activation scales, uses SM120-swizzled B scales, and launches CUTLASS
block-scaled GEMM.

Accuracy is compared against an activation-quantized FP4 reference in the Python
harness. This is intentional: the native path is not bitwise-equivalent to the
default weight-only FP4 path because it also quantizes activations.

Representative result on Blackwell GPU 0 for `M=16,N=11008,K=4096,fp16`:

| Path | Mean latency | TFLOP/s | Accuracy reference |
|------|--------------|---------|--------------------|
| Default dequant + cuBLAS | about 0.71 ms | about 2.0 | weight-only FP4 reference |
| Native SM120 FP4 x FP4, with prepacked B scales | about 0.15-0.16 ms | about 9.1-9.5 | activation-quantized FP4 reference |

---

## 2. Prepacking SM120 Swizzled B Scales

The native CUTLASS GEMM expects B scales in an SM120 swizzled scale layout, not
the operator's original `[N, K / 16]` row-major scale tensor. Repacking the scale
tensor on every run was measurable overhead, so `PrePack` now repacks initializer
`weight_scale` once into `b_scale_prepacked_`.

Important detail: `PrePack` does **not** mark `weight_scale` as removable. The
original unswizzled scale input is still needed by:

- decode GEMV,
- default dequantize + cuBLAS fallback,
- dynamic cases where native SM120 is not selected.

Measured effect on the same representative prefill shape:

| Variant | Mean latency |
|---------|--------------|
| Native SM120 with per-run B-scale repack | about 0.19-0.20 ms |
| Native SM120 with `PrePack` cached B-scale repack | about 0.15-0.16 ms |

This optimization is kept.

---

## 3. Rejected Experiment - Reuse Swizzled B Scales for Decode GEMV

Question tested: can the decode GEMV use the same prepacked SM120 swizzled B
scale buffer and avoid reading the original unswizzled `weight_scale`?

Implementation attempted:

- Add a sibling GEMV launcher that accepts `b_scale_prepacked_`.
- Add a swizzled-scale accessor equivalent to the SM120 CUTLASS scale layout.
- Route decode GEMV to this launcher when `b_scale_prepacked_` exists.

Result: correct, but slower.

Reason: the existing decode GEMV maps one warp to one output column. For that
access pattern, the original layout gives contiguous per-column scale loads:

```
weight_scale[col * k_blocks + kb]
```

The SM120 swizzled layout is optimized for tiled block-scaled GEMM. For one
fixed output column, consecutive K-block scale loads jump through memory. Even
after precomputing the row base and reducing accessor arithmetic, the swizzled
scale layout was slower for decode.

Representative `M=1,N=11008,K=4096,fp16` result:

| Decode variant | Mean latency | Notes |
|----------------|--------------|-------|
| Original GEMV, unswizzled scales | about 0.116 ms | contiguous scale row |
| Swizzled-scale GEMV attempt | about 0.138 ms | strided scale access |

Representative `M=8,N=11008,K=4096,fp16` result:

| Decode variant | Mean latency | Notes |
|----------------|--------------|-------|
| Original GEMV, unswizzled scales | about 0.382 ms | contiguous scale row |
| Swizzled-scale GEMV attempt | about 0.507 ms | strided scale access |

Decision: remove the swizzled-scale GEMV path and keep decode on the original
unswizzled scale layout.

---

## 4. Benchmark Commands

The commands below use `ORT_REPO` and `ORT_BUILD` so they can be copied without
editing developer-specific paths. Set them once:

```bash
export ORT_REPO=$(git rev-parse --show-toplevel)
export ORT_BUILD="$ORT_REPO/build/cu130/Release"
```

Provider rebuild and Python-provider sync:

```bash
cmake --build "$ORT_BUILD" --target onnxruntime_providers_cuda --parallel
cp "$ORT_BUILD/libonnxruntime_providers_cuda.so" \
  "$ORT_BUILD/onnxruntime/capi/libonnxruntime_providers_cuda.so"
cp "$ORT_BUILD/libonnxruntime_providers_cuda.so" \
  "$ORT_BUILD/build/lib/onnxruntime/capi/libonnxruntime_providers_cuda.so"
```

Decode benchmarks:

```bash
cd /tmp && PYTHONPATH="$ORT_BUILD" CUDA_VISIBLE_DEVICES=0 \
  ORT_MATMUL_BLOCK_SCALED_FP4_NATIVE_SM120=1 \
  python "$ORT_REPO/onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py" \
  --op fp4 --activation-dtype fp16 --m 1 --n 11008 --k 4096 --warmup 100 --repeat 500

cd /tmp && PYTHONPATH="$ORT_BUILD" CUDA_VISIBLE_DEVICES=0 \
  ORT_MATMUL_BLOCK_SCALED_FP4_NATIVE_SM120=1 \
  python "$ORT_REPO/onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py" \
  --op fp4 --activation-dtype fp16 --m 8 --n 11008 --k 4096 --warmup 100 --repeat 500
```

Native prefill benchmark:

```bash
cd /tmp && PYTHONPATH="$ORT_BUILD" CUDA_VISIBLE_DEVICES=0 \
  ORT_MATMUL_BLOCK_SCALED_FP4_NATIVE_SM120=1 \
  python "$ORT_REPO/onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py" \
  --op fp4 --activation-dtype fp16 --m 16 --n 11008 --k 4096 --warmup 50 --repeat 200
```

Focused C++ tests:

```bash
CUDA_VISIBLE_DEVICES=0 "$ORT_BUILD/onnxruntime_provider_test" \
  --gtest_filter='MatMulBlockScaledFp4OpTest.*'
```

---

## 5. Lessons

- Keep the native SM120 swizzled scale layout for native GEMM only.
- Keep decode GEMV on the original `[N, K / 16]` scale layout.
- Prepacking can still cache the native GEMM swizzled scale buffer, but it must
  leave the original `weight_scale` input available.
- When a native path changes arithmetic semantics by quantizing activations,
  validate against an activation-quantized reference, not the weight-only FP4
  reference.
