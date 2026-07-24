# MatMulBlockQuantizedFp8Weight - CUDA Experiments

This document records CUDA performance experiments for
**MatMulBlockQuantizedFp8Weight** (`com.microsoft::MatMulBlockQuantizedFp8Weight`).
It includes retained optimizations and their measured effect so future tuning
does not repeat work whose results are already known.

Related documentation:

- [matmul_block_scaled_fp8.md](matmul_block_scaled_fp8.md) - operator behavior and current dispatch chain.
- [matmul_block_scaled_fp4_experiments.md](matmul_block_scaled_fp4_experiments.md) - related FP4 experiments.

---

## Table of Contents

1. [Test Environment](#1-test-environment)
2. [Baseline Latency Profile](#2-baseline-latency-profile)
3. [Prefill Bottleneck - Weight Dequantization](#3-prefill-bottleneck---weight-dequantization)
4. [Optimization - Vectorized Dequantization Kernel](#4-optimization---vectorized-dequantization-kernel)
5. [Decode GEMV - Kept As Is](#5-decode-gemv---kept-as-is)
6. [Benchmark Commands](#6-benchmark-commands)
7. [Lessons](#7-lessons)

---

## 1. Test Environment

- GPU: NVIDIA GeForce RTX 5060 Ti, SM120 (Blackwell), 36 SMs, about 448 GB/s memory bandwidth.
- CUDA toolkit: 13.0.
- CUTLASS: 4.4.2.
- Build directory: `build/cu130/Release`.
- Benchmark shape unless stated otherwise: `N=4096`, `K=4096`, `block_size=128`.
- Data: FP16 activation `A`, FP8 E4M3 weight `B`, FP32 per-block scales, FP16 output.
- Timing: warmup then measured iterations (see commands in section 6); latency is the mean over measured iterations.
- Device selection: `CUDA_VISIBLE_DEVICES=0`.

The operator is weight-only: `A` stays FP16/BF16, `B` is dequantized to the
activation type, and the product runs either as a fused decode GEMV (small `M`)
or as a cuBLAS GEMM on the dequantized weight (larger `M`). There is no native
FP8 block-scaled tensor-core path; the design is architecture independent
(SM80+).

---

## 2. Baseline Latency Profile

Measured latency of the two dispatch paths at `N=4096, K=4096, fp16` before the
dequantization kernel was optimized:

| M | Path | Mean latency | Notes |
|----|------|--------------|-------|
| 1 | decode GEMV | 0.076 ms | one warp per output column, row group 1 |
| 2 | decode GEMV | 0.097 ms | row group 2 |
| 4 | decode GEMV | 0.123 ms | row group 4 |
| 8 | decode GEMV | 0.132 ms | row group 8 |
| 16 | dequant + cuBLAS | 0.396 ms | dequant dominated |
| 32 | dequant + cuBLAS | 0.396 ms | dequant dominated |
| 64 | dequant + cuBLAS | 0.399 ms | dequant dominated |
| 128 | dequant + cuBLAS | 0.453 ms | dequant + growing GEMM |
| 256 | dequant + cuBLAS | 0.596 ms | GEMM growing |
| 512 | dequant + cuBLAS | 0.732 ms | GEMM significant |

The flat 0.396-0.399 ms across `M = 16, 32, 64` is the tell: the prefill cost
does not depend on `M` there, so the GEMM is negligible and a fixed per-shape
cost dominates.

---

## 3. Prefill Bottleneck - Weight Dequantization

The default prefill path expands the whole `[N, K]` weight into an FP16/BF16
scratch buffer before the cuBLAS GEMM. That expansion moves `N*K` bytes in
(FP8) and `2*N*K` bytes out (FP16/BF16), independent of `M`. To confirm this is
the bottleneck, prefill latency at `M=16` was measured while varying `N*K`:

| N | K | N*K | Mean latency |
|------|------|--------|--------------|
| 4096 | 2048 | 8.4 M | 0.203 ms |
| 2048 | 4096 | 8.4 M | 0.194 ms |
| 4096 | 4096 | 16.8 M | 0.399 ms |
| 4096 | 8192 | 33.6 M | 0.730 ms |
| 8192 | 4096 | 33.6 M | 0.733 ms |

Latency scales linearly with `N*K` and is independent of whether `N` or `K`
grows, which matches a memory-bound dequantization and rules out the GEMM as the
prefill bottleneck. At 16.8 M elements in 0.40 ms the effective traffic is only
about 125 GB/s, roughly 28% of the 448 GB/s peak.

The original kernel `DequantizeBlockScaledFp8Kernel` mapped one thread to one
output element and computed, per element:

```cpp
const int row = static_cast<int>(idx / k);              // 64-bit division per element
const int col = static_cast<int>(idx - (long long)row * k);
const int blk = col / block_size;
out[idx] = FromFloat<T>(static_cast<float>(b_fp8[idx]) * weight_scale[row * k_blocks + blk]);
```

Two problems limited bandwidth: a 64-bit `idx / k` division on every element, and
scalar 1-byte loads / 2-byte stores that do not form wide coalesced memory
transactions.

---

## 4. Optimization - Vectorized Dequantization Kernel

`DequantizeBlockScaledFp8Vec16Kernel` replaces the scalar kernel when
`K % 16 == 0` (the common layout; the scalar kernel is kept for `K % 16 != 0`).
Each thread converts one aligned 16-element K chunk of a single row:

- **Coalesced wide memory access.** The 16 FP8 values load as one 16-byte
  `uint4`; the 16 FP16/BF16 results store as a 32-byte pair of `uint4`. Because
  `K % 16 == 0` every row begins 16-byte aligned, so both accesses are aligned
  and 32 threads of a warp cover 512 contiguous bytes.
- **No per-element division.** The row index comes from a 2D grid
  (`row = blockIdx.y`, with a grid-stride loop when `N` exceeds 65535), so the
  expensive `idx / k` division is gone entirely.
- **One scale load per chunk.** When `block_size % 16 == 0`, all 16 elements of a
  chunk fall in the same K block, so the kernel loads a single `b_scale` value
  per chunk instead of one per element. The general case still does a per-element
  block lookup.

Measured effect at `N=4096, K=4096, fp16`:

| M | Before | After | Speedup |
|----|--------|-------|---------|
| 16 | 0.396 ms | 0.262 ms | 1.51x |
| 32 | 0.396 ms | 0.260 ms | 1.52x |
| 64 | 0.399 ms | 0.261 ms | 1.53x |
| 128 | 0.453 ms | 0.303 ms | 1.50x |
| 256 | 0.596 ms | 0.392 ms | 1.52x |
| 512 | 0.732 ms | 0.584 ms | 1.25x |

Larger weight, `N=11008, K=4096, fp16`:

| M | Before | After | Speedup |
|----|--------|-------|---------|
| 32 | 0.970 ms | 0.662 ms | 1.47x |
| 128 | 1.005 ms | 0.713 ms | 1.41x |

Effective dequantization bandwidth (counting 3 bytes per element: 1 read + 2
write) after the change, from the `N*K` scaling sweep at `M=16`:

| N*K | Mean latency | Effective bandwidth |
|--------|--------------|---------------------|
| 8.4 M | 0.133 ms | about 189 GB/s |
| 16.8 M | 0.258 ms | about 195 GB/s |
| 33.6 M | 0.484 ms | about 208 GB/s |

Bandwidth rose from about 125 GB/s to about 195-208 GB/s (about 1.55x), and the
marginal throughput between the smallest and largest sweep points is about
215 GB/s. Accuracy is unchanged: the focused C++ tests and the Python harness
accuracy checks pass at every measured shape.

This optimization is kept.

---

## 5. Decode GEMV - Kept As Is

The decode GEMV `MatMulBlockScaledFp8GemvKernel` is already well tuned and was
not changed. It groups all rows of an `M <= 8` tile into one warp
(`RowsPerWarp` = 1, 2, 4, or 8) so each weight row streams exactly once, uses
`uint4` vectorized FP8 loads, folds one `b_scale` value per 16-element chunk
(valid because `block_size % 16 == 0` on this path), and reduces with warp
shuffles. Decode latency (0.076-0.145 ms for `M = 1..8`) is far below the
prefill dequant floor, so it is not the optimization target. Its numbers are
unchanged by the dequant kernel work, as expected.

---

## 6. Benchmark Commands

The commands below use `ORT_REPO` and `ORT_BUILD` so they can be copied without
editing developer-specific paths. Set them once:

```bash
export ORT_REPO=$(git rev-parse --show-toplevel)
export ORT_BUILD="$ORT_REPO/build/cu130/Release"
```

Provider rebuild and Python-provider sync after editing the `.cu` kernel:

```bash
cmake --build "$ORT_BUILD" --target onnxruntime_providers_cuda --parallel
cp "$ORT_BUILD/libonnxruntime_providers_cuda.so" \
  "$ORT_BUILD/onnxruntime/capi/libonnxruntime_providers_cuda.so"
```

Decode GEMV benchmarks (small M):

```bash
cd /tmp && PYTHONPATH="$ORT_BUILD" CUDA_VISIBLE_DEVICES=0 \
  python "$ORT_REPO/onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py" \
  --op fp8 --activation-dtype fp16 --m 1 --n 4096 --k 4096 --warmup 100 --repeat 500
```

Default prefill (dequantize + cuBLAS):

```bash
cd /tmp && PYTHONPATH="$ORT_BUILD" CUDA_VISIBLE_DEVICES=0 \
  python "$ORT_REPO/onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py" \
  --op fp8 --activation-dtype fp16 --m 32 --n 4096 --k 4096 --warmup 50 --repeat 200
```

`N*K` scaling sweep used to isolate dequantization cost:

```bash
for shape in "4096 2048" "4096 4096" "4096 8192" "2048 4096" "8192 4096"; do
  set -- $shape
  cd /tmp && PYTHONPATH="$ORT_BUILD" CUDA_VISIBLE_DEVICES=0 \
    python "$ORT_REPO/onnxruntime/test/python/contrib_ops/profile_matmul_block_scaled.py" \
    --op fp8 --activation-dtype fp16 --m 16 --n $1 --k $2 --warmup 50 --repeat 300
done
```

Focused C++ tests:

```bash
CUDA_VISIBLE_DEVICES=0 "$ORT_BUILD/onnxruntime_provider_test" \
  --gtest_filter='MatMulBlockQuantizedFp8WeightOpTest.*'
```

---

## 7. Lessons

- The prefill path is memory bound on weight dequantization, not on the GEMM;
  latency there scales with `N*K` and is independent of `M`.
- Isolate a memory-bound helper by sweeping the dimension it depends on (`N*K`
  here) rather than the FLOP-bearing dimension (`M`). The flat latency across
  `M = 16..64` and the linear `N*K` scaling both pointed at dequantization.
- For a byte-to-halfword expansion, wide coalesced `uint4` loads/stores plus a
  2D grid that removes per-element integer division recovered about 1.55x and
  lifted effective bandwidth from about 28% to about 46% of peak.
- Keep the scalar dequant kernel as a correctness fallback for `K % 16 != 0`; the
  vectorized kernel requires the 16-element alignment that `K % 16 == 0`
  guarantees.
- The decode GEMV already streams weights once per tile and is well below the
  prefill floor, so it was left unchanged.
