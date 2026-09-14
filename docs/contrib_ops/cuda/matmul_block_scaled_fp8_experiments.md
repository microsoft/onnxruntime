# MatMulBlockQuantizedFp8Weight - CUDA Experiments

This document records CUDA performance experiments for
**MatMulBlockQuantizedFp8Weight** (`com.microsoft::MatMulBlockQuantizedFp8Weight`).
It includes retained optimizations and their measured effect so future tuning
does not repeat work whose results are already known.

Related documentation:

- [matmul_block_scaled_fp8.md](matmul_block_scaled_fp8.md) - operator behavior and current dispatch chain.

---

## Table of Contents

1. [Test Environment](#1-test-environment)
2. [Baseline Latency Profile](#2-baseline-latency-profile)
3. [Prefill Bottleneck - Weight Dequantization](#3-prefill-bottleneck---weight-dequantization)
4. [Optimization - Vectorized Dequantization Kernel](#4-optimization---vectorized-dequantization-kernel)
5. [Decode GEMV - Memory-Level Parallelism](#5-decode-gemv---memory-level-parallelism)
6. [Decode GEMV - Tensor Cores](#6-decode-gemv---tensor-cores)
  - [RTX 4090 Split-K Qualification and Dispatch Refinement](#65-rtx-4090-split-k-qualification-and-dispatch-refinement)
  - [RTX 5060 Ti and RTX 3060 Split-K Validation](#66-rtx-5060-ti-and-rtx-3060-split-k-validation)
7. [Benchmark Commands](#7-benchmark-commands)
8. [Lessons](#8-lessons)

---

## 1. Test Environment

- GPU: NVIDIA GeForce RTX 5060 Ti, SM120 (Blackwell), 36 SMs, about 448 GB/s memory bandwidth.
  Section 5 was measured separately on an NVIDIA H200, SM90 (Hopper), 132 SMs, about 4.8 TB/s.
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

## 5. Decode GEMV - Memory-Level Parallelism

An earlier round of this document concluded the decode GEMV was "already well
tuned and not the optimization target". That conclusion came from measurements
taken through the ORT Python API without CUDA graphs, which on a fast GPU are
dominated by per-node host overhead rather than by the kernel. Re-measuring on
H200 (SM90) with the launches captured in a CUDA graph changed the picture.

### 5.1 What the measurement was actually reporting

At `N = 8192, K = 2048, M = 1` the op-level measurement reported 21.0 us while
the kernel itself takes 7.2 us. The difference is ORT per-node host work; the
host, not the GPU, was the limiter in that harness. Two rules follow:

- Measure kernels standalone, or with the launches captured in a CUDA graph.
- On H200 an empty kernel costs **1.79 us** as a stream launch and **0.68 us**
  as a CUDA graph node. That 0.68 us is a floor no kernel optimization can go
  below, and it dominates any op whose useful work is smaller.

### 5.2 Nsight Compute diagnosis

`ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy
--section WarpStateStats` on the original kernel at `N = 8192, K = 2048, M = 1`:

| Metric | Value | Reading |
|---|---|---|
| DRAM throughput | 35.3% | not bandwidth bound |
| Compute (SM) throughput | 52.3% | not compute bound either |
| Waves | 1.0 | grid barely fills the GPU once |
| L1/TEX hit rate | 79.6% | the `A` row is already resident in L1 |
| L2 hit rate | 5.2% | `A` re-reads never reach L2 |
| Block limit (registers) | 8 blocks/SM | occupancy already register capped |
| Achieved occupancy | 71.0% | vs 100% theoretical |
| Warp cycles / issued instr | 16.7, of which 5.5 on L1TEX | latency bound |

The kernel is short of **outstanding loads**, not of bandwidth or instructions.
Each thread moves only `K / 32 = 64` bytes of `B`, and because `k` is a runtime
value the K loop does not unroll, so a thread has exactly one `B` load in flight
and pays full L1 latency every iteration.

Two hypotheses were tested and rejected:

- *Reduce conversion instructions.* Replacing the 16 scalar `static_cast<float>`
  FP8 converts with `__nv_cvt_fp8x2_to_halfraw2` (one `cvt.rn.f16x2.e4m3x2` per
  pair) is bit-exact and worth only 0-15%. Instruction count was not the limit.
- *Stage `A` in shared memory.* The concern was that all `N` warps re-read the
  whole `A` row. The 79.6% L1 hit rate shows L1 already absorbs this; a shared
  memory variant was neutral to slower except at very small `N`.

### 5.3 Change

`MatMulBlockScaledFp8GemvKernel` is now templated on
`<RowsPerWarp, ColsPerWarp, Unroll, AType>`, where `<R, 1, 1, A>` reproduces the
original geometry exactly:

- `Unroll` pre-issues `Unroll` independent `B`/`A` loads before consuming any of
  them, so several requests are in flight per thread.
- `ColsPerWarp` gives one warp several output columns, so each `A` load feeds
  several independent FMA chains.

Both trade occupancy - already register capped, and irrelevant at one wave - for
per-thread memory-level parallelism. The FP8 to FP16 conversion is also
vectorized. FP32 accumulation is unchanged, so results are **bit-identical** to
the previous kernel.

Dispatch (only `M == 1`, the batch-1 decode case, uses wide tiles):

| Condition | Config |
|---|---|
| `M == 1, N >= 8192` | `<1, 4, 2>` |
| `M == 1, N >= 4096` | `<1, 2, 2>` |
| otherwise | `<RowsPerWarp, 1, 1>` (unchanged) |

Below `N = 4096` the wider tiles leave too few warps to fill the GPU, and for
`M > 1` the extra live registers (accumulators plus pre-issued loads) cost more
than the added parallelism returns. Both measured slower, hence the guards.

> Superseded for `M > 1` by section 6.4, which re-tunes `ColsPerWarp` / `Unroll`
> for the speculative-decode tiles. Because `Unroll` changes the K chunk each
> lane accumulates first, the `M > 1` dispatch there is not bit-identical to
> `<RowsPerWarp, 1, 1>` (last-ulp only; the accumulation is still FP32).

### 5.4 Results (H200, `M = 1`, CUDA graph, us, includes 0.68 us node overhead)

| Shape (N x K) | cuBLAS FP16 | GEMV before | GEMV after | vs before | vs cuBLAS |
|---|---|---|---|---|---|
| 8192 x 2048 | 10.2 | 7.2 | **5.4** | 1.33x | 1.89x |
| 4096 x 2048 | 7.1 | 4.4 | **4.0** | 1.10x | 1.78x |
| 4096 x 4096 | 9.2 | 7.1 | **6.0** | 1.18x | 1.53x |
| 2048 x 4096 | 7.6 | 4.6 | 4.6 | 1.00x | 1.65x |
| 512 x 2048 | 5.4 | 2.7 | 2.7 | 1.00x | 2.00x |

At `8192 x 2048` this is 3.1 TB/s of the 4.8 TB/s HBM peak, up from 2.3 TB/s.

Note the last column: the weight-only FP8 GEMV is **1.5-2.0x faster than cuBLAS
FP16** at `M = 1`, so quantizing a projection to FP8 is a decode win on latency
as well as on footprint. At `M >= 4` cuBLAS wins and the GEMV path should not be
preferred on speed alone.

---

## 6. Decode GEMV - Tensor Cores

Section 5 tuned memory-level parallelism at `M = 1`. At `M = 4` - the width of a
speculative-decode / MTP verify forward - the kernel is limited by something
else. With `RowsPerWarp = 4` a lane executes roughly 240 instructions per 32
weight bytes, only 128 of which are the FMAs that do useful work, and effective
bandwidth falls from about 2.35 TB/s at `M = 1` to about 1.25 TB/s at `M = 4`.
More ILP cannot fix that; the dot products have to leave the FMA pipe.

### 6.1 Design

`MatMulBlockScaledFp8MmaGemvKernel` uses `mma.m16n8k16` with FP32 accumulation.
The operand assignment is the key decision:

| mma operand | fed from | why it fits |
|---|---|---|
| `A[16, 16]` row-major | weight `[16 output cols][16 k]` | `B` is `[N, K]` row-major |
| `B[16, 8]` col-major | activation `[16 k][8 rows]` | `A` is `[M, K]` row-major |
| `D[16, 8]` | `y[16 output cols][8 rows]` | |

So the mma "M" extent is the output column count and the mma "N" extent is `M`.
At `M = 4` half the mma N lanes are idle, which is irrelevant: the kernel is
bound by weight traffic and instruction issue, and both improve about 10x per
weight byte.

The naive fragment load is badly coalesced - a lane needs bytes
`{2t, 2t+1, 2t+8, 2t+9}` of a row, which spreads a warp across 16 rows x 16
bytes and over-fetches every 32-byte sector 2x. The fix is to **permute the K
axis**. K is a reduction axis, so any permutation applied to *both* operands
leaves the result unchanged. Inside a 64-element K window the permutation used is

```
mma k-slot (of step j)  ->  actual k
  2t,   2t+1                 16t + 4j,     16t + 4j + 1
  2t+8, 2t+9                 16t + 4j + 2, 16t + 4j + 3
```

so lane `(g = lane >> 2, t = lane & 3)` loads one contiguous `uint4` of weight
bytes `[16t, 16t + 16)` and the matching 32 activation bytes, four lanes cover 64
contiguous bytes of one weight row, and that single `uint4` feeds all four mma
steps.

16 columns per warp gives about 8x fewer warps than the FMA kernel, which alone
costs more in lost memory-level parallelism than the instruction saving is worth.
`KSplit` warps per block therefore take a strided share of the K windows and are
reduced through shared memory at the end. The generic policy uses `KSplit = 8`
for `N >= 8192` and 16 otherwise, subject to the short-K window clamp. Three
qualified low-M configurations select 8 earlier:

- SM90 with 132 SMs (measured on H200), `M <= 8`: more than `2 * sm_count`
  output blocks.
- SM89 with 128 SMs (measured on RTX 4090), `M <= 8`, `16 <= K/64 <= 96`:
  more than `3 * sm_count` output blocks. This preserves the pinned KS16 window
  and changes only `6144 < N < 8192`, `1024 <= K <= 6144` relative to the
  generic policy.
- SM120 with 36 SMs (measured on RTX 5060 Ti), `M <= 8`, `40 <= K/64 <= 96`:
  more than `2 * sm_count` output blocks.

Other configurations retain the generic policy and the existing SM121 KS32
override. The residency hint remains active where the selected KS16 qualifies.

On H200, plain KSplit 16 is 512 threads at 48 registers and fits two blocks per
SM, while KSplit 8 fits five. The following H200 measurements motivated its
qualified rule (132 SMs, boundary at 264 blocks; boost clocks):

| output blocks | blocks/SM | shipped us | KSplit 8 us | shipped / KSplit 8 |
| --- | --- | --- | --- | --- |
| 262 | 1.985 | 6.592 | 7.296 | 0.904 |
| 264 | 2.000 | 6.784 | 7.328 | 0.928 |
| 265 | 2.008 | 9.072 | 8.288 | **1.096** |
| 320 | 2.424 | 9.296 | 8.640 | **1.076** |
| 384 | 2.909 | 9.456 | 8.720 | **1.083** |
| 429 | 3.250 | 12.000 | 10.624 | **1.130** |
| 495 | 3.750 | 12.496 | 11.392 | **1.098** |
| 512 | 3.879 | 11.744 | 11.712 | 0.996 (already KSplit 8) |

On this H200, the boundary tracks `2 * sm_count` rather than a fixed N, and holds for
`K = 2560`, `5120` and `6144` (windows 40, 80 and 96) and for both `M = 1` and
`M = 8`. At `M = 16` two row tiles cost 72 registers, which drops KSplit 16 to a
single block per SM, and KSplit 8 wins at every width measured (1.23-1.43x); the
selector does not act on that.

The RTX 4090 experiments and the rationale for its narrower qualification are
recorded in [section 6.5](#65-rtx-4090-split-k-qualification-and-dispatch-refinement).

Cross-device validation used CUDA graph replay and Nsight Systems kernel timing.
On an RTX 5060 Ti (SM120, 36 SMs), the crossover lands at 72/73 output blocks and
KSplit 8 improves `M=4, N=5120, K=6144` by 9.5%. On an RTX 3060 (SM86, 28 SMs),
KSplit 8 regresses `M=1, N=897, K=5120` by 12.1% and `N=2048, K=5120` by about
8%, while results at larger widths are mixed. The SM86 configuration therefore
retains the generic policy.

Preconditions: SM80+, `K % 64 == 0`, `K >= 256`, `block_size % 64 == 0`, `M <= 8`.
Otherwise the FMA kernel runs unchanged. `ORT_FP8_GEMV_MMA=0` forces the FMA
kernel for A/B testing in a single binary.

### 6.2 Accuracy

Not bit-identical to the FMA kernel (different summation order), but not less
accurate either. E4M3 to FP16 is lossless, E4M3 to BF16 is lossless, FP16 x FP16
products are exact in FP32, and the mma accumulates in FP32 exactly as the FMA
path does. Scored against an FP64 CPU reference on the shapes below, the maximum
error is *identical* for the two kernels (2-4e-4, i.e. pure FP16 output
rounding).

### 6.3 Results (H200, us, standalone, `fp8_gemv_m4_bench.cu`)

| Shape (N x K) | M | cuBLAS FP16 | FMA kernel | mma kernel | vs FMA | vs cuBLAS |
|---|---|---|---|---|---|---|
| 8192 x 2048 | 1 | 11.0 | 6.3 | **5.1** | 1.23x | 2.16x |
| 4096 x 2048 | 1 | 8.4 | 4.8 | **4.0** | 1.20x | 2.09x |
| 2048 x 4096 | 1 | 9.0 | 5.1 | **4.2** | 1.21x | 2.13x |
| 512 x 2048 | 1 | 6.7 | 3.3 | **3.1** | 1.06x | 2.12x |
| 8192 x 2048 | 4 | 11.0 | 9.8 | **5.2** | 1.87x | 2.10x |
| 4096 x 2048 | 4 | 8.5 | 6.9 | **4.1** | 1.69x | 2.09x |
| 2048 x 4096 | 4 | 8.5 | 8.0 | **4.4** | 1.82x | 1.92x |
| 512 x 2048 | 4 | 6.8 | 4.7 | **3.3** | 1.43x | 2.08x |
| 8192 x 2048 | 8 | 10.9 | 17.5 | **5.7** | 3.09x | 1.93x |
| 2048 x 4096 | 8 | 8.5 | 13.0 | **4.5** | 2.90x | 1.90x |

The mma kernel is faster at every measured `M`, so it is preferred whenever its
preconditions hold rather than only for `M > 1`. Note also that the FMA kernel
crosses over and loses to cuBLAS at `M = 8`, while the mma kernel stays about 1.9x
ahead.

End to end on a 40-layer Qwen3.6-35B-A3B NVFP4 MTP decode (130 FP8 matmul nodes
per step, `M = 4`), CUDA graphs on:

| | FMA kernel | mma kernel |
|---|---|---|
| FP8 GEMV kernel time | 1.052 ms/step | **0.713 ms/step** |
| total kernel time | 7.368 ms/step | **7.021 ms/step** |
| wall | 9.80 ms/step | **9.54 ms/step** |

No other kernel family moved. This optimization is kept.

### 6.4 FMA fallback re-tune for `M > 1`

The FMA kernel still runs when the tensor-core preconditions do not hold (pre-SM80,
`K < 256`, `K % 64 != 0` or `block_size % 64 != 0`), so the `M > 1` tiles were
re-tuned there as well. Widening `A` to FP32 is now hoisted out of the column loop
(one widening per row instead of one per row/column pair), which makes `ColsPerWarp`
profitable at `M > 1` for a second reason beyond memory-level parallelism:

| Condition | Config |
|---|---|
| `2 <= M <= 2, N >= 8192` | `<2, 4, 1>` |
| `2 <= M <= 2, N >= 2048` | `<2, 2, 1>` |
| `2 <= M <= 2` otherwise | `<2, 1, 2>` |
| `3 <= M <= 4, N >= 4096` | `<4, 4, 1>` |
| `3 <= M <= 4, N >= 2048` | `<4, 2, 2>` |
| `3 <= M <= 4` otherwise | `<4, 1, 2>` |
| `M > 4` | `<8, 1, 1>` |

Measured on H200 (us, `M = 4`, versus the previous `<R, 1, 1>` and cuBLAS FP16):

| Shape (N x K) | cuBLAS | `<4, 1, 1>` | tuned |
|---|---|---|---|
| 8192 x 2048 | 10.9 | 13.7 | **9.7** (`<4, 4, 1>`) |
| 4096 x 2048 | 8.3 | 8.1 | **6.9** (`<4, 4, 1>`) |
| 2048 x 4096 | 8.3 | 9.0 | **7.9** (`<4, 2, 2>`) |
| 512 x 2048 | 7.3 | 5.0 | **4.6** (`<4, 1, 2>`) |

The hoisting itself is bit-identical (the per-lane `fmaf` sequence is unchanged),
but a different `Unroll` changes which K chunk a lane accumulates first, so the
re-tuned dispatch is a last-ulp change relative to `<RowsPerWarp, 1, 1>`.

---

### 6.5 RTX 4090 Split-K Qualification and Dispatch Refinement

#### Environment and Method

Measured on September 14, 2026 for [PR #32594](https://github.com/microsoft/onnxruntime/pull/32594):

- RTX 4090, SM89, 128 SMs, 72 MiB L2; Windows WDDM with an active display.
- CUDA 13.3.73, driver 610.60, Visual Studio 18 2026, Release `sm_89` compilation.
- Main baseline `b03fb522be1e1574e86835c9dca84b2b94ef2ca5` versus original PR
  `c2de46090018e9cbcba84e1f7654a9c405ab6476`. Their production device bodies were
  identical; a standalone same-binary harness compared the exact selectors and
  kernel entry points. No full ORT DLL or model was used for these timings.
- FP8 E4M3 weights, FP32 scales, block size 128, FP16/BF16 activations and
  outputs; no bias or activation QDQ. Inputs and buffers were prepared before timing.
- CUDA events measured batches of graph nodes after 50 eager and 10 graph
  warmups. A/B order alternated within paired samples. The initial sweep covered
  188 shape/dtype cases with 31 pairs and 128 nodes per graph; 18 cases received
  three independent confirmation runs with 101 pairs and 256 nodes per graph.
- Follow-up qualification added 168 focused cases and two-run cache-mode
  comparisons, totaling 14,592 additional pairs. It included intermediate
  `M={9,12,17,24,31}`, `K={1024,2560,5120,6144,8192,16384}`, and grid boundaries.
- Reused-weight tests repeatedly accessed the same allocation. Streaming tests
  rotated weight/scale allocations totaling at least three times L2 capacity,
  while reusing activations and outputs. This probes cache sensitivity; it is
  not a cache-flush test or a full-model workload.

Times below are unprofiled, graph-amortized **microseconds per kernel node**.
Speedup is the median paired `main_time / candidate_time`; values above 1 favor
the candidate. Separate latency medians need not divide to that paired statistic.
Clocks were not locked, WDDM and a concurrent CPU build could perturb timing,
and no outliers were discarded.

#### Original Rule: Pinned KS16 Is the Relevant Baseline

The original PR chose KS8 once `ceil(N/16) > 2 * sm_count`. On this GPU that
means `N > 4096`. However, for low M and sufficiently long K, main already used
**pinned KS16** for `4096 < N <= 6144`, rather than plain KS16.

| Type | M | N | K | Main us | Original PR us | Paired speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 8 | 4097 | 5120 | 9.820 | 10.656 | 0.927x |
| BF16 | 8 | 4097 | 5120 | 9.980 | 10.884 | 0.918x |
| FP16 | 1 | 5120 | 6144 | 9.956 | 10.216 | 0.981x |
| FP16 | 8 | 6144 | 5120 | 12.540 | 13.092 | 0.956x |
| FP16 | 8 | 6145 | 5120 | 14.968 | 12.948 | 1.166x |
| FP16 | 16 | 6144 | 5120 | 27.264 | 17.744 | 1.543x |

The first FP16 regression persisted across three runs (0.910-0.939x run-median
speedups). Nsight Systems confirmed pinned KS16 versus plain KS8, with traced
kernel medians of 10.208 versus 11.232 us and similar inter-node gaps. Traced
timings are diagnostic and are not mixed into the table's unprofiled results.

Pinned KS16 used 40 registers/thread and could theoretically host three blocks
per SM; plain KS8 used 48 registers/thread and could host five. Thus main already
fit the boundary grid in one residency round. The unhinted-KS16 occupancy argument
does not justify replacing this path. Actual achieved occupancy and bandwidth
were not measured because performance-counter access was denied.

#### Streaming Weights Reject a Broad Larger-M Rule

A candidate preserving the low-M pinned window but selecting KS8 above two
blocks per SM for all larger M passed the original sweep. Additional testing
found that some reused-weight gains reverse with streaming weights:

| Type | M | N | K | Reused speedup | Streaming speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 9 | 7168 | 5120 | 1.161x | 0.966x |
| BF16 | 12 | 7168 | 6144 | 1.245x | 0.966x |
| BF16 | 9 | 7168 | 8192 | 1.125x | 0.930x |
| BF16 | 9 | 7168 | 16384 | 0.952x | 0.945x |

These results reject both an unrestricted larger-M rule and a K<=8192 cap.
Low-M long-K reused-weight cases also showed small losses, so the retained rule
is bounded in both M and K. Some larger-M gains remain real but are deferred
rather than adding another partially qualified dispatch region.

#### Retained Region and Dispatch Rules

Representative FP16 results within the retained low-M region:

| M | N | K | Reused main / KS8 us | Reused speedup | Streaming main / KS8 us | Streaming speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7168 | 1024 | 4.023 / 3.448 | 1.166x | 9.946 / 9.585 | 1.038x |
| 8 | 7168 | 2560 | 9.472 / 7.981 | 1.178x | 21.790 / 21.651 | 1.005x |
| 1 | 7168 | 6144 | 13.867 / 13.048 | 1.064x | 49.699 / 49.725 | 1.002x |
| 8 | 7168 | 6144 | 19.908 / 17.568 | 1.154x | 49.798 / 49.809 | 0.998x |

Across both dtypes, the retained original sweep cases gave 1.064-1.194x.
Selected streaming confirmations gave 0.998-1.038x, with no qualified loss above
3%. Qualification required a paired 95% bootstrap interval entirely beyond
1.03 for gains or below 1/1.03 for losses; this does not prove every unmeasured
shape in the region improves.

The refined policy in
[`matmul_block_scaled_fp8_tiling.h`](../../../onnxruntime/contrib_ops/cuda/math/matmul_block_scaled_fp8_tiling.h)
uses `output_blocks = ceil(N/16)` and `windows = K/64`:

| Configuration | Rule before the short-K clamp |
| --- | --- |
| Default, including unqualified devices and larger M | KS8 if `N >= 8192`, otherwise KS16 |
| SM90, 132 SMs, `M <= 8` | Select KS8 above `2 * sm_count` output blocks |
| SM89, 128 SMs, `M <= 8`, `16 <= windows <= 96` | Select KS8 above `3 * sm_count` output blocks |

The SM89 change relative to main is therefore limited to `6144 < N < 8192`,
`M <= 8`, and `1024 <= K <= 6144`, subject to the MMA path's existing alignment
requirements. `N=6144` keeps pinned KS16; `N=6145` is the first changed width.
M>8 and K>6144 retain main's policy. Architecture and SM-count checks qualify
configurations, not GPU marketing names; other devices sharing those properties
will also match.

After selection, the existing window clamp selects KS8 or KS4 when the reduction
cannot feed the requested split. The SM121 KS32 override remains unchanged, as
does the residency predicate: selected KS16 still uses the hinted entry point
when eligible. H200's low-M rule is retained from the earlier evidence, not
remeasured on this machine. No runtime autotuning was introduced.

All new timed cases passed the independent FP64-reference check. Final validation
against the edited production header passed 192 FP16/BF16 GPU correctness/dispatch
cases and the extracted host selector/residency-boundary tests. The production
CUDA translation unit also compiled separately. These checks do not establish
full-provider test coverage or end-to-end model speedup.

### 6.6 RTX 5060 Ti and RTX 3060 Split-K Validation

#### Environment and Method

Measured on September 14, 2026 while evaluating the Split-K dispatch change:

- RTX 5060 Ti, SM120, 36 SMs, about 448 GB/s memory bandwidth.
- RTX 3060, SM86, 28 SMs.
- CUDA 13.0, Visual Studio 2022, Release build containing both KSplit 8 and
  KSplit 16 kernel instantiations.
- FP8 E4M3 weights, FP32 scales, block size 128, and FP16 activations and
  outputs. The focused cases used no bias or activation QDQ.
- A NumPy/ORT harness allocated CUDA `OrtValue`s, captured the operator in a
  CUDA graph, warmed it up, and replayed it hundreds of times. Nsight Systems
  2026.3.2 CUDA graph-node traces supplied kernel durations. The final A/B used
  one freshly built binary and `ORT_FP8_GEMV_KSPLIT` to force each choice, which
  avoids compiler or binary differences between the two arms.
- The final fresh-binary cases ran an exact output check after replay. Separate
  selector tests checked the default route, including the short-K clamp and
  device qualification.

The numbers below are median microseconds per CUDA graph kernel node. Speedup is
`KSplit 16 / KSplit 8`, so values above 1 favor KSplit 8.

#### RTX 5060 Ti: Residency Boundary Holds

With 36 SMs, two output blocks per SM is 72 blocks. Since the MMA kernel emits
one block per 16 output columns, the boundary lies between `N=1152` and the next
block at `N=1153..1168`. An exploratory same-binary forced-kernel sweep at the
full-block endpoints showed a sharp crossover:

| M | N | K | output blocks | KSplit 16 | KSplit 8 | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1152 | 5120 | 72 | **6.720 us** | 7.583 us | 0.886x |
| 8 | 1168 | 5120 | 73 | 9.248 us | **8.288 us** | 1.116x |

The exploratory sweep also found that the first block above the boundary favored
KSplit 8 at `M=1` over each tested reduction length:

| M | N | K | KSplit 16 | KSplit 8 | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1168 | 2560 | 4.512 us | **3.904 us** | 1.156x |
| 1 | 1168 | 5120 | 7.104 us | **6.624 us** | 1.072x |
| 1 | 1168 | 6144 | 8.096 us | **7.680 us** | 1.054x |

The final fresh-binary confirmation used a model projection shape. KSplit 8
reduced the median from 28.512 us to 26.047 us, a 1.095x speedup:

| M | N | K | KSplit 16 | KSplit 8 | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 5120 | 6144 | 28.512 us | **26.047 us** | **1.095x** |

Nsight reported 54 registers per thread for both variants in this Windows
SM120 build. A KSplit 16 block has 512 threads and fits two blocks per SM by the
register limit, while a KSplit 8 block has 256 threads and fits four. The exact
72/73-block timing discontinuity, rather than an assumed cross-architecture
register count, is the evidence for the retained `2 * sm_count` boundary.

#### RTX 3060: Residency Alone Does Not Predict the Choice

The analogous two-block boundary on the 28-SM RTX 3060 is 56 output blocks.
The final fresh-binary test at the first ragged width in block 57 showed the
opposite result from SM120:

| M | N | K | output blocks | KSplit 16 | KSplit 8 | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 897 | 5120 | 57 | **42.879 us** | 48.063 us | **0.892x** |

KSplit 8 was therefore 12.1% slower at the exact proposed dispatch boundary.
An exploratory reduction sweep showed that the result also depends on K: at the
nearby full-block width `N=912`, KSplit 8 slightly won for `K=2560` but lost for
the longer reductions.

| M | N | K | KSplit 16 | KSplit 8 | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 912 | 2560 | 20.320 us | **20.064 us** | 1.013x |
| 1 | 912 | 5120 | **37.600 us** | 38.688 us | 0.972x |
| 1 | 912 | 6144 | **44.480 us** | 45.760 us | 0.972x |

Nor was there a monotonic width threshold. At `M=1, K=5120`, KSplit 8 was about
8% slower at `N=2048`, slightly faster at `N=3072`, and approximately tied again
at `N=4096`. At larger `M=8` model projections it produced modest gains: 1.022x
at `N=5120, K=6144` and 1.035x at `N=6144, K=5120`. A single occupancy-derived
rule would therefore trade regressions in some decode shapes for gains in others.

Nsight reported 56 registers per thread for both plain variants on SM86, again
giving approximately two resident KSplit 16 blocks versus four KSplit 8 blocks
by the register limit. SM86 does not have native FP8 tensor-core instructions,
but that fact is not by itself the explanation: this kernel converts E4M3
weights to FP16/BF16 fragments and issues FP16/BF16 MMA instructions on every
supported architecture. Differences in memory behavior, scheduling, and the
cost of the split reduction still make the best KSplit architecture- and
shape-dependent.

#### Dispatch Decision

The retained selector consequently qualifies the measured RTX 5060 Ti
configuration (`SM120`, 36 SMs, `M <= 8`, and `40 <= K/64 <= 96`) for KSplit 8
above `2 * sm_count` output blocks. The RTX 3060 and other unqualified devices
keep the legacy `N >= 8192` crossover. The final default-route traces confirmed
KSplit 8 on the RTX 5060 Ti boundary case and KSplit 16 on the RTX 3060 boundary
case. These results do not justify extending either decision to unmeasured RTX
40- or RTX 50-series configurations solely from compute capability.

---

## 7. Benchmark Commands

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

## 8. Lessons

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
- Never benchmark a fast kernel through the ORT Python API without CUDA graphs.
  At `8192 x 2048, M = 1` that harness reported 21.0 us for a 7.2 us kernel; the
  measurement was host bound and led to the wrong conclusion that the decode
  GEMV was fine and that FP8 was slower than FP16.
- A decode GEMV runs one wave and is usually starved of *outstanding loads*, not
  of bandwidth or instructions. When SOL shows both DRAM and SM well under 60%
  with a large L1TEX stall share, add per-thread memory-level parallelism
  (unroll to pre-issue loads, widen the tile) rather than cutting instructions
  or adding shared memory staging. Trading register-capped occupancy for ILP is
  the right move at one wave.
- Do not pass an array by reference (`__half2 (&)[8]`) to a `__device__` helper.
  It is placed in local memory; inlining the same code via a macro was about 2x
  faster here and much more at `RowsPerWarp > 1`.
- Know the launch floor before optimizing: on H200 an empty kernel costs 0.68 us
  as a CUDA graph node. Ops cheaper than that are launch bound and should be
  fused, not tuned.
- The FP8 weight-only GEMV is 1.5-2.0x faster than cuBLAS FP16 at `M = 1`, so
  quantizing a projection is a decode latency win, not just a footprint win. The
  ordering reverses by `M = 4`.
