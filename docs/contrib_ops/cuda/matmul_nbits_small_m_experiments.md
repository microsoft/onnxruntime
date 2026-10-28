# MatMulNBits Small-M GEMV Profiling Experiments

This file records CUDA `MatMulNBits` 4-bit (M = 2..16) and 8-bit small-M
profiling results. The historical 8-bit baseline covered M = 2..5; the
general 8-bit dispatch still covers M = 2..5, with M = 6..8 enabled only
for the exact SM121 FP16 configuration qualified below. These measurements
provide stable comparison points for future kernel and dispatch changes. The
last section covers the fpA_intB (CUTLASS weight-only) GEMV `M = 8` column-tile
dispatch rather than the generic small-M GEMV.

> **Note**: These are **point-in-time** measurements captured on the specific GPU, driver, CUDA
> toolkit, and ORT build noted in the section header. Treat the numbers as a historical baseline
> for regression comparison, not as current performance guidance — re-run the benchmark script on
> your own hardware before drawing conclusions.

## 2026-06-30 Baseline: A100 (SM80), Warmup 25, Repeat 200

### Setup

- Machine/GPU: A100 (SM80).
- CUDA toolkit: 13.0. ONNX Runtime build: `build/Release`.
- Benchmark script: `onnxruntime/test/python/transformers/profile_matmul_nbits.py`.
- Warmup: 25 ORT runs before the measured `benchmark` NVTX range. Repeat: best of 10 trials x 200 runs.
- Weights: 4-bit, block_size 32, asymmetric (with zero points), per the standard `[N, blocks, blob]` layout.
- Matrices sized after a Qwen3-8B-class dense decoder + lm_head.

Command template:

```bash
# Host-timing table across all matrices:
python onnxruntime/test/python/transformers/profile_matmul_nbits.py --warmup 25 --repeat 200

# Single case:
python onnxruntime/test/python/transformers/profile_matmul_nbits.py --k 4096 --n 12288 --m 8

# Kernel-level via nsys + the repo parser:
nsys profile -t cuda,nvtx -o mnb --export=sqlite \
    python onnxruntime/test/python/transformers/profile_matmul_nbits.py --k 4096 --n 12288 --m 8
python onnxruntime/test/python/transformers/parse_nsys.py mnb.sqlite --nvtx-range benchmark --pattern '%'
```

### Before / After (4-bit)

`before` is the prior dispatch (M=1 single-row GEMV; M>1 falls back to weight dequantization + cuBLAS
GEMM, which is M-independent and dequantizes the full weight regardless of M). `after` is the batched
small-M GEMV (`MatMulFloat4BatchedKernel`) covering M=2..16. Values are average op latency in microseconds
(lower is better).

Before (dequant + cuBLAS for M>1):

| matrix  | K     | N      | M=1   | M=2    | M=4    | M=8    | M=16   |
|---------|-------|--------|-------|--------|--------|--------|--------|
| qkv     | 4096  | 4096   | 25.9  | 77.2   | 69.3   | 69.5   | 69.9   |
| o_proj  | 4096  | 4096   | 23.2  | 69.2   | 69.1   | 69.3   | 69.7   |
| gate_up | 4096  | 12288  | 39.3  | 172.0  | 172.1  | 172.1  | 172.4  |
| down    | 12288 | 4096   | 38.8  | 174.8  | 175.1  | 175.4  | 175.6  |
| lm_head | 4096  | 151936 | 301.6 | 1868.0 | 1871.1 | 1877.8 | 1885.1 |

After (batched small-M GEMV for M=2..16):

| matrix  | K     | N      | M=1   | M=2   | M=4   | M=8   | M=16   |
|---------|-------|--------|-------|-------|-------|-------|--------|
| qkv     | 4096  | 4096   | 25.4  | 30.1  | 32.3  | 43.2  | 70.0   |
| o_proj  | 4096  | 4096   | 22.6  | 26.4  | 28.1  | 36.7  | 58.7   |
| gate_up | 4096  | 12288  | 37.5  | 43.5  | 49.5  | 71.9  | 125.9  |
| down    | 12288 | 4096   | 37.9  | 47.4  | 52.9  | 76.8  | 150.1  |
| lm_head | 4096  | 151936 | 300.7 | 329.9 | 424.1 | 635.3 | 1226.9 |

Speedup (before / after, >1 means the batched GEMV is faster):

| matrix  | M=2   | M=4   | M=8   | M=16  |
|---------|-------|-------|-------|-------|
| qkv     | 2.56x | 2.15x | 1.61x | 1.00x |
| o_proj  | 2.62x | 2.46x | 1.89x | 1.19x |
| gate_up | 3.95x | 3.48x | 2.39x | 1.37x |
| down    | 3.69x | 3.31x | 2.28x | 1.17x |
| lm_head | 5.66x | 4.41x | 2.96x | 1.54x |

### Before / After (8-bit)

At the time of this baseline, the 8-bit batched GEMV (`MatMulFloat8bKernelBatched`) covered M=2..5.
8-bit weights are twice the bytes of 4-bit and the GEMV runs on CUDA cores, so this experiment kept
M>=6 on the dequantize + cuBLAS (tensor-core) fallback. Values are average op latency in microseconds.

Before (dequant + cuBLAS for M>1):

| matrix  | K     | N      | M=1   | M=2    | M=3    | M=4    | M=5    |
|---------|-------|--------|-------|--------|--------|--------|--------|
| qkv     | 4096  | 4096   | 36.2  | 80.6   | 72.9   | 72.8   | 73.3   |
| o_proj  | 4096  | 4096   | 31.1  | 73.2   | 72.7   | 73.6   | 73.4   |
| gate_up | 4096  | 12288  | 63.5  | 184.4  | 184.2  | 184.1  | 184.3  |
| down    | 12288 | 4096   | 67.8  | 187.3  | 187.4  | 187.8  | 188.0  |
| lm_head | 4096  | 151936 | 535.9 | 2025.0 | 2025.9 | 2028.1 | 2029.8 |

After (batched small-M GEMV for M=2..5):

| matrix  | K     | N      | M=1   | M=2   | M=3   | M=4    | M=5    |
|---------|-------|--------|-------|-------|-------|--------|--------|
| qkv     | 4096  | 4096   | 36.2  | 46.1  | 48.7  | 57.9   | 67.0   |
| o_proj  | 4096  | 4096   | 31.6  | 39.5  | 48.6  | 57.9   | 67.1   |
| gate_up | 4096  | 12288  | 63.4  | 80.9  | 104.6 | 128.9  | 152.6  |
| down    | 12288 | 4096   | 68.0  | 96.2  | 119.3 | 146.4  | 172.0  |
| lm_head | 4096  | 151936 | 536.3 | 647.0 | 896.9 | 1157.1 | 1420.1 |

Speedup (before / after, >1 means the batched GEMV is faster):

| matrix  | M=2   | M=3   | M=4   | M=5   |
|---------|-------|-------|-------|-------|
| qkv     | 1.75x | 1.50x | 1.26x | 1.09x |
| o_proj  | 1.85x | 1.50x | 1.27x | 1.09x |
| gate_up | 2.28x | 1.76x | 1.43x | 1.21x |
| down    | 1.95x | 1.57x | 1.28x | 1.09x |
| lm_head | 3.13x | 2.26x | 1.75x | 1.43x |

### Observations

- The previous M>1 path dequantizes the entire weight matrix to a temporary buffer and calls cuBLAS, so
  its latency is flat across M (e.g. `gate_up` ~172 us, `lm_head` ~1.9 ms even at M=2). The batched small-M
  GEMV reads the quantized weight once and scales with M, giving 2.6-5.7x at M=2 (4-bit) / 1.8-3.1x at M=2
  (8-bit).
- 4-bit stays at or above parity through M=16. The 8-bit path keeps the M<=5 cutoff for all devices
  and shapes except the exact SM121 FP16 configuration qualified below.
- M=1 decode is unchanged (same single-row GEMV in both builds).
- No prepacking is used, so there is no extra resident weight memory and no GEMM tactic profiling at
  session init.

## 2026-09-21 SM121 W8 M=8 Dispatch Validation

This measurement used FP16 activations, W8 block size 64, M=8, K=3584, and N=200064.

| Path | Average op latency |
|------|-------------------:|
| Dequantize + GEMM fallback | 15.257 ms |
| Batched small-M GEMV | 5.846 ms |

The batched path was **2.61x faster** for this shape. The W8 small-M dispatch is extended through
M=8 only on SM121 for FP16 activations with K=3584, N=200064, and block size 64. All other devices,
types, and shapes retain the existing M<=5 cutoff. M>=9 always uses the fallback.
This is a shape- and device-specific result rather than a general crossover claim for every GPU.

### Next Experiments

- Sweep block_size {16, 32, 64, 128} and bf16 activations.
- Add kernel-level (nsys) breakdowns to separate compute from launch/dispatch overhead.

## 2026-10-27 A100 fpA_intB GEMV M=8 Tile Dispatch Validation

This section covers the CUTLASS weight-only (fpA_intB) GEMV, not the generic small-M GEMV measured
above. It compares the `M = 8` narrow column tile (`CtaN = 2`) with the default tile (`CtaN = 4`) on an
A100 to qualify the dispatch described in §6.4 of [matmul_nbits.md](./matmul_nbits.md).

### Setup

- Machine/GPU: A100-SXM4-80GB (SM80, 108 SMs, 40 MiB L2), driver 580.159.04, CUDA 13.0 (nvcc 13.0.48),
  GCC 13.2, `-O3`, `sm_80`.
- Kernels: the production `fpA_intB_gemv` kernel template instantiated directly (not an ORT session)
  with M = 8, 128 threads, `CtaN` = 2 and 4, scale-only SM80-interleaved weights (no activation scale,
  zero points, bias, or alpha-in-advance), FP16 and BF16 activations, INT4 and INT8 weights.
- Shapes: `block_size` 32 at K = 1024 and 4096, plus spot checks at K = 8192 and `block_size` 64 / 128;
  1,216 unique dtype x bits x N x K x group x residency conditions in total.
- Residency: *cache* reuses one weight buffer (0.13-34 MiB, L2-resident); *rotating* cycles through
  distinct weight buffers (256-276 MiB, more than 6.4x the L2) so the weights stream from DRAM.
- Timing: CUDA graph replay of at least 128 kernels per case. A case reports the median of 11
  batch-average per-kernel latencies; a condition reports the median of at least 3 independent rounds
  (5 for the main sweep; separate processes, shuffled shape and tile order). Decisions use the paired
  per-round ratio. Both tiles run on identical buffers and produce bit-identical outputs (0 mismatches,
  no non-finite values).

### Results (K = 4096, `block_size` 32, rotating weights)

Median per-kernel latency in microseconds, narrow (`CtaN = 2`) / default (`CtaN = 4`), with the median
paired-round ratio narrow / default in parentheses (below 1 favors the narrow tile).

| Weights | N | FP16 | BF16 |
|---------|-----:|-------------------------|-------------------------|
| INT4 | 1024 | 12.912 / 11.754 (1.097) | 13.069 / 11.834 (1.105) |
| INT4 | 2048 | 13.743 / 13.755 (1.000) | 13.881 / 13.995 (0.992) |
| INT4 | 2880 | 15.223 / 14.411 (1.056) | 15.235 / 14.667 (1.039) |
| INT4 | 4096 | 17.168 / 18.587 (0.924) | 17.168 / 18.361 (0.935) |
| INT4 | 4320 | 17.522 / 18.705 (0.936) | 17.514 / 18.462 (0.949) |
| INT4 | 4336 | 23.529 / 18.789 (1.252) | 23.681 / 18.500 (1.280) |
| INT4 | 4352 | 24.902 / 18.675 (1.333) | 25.008 / 18.485 (1.353) |
| INT4 | 5120 | 29.874 / 19.152 (1.560) | 29.807 / 18.833 (1.581) |
| INT8 | 640 | 9.802 / 9.359 (1.044) | 9.981 / 10.087 (0.989) |
| INT8 | 1024 | 10.335 / 10.800 (0.957) | 10.721 / 11.281 (0.951) |
| INT8 | 1280 | 10.818 / 11.243 (0.960) | 11.229 / 11.759 (0.955) |
| INT8 | 1536 | 11.557 / 11.532 (1.002) | 12.151 / 12.101 (1.005) |
| INT8 | 2560 | 14.295 / 13.872 (1.031) | 15.271 / 14.896 (1.025) |

### Findings

- INT4 narrow wins for `3456 < N <= 4320` at every tested K (up to 19% faster at N = 4320, K = 8192) and
  loses from N = 4336: the narrow tile fits five resident blocks per SM, so N = 4320 launches exactly
  108 x 5 = 540 blocks and larger N starts a second wave. It is 4-5% slower at N = 4336, K = 1024 and up to
  58% slower at N = 5120, K = 4096. The previous grid.y cutoff (`N <= 5120`) selected the narrow tile across
  that whole range on this device.
- INT4 narrow also wins for `1728 < N <= 2560` at K <= 3072: the default tile is up to 8.4% slower at
  K = 1024 and 4.9% slower at K = 3072. At K = 4096 the result is mixed (paired ratio 0.968-1.022), so that
  window is limited to K <= 3072 (validated at K = 1024, 2048, and 3072 only). N = 1024 and N = 2880 favor
  the default tile at K = 4096.
- INT8 narrow wins for `864 < N <= 1304` (4-5% at N = 1024-1280, K = 4096; up to 9% at N = 872, K = 1024).
  Both edges are soft: N = 864 is mixed (paired ratio 0.956-1.032 at K = 4096; the default tile loses up to
  4.6% in one setting), N = 1312 still favors the narrow tile in some settings (excluding it costs up to
  5.8%), and at K = 8192 the benefit is marginal or reverses. N = 1376 shows a clear cache-resident
  K = 8192 loss (up to 5.2% for FP16), which argues against a higher cap.
- The dispatcher applies the windows only to scale-only launches (no activation scale, bias, or
  alpha-in-advance) with `block_size` 32, 64, or 128; other A100 launches keep the default tile.

### Policy audit

Applying the implemented windows to all 1,216 conditions: 428 conditions (1,736 paired observations)
select the narrow tile. The worst regret of a selected narrow tile is 1.95% (cache, BF16/INT8, N = 1304,
K = 8192, `block_size` 32) and none exceeds 3%.

The conservative windows leave measured wins unclaimed, so they are not universally optimal. Choosing the
default tile at very small N loses up to 6.5% at INT4 N = 256 (4.0% at N = 512) and up to 9.1% at
INT8 N = 128 (INT8 N = 256 narrow is not uniformly better). A third tile (`CtaN = 8`) beats the better of
`CtaN` 2 and 4 at some large N (rotating BF16/INT4, N = 6144, K = 8192: 47.1 us versus 55.9 us); it is
outside these windows and is not enabled.

### Production check

86 representative launches x 3 rounds (258 debug tile choices) through the actual dispatcher header
matched the windows for all four FP16/BF16 x INT4/INT8 combinations, and production, forced-`CtaN = 2`,
and forced-`CtaN = 4` outputs were bit-identical. `M = 7` and `M = 9` keep `CtaN = 4`, and a
bias-enabled BF16/INT4 launch at N = 4320 keeps `CtaN = 4`. The caller reports the device in
`Params::device_sm` and `Params::device_sm_count`; the dispatcher makes no CUDA runtime call (0
`cudaGetDevice` / `cudaDeviceGetAttribute` calls over cold, 1,000 warm, `M = 7` / `M = 9`, and bias-enabled
launches), and a launch that reports no device gets the device-independent cutoffs. Warm CPU enqueue time of
the dispatched launch exceeds a forced launch of the same tile by 0.003-0.022 us (0.1-0.8%; median paired
delta per case, 750 paired trials over 10 cases) and device time is unchanged (median paired ratio
0.9994-1.0005). An earlier revision that queried the device inside the dispatcher (`cudaGetDevice` plus
per-thread cached attributes) added 0.08-0.11 us (about 3%) per warm launch, so the query was moved out of
the launch path.

### Caveats

- Only an A100-SXM4-80GB was measured. A100 40 GB and PCIe parts share SM80, 108 SMs, and a 40 MiB L2,
  so they use the same windows, but they were not measured separately.
- Other devices keep the H200-tuned grid.y cutoffs. H200 data cover FP16/INT4 and BF16/INT8 only;
  FP16/INT8 and BF16/INT4 on non-A100 devices are unmeasured.
- Kernel-level timings with synthetic weights and activations; no end-to-end ORT latency was measured.
