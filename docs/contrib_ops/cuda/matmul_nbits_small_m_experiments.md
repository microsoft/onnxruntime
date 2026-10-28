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

The opt-in wave-aware tile (`ep.cuda.fpa_intb_gemv_wave_aware`; FP16/INT4, `M = 8`, sm_12x) had the same
per-launch query in `GemvPickSmCount()` and now reads the same `Params` fields. 52 launches (13 N values x 4
reported devices) ran on the A100 with the dispatcher's debug output captured. With a reported SM 12.0 /
170-SM device and the flag set, the dispatcher chose `CtaN = 2` at N = 5184, 8192, 10240, and 16384 and
`CtaN = 4` at N = 5760, 6144, and 12288. The same launches kept the non-wave-aware rule with the flag off,
with no reported device, or with the A100's own values. Every choice matched an independent restatement of the
policy, and production, forced-`CtaN = 2`, and forced-`CtaN = 4` outputs were bit-identical. The dispatcher
made 0 runtime calls over cold and 1,000 warm launches; the previous header made one `cudaGetDevice` call per
warm launch and ignored the reported device. With the flag set, warm CPU enqueue time exceeded a forced launch
of the same tile by 0.016-0.021 us (0.5-0.7%) after the change, versus 0.106-0.112 us (3.4-3.6%) before it,
and 0.019 us with the flag off in both builds (median paired deltas, 5 rounds x 15 trials x 256 launches per
case). These runs check the dispatch decision and its host cost on an A100 only; no sm_12x device was
available, so the speed-up of the wave-aware tile itself was not re-measured.

### Caveats

- Only an A100-SXM4-80GB was measured. A100 40 GB and PCIe parts share SM80, 108 SMs, and a 40 MiB L2,
  so they use the same windows, but they were not measured separately.
- Other devices keep the device-independent grid.y cutoffs of the first H200 pass (FP16/INT4 and
  BF16/INT8 only). The H200 now has its own rule, measured for all four combinations (next section);
  FP16/INT8 and BF16/INT4 on every other device are unmeasured.
- Kernel-level timings with synthetic weights and activations; no end-to-end ORT latency was measured.

## 2026-10-28 H200 fpA_intB GEMV M=8 Tile Dispatch Tuning

This section tunes the `M = 8` tile dispatch of §6.4 of [matmul_nbits.md](./matmul_nbits.md) for the
H200, following the A100 study above. It compares the narrow column tile (`CtaN = 2`) with the default
tile (`CtaN = 4`) of the CUTLASS weight-only (fpA_intB) GEMV. Before this change the H200 used the
device-independent `grid.y` cutoffs (INT4 `N / 16 <= 320`, INT8 `N / 8 <= 160`).

### Setup

- Machine/GPU: H200 (SM90, 132 SMs, 60 MiB L2), driver 580.159.04, CUDA 13.0, `-O3`, `sm_90a`.
- Kernels: the production `fpA_intB_gemv` kernel template instantiated directly (not an ORT session) with
  M = 8, 128 threads, `CtaN` = 2 and 4, SM80-interleaved weights without zero points, activation scale, or
  alpha-in-advance, FP16 and BF16 activations, INT4 and INT8 weights, scale-only and with a bias epilogue.
- Shapes, scale-only: `block_size` 32 with N = 32-12288 in steps of 32 and K = 256-31744 (35 values for INT4,
  26 for INT8) in both residency modes (32,486 paired conditions); `block_size` 64 and 128 with N = 32-6144 and
  eight values of K between 1024 and 17408 (6,144 conditions each, rotating weights only).
- Shapes, bias: N = 32-6144 at K = 1024, 5120, 13312, and 17408, plus ten N between 4256 and 5280 at 13-14 values
  of K between 1024 and 17408 (`block_size` 32 and 128: 3,272 conditions each); `block_size` 64 has the ten N only
  (260 conditions).
- Residency: *cache* reuses one weight buffer; *rotating* cycles through a pool of more than 6.4x the L2 so the
  weights stream from DRAM, as they do when decode steps through the layers of a model. Decisions use the
  rotating mode.
- Timing: CUDA graph replay of at least 128 kernels per case. A condition reports the minimum per-kernel time of
  each tile over its independent rounds and passes, and the ratio narrow / default of those minima: other work on
  a shared GPU only adds time, so the minimum is the least contaminated estimate. Run-to-run noise is 0.3-0.9%. A
  bias data set that was collected while another job used the GPU showed wide spreads and was discarded and
  re-collected on idle GPUs.

### Results (K = 4096, `block_size` 32, rotating weights)

Minimum per-kernel latency in microseconds, narrow (`CtaN = 2`) / default (`CtaN = 4`), with the ratio narrow /
default in parentheses (below 1 favors the narrow tile).

| Weights | N | FP16 | BF16 |
|---------|-----:|-------------------------|-------------------------|
| INT4 | 1024 | 9.028 / 9.147 (0.987) | 9.068 / 9.185 (0.987) |
| INT4 | 2048 | 9.121 / 9.483 (0.962) | 9.158 / 9.501 (0.964) |
| INT4 | 3072 | 9.518 / 10.054 (0.947) | 9.569 / 10.108 (0.947) |
| INT4 | 4096 | 10.171 / 10.342 (0.983) | 10.248 / 10.398 (0.986) |
| INT4 | 4224 | 10.229 / 10.358 (0.988) | 10.273 / 10.422 (0.986) |
| INT4 | 4256 | 10.657 / 12.398 (0.860) | 10.701 / 12.510 (0.855) |
| INT4 | 5120 | 11.084 / 12.874 (0.861) | 11.130 / 12.961 (0.859) |
| INT4 | 5280 | 11.137 / 12.880 (0.865) | 11.149 / 12.970 (0.860) |
| INT4 | 5312 | 15.558 / 12.914 (1.205) | 15.660 / 12.987 (1.206) |
| INT4 | 6144 | 18.699 / 12.983 (1.440) | 18.797 / 13.095 (1.435) |
| INT8 | 640 | 7.026 / 6.746 (1.041) | 7.669 / 7.281 (1.053) |
| INT8 | 1024 | 7.063 / 6.946 (1.017) | 7.718 / 7.504 (1.028) |
| INT8 | 1280 | 7.255 / 7.197 (1.008) | 7.902 / 7.654 (1.032) |
| INT8 | 1536 | 7.343 / 7.311 (1.004) | 8.007 / 7.808 (1.025) |
| INT8 | 2560 | 8.633 / 8.236 (1.048) | 9.136 / 8.920 (1.024) |

### Findings

- **Wave model (INT4).** A narrow block (92 registers, five resident per SM) covers 8 columns; a default block
  (142 registers for FP16, 146 for BF16, three resident per SM) covers 16. One wave therefore holds
  `132 * 5 * 8 = 5280` narrow columns and `132 * 3 * 16 = 6336` default columns. For `N <= 4224` (`132 * 32`) the
  busiest SM carries at most 32 columns with either tile. For `4224 < N <= 5280` it carries 48 with the default
  tile and 40 with the narrow one (ideal ratio 0.833; 0.86 measured at K = 4096). From N = 5312 the narrow tile
  needs a second wave and is 20% (N = 5312) to 44% (N = 6144) slower at K = 4096.
- **`N <= 4224`.** The narrow tile is a small latency win while `W`, the bytes of packed weights plus 16-bit
  scales (`N * K / 2 + 2 * N * (K / block_size)`), is small, and a tie or a small loss beyond. For
  3168 < N <= 4224 (`block_size` 32, K >= 1024) it is 2.5% faster on average below 28 MB; above 28 MB it is slower
  in 76% (FP16) and 99% (BF16) of the 80 conditions, by a median of 0.5% / 2.6% and at most 2.2% / 4.6%. N = 4224
  (FP16) shows the transition: ratio 0.98-1.00 up to 20 MB, 1.001 from 22 to 28 MB, and 1.002-1.022 from 29 to
  41 MB.
- **`4224 < N <= 5280`.** The narrow tile is 13% faster on average. The gain fades with `W` (N = 5120,
  `block_size` 32, FP16: ratio 0.80-0.86 from 9 to 35 MB, 0.93 at 39.8 MB, 0.968 at 41.3 MB, 0.984 at 42.8 MB,
  1.010 at 45.7 MB; BF16 is within 2.4 points of these).
- **`N > 5280`.** The narrow tile needs a second wave and is slower in all 402 / 216 / 216 conditions per dtype
  (`block_size` 32 / 64 / 128) up to N = 6144: by at least 6.4% (N = 5312), by a median of 35-42%, and by up to
  83% (N = 6144, K = 27648, `block_size` 32, BF16). Further up (`block_size` 32 only, N <= 12288) it is slower in
  78% of the shapes and faster by at most 5.3% in scattered, K-dependent windows (70-74% of the conditions below
  0.97 lie at N = 7000-8500, all at K = 2048-6144), which the rule does not chase. Three single conditions in which
  the default tile ran 14-110% slower than at the neighboring N (FP16: N = 10944 and 11488 at K = 3072; BF16:
  N = 8192 at K = 17408) are measurement artifacts and are excluded from that 5.3%.
- **The fade follows bytes, not weight elements.** `block_size` 32, 64, and 128 stream 0.5625, 0.5313, and 0.5156
  bytes per weight, and their mean ratios collapse onto one curve in `W` (4224 < N <= 5280, FP16 and BF16,
  6-36 conditions per cell):

  | `W` | `block_size` 32 | `block_size` 64 | `block_size` 128 |
  |-----|----------------:|----------------:|-----------------:|
  | 30-32 MB | 0.815 | 0.822 | 0.834 |
  | 32-34 MB | 0.816 | 0.826 | 0.824 |
  | 34-36 MB | 0.841 | 0.825 | 0.806 |
  | 36-38 MB | 0.874 | 0.854 | 0.834 |
  | 38-40 MB | 0.923 | 0.931 | 0.919 |
  | 40-42 MB | 0.971 | 0.971 | 0.966 |
  | 42-44 MB | 0.994 | 0.995 | 0.997 |
  | 44-46 MB | 1.010 | 1.008 | 1.011 |
  | 46-48 MB | 1.014 | 1.018 | 1.021 |
  | 48-52 MB | 1.024 | 1.030 | n/a |

  The mean ratio crosses 1.0 at about 43-44 MB. For every group size and dtype, the first K from which every ratio
  of an N stays at or above 0.99 lies at 41.8-42.3 MB or later (median 44-47 MB), so the cap for this window is
  42 MB. The mechanism of the fade is not explained.
- **Gain of the selected shapes** (the rule below, FP16, `block_size` 32, K >= 1024, `1 - narrow / default`; the BF16
  means are within 0.1 points):

  | N | Shapes | Mean gain | Middle 80% | Worst |
  |---|-------:|----------:|-----------:|------:|
  | <= 1024 | 385 | 2.6% | 1.5-3.8% | 0.5% |
  | 1024-2112 | 415 | 2.7% | 1.7-3.7% | 0.7% |
  | 2112-3168 | 370 | 4.8% | 3.6-6.2% | 2.4% |
  | 3168-4224 | 346 | 2.5% | 0.7-5.0% | -0.3% |
  | 4224-4800 | 208 | 12.8% | 6.7-17.6% | 1.8% |
  | 4800-5280 | 167 | 12.8% | 6.6-17.9% | 1.3% |

  At `block_size` 64 and 128 the mean gains are 0.4-1.3 points lower for N <= 3168 and 13.2-13.8% for
  4224 < N <= 5280; for 3168 < N <= 4224 the mean gain is only 0.7-1.1% and the worst case a 1.2-1.8% loss.
- **Bias.** With a bias the narrow tile is 6-32% slower than the default one (median 19%) for `N <= 4224` at all
  four K measured (528 conditions per dtype at `block_size` 32 and 128; N = 4096, FP16: 1.10 at K = 1024, 1.18 at
  K = 5120, 1.27 at K = 17408), with a sharp edge between N = 4224 and 4256 (K = 1024, FP16: 1.10 to 0.96). The
  bias epilogue makes the narrow kernel compile to 84 registers instead of 92 and its K loop costs about 0.5 ns
  more per K element (N = 512: +0.6 us at K = 1024 and +8.7 us at K = 17408); the default tile (142-146
  registers) loses only 0.1-0.3 us there. The register counts are identical with CUDA 12.8, 13.0, and 13.1. For
  `4224 < N <= 5280` the wave gain survives only while the penalty is small: N = 5120 (FP16) gains 1.7-10.4% up
  to 32.4 MB (ratio 0.954 there), is 1.040 at 35.4 MB and 1.210 at 41.3 MB. The first K from which all ratios of an
  N stay at or above 1.0 sits at these `W`:

  | `block_size` | Median over N (FP16 - BF16) | Earliest |
  |-------------:|----------------------------:|---------:|
  | 32 | 36.3 - 36.6 MB | 34.5 - 35.1 MB |
  | 64 | 37.0 - 37.5 MB | 35.5 - 36.6 MB |
  | 128 | 36.8 - 37.8 MB | 36.0 MB |

  The bias window therefore ends at 34 MB, below the earliest of them. Inside it the narrow tile wins in every
  measured condition (141 / 85 / 162 per dtype at `block_size` 32 / 64 / 128), by 1.6-11.9% (median 5-8%).
- **INT8.** The narrow tile has no window on the H200. Ratios narrow / default of the shapes the legacy cutoff
  selected (N <= 1280, `block_size` 32, rotating weights; median and range):

  | K | FP16 | BF16 |
  |---|-----:|-----:|
  | 256-2048 | 0.995 (0.942-1.028) | 1.013 (0.962-1.053) |
  | 3072-8192 | 1.037 (0.994-1.070) | 1.053 (1.019-1.082) |
  | 9216-17408 | 1.077 (1.044-1.095) | 1.079 (1.044-1.105) |
  | 19456-31744 | 1.104 (1.060-1.203) | 1.095 (1.063-1.176) |

  The narrow tile is faster in 55% (FP16) and 26% (BF16) of the K <= 2048 conditions, by at most 5.8% and 3.8%,
  and in 3% and 0% of the K = 3072-8192 conditions. The spot checks with a bias (four K) behave like the scale-only
  ones. The earlier H200 datapoint that motivated the legacy cutoff (BF16, N = 1280, K = 1280) was not reproduced:
  the ratio is 1.013 with rotating and 1.066 with cached weights (FP16: 0.988 and 1.033).

### Rule and policy audit

The H200 rule (`IsH200Int4M8NarrowTile` in `fpA_intB_gemv.h`) selects the narrow tile for INT4 when:

- scale-only: `N <= 4224` and `W <= 28 MB`, or `4224 < N <= 5280` and `W <= 42 MB`;
- with a bias: `4224 < N <= 5280` and `W <= 34 MB`.

INT8 always keeps the default tile. *Regret* is how much slower the narrow tile is on a shape the rule sends to
it (`narrow / default - 1`); *forgone gain* is how much faster it would have been on a shape the rule keeps on the
default tile (`default / narrow - 1`). Rotating weights, INT4, N <= 6144; each cell is FP16 / BF16:

| `block_size` | Epilogue | Rule | Worst regret | Worst forgone gain |
|-------------:|----------|------|-------------:|-------------------:|
| 32 | scale-only | legacy `N <= 5120` | 2.17% / 4.60% | 23.3% / 23.8% |
| 32 | scale-only | element caps 56M / 76M | 0.88% / 1.36% | 2.92% / 1.41% |
| 32 | scale-only | **28 / 42 MB** | 0.25% / 0.51% | 3.14% / 4.40% |
| 64 | scale-only | legacy | 1.53% / 2.97% | 24.8% / 25.3% |
| 64 | scale-only | element caps | 1.29% / 1.15% | 4.82% / 5.34% |
| 64 | scale-only | **28 / 42 MB** | 1.29% / 1.15% | 2.56% / 2.64% |
| 128 | scale-only | legacy | 1.83% / 2.52% | 23.7% / 24.2% |
| 128 | scale-only | element caps | 1.78% / 1.48% | 7.75% / 9.28% |
| 128 | scale-only | **28 / 42 MB** | 1.78% / 1.48% | 2.30% / 2.59% |
| 32 | bias | legacy | 29.3% / 32.4% | 11.7% / 11.6% |
| 32 | bias | never narrow | 0% | 12.7% / 12.5% |
| 32 | bias | **34 MB window** | 0% | 6.42% / 6.67% |
| 64 | bias | legacy | 27.9% / 30.7% | 12.1% / 11.6% |
| 64 | bias | never narrow | 0% | 13.2% / 13.1% |
| 64 | bias | **34 MB window** | 0% | 8.34% / 7.70% |
| 128 | bias | legacy | 29.2% / 30.8% | 10.9% / 10.5% |
| 128 | bias | never narrow | 0% | 13.5% / 12.8% |
| 128 | bias | **34 MB window** | 0% | 8.83% / 8.08% |

- The legacy cutoff selects 2,477 of the 2,970 `block_size` 32 conditions per dtype. It loses up to 2.2% (FP16) and
  4.6% (BF16; 55 conditions above 3%) on the large-K shapes it selects and forgoes up to 23-24% for
  5120 < N <= 5280 (72 / 67 conditions above 3%). The mean gap to always picking the faster tile (regret on the
  shapes sent to the narrow tile plus forgone gain on the rest) is 0.36% / 0.46% for the legacy cutoff and
  0.025% / 0.013% for the rule at `block_size` 32 (0.02-0.03% against 0.38-0.44% at 64 and 128). With a bias the
  mean gap of the legacy cutoff is 12.4% / 12.8% at `block_size` 32 and 11.9% / 12.2% at 128, because it selects the
  narrow tile for every `N <= 5120`. Never narrowing costs 1.0% at `block_size` 32 and 1.3-1.4% at 128; the 34 MB
  window costs 0.04% and 0.07%.
- Element caps (`N * K`) fit `block_size` 32, but forgo up to 5.3% at `block_size` 64 and 9.3% at 128 because those
  shapes stream fewer bytes per weight. The byte caps remove that dependency.
- The caps sit on a flat part of the trade-off (`block_size` 32, FP16 / BF16, worst regret and worst forgone gain):
  28 / 43 MB gives the same regret (0.25% / 0.51%) and a forgone gain of 3.14% / 3.45%; 28 / 44 MB raises the regret to
  0.38% / 1.17%; 28 / 41 MB forgoes up to 5.0% / 6.2%; 30 / 42 MB raises the regret to 0.88% / 1.29%; 24 / 42 MB
  forgoes up to 3.7% / 4.4% (13 / 16 conditions above 3%). 42 MB sits at the earliest 0.99 crossing, just below the
  43-44 MB crossover of the mean ratio.
- The 34 MB bias window has no regret at any group size; its forgone gain is confined to shapes between 34.0 and
  36.8 MB, just above the cap. A 36 MB window would regret up to 4.0% / 3.6% (N = 5120, K = 12288, `block_size` 32).
- Cache-resident launches (the same weights repeated; `block_size` 32; not used for decisions) are less favorable to
  every rule. For the new rule the worst regret is 5.2% / 4.7% (always N = 1312, K = 1024) with 88 / 55 of 2,748
  conditions above 3%, and the mean gap is 0.38% / 0.29%; the legacy cutoff has 12.8% / 12.0% and 0.66% / 0.56%.
  The selected shapes are still faster on average (mean ratio 0.91 in the `4224 < N <= 5280` window and 0.98-0.99
  for `N <= 4224`), but 28-31% of the latter are slower than the default tile.

### Production check

On the H200, `FpAIntBGemvTest.*` and the four `*M8NarrowTile` fixtures (INT4 and INT8, FP16 and BF16; 19 tests in
all) pass. The fixtures launch through the production dispatcher with the device reported and check the executed
`CtaN` at K = 1024 and 4096 (the 5280-column wave edge and the first N past it), at K = 14336 and 16384 (the byte
caps) and for M = 7 and 9, compare `M = 8` bit for bit with two `M = 4` launches and with the CUTLASS reference, and
check that a launch that reports no device keeps the device-independent cutoffs. The fixtures are scale-only
`block_size` 32 because the compact fpA_intB build has neither a bias nor other group sizes, so those are covered by
pure-function tests of the picker that put every byte cap at its exact edge for `block_size` 32, 64, and 128.

### Caveats

- Only an H200 was measured. H100 SXM, H800, and GH200 report the same SM 9.0 / 132-SM pair and therefore share
  the rule, but their memory systems differ and the fade point and the bias window may move there. H100 PCIe (114
  SMs) and H20 (78 SMs) keep the device-independent cutoffs.
- `block_size` 32 is the main data set; `block_size` 64 and 128 are spot checks on eight values of K, and the bias
  data use the limited grids listed in the setup. K > 31744 and, for `block_size` 64 and 128, N > 6144 were not
  measured.
- The fade at 44 MB and the bias slowdown are measured, not explained. The register counts of the bias kernel were
  read for CUDA 12.8, 13.0, and 13.1 only.
- Decisions use rotating (DRAM-streaming) weights. A workload that relaunches the same small weights back to back
  can see the cache-resident regressions listed above (up to about 5%) on some selected shapes.
- Other devices keep the legacy cutoffs, also for launches with a bias, which cost up to 32% on the H200 and were
  not measured elsewhere. Activation scales and alpha-in-advance were not timed; they keep the default tile on the
  A100 and H200.
- Kernel-level timings with synthetic weights and activations; no end-to-end ORT latency was measured.
- The wave-aware path was checked on an A100 with a reported sm_12x device (dispatch decision and host cost
  only); no sm_12x GPU was used.
