# MatMulBlockScaledFp8 - CUDA Experiments

This document records CUDA performance experiments for
**MatMulBlockScaledFp8** (`com.microsoft::MatMulBlockScaledFp8`). It includes
both retained optimizations and rejected variants so future tuning does not
repeat experiments whose crossover points were already measured.

Related documentation:

- [matmul_block_scaled_fp8.md](matmul_block_scaled_fp8.md) - operator behavior and current dispatch chain.
- [matmul_block_scaled_fp4_experiments.md](matmul_block_scaled_fp4_experiments.md) - related FP4 experiments.

---

## Test Environment

- GPU: NVIDIA GeForce RTX 5060 Ti, SM120 (Blackwell), 36 SMs.
- CUDA toolkit: 13.0.48.
- CUTLASS: 4.4.2.
- Build directory: `build/cu130_fp8_fp4/Release`.
- Benchmark shape unless stated otherwise: `N=4096`, `K=4096`.
- Data: FP8 E4M3 A and B, FP32 scales, BF16 output.
- Scale granularity: `(1, 1, 128)`.
- Timing: 100 warmup iterations and 500 measured iterations.

Build and install the onnxruntime-gpu wheel, and run one benchmark case with:

```bash
export ORT_REPO=$(git rev-parse --show-toplevel)
cd $ORT_REPO/onnxruntime/test/python/contrib_ops
"$ORT_REPO/venv/bin/python" profile_matmul_block_scaled.py" \
  --op fp8 --activation-dtype fp8 --block-size 128 \
  --m 64 --n 4096 --k 4096 --warmup 100 --repeat 500
```

Process-level timing varies on this workstation. Compare adjacent builds with
identical commands and use mean, minimum, p50, and p90 rather than one timing
statistic alone. Every performance run must also pass the harness accuracy
check.

---

## 1. Eight-Row Decode GEMV

The decode kernel maps one warp to an output column and a small group of A rows.
The original dispatch grouped at most four rows per warp. For `M=5..8`, that
launched a second warp over the same B row and streamed the weights twice.

The retained variant adds `RowsPerWarp=8` and dispatches row groups of 1, 2, 4,
or 8. This allows one warp to reuse each loaded B vector across all active rows
for `M<=8`.

Representative adjacent-build results for `N=4096`, `K=4096`:

| M | Four-row grouping | Eight-row grouping | Result |
|---:|---:|---:|---:|
| 6 | 0.1835 ms | 0.1521 ms | 17.1% faster |
| 7 | 0.1725 ms | 0.1534 ms | 11.1% faster |
| 8 | 0.1865 ms | 0.1411 ms | 24.3% faster |

Decision: retained. The focused CUDA operator suite includes an eight-row
correctness case.

---

## 2. SM120 64x128x128 Ping-Pong Kernel

The initial SM120 prefill implementation used one CUTLASS kernel for all shapes:

- MMA tile: `128x128x128`.
- Schedule: `KernelScheduleSm120Blockwise`.
- Cluster: `1x1x1`.

For small prefill M, a 128-row tile wastes work and produces too few independent
tiles to fill the 36-SM GPU. A second kernel was instantiated with:

- MMA tile: `64x128x128`.
- Schedule: `KernelTmaWarpSpecializedBlockwisePingpongSm120`.
- Cluster: `1x1x1`.

The first experiment routed all `M<=256` cases to ping-pong. Accuracy passed at
all measured shapes, but the performance crossover was below 128:

| M | Original mean | Ping-pong mean | Result |
|---:|---:|---:|---:|
| 9 | 0.1238 ms | 0.1067 ms | 13.8% faster |
| 16 | 0.1136 ms | 0.1243 ms | 9.4% slower in this run |
| 32 | 0.1480 ms | 0.1056 ms | 28.7% faster |
| 64 | 0.1648 ms | 0.1091 ms | 33.8% faster |
| 128 | 0.1319 ms | 0.1381 ms | 4.7% slower |
| 256 | 0.1748 ms | 0.1989 ms | 13.8% slower |

The dispatch cutoff was therefore narrowed to `M<=64`. A final rebuild of the
retained routing produced:

| M | Mean | Minimum | p50 | p90 | TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 9 | 0.1092 ms | 0.0818 ms | 0.1042 ms | 0.1296 ms | 2.77 |
| 16 | 0.1031 ms | 0.0756 ms | 0.0992 ms | 0.1180 ms | 5.21 |
| 32 | 0.1139 ms | 0.0818 ms | 0.1094 ms | 0.1277 ms | 9.43 |
| 64 | 0.1072 ms | 0.0918 ms | 0.1052 ms | 0.1188 ms | 20.04 |
| 128 | 0.1274 ms | 0.0990 ms | 0.1247 ms | 0.1452 ms | 33.73 |
| 256 | 0.1859 ms | 0.1531 ms | 0.1801 ms | 0.2121 ms | 46.20 |

The `M=128` and `M=256` rows use the original kernel in this final table.

Decision: retain ping-pong only for `M<=64`. Use the same cutoff for workspace
size calculation and launch dispatch.

### External Comparison

A local vLLM SM120 blockwise FP8 implementation also uses a `64x128x128`
ping-pong kernel through `M=256`. That is useful precedent for the kernel shape,
but not for the cutoff: vLLM uses scale granularity `(1,128,128)`, while this
operator uses `(1,1,128)` and has substantially more B-scale traffic. The local
measurements above are authoritative for ORT's layout.

---

## 3. Validation

## 3. Rejected 128x64x128 Cooperative Kernel

The next experiment targeted the occupancy-sensitive `65<=M<=128` range. At
`M=128,N=4096`, the default `128x128` MN tile creates only 32 output tiles for
a 36-SM GPU. A `128x64x128` cooperative kernel creates 64 tiles while changing
only tile N.

The kernel compiled and passed accuracy, but its gains were not consistent
enough to justify a third production kernel:

| M | 128x128x128 mean | 128x64x128 mean | Result |
|---:|---:|---:|---:|
| 65 | 0.1449 ms | 0.1409 ms | 2.8% faster, but p50 regressed |
| 96 | 0.1561 ms | 0.1460 ms | 6.4% faster |
| 128 | 0.1398 ms | 0.1443 ms | 3.3% slower |

Decision: rejected. The modest, noisy improvement around `M=96` does not
offset the added binary size and dispatch complexity, especially with a
regression at the natural tile boundary.

---

## 4. Rejected 128x256x128 Cooperative Kernel

A wider N tile was tested for `M>=384` to reduce output-tile scheduler and
epilogue overhead while retaining enough grid parallelism. CUTLASS's SM120
blockwise generator does not emit this tile; it generates only `128x128x128`
cooperative and `64x128x128` ping-pong kernels.

The candidate was not viable:

1. With automatic stage carveout, CUTLASS selected one mainloop stage and
  compilation failed because SM120 blockwise MMA requires at least two stages.
2. Forcing `StageCount<2>` compiled and linked, but kernel initialization failed
  at runtime for `M=512,N=4096,K=4096`, consistent with the two-stage shared
  memory requirement exceeding the available configuration.

Decision: rejected. Keep tile N at 128 for this blockwise kernel.

---

## 5. Rejected M-Only Stream-K Scheduler

CUTLASS maps `StreamKScheduler` on SM120 to
`PersistentTileSchedulerSm100StreamK`. It compiled, initialized, and passed
accuracy with the existing `128x128x128` cooperative kernel. The target was the
underfilled `65<=M<=128` range, where `N=4096` creates only 32 output tiles on a
36-SM GPU.

An adjacent-build sweep did not show a stable crossover:

| M | Default mean | Stream-K mean | Result |
|---:|---:|---:|---:|
| 72 | 0.1161 ms | 0.1196 ms | 3.0% slower |
| 80 | 0.1399 ms | 0.1368 ms | 2.2% faster |
| 88 | 0.1272 ms | 0.1390 ms | 9.3% slower |
| 96 | 0.1415 ms | 0.1245 ms | 12.0% faster |
| 104 | 0.1164 ms | 0.1364 ms | 17.2% slower |
| 112 | 0.1378 ms | 0.1398 ms | 1.5% slower |
| 120 | 0.1434 ms | 0.1391 ms | 3.0% faster |

Decision: rejected for an M-only dispatch. K splitting has enough reduction and
fixup overhead that neighboring M values alternate between wins and losses.

---

## 6. Rejected Extended Ping-Pong Range

The retained ping-pong kernel was also measured throughout the previously
untested `65<=M<128` range. Extending its cutoff above 64 generally regressed
latency. For example, compared with the adjacent default build, mean latency
changed from 0.1161 to 0.1340 ms at `M=72`, from 0.1272 to 0.1400 ms at `M=88`,
and from 0.1164 to 0.1463 ms at `M=104`.

Decision: keep the `M<=64` cutoff. The smaller M tile's occupancy benefit no
longer compensates for its additional work tiles once M exceeds 64.

---

## 7. N/K-Aware Stream-K Scheduler

The M-only experiment mixed wins and regressions because output-tile occupancy
depends on N and Stream-K reduction overhead is amortized by K. A second matrix
held `M=96` constant and varied N and K. The retained region uses Stream-K when:

```text
64 < M <= 128
N <= 2048
K >= 4096
K >= 4 * N
```

The last comparison uses widened integer arithmetic in the implementation. The
same predicate selects workspace sizing and kernel launch.

Representative adjacent-build results on SM120:

| M | N | K | Default mean | Stream-K mean | Result |
|---:|---:|---:|---:|---:|---:|
| 65 | 1024 | 4096 | 0.1413 ms | 0.1160 ms | 17.9% faster |
| 80 | 1024 | 4096 | 0.1446 ms | 0.1243 ms | 14.0% faster |
| 112 | 1024 | 4096 | 0.1540 ms | 0.1117 ms | 27.5% faster |
| 128 | 1024 | 4096 | 0.1428 ms | 0.1196 ms | 16.2% faster |
| 96 | 1024 | 8192 | 0.2054 ms | 0.1414 ms | 31.2% faster |
| 96 | 2048 | 8192 | 0.2020 ms | 0.1647 ms | 18.5% faster |

Longer repeated runs for `N=2048,K=8192` confirmed Stream-K means of
0.160-0.169 ms at `M=80` and 0.159-0.163 ms at `M=112`, compared with default
means of about 0.203 ms and 0.199 ms respectively.

The predicate excludes measured mixed or losing regions, including
`N=512,K<=2048`, `N=1024,K=2048`, and wider N with insufficient K. This is
intentionally conservative rather than an attempt to model every shape.

Decision: retained.

---

## 8. Validation

The retained decode and SM120 changes were validated with:

```bash
cd "$ORT_REPO/build/Linux/Release"
./onnxruntime_provider_test --gtest_filter='MatMulBlockScaledFp8OpTest.*'
```

All focused tests passed on SM120. The Python benchmark matrices also passed their
FP32-reference accuracy check at every shape.

---

## 9. Lessons and Next Tuning Area

- Reusing B loads across decode rows is more important than exposing extra warps
  for `M=5..8`.
- A smaller M tile improves occupancy and avoids padded work for small prefill,
  but the crossover must be measured for this operator's scale layout.
- A cutoff copied from a kernel with different scale granularity is not a valid
  dispatch heuristic.
- Workspace and launch selection must use exactly the same dispatch predicate.
- The SM120 blockwise mainloop requires at least two stages; a 256-column tile
  cannot satisfy that requirement with this epilogue and shared-memory budget.
- Stream-K is supported for this kernel, but an M-only Stream-K range is not
  stable enough to retain. N/K-aware dispatch is required to account for output
  tile occupancy and reduction/fixup amortization.
