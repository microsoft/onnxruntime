# CPU MultiHeadAttention masked-path performance experiments

Tracking work for issue [#29655](https://github.com/microsoft/onnxruntime/issues/29655):
*CPU MultiHeadAttention: redundant per-head mask copy makes masked attention ~1.6× slower.*

The affected code is `AttentionCPUBase::ComputeAttentionProbs` in
[`onnxruntime/contrib_ops/cpu/bert/attention_cpu_base.h`](../../../../onnxruntime/contrib_ops/cpu/bert/attention_cpu_base.h),
used by the CPU `Attention`, `MultiHeadAttention`, and `DecoderMaskedMultiHeadAttention` ops.

## Background

Original masked path, per `(batch × head)` unit inside a `TryParallelFor` over `batch*num_heads`:

1. `PrepareMask(...)` materializes the mask as a full `[B, S, T]` float tensor.
2. For every head, `memcpy` the `[S, T]` mask tile into the head's score buffer.
3. `math::Gemm(..., beta = 1.0f, ...)` accumulates `Q·Kᵀ` on top (read-modify-write),
   with a `nullptr` thread pool (so each head's GEMM is single-threaded).
4. A separate `MlasComputeSoftmax` over `N = B*num_heads*S` rows.

The `batch*num_heads` parallel granularity limits GEMM parallelism; extra threads only help
the softmax pass.

## Experiment setup

- Machine: AMD EPYC 7763, 16 physical cores / 32 logical (SMT on). Pinned to physical cores
  `taskset -c 0-15`.
- Build: `build/cpu/Release` (CPU EP only, Release), Python bindings enabled.
- Benchmark: `onnxruntime/test/python/transformers/benchmark_mha.py` with `--causal`
  (routes through the masked `AttentionCPUBase` path; kernel `ort:math`) and
  `--intra_op_num_threads` to control threads.
- Metric: min latency (ms) over 3 runs × 500 iterations per run.
- Command shape:

  ```bash
  PYTHONPATH=build/cpu/Release taskset -c 0-15 \
    python onnxruntime/test/python/transformers/benchmark_mha.py \
      -b <B> -s <S> -n <heads> -d <head_size> --causal \
      --intra_op_num_threads <T> -r 500
  ```

## Experiment 1 — separate fully-parallel mask/bias pass (beta=0 GEMM)

Change: GEMM writes with `beta=0` (clean write); the additive mask and attention bias are
applied afterwards in a **dedicated `TryParallelFor` over `B*N*S` rows** using vectorized
`MlasEltwiseAdd`, before the softmax. This removes the per-head mask `memcpy` and the
read-modify-write GEMM, and spreads the mask/bias work across all threads.

Results (min latency ms over 3×500):

| config (b, s, heads, head_size) | threads | baseline | patched | change |
|---|---:|---:|---:|---:|
| 1, 512, 4, 32 (issue repro) | 4 | 0.872 | 0.926 | −6% (slower) |
| 1, 512, 4, 32 | 8 | 0.885 | 0.860 | +3% |
| 4, 512, 4, 32 | 4 | 3.711 | 3.433 | +7.5% |
| 4, 512, 4, 32 | 8 | 3.095 | 2.937 | +5% |
| 1, 512, 12, 64 (bert-base) | 4 | 3.837 | 3.723 | +3% |
| 1, 512, 12, 64 | 8 | 3.550 | 2.739 | +23% |

**Assessment:** modest, mixed. Helps configs with enough `batch·heads·seq` work (up to
+23%), but slightly regresses the exact issue repro (`b=1`, 4 heads) at 4 threads. With only
`B·N = 4` GEMM units the extra separate pass adds dispatch overhead and re-reads the
`[B,N,S,T]` probs from memory (cache eviction between the GEMM loop and the mask pass), which
isn't offset by the small mask savings.

**Root cause revealed by the data:** the dominant serial factor is the **per-head `Q·Kᵀ`
GEMM**, parallelized only over `batch·num_heads` units (GEMM called with a `nullptr` thread
pool). The mask copy is a secondary cost. The unfused decomposition in the issue is faster
mainly because MLAS runs `Q·Kᵀ` as a **batched GEMM across all threads**.

## Next: deeper fix

Direction: give the GEMM real parallelism and avoid extra passes over the score buffer:

- Replace the per-head `math::Gemm` loop with a batched GEMM over all `B·N` heads driven by
  the thread pool (e.g. `MlasGemmBatch`), or otherwise parallelize the GEMM across all
  threads.
- Fuse the additive mask/bias directly into the softmax pass (streamed per `[T]` row) so
  there is no separate `[B,S,T]` materialization and no extra read-modify-write pass.

## Experiment 2 — batched GEMM + mask/bias fused into softmax (deeper fix)

Change (in `ComputeAttentionProbs`):

1. **Batched GEMM.** Collect the per-head `Q`/`K`/`C` pointers (including any past→present
   `K` concatenation) into an array of `MLAS_SGEMM_DATA_PARAMS` and issue all `B·N` matmuls as
   a single `MlasGemmBatch(..., beta=0, tp)`. MLAS now parallelizes across all threads
   (partitioning both the batch and the `M` dimension) instead of running one single-threaded
   GEMM per head (which capped parallelism at `B·N` units). Float-only path; a per-head
   fallback is kept under `if constexpr` for non-float `T` (not currently instantiated).
2. **Fused mask/bias + softmax.** When a mask, attention bias, or QK output is present, a
   single `TryParallelFor` over `B·N·S` rows adds the mask and attention bias (vectorized
   `MlasEltwiseAdd`), optionally emits the pre-softmax QK, then softmaxes the row — all while
   the row is hot in cache. No separate materialization/add pass over the `[B,N,S,T]` score
   buffer. When none are present, the fast path calls the batched `MlasComputeSoftmax`
   directly.

Results (min latency ms over **5 runs × 1000 iters**, `--causal`, `taskset -c 0-15`):

| config (b, s, heads, head_size) | threads | baseline | deep fix | speedup |
|---|---:|---:|---:|---:|
| 1, 512, 4, 32 (issue repro) | 1 | 2.503 | 2.514 | ~0% |
| 1, 512, 4, 32 | 4 | 0.852 | 0.854 | ~0% |
| 1, 512, 4, 32 | 8 | 0.910 | 0.733 | **+19%** |
| 1, 512, 4, 32 | 16 | 1.090 | 0.944 | **+13%** |
| 4, 512, 4, 32 | 1 | 10.631 | 10.519 | +1% |
| 4, 512, 4, 32 | 4 | 3.589 | 3.387 | +6% |
| 4, 512, 4, 32 | 8 | 3.002 | 2.466 | **+18%** |
| 4, 512, 4, 32 | 16 | 2.933 | 2.582 | **+12%** |

**Assessment:** consistent win, and it fixes the pathological thread-scaling the issue calls
out. Key observations:

- **Single thread:** identical to baseline — the batched GEMM does the same work as the
  per-head GEMM, so there is no compute regression.
- **Negative scaling fixed.** Baseline gets *slower* going 8→16 threads on the 4-head shape
  (`b=1`: 0.910 → 1.090 ms) because the `Q·Kᵀ` GEMM can only use `B·N = 4` units, so extra
  threads only contend. The deep fix keeps scaling (best `b=1` latency 0.733 ms at 8 threads
  vs baseline best 0.852 ms at 4 threads), because the batched GEMM spreads across all
  threads.
- **10–20% faster** at the thread counts people actually run inference with (8–16).

Correctness: `onnxruntime_provider_test` attention suites pass (287 passed) for both the
mask (causal) and attention-bias paths, and the `DecoderMaskedMultiHeadAttention` QK-output
path.

## Experiment 3 — popular model shapes (head_size = 64)

`head_size = 32` (Experiment 1/2) is an unusually small head. The shapes below use the
realistic `head_size = 64` of common transformers. All runs use `--causal` (the masked
`AttentionCPUBase` path; a BERT/DistilBERT padding mask hits the same code as `mask_data`).
Min latency ms over 3 runs × 500 iters, `taskset -c 0-15`.

| model shape (b, s, heads, head_size) | threads | baseline | deep fix | speedup |
|---|---:|---:|---:|---:|
| BERT-base    (1, 384, 12, 64)  | 4  | 2.151 | 1.972 | +8% |
| BERT-base    (1, 384, 12, 64)  | 8  | 1.710 | 1.618 | +5% |
| BERT-base    (1, 384, 12, 64)  | 16 | 1.883 | 1.794 | +5% |
| BERT-base    (8, 384, 12, 64)  | 4  | 20.145 | 19.529 | +3% |
| BERT-base    (8, 384, 12, 64)  | 8  | 16.159 | 13.900 | **+14%** |
| BERT-base    (8, 384, 12, 64)  | 16 | 13.587 | 12.155 | **+11%** |
| BERT-large   (1, 512, 16, 64)  | 4  | 5.022 | 5.074 | ~0% |
| BERT-large   (1, 512, 16, 64)  | 8  | 4.032 | 4.033 | ~0% |
| BERT-large   (1, 512, 16, 64)  | 16 | 3.514 | 3.198 | +9% |
| BERT-large   (4, 512, 16, 64)  | 4  | 23.802 | 21.540 | +10% |
| BERT-large   (4, 512, 16, 64)  | 8  | 17.769 | 15.261 | **+14%** |
| BERT-large   (4, 512, 16, 64)  | 16 | 14.506 | 13.074 | +10% |
| DistilBERT   (8, 256, 12, 64)  | 4  | 9.360 | 9.338 | ~0% |
| DistilBERT   (8, 256, 12, 64)  | 8  | 7.367 | 6.888 | +7% |
| DistilBERT   (8, 256, 12, 64)  | 16 | 6.662 | 6.435 | +3% |
| GPT2-small   (1, 1024, 12, 64) | 4  | 15.568 | 15.270 | +2% |
| GPT2-small   (1, 1024, 12, 64) | 8  | 14.856 | 11.156 | **+25%** |
| GPT2-small   (1, 1024, 12, 64) | 16 | 12.547 | 10.935 | **+13%** |
| GPT2-medium  (1, 512, 16, 64)  | 4  | 4.952 | 4.933 | ~0% |
| GPT2-medium  (1, 512, 16, 64)  | 8  | 4.117 | 3.439 | **+17%** |
| GPT2-medium  (1, 512, 16, 64)  | 16 | 3.320 | 3.166 | +5% |

**Assessment:** the deeper fix is faster or neutral on every popular-model shape, with the
biggest gains (11–25%) at 8–16 threads. It never regresses meaningfully (worst point is
BERT-large `b=1` at 4 threads, −1%, within noise). The pattern matches Experiment 2: at high
thread counts the batched GEMM keeps scaling where the baseline plateaus or slows, and the
fused mask/softmax removes an extra pass over the score buffer. Even at `head_size = 64`
(where the per-head mask copy is a smaller fraction of the GEMM), the batched-GEMM
parallelism alone delivers a solid speedup.

### Notes / follow-ups


- The `[B,S,T]` mask is still materialized by `PrepareMask`. A further optimization could
  broadcast a `[B,T]` padding mask directly during the fused softmax pass (no `S`-axis
  materialization), but that is a separate change with causal-mask handling to consider.
  **(Implemented in Experiment 4 below.)**
- The batched-GEMM path is `float`-only (the only instantiation of this base). If a
  half-precision CPU `Attention`/`MHA` is ever added, it will use the per-head fallback until
  a batched half GEMM is wired up.

## Experiment 4 — `[B,T]` padding mask broadcast in softmax

A pure padding mask (1D `mask_index` of shape `(B)`/`(2B)`, or a 2D key mask `(B, T)`) with
**no** causal mask does not depend on the query position `S`. Instead of expanding it to a
full `[B, S, T]` tensor (`PrepareMask`), it is now kept as `[B, T]` (`PreparePaddingMask`) and
broadcast across `S` during the fused softmax via a `mask_seq_stride = 0`.

`ComputeAttentionProbs` takes `mask_batch_stride` / `mask_seq_stride`; the mask row for
`(batch b, query s)` is `mask_data + b * mask_batch_stride + s * mask_seq_stride`:

- full `[B,S,T]` mask (causal, 3D mask, or causal+padding): `batch_stride = S*T`, `seq_stride = T`.
- `[B,T]` padding mask (this optimization): `batch_stride = T`, `seq_stride = 0`.

This shrinks the mask buffer by a factor of `S`, removes the `S`-axis broadcast/`memcpy` in
mask preparation, and cuts the mask bytes read during softmax from `O(B·N·S·T)` to
`O(B·N·S·T)` reads of only `B·T` distinct bytes (hot in cache).

**Why this matters:** CPU flash attention is skipped whenever `key_padding_mask != nullptr`
(see `multihead_attention.cc`), so a **non-causal BERT/DistilBERT with a padding mask** runs
through exactly this `AttentionCPUBase` path — the common real-world inference case.

Benchmark uses a **2D key padding mask, non-causal** (`bench_padmask.py`, min over 3 runs ×
500 iters, `taskset -c 0-15`). Three builds: **baseline** (`origin/main`), **deep fix only**
(Experiment 2, no `[B,T]`), **+[B,T]** (this change on top of the deep fix). The deep-fix-only
number is obtained by forcing `padding_mask_only = false`.

| model shape (b, s, heads, 64) | threads | baseline | deep-fix only | +[B,T] | +[B,T] vs baseline |
|---|---:|---:|---:|---:|---:|
| BERT-base   (1, 384, 12) | 8  | 1.718 | 2.012 | 1.588 | **+8%** |
| BERT-base   (1, 384, 12) | 16 | 1.846 | 1.919 | 1.609 | **+13%** |
| BERT-base   (8, 384, 12) | 8  | 15.213 | 14.529 | 12.783 | **+16%** |
| BERT-base   (8, 384, 12) | 16 | 13.489 | 12.615 | 11.556 | **+14%** |
| BERT-large  (4, 512, 16) | 8  | 17.194 | 15.632 | 14.945 | **+13%** |
| BERT-large  (4, 512, 16) | 16 | 14.512 | 13.567 | 12.302 | **+15%** |
| DistilBERT  (8, 256, 12) | 8  | 7.388 | 6.977 | 6.199 | **+16%** |
| DistilBERT  (8, 256, 12) | 16 | 6.603 | 6.340 | 6.185 | +6% |
| BERT-base   (1, 512, 12) | 8  | 3.494 | 3.088 | 2.888 | **+17%** |
| BERT-base   (1, 512, 12) | 16 | 2.925 | 2.811 | 2.686 | +8% |

**Assessment:**

- **Deep-fix-only is mixed vs baseline for padding masks:** it helps larger batches (BERT-base
  `b=8 @8`: 15.213 → 14.529) but slightly *hurts* tiny `b=1` shapes (BERT-base `b=1 s=384 @8`:
  1.718 → 2.012) — when `B·N ≥ threads` the batched GEMM adds little parallelism, and per-row
  softmax is marginally slower than baseline's single batched `MlasComputeSoftmax`.
- **`[B,T]` is the decisive optimization for padding masks:** it is faster than deep-fix-only
  by **+4% to +21%** (e.g. BERT-base `b=1 s=384 @8`: 2.012 → 1.588 ms, +21%), and turns the
  combined change into a **+6% to +17%** net win over `origin/main` across every
  BERT/DistilBERT shape — including the small `b=1` cases where deep-fix-only alone regressed.
- Correctness: `onnxruntime_provider_test` — all 112 mask tests pass (1D left/right padding,
  2D raw mask, 3D mask, causal+mask, clamp-OOB), covering both the reduced `[B,T]` branch and
  the full `[B,S,T]` fallback.
