# GroupQueryAttention CPU Implementation Notes

This document describes the current CPU implementation of the `com.microsoft.GroupQueryAttention` operator in ONNX Runtime, with emphasis on quantized KV-cache execution.

## Scope

The CPU GroupQueryAttention kernel is implemented in:

- `onnxruntime/contrib_ops/cpu/bert/group_query_attention.cc`
- `onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h`
- `onnxruntime/contrib_ops/cpu/bert/group_query_attention_helper.h`

Quantized KV-cache GEMM helpers are implemented in MLAS:

- `onnxruntime/core/mlas/inc/mlas_qkv_quant.h`
- `onnxruntime/core/mlas/lib/qkv_quant.cpp`
- `onnxruntime/core/mlas/lib/qkv_quant_kernel_avx2.cpp`
- `onnxruntime/core/mlas/lib/qkv_quant_kernel_avx512vnni.cpp`
- `onnxruntime/core/mlas/lib/qkv_quant_kernel_neon.cpp`
- `onnxruntime/core/mlas/lib/flashattn_qkv.cpp` (flash attention tiled kernel)

The operator schema itself is defined in:

- `onnxruntime/core/graph/contrib_ops/contrib_defs.cc`

Relevant tests and benchmarks are in:

- `onnxruntime/test/contrib_ops/group_query_attention_op_test.cc`
- `onnxruntime/test/python/transformers/test_gqa_cpu_quantized.py`
- `onnxruntime/test/python/transformers/test_gqa.py`
- `onnxruntime/test/mlas/unittest/test_qkv_quant.cpp`
- `onnxruntime/test/mlas/bench/bench_qkv_quant.cpp`

This document focuses on runtime behavior on the CPU Execution Provider, not on the full operator schema.

## High-Level Execution Flow

At a high level, the CPU kernel executes GroupQueryAttention in these stages:

1. Validate attributes, input ranks, cache shapes, sequence lengths, and optional scale tensors.
2. Prepare query, key, and value views, including packed QKV input handling when applicable.
3. Apply rotary embeddings when requested.
4. Build or extend the present K/V cache.
5. Compute QK attention scores.
6. Apply attention bias, local-window masking, causal masking, softcap, and softmax.
7. Compute softmax-times-V and write the output tensor.
8. Return present K/V cache tensors.

The non-quantized and quantized paths share the surrounding validation, masking, softmax, and output flow. Their main difference is how the K/V cache is stored and read during QK and SV GEMMs.

The quantized path has two execution strategies:

- **Naive (full materialization)**: Computes the full `[S, T]` attention score matrix, applies masking and softmax, then computes the SV product. Simple but memory-intensive for long sequences.
- **Flash Attention (tiled, online softmax)**: Processes K/V in L2-cache-sized blocks using the online softmax algorithm (Milakov & Gimelshein, 2018). Avoids materializing the full attention matrix, reducing peak memory from O(S×T) to O(S×Bc) per head. Multi-threaded via the MLAS thread pool.

The flash path is selected by default when conditions are met (see below). Set `ORT_GQA_DISABLE_FLASH_ATTENTION=1` to force the naive path.

## Supported Cache Modes

### Non-quantized cache

When `k_quant_type` and `v_quant_type` are `NONE`, `kv_cache_bit_width` must be `0`. The past and present K/V tensors use the same floating-point element type as the kernel specialization.

### Quantized cache

When quantization is enabled, `k_quant_type` and `v_quant_type` must match and may be:

- `PER_TENSOR`
- `PER_CHANNEL`

The CPU quantized path supports:

- `kv_cache_bit_width = 8` for signed INT8 cache values
- `kv_cache_bit_width = 4` for signed INT4 cache values packed into `uint8`

The scale inputs are always `float` tensors:

- `PER_TENSOR`: `k_scale` and `v_scale` each contain exactly one element
- `PER_CHANNEL`: each scale tensor contains `kv_num_heads * head_size` elements

For per-channel scales, each KV head uses a contiguous `head_size` scale slice. For per-tensor scales, all heads share the single scale value.

## Quantized Cache Layout

The past and present KV cache tensors use BNSH layout:

- `batch_size`
- `kv_num_heads`
- `cache_sequence_length`
- `head_size`

For INT4, two signed 4-bit values are stored in each byte. The packed head dimension is `ceil(head_size / 2)` bytes. For INT8, the packed head dimension is `head_size` bytes.

During quantized execution, new key/value vectors are quantized on write into the present cache. Existing past-cache data and newly written present-cache data are then consumed by MLAS quantized GEMM helpers.

## Naive Path: QK GEMM + Softmax + SV GEMM

The naive (full materialization) path executes attention as three separate stages:

### QK GEMM

The QK stage computes:

```text
attention_scores = scale * query * K_cache^T
```

For quantized K cache, the CPU path calls `MlasQKGemm` with:

- FP32 query input `A`
- packed quantized K cache `B`
- a scalar or per-channel K scale buffer
- FP32 attention score output

The default MLAS contract is exact with respect to the FP32 query operand: only the K cache is dequantized on the fly. The query row is not quantized by default.

### Softmax and Masking

After QK GEMM, the CPU path applies the same attention-score processing used by the non-quantized path, including supported combinations of:

- attention bias
- causal masking
- local-window masking
- softcap
- smooth softmax / head sink
- optional QK output capture

The quantized cache mode does not change these score-processing semantics.

### SV GEMM

The SV stage computes:

```text
output = attention_probs * V_cache
```

For quantized V cache, the CPU path calls `MlasSVGemm` with:

- FP32 attention probabilities `A`
- packed quantized V cache `B`
- a scalar or per-channel V scale buffer
- FP32 output tile

As with QK GEMM, the default MLAS contract preserves the FP32 left-hand operand and dequantizes only the cached V values on the fly.

## Flash Attention Path

The flash attention path (`MlasFlashAttentionQuantizedKV`) processes K/V in blocks with online softmax, fusing QK, masking, softmax, and SV into a single tiled loop. This avoids the O(S×T) memory allocation for the full attention matrix.

### Algorithm

For each (batch, head, q_block) tile:

1. **QK GEMM** — `MlasQKGemm` on a block slice of quantized K cache (Bc rows at a time)
2. **Causal + local window masking** — Set masked positions to −∞ before softmax
3. **Online softmax** — Track running max `m` and sum `l`, rescale accumulated output with `exp(m_old − m_new)`
4. **V dequantize** — `MlasKVDequantize` the current V block to an FP32 temp buffer
5. **SV accumulate** — `MlasSgemmOperation(beta=1.0)` to accumulate `softmax(QK_block) × V_block` into the output
6. **Finalize** — Normalize accumulated output by `1/l` after all KV blocks are processed

### Activation Conditions

The flash path is selected when ALL of the following hold:

- `ORT_GQA_DISABLE_FLASH_ATTENTION` environment variable is not set (or set to `0`)
- `total_sequence_length > 1`
- No attention bias
- No softcap
- No smooth softmax
- No head sink
- No output QK capture

When any condition is not met, the kernel falls back to the naive full-materialization path.

### Block Size Selection

Block sizes are chosen based on L2 cache size:

- `kv_block_size (Bc)`: Sized so that a full KV block's scores + dequantized V fit within L2. Typical values: 128–256.
- `q_block_size (Br)`: Sized for the query tile. Typical value: 64.

### Threading

The flash kernel parallelizes across `(batch, head, q_block)` tiles using the ORT intra-op thread pool. Each thread gets a private working buffer containing space for:

- `l[Br]` and `m[Br]` — running softmax statistics
- `scores[Br × Bc]` — QK scores for current KV block
- `temp_output[Br × H]` — accumulated output
- `v_dequant[Bc × H]` — dequantized V block

## MLAS Dispatch Paths

MLAS selects the best available quantized KV-cache GEMM implementation through the platform dispatch table.

Current CPU paths include:

- scalar fallback in `qkv_quant.cpp`
- AVX2 FP32 fused-dequant kernels
- AVX512 FP32 fused-dequant kernels
- ARM NEON kernels

The AVX512 implementation also contains an optional approximate VNNI QK path for INT8 per-tensor K cache. It is disabled by default because it quantizes the FP32 query row before the dot product, which changes the `MlasQKGemm` numeric contract and can make results differ from scalar, AVX2, and NEON paths.

To opt in explicitly, set:

```bash
ORT_MLAS_QKGEMM_S8_APPROX_VNNI=1
```

This opt-in currently applies only to AVX512 INT8 per-tensor `MlasQKGemm`. It does not affect INT8 per-channel, INT4, or `MlasSVGemm` paths.

## Tests and Benchmarks

### Test locations

CPU GroupQueryAttention coverage is split across operator-level and MLAS-level tests:

- `onnxruntime/test/contrib_ops/group_query_attention_op_test.cc`
  - C++ operator tests, including quantized INT8/INT4 prompt cases and validation failures.
- `onnxruntime/test/python/transformers/test_gqa_cpu_quantized.py`
  - Python integration tests for CPU quantized KV-cache decoding accuracy.
- `onnxruntime/test/python/transformers/test_gqa.py`
  - Broader GroupQueryAttention test helper coverage, including cache-type configuration.
- `onnxruntime/test/mlas/unittest/test_qkv_quant.cpp`
  - MLAS `MlasKVQuantize`, `MlasKVDequantize`, `MlasQKGemm`, and `MlasSVGemm` contract tests.

The MLAS benchmark for quantized KV-cache GEMM and flash attention is:

- `onnxruntime/test/mlas/bench/bench_qkv_quant.cpp`

### Running tests

Build and test commands depend on the local build directory. For an existing CPU Release build rooted at `build/cpu_test/Release`, the focused tests can be run with:

```bash
cd build/cpu_test/Release

# MLAS quantized KV-cache primitive tests.
./onnxruntime_mlas_test

# CPU operator tests for quantized GroupQueryAttention.
./onnxruntime_test_all --gtest_filter="*GroupQueryAttentionQuantized*"
```

The Python integration test can be run from the repository root after activating the build/test environment:

```bash
python onnxruntime/test/python/transformers/test_gqa_cpu_quantized.py
```

### Running benchmarks

Rebuild the benchmark target after changing MLAS code:

```bash
cmake --build build/cpu_test/Release --target onnxruntime_mlas_benchmark -j $(nproc)
```

Run the representative INT8 per-tensor QKGemm decode benchmark for scalar, AVX2, and the default platform dispatch:

```bash
cd build/cpu_test/Release
unset ORT_MLAS_QKGEMM_S8_APPROX_VNNI
./onnxruntime_mlas_benchmark \
  --benchmark_filter='BM_QKGemm(/M:1/N_seqlen:512/K_head:128/QuantType:0|_Scalar/M:1/N:512/K:128/QuantType:0|_Avx2/M:1/N:512/K:128/QuantType:0)' \
  --benchmark_min_time=0.5s \
  --benchmark_repetitions=3 \
  --benchmark_report_aggregates_only=true
```

Run the opt-in approximate AVX512 VNNI path with:

```bash
cd build/cpu_test/Release
ORT_MLAS_QKGEMM_S8_APPROX_VNNI=1 ./onnxruntime_mlas_benchmark \
  --benchmark_filter='BM_QKGemm/M:1/N_seqlen:512/K_head:128/QuantType:0' \
  --benchmark_min_time=0.5s \
  --benchmark_repetitions=3 \
  --benchmark_report_aggregates_only=true
```

Run flash vs naive full-attention benchmark:

```bash
cd build/cpu_test/Release
./onnxruntime_mlas_benchmark \
  --benchmark_filter='BM_GQA_(Naive|Flash)' \
  --benchmark_min_time=0.5s \
  --benchmark_repetitions=3 \
  --benchmark_report_aggregates_only=true
```

To force the naive path at the operator level (for A/B testing during inference):

```bash
ORT_GQA_DISABLE_FLASH_ATTENTION=1 ./your_inference_app
```

### Updated benchmark results

The following results were measured on an Intel Xeon Platinum 8480C, 96 CPUs, using the CPU Release benchmark binary. Shape: `M=1`, `N=512`, `K=128`, INT8 per-tensor QKGemm.

| Implementation | Latency (ns, mean) | vs Scalar |
|---|---:|---:|
| Scalar fallback | 31,027 | 1.0x |
| AVX2 FP32 fused dequant-dot | 4,234 | 7.3x |
| AVX512 FP32 fused dequant-dot, default | 3,736 | 8.3x |
| AVX512 VNNI approximate, `ORT_MLAS_QKGEMM_S8_APPROX_VNNI=1` | 2,020 | 15.4x |

For comparison, the earlier PR description reported the approximate AVX512 VNNI path at 1,938 ns for this shape, with scalar at 30,179 ns and AVX2 at 4,219 ns. The default AVX512 path is now the exact FP32 fused-dequant implementation, so it is slower than approximate VNNI but preserves the `MlasQKGemm` FP32-query contract.

### Flash Attention vs Naive benchmark results

Measured on Intel Xeon Platinum 8480C, 96 CPUs. Shape: B=1, num_heads=16, kv_num_heads=8, head_size=128, INT8 per-tensor.

These results are from the operator-level Python benchmark. To reproduce:

```bash
# Build ORT with Python enabled (CPU only)
python tools/ci_build/build.py --build_dir build/cpu --config Release \
  --parallel --build_wheel --skip_tests

# Install the wheel, then run from a directory outside the ORT source tree:
cd /tmp
python <ort_src>/onnxruntime/test/python/transformers/benchmark_gqa_cpu_flash.py \
  --warmup 5 --repeats 20
```

The MLAS-level C++ benchmark (`bench_qkv_quant.cpp`) produces similar results without operator overhead:

```bash
cd build/cpu/Release
./onnxruntime_mlas_benchmark \
  --benchmark_filter='BM_GQA_(Naive|Flash)' \
  --benchmark_min_time=0.5s \
  --benchmark_repetitions=3 \
  --benchmark_report_aggregates_only=true
```

#### Latency — Prefill (S = T, prompt phase)

| Seq Length | Naive (ms) | Flash (ms) | Speedup | Threads |
|---:|---:|---:|---:|---:|
| 512 | 8.9 | 6.6 | 1.3x | 8 |
| 1024 | 44.3 | 26.5 | 1.7x | 8 |
| 2048 | 174.7 | 91.2 | 1.9x | 8 |
| 4096 | 824.1 | 358.4 | 2.3x | 8 |
| 512 | 60.7 | 45.9 | 1.3x | 1 |
| 1024 | 310.0 | 177.3 | 1.7x | 1 |
| 2048 | 1322.2 | 708.7 | 1.9x | 1 |
| 4096 | 6335.7 | 2779.5 | 2.3x | 1 |

#### Latency — Decode (S = 1, token generation)

| Total Seqlen | Naive (ms) | Flash (ms) | Speedup | Threads |
|---:|---:|---:|---:|---:|
| 512 | 0.23 | 0.25 | 0.9x | 8 |
| 1024 | 0.37 | 0.36 | 1.0x | 8 |
| 2048 | 0.62 | 0.64 | 1.0x | 8 |
| 4096 | 1.02 | 1.15 | 0.9x | 8 |

#### Peak Memory — Prefill (S = T, prompt phase)

| Seq Length | Naive Peak (MB) | Flash Peak (MB) | Memory Reduction |
|---:|---:|---:|---:|
| 2048 | +294 | +44 | 6.7x |
| 4096 | +1107 | +82 | 13.5x |
| 4096 (N=32) | +2131 | +87 | 24.5x |

The flash path's primary benefit is **memory reduction**: avoiding the full O(N×S×T) attention matrix allocation. For S=4096 with 16 heads, the naive path allocates ~1 GB for the attention scores alone, while the flash path uses only ~80 MB of working memory regardless of sequence length. The latency speedup (1.3–2.3x) comes from improved cache locality — the tiled algorithm keeps working data within L2 cache. The speedup ratio is the same regardless of thread count, confirming it is purely algorithmic (not parallelism-driven). Decode (S=1) shows no benefit because the single-row attention vector is already small.

## Current CPU Limitations

The current CPU GroupQueryAttention implementation has a few important limitations:

- Quantized K and V cache modes must match.
- Quantized CPU cache scales are `float` only.
- `kv_cache_bit_width` must be `0` when quantization is disabled, and `4` or `8` when quantization is enabled.
- INT4 cache storage uses packed `uint8` bytes and requires consumers to use the packed head dimension.
- The default AVX512 quantized KV-cache GEMM path preserves FP32 query and attention-probability operands; the approximate VNNI QK path is opt-in only.
- Hardware dispatch affects performance, but should not change default numeric semantics.
- The flash attention path does not support attention bias, softcap, smooth softmax, head sink, or QK output capture. These features fall back to the naive path.
- The MLAS quantized GEMM helpers operate on one per-batch/per-head tile at a time; outer parallelism is managed by the GQA kernel (or by the flash attention kernel internally).

## Future Work

Further optimization opportunities include:

- Fused `MlasSVGemmAccumulate(beta=1.0)` to avoid the V dequantization temporary buffer in the flash path, performing dequant-and-accumulate in one pass.
- Extend the flash attention path to support attention bias within the tiled loop.
- Improve the exact AVX512 INT8 per-tensor QK path without quantizing the FP32 query, for example by processing multiple K-cache rows per query row while keeping FP32 FMA semantics.
- Add AVX512-specific exact micro-kernels for common decode shapes such as `M=1`, `N=512/2048`, and `K=64/128`.
- Add dedicated accuracy/performance tests for the approximate VNNI opt-in path before enabling it in any production configuration.
- Reduce temporary copies in quantized cache concatenation when past and present buffers cannot be shared directly.
- Explore prepacking or layout transforms for long-lived quantized KV caches when the cache update pattern makes that worthwhile.

CPU features that are limited or not implemented relative to the broader operator schema include:

- Float8 KV-cache execution is described by the contrib operator schema, but the current CPU quantized path covered here is INT8/INT4.
- The CPU quantized path requires K and V quantization modes to match.
- Scale shapes are restricted to the CPU implementation's scalar or per-head-channel forms, rather than arbitrary broadcastable shapes.
- The approximate AVX512 VNNI QK path is opt-in only and is not part of the default numeric contract.
- Some schema features are supported by validation and shared attention code, but may not have the same optimized coverage in every CPU quantized path.

## Code Structure Summary

- `GroupQueryAttentionBase<T>::GroupQueryAttentionBase(...)`
  - reads attributes such as `num_heads`, `kv_num_heads`, quantization type, and `kv_cache_bit_width`
- `GroupQueryAttention<T>::Compute(...)`
  - validates input tensors, scale tensors, and sequence lengths
  - allocates output and present-cache tensors
  - dispatches to the quantized or non-quantized attention path
- `GroupQueryAttentionBase<T>::ApplyAttentionQuantized(...)`
  - quantizes new K/V values into the present cache
  - concatenates past and present cache chunks when needed
  - calls `MlasQKGemm` and `MlasSVGemm`
- `GroupQueryAttentionBase<T>::ApplyAttentionQuantizedFlash(...)`
  - concatenates new K/V into present cache (parallel over batch × kv_heads)
  - invokes `MlasFlashAttentionQuantizedKV` with L2-cache-aware block sizes
- `MlasQKGemm(...)`
  - computes FP32 query times quantized K cache transpose
- `MlasSVGemm(...)`
  - computes FP32 attention probabilities times quantized V cache
- `MlasFlashAttentionQuantizedKV(...)`
  - flash attention kernel with online softmax, tiled QK/SV over quantized KV cache
  - parallelizes across (batch, head, q_block) tiles via thread pool
