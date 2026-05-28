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

## QK GEMM

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

## Softmax and Masking

After QK GEMM, the CPU path applies the same attention-score processing used by the non-quantized path, including supported combinations of:

- attention bias
- causal masking
- local-window masking
- softcap
- smooth softmax / head sink
- optional QK output capture

The quantized cache mode does not change these score-processing semantics.

## SV GEMM

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

The MLAS benchmark for quantized KV-cache GEMM is:

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

### Updated benchmark results

The following results were measured on an Intel Xeon Platinum 8480C, 96 CPUs, using the CPU Release benchmark binary. Shape: `M=1`, `N=512`, `K=128`, INT8 per-tensor QKGemm.

| Implementation | Latency (ns, mean) | vs Scalar |
|---|---:|---:|
| Scalar fallback | 31,027 | 1.0x |
| AVX2 FP32 fused dequant-dot | 4,234 | 7.3x |
| AVX512 FP32 fused dequant-dot, default | 3,736 | 8.3x |
| AVX512 VNNI approximate, `ORT_MLAS_QKGEMM_S8_APPROX_VNNI=1` | 2,020 | 15.4x |

For comparison, the earlier PR description reported the approximate AVX512 VNNI path at 1,938 ns for this shape, with scalar at 30,179 ns and AVX2 at 4,219 ns. The default AVX512 path is now the exact FP32 fused-dequant implementation, so it is slower than approximate VNNI but preserves the `MlasQKGemm` FP32-query contract.

## Current CPU Limitations

The current CPU GroupQueryAttention implementation has a few important limitations:

- Quantized K and V cache modes must match.
- Quantized CPU cache scales are `float` only.
- `kv_cache_bit_width` must be `0` when quantization is disabled, and `4` or `8` when quantization is enabled.
- INT4 cache storage uses packed `uint8` bytes and requires consumers to use the packed head dimension.
- The default AVX512 quantized KV-cache GEMM path preserves FP32 query and attention-probability operands; the approximate VNNI QK path is opt-in only.
- Hardware dispatch affects performance, but should not change default numeric semantics.
- The MLAS quantized GEMM helpers operate on one per-batch/per-head tile at a time; outer parallelism is managed by the GQA kernel.

## Future Work

Further optimization opportunities include:

- Improve the exact AVX512 INT8 per-tensor QK path without quantizing the FP32 query, for example by processing multiple K-cache rows per query row while keeping FP32 FMA semantics.
- Add AVX512-specific exact micro-kernels for common decode shapes such as `M=1`, `N=512/2048`, and `K=64/128`.
- Add dispatch-specific benchmark coverage for prefill shapes (`M > 1`) and longer cache lengths.
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
- `MlasQKGemm(...)`
  - computes FP32 query times quantized K cache transpose
- `MlasSVGemm(...)`
  - computes FP32 attention probabilities times quantized V cache
