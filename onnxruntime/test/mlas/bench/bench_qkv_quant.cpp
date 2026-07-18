// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// Benchmark for quantized KV-cache GEMM kernels (MlasQKGemm / MlasSVGemm).
// Measures throughput for typical attention shapes:
//   - Decoding: M=1, K=head_size, N=total_seqlen
//   - Prefill:  M=128, K=head_size, N=total_seqlen
//
// Includes comparison between fused AVX2 path and scalar fallback.
//

#include "mlas_qkv_quant.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/qkv_quant_kernel.h"
#include "core/util/thread_utils.h"
#include "benchmark/benchmark.h"
#include "bench_util.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

// Generate random FP32 data in [-1, 1]
std::vector<float> RandomFloats(size_t n, unsigned seed) {
  std::default_random_engine gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(gen);
  return v;
}

// Quantize a FP32 matrix [rows, cols] into the packed format and compute scales.
void QuantizeMatrix(
    const float* src, size_t rows, size_t cols,
    MLAS_KV_QUANT_TYPE qt,
    std::vector<uint8_t>& dst_buf,
    std::vector<float>& scales_buf) {
  bool per_channel = (qt == MLAS_KV_QUANT_TYPE::S8_PerChannel ||
                      qt == MLAS_KV_QUANT_TYPE::S4_PerChannel);
  bool int4 = (qt == MLAS_KV_QUANT_TYPE::S4_PerTensor ||
               qt == MLAS_KV_QUANT_TYPE::S4_PerChannel);

  // Compute scales: max abs per column (per_channel) or global (per_tensor)
  size_t num_scales = per_channel ? cols : 1;
  scales_buf.resize(num_scales, 0.0f);

  if (per_channel) {
    for (size_t c = 0; c < cols; ++c) {
      float max_abs = 0.0f;
      for (size_t r = 0; r < rows; ++r) {
        max_abs = std::max(max_abs, std::abs(src[r * cols + c]));
      }
      float qmax = int4 ? 7.0f : 127.0f;
      scales_buf[c] = max_abs / qmax;
    }
  } else {
    float max_abs = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
      max_abs = std::max(max_abs, std::abs(src[i]));
    }
    float qmax = int4 ? 7.0f : 127.0f;
    scales_buf[0] = max_abs / qmax;
  }

  size_t row_bytes = MlasKVQuantPackedRowBytes(qt, cols);
  dst_buf.resize(rows * row_bytes);
  MlasKVQuantize(src, dst_buf.data(), rows, cols, cols, qt, scales_buf.data(), nullptr);
}

}  // namespace

//
// Benchmark MlasQKGemm: C[M,N] = alpha * A[M,K] * B^T[K,N]
//
static void BM_QKGemm(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const auto qt = static_cast<MLAS_KV_QUANT_TYPE>(state.range(3));

  // A = query [M, K]
  auto A = RandomFloats(M * K, 42);
  // B_fp = K-cache [N, K] (row-major, N = total_seqlen)
  auto B_fp = RandomFloats(N * K, 123);

  std::vector<uint8_t> B_quant;
  std::vector<float> scales;
  QuantizeMatrix(B_fp.data(), N, K, qt, B_quant, scales);

  std::vector<float> C(M * N, 0.0f);
  const float alpha = 1.0f / std::sqrt(static_cast<float>(K));

  // Warmup
  MlasQKGemm(M, N, K, alpha, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);

  for (auto _ : state) {
    MlasQKGemm(M, N, K, alpha, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);
  }

  // Report throughput
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * M * N * K * 2);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (M * K * sizeof(float) + N * MlasKVQuantPackedRowBytes(qt, K)));
}

//
// Benchmark MlasSVGemm: C[M,N] = A[M,K] * B[K,N]
//
static void BM_SVGemm(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const auto qt = static_cast<MLAS_KV_QUANT_TYPE>(state.range(3));

  // A = attention probs [M, K] (K = total_seqlen for SV)
  auto A = RandomFloats(M * K, 42);
  // B_fp = V-cache [K, N] (row-major, N = head_size)
  auto B_fp = RandomFloats(K * N, 456);

  std::vector<uint8_t> B_quant;
  std::vector<float> scales;
  QuantizeMatrix(B_fp.data(), K, N, qt, B_quant, scales);

  std::vector<float> C(M * N, 0.0f);

  // Warmup
  MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, 0.0f, nullptr);

  for (auto _ : state) {
    MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, 0.0f, nullptr);
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * M * N * K * 2);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (M * K * sizeof(float) + K * MlasKVQuantPackedRowBytes(qt, N)));
}

// QKGemm benchmark configurations
// Args: M, N (total_seqlen), K (head_size), QuantType
static void QKGemmArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N_seqlen", "K_head", "QuantType"});
  // Decoding (M=1) and prefill (M=128) with typical shapes
  for (int qt : {0, 1, 2, 3}) {    // S8_PerTensor, S8_PerChannel, S4_PerTensor, S4_PerChannel
    for (int K : {64, 128}) {      // head_size
      for (int N : {512, 2048}) {  // total_seqlen
        b->Args({1, N, K, qt});    // decoding
        b->Args({128, N, K, qt});  // prefill
      }
    }
  }
}

// SVGemm benchmark configurations
// Args: M, N (head_size), K (total_seqlen), QuantType
static void SVGemmArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N_head", "K_seqlen", "QuantType"});
  for (int qt : {0, 1, 2, 3}) {
    for (int N : {64, 128}) {      // head_size
      for (int K : {512, 2048}) {  // total_seqlen
        b->Args({1, N, K, qt});    // decoding
        b->Args({128, N, K, qt});  // prefill
      }
    }
  }
}

BENCHMARK(BM_QKGemm)->Apply(QKGemmArgs)->UseRealTime();
BENCHMARK(BM_SVGemm)->Apply(SVGemmArgs)->UseRealTime();

//
// Scalar fallback benchmarks: temporarily null the dispatch to force the scalar path.
//
static void BM_QKGemm_Scalar(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const auto qt = static_cast<MLAS_KV_QUANT_TYPE>(state.range(3));

  auto A = RandomFloats(M * K, 42);
  auto B_fp = RandomFloats(N * K, 123);

  std::vector<uint8_t> B_quant;
  std::vector<float> scales;
  QuantizeMatrix(B_fp.data(), N, K, qt, B_quant, scales);

  std::vector<float> C(M * N, 0.0f);
  const float alpha = 1.0f / std::sqrt(static_cast<float>(K));

  // Save and null the dispatch
  auto& platform = GetMlasPlatform();
  auto* saved_dispatch = platform.KVQuantGemmDispatch;
  platform.KVQuantGemmDispatch = nullptr;

  MlasQKGemm(M, N, K, alpha, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);

  for (auto _ : state) {
    MlasQKGemm(M, N, K, alpha, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);
  }

  // Restore dispatch
  platform.KVQuantGemmDispatch = saved_dispatch;

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * M * N * K * 2);
}

static void BM_SVGemm_Scalar(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const auto qt = static_cast<MLAS_KV_QUANT_TYPE>(state.range(3));

  auto A = RandomFloats(M * K, 42);
  auto B_fp = RandomFloats(K * N, 456);

  std::vector<uint8_t> B_quant;
  std::vector<float> scales;
  QuantizeMatrix(B_fp.data(), K, N, qt, B_quant, scales);

  std::vector<float> C(M * N, 0.0f);

  auto& platform = GetMlasPlatform();
  auto* saved_dispatch = platform.KVQuantGemmDispatch;
  platform.KVQuantGemmDispatch = nullptr;

  MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, 0.0f, nullptr);

  for (auto _ : state) {
    MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, 0.0f, nullptr);
  }

  platform.KVQuantGemmDispatch = saved_dispatch;

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * M * N * K * 2);
}

// Use a subset of shapes for scalar comparison (it's slow)
static void ScalarArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "QuantType"});
  for (int qt : {0, 2}) {          // S8_PerTensor and S4_PerTensor as representative
    b->Args({1, 512, 128, qt});    // decoding
    b->Args({128, 512, 128, qt});  // prefill
  }
}

BENCHMARK(BM_QKGemm_Scalar)->Apply(ScalarArgs)->UseRealTime();
BENCHMARK(BM_SVGemm_Scalar)->Apply(ScalarArgs)->UseRealTime();

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)
//
// AVX2-only benchmarks: force the AVX2 dispatch to compare against AVX512-VNNI.
//
static void BM_QKGemm_Avx2(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const auto qt = static_cast<MLAS_KV_QUANT_TYPE>(state.range(3));

  auto A = RandomFloats(M * K, 42);
  auto B_fp = RandomFloats(N * K, 123);

  std::vector<uint8_t> B_quant;
  std::vector<float> scales;
  QuantizeMatrix(B_fp.data(), N, K, qt, B_quant, scales);

  std::vector<float> C(M * N, 0.0f);
  const float alpha = 1.0f / std::sqrt(static_cast<float>(K));

  // Force AVX2 dispatch
  auto& platform = GetMlasPlatform();
  auto* saved_dispatch = platform.KVQuantGemmDispatch;
  platform.KVQuantGemmDispatch = &MlasKVQuantGemmDispatchAvx2;

  MlasQKGemm(M, N, K, alpha, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);

  for (auto _ : state) {
    MlasQKGemm(M, N, K, alpha, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);
  }

  platform.KVQuantGemmDispatch = saved_dispatch;

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * M * N * K * 2);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (M * K * sizeof(float) + N * MlasKVQuantPackedRowBytes(qt, K)));
}

static void BM_SVGemm_Avx2(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const auto qt = static_cast<MLAS_KV_QUANT_TYPE>(state.range(3));

  auto A = RandomFloats(M * K, 42);
  auto B_fp = RandomFloats(K * N, 456);

  std::vector<uint8_t> B_quant;
  std::vector<float> scales;
  QuantizeMatrix(B_fp.data(), K, N, qt, B_quant, scales);

  std::vector<float> C(M * N, 0.0f);

  auto& platform = GetMlasPlatform();
  auto* saved_dispatch = platform.KVQuantGemmDispatch;
  platform.KVQuantGemmDispatch = &MlasKVQuantGemmDispatchAvx2;

  MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, 0.0f, nullptr);

  for (auto _ : state) {
    MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, 0.0f, nullptr);
  }

  platform.KVQuantGemmDispatch = saved_dispatch;

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * M * N * K * 2);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (M * K * sizeof(float) + K * MlasKVQuantPackedRowBytes(qt, N)));
}

BENCHMARK(BM_QKGemm_Avx2)->Apply(ScalarArgs)->UseRealTime();
BENCHMARK(BM_SVGemm_Avx2)->Apply(ScalarArgs)->UseRealTime();

#endif  // MLAS_TARGET_AMD64 || MLAS_TARGET_IX86

//
// Flash Attention vs Naive (full materialization) benchmark.
// Compares MlasFlashAttentionQuantizedKV against the manual
// QKGemm + softmax + SVGemm pipeline for realistic GQA shapes.
//
// Args: batch_size, num_heads, kv_num_heads, seq_len, total_seqlen, head_size, QuantType
//

static MLAS_THREADPOOL* GetBenchThreadPool() {
  static OrtThreadPoolParams tpo;
  static bool init = [&]() {
    tpo.thread_pool_size = 8;
    tpo.auto_set_affinity = true;
    return true;
  }();
  (void)init;
  static std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));
  return tp.get();
}

// Naive path: QKGemm + row-wise softmax + SVGemm (full attention matrix materialized)
static void BM_GQA_Naive(benchmark::State& state) {
  const int batch_size = static_cast<int>(state.range(0));
  const int num_heads = static_cast<int>(state.range(1));
  const int kv_num_heads = static_cast<int>(state.range(2));
  const int seq_len = static_cast<int>(state.range(3));
  const int total_seqlen = static_cast<int>(state.range(4));
  const int head_size = static_cast<int>(state.range(5));
  const auto qt = static_cast<MLAS_KV_QUANT_TYPE>(state.range(6));

  const int groups = num_heads / kv_num_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

  // Allocate query [B, N, S, H]
  auto query = RandomFloats(static_cast<size_t>(batch_size) * num_heads * seq_len * head_size, 42);

  // Allocate and quantize K cache [B, kv_N, T, H]
  auto k_fp = RandomFloats(static_cast<size_t>(batch_size) * kv_num_heads * total_seqlen * head_size, 123);
  auto v_fp = RandomFloats(static_cast<size_t>(batch_size) * kv_num_heads * total_seqlen * head_size, 456);

  size_t k_row_bytes = MlasKVQuantPackedRowBytes(qt, head_size);
  size_t v_row_bytes = MlasKVQuantPackedRowBytes(qt, head_size);
  size_t k_cache_size = static_cast<size_t>(batch_size) * kv_num_heads * total_seqlen * k_row_bytes;
  size_t v_cache_size = static_cast<size_t>(batch_size) * kv_num_heads * total_seqlen * v_row_bytes;

  std::vector<uint8_t> k_cache(k_cache_size);
  std::vector<uint8_t> v_cache(v_cache_size);

  bool per_channel = (qt == MLAS_KV_QUANT_TYPE::S8_PerChannel || qt == MLAS_KV_QUANT_TYPE::S4_PerChannel);
  size_t num_scales = per_channel ? static_cast<size_t>(kv_num_heads * head_size) : 1;
  std::vector<float> k_scale(num_scales, 0.01f);
  std::vector<float> v_scale(num_scales, 0.01f);

  // Quantize K and V caches per kv-head
  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < kv_num_heads; ++h) {
      size_t offset_fp = (static_cast<size_t>(b) * kv_num_heads + h) * total_seqlen * head_size;
      size_t offset_q = (static_cast<size_t>(b) * kv_num_heads + h) * total_seqlen * k_row_bytes;
      MlasKVQuantize(k_fp.data() + offset_fp, k_cache.data() + offset_q,
                     total_seqlen, head_size, head_size, qt,
                     per_channel ? k_scale.data() + h * head_size : k_scale.data(), nullptr);
      offset_q = (static_cast<size_t>(b) * kv_num_heads + h) * total_seqlen * v_row_bytes;
      MlasKVQuantize(v_fp.data() + offset_fp, v_cache.data() + offset_q,
                     total_seqlen, head_size, head_size, qt,
                     per_channel ? v_scale.data() + h * head_size : v_scale.data(), nullptr);
    }
  }

  // Allocate working buffers: scores[B*N, S, T] (one per head) + output[B, S, N, H]
  std::vector<float> scores(static_cast<size_t>(batch_size) * num_heads * seq_len * total_seqlen);
  std::vector<float> output(static_cast<size_t>(batch_size) * seq_len * num_heads * head_size, 0.0f);

  auto* tp = GetBenchThreadPool();
  const ptrdiff_t loop_len = batch_size * num_heads;

  for (auto _ : state) {
    // Pass 1: QK GEMM + Softmax (matches operator's first TryParallelFor)
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, loop_len, [&](std::ptrdiff_t i) {
          const int b = static_cast<int>(i) / num_heads;
          const int h = static_cast<int>(i) % num_heads;
          const int kv_h = h / groups;
          float* my_scores = scores.data() + static_cast<size_t>(i) * seq_len * total_seqlen;
          const float* q_ptr = query.data() + (static_cast<size_t>(b) * num_heads + h) * seq_len * head_size;
          const uint8_t* k_ptr = k_cache.data() + (static_cast<size_t>(b) * kv_num_heads + kv_h) * total_seqlen * k_row_bytes;

          // QK GEMM: scores[S, T] = scale * Q[S,H] * K[T,H]^T
          MlasQKGemm(seq_len, total_seqlen, head_size, scale,
                     q_ptr, head_size, k_ptr, qt,
                     per_channel ? k_scale.data() + kv_h * head_size : k_scale.data(),
                     my_scores, total_seqlen, nullptr);

          // Causal masking + MLAS-optimized softmax (matches operator)
          for (int s = 0; s < seq_len; ++s) {
            float* row = my_scores + s * total_seqlen;
            int valid_len = total_seqlen - seq_len + s + 1;
            // Zero out future positions (operator sets them to 0 before softmax)
            for (int t = valid_len; t < total_seqlen; ++t) row[t] = 0.f;
            // Use MLAS optimized softmax on valid range only
            MlasComputeSoftmax(row, row, static_cast<size_t>(1),
                               static_cast<size_t>(valid_len), false, false, 0.0f, nullptr);
          }
        });

    // Pass 2: SV GEMM (matches operator's second TryParallelFor)
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, loop_len, [&](std::ptrdiff_t i) {
          const int b = static_cast<int>(i) / num_heads;
          const int h = static_cast<int>(i) % num_heads;
          const int kv_h = h / groups;
          float* my_scores = scores.data() + static_cast<size_t>(i) * seq_len * total_seqlen;
          const uint8_t* v_ptr = v_cache.data() + (static_cast<size_t>(b) * kv_num_heads + kv_h) * total_seqlen * v_row_bytes;
          float* out_ptr = output.data() + (static_cast<size_t>(b) * seq_len * num_heads + h) * head_size;

          // SV GEMM: out[S, H] = scores[S,T] * V[T,H]
          MlasSVGemm(seq_len, head_size, total_seqlen,
                     my_scores, total_seqlen, v_ptr, qt,
                     per_channel ? v_scale.data() + kv_h * head_size : v_scale.data(),
                     out_ptr, num_heads * head_size, 0.0f, nullptr);
        });
    benchmark::DoNotOptimize(output.data());
  }

  int64_t flops = static_cast<int64_t>(batch_size) * num_heads * seq_len *
                  (2LL * total_seqlen * head_size + 2LL * total_seqlen * head_size);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

// Flash path: MlasFlashAttentionQuantizedKV (tiled, online softmax)
static void BM_GQA_Flash(benchmark::State& state) {
  const int batch_size = static_cast<int>(state.range(0));
  const int num_heads = static_cast<int>(state.range(1));
  const int kv_num_heads = static_cast<int>(state.range(2));
  const int seq_len = static_cast<int>(state.range(3));
  const int total_seqlen = static_cast<int>(state.range(4));
  const int head_size = static_cast<int>(state.range(5));
  const auto qt = static_cast<MLAS_KV_QUANT_TYPE>(state.range(6));

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  bool per_channel = (qt == MLAS_KV_QUANT_TYPE::S8_PerChannel || qt == MLAS_KV_QUANT_TYPE::S4_PerChannel);

  // Allocate query [B, N, S, H] in BNSH layout
  auto query = RandomFloats(static_cast<size_t>(batch_size) * num_heads * seq_len * head_size, 42);

  // Allocate and quantize K/V caches
  auto k_fp = RandomFloats(static_cast<size_t>(batch_size) * kv_num_heads * total_seqlen * head_size, 123);
  auto v_fp = RandomFloats(static_cast<size_t>(batch_size) * kv_num_heads * total_seqlen * head_size, 456);

  size_t k_row_bytes = MlasKVQuantPackedRowBytes(qt, head_size);
  size_t v_row_bytes = MlasKVQuantPackedRowBytes(qt, head_size);
  std::vector<uint8_t> k_cache(static_cast<size_t>(batch_size) * kv_num_heads * total_seqlen * k_row_bytes);
  std::vector<uint8_t> v_cache(static_cast<size_t>(batch_size) * kv_num_heads * total_seqlen * v_row_bytes);

  size_t num_scales = per_channel ? static_cast<size_t>(kv_num_heads * head_size) : 1;
  std::vector<float> k_scale(num_scales, 0.01f);
  std::vector<float> v_scale(num_scales, 0.01f);

  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < kv_num_heads; ++h) {
      size_t offset_fp = (static_cast<size_t>(b) * kv_num_heads + h) * total_seqlen * head_size;
      size_t offset_q = (static_cast<size_t>(b) * kv_num_heads + h) * total_seqlen * k_row_bytes;
      MlasKVQuantize(k_fp.data() + offset_fp, k_cache.data() + offset_q,
                     total_seqlen, head_size, head_size, qt,
                     per_channel ? k_scale.data() + h * head_size : k_scale.data(), nullptr);
      offset_q = (static_cast<size_t>(b) * kv_num_heads + h) * total_seqlen * v_row_bytes;
      MlasKVQuantize(v_fp.data() + offset_fp, v_cache.data() + offset_q,
                     total_seqlen, head_size, head_size, qt,
                     per_channel ? v_scale.data() + h * head_size : v_scale.data(), nullptr);
    }
  }

  // Output [B, S, N, H]
  std::vector<float> output(static_cast<size_t>(batch_size) * seq_len * num_heads * head_size, 0.0f);

  // Fixed block sizes for reproducible benchmarks (operator computes from L2 cache size)
  int q_block_size = 64;
  int kv_block_size = 256;

  // Thread pool
  auto* tp = GetBenchThreadPool();
  int thread_count = 8;

  // Flash decoding: for decode (seq_len=1), partition KV across threads
  int kv_chunk_count = (total_seqlen + kv_block_size - 1) / kv_block_size;
  bool use_flash_decoding = (seq_len == 1 &&
                             batch_size * num_heads < thread_count &&
                             kv_chunk_count > 1);

  // Working buffer
  size_t buffer_size_per_thread;
  size_t partials_buffer_bytes = 0;
  if (use_flash_decoding) {
    buffer_size_per_thread = static_cast<size_t>(kv_block_size) * sizeof(float);
    partials_buffer_bytes = static_cast<size_t>(batch_size) * num_heads *
                            kv_chunk_count * (2 + head_size) * sizeof(float);
  } else {
    buffer_size_per_thread =
        (static_cast<size_t>(q_block_size) * 2 +                                   // l + m
         static_cast<size_t>(q_block_size) * static_cast<size_t>(kv_block_size) +  // scores
         static_cast<size_t>(q_block_size) * static_cast<size_t>(head_size)) *     // temp_output
        sizeof(float);
  }
  size_t total_buffer_floats = (buffer_size_per_thread * thread_count + partials_buffer_bytes) / sizeof(float);
  std::vector<float> buffer(total_buffer_floats);
  float* partials_ptr = use_flash_decoding
                            ? buffer.data() + (buffer_size_per_thread * thread_count) / sizeof(float)
                            : nullptr;

  MlasFlashAttentionQuantizedKVArgs args{};
  args.batch_size = batch_size;
  args.num_heads = num_heads;
  args.kv_num_heads = kv_num_heads;
  args.sequence_length = seq_len;
  args.total_seqlen = total_seqlen;
  args.head_size = head_size;
  args.past_seqlen = total_seqlen - seq_len;
  args.local_window_size = -1;
  args.seqlen_present_kv = total_seqlen;
  args.q_block_size = q_block_size;
  args.kv_block_size = kv_block_size;
  args.scale = scale;
  args.quant_type = qt;
  args.per_channel_k = per_channel;
  args.per_channel_v = per_channel;
  args.thread_count = thread_count;
  args.buffer = buffer.data();
  args.buffer_size_per_thread = buffer_size_per_thread;
  args.query = query.data();
  args.q_batch_stride = static_cast<size_t>(num_heads) * seq_len * head_size;
  args.k_cache = k_cache.data();
  args.v_cache = v_cache.data();
  args.k_scale = k_scale.data();
  args.v_scale = v_scale.data();
  args.output = output.data();
  args.attention_bias = nullptr;
  args.attention_bias_seqlen_stride = 0;
  args.attention_bias_broadcast_batch = true;
  args.attention_bias_broadcast_head = true;
  args.flash_decoding_partials = partials_ptr;
  args.kv_chunk_count = kv_chunk_count;

  // Warmup
  MlasFlashAttentionQuantizedKV(&args, tp);

  for (auto _ : state) {
    MlasFlashAttentionQuantizedKV(&args, tp);
    benchmark::DoNotOptimize(output.data());
  }

  int64_t flops = static_cast<int64_t>(batch_size) * num_heads * seq_len *
                  (2LL * total_seqlen * head_size + 2LL * total_seqlen * head_size);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * flops);
}

// Flash vs Naive benchmark configurations
// Args: batch, num_heads, kv_num_heads, seq_len, total_seqlen, head_size, QuantType
static void FlashGQAArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"B", "N", "N_kv", "S", "T", "H", "QType"});
  // INT8 per-tensor (qt=0), INT8 per-channel (qt=1)
  for (int qt : {0, 1}) {
    // Prompt (prefill): seq_len = total_seqlen
    for (int T : {512, 1024, 2048, 4096}) {
      b->Args({1, 16, 8, T, T, 128, qt});  // B=1, GQA ratio 2
    }
    // Decode: seq_len=1, past grows
    for (int T : {512, 1024, 2048, 4096}) {
      b->Args({1, 16, 8, 1, T, 128, qt});  // B=1, decode
    }
    // Larger batch decode
    b->Args({4, 16, 8, 1, 2048, 128, qt});
    // Flash decoding cases: B*N < thread_count (8), triggers KV partitioning
    for (int T : {512, 1024, 2048, 4096}) {
      b->Args({1, 4, 4, 1, T, 128, qt});  // B=1, N=4, flash decoding enabled
    }
  }
}

BENCHMARK(BM_GQA_Naive)->Apply(FlashGQAArgs)->UseRealTime();
BENCHMARK(BM_GQA_Flash)->Apply(FlashGQAArgs)->UseRealTime();
