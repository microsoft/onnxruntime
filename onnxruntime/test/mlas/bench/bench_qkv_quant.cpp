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
  MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);

  for (auto _ : state) {
    MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);
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

  MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);

  for (auto _ : state) {
    MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);
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

  MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);

  for (auto _ : state) {
    MlasSVGemm(M, N, K, A.data(), K, B_quant.data(), qt, scales.data(), C.data(), N, nullptr);
  }

  platform.KVQuantGemmDispatch = saved_dispatch;

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * M * N * K * 2);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (M * K * sizeof(float) + K * MlasKVQuantPackedRowBytes(qt, N)));
}

BENCHMARK(BM_QKGemm_Avx2)->Apply(ScalarArgs)->UseRealTime();
BENCHMARK(BM_SVGemm_Avx2)->Apply(ScalarArgs)->UseRealTime();

#endif  // MLAS_TARGET_AMD64 || MLAS_TARGET_IX86
