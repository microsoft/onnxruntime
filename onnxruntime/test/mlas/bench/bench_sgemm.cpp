// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "bench_util.h"

#include <stdexcept>
#include <numeric>

static const std::vector<std::string> sgemm_bench_arg_names = {"M", "N", "K"};

void SGEMM(benchmark::State& state, bool pack_b, bool trans_a, bool trans_b, float alpha = 1.0f, float beta = 0.0f) {
  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));


  auto A = RandomVectorUniform(static_cast<size_t>(M * K), -1.0f, 1.0f);
  auto B = RandomVectorUniform(static_cast<size_t>(N * K), -1.0f, 1.0f);
  std::vector<float> C(static_cast<size_t>(M * N));

  if (pack_b) {
    size_t pack_b_size = MlasGemmPackBSize(N, K);
    std::vector<float> B_packed(pack_b_size);
    MlasGemmPackB(CblasNoTrans, N, K, B.data(), N, B_packed.data());

    MlasGemm(
        trans_a ? CblasTrans : CblasNoTrans,
        static_cast<size_t>(M),
        static_cast<size_t>(N),
        static_cast<size_t>(K),
        alpha,
        A.data(),
        trans_a ? M : K,
        B_packed.data(),
        beta,
        C.data(),
        N,
        nullptr);

    for (auto _ : state) {
      MlasGemm(
          trans_a ? CblasTrans : CblasNoTrans,
          static_cast<size_t>(M),
          static_cast<size_t>(N),
          static_cast<size_t>(K),
          alpha,
          A.data(),
          trans_a ? M : K,
          B_packed.data(),
          beta,
          C.data(),
          N,
          nullptr);
    }

  } else {
    MlasGemm(
        trans_a ? CblasTrans : CblasNoTrans,
        trans_b ? CblasTrans : CblasNoTrans,
        static_cast<size_t>(M),
        static_cast<size_t>(N),
        static_cast<size_t>(K),
        alpha,
        A.data(),
        trans_a ? M : K,
        B.data(),
        trans_b ? K : N,
        beta,
        C.data(),
        N,
        nullptr);

    for (auto _ : state) {
      MlasGemm(
          trans_a ? CblasTrans : CblasNoTrans,
          trans_b ? CblasTrans : CblasNoTrans,
          static_cast<size_t>(M),
          static_cast<size_t>(N),
          static_cast<size_t>(K),
          alpha,
          A.data(),
          trans_a ? M : K,
          B.data(),
          trans_b ? K : N,
          beta,
          C.data(),
          N,
          nullptr);
    }
  }
}

static void GemmSizeWithOne(benchmark::internal::Benchmark* b) {
  b->ArgNames(sgemm_bench_arg_names);
  ArgsProduct(b, {{1}, {63, 255, 1023}, {63, 255, 1023}});
  ArgsProduct(b, {{63, 255, 1023}, {1}, {63, 255, 1023}});
  ArgsProduct(b, {{63, 255, 1023}, {63, 255, 1023}, {1}});
}

static void GemmSizeProducts(benchmark::internal::Benchmark* b) {
  b->ArgNames(sgemm_bench_arg_names);
  ArgsProduct(b, {{63, 255, 1023}, {63, 255, 1023}, {63, 255, 1023}});
}

BENCHMARK_CAPTURE(SGEMM, NORMAL_NoTrans, false, false, false)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(SGEMM, NORMAL_TransA, false, true, false)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(SGEMM, NORMAL_TransB, false, false, true)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(SGEMM, NORMAL_ABTrans, false, true, true)->Apply(GemmSizeProducts)->UseRealTime();

BENCHMARK_CAPTURE(SGEMM, GEMV_NoTrans, false, false, false)->Apply(GemmSizeWithOne)->UseRealTime();
BENCHMARK_CAPTURE(SGEMM, GEMV_TransA, false, true, false)->Apply(GemmSizeWithOne)->UseRealTime();
BENCHMARK_CAPTURE(SGEMM, GEMV_TransB, false, false, true)->Apply(GemmSizeWithOne)->UseRealTime();
BENCHMARK_CAPTURE(SGEMM, GEMV_ABTrans, false, true, true)->Apply(GemmSizeWithOne)->UseRealTime();

BENCHMARK_CAPTURE(SGEMM, PACKB_NoTransA, true, false, false)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(SGEMM, PACKB_TransA, true, true, false)->Apply(GemmSizeProducts)->UseRealTime();
