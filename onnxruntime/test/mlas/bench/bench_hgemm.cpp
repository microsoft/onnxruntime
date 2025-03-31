// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <stdexcept>
#include <numeric>

static const std::vector<std::string> hgemm_bench_arg_names = {"M", "N", "K"};

void HGEMM(benchmark::State& state, bool transA, bool transB) {
  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  auto A = RandomVectorUniform(static_cast<size_t>(M * K), MLAS_FP16(-1.0f), MLAS_FP16(1.0f));
  auto B = RandomVectorUniform(static_cast<size_t>(N * K), MLAS_FP16(-1.0f), MLAS_FP16(1.0f));
  std::vector<MLAS_FP16> C(static_cast<size_t>(M * N));

  MLAS_FP16 alpha = MLAS_FP16(1.0f);
  MLAS_FP16 beta = MLAS_FP16(0.0f);
  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = 8;
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));
  MlasGemm(
      transA ? CblasTrans : CblasNoTrans,
      transB ? CblasTrans : CblasNoTrans,
      static_cast<size_t>(M),
      static_cast<size_t>(N),
      static_cast<size_t>(K),
      A.data(),
      transA ? M : K,
      B.data(),
      transB ? K : N,
      C.data(),
      N,
      alpha.val,
      beta.val,
      tp.get());

  for (auto _ : state) {
    MlasGemm(
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        static_cast<size_t>(M),
        static_cast<size_t>(N),
        static_cast<size_t>(K),
        A.data(),
        transA ? M : K,
        B.data(),
        transB ? K : N,
        C.data(),
        N,
        alpha.val,
        beta.val,
        tp.get());
  }
}

static void GemmSizeWithOne(benchmark::internal::Benchmark* b) {
  b->ArgNames(hgemm_bench_arg_names);
  b->ArgsProduct({{1}, {63, 255, 1023}, {63, 255, 1023}});
  b->ArgsProduct({{63, 255, 1023}, {1}, {63, 255, 1023}});
  b->ArgsProduct({{63, 255, 1023}, {63, 255, 1023}, {1}});
}
BENCHMARK_CAPTURE(HGEMM, GEMV_TransB, false, true)->Apply(GemmSizeWithOne)->UseRealTime();
BENCHMARK_CAPTURE(HGEMM, GEMV_B, false, false)->Apply(GemmSizeWithOne)->UseRealTime();

static void GemmSizeProducts(benchmark::internal::Benchmark* b) {
  b->ArgNames(hgemm_bench_arg_names);
  b->ArgsProduct({{63, 255, 1023}, {63, 255, 1023}, {63, 255, 1023}});
}
BENCHMARK_CAPTURE(HGEMM, NORMAL_TransB, false, true)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(HGEMM, NORMAL_B, false, false)->Apply(GemmSizeProducts)->UseRealTime();

static void GemmLLMSizeProducts(benchmark::internal::Benchmark* b) {
  b->ArgNames(hgemm_bench_arg_names);
  b->ArgsProduct({{1, 1024, 2048}, {4096, 11008}, {4096, 11008}});
}
BENCHMARK_CAPTURE(HGEMM, LLM_TransB, false, true)->Apply(GemmLLMSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(HGEMM, LLM_B, false, false)->Apply(GemmLLMSizeProducts)->UseRealTime();
