// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas_q4.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <stdexcept>
#include <numeric>

static const std::vector<std::string> q4gemm_bench_arg_names = {"M", "N", "K", "Threads"};

void Q4GEMM(benchmark::State& state, MLAS_BLK_QUANT_TYPE qtype) {
  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  if (state.range(3) <= 0) throw std::invalid_argument("Threads must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t threads = static_cast<size_t>(state.range(3));
  const size_t pack_b_size = MlasQ4GemmPackBSize(qtype, N, K);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = int(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto A1 = RandomVectorUniform(static_cast<size_t>(M * K), -1.0f, 1.0f);
  auto B1 = RandomVectorUniform(static_cast<size_t>(N * K), -1.0f, 1.0f);
  std::vector<float> C1(static_cast<size_t>(M * N));

  std::vector<uint8_t> B1_packed(pack_b_size);
  MlasQ4GemmPackB(qtype, B1_packed.data(), B1.data(), N, K, N);

  MLAS_Q4_GEMM_DATA_PARAMS params1;
  params1.A = A1.data();
  params1.lda = K;
  params1.Bias = nullptr;
  params1.C = C1.data();
  params1.ldc = N;
  params1.B = B1_packed.data();
  params1.OutputProcessor = nullptr;

  MlasQ4GemmBatch(qtype, M, N, K, 1, &params1, tp.get());

  for (auto _ : state) {
    MlasQ4GemmBatch(qtype, M, N, K, 1, &params1, tp.get());
  }
}

void Q8Q4GEMM(benchmark::State& state, MLAS_BLK_QUANT_TYPE qtype) {
  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  if (state.range(3) <= 0) throw std::invalid_argument("Threads must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t threads = static_cast<size_t>(state.range(3));
  const size_t pack_b_size = MlasQ4GemmPackBSize(qtype, N, K);
  const size_t quant_a_size = MlasQ80BlkQuantSize(qtype, M, K);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = int(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto A1 = RandomVectorUniform(static_cast<size_t>(M * K), -1.0f, 1.0f);
  auto B1 = RandomVectorUniform(static_cast<size_t>(N * K), -1.0f, 1.0f);
  std::vector<float> C1(static_cast<size_t>(M * N));

  std::vector<uint8_t> B1_packed(pack_b_size);
  MlasQ4GemmPackB(qtype, B1_packed.data(), B1.data(), N, K, N);

  std::vector<int8_t> A1_quant(quant_a_size);

  MlasQ80BlkQuant(BlkQ4Sym, A1_quant.data(), A1.data(), M, K, K, tp.get());

  MLAS_Q8Q4_GEMM_DATA_PARAMS params1;
  params1.A = A1.data();
  params1.B = B1_packed.data();
  params1.Bias = nullptr;
  params1.C = C1.data();
  params1.ldc = N;
  params1.OutputProcessor = nullptr;

  MlasQ8Q4GemmBatch(qtype, M, N, K, 1, &params1, tp.get());

  for (auto _ : state) {
    MlasQ80BlkQuant(BlkQ4Sym, A1_quant.data(), A1.data(), M, K, K, tp.get());

    MLAS_Q8Q4_GEMM_DATA_PARAMS params;
    params.A = A1.data();
    params.B = B1_packed.data();
    params.Bias = nullptr;
    params.C = C1.data();
    params.ldc = N;
    params.OutputProcessor = nullptr;
    MlasQ8Q4GemmBatch(qtype, M, N, K, 1, &params, tp.get());
  }
}

static void GemmSizeProducts(benchmark::internal::Benchmark* b) {
  b->ArgNames(q4gemm_bench_arg_names);
  ArgsProduct(b, {{1, 1024, 2048}, {4096}, {4096}, {8}});
}

BENCHMARK_CAPTURE(Q4GEMM, Q4Sym, BlkQ4Sym)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM, Q4Zp8, BlkQ4Zp8)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM, Q4Sym128, BlkQ4Sym)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q8Q4GEMM, Q4Sym, BlkQ4Sym)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q8Q4GEMM, Q4Zp8, BlkQ4Zp8)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q8Q4GEMM, Q4Sym128, BlkQ4Zp8)->Apply(GemmSizeProducts)->UseRealTime();
