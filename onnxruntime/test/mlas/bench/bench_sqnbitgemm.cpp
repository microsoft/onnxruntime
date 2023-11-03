// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas_qnbit.h"

#include <stdexcept>

#include "benchmark/benchmark.h"

#include "bench_util.h"
#include "core/util/thread_utils.h"

template <size_t BlkBitWidth, size_t BlkLen, bool Symmetric>
void SQNBITGEMM(benchmark::State& state) {
  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  if (state.range(3) <= 0) throw std::invalid_argument("Threads must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t threads = static_cast<size_t>(state.range(3));

  size_t PackedBDataSize, PackedBScaleSize, PackedBZeroPointSize;
  MlasReferenceQNBitPacking<BlkBitWidth, BlkLen>::GetPackedBSizes(
      N, K,
      PackedBDataSize, PackedBScaleSize, &PackedBZeroPointSize);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(threads);
  tpo.auto_set_affinity = true;

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto A = RandomVectorUniform(static_cast<size_t>(M * K), -1.0f, 1.0f);
  auto B = RandomVectorUniform(static_cast<size_t>(N * K), -1.0f, 1.0f);
  std::vector<float> C(static_cast<size_t>(M * N));

  std::vector<uint8_t> PackedBData(PackedBDataSize);
  std::vector<float> PackedBScale(PackedBScaleSize);
  std::vector<uint8_t> PackedBZeroPoint(Symmetric ? 0 : PackedBZeroPointSize);

  MlasReferenceQNBitPacking<BlkBitWidth, BlkLen>::PackB(
      N, K, B.data(), N,
      PackedBData.data(), PackedBScale.data(), Symmetric ? nullptr : PackedBZeroPoint.data());

  MLAS_SQNBIT_GEMM_DATA_PARAMS params{};
  params.A = A.data();
  params.lda = K;
  params.PackedBData = PackedBData.data();
  params.PackedBScale = PackedBScale.data();
  params.PackedBZeroPoint = Symmetric ? nullptr : PackedBZeroPoint.data();
  params.Bias = nullptr;
  params.C = C.data();
  params.ldc = N;

  // warm up run
  MlasSQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, &params, tp.get());

  for (auto _ : state) {
    MlasSQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, &params, tp.get());
  }
}

static void GemmSizeProducts(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "Threads"});
  ArgsProduct(b, {{1, 1024, 2048}, {4096}, {4096}, {8}});
}

BENCHMARK(SQNBITGEMM<4, 16, false>)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK(SQNBITGEMM<4, 16, true>)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK(SQNBITGEMM<4, 32, false>)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK(SQNBITGEMM<4, 32, true>)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK(SQNBITGEMM<4, 64, false>)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK(SQNBITGEMM<4, 64, true>)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK(SQNBITGEMM<4, 128, false>)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK(SQNBITGEMM<4, 128, true>)->Apply(GemmSizeProducts)->UseRealTime();
