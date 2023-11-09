// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas_q4.h"
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

  size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
  MlasBlockwiseQuantizedBufferSizes(
      BlkBitWidth, BlkLen, /* columnwise */ true,
      static_cast<int>(K), static_cast<int>(N),
      QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(threads);
  tpo.auto_set_affinity = true;

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto A = RandomVectorUniform(static_cast<size_t>(M * K), -1.0f, 1.0f);
  auto B = RandomVectorUniform(static_cast<size_t>(K * N), -1.0f, 1.0f);
  std::vector<float> C(static_cast<size_t>(M * N));

  std::vector<uint8_t> QuantBData(QuantBDataSizeInBytes);
  std::vector<float> QuantBScale(QuantBScaleSize);
  std::vector<uint8_t> QuantBZeroPoint(Symmetric ? 0 : QuantBZeroPointSizeInBytes);

  MlasQuantizeBlockwise<float, BlkBitWidth>(QuantBData.data(), QuantBScale.data(),
                                            Symmetric ? nullptr : QuantBZeroPoint.data(),
                                            B.data(), BlkLen, /* columnwise */ true,
                                            static_cast<int>(K), static_cast<int>(N), static_cast<int>(N),
                                            tp.get());

  MLAS_SQNBIT_GEMM_DATA_PARAMS params{};
  params.A = A.data();
  params.lda = K;
  params.QuantBData = QuantBData.data();
  params.QuantBScale = QuantBScale.data();
  params.QuantBZeroPoint = Symmetric ? nullptr : QuantBZeroPoint.data();
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
BENCHMARK(SQNBITGEMM<4, 256, false>)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK(SQNBITGEMM<4, 256, true>)->Apply(GemmSizeProducts)->UseRealTime();
