// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas_q4.h"
#include "mlas_qnbit.h"

#include <stdexcept>

#include "benchmark/benchmark.h"

#include "bench_util.h"
#include "core/util/thread_utils.h"
#include "core/common/narrow.h"

using onnxruntime::narrow;

template <size_t BlkBitWidth>
void SQNBITGEMM(benchmark::State& state) {
  const auto BlkLen = narrow<size_t>(state.range(0));
  const auto M = narrow<size_t>(state.range(1));
  const auto N = narrow<size_t>(state.range(2));
  const auto K = narrow<size_t>(state.range(3));
  const auto Threads = narrow<size_t>(state.range(4));
  const auto Symmetric = narrow<bool>(state.range(5));
  const auto ComputeType = static_cast<MLAS_SQNBITGEMM_COMPUTE_TYPE>(state.range(6));

  size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
  MlasBlockwiseQuantizedBufferSizes(
      BlkBitWidth, static_cast<int>(BlkLen), /* columnwise */ true,
      static_cast<int>(K), static_cast<int>(N),
      QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(Threads);
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
                                            B.data(), static_cast<int>(BlkLen), /* columnwise */ true,
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
  MlasSQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType, &params, tp.get());

  for (auto _ : state) {
    MlasSQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType, &params, tp.get());
  }
}

static void SQNBitGemmArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"BlkLen", "M", "N", "K", "Threads", "Symmetric", "ComputeType"});

  ArgsProductWithFilter(b,

                        {{16, 32, 64, 128, 256},                   // BlkLen
                         {1, 1024, 2048},                          // M
                         {4096, 11008},                            // N
                         {4096, 11008},                            // K
                         {8},                                      // Threads
                         {int64_t{false}, int64_t{true}},          // Symmetric
                         {int64_t{CompFp32}, int64_t{CompInt8}}},  // ComputeType

                        [](const std::vector<int64_t>& args) {
                          return MlasIsSQNBitGemmAvailable(
                              // M, N, K
                              narrow<size_t>(args[1]), narrow<size_t>(args[2]), narrow<size_t>(args[3]),
                              // BlkBitWidth, BlkLen
                              4, narrow<size_t>(args[0]),
                              // ComputeType
                              static_cast<MLAS_SQNBITGEMM_COMPUTE_TYPE>(args[6]));
                        });
}

BENCHMARK(SQNBITGEMM<4>)->Apply(SQNBitGemmArgs)->UseRealTime();
