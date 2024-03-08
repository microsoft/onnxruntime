// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas_q4.h"
#include "mlas_qnbit.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "benchmark/benchmark.h"

#include "bench_util.h"
#include "core/common/narrow.h"
#include "core/util/thread_utils.h"
#include "core/platform/env_var_utils.h"

using onnxruntime::narrow;

template <size_t BlkBitWidth>
void RunSQNBitGemmBenchmark(size_t BlkLen,
                            size_t M, size_t N, size_t K,
                            size_t Threads,
                            bool Symmetric,
                            MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
                            benchmark::State& state) {
  if (!MlasIsSQNBitGemmAvailable(BlkBitWidth, BlkLen, ComputeType)) {
    state.SkipWithMessage("SQNBitGemm is not available with the given configuration on the current machine.");
    return;
  }

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

  std::unique_ptr<std::byte[]> Workspace;
  if (const auto WorkspaceSize = MlasSQNBitGemmBatchWorkspaceSize(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType);
      WorkspaceSize > 0) {
    Workspace = std::make_unique<std::byte[]>(WorkspaceSize);
  }

  std::unique_ptr<std::byte[]> PackedQuantBData;
  if (const auto PackedQuantBDataSize = MlasSQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen, ComputeType);
      PackedQuantBDataSize > 0) {
    PackedQuantBData = std::make_unique<std::byte[]>(PackedQuantBDataSize);
    MlasSQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, ComputeType, QuantBData.data(), PackedQuantBData.get(),
                                 tp.get());
  }

  MLAS_SQNBIT_GEMM_DATA_PARAMS params{};
  params.A = A.data();
  params.lda = K;
  params.QuantBData = PackedQuantBData != nullptr
                          ? static_cast<const void*>(PackedQuantBData.get())
                          : static_cast<const void*>(QuantBData.data());
  params.QuantBScale = QuantBScale.data();
  params.QuantBZeroPoint = Symmetric ? nullptr : QuantBZeroPoint.data();
  params.Bias = nullptr;
  params.C = C.data();
  params.ldc = N;

  // warm up run
  MlasSQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType, &params, Workspace.get(), tp.get());

  for (auto _ : state) {
    MlasSQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType, &params, Workspace.get(), tp.get());
  }
}

template <size_t BlkBitWidth>
void SQNBITGEMM(benchmark::State& state) {
  const auto BlkLen = narrow<size_t>(state.range(0));
  const auto M = narrow<size_t>(state.range(1));
  const auto N = narrow<size_t>(state.range(2));
  const auto K = narrow<size_t>(state.range(3));
  const auto Threads = narrow<size_t>(state.range(4));
  const auto Symmetric = narrow<bool>(state.range(5));
  const auto ComputeType = static_cast<MLAS_SQNBIT_GEMM_COMPUTE_TYPE>(state.range(6));

  RunSQNBitGemmBenchmark<BlkBitWidth>(BlkLen, M, N, K, Threads, Symmetric, ComputeType, state);
}

// This test gets benchmark arguments from environment variables.
template <size_t BlkBitWidth>
void SQNBITGEMM_ENV(benchmark::State& state) {
  using onnxruntime::ParseEnvironmentVariableWithDefault;

  const auto BlkLen = ParseEnvironmentVariableWithDefault<size_t>("ORT_SQNBITGEMM_BLKLEN", 32);
  const auto M = ParseEnvironmentVariableWithDefault<size_t>("ORT_SQNBITGEMM_M", 1);
  const auto N = ParseEnvironmentVariableWithDefault<size_t>("ORT_SQNBITGEMM_N", 4096);
  const auto K = ParseEnvironmentVariableWithDefault<size_t>("ORT_SQNBITGEMM_K", 4096);
  const auto Threads = ParseEnvironmentVariableWithDefault<size_t>("ORT_SQNBITGEMM_THREADS", 1);
  const auto Symmetric = ParseEnvironmentVariableWithDefault<bool>("ORT_SQNBITGEMM_SYMMETRIC", true);
  const auto ComputeType = ParseEnvironmentVariableWithDefault<int32_t>("ORT_SQNBITGEMM_COMPUTE_TYPE",
                                                                        static_cast<int32_t>(CompFp32));

  RunSQNBitGemmBenchmark<BlkBitWidth>(BlkLen, M, N, K, Threads, Symmetric,
                                      static_cast<MLAS_SQNBIT_GEMM_COMPUTE_TYPE>(ComputeType),
                                      state);

  std::ostringstream s;
  s << "BlkBitWidth:" << BlkBitWidth << "/BlkLen:" << BlkLen
    << "/M:" << M << "/N:" << N << "/K:" << K
    << "/Threads:" << Threads << "/Symmetric:" << Symmetric << "/ComputeType:" << ComputeType;
  state.SetLabel(s.str());
}

static void SQNBitGemmArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"BlkLen", "M", "N", "K", "Threads", "Symmetric", "ComputeType"});

  b->ArgsProduct({
      {16, 32, 64, 128, 256},                  // BlkLen
      {1, 1024, 2048},                         // M
      {4096, 11008},                           // N
      {4096, 11008},                           // K
      {1, 8},                                  // Threads
      {int64_t{false}, int64_t{true}},         // Symmetric
      {int64_t{CompFp32}, int64_t{CompInt8}},  // ComputeType
  });
}

BENCHMARK(SQNBITGEMM<4>)->Apply(SQNBitGemmArgs)->UseRealTime();
BENCHMARK(SQNBITGEMM_ENV<4>)->UseRealTime();
