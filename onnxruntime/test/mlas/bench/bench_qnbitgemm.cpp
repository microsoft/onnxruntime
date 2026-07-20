// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas_q4.h"
#include "mlas_qnbit.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <type_traits>

#include "benchmark/benchmark.h"

#include "bench_util.h"
#include "core/common/narrow.h"
#include "core/util/thread_utils.h"
#include "core/platform/env_var_utils.h"

template <typename AType, size_t BlkBitWidth>
void RunQNBitGemmBenchmark(size_t BlkLen,
                           size_t M, size_t N, size_t K,
                           size_t Threads,
                           bool Symmetric,
                           bool HasBias,
                           MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
                           benchmark::State& state) {
  if (!MlasIsQNBitGemmAvailable(BlkBitWidth, BlkLen, ComputeType)) {
    state.SkipWithMessage("QNBitGemm is not available with the given configuration on the current machine.");
    return;
  }

  size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
  MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(
      static_cast<int>(BlkLen), /* columnwise */ true,
      static_cast<int>(K), static_cast<int>(N),
      QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(Threads);
  tpo.auto_set_affinity = true;

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  const auto A = RandomVectorUniform(M * K, AType(-1.0f), AType(1.0f));
  const auto B = RandomVectorUniform(K * N, AType(-1.0f), AType(1.0f));

  const auto Bias = HasBias ? RandomVectorUniform(N, AType(-1.0f), AType(1.0f)) : std::vector<AType>();

  std::vector<AType> C(static_cast<size_t>(M * N));

  std::vector<uint8_t> QuantBData(QuantBDataSizeInBytes);
  std::vector<AType> QuantBScale(QuantBScaleSize);
  std::vector<uint8_t> QuantBZeroPoint(Symmetric ? 0 : QuantBZeroPointSizeInBytes);
  bool has_zp_input = !Symmetric;

  MlasQuantizeBlockwise<AType, BlkBitWidth>(QuantBData.data(), QuantBScale.data(),
                                            Symmetric ? nullptr : QuantBZeroPoint.data(),
                                            B.data(), static_cast<int>(BlkLen), /* columnwise */ true,
                                            static_cast<int>(K), static_cast<int>(N), static_cast<int>(N),
                                            tp.get());

  std::unique_ptr<std::byte[]> Workspace;
  if (const auto WorkspaceSize = MlasQNBitGemmBatchWorkspaceSize(M, N, K, 1, BlkBitWidth, BlkLen, !Symmetric, ComputeType, nullptr);
      WorkspaceSize > 0) {
    Workspace = std::make_unique<std::byte[]>(WorkspaceSize);
  }

  std::unique_ptr<std::byte[]> PackedQuantBData;
  if (const auto PackedQuantBDataSize = MlasQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen, !Symmetric, ComputeType, nullptr);
      PackedQuantBDataSize > 0) {
    PackedQuantBData = std::make_unique<std::byte[]>(PackedQuantBDataSize);
    MlasQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, ComputeType, QuantBData.data(), PackedQuantBData.get(),
                                QuantBScale.data(), has_zp_input, QuantBZeroPoint.data(),
                                tp.get(), nullptr);
  }

  MLAS_QNBIT_GEMM_DATA_PARAMS<AType> params{};
  params.A = A.data();
  params.lda = K;
  if (PackedQuantBData != nullptr)
    params.QuantBDataWorkspace = PackedQuantBData.get();
  else
    params.QuantBDataWorkspace = static_cast<const void*>(QuantBData.data());

  params.PackedQuantBData = PackedQuantBData.get();
  params.QuantBScale = QuantBScale.data();
  params.QuantBZeroPoint = Symmetric ? nullptr : QuantBZeroPoint.data();
  params.Bias = HasBias ? Bias.data() : nullptr;
  params.C = C.data();
  params.ldc = N;

  // warm up run
  MlasQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType, &params, Workspace.get(), tp.get(), nullptr);

  for (auto _ : state) {
    MlasQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType, &params, Workspace.get(), tp.get(), nullptr);
  }
}

template <typename AType, size_t BlkBitWidth>
void QNBITGEMM(benchmark::State& state) {
  using onnxruntime::narrow;

  const auto BlkLen = narrow<size_t>(state.range(0));
  const auto M = narrow<size_t>(state.range(1));
  const auto N = narrow<size_t>(state.range(2));
  const auto K = narrow<size_t>(state.range(3));
  const auto Threads = narrow<size_t>(state.range(4));
  const auto Symmetric = narrow<bool>(state.range(5));
  const bool HasBias = narrow<bool>(state.range(6));
  const auto ComputeType = static_cast<MLAS_QNBIT_GEMM_COMPUTE_TYPE>(state.range(7));

  RunQNBitGemmBenchmark<AType, BlkBitWidth>(BlkLen, M, N, K, Threads, Symmetric, HasBias, ComputeType, state);
}

template <typename AType>
static void QNBitGemmArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"BlkLen", "M", "N", "K", "Threads", "Symmetric", "HasBias", "ComputeType"});

  b->ArgsProduct({
      {128},                            // BlkLen
      {1, 4096},                        // M
      {4096, 11008},                    // N
      {4096, 11008},                    // K
      {1, 8},                           // Threads
      {int64_t{false}, int64_t{true}},  // Symmetric
      {int64_t{false}, int64_t{true}},  // HasBias
      std::is_same_v<AType, MLAS_FP16>
          ? std::vector<int64_t>{int64_t{HQNBIT_CompFp16}}
          : std::vector<int64_t>{int64_t{SQNBIT_CompFp32}, int64_t{SQNBIT_CompInt8}},  // ComputeType
  });
}

// Standard sweep for the native W2 kernel. W2 has fewer free dimensions
// than W4 (symmetric-only, SQNBIT_CompInt8 only), so the grid uses fixed
// values for those axes and sweeps the rest like QNBitGemmArgs.
static void QNBit2BitArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"BlkLen", "M", "N", "K", "Threads", "Symmetric", "HasBias", "ComputeType"});

  b->ArgsProduct({
      {32, 64, 128},                    // BlkLen      (W2 native kernel supports all three)
      {1, 4096},                        // M           (decode + prefill)
      {4096, 11008},                    // N
      {4096, 11008},                    // K
      {1, 8},                           // Threads
      {int64_t{true}},                  // Symmetric   (W2 native kernel constraint)
      {int64_t{false}, int64_t{true}},  // HasBias
      {int64_t{SQNBIT_CompInt8}},       // ComputeType (W2 native kernel constraint)
  });
}
BENCHMARK(QNBITGEMM<float, 4>)->Apply(QNBitGemmArgs<float>)->UseRealTime();
BENCHMARK(QNBITGEMM<float, 8>)->Apply(QNBitGemmArgs<float>)->UseRealTime();
BENCHMARK(QNBITGEMM<MLAS_FP16, 4>)->Apply(QNBitGemmArgs<MLAS_FP16>)->UseRealTime();
BENCHMARK(QNBITGEMM<float, 2>)->Apply(QNBit2BitArgs)->UseRealTime();

// Representative MatMulNBits shapes mirrored at 4-bit for a head-to-head
// comparison vs the W2 LUT path (LUTGEMM_COMPUTE/RealisticShapes). These
// shapes use BlkLen=64 and reflect proportions that appear in real W2
// production models. Five distinct (K, N) pairs:
//   (K=384,  N=1024): 20 nodes
//   (K=1024, N=192):  40 nodes
//   (K=1024, N=384):  20 nodes
//   (K=1024, N=4096): 20 nodes
//   (K=4096, N=1024): 20 nodes
// Both M=1 (decode) and M=128 (prefill) are exercised — paired with the W2
// rows below so we get a 3-way (W2 / W4 / W8) comparison at each M.
static void QNBitGemmRealisticShapesArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"BlkLen", "M", "N", "K", "Threads", "Symmetric", "HasBias", "ComputeType"});
  const int64_t BlkLen = 64;
  const int64_t Threads = 8;
  const int64_t Symmetric = 1;
  const int64_t HasBias = 1;
  for (int64_t M : {int64_t{1}, int64_t{128}}) {
    for (auto kn : {std::pair<int64_t, int64_t>{384, 1024},
                    std::pair<int64_t, int64_t>{1024, 192},
                    std::pair<int64_t, int64_t>{1024, 384},
                    std::pair<int64_t, int64_t>{1024, 4096},
                    std::pair<int64_t, int64_t>{4096, 1024}}) {
      for (int64_t ct : {int64_t{SQNBIT_CompFp32}, int64_t{SQNBIT_CompInt8}}) {
        b->Args({BlkLen, M, kn.second, kn.first, Threads, Symmetric, HasBias, ct});
      }
    }
  }
}

BENCHMARK(QNBITGEMM<float, 4>)->Apply(QNBitGemmRealisticShapesArgs)->UseRealTime();

// 2-bit weight rows for the same representative shapes. Exercises the AVX-512
// W2 native path (VNNI variant on AVX-512-VNNI hosts; non-VNNI variant on
// AVX-512BW hosts). W2 is registered only for SQNBIT_CompInt8 and BlkLen=64,
// so we emit just that one ComputeType. Covers both M=1 (decode) and M=128
// (prefill).
static void QNBit2BitRealisticShapesArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"BlkLen", "M", "N", "K", "Threads", "Symmetric", "HasBias", "ComputeType"});
  const int64_t BlkLen = 64;
  const int64_t Threads = 8;
  const int64_t Symmetric = 1;  // W2 native path is symmetric-only.
  const int64_t HasBias = 1;
  for (int64_t M : {int64_t{1}, int64_t{128}}) {
    for (auto kn : {std::pair<int64_t, int64_t>{384, 1024},
                    std::pair<int64_t, int64_t>{1024, 192},
                    std::pair<int64_t, int64_t>{1024, 384},
                    std::pair<int64_t, int64_t>{1024, 4096},
                    std::pair<int64_t, int64_t>{4096, 1024}}) {
      b->Args({BlkLen, M, kn.second, kn.first, Threads, Symmetric, HasBias, int64_t{SQNBIT_CompInt8}});
    }
  }
}

BENCHMARK(QNBITGEMM<float, 2>)->Apply(QNBit2BitRealisticShapesArgs)->UseRealTime();

// This test gets benchmark arguments from environment variables.
template <typename AType, size_t BlkBitWidth>
void QNBITGEMM_ENV(benchmark::State& state) {
  using onnxruntime::ParseEnvironmentVariableWithDefault;

  const auto BlkLen = ParseEnvironmentVariableWithDefault<size_t>("ORT_QNBITGEMM_BLKLEN", 32);
  const auto M = ParseEnvironmentVariableWithDefault<size_t>("ORT_QNBITGEMM_M", 1);
  const auto N = ParseEnvironmentVariableWithDefault<size_t>("ORT_QNBITGEMM_N", 4096);
  const auto K = ParseEnvironmentVariableWithDefault<size_t>("ORT_QNBITGEMM_K", 4096);
  const auto Threads = ParseEnvironmentVariableWithDefault<size_t>("ORT_QNBITGEMM_THREADS", 1);
  const auto Symmetric = ParseEnvironmentVariableWithDefault<bool>("ORT_QNBITGEMM_SYMMETRIC", true);
  const auto HasBias = ParseEnvironmentVariableWithDefault<bool>("ORT_QNBITGEMM_HAS_BIAS", false);
  const auto ComputeType = ParseEnvironmentVariableWithDefault<int32_t>("ORT_QNBITGEMM_COMPUTE_TYPE",
                                                                        static_cast<int32_t>(SQNBIT_CompFp32));

  RunQNBitGemmBenchmark<AType, BlkBitWidth>(BlkLen, M, N, K, Threads, Symmetric, HasBias,
                                            static_cast<MLAS_QNBIT_GEMM_COMPUTE_TYPE>(ComputeType),
                                            state);

  std::ostringstream s;
  s << "BlkBitWidth:" << BlkBitWidth << "/BlkLen:" << BlkLen
    << "/M:" << M << "/N:" << N << "/K:" << K
    << "/Threads:" << Threads << "/Symmetric:" << Symmetric << "/HasBias:" << HasBias
    << "/ComputeType:" << ComputeType;
  state.SetLabel(s.str());
}

BENCHMARK(QNBITGEMM_ENV<float, 4>)->UseRealTime();
