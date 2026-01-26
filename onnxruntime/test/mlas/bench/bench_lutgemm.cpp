// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas_q4.h"
#include "mlas_qnbit.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <stdexcept>

static const std::vector<std::string> lutgemm_bench_arg_names = {"BlkLen", "N", "K", "Threads", "HasZP"};
static const std::vector<std::string> lutgemm_compute_arg_names = {"BlkLen", "M", "N", "K", "Threads", "HasZP"};

template <size_t BlkBitWidth>
void LUTGEMM_PACK(benchmark::State& state) {
  if (state.range(0) <= 0) throw std::invalid_argument("BlkLen must be greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must be greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must be greater than 0!");
  if (state.range(3) <= 0) throw std::invalid_argument("Threads must be greater than 0!");

  const size_t BlkLen = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t Threads = static_cast<size_t>(state.range(3));
  const bool HasZeroPoint = static_cast<bool>(state.range(4));

  if (!MlasIsLutGemmAvailable(N, K, BlkBitWidth, BlkLen)) {
    state.SkipWithMessage("LUT GEMM is not available with the given configuration.");
    return;
  }

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(Threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
  MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(
      static_cast<int>(BlkLen), true,
      static_cast<int>(K), static_cast<int>(N),
      QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

  auto B = RandomVectorUniform(K * N, -1.0f, 1.0f);
  std::vector<uint8_t> QuantBData(QuantBDataSizeInBytes);
  std::vector<float> QuantBScale(QuantBScaleSize);
  std::vector<uint8_t> QuantBZeroPoint(HasZeroPoint ? QuantBZeroPointSizeInBytes : 0);

  MlasQuantizeBlockwise<float, BlkBitWidth>(
      QuantBData.data(), QuantBScale.data(),
      HasZeroPoint ? QuantBZeroPoint.data() : nullptr,
      B.data(), static_cast<int>(BlkLen), true,
      static_cast<int>(K), static_cast<int>(N), static_cast<int>(N),
      tp.get());

  MlasClearLutGemmKernelConfig();
  MlasInitLutGemmKernelConfig(N, K, BlkBitWidth, BlkLen, HasZeroPoint);

  size_t PackedBufSize = MlasLutGemmPackedSize(N, K, BlkBitWidth, BlkLen, HasZeroPoint);
  std::vector<std::byte> PackedBuf(PackedBufSize);

  MlasLutGemmPack(
      N, K, BlkBitWidth, BlkLen, HasZeroPoint,
      reinterpret_cast<std::byte*>(QuantBData.data()),
      QuantBScale.data(),
      HasZeroPoint ? QuantBZeroPoint.data() : nullptr,
      PackedBuf.data(),
      tp.get());

  for (auto _ : state) {
    MlasLutGemmPack(
        N, K, BlkBitWidth, BlkLen, HasZeroPoint,
        reinterpret_cast<std::byte*>(QuantBData.data()),
        QuantBScale.data(),
        HasZeroPoint ? QuantBZeroPoint.data() : nullptr,
        PackedBuf.data(),
        tp.get());
  }
}

template <size_t BlkBitWidth>
void LUTGEMM_COMPUTE(benchmark::State& state) {
  if (state.range(0) <= 0) throw std::invalid_argument("BlkLen must be greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("M must be greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("N must be greater than 0!");
  if (state.range(3) <= 0) throw std::invalid_argument("K must be greater than 0!");
  if (state.range(4) <= 0) throw std::invalid_argument("Threads must be greater than 0!");

  const size_t BlkLen = static_cast<size_t>(state.range(0));
  const size_t M = static_cast<size_t>(state.range(1));
  const size_t N = static_cast<size_t>(state.range(2));
  const size_t K = static_cast<size_t>(state.range(3));
  const size_t Threads = static_cast<size_t>(state.range(4));
  const bool HasZeroPoint = static_cast<bool>(state.range(5));

  if (!MlasIsLutGemmAvailable(N, K, BlkBitWidth, BlkLen)) {
    state.SkipWithMessage("LUT GEMM is not available with the given configuration.");
    return;
  }

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(Threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto A = RandomVectorUniform(M * K, -1.0f, 1.0f);
  std::vector<float> C(M * N);

  size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
  MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(
      static_cast<int>(BlkLen), true,
      static_cast<int>(K), static_cast<int>(N),
      QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

  auto B = RandomVectorUniform(K * N, -1.0f, 1.0f);
  std::vector<uint8_t> QuantBData(QuantBDataSizeInBytes);
  std::vector<float> QuantBScale(QuantBScaleSize);
  std::vector<uint8_t> QuantBZeroPoint(HasZeroPoint ? QuantBZeroPointSizeInBytes : 0);

  MlasQuantizeBlockwise<float, BlkBitWidth>(
      QuantBData.data(), QuantBScale.data(),
      HasZeroPoint ? QuantBZeroPoint.data() : nullptr,
      B.data(), static_cast<int>(BlkLen), true,
      static_cast<int>(K), static_cast<int>(N), static_cast<int>(N),
      tp.get());

  MlasClearLutGemmKernelConfig();
  MlasInitLutGemmKernelConfig(N, K, BlkBitWidth, BlkLen, HasZeroPoint);

  size_t PackedBufSize = MlasLutGemmPackedSize(N, K, BlkBitWidth, BlkLen, HasZeroPoint);
  std::vector<std::byte> PackedBuf(PackedBufSize);

  MlasLutGemmPack(
      N, K, BlkBitWidth, BlkLen, HasZeroPoint,
      reinterpret_cast<std::byte*>(QuantBData.data()),
      QuantBScale.data(),
      HasZeroPoint ? QuantBZeroPoint.data() : nullptr,
      PackedBuf.data(),
      tp.get());

  MlasLutGemm(A.data(), BlkLen, PackedBuf.data(), C.data(),
              static_cast<int>(K), static_cast<int>(M), static_cast<int>(N),
              HasZeroPoint, tp.get());

  for (auto _ : state) {
    MlasLutGemm(A.data(), BlkLen, PackedBuf.data(), C.data(),
                static_cast<int>(K), static_cast<int>(M), static_cast<int>(N),
                HasZeroPoint, tp.get());
  }
}

static void LutGemmPackArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames(lutgemm_bench_arg_names);
  b->ArgsProduct({
      {128},             // BlkLen
      {4096},            // N
      {4096},            // K
      {8},               // Threads
      {int64_t{false}},  // HasZeroPoint
  });
}

static void LutGemmComputeArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames(lutgemm_compute_arg_names);
  b->ArgsProduct({
      {128},             // BlkLen
      {1, 32},           // M
      {4096},            // N
      {4096},            // K
      {8},               // Threads
      {int64_t{false}},  // HasZeroPoint
  });
}

[[maybe_unused]] static const bool benchmarks_registered = []() {
  const bool is_lutgemm_supported = MlasIsLutGemmAvailable(4096, 4096, 2, 128);
  if (is_lutgemm_supported) {
    BENCHMARK(LUTGEMM_PACK<2>)->Apply(LutGemmPackArgs)->UseRealTime();
    BENCHMARK(LUTGEMM_COMPUTE<2>)->Apply(LutGemmComputeArgs)->UseRealTime();
    return true;
  }
  return false;
}();
