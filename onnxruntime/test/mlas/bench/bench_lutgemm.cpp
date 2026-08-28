// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "mlas_q4.h"
#include "mlas_qnbit.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <stdexcept>

static const std::vector<std::string> lutgemm_bench_arg_names = {"BlkLen", "N", "K", "Threads", "HasZP"};
static const std::vector<std::string> lutgemm_compute_arg_names = {"BlkLen", "M", "N", "K", "Threads", "HasZP", "HasBias"};

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
      false,  // IsFloatZeroPoint
      PackedBuf.data(),
      tp.get());

  for (auto _ : state) {
    MlasLutGemmPack(
        N, K, BlkBitWidth, BlkLen, HasZeroPoint,
        reinterpret_cast<std::byte*>(QuantBData.data()),
        QuantBScale.data(),
        HasZeroPoint ? QuantBZeroPoint.data() : nullptr,
        false,  // IsFloatZeroPoint
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
  const bool HasBias = static_cast<bool>(state.range(6));

  const bool lut_available = MlasIsLutGemmAvailable(N, K, BlkBitWidth, BlkLen);

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

  std::vector<float> Bias;
  const float* BiasPtr = nullptr;
  if (HasBias) {
    Bias = RandomVectorUniform(N, -1.0f, 1.0f);
    BiasPtr = Bias.data();
  }

  if (lut_available) {
    state.SetLabel("path=LUT");

    MlasClearLutGemmKernelConfig();
    MlasInitLutGemmKernelConfig(N, K, BlkBitWidth, BlkLen, HasZeroPoint);

    size_t PackedBufSize = MlasLutGemmPackedSize(N, K, BlkBitWidth, BlkLen, HasZeroPoint);
    std::vector<std::byte> PackedBuf(PackedBufSize);

    MlasLutGemmPack(
        N, K, BlkBitWidth, BlkLen, HasZeroPoint,
        reinterpret_cast<std::byte*>(QuantBData.data()),
        QuantBScale.data(),
        HasZeroPoint ? QuantBZeroPoint.data() : nullptr,
        false,  // IsFloatZeroPoint
        PackedBuf.data(),
        tp.get());

    // warm-up
    MlasLutGemm(A.data(), BlkLen, PackedBuf.data(), C.data(),
                static_cast<int>(K), static_cast<int>(M), static_cast<int>(N),
                HasZeroPoint, tp.get(), BiasPtr);

    for (auto _ : state) {
      MlasLutGemm(A.data(), BlkLen, PackedBuf.data(), C.data(),
                  static_cast<int>(K), static_cast<int>(M), static_cast<int>(N),
                  HasZeroPoint, tp.get(), BiasPtr);
    }
  } else {
    // Fall back to ComputeBUnpacked-equivalent: dequant B to fp32 then SGEMM.
    // This matches what the runtime does (matmul_nbits.cc ComputeBUnpacked)
    // when LUT is gated out for the given shape (e.g. N % n_div != 0).
    state.SetLabel("path=Dequant+SGEMM");

    // MlasDequantizeBlockwise(columnwise=true, K, N) emits B in [N, K] row-major
    // layout (equivalently [K, N] column-major) -- this is "B^T" from the
    // MatMul perspective and matches what matmul_nbits.cc::ComputeBUnpacked
    // feeds to MlasGemmBatch.
    std::vector<float> DequantB(K * N);

    // Time dequant+SGEMM as a unit (this is what the runtime pays per call,
    // since the dequantized buffer is not cached across MatMulNBits calls).
    auto dequant_and_gemm = [&]() {
      MlasDequantizeBlockwise<float, BlkBitWidth>(
          DequantB.data(), QuantBData.data(), QuantBScale.data(),
          HasZeroPoint ? QuantBZeroPoint.data() : nullptr,
          static_cast<int>(BlkLen), /*columnwise*/ true,
          static_cast<int>(K), static_cast<int>(N), tp.get());

      // DequantB is [N, K] row-major; use CblasTrans + ldb=K to match
      // matmul_nbits.cc::ComputeBUnpacked (which calls
      // MlasGemmBatch(CblasNoTrans, CblasTrans, ..., ldb=K, ...) on the
      // same dequantized buffer).
      MlasGemm(CblasNoTrans, CblasTrans,
               M, N, K,
               1.0f,
               A.data(), K,
               DequantB.data(), K,
               0.0f,
               C.data(), N,
               tp.get(), nullptr);

      if (BiasPtr != nullptr) {
        // Broadcast-add bias [N] over rows of C [M, N].
        for (size_t m = 0; m < M; ++m) {
          float* row = C.data() + m * N;
          for (size_t n = 0; n < N; ++n) {
            row[n] += BiasPtr[n];
          }
        }
      }
    };

    // warm-up
    dequant_and_gemm();

    for (auto _ : state) {
      dequant_and_gemm();
    }
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
      {128},                            // BlkLen
      {1, 32},                          // M
      {4096},                           // N
      {4096},                           // K
      {8},                              // Threads
      {int64_t{false}},                 // HasZeroPoint
      {int64_t{false}, int64_t{true}},  // HasBias
  });
}

// Representative 2-bit MatMulNBits shapes (BlkLen=64) drawn from real W2
// production models. Five distinct (K, N) pairs:
//   (K=384,  N=1024): 20 nodes
//   (K=1024, N=192):  40 nodes
//   (K=1024, N=384):  20 nodes
//   (K=1024, N=4096): 20 nodes
//   (K=4096, N=1024): 20 nodes
// Covers both M=1 (decode) and M=128 (prefill) so the LUT path can be
// compared apples-to-apples against the W4 CompInt8 and W2 kernels
// (QNBITGEMM<float, 4>/QNBitGemmRealisticShapesArgs and
// QNBITGEMM<float, 2>/QNBit2BitRealisticShapesArgs).
static void LutGemmRealisticShapesArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames(lutgemm_compute_arg_names);
  // Separate Args() entries so we only run the exact (M, K, N) tuples that
  // appear in the representative production model.
  const int64_t BlkLen = 64;
  const int64_t Threads = 8;
  const int64_t HasZP = 0;
  const int64_t HasBias = 1;
  for (int64_t M : {int64_t{1}, int64_t{128}}) {
    for (auto kn : {std::pair<int64_t, int64_t>{384, 1024},
                    std::pair<int64_t, int64_t>{1024, 192},
                    std::pair<int64_t, int64_t>{1024, 384},
                    std::pair<int64_t, int64_t>{1024, 4096},
                    std::pair<int64_t, int64_t>{4096, 1024}}) {
      b->Args({BlkLen, M, kn.second, kn.first, Threads, HasZP, HasBias});
    }
  }
}

[[maybe_unused]] static const bool benchmarks_registered = []() {
  const bool is_lutgemm_supported = MlasIsLutGemmAvailable(4096, 4096, 2, 128);
  if (is_lutgemm_supported) {
    BENCHMARK(LUTGEMM_PACK<2>)->Apply(LutGemmPackArgs)->UseRealTime();
    BENCHMARK(LUTGEMM_COMPUTE<2>)->Apply(LutGemmComputeArgs)->UseRealTime();
    BENCHMARK(LUTGEMM_COMPUTE<2>)->Apply(LutGemmRealisticShapesArgs)->UseRealTime();
    return true;
  }
  return false;
}();
