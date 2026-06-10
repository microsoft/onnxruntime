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

// Prototype W2 super-block kernel + scalar pack helper (Phase 3 of the
// W2-vs-W4 parity work). Not yet wired into the platform dispatch, so the
// bench drives it directly via the test-entry forwarder.
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/qnbitgemm.h"
#include "core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.h"
#include "core/mlas/lib/sqnbitgemm_kernel_avx512_2bit_superblock.h"

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

BENCHMARK(QNBITGEMM<float, 4>)->Apply(QNBitGemmArgs<float>)->UseRealTime();
BENCHMARK(QNBITGEMM<float, 8>)->Apply(QNBitGemmArgs<float>)->UseRealTime();
BENCHMARK(QNBITGEMM<MLAS_FP16, 4>)->Apply(QNBitGemmArgs<MLAS_FP16>)->UseRealTime();

// Customer MatMulNBits shapes mirrored at 4-bit for a head-to-head comparison
// vs the W2 LUT path (LUTGEMM_COMPUTE/CUSTOMER). Customer model uses BlkLen=64.
// Five distinct (K, N) pairs:
//   (K=384,  N=1024): 20 nodes
//   (K=1024, N=192):  40 nodes
//   (K=1024, N=384):  20 nodes
//   (K=1024, N=4096): 20 nodes
//   (K=4096, N=1024): 20 nodes
// Both M=1 (decode) and M=128 (prefill) are exercised — paired with the W2
// rows below so we get a 3-way (W2 / W4 / W8) comparison at each M.
static void QNBitGemmCustomerArgs(benchmark::internal::Benchmark* b) {
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

BENCHMARK(QNBITGEMM<float, 4>)->Apply(QNBitGemmCustomerArgs)->UseRealTime();

// 2-bit weight rows for the customer shapes. Exercises the AVX-512 W2 native
// path (VNNI variant on AVX-512-VNNI hosts; non-VNNI variant on AVX-512BW
// hosts). W2 is registered only for SQNBIT_CompInt8 and BlkLen=64, so we
// emit just that one ComputeType. Covers both M=1 (decode) and M=128 (prefill).
static void QNBit2BitCustomerArgs(benchmark::internal::Benchmark* b) {
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

BENCHMARK(QNBITGEMM<float, 2>)->Apply(QNBit2BitCustomerArgs)->UseRealTime();

// ---------------------------------------------------------------------------
// W2 SUPER-BLOCK PROTOTYPE BENCHMARK
// ---------------------------------------------------------------------------
// Drives the Phase-3 super-block W2 kernel directly via its test-entry
// forwarder. Mirrors what MlasQNBitGemmBatch would do for our path: pre-pack
// B via the super-block helpers, quantize each A row using the dispatch's
// AVX-512 A-quantizer (the same one the production W2 path uses), and call
// the SIMD kernel per-thread on the N-tile chunks the dispatcher splits over.
//
// Constraints (must be satisfied for the kernel to handle the rows; bench
// skips otherwise):
//   * BlkLen == 64
//   * K a multiple of (BlkLen * kSuperBlockBlks) = 256
//   * M a multiple of kNRows2 (=2)
//   * N a multiple of kNCols4 (=4)
//
namespace bench_super {

namespace sq2sb = onnxruntime::mlas::sq2bit_avx512_super;
namespace sq2 = onnxruntime::mlas::sq2bit_avx512;

void RunQ2SuperBlockBenchmark(size_t M, size_t N, size_t K, size_t Threads,
                              bool HasBias, benchmark::State& state) {
  using onnxruntime::narrow;
  constexpr size_t BlkBitWidth = 2;
  constexpr size_t BlkLen = sq2::kBlkLen;
  constexpr size_t kSuperBlockBlks = sq2::kSuperBlockBlks;
  // R2xC4 SIMD tile (kNRows2 lives in the SIMD-only header which the bench
  // TU is not built against -- mirror its value here, used only for the
  // N alignment gate below). The kernel handles any M >= 1 and any K >= 1
  // (K-tail handler covers K not a multiple of 256).
  constexpr size_t kNCols4 = sq2::kNCols4;
  (void)kSuperBlockBlks;  // retained for documentation reference only

  // Gate on the kernel's hard constraints.
  if (K % BlkLen != 0 || K == 0) {
    state.SkipWithMessage("Super-block requires K > 0 and K % BlkLen == 0.");
    return;
  }
  if (M == 0 || (N % kNCols4) != 0) {
    state.SkipWithMessage("Super-block requires M>=1 and N%4==0.");
    return;
  }

  // Gate on host having AVX-512(-VNNI) -- the kernel is AVX-512 BW + (optional) VNNI.
  // QuantizeARowComputeBlkSum_CompInt8 is AVX-512 and required for the A-quant step.
  const auto& platform = GetMlasPlatform();
  if (platform.QNBitGemmDispatch == nullptr ||
      platform.QNBitGemmDispatch->QuantizeARowComputeBlkSum_CompInt8 == nullptr) {
    state.SkipWithMessage("AVX-512 dispatch table not available on this host.");
    return;
  }

  const size_t BlockCountK = K / BlkLen;

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(Threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo,
                                                 onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  // ----- Source data -----
  const auto A = RandomVectorUniform(M * K, float{-1.0f}, float{1.0f});
  const auto B = RandomVectorUniform(K * N, float{-1.0f}, float{1.0f});
  const auto Bias = HasBias ? RandomVectorUniform(N, float{-1.0f}, float{1.0f}) : std::vector<float>();

  size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
  MlasBlockwiseQuantizedBufferSizes<BlkBitWidth>(
      static_cast<int>(BlkLen), /*columnwise=*/true,
      static_cast<int>(K), static_cast<int>(N),
      QuantBDataSizeInBytes, QuantBScaleSize, &QuantBZeroPointSizeInBytes);

  std::vector<uint8_t> QuantBDataSrc(QuantBDataSizeInBytes);
  std::vector<float> QuantBScale(QuantBScaleSize);
  MlasQuantizeBlockwise<float, BlkBitWidth>(
      QuantBDataSrc.data(), QuantBScale.data(), /*zp=*/nullptr,
      B.data(), static_cast<int>(BlkLen), /*columnwise=*/true,
      static_cast<int>(K), static_cast<int>(N), static_cast<int>(N), tp.get());

  // ----- Pack into the super-block layout -----
  const size_t PackedSize = sq2sb::Q2BitGemmPackQuantBDataSize_SuperBlock(
      N, K, BlkLen, /*HasZeroPoint=*/false, SQNBIT_CompInt8, nullptr);
  if (PackedSize == 0) {
    state.SkipWithMessage("Super-block pack size returned 0 for this shape.");
    return;
  }
  std::vector<std::byte> PackedBuf(PackedSize, std::byte{0});
  PackedQuantBDataStruct<float, BlkBitWidth> packed_b(
      PackedBuf.data(), N, BlockCountK, BlkLen, /*QuantAUnsigned=*/false);

  // Same 3-call prepack pattern matmul_nbits.cc uses.
  sq2sb::SQ2BitGemmPackQuantBDataAndBlkSum_SuperBlockScalar(
      N, K, BlkLen, SQNBIT_CompInt8,
      reinterpret_cast<const std::byte*>(QuantBDataSrc.data()),
      /*scales=*/nullptr,
      /*has_zp=*/false, /*zp=*/nullptr,
      packed_b, tp.get(), nullptr);
  sq2sb::SQ2BitGemmPackQuantBDataAndBlkSum_SuperBlockScalar(
      N, K, BlkLen, SQNBIT_CompInt8,
      /*B=*/nullptr, QuantBScale.data(),
      /*has_zp=*/false, /*zp=*/nullptr,
      packed_b, tp.get(), nullptr);

  // ----- Quantize A once via the dispatch's AVX-512 A-quantizer -----
  std::vector<std::byte> QuantAData(M * BlockCountK * BlkLen, std::byte{0});
  std::vector<float> QuantAScale(M * BlockCountK, 0.0f);
  std::vector<float> ABlockSum(M * BlockCountK, 0.0f);
  auto QuantizeARow = platform.QNBitGemmDispatch->QuantizeARowComputeBlkSum_CompInt8;
  for (size_t m = 0; m < M; ++m) {
    QuantizeARow(BlkLen, A.data() + m * K, K,
                 QuantAData.data() + m * BlockCountK * BlkLen,
                 QuantAScale.data() + m * BlockCountK,
                 ABlockSum.data() + m * BlockCountK);
  }

  std::vector<float> C(M * N, 0.0f);

  // Pick the best SIMD variant for the host. Prefer the VNNI variant when
  // the platform dispatch table is the VNNI one (same logic the production
  // dispatch wiring would use).
  const bool use_vnni = (platform.QNBitGemmDispatch == &MlasSQNBitGemmDispatchAvx512vnni);
  auto kernel = use_vnni
                    ? sq2sb::SQ2BitGemmKernel_BlkSum_CompInt8_Super_Avx512Vnni_TestEntry
                    : sq2sb::SQ2BitGemmKernel_BlkSum_CompInt8_Super_Avx512_TestEntry;

  // Mirror the production dispatcher's N-tile parallel split. SQ2BitGemm_CompInt8
  // tiles N in chunks of 128 and parallelizes across them; we do the same.
  constexpr size_t kNTile = 128;
  const size_t num_n_tiles = (N + kNTile - 1) / kNTile;

  auto run_one = [&]() {
    onnxruntime::concurrency::ThreadPool::TryBatchParallelFor(
        tp.get(), static_cast<ptrdiff_t>(num_n_tiles),
        [&](ptrdiff_t t) {
          const size_t n_start = static_cast<size_t>(t) * kNTile;
          const size_t n_count = std::min(kNTile, N - n_start);
          if (n_count == 0) return;

          // Pointer arithmetic mirroring SQ2BitGemm_CompInt8.
          const size_t ldb_bytes = BlockCountK * sq2::kBlkBytes;
          const size_t ldb_scale = BlockCountK;
          const std::byte* b_tile = packed_b.PackedQuantBData + n_start * ldb_bytes;
          const float* bscale_tile = packed_b.PackedQuantBScale + n_start * ldb_scale;
          const float* bblksum_tile = packed_b.QuantBBlkSum + n_start * ldb_scale;
          float* c_tile = C.data() + n_start;
          const float* bias_tile = HasBias ? (Bias.data() + n_start) : nullptr;

          kernel(BlkLen,
                 QuantAData.data(), QuantAScale.data(),
                 b_tile, bscale_tile,
                 /*QuantBZeroPoint=*/nullptr,
                 c_tile,
                 M, n_count, K, BlockCountK,
                 bias_tile,
                 /*ldc=*/N,
                 ABlockSum.data(), bblksum_tile);
        },
        /*cost=*/0);
  };

  run_one();  // warm up
  for (auto _ : state) {
    run_one();
  }
}

}  // namespace bench_super

void QNBITGEMM_SUPER(benchmark::State& state) {
  using onnxruntime::narrow;
  const auto BlkLen = narrow<size_t>(state.range(0));
  (void)BlkLen;  // The super-block kernel only supports BlkLen=64; gated inside.
  const auto M = narrow<size_t>(state.range(1));
  const auto N = narrow<size_t>(state.range(2));
  const auto K = narrow<size_t>(state.range(3));
  const auto Threads = narrow<size_t>(state.range(4));
  // state.range(5) (Symmetric) and (7) (ComputeType) are unused; the
  // super-block kernel is symmetric CompInt8 only.
  const bool HasBias = narrow<bool>(state.range(6));
  bench_super::RunQ2SuperBlockBenchmark(M, N, K, Threads, HasBias, state);
}

// Customer-shape rows for the super-block prototype. Uses the same argument
// schema as QNBITGEMM<float, 2> so the bench rows line up one-to-one in the
// output (modulo skipped shapes that violate the super-block K constraint).
BENCHMARK(QNBITGEMM_SUPER)->Apply(QNBit2BitCustomerArgs)->UseRealTime();

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
