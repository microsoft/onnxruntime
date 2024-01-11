// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas_q4.h"
#include "mlas_qnbit.h"

#include <memory>
#include <stdexcept>
#include <vector>

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
  const auto ComputeType = static_cast<MLAS_SQNBIT_GEMM_COMPUTE_TYPE>(state.range(6));

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
  if (const auto PackedQuantBDataSize = MlasSQNBitGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen);
      PackedQuantBDataSize > 0) {
    PackedQuantBData = std::make_unique<std::byte[]>(PackedQuantBDataSize);
    MlasSQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, QuantBData.data(), PackedQuantBData.get(), tp.get());
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
                              static_cast<MLAS_SQNBIT_GEMM_COMPUTE_TYPE>(args[6]));
                        });
}

BENCHMARK(SQNBITGEMM<4>)->Apply(SQNBitGemmArgs)->UseRealTime();

#if defined(MLAS_JBLAS)

void Q4GEMM_Jblas(benchmark::State& state, int block_size, bool is_asym, MLAS_SQNBIT_COMPUTE_TYPE cmp_type) {
  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  if (state.range(3) <= 0) throw std::invalid_argument("Threads must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t threads = static_cast<size_t>(state.range(3));
  block_size = block_size == -1 ? static_cast<int>(K) : block_size;
  const size_t pack_b_size = MlasNBitsGemmPackBSize(N, K, block_size, 4, is_asym, cmp_type);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = static_cast<int>(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(onnxruntime::concurrency::CreateThreadPool(
      &onnxruntime::Env::Default(), tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto A1 = RandomVectorUniform(static_cast<size_t>(M * K), -1.0f, 1.0f);
  auto B1 = RandomVectorUniform<uint8_t>(static_cast<size_t>(N * K / 2), 0, 255);
  auto blk_num = static_cast<size_t>((K + block_size - 1) / block_size);
  auto B_scale = RandomVectorUniform(static_cast<size_t>(N * blk_num), 0.003f, 0.005f);
  std::vector<float> C1(static_cast<size_t>(M * N));
  auto B_zp = RandomVectorUniform<uint8_t>(static_cast<size_t>(N * blk_num / 2), 0, 255);

  std::vector<int8_t> B1_packed(pack_b_size);
  MlasNBitsGemmPackB(B1_packed.data(), B1.data(), B_scale.data(), is_asym ? B_zp.data() : nullptr, N, K, K, block_size,
                     4, is_asym, true, cmp_type, tp.get());

  MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS params1;
  params1.A = A1.data();
  params1.lda = K;
  params1.C = C1.data();
  params1.ldc = N;
  params1.B = B1_packed.data();
  std::vector<int8_t> workspace(static_cast<size_t>(M <= 32 ? 32 : M) * K * 4);
  MlasSQNBitsGemmBatchPackedB(M, N, K, 1, &params1, workspace.data(), tp.get());

  for (auto _ : state) {
    MlasSQNBitsGemmBatchPackedB(M, N, K, 1, &params1, workspace.data(), tp.get());
  }
}

static void GemmSizeProducts(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "Threads"});
  b->ArgsProduct({{1, 1024, 2048}, {4096, 11008}, {4096, 11008}, {8}});
}

BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4G32SymInt8, 32, false, CompInt8)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4G128SymInt8, 128, false, CompInt8)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4GPerNSymInt8, -1, false, CompInt8)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4G32SymFp32, 32, false, CompFp32)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4G128SymFp32, 128, false, CompFp32)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4GPerNSymFp32, -1, false, CompFp32)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4G32AsymFp32, 32, true, CompFp32)->Apply(GemmSizeProducts)->UseRealTime();

#endif  // defined(MLAS_JBLAS)
