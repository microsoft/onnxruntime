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
  ArgsProduct(b, {{1, 1024, 2048}, {4096, 11008}, {4096, 11008}, {8}});
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

#ifdef MLAS_JBLAS
void Q4GEMM_Jblas(benchmark::State& state, int block_size, bool is_asym, MLAS_COMPUTE_TYPE cmp_type) {
  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  if (state.range(3) <= 0) throw std::invalid_argument("Threads must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t threads = static_cast<size_t>(state.range(3));
  block_size = block_size == -1 ? int(K) : block_size;
  const size_t pack_b_size = MlasNBitsGemmPackBSize(N, K, block_size, 4, is_asym, cmp_type);

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = int(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto A1 = RandomVectorUniform(static_cast<size_t>(M * K), -1.0f, 1.0f);
  auto B1 = RandomVectorUniform<uint8_t>(static_cast<size_t>(N * K / 2), 0, 255);
  auto blk_num = (K + block_size - 1) / block_size;
  auto B_scale = RandomVectorUniform(static_cast<size_t>(N * blk_num), 0.003f, 0.005f);
  std::vector<float> C1(static_cast<size_t>(M * N));
  auto B_zp = RandomVectorUniform<uint8_t>(static_cast<size_t>(N * blk_num / 2), 0, 255);

  std::vector<int8_t> B1_packed(pack_b_size);
  MlasNBitsGemmPackB(B1_packed.data(), B1.data(), B_scale.data(), is_asym ? B_zp.data() : nullptr, N, K, N, block_size, 4, is_asym, true, cmp_type, tp.get());

  MLAS_NBITS_GEMM_DATA_SIMPLE_PARAMS params1;
  params1.A = A1.data();
  params1.lda = K;
  params1.C = C1.data();
  params1.ldc = N;
  params1.B = B1_packed.data();
  std::vector<int8_t> workspace(size_t(M <= 32 ? 32 : M) * K * 4);
  MlasNBitsGemmBatchPackedB(M, N, K, 1, &params1, workspace.data(), tp.get());

  for (auto _ : state) {
    MlasNBitsGemmBatchPackedB(M, N, K, 1, &params1, workspace.data(), tp.get());
  }
}

BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4G32SymInt8, 32, false, CompInt8)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4G128SymInt8, 128, false, CompInt8)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4GPerNSymInt8, -1, false, CompInt8)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4G32SymFp32, 32, false, CompFp32)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4G128SymFp32, 128, false, CompFp32)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4GPerNSymFp32, -1, false, CompFp32)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(Q4GEMM_Jblas, Q4G32AsymFp32, 32, true, CompFp32)->Apply(GemmSizeProducts)->UseRealTime();
#endif
