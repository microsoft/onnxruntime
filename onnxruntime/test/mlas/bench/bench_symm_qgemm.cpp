// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <stdexcept>
#include <memory>
#include <numeric>
#include <algorithm>

static const std::vector<std::string> qgemm_arg_names = {"M", "N", "K", "Batch", "Threads"};

void SYMMQGEMM(benchmark::State& state, bool a_signed) {
  const int8_t a_zero_point = 29;

  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  if (state.range(3) <= 0) throw std::invalid_argument("Batch must greater than 0!");
  if (state.range(4) <= 0) throw std::invalid_argument("Threads must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  const size_t batch = static_cast<size_t>(state.range(3));
  const size_t threads = static_cast<size_t>(state.range(4));

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = int(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));
  void* a_data = nullptr;
  std::vector<int8_t> signed_a;
  std::vector<uint8_t> unsigned_a;
  if (a_signed) {
    signed_a = RandomVectorUniform<int8_t>(static_cast<size_t>(M * K * batch) + 512, int8_t(-120), int8_t(120));
    a_data = signed_a.data();
  } else {
    unsigned_a = RandomVectorUniform<uint8_t>(static_cast<size_t>(M * K * batch) + 512, uint8_t(7), uint8_t(248));
    a_data = unsigned_a.data();
  }

  auto B_holder = RandomVectorUniform<int8_t>(static_cast<size_t>(N * K * batch), int8_t(-122), int8_t(122));
  std::vector<int32_t> C_holder(static_cast<size_t>(M * N * batch));
  std::vector<uint8_t> pack_b_holder;

  size_t packed_b_size = MlasSymmQgemmPackBSize(N, K, a_signed);
  if (packed_b_size == 0) return;

  pack_b_holder.resize(packed_b_size * batch);

  MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;

  gemm_shape.M = static_cast<size_t>(M);
  gemm_shape.N = static_cast<size_t>(N);
  gemm_shape.K = static_cast<size_t>(K);
  gemm_shape.AIsSigned = a_signed;
  gemm_shape.BIsSigned = true;

  std::vector<MLAS_SYMM_QGEMM_DATA_PARAMS> gemm_data_vec(batch);
  for (size_t i = 0; i < batch; i++) {
    auto& gemm_params = gemm_data_vec[i];
    gemm_params.lda = gemm_shape.K;
    gemm_params.ldc = gemm_shape.N;
    gemm_params.A = (uint8_t*)a_data + M * K * i;
    gemm_params.C = C_holder.data() + M * N * i;
    gemm_params.B = (void*)(pack_b_holder.data() + packed_b_size * i);

    MlasSymmQgemmPackB(N, K, (const int8_t*)gemm_params.B, N, a_signed, a_zero_point, (void*)(pack_b_holder.data() + packed_b_size * i));
  }
  for (auto _ : state) {
    MlasSymmQgemmBatch(gemm_shape, gemm_data_vec.data(), batch, tp.get());
  }
}

static void SymmQGemmSize(benchmark::internal::Benchmark* b) {
  b->ArgNames(qgemm_arg_names);
  // Args for  "M", "N", "K", "Batch",

  b->Args({1, 4096, 4096, 1, 8});
  b->Args({1, 12288, 4096, 1, 8});
  b->Args({1024, 4096, 4096, 1, 8});
  b->Args({1024, 12288, 4096, 1, 8});
}

BENCHMARK_CAPTURE(SYMMQGEMM, U8S8, false)->Apply(SymmQGemmSize)->UseRealTime();

