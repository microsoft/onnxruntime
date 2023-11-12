// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas_q4.h"
#include "bench_util.h"
#include "contrib_ops/cpu/quantization/dequantize_blockwise.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/thread_utils.h"

#include <stdexcept>
#include <numeric>

extern const float* G_scales_data;
extern const uint8_t* G_zero_points_data;
extern int64_t G_block_size_;
extern int64_t G_nbits_;
extern size_t G_ldfb;
extern size_t G_K;

/**
 * @brief Define types of block quantization
 */
typedef enum {
    BeforePacking = 0,    /*!< int4 Symmetric Block Quantization, zero_point = 0 */
    InPacking = 1,    /*!< int4 Block Quantization, zero_point is int8 type */
} MLAS_BLK_DEQUANT_TYPE;

static const std::vector<std::string> f32q4gemm_bench_arg_names = {"M", "N", "K", "block_size", "Threads"};

static inline bool IsPowerOfTwo(unsigned x) {
  return (x != 0) && ((x & (x - 1)) == 0);
}

namespace onnxruntime {
namespace test {

void QuantizeDequantize(std::vector<float>& raw_vals,
                        std::vector<uint8_t>& quant_vals,
                        std::vector<float>& scales,
                        std::vector<uint8_t>* zp,
                        int32_t N,
                        int32_t K,
                        int32_t block_size) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);
  contrib::QuantizeBlockwise<float>(
      quant_vals.data(),
      raw_vals.data(),
      scales.data(),
      zp != nullptr ? zp->data() : nullptr,
      block_size,
      4,
      N,
      K,
      tp.get());

  // Note that input1_f_vals is NxK after dequant
  contrib::DequantizeBlockwise<float>(
      raw_vals.data(),
      quant_vals.data(),
      scales.data(),
      zp != nullptr ? zp->data() : nullptr,
      block_size,
      4,
      N,
      K,
      tp.get());
}

}  // namespace test
}  // namespace onnxruntime

void F32Q4GEMM(benchmark::State& state, MLAS_BLK_DEQUANT_TYPE dqtype) {
  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  const size_t block_size = static_cast<size_t>(state.range(3));
  if (!(block_size >= 16 && block_size <= 256 && IsPowerOfTwo(block_size))) throw std::invalid_argument("block_size must be 16, 32, 64, 128, 256!");
  if (state.range(4) <= 0) throw std::invalid_argument("Threads must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t threads = static_cast<size_t>(state.range(4));

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = int(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

#if 1
  auto input0_vals = RandomVectorUniform(static_cast<size_t>(M * K), -1.0f, 1.0f);
  auto input1_f_vals = RandomVectorUniform(static_cast<size_t>(N * K), -1.0f, 1.0f);
#else
  std::vector<float> input0_vals(M * K);
  for (int64_t i = 0; i < M * K; i++) {
    input0_vals[i] = static_cast<float>(i)/(M * K);
  }
  std::vector<float> input1_f_vals(K * N);
  for (int64_t i = 0; i < K * N; i++) {
    input1_f_vals[i] = static_cast<float>(i)/(K * N);
  }
#endif
  std::vector<float> y(static_cast<size_t>(M * N));

#if 0  // for Debugging
  std::vector<float> input1_f_vals_trans(N * K);
  MlasTranspose(input1_f_vals.data(), input1_f_vals_trans.data(), K, N);
#endif

  int64_t block_per_k = (K + block_size - 1) / block_size;
  int64_t number_of_block = block_per_k * N;
  int64_t block_blob_size = block_size * 4 / 8;
  int64_t buf_size = number_of_block * (block_size * 4 / 8);
  std::vector<uint8_t> input1_vals(buf_size);
  std::vector<float> scales(number_of_block);
  std::vector<uint8_t> zp((N * block_per_k + 1) / 2);

  onnxruntime::test::QuantizeDequantize(input1_f_vals,
                     input1_vals,
                     scales,
                     /*has_zeropoint ? &zp :*/ nullptr,
                     static_cast<int32_t>(N),
                     static_cast<int32_t>(K),
                     static_cast<int32_t>(block_size));

  G_scales_data = scales.data();

  G_block_size_ = block_size;
  G_nbits_ = 4;

  //TensorShape b_shape({N, K});

  //MatMulComputeHelper helper;
  //ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  const size_t lda = K; //helper.Lda(false);
  //const size_t ldb = helper.Ldb(true);

  G_K = K;

  MLAS_SGEMM_DATA_PARAMS data;
  data.BIsPacked = false;
  data.A = input0_vals.data();// + helper.LeftOffsets()[i];
  data.lda = lda;
  data.B = input1_vals.data();// + helper.RightOffsets()[i];
  data.ldb = -1;
  data.C = y.data();// + helper.OutputOffsets()[i];
  data.ldc = N;
  data.alpha = 1.f;
  data.beta = 0.0f;

  MlasGemmBatch(CblasNoTrans, CblasTrans,
                M, N, K, &data, 1, tp.get());

  for (auto _ : state) {
    MlasGemmBatch(CblasNoTrans, CblasTrans,
                  M, N, K, &data, 1, tp.get());
  }
}

static void GemmSizeProducts(benchmark::internal::Benchmark* b) {
  b->ArgNames(f32q4gemm_bench_arg_names);
  for (auto M : {/*1,*/ /*2, 4, 16, 64, 100, 256, 1024,*/ 2048}) {
    for (auto N : {/*128, 256,*/ 4096, 12288}) {
      for (auto K : {/*128, 256,*/ 4096, 12288}) {
        for (auto block_size : {16, 32, 64, 128/*, 256*/}) {
          if (N == 12288 && K == 12288) continue;
          b->Args({M, N, K, block_size, 8});
        }
      }
    }
  }
}

BENCHMARK_CAPTURE(F32Q4GEMM, Before, BeforePacking)->Apply(GemmSizeProducts)->UseRealTime();
BENCHMARK_CAPTURE(F32Q4GEMM, In, InPacking)->Apply(GemmSizeProducts)->UseRealTime();
