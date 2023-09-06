// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_MINIMAL_BUILD

#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/qmath.h"
#include "contrib_ops/cpu/quantization/dequantize_blockwise.h"

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {

template <typename T>
void QuantizeDequantize(std::vector<T>& raw_vals,
                        std::vector<uint8_t>& quant_vals,
                        std::vector<T>& scales,
                        std::vector<uint8_t>* zp,
                        int64_t N,
                        int64_t K,
                        int64_t block_size) {
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

void RunTest(int64_t M, int64_t N, int64_t K, int64_t block_size) {
  OpTester test("MatMulWithCompressWeight", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", 4);

  RandomValueGenerator random{1234};
  std::vector<float> input0_vals(random.Gaussian<float>(std::vector<int64_t>({M, K}), 0.0f, 0.25f));
  std::vector<float> input1_f_vals(random.Gaussian<float>(std::vector<int64_t>({K, N}), 0.0f, 0.25f));

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

  QuantizeDequantize<float>(input1_f_vals, input1_vals, scales, nullptr, N, K, block_size);

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input0_vals[m * K + k] * input1_f_vals[n * K + k];
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddInput<float>("A", {M, K}, input0_vals, false);
  test.AddInput<uint8_t>("B", {N, block_per_k, block_blob_size}, input1_vals, true);
  test.AddInput<float>("scales", {N, block_per_k}, scales, true);

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}

TEST(MatMulWithCompressWeight, MatMul2DSym) {
  // RunTest(1, 288, 1024, 16);
  // RunTest(2, 288, 1024, 16);

  for (auto M : {1, 2, 100}) {
    for (auto N : {1, 2, 32, 288}) {
      for (auto K : {16, 32, 64, 128, 256, 1024, 93, 1234}) {
        for (auto block_size : {16, 32, 64, 128}) {
          std::cout << "Begin M:" << M << ",N:" << N << ",K:" << K << ",block_size:" << block_size << std::endl;
          RunTest(M, N, K, block_size);
          std::cout << "End M:" << M << ",N:" << N << ",K:" << K << ",block_size:" << block_size << std::endl;
        }
      }
    }
  }
}

TEST(MatMulWithCompressWeight, MatMul2DSym_1024) {
  // RunTest(1, 288, 1024, 16);
  // RunTest(2, 288, 1024, 16);

  for (auto M : {100}) {
    for (auto N : {32}) {
      for (auto K : {1024}) {
        for (auto block_size : {16}) {
          std::cout << "Begin M:" << M << ",N:" << N << ",K:" << K << ",block_size:" << block_size << std::endl;
          RunTest(M, N, K, block_size);
          std::cout << "End M:" << M << ",N:" << N << ",K:" << K << ",block_size:" << block_size << std::endl;
        }
      }
    }
  }
}

TEST(MatMulWithCompressWeight, MatMul2DBlkZp) {
  // (100 x 41) X (41 x 288)
  constexpr int64_t M = 100;
  constexpr int64_t N = 288;
  constexpr int64_t K = 41;

  OpTester test("MatMulWithCompressWeight", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", 32);
  test.AddAttribute<int64_t>("bits", 4);

  std::vector<float> input0_vals(M * K);
  float fv = -135.f;
  for (auto& f : input0_vals) {
    f = fv / 128;
    fv++;
    if (fv > 135.f) {
      fv = -135.f;
    }
  }

  std::vector<float> input1_f_vals(N * K);
  int v = 0;
  for (size_t i = 0; i < N * K; i++) {
    input1_f_vals[i] = (float)v;
    if (++v >= 16) {
      v = 0;
    }
  }

  constexpr size_t number_of_blob = (K + 32 - 1) / 32 * N;
  constexpr size_t buf_size = number_of_blob * (32 * 4 / 8);
  std::vector<uint8_t> input1_vals(buf_size);
  std::vector<float> scales(number_of_blob);
  std::vector<uint8_t> zp(number_of_blob);

  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);
  contrib::QuantizeBlockwise<float, 32, 4>(
      input1_vals.data(),
      input1_f_vals.data(),
      scales.data(),
      zp.data(),
      N,
      K,
      tp.get());

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input0_vals[m * K + k] * input1_f_vals[k * N + n];
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddInput<float>("A", {M, K}, input0_vals, false);
  test.AddInput<uint8_t>("B", {N, (K + 31) / 32, 16}, input1_vals, true);
  test.AddInput<float>("scales", {N, (K + 31) / 32}, scales, true);
  test.AddInput<uint8_t>("zero_points", {N, (K + 31) / 32}, zp, true);

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
