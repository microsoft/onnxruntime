// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_MINIMAL_BUILD

#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/qmath.h"
#include "contrib_ops/cpu/quantization/dequantize_blockwise_weight.h"

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {

TEST(MatMulWithQuantWeight, MatMul2DSym) {
  // (100 x 41) X (41 x 288)
  constexpr int64_t M = 100;
  constexpr int64_t N = 288;
  constexpr int64_t K = 41;

  OpTester test("MatMulWithQuantWeight", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", 32);
  test.AddAttribute<int64_t>("bits", 4);
  test.AddAttribute<int64_t>("has_zero_point", 0);

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
  int v = -2;
  for (size_t i = 0; i < N * K; i++) {
    if (v == 0 || v == -3 || v == 3) v++;
    input1_f_vals[i] = (float)v;
    if (++v >= 8) {
      v = -8;
    }
  }

  constexpr size_t number_of_blob = (K + 32 - 1) / 32 * N;
  constexpr size_t buf_size = number_of_blob * (32 * 4 / 8);
  std::vector<uint8_t> input1_vals(buf_size);
  std::vector<float> scales(number_of_blob);

  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);
  contrib::QuantizeBlockwiseWeight<float, 32, 4>(
      reinterpret_cast<contrib::SubByteBlob<32, 4>*>(input1_vals.data()),
      scales.data(),
      nullptr,
      input1_f_vals.data(),
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

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}

TEST(MatMulWithQuantWeight, MatMul2DBlkZp) {
  // (100 x 41) X (41 x 288)
  constexpr int64_t M = 100;
  constexpr int64_t N = 288;
  constexpr int64_t K = 41;

  OpTester test("MatMulWithQuantWeight", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", 32);
  test.AddAttribute<int64_t>("bits", 4);
  test.AddAttribute<int64_t>("has_zero_point", 1);

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
  contrib::QuantizeBlockwiseWeight<float, 32, 4>(
      reinterpret_cast<contrib::SubByteBlob<32, 4>*>(input1_vals.data()),
      scales.data(),
      zp.data(),
      input1_f_vals.data(),
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
