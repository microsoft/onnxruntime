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

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {

static constexpr int QBits = 4;

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

  MlasQuantizeBlockwise<float>(
      quant_vals.data(),
      scales.data(),
      zp != nullptr ? zp->data() : nullptr,
      raw_vals.data(),
      block_size,
      true,
      K,
      N,
      N,
      tp.get());

  // Note that input1_f_vals is NxK after dequant
  MlasDequantizeBlockwise<float>(
      raw_vals.data(),                       // dequantized output
      quant_vals.data(),                     // quantized input
      scales.data(),                         // quantization scales
      zp != nullptr ? zp->data() : nullptr,  // quantization zero points
      block_size,                            // quantization block size
      true,                                  // columnwise quantization
      K,                                     // number of rows
      N,                                     // number of columns
      tp.get());
}

void RunTest(int64_t M, int64_t N, int64_t K, int64_t block_size, bool has_zeropoint, bool use_float16) {
  RandomValueGenerator random{1234};
  std::vector<float> input0_vals(random.Gaussian<float>(std::vector<int64_t>({M, K}), 0.0f, 0.25f));
  std::vector<float> input1_f_vals(random.Gaussian<float>(std::vector<int64_t>({K, N}), 0.0f, 0.25f));

#if 0  // for Debugging
  std::vector<float> input1_f_vals_trans(N * K);
  MlasTranspose(input1_f_vals.data(), input1_f_vals_trans.data(), K, N);
#endif

  int meta_rows;
  int meta_cols;
  MlasBlockwiseQuantMetaShape<float>((int)block_size, true, (int)K, (int)N, meta_rows, meta_cols);

  int q_rows;
  int q_cols;
  MlasBlockwiseQuantizedShape<float>((int)block_size, true, (int)K, (int)N, q_rows, q_cols);

  std::vector<uint8_t> input1_vals(q_rows * q_cols);
  std::vector<float> scales(meta_rows * meta_cols);

  // TODO!! THIS SHOULD BE PROVIDED BY MLAS
  // sub 8b packing always happen on the column dimension
  const int packed_meta_rows = (meta_rows * QBits + 7) / 8;
  std::vector<uint8_t> zp(packed_meta_rows * meta_cols);

  QuantizeDequantize(input1_f_vals,
                     input1_vals,
                     scales,
                     has_zeropoint ? &zp : nullptr,
                     static_cast<int32_t>(N),
                     static_cast<int32_t>(K),
                     static_cast<int32_t>(block_size));

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

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", QBits);
  if (use_float16) {
    test.AddInput<MLFloat16>("A", {M, K}, ToFloat16(input0_vals), false);
    test.AddInput<uint8_t>("B", {q_cols, q_rows}, input1_vals, true);
    test.AddInput<MLFloat16>("scales", {meta_cols * meta_rows}, ToFloat16(scales), true);
    if (has_zeropoint) {
      test.AddInput<uint8_t>("zero_points", {meta_cols * packed_meta_rows}, zp, true);
    }

    test.AddOutput<MLFloat16>("Y", {M, N}, ToFloat16(expected_vals));
    test.SetOutputAbsErr("Y", 0.02f);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  } else {
    test.AddInput<float>("A", {M, K}, input0_vals, false);
    test.AddInput<uint8_t>("B", {q_cols, q_rows}, input1_vals, true);
    test.AddInput<float>("scales", {meta_cols * meta_rows}, scales, true);
    if (has_zeropoint) {
      test.AddInput<uint8_t>("zero_points", {meta_cols * packed_meta_rows}, zp, true);
    }

    test.AddOutput<float>("Y", {M, N}, expected_vals);

    test.Run();
  }
}

TEST(MatMulNBits, Float32) {
  for (auto M : {1, 2, 100}) {
    for (auto N : {1, 2, 32, 288}) {
      for (auto K : {16, 32, 64, 128, 256, 1024, 93, 1234}) {
        for (auto block_size : {16, 32, 64, 128}) {
          RunTest(M, N, K, block_size, false, false);
          RunTest(M, N, K, block_size, true, false);
        }
      }
    }
  }
}

#if defined(USE_CUDA)
TEST(MatMulNBits, Float16) {
  for (auto M : {1, 2, 100}) {
    for (auto N : {1, 2, 32, 288}) {
      for (auto K : {16, 32, 64, 128, 256, 1024, 93, 1234}) {
        for (auto block_size : {16, 32, 64, 128}) {
          RunTest(M, N, K, block_size, false, true);
          RunTest(M, N, K, block_size, true, true);
        }
      }
    }
  }
}

#endif
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
