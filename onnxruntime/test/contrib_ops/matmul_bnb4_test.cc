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
#include "contrib_ops/cpu/quantization/dequantize_blockwise_bnb4.h"

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {

void QuantizeDequantizeBnb4(std::vector<float>& raw_vals,  // N X K
                            std::vector<uint8_t>& quant_vals,
                            std::vector<float>& absmax,
                            int32_t quant_type,
                            int32_t N,
                            int32_t K,
                            int32_t block_size) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  contrib::QuantizeBlockwiseBnb4<float>(
      quant_vals.data(),
      raw_vals.data(),
      absmax.data(),
      block_size,
      quant_type,
      N,
      K,
      tp.get());

  contrib::DequantizeBlockwiseBnb4<float>(
      raw_vals.data(),
      quant_vals.data(),
      absmax.data(),
      block_size,
      quant_type,
      N,
      K,
      tp.get());
}

void RunTest(int64_t quant_type, int64_t M, int64_t N, int64_t K, int64_t block_size, bool use_float16) {
  RandomValueGenerator random{1234};
  std::vector<float> input0_vals(random.Gaussian<float>(std::vector<int64_t>({M, K}), 0.0f, 0.25f));
  // quantizer expects transposed weights, N X K
  std::vector<float> input1_f_vals(random.Gaussian<float>(std::vector<int64_t>({N, K}), 0.0f, 0.25f));

  int64_t numel = N * K;
  int64_t quantized_numel = (numel + 1) / 2;
  int64_t total_block_count = (numel + block_size - 1) / block_size;
  std::vector<uint8_t> input1_vals(quantized_numel);
  std::vector<float> absmax(total_block_count);

  QuantizeDequantizeBnb4(input1_f_vals,
                         input1_vals,
                         absmax,
                         static_cast<int32_t>(quant_type),
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

  OpTester test("MatMulBnb4", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("quant_type", quant_type);
  if (use_float16) {
    test.AddInput<MLFloat16>("A", {M, K}, ToFloat16(input0_vals), false);
    test.AddInput<uint8_t>("B", {quantized_numel}, input1_vals, true);
    test.AddInput<MLFloat16>("absmax", {total_block_count}, ToFloat16(absmax), true);

    test.AddOutput<MLFloat16>("Y", {M, N}, ToFloat16(expected_vals));
    test.SetOutputAbsErr("Y", 0.02f);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  } else {
    test.AddInput<float>("A", {M, K}, input0_vals, false);
    test.AddInput<uint8_t>("B", {quantized_numel}, input1_vals, true);
    test.AddInput<float>("absmax", {total_block_count}, absmax, true);

    test.AddOutput<float>("Y", {M, N}, expected_vals);

    test.Run();
  }
}

TEST(MatMulBnb4, Float32) {
  for (auto qt : {0, 1}) {
    for (auto M : {1, 2, 100}) {
      for (auto N : {1, 2, 32, 288}) {
        for (auto K : {16, 32, 64, 128, 256, 1024, 93, 1234}) {
          for (auto block_size : {16, 32, 64, 128}) {
            RunTest(qt, M, N, K, block_size, false);
          }
        }
      }
    }
  }
}

#if defined(USE_CUDA)
TEST(MatMulBnb4, Float16) {
  for (auto qt : {0, 1}) {
    for (auto M : {1, 2, 100}) {
      for (auto N : {1, 2, 32, 288}) {
        for (auto K : {16, 32, 64, 128, 256, 1024, 93, 1234}) {
          for (auto block_size : {16, 32, 64, 128}) {
            RunTest(qt, M, N, K, block_size, true);
          }
        }
      }
    }
  }
}

#endif
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
