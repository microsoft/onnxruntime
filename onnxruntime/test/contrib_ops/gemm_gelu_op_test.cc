// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA)
static void RunGemmGeluGpuTest(const std::vector<float>& input_data, const std::vector<float>& weight_data,
                                   const std::vector<float>& bias_data, const std::vector<float>& output_data,
                                   const std::vector<int64_t>& input_dims, const std::vector<int64_t>& weight_dims,
                                   const std::vector<int64_t>& bias_dims, const std::vector<int64_t>& output_dims,
                                   bool has_bias, bool use_float16 = false) {
  OpTester tester("GemmGelu", 1, onnxruntime::kMSDomain);

  if (use_float16) {
    tester.AddInput<MLFloat16>("X", input_dims, ToFloat16(input_data));
    tester.AddInput<MLFloat16>("W", weight_dims, ToFloat16(weight_data));
    if (has_bias) {
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    }
    tester.AddOutput<MLFloat16>("Y", output_dims, ToFloat16(output_data));
  } else {
    tester.AddInput<float>("X", input_dims, input_data);
    tester.AddInput<float>("W", weight_dims, weight_data);
    if (has_bias) {
      tester.AddInput<float>("bias", bias_dims, bias_data);
    }
    tester.AddOutput<float>("Y", output_dims, output_data);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(GemmGeluTest, GemmGeluWithoutBiasFloat16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int dense_size = 6;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.7f, -0.5f, 0.7f, 1.2f,
      0.3f, 0.1f, 0.8f, -1.6f,
      0.9f, -0.1f, 3.0f, 2.f,
      0.4f, -0.7f, -0.3f, 0.6f};

  std::vector<float> bias_data = {};

  std::vector<float> output_data = {
      3.4894f, 1.8455f, 0.0260f, 0.2229f, -0.1003f, 0.0902f,
      -0.1323f, -0.0953f, 0.0778f, 0.2152f, 0.6715f, -0.0240f};

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weight_dims = {hidden_size, dense_size};
  std::vector<int64_t> bias_dims = {dense_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, dense_size};
 
  RunGemmGeluGpuTest(input_data, weight_data, bias_data, output_data,
                         input_dims, weight_dims, bias_dims, output_dims,
                         false, true);
}

TEST(GemmGeluTest, GemmGeluWithBiasFloat16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int dense_size = 6;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.7f, -0.5f, 0.7f, 1.2f,
      0.3f, 0.1f, 0.8f, -1.6f,
      0.9f, -0.1f, 3.0f, 2.f,
      0.4f, -0.7f, -0.3f, 0.6f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, -0.6f, 0.4f};

  std::vector<float> output_data = {
      2.9862f, 2.4849f, 1.1177f, 2.4329f, -0.1681f, 0.3988f,
      -0.0702f, -0.1633f, 1.2190f, 2.4225f, 0.1428f, 0.2229f};

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weight_dims = {hidden_size, dense_size};
  std::vector<int64_t> bias_dims = {dense_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, dense_size};

  RunGemmGeluGpuTest(input_data, weight_data, bias_data, output_data,
                         input_dims, weight_dims, bias_dims, output_dims,
                         true, true);
}

#endif

}  // namespace test
}  // namespace onnxruntime
