// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
constexpr float epsilon_ = 1e-12f;

static void RunTest(
    const std::vector<float>& input_data,
    const std::vector<float>& skip_data,
    const std::vector<float>& gamma_data,
    const std::vector<float>& beta_data,
    const std::vector<float>& bias_data,
    const std::vector<float>& output_data,
    float epsilon,
    int batch_size,
    int sequence_length,
    int hidden_size,
    bool use_float16 = false) {
  // Input and output shapes
  //   Input 0 - input: (batch_size, sequence_length, hidden_size)
  //   Input 1 - skip : (batch_size, sequence_length, hidden_size)
  //   Input 2 - gamma: (hidden_size)
  //   Input 3 - beta : (hidden_size)
  //   Output         : (batch_size, sequence_length, hidden_size)
  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> skip_dims = input_dims;
  std::vector<int64_t> gamma_dims = {hidden_size};
  std::vector<int64_t> beta_dims = gamma_dims;
  std::vector<int64_t> bias_dims = gamma_dims;
  std::vector<int64_t> output_dims = input_dims;

  if (!use_float16) {
    OpTester test("SkipLayerNormalization", 1, onnxruntime::kMSDomain);
    test.AddInput<float>("input", input_dims, input_data);
    test.AddInput<float>("skip", skip_dims, skip_data);
    test.AddInput<float>("gamma", gamma_dims, gamma_data);
    test.AddInput<float>("beta", beta_dims, beta_data);
    test.AddAttribute("epsilon", epsilon);
    if (!bias_data.empty()) {
      test.AddInput<float>("bias", bias_dims, bias_data);
    }

    test.AddOutput<float>("output", output_dims, output_data);
    test.Run();
  } else if (HasCudaEnvironment(530 /*min_cuda_architecture*/)) {
    OpTester test("SkipLayerNormalization", 1, onnxruntime::kMSDomain);
    test.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
    test.AddInput<MLFloat16>("skip", skip_dims, ToFloat16(skip_data));
    test.AddInput<MLFloat16>("gamma", gamma_dims, ToFloat16(gamma_data));
    test.AddInput<MLFloat16>("beta", beta_dims, ToFloat16(beta_data));
    test.AddAttribute("epsilon", epsilon);
    if (!bias_data.empty()) {
      test.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    }

    test.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(SkipLayerNormTest, SkipLayerNormBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> skip_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          std::vector<float>(),
          output_data,
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size);
}

TEST(SkipLayerNormTest, SkipLayerNormBatch1_Float16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> skip_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          std::vector<float>(),
          output_data,
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size,
          true);
}

TEST(SkipLayerNormTest, SkipLayerNormBatch2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> skip_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      1.8f, -0.3f, 0.0f, 1.f,
      -0.5f, 0.4f, 0.8f, -0.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438,
      0.55470430850982666, -0.15080101788043976, -2.3229825496673584, 3.255286693572998,
      0.15631480515003204, 0.21066918969154358, 4.9432611465454102, -1.7957965135574341};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          std::vector<float>(),
          output_data,
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size);
}

TEST(SkipLayerNormTest, SkipLayerNormBatch2_Bias) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.7f, -0.4f, -0.2f, 1.2f,
      0.4f, 0.3f, 0.1f, -0.4f,
      0.7f, -0.4f, -0.2f, 1.2f,
      0.4f, 0.3f, 0.1f, -0.4f};

  std::vector<float> skip_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      1.8f, -0.3f, 0.0f, 1.f,
      -0.5f, 0.4f, 0.8f, -0.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> bias_data = {
      0.1f, -0.1f, 0.2f, -0.2f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438,
      0.55470430850982666, -0.15080101788043976, -2.3229825496673584, 3.255286693572998,
      0.15631480515003204, 0.21066918969154358, 4.9432611465454102, -1.7957965135574341};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          bias_data,
          output_data,
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size);
}

}  // namespace test
}  // namespace onnxruntime
