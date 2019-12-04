// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {

const std::vector<float> ComputeGelu(const std::vector<float>& input_data) {
  std::vector<float> output;
  output.reserve(input_data.size());

  for (size_t i = 0; i < input_data.size(); i++) {
    float x = input_data[i];
    float y = x * (0.5f + 0.5f * tanh(x * (0.035677408136300125f * x * x + 0.7978845608028654f)));
    output.push_back(y);
  }
  return output;
}

const std::vector<float> AddBias(const std::vector<float>& input_data, const std::vector<float>& bias_data) {
  size_t bias_length = bias_data.size();

  std::vector<float> output;
  output.reserve(input_data.size());

  for (size_t i = 0; i < input_data.size(); i++) {
    output.push_back(input_data[i] + bias_data[i % bias_length]);
  }
  return output;
}

const std::vector<float> GetExpectedResult(const std::vector<float>& input_data, const std::vector<float>& bias_data) {
  std::vector<float> add_bias_data = AddBias(input_data, bias_data);
  return ComputeGelu(add_bias_data);
}

static void RunFastGeluTest(
    const std::vector<float>& input_data,
    const std::vector<float>& bias_data,
    const std::vector<float>& output_data,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& bias_dims,
    const std::vector<int64_t>& output_dims,
    bool has_bias = true,
    bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  if (HasCudaEnvironment(min_cuda_architecture)) {
    OpTester tester("FastGelu", 1, onnxruntime::kMSDomain);

    if (use_float16) {
      tester.AddInput<MLFloat16>("X", input_dims, ToFloat16(input_data));
      if (has_bias) {
        tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      }
      tester.AddOutput<MLFloat16>("Y", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("X", input_dims, input_data);
      if (has_bias) {
        tester.AddInput<float>("bias", bias_dims, bias_data);
      }
      tester.AddOutput<float>("Y", output_dims, output_data);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

// This test simulates Gelu in Bert model for float32
static void RunFastGeluTest(
    const std::vector<float>& input_data,
    const std::vector<float>& bias_data,
    int batch_size,
    int sequence_length,
    int hidden_size) {
  std::vector<float> output_data;

  bool has_bias = (bias_data.size() > 0);
  if (has_bias) {
    output_data = GetExpectedResult(input_data, bias_data);
  } else {
    output_data = ComputeGelu(input_data);
  }
  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = input_dims;
  RunFastGeluTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, has_bias);
}

TEST(FastGeluTest, FastGeluWithBiasFloat32) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  RunFastGeluTest(input_data, bias_data, batch_size, sequence_length, hidden_size);
}

TEST(FastGeluTest, FastGeluWithoutBiasFloat32) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> bias_data = {};

  RunFastGeluTest(input_data, bias_data, batch_size, sequence_length, hidden_size);
}

TEST(FastGeluTest, FastGeluWithBiasFloat16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> output_data = {
      0.1851806640625f, 0.054046630859375f, 1.0615234375f, 3.095703125f,
      0, 0.63037109375f, 1.3984375f, 1.3984375f};

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = input_dims;

  RunFastGeluTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, true, true);
}

TEST(FastGeluTest, FastGeluWithoutBiasFloat16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> bias_data = {};

  std::vector<float> output_data = {
      0.63037109375f, -0.154296875f, 0.0f, 0.8408203125f,
      0.345703125f, 0.11578369140625f, 0.1854248046875f, -0.1646728515625f };

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {};
  std::vector<int64_t> output_dims = input_dims;

  RunFastGeluTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, false, true);
}

}  // namespace test
}  // namespace onnxruntime
