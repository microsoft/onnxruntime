// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {
namespace bias_split_gelu_test {
std::vector<float> ComputeGelu(const std::vector<float>& input_data) {
  std::vector<float> output;
  output.reserve(input_data.size());

  for (size_t i = 0; i < input_data.size(); i++) {
    float x = input_data[i];
    float y = x * (0.5f * (1.0f + std::erff(x / static_cast<float>(M_SQRT2))));
    output.push_back(y);
  }
  return output;
}

std::vector<float> AddBias(const std::vector<float>& input_data, const std::vector<float>& bias_data) {
  size_t bias_length = bias_data.size();

  std::vector<float> output;
  output.reserve(input_data.size());

  for (size_t i = 0; i < input_data.size(); i++) {
    output.push_back(input_data[i] + bias_data[i % bias_length]);
  }
  return output;
}

void Split(const std::vector<float>& input_data,
           const std::vector<int64_t>& input_dims,
           std::vector<float>& left_half_data, std::vector<float>& right_half_data) {
  std::size_t length = input_data.size();
  left_half_data.reserve(length / 2);
  right_half_data.reserve(length / 2);

  int64_t index = 0;
  for (int64_t i = 0; i < input_dims[0]; i++) {
    for (int64_t j = 0; j < input_dims[1]; j++) {
      for (int64_t k = 0; k < input_dims[2]; k++, index++) {
        if (k < input_dims[2] / 2) {
          left_half_data.push_back(input_data[index]);
        } else {
          right_half_data.push_back(input_data[index]);
        }
      }
    }
  }
}

std::vector<float> GetExpectedResult(const std::vector<float>& input_data,
                                     const std::vector<int64_t>& input_dims,
                                     const std::vector<float>& bias_data) {
  std::vector<float> add_bias_data = AddBias(input_data, bias_data);
  std::vector<float> left_half_data;
  std::vector<float> right_half_data;
  Split(add_bias_data, input_dims, left_half_data, right_half_data);
  std::vector<float> right_gelu_data = ComputeGelu(right_half_data);

  std::vector<float> output_data;
  output_data.reserve(left_half_data.size());
  for (std::size_t i = 0; i < left_half_data.size(); i++) {
    output_data.push_back(left_half_data[i] * right_gelu_data[i]);
  }
  return output_data;
}
}  // namespace bias_split_gelu_test

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML)

static void RunBiasSplitGeluGpuTest(const std::vector<float>& input_data,
                                    const std::vector<float>& bias_data,
                                    const std::vector<float>& output_data,
                                    const std::vector<int64_t>& input_dims,
                                    const std::vector<int64_t>& bias_dims,
                                    const std::vector<int64_t>& output_dims,
                                    bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  bool enable_dml = (nullptr != DefaultDmlExecutionProvider().get());

  if (!enable_cuda && !enable_rocm && !enable_dml) {
    return;
  }

  OpTester tester("BiasSplitGelu", 1, onnxruntime::kMSDomain);

  if (use_float16) {
    tester.AddInput<MLFloat16>("X", input_dims, ToFloat16(input_data));
    tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    tester.AddOutput<MLFloat16>("Y", output_dims, ToFloat16(output_data));
  } else {
    tester.AddInput<float>("X", input_dims, input_data);
    tester.AddInput<float>("bias", bias_dims, bias_data);
    tester.AddOutput<float>("Y", output_dims, output_data);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (enable_cuda) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  }
  if (enable_rocm) {
    execution_providers.push_back(DefaultRocmExecutionProvider());
  }
  if (enable_dml) {
    execution_providers.push_back(DefaultDmlExecutionProvider());
  }

  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

static void RunBiasSplitGeluTest(int64_t batch_size, int64_t sequence_length, int64_t hidden_size) {
  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size / 2};

  RandomValueGenerator random{};
  std::vector<float> input_data = random.Gaussian<float>(input_dims, 0.0f, 0.3f);
  std::vector<float> bias_data = random.Gaussian<float>(bias_dims, 0.0f, 0.3f);
  std::vector<float> output_data = bias_split_gelu_test::GetExpectedResult(input_data, input_dims, bias_data);

  RunBiasSplitGeluGpuTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims);
}

TEST(BiasSplitGeluTest, BiasSplitGeluTest_HiddenSize_2560) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t sequence_length = 5;
  constexpr int64_t hidden_size = 2560;
  RunBiasSplitGeluTest(batch_size, sequence_length, hidden_size);
}

TEST(BiasSplitGeluTest, BiasSplitGeluTest_HiddenSize_5120) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t sequence_length = 1;
  constexpr int64_t hidden_size = 5120;
  RunBiasSplitGeluTest(batch_size, sequence_length, hidden_size);
}

TEST(BiasSplitGeluTest, BiasSplitGeluTest_HiddenSize_10240) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t sequence_length = 2;
  constexpr int64_t hidden_size = 10240;
  RunBiasSplitGeluTest(batch_size, sequence_length, hidden_size);
}

#endif

}  // namespace test
}  // namespace onnxruntime
