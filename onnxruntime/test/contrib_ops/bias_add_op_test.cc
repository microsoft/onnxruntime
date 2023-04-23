// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <functional>
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML)
static std::vector<float> GetExpectedResult(const std::vector<float>& input_data,
                                            const std::vector<float>& bias_data,
                                            const std::vector<float>& skip_data) {
  std::vector<float> output_data;
  output_data.reserve(input_data.size());

  size_t bias_length = bias_data.size();
  for (size_t i = 0; i < input_data.size(); i++) {
    output_data.push_back(input_data[i] + bias_data[i % bias_length] + skip_data[i]);
  }
  return output_data;
}

static void RunSkipBiasGpuTest(const std::vector<float>& input_data,
                               const std::vector<float>& bias_data,
                               const std::vector<float>& skip_data,
                               const std::vector<float>& output_data,
                               const std::vector<int64_t>& input_dims,
                               const std::vector<int64_t>& bias_dims,
                               const std::vector<int64_t>& skip_dims,
                               const std::vector<int64_t>& output_dims,
                               bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  bool enable_dml = (nullptr != DefaultDmlExecutionProvider().get());

  if (!enable_cuda && !enable_rocm && !enable_dml) {
    return;
  }

  OpTester tester("BiasAdd", 1, onnxruntime::kMSDomain);

  if (use_float16) {
    tester.AddInput<MLFloat16>("X", input_dims, ToFloat16(input_data));
    tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    tester.AddInput<MLFloat16>("skip", skip_dims, ToFloat16(skip_data));
    tester.AddOutput<MLFloat16>("Y", output_dims, ToFloat16(output_data));
  } else {
    tester.AddInput<float>("X", input_dims, input_data);
    tester.AddInput<float>("bias", bias_dims, bias_data);
    tester.AddInput<float>("skip", skip_dims, skip_data);
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

static void RunBiasAddTest(int64_t batch_size, int64_t image_size, int64_t num_channels) {
  std::vector<int64_t> input_dims = {batch_size, image_size, num_channels};
  std::vector<int64_t> bias_dims = {num_channels};
  std::vector<int64_t>& skip_dims = input_dims;
  std::vector<int64_t>& output_dims = input_dims;

  RandomValueGenerator random{};
  std::vector<float> input_data = random.Gaussian<float>(input_dims, 0.0f, 0.3f);
  std::vector<float> bias_data = random.Gaussian<float>(bias_dims, 0.0f, 0.3f);
  std::vector<float> skip_data = random.Gaussian<float>(skip_dims, 0.0f, 0.3f);
  std::vector<float> output_data = GetExpectedResult(input_data, bias_data, skip_data);

  RunSkipBiasGpuTest(input_data, bias_data, skip_data, output_data, input_dims, bias_dims, skip_dims, output_dims);
}

TEST(BiasAddTest, BiasAddTest_HiddenSize_320) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t image_size = 5;
  constexpr int64_t num_channels = 320;
  RunBiasAddTest(batch_size, image_size, num_channels);
}

TEST(BiasAddTest, BiasAddTest_HiddenSize_640) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t image_size = 1;
  constexpr int64_t num_channels = 640;
  RunBiasAddTest(batch_size, image_size, num_channels);
}

TEST(BiasAddTest, BiasAddTest_HiddenSize_1280) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t image_size = 2;
  constexpr int64_t num_channels = 1280;
  RunBiasAddTest(batch_size, image_size, num_channels);
}
#endif

}  // namespace test
}  // namespace onnxruntime
