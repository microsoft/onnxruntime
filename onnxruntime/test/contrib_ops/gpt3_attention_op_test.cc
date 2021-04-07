// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

//dummy test
static void RunDummyGPT3AttentionTest(
    const std::vector<float>& input_data,
    const std::vector<float>& hidden_state_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = false;
  if (enable_cpu || enable_cuda) {
    OpTester tester("Gpt3Attention", 1, onnxruntime::kMSDomain);

    std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> hidden_state_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> output_dims = input_dims;

    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      tester.AddInput<MLFloat16>("hidden_state", hidden_state_dims, ToFloat16(hidden_state_data));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      tester.AddInput<float>("hidden_state", hidden_state_dims, hidden_state_data);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

TEST(Gpt3AttentionTest, Dummy_FP32) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> hidden_state_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> output_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  RunDummyGPT3AttentionTest(input_data, hidden_state_data, output_data, 1, 2, 3, false);
}

TEST(Gpt3AttentionTest, Dummy_FP16) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> hidden_state_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> output_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  RunDummyGPT3AttentionTest(input_data, hidden_state_data, output_data, 1, 2, 3, true);
}

}  // namespace test
}  // namespace onnxruntime
