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
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& value_data,
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
    std::vector<int64_t> output_dims = input_dims;

    if (use_float16) {
      tester.AddInput<MLFloat16>("query", input_dims, ToFloat16(query_data));
      tester.AddInput<MLFloat16>("key", input_dims, ToFloat16(key_data));
      tester.AddInput<MLFloat16>("value", input_dims, ToFloat16(value_data));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("query", input_dims, query_data);
      tester.AddInput<float>("query", input_dims, key_data);
      tester.AddInput<float>("query", input_dims, value_data);
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
  std::vector<float> query_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> key_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> value_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> output_data = {0.0f, 3.0f, 6.0f, 9.0f, 12.0f, 15.0f};

  RunDummyGPT3AttentionTest(query_data, key_data, value_data, output_data, 1, 2, 3, false);
}

TEST(Gpt3AttentionTest, Dummy_FP16) {
  std::vector<float> query_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> key_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> value_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> output_data = {0.0f, 3.0f, 6.0f, 9.0f, 12.0f, 15.0f};

  RunDummyGPT3AttentionTest(query_data, key_data, value_data, output_data, 1, 2, 3, true);
}

}  // namespace test
}  // namespace onnxruntime
