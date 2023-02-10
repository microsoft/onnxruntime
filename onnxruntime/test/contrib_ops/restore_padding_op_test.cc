// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunRestorePadding(
    const std::vector<float>& input_data,
    const std::vector<float>& output_data,
    const std::vector<int32_t>& token_offset_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    int total_tokens,
    bool use_float16 = false,
    const bool disable_cpu = true,
    const bool disable_cuda = false,
    const bool disable_rocm = true) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !disable_cuda;
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get()) && !disable_rocm;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !disable_cpu;

  if (enable_cpu || enable_cuda || enable_rocm) {
    OpTester tester("RestorePadding", 1, onnxruntime::kMSDomain);

    // shape of inputs:
    //   input:                (total_tokens, hidden_size)
    //   token_offset:         (batch_size, sequence_length)
    // shape of outputs:
    //   output:               (batch_size, sequence_length, hidden_size)
    std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> input_dims = {total_tokens, hidden_size};
    std::vector<int64_t> token_offset_dims = {batch_size, sequence_length};

    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    tester.AddInput<int32_t>("token_offset", token_offset_dims, token_offset_data);

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_rocm) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultRocmExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

static void RunRestorePaddingTests(
    const std::vector<float>& input_data,
    const std::vector<float>& output_data,
    const std::vector<int32_t>& token_offset_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    int total_tokens) {
  bool use_float16 = false;
  constexpr bool disable_cpu = true;
  constexpr bool disable_cuda = false;
  constexpr bool disable_rocm = true;
  RunRestorePadding(input_data, output_data, token_offset_data, batch_size, sequence_length, hidden_size, total_tokens,
                    use_float16, disable_cpu, disable_cuda, disable_rocm);

  use_float16 = true;
  RunRestorePadding(input_data, output_data, token_offset_data, batch_size, sequence_length, hidden_size, total_tokens,
                    use_float16, disable_cpu, disable_cuda, disable_rocm);
}

TEST(RestorePaddingTest, RestorePaddingBatch1_NoPadding) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int total_tokens = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> output_data = input_data;

  std::vector<int32_t> token_offset_data = {0, 1};

  RunRestorePaddingTests(
      input_data,
      output_data,
      token_offset_data,
      batch_size,
      sequence_length,
      hidden_size,
      total_tokens);
}

TEST(RestorePaddingTest, RestorePaddingBatch3_TwoWithPadding) {
  int batch_size = 3;
  int sequence_length = 4;
  int hidden_size = 8;
  int total_tokens = 7;

  std::vector<float> output_data = {
      0.8f, -0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.15f, 0.25f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f,
      0.35f, 0.45f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.135f, 0.235f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f,
      0.335f, 0.435f, 4.7f, 4.8f, 4.9f, 5.0f, 5.1f, 5.2f,
      0.535f, 0.635f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f,
      0.735f, 0.835f, 5.9f, 6.0f, 6.1f, 6.2f, 6.3f, 6.4f};

  std::vector<int32_t> sequence_token_count_data = {1, 2, 4};

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
      0.15f, 0.25f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f,
      0.35f, 0.45f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f,
      0.135f, 0.235f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f,
      0.335f, 0.435f, 4.7f, 4.8f, 4.9f, 5.0f, 5.1f, 5.2f,
      0.535f, 0.635f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f,
      0.735f, 0.835f, 5.9f, 6.0f, 6.1f, 6.2f, 6.3f, 6.4f};

  std::vector<int32_t> token_offset_data = {0, 4, 5, 8, 9, 10, 11, 1, 2, 3, 6, 7};

  RunRestorePaddingTests(
      input_data,
      output_data,
      token_offset_data,
      batch_size,
      sequence_length,
      hidden_size,
      total_tokens);
}

TEST(RestorePaddingTest, RestorePaddingBatch3_AllWithPadding) {
  int batch_size = 3;
  int sequence_length = 4;
  int hidden_size = 8;
  int total_tokens = 6;

  std::vector<float> output_data = {
      0.8f, -0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.15f, 0.25f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f,
      0.35f, 0.45f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.135f, 0.235f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f,
      0.335f, 0.435f, 4.7f, 4.8f, 4.9f, 5.0f, 5.1f, 5.2f,
      0.535f, 0.635f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f,
      0.0f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
      0.15f, 0.25f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f,
      0.35f, 0.45f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f,
      0.135f, 0.235f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f,
      0.335f, 0.435f, 4.7f, 4.8f, 4.9f, 5.0f, 5.1f, 5.2f,
      0.535f, 0.635f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f};

  std::vector<int32_t> token_offset_data = {0, 4, 5, 8, 9, 10, 1, 2, 3, 6, 7, 11};

  RunRestorePaddingTests(
      input_data,
      output_data,
      token_offset_data,
      batch_size,
      sequence_length,
      hidden_size,
      total_tokens);
}

}  // namespace test
}  // namespace onnxruntime
