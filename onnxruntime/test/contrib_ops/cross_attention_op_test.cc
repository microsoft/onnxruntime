// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env_var_utils.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace test {

static void RunCrossAttentionTest(
    const std::vector<float>& query_data,               // query:  [batch_size, sequence_length, hidden_size]
    const std::vector<float>& key_data,                 // key:    [batch_size, kv_sequence_length, hidden_size]
    const std::vector<float>& value_data,               // value:  [batch_size, kv_sequence_length, v_hidden_size]
    const std::vector<float>& bias_data,                // bias:   [hidden_size + hidden_size + v_hidden_size]
    const std::vector<int32_t>& key_padding_mask_data,  // key_padding_mask: see below
    AttentionMaskType mask_type,                        // 1 for [batch_size], 2 for [batch_size, kv_sequence_length]
    const std::vector<float>& output_data,              // output: [batch_size, sequence_length, v_hidden_size]
    int number_of_heads,
    int batch_size,
    int sequence_length,
    int kv_sequence_length,
    int hidden_size,
    int v_hidden_size,
    bool use_float16 = false,
    const bool disable_cpu = false,
    const bool disable_cuda = false,
    const bool disable_rocm = false) {
  kv_sequence_length = (kv_sequence_length == 0 ? sequence_length : kv_sequence_length);

  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !disable_cuda;
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get()) && !disable_rocm;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !disable_cpu;

  if (enable_cpu || enable_cuda || enable_rocm) {
    OpTester tester("CrossAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));

    std::vector<int64_t> query_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> key_dims = {batch_size, kv_sequence_length, hidden_size};
    std::vector<int64_t> value_dims = {batch_size, kv_sequence_length, v_hidden_size};
    std::vector<int64_t> bias_dims = {hidden_size + hidden_size + v_hidden_size};
    std::vector<int64_t> output_dims = {batch_size, sequence_length, v_hidden_size};

    std::vector<int64_t> mask_dims_1 = {batch_size};
    std::vector<int64_t> mask_dims_2 = {batch_size, kv_sequence_length};
    std::vector<int64_t>& key_padding_mask_dims = (mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN)
                                                      ? mask_dims_1
                                                      : mask_dims_2;

    if (use_float16) {
      tester.AddInput<MLFloat16>("query", query_dims, ToFloat16(query_data));
      tester.AddInput<MLFloat16>("key", key_dims, ToFloat16(key_data));
      tester.AddInput<MLFloat16>("value", value_dims, ToFloat16(value_data));

      if (bias_data.size()) {
        tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }

      if (key_padding_mask_data.size()) {
        tester.AddInput<int32_t>("key_padding_mask", key_padding_mask_dims, key_padding_mask_data);
      } else {
        tester.AddOptionalInputEdge<int32_t>();
      }

      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("query", query_dims, query_data);
      tester.AddInput<float>("key", key_dims, key_data);
      tester.AddInput<float>("value", value_dims, value_data);

      if (bias_data.size()) {
        tester.AddInput<float>("bias", bias_dims, bias_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }

      if (key_padding_mask_data.size()) {
        tester.AddInput<int32_t>("key_padding_mask", key_padding_mask_dims, key_padding_mask_data);
      } else {
        tester.AddOptionalInputEdge<int32_t>();
      }

      tester.AddOutput<float>("output", output_dims, output_data);
    }

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

TEST(CrossAttentionTest, CrossAttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;
  int kv_sequence_length = 3;
  int v_hidden_size = 2;

  std::vector<float> query_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> key_data = {0.1f, 0.2f, 0.3f, 0.4f,
                                 0.5f, 0.6f, 0.7f, 0.8f,
                                 0.9f, 1.0f, 1.1f, 1.2f};

  std::vector<float> value_data = {0.6f, 0.5f,
                                   0.4f, 0.3f,
                                   0.2f, 0.1f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f,
      0.5f, 0.7f, 0.2f, 1.2f,
      0.5f, 0.4f};

  std::vector<float> output_data = {0.99434918f, 0.0f,
                                    0.9887343f, 0.74572039f};

  std::vector<int32_t> key_padding_mask_data = {2L};
  constexpr AttentionMaskType mask_type = AttentionMaskType::MASK_1D_KEY_SEQ_LEN;

  bool use_float16 = false;

  constexpr bool disable_cpu = true;  // not supported in cpu right now.
  constexpr bool disable_cuda = false;
  constexpr bool disable_rocm = true;  // not supported in rocm right now.

  RunCrossAttentionTest(
      query_data, key_data, value_data, bias_data, key_padding_mask_data, mask_type, output_data,
      number_of_heads, batch_size, sequence_length, kv_sequence_length, hidden_size, v_hidden_size,
      use_float16, disable_cpu, disable_cuda, disable_rocm);
}

}  // namespace test
}  // namespace onnxruntime
