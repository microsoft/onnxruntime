// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/narrow.h"
#include "core/platform/env_var_utils.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "test/contrib_ops/attention_op_test_helper.h"

namespace onnxruntime {
using contrib::AttentionMaskType;
namespace test {

static void RunPackedAttentionTest(
    const std::vector<float>& input_data,                    // input:      [token_count, hidden_size]
    const std::vector<float>& weights_data,                  // weights:    [hidden_size, 3 * hidden_size]
    const std::vector<float>& bias_data,                     // bias:       [3 * hidden_size]
    const std::vector<int32_t>& token_offset,                // token_offset: [batch_size, sequence_length]
    const std::vector<int32_t>& cumulative_sequence_length,  // cum_seq_len: [batch_size + 1]
    const std::vector<float>& output_data,                   // output:     [token_count, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    int token_count,
    bool use_float16,
    bool use_scale,
    std::vector<int32_t> qkv_sizes,
    const std::vector<float>& relative_position_bias_data) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);

  if (enable_cuda) {
    OpTester tester("PackedAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));

    int32_t qkv_hidden_size_sum;
    int32_t v_hidden_size;
    int32_t head_size_k;
    if (qkv_sizes.size() != 0) {
      qkv_hidden_size_sum = qkv_sizes[0] + qkv_sizes[1] + qkv_sizes[2];
      std::vector<int64_t> sizes_attribute{qkv_sizes[0], qkv_sizes[1], qkv_sizes[2]};
      tester.AddAttribute<std::vector<int64_t>>("qkv_hidden_sizes", sizes_attribute);
      v_hidden_size = qkv_sizes[2];
      head_size_k = qkv_sizes[1] / number_of_heads;
    } else {
      qkv_hidden_size_sum = 3 * hidden_size;
      v_hidden_size = hidden_size;
      head_size_k = hidden_size / number_of_heads;
    }

    if (use_scale) {
      tester.AddAttribute<float>("scale", static_cast<float>(1.f / sqrt(head_size_k)));
    }

    std::vector<int64_t> input_dims = {token_count, hidden_size};
    std::vector<int64_t> weights_dims = {hidden_size, qkv_hidden_size_sum};
    std::vector<int64_t> bias_dims = {qkv_hidden_size_sum};
    std::vector<int64_t> token_offset_dims = {batch_size, sequence_length};
    std::vector<int64_t> cum_seq_len_dims = {batch_size + 1};
    std::vector<int64_t> relative_position_bias_data_dims = {batch_size, number_of_heads, sequence_length, sequence_length};
    std::vector<int64_t> output_dims = {token_count, v_hidden_size};
    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      tester.AddInput<MLFloat16>("weight", weights_dims, ToFloat16(weights_data));
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      tester.AddInput<int32_t>("token_offset", token_offset_dims, token_offset);
      tester.AddInput<int32_t>("cumulative_sequence_length", cum_seq_len_dims, cumulative_sequence_length);
      if (relative_position_bias_data.size() > 0) {
        tester.AddInput<MLFloat16>("relative_position_bias", relative_position_bias_data_dims, ToFloat16(relative_position_bias_data));
      }

      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      tester.AddInput<float>("weight", weights_dims, weights_data);
      tester.AddInput<float>("bias", bias_dims, bias_data);
      tester.AddInput<int32_t>("token_offset", token_offset_dims, token_offset);
      tester.AddInput<int32_t>("cumulative_sequence_length", cum_seq_len_dims, cumulative_sequence_length);
      if (relative_position_bias_data.size() > 0) {
        tester.AddInput<float>("relative_position_bias", relative_position_bias_data_dims, relative_position_bias_data);
      }

      tester.AddOutput<float>("output", output_dims, output_data);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

static void RunPackedAttentionTest(
    const std::vector<float>& input_data,                    // input:      [token_count, hidden_size]
    const std::vector<float>& weights_data,                  // weights:    [hidden_size, 3 * hidden_size]
    const std::vector<float>& bias_data,                     // bias:       [3 * hidden_size]
    const std::vector<int32_t>& token_offset,                // token_offset: [batch_size, sequence_length]
    const std::vector<int32_t>& cumulative_sequence_length,  // cum_seq_len: [batch_size + 1]
    const std::vector<float>& output_data,                   // output:     [token_count, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    int token_count,
    std::vector<int32_t> qkv_sizes = {},
    const std::vector<float>& relative_position_bias_data = {}) {
#define InvokePackedAttentionTest(use_float16, use_scale) \
  RunPackedAttentionTest(                                 \
      input_data,                                         \
      weights_data,                                       \
      bias_data,                                          \
      token_offset,                                       \
      cumulative_sequence_length,                         \
      output_data,                                        \
      batch_size,                                         \
      sequence_length,                                    \
      hidden_size,                                        \
      number_of_heads,                                    \
      token_count,                                        \
      use_float16,                                        \
      use_scale,                                          \
      qkv_sizes,                                          \
      relative_position_bias_data);

  InvokePackedAttentionTest(true, true);
  InvokePackedAttentionTest(true, false);
  InvokePackedAttentionTest(false, true);
  InvokePackedAttentionTest(false, false);
}

TEST(PackedAttentionTest, NoPack) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> token_offset{0, 1};
  std::vector<int32_t> cum_seq_len{0, 2};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunPackedAttentionTest(
      input_data,
      weight_data,
      bias_data,
      token_offset,
      cum_seq_len,
      output_data,
      batch_size,
      sequence_length,
      hidden_size,
      number_of_heads,
      batch_size * sequence_length);
}

TEST(PackedAttentionTest, NoPackWithRelativePositionBias) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> token_offset{0, 1, 2, 3};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  std::vector<float> relative_position_bias = {
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f,
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f};

  std::vector<float> output_data = {
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f,
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f};

  RunPackedAttentionTest(
      input_data,
      weight_data,
      bias_data,
      token_offset,
      cum_seq_len,
      output_data,
      batch_size,
      sequence_length,
      hidden_size,
      number_of_heads,
      batch_size * sequence_length,
      {},
      relative_position_bias);
}

TEST(PackedAttentionTest, PackedWithRelativePositionBias) {
  int batch_size = 2;
  int sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,   // b0:s0
      0.5f, 0.2f, 0.3f, -0.6f,  // b0:s1
      0.8f, -0.5f, 0.0f, 1.f,   // b1:s0
      0.5f, 0.2f, 0.3f, -0.6f   // b1:s1
  };

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> token_offset{0, 1, 4, 5, 2, 3, 6, 7};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  std::vector<float> relative_position_bias = {
      0.2f, -0.1f, 0.f, 0.f, 0.4f, 2.5f, 0.f, 0.f,
      1.6f, -1.1f, 0.f, 0.f, 0.4f, -2.5f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

      0.2f, -0.1f, 0.f, 0.f, 0.4f, 2.5f, 0.f, 0.f,
      1.6f, -1.1f, 0.f, 0.f, 0.4f, -2.5f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  std::vector<float> output_data = {
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f,
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f};

  RunPackedAttentionTest(
      input_data,
      weight_data,
      bias_data,
      token_offset,
      cum_seq_len,
      output_data,
      batch_size,
      sequence_length,
      hidden_size,
      number_of_heads,
      4,
      {},
      relative_position_bias);
}

TEST(PackedAttentionTest, PackedBatch) {
  int batch_size = 2;
  int sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,   // b0:s0
      0.5f, 0.2f, 0.3f, -0.6f,  // b0:s1
      0.8f, -0.5f, 0.0f, 1.f,   // b1:s0
      0.5f, 0.2f, 0.3f, -0.6f   // b1:s1
  };

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> token_offset{0, 1, 4, 5, 2, 3, 6, 7};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunPackedAttentionTest(
      input_data,
      weight_data,
      bias_data,
      token_offset,
      cum_seq_len,
      output_data,
      batch_size,
      sequence_length,
      hidden_size,
      number_of_heads,
      4);
}

TEST(PackedAttentionTest, PackedBatchWithQKV) {
  int batch_size = 2;
  int sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,

      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<int32_t> qkv_sizes = {6, 6, 4};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f,

      0.3f, 0.2f, 4.0f, 2.2f, 2.4f, 3.3f, 2.1f, 4.2f, 0.5f, 0.1f, 0.4f, 1.6f,
      0.4f, 0.8f, 0.9f, 0.1f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f,
      0.5f, 0.7f, 0.2f, 1.2f};

  std::vector<int32_t> token_offset{0, 1, 4, 5, 2, 3, 6, 7};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  std::vector<float> output_data = {
      3.1967618465423584f, 0.51903456449508667f, 0.63051539659500122f, 2.9394614696502686f,
      0.65332180261611938f, 1.000949501991272f, 0.74175024032592773f, 2.8231701850891113f,

      3.1967618465423584f, 0.51903456449508667f, 0.63051539659500122f, 2.9394614696502686f,
      0.65332180261611938f, 1.000949501991272f, 0.74175024032592773f, 2.8231701850891113f};

  RunPackedAttentionTest(
      input_data,
      weight_data,
      bias_data,
      token_offset,
      cum_seq_len,
      output_data,
      batch_size,
      sequence_length,
      hidden_size,
      number_of_heads,
      4,
      qkv_sizes);
}

static void RunModelWithRandomInput(
    int64_t batch_size,
    int64_t sequence_length,
    std::string& onnx_model,
    bool is_float16,
    bool has_rbp = false) {
  RandomValueGenerator random{234};

  constexpr int hidden_size = 768;
  constexpr int num_heads = 12;

  int64_t token_count = 0;
  std::vector<int32_t> cum_seq_len(batch_size + 1);
  cum_seq_len[0] = 0;

  int64_t original_offset = 0;
  int64_t token_offset_idx = 0;
  std::vector<int32_t> token_offset(batch_size * sequence_length);
  for (int64_t b = 0; b < batch_size; b++) {
    int64_t actual_seq_len = (sequence_length / (b + 1));
    token_count += actual_seq_len;
    cum_seq_len[b + 1] = narrow<int32_t>(token_count);

    original_offset = b * sequence_length;
    for (int64_t s = 0; s < actual_seq_len; s++) {
      token_offset[token_offset_idx++] = narrow<int32_t>(original_offset++);
    }
  }

  for (int64_t b = 0; b < batch_size; b++) {
    int64_t actual_seq_len = (sequence_length / (b + 1));
    original_offset = b * sequence_length + actual_seq_len;
    for (int64_t s = actual_seq_len; s < sequence_length; s++) {
      token_offset[token_offset_idx++] = narrow<int32_t>(original_offset++);
    }
  }

  assert(token_offset_idx == batch_size * sequence_length);

  std::vector<int64_t> input_dims{token_count, hidden_size};
  std::vector<float> input_data = random.Gaussian<float>(input_dims, 0.0f, 0.3f);

  std::vector<int64_t> weight_dims{hidden_size, 3 * hidden_size};
  std::vector<float> weight_data = random.Gaussian<float>(weight_dims, 0.0f, 0.3f);

  std::vector<int64_t> bias_dims{3 * hidden_size};
  std::vector<float> bias_data = random.Gaussian<float>(bias_dims, 0.0f, 0.1f);

  std::vector<int64_t> token_offset_dims{batch_size, sequence_length};
  std::vector<int64_t> cum_seq_len_dims{batch_size + 1};

  // TF32 in SM >= 80 is enabled by default, need larger threshold for float when TF32 is enabled.
  float gpu_threshold = is_float16 ? 0.15f : (HasCudaEnvironment(800) ? 0.05f : 0.005f);
  gpu_threshold *= sequence_length > 1024 ? 4.0f : 1.0f;  // threshold should increase with sequence length
  bool enable_cuda = HasCudaEnvironment(is_float16 ? 530 : 0);
  if (enable_cuda) {
    OpTester test("PackedAttention", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    if (is_float16) {
      test.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      test.AddInput<MLFloat16>("weight", weight_dims, ToFloat16(weight_data));
      test.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    } else {
      test.AddInput<float>("input", input_dims, input_data);
      test.AddInput<float>("weight", weight_dims, weight_data);
      test.AddInput<float>("bias", bias_dims, bias_data);
    }
    test.AddInput<int32_t>("token_offset", token_offset_dims, token_offset);
    test.AddInput<int32_t>("cumulative_sequence_length", cum_seq_len_dims, cum_seq_len);

    if (has_rbp) {
      std::vector<int64_t> rbp_dims{1, num_heads, sequence_length, sequence_length};
      std::vector<float> rbp_data = random.Gaussian<float>(rbp_dims, 0.0f, 0.1f);
      if (is_float16) {
        test.AddInput<MLFloat16>("rbp", rbp_dims, ToFloat16(rbp_data));
      } else {
        test.AddInput<float>("rbp", rbp_dims, rbp_data);
      }
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.AddReferenceOutputs(onnx_model, gpu_threshold, DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(PackedAttentionTest, TestWithRandomData) {
  std::string onnx_model = "testdata/packed_attention_fp32.onnx";
  std::string onnx_model_fp16 = "testdata/packed_attention_fp16.onnx";
  for (int batch_size : std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8})) {
    for (int sequence_length : std::vector<int>({32, 48, 64, 95, 128})) {
      RunModelWithRandomInput(
          batch_size,
          sequence_length,
          onnx_model,
          false);
      RunModelWithRandomInput(
          batch_size,
          sequence_length,
          onnx_model_fp16,
          true);
    }
  }
}

TEST(PackedAttentionTest, TestWithRandomDataWithRBP) {
  std::string onnx_model_fp16 = "testdata/packed_attention_fp16.rbp.onnx";  // mainly for cutlass
  for (int batch_size : std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8})) {
    for (int sequence_length : std::vector<int>({32, 48, 64, 95, 128})) {
      RunModelWithRandomInput(
          batch_size,
          sequence_length,
          onnx_model_fp16,
          true /*is_float16*/,
          true /*has_rbp*/);
    }
  }
}

TEST(PackedAttentionTest, TestWithRandomDataLargeSeq) {
  int batch_size = 2;
  int sequence_length = 1152;  // > 1024
  std::string onnx_model = "testdata/packed_attention_fp32.onnx";
  std::string onnx_model_fp16 = "testdata/packed_attention_fp16.onnx";
  RunModelWithRandomInput(
      batch_size,
      sequence_length,
      onnx_model,
      false);
  RunModelWithRandomInput(
      batch_size,
      sequence_length,
      onnx_model_fp16,
      true);
}

}  // namespace test
}  // namespace onnxruntime
