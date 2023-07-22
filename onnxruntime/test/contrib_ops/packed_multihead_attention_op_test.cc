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

#define InvokePackedMultiHeadAttentionTest(use_float16, use_scale) \
  RunPackedMultiHeadAttentionTest(                                 \
      query_data,                                                  \
      key_data,                                                    \
      value_data,                                                  \
      token_offset,                                                \
      cumulative_sequence_length,                                  \
      output_data,                                                 \
      batch_size,                                                  \
      sequence_length,                                             \
      hidden_size,                                                 \
      v_hidden_size,                                               \
      number_of_heads,                                             \
      token_count,                                                 \
      use_float16,                                                 \
      use_scale,                                                   \
      relative_position_bias_data);

static void RunPackedMultiHeadAttentionTest(
    const std::vector<float>& query_data,                    // query:      [token_count, num_heads, 3, head_size]
                                                             //          or [token_count, hidden_size]
    const std::vector<float>& key_data,                      // key:        [token_count, hidden_size]
    const std::vector<float>& value_data,                    // value:      [token_count, v_hidden_size]
    const std::vector<int32_t>& token_offset,                // token_offset: [batch_size, sequence_length]
    const std::vector<int32_t>& cumulative_sequence_length,  // cum_seq_len: [batch_size + 1]
    const std::vector<float>& output_data,                   // output:     [token_count, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int v_hidden_size,
    int number_of_heads,
    int token_count,
    bool use_float16,
    bool use_scale,
    const std::vector<float>& relative_position_bias_data) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);

  if (enable_cuda) {
    OpTester tester("PackedMultiHeadAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
    if (use_scale) {
      tester.AddAttribute<float>("scale", static_cast<float>(1.f / sqrt(head_size)));
    }

    int64_t head_size = static_cast<int64_t>(hidden_size / number_of_heads);
    std::vector<int64_t> packed_qkv_dims = {token_count, number_of_heads, 3, head_size};
    std::vector<int64_t> query_dims = {token_count, hidden_size};
    std::vector<int64_t> key_dims = {token_count, hidden_size};
    std::vector<int64_t> value_dims = {token_count, hidden_size};
    std::vector<int64_t> token_offset_dims = {batch_size, sequence_length};
    std::vector<int64_t> cum_seq_len_dims = {batch_size + 1};
    std::vector<int64_t> relative_position_bias_data_dims = {batch_size, number_of_heads, sequence_length, sequence_length};
    std::vector<int64_t> output_dims = {token_count, v_hidden_size};

    is_packed_qkv = (key_data.size() == 0 && value_data.size() == 0);  // packed QKV format

    if (use_float16) {
      if (is_packed_qkv) {
        tester.AddInput<MLFloat16>("query", packed_qkv_dims, ToFloat16(query_data));
        tester.AddOptionalInputEdge<MLFloat16>();
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddInput<MLFloat16>("query", query_data, ToFloat16(query_data));
        tester.AddInput<MLFloat16>("key", key_dims, ToFloat16(key_data));
        tester.AddInput<MLFloat16>("value", value_dims, ToFloat16(value_data));
      }

      tester.AddInput<int32_t>("token_offset", token_offset_dims, token_offset);
      tester.AddInput<int32_t>("cumulative_sequence_length", cum_seq_len_dims, cumulative_sequence_length);
      if (relative_position_bias_data.size() > 0) {
        tester.AddInput<MLFloat16>("relative_position_bias", relative_position_bias_data_dims, ToFloat16(relative_position_bias_data));
      }

      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      if (is_packed_qkv) {
        tester.AddInput<float>("query", packed_qkv_dims, query_data);
        tester.AddOptionalInputEdge<float>();
        tester.AddOptionalInputEdge<float>();
      } else {
        tester.AddInput<float>("query", query_data, query_data);
        tester.AddInput<float>("key", key_dims, key_data);
        tester.AddInput<float>("value", value_dims, value_data);
      }

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

static void RunPackedMultiHeadAttentionTest(
    const std::vector<float>& query_data,                    // query:      [token_count, num_heads, 3, head_size]
                                                             //          or [token_count, hidden_size]
    const std::vector<float>& key_data,                      // key:        [token_count, hidden_size]
    const std::vector<float>& value_data,                    // value:      [token_count, v_hidden_size]
    const std::vector<int32_t>& token_offset,                // token_offset: [batch_size, sequence_length]
    const std::vector<int32_t>& cumulative_sequence_length,  // cum_seq_len: [batch_size + 1]
    const std::vector<float>& output_data,                   // output:     [token_count, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int v_hidden_size,
    int number_of_heads,
    int token_count,
    const std::vector<float>& relative_position_bias_data = {}) {
  InvokePackedMultiHeadAttentionTest(true, true);
  InvokePackedMultiHeadAttentionTest(true, false);

  // packed qkv format can only run in float16 (due to limitation of TRT attention kernel)
  if (key_data.size() > 0 && value_data.size() > 0) {
    InvokePackedMultiHeadAttentionTest(false, true);
    InvokePackedMultiHeadAttentionTest(false, false);
  }
}

TEST(PackedMultiHeadAttentionTest, PackedQKV_NoPadding) {
  AttentionTestData data;
  GetSelfAttentionData_Batch2_HeadSize32_NoBias_NoMask_PackedQKV(data);
  std::vector<float> empty_data = {};
  std::vector<int32_t> token_offset{0, 1, 2, 3};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  RunPackedMultiHeadAttentionTest(
      data.qkv_data,
      empty_data,
      empty_data,
      token_offset,
      cum_seq_len,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      data.batch_size * data.sequence_length);
}

TEST(PackedMultiHeadAttentionTest, Q_K_V_Padding) {
  AttentionTestData data;
  GetPackedMultiHeadAttentionData_Batch2_HeadSize8_Mask(data);
  std::vector<float> empty_data = {};
  int token_count = 3;
  std::vector<int32_t> token_offset{0, 2, 3, 1};
  std::vector<int32_t> cum_seq_len{0, 1, 3};
  RunPackedMultiHeadAttentionTest(
      data.query_data,
      data.key_data,
      data.value_data,
      token_offset,
      cum_seq_len,
      data.fp16_output_data,
      data.batch_size,
      data.sequence_length,
      data.hidden_size,
      data.v_hidden_size,
      data.num_heads,
      token_count);
}

}  // namespace test
}  // namespace onnxruntime
