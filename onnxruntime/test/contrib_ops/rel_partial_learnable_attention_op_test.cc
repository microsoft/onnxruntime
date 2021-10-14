// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunRelPartialLearnableAttentionTest(
    const std::vector<float>& input_data,            // input: [batch_size, sequence_length, d_model]
    const std::vector<float>& input_weights_data,    // input_weights: [d_model, 3 * num_heads * head_size]
    const std::vector<float>& pos_emb_data,          // pos_emb: [batch_size, sequence_length, d_model]
    const std::vector<float>& pos_emb_weights_data,  // pos_emb_weights: [d_model, num_heads * head_size]
    const std::vector<float>& r_w_bias_data,         // r_w_bias: [num_heads, head_size]
    const std::vector<float>& r_r_bias_data,         // r_r_bias: [num_heads, head_size]
    const std::vector<float>& output_weights_data,   // output_weights: [num_heads * head_size, d_model]
    const std::vector<int32_t>& attn_mask_data,      // attn_mask: [sequence_length, sequence_length] or empty
    const std::vector<float>& mems_data,             // mems: [batch_size, sequence_length + memory_length, d_model] or empty
    const std::vector<float>& output_data,           // output: [batch_size, sequence_length, d_model]
    int batch_size,
    int sequence_length,
    int d_model,
    int number_of_heads,
    int head_size,
    bool use_float16 = false,
    int memory_length = 0,
    int input_d_model = 0,
    bool only_enable_cuda = false,
    bool only_enable_cpu = false) {
  input_d_model = (input_d_model == 0 ? d_model : input_d_model);  // By default, no pruning.

  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !only_enable_cpu;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !only_enable_cuda;

  if (enable_cpu || enable_cuda) {
    OpTester tester("RelPartialLearnableAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
    tester.AddAttribute<int64_t>("head_size", static_cast<int64_t>(head_size));
    tester.AddAttribute<int64_t>("d_model", static_cast<int64_t>(d_model));

    std::vector<int64_t> input_dims = {batch_size, sequence_length, input_d_model};
    std::vector<int64_t> input_weights_dims = {input_d_model, 3 * number_of_heads * head_size};
    std::vector<int64_t> pos_emb_dims = {batch_size, sequence_length, input_d_model};
    std::vector<int64_t> pos_emb_weights_dims = {input_d_model, number_of_heads * head_size};
    std::vector<int64_t> r_w_bias_dims = {number_of_heads, head_size};
    std::vector<int64_t> r_r_bias_dims = {number_of_heads, head_size};
    std::vector<int64_t> output_weights_dims = {number_of_heads * head_size, input_d_model};
    std::vector<int64_t> attn_mask_dims = {sequence_length, sequence_length};
    std::vector<int64_t> mems_dims = {batch_size, sequence_length + memory_length, d_model};

    std::vector<int64_t> output_dims = {batch_size, sequence_length, d_model};

    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      tester.AddInput<MLFloat16>("input_weights", input_weights_dims, ToFloat16(input_weights_data));
      tester.AddInput<MLFloat16>("pos_emb", pos_emb_dims, ToFloat16(pos_emb_data));
      tester.AddInput<MLFloat16>("pos_emb_weights", pos_emb_weights_dims, ToFloat16(pos_emb_weights_data));
      tester.AddInput<MLFloat16>("r_w_bias", r_w_bias_dims, ToFloat16(r_w_bias_data));
      tester.AddInput<MLFloat16>("r_r_bias", r_r_bias_dims, ToFloat16(r_r_bias_data));
      tester.AddInput<MLFloat16>("output_weights", output_weights_dims, ToFloat16(output_weights_data));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      tester.AddInput<float>("input_weights", input_weights_dims, input_weights_data);
      tester.AddInput<float>("pos_emb", pos_emb_dims, pos_emb_data);
      tester.AddInput<float>("pos_emb_weights", pos_emb_weights_dims, pos_emb_weights_data);
      tester.AddInput<float>("r_w_bias", r_w_bias_dims, r_w_bias_data);
      tester.AddInput<float>("r_r_bias", r_r_bias_dims, r_r_bias_data);
      tester.AddInput<float>("output_weights", output_weights_dims, output_weights_data);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    if (attn_mask_data.size() > 0) {  // attention mask is optional.
      tester.AddInput<int32_t>("attn_mask", attn_mask_dims, attn_mask_data);
    } else {
      tester.AddOptionalInputEdge<int32_t>();
    }

    if (mems_data.size() > 0) {  // memories are optional.
      if (use_float16) {
        tester.AddInput<MLFloat16>("mems", mems_dims, ToFloat16(mems_data));
      } else {
        tester.AddInput<float>("mems", mems_dims, mems_data);
      }
    } else {
      if (use_float16) {
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddOptionalInputEdge<float>();
      }
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

TEST(RelPartialLearnableAttentionTest, RelPartialLearnableAttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int d_model = 4;
  int number_of_heads = 2;
  int head_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> input_weights_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> pos_emb_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> pos_emb_weights_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> r_w_bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<float> r_r_bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<float> output_weights_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<int32_t> attn_mask_data = {};

  std::vector<float> mems_data = {};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunRelPartialLearnableAttentionTest(input_data, input_weights_data, pos_emb_data, pos_emb_weights_data,
                                     r_w_bias_data, r_r_bias_data, output_weights_data, attn_mask_data,
                                     mems_data, output_data, batch_size, sequence_length, d_model,
                                     number_of_heads, head_size);
}

}  // namespace test
}  // namespace onnxruntime
