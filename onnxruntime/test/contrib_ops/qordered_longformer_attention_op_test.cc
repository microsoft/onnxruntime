// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/contrib_ops/qordered_test_utils.h"

namespace onnxruntime {
namespace test {

#ifdef USE_CUDA

void run_qordered_longformer_attention_op_test(
    const unsigned int seed,
    const int64_t batch_size = 1,
    const int64_t sequence_len = 128,
    const int64_t num_heads = 2,
    const int64_t head_size = 16,
    const int64_t window = 32,
    int64_t input_hidden_size = 0
) {
  const int64_t hidden_size = num_heads * head_size;
  if (!input_hidden_size) input_hidden_size = hidden_size;

  RandomValueGenerator random_gen{seed};
  auto inputq = GenData<int8_t>({batch_size, sequence_len, hidden_size}, 1.0f, &random_gen);
  // clear some tail
  for (int64_t b = 0; b < batch_size; b++) {
    std::fill_n(&inputq[(b + 1) * sequence_len - 8], 8, 0);
  }
  float scale_input = 1.0f / 32.0f;
  auto weightq = GenData<int8_t>({3 * hidden_size, input_hidden_size}, 1.0f, &random_gen);
  float scale_weight = 1.0f / 64.0f;
  auto bias = GenData<float>({3 * hidden_size}, 1.0 / 128.0f / 16.0f, &random_gen);
  float scale_bias = 1.0f / 8.0f;
  float scale_qkv_gemm = 1.0f / 4.0f;
  auto attention_mask = std::vector<MLFloat16>(batch_size * sequence_len, MLFloat16(1.0f));
  // clear some tail
  for (int64_t b = 0; b < batch_size; b++) {
    std::fill_n(&attention_mask[(b + 1) * sequence_len - 8], 8, MLFloat16(0.0f));
  }
  auto global_weightq = GenData<int8_t>({3 * hidden_size, input_hidden_size}, 1.0f, &random_gen);
  float scale_global_weight = 1.0f / 64.0f;
  auto global_bias = GenData<float>({3 * hidden_size}, 1.0 / 128.0f / 16.0f, &random_gen);
  float scale_global_gemm = 1.0f / 4.0f;
  auto global_attention_mask = std::vector<int32_t>(batch_size * sequence_len, 0);
  // set each batch first 16 token as global attention token
  for (int b = 0; b < batch_size; b++) {
    std::fill_n(&global_attention_mask[b * sequence_len], 16, 1);
  }
  float scale_output = 1.0f / 16.0f;

  auto outputq = GenData<int8_t>({batch_size, sequence_len, hidden_size}, 1.0f, &random_gen);

  OpTester test_qorder("QOrderedLongformerAttention", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("num_heads", (int64_t)num_heads);
  test_qorder.AddAttribute("window", (int64_t)window);
  test_qorder.AddAttribute("order_input", (int64_t)ORDER_ROW);
  test_qorder.AddAttribute("order_output", (int64_t)ORDER_ROW);
  test_qorder.AddAttribute("order_weight", (int64_t)ORDER_COL);
  test_qorder.AddAttribute("order_global_weight", (int64_t)ORDER_COL);

  test_qorder.AddInput<int8_t>("input", {batch_size, sequence_len, input_hidden_size}, inputq);
  test_qorder.AddInput<float>("scale_input", {}, {scale_input}, true);
  test_qorder.AddInput<int8_t>("weight", {input_hidden_size, 3 * hidden_size}, weightq, true); // COL major
  test_qorder.AddInput<float>("scale_weight", {}, {scale_weight}, true);
  test_qorder.AddInput<float>("bias", {3 * hidden_size}, bias, true);
  test_qorder.AddInput<float>("scale_bias", {}, {scale_bias}, true);
  test_qorder.AddInput<float>("scale_qkv_gemm", {}, {scale_qkv_gemm}, true);
  test_qorder.AddInput<MLFloat16>("mask", {batch_size, sequence_len}, attention_mask);
  test_qorder.AddInput<int8_t>("global_weight", {input_hidden_size, 3 * hidden_size}, global_weightq, true); // COL major
  test_qorder.AddInput<float>("scale_global_weight", {}, {scale_global_weight}, true);
  test_qorder.AddInput<float>("global_bias", {3 * hidden_size}, global_bias, true);
  test_qorder.AddInput<float>("scale_global_gemm", {}, {scale_global_gemm}, true);
  test_qorder.AddInput<int32_t>("global", {batch_size, sequence_len}, global_attention_mask);
  test_qorder.AddInput<float>("scale_output", {}, {scale_output}, true);
  test_qorder.AddOutput<int8_t>("output", {batch_size, sequence_len, hidden_size}, outputq);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}


TEST(QOrderedTest, LongformerAttention_1x128x2x16_window_32) {
  run_qordered_longformer_attention_op_test(666, 1, 128, 2, 16, 32);
}

#endif

}  // namespace test
}  // namespace onnxruntime
