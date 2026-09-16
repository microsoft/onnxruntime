// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/tensor.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

void RunMalformedSeqlensKNoOOB(int32_t seqlens_k) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 2;
  constexpr int past_seq_len = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int present_seq_len = past_seq_len + sequence_length;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", num_heads);
  tester.AddAttribute<int64_t>("kv_num_heads", kv_num_heads);

  tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size},
                         std::vector<float>(batch_size * sequence_length * hidden_size, 0.1f));
  tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size},
                         std::vector<float>(batch_size * sequence_length * kv_hidden_size, 0.2f));
  tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size},
                         std::vector<float>(batch_size * sequence_length * kv_hidden_size, 0.3f));
  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_seq_len, head_size},
                         std::vector<float>(batch_size * kv_num_heads * past_seq_len * head_size, 0.4f));
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_seq_len, head_size},
                         std::vector<float>(batch_size * kv_num_heads * past_seq_len * head_size, 0.5f));
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {seqlens_k});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {present_seq_len}, /*is_initializer=*/true);

  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  constexpr int output_size = batch_size * sequence_length * hidden_size;
  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(output_size, 0.0f));
  constexpr int present_size = batch_size * kv_num_heads * present_seq_len * head_size;
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, present_seq_len, head_size},
                          std::vector<float>(present_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, present_seq_len, head_size},
                          std::vector<float>(present_size, 0.0f));

  tester.SetOutputTolerance(1e6f);
  tester.SetCustomOutputVerifier([](const std::vector<OrtValue>& fetches,
                                    const std::string& /*provider*/) {
    ASSERT_FALSE(fetches.empty());
    ASSERT_TRUE(fetches[0].IsTensor());
    EXPECT_EQ(fetches[0].Get<Tensor>().Shape().Size(), static_cast<int64_t>(output_size));
  });

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

TEST(GroupQueryAttentionTest, OversizedSeqlensK_CacheAppend_NoOOB_WebGPU) {
  RunMalformedSeqlensKNoOOB(/*seqlens_k=*/106);
}

TEST(GroupQueryAttentionTest, NegativeSeqlensK_CacheAppend_NoOOB_WebGPU) {
  RunMalformedSeqlensKNoOOB(/*seqlens_k=*/-1);
}

}  // namespace test
}  // namespace onnxruntime
