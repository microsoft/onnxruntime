// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/session_options.h"
#include "core/framework/tensor.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

void RunOnWebGpu(OpTester& tester, std::unique_ptr<IExecutionProvider> webgpu_ep) {
  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  tester.Config(session_options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

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

  RunOnWebGpu(tester, std::move(webgpu_ep));
}

void RunPackedRotaryWithAsymmetricCachesNoOOB() {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;
  constexpr int past_sequence_length = 1;
  constexpr int total_sequence_length = past_sequence_length + sequence_length;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int packed_hidden_size = hidden_size + 2 * kv_hidden_size;
  constexpr int half_rotary_dimension = head_size / 2;
  constexpr int cos_cache_length = 4;
  constexpr int sin_cache_length = total_sequence_length;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", num_heads);
  tester.AddAttribute<int64_t>("kv_num_heads", kv_num_heads);
  tester.AddAttribute<int64_t>("do_rotary", 1);

  tester.AddInput<float>("query", {batch_size, sequence_length, packed_hidden_size},
                         std::vector<float>(batch_size * sequence_length * packed_hidden_size, 0.1f));
  tester.AddOptionalInputEdge<float>();  // key
  tester.AddOptionalInputEdge<float>();  // value
  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_sequence_length, head_size},
                         std::vector<float>(batch_size * kv_num_heads * past_sequence_length * head_size, 0.2f));
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_sequence_length, head_size},
                         std::vector<float>(batch_size * kv_num_heads * past_sequence_length * head_size, 0.3f));

  // The device-side position is 2: valid for cos_cache, but one past sin_cache.
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {2});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length}, /*is_initializer=*/true);
  tester.AddInput<float>("cos_cache", {cos_cache_length, half_rotary_dimension},
                         std::vector<float>(cos_cache_length * half_rotary_dimension, 1.0f));
  tester.AddInput<float>("sin_cache", {sin_cache_length, half_rotary_dimension},
                         std::vector<float>(sin_cache_length * half_rotary_dimension, 0.0f));
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(batch_size * sequence_length * hidden_size, 0.0f));
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, total_sequence_length, head_size},
                          std::vector<float>(batch_size * kv_num_heads * total_sequence_length * head_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, total_sequence_length, head_size},
                          std::vector<float>(batch_size * kv_num_heads * total_sequence_length * head_size, 0.0f));

  tester.SetOutputTolerance(1e6f);
  RunOnWebGpu(tester, std::move(webgpu_ep));
}

}  // namespace

TEST(GroupQueryAttentionTest, OversizedSeqlensK_CacheAppend_NoOOB_WebGPU) {
  RunMalformedSeqlensKNoOOB(/*seqlens_k=*/106);
}

TEST(GroupQueryAttentionTest, NegativeSeqlensK_CacheAppend_NoOOB_WebGPU) {
  RunMalformedSeqlensKNoOOB(/*seqlens_k=*/-1);
}

TEST(GroupQueryAttentionTest, PackedRotaryAsymmetricCaches_NoOOB_WebGPU) {
  RunPackedRotaryWithAsymmetricCachesNoOOB();
}

}  // namespace test
}  // namespace onnxruntime
