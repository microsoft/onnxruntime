// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

void RunSparseAttentionInvalidInputTest(const std::vector<int32_t>& total_key_lengths_data,
                                        const std::vector<int64_t>& total_key_lengths_dims,
                                        const std::string& expected_error,
                                        int32_t total_sequence_length = 4) {
  OpTester test("SparseAttention", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 2);
  test.AddAttribute<int64_t>("kv_num_heads", 2);
  test.AddAttribute<int64_t>("sparse_block_size", 1);
  test.AddAttribute<float>("scale", 1.0f);
  test.AddAttribute<int64_t>("do_rotary", 0);
  test.AddAttribute<int64_t>("rotary_interleaved", 0);

  test.AddInput<float>("query", {1, 1, 16}, std::vector<float>(16, 0.0f));
  test.AddInput<float>("key", {1, 1, 16}, std::vector<float>(16, 0.0f));
  test.AddInput<float>("value", {1, 1, 16}, std::vector<float>(16, 0.0f));
  test.AddInput<float>("past_key", {1, 2, 4, 8}, std::vector<float>(64, 0.0f));
  test.AddInput<float>("past_value", {1, 2, 4, 8}, std::vector<float>(64, 0.0f));
  test.AddInput<int32_t>("block_row_indices", {1, 5}, {0, 1, 2, 3, 4});
  test.AddInput<int32_t>("block_col_indices", {1, 1}, {0});
  test.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length});
  test.AddInput<int32_t>("key_total_sequence_lengths", total_key_lengths_dims, total_key_lengths_data);
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();

  test.AddOutput<float>("output", {1, 1, 16}, std::vector<float>(16, 0.0f));
  test.AddOutput<float>("present_key", {1, 2, 4, 8}, std::vector<float>(64, 0.0f));
  test.AddOutput<float>("present_value", {1, 2, 4, 8}, std::vector<float>(64, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, expected_error, {}, nullptr, &execution_providers);
}

void RunSparseAttentionPromptInputTest(const std::vector<int32_t>& total_key_lengths_data,
                                       int64_t batch_size,
                                       int64_t sequence_length,
                                       int32_t total_sequence_length) {
  OpTester test("SparseAttention", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 2);
  test.AddAttribute<int64_t>("kv_num_heads", 2);
  test.AddAttribute<int64_t>("sparse_block_size", 1);
  test.AddAttribute<float>("scale", 1.0f);
  test.AddAttribute<int64_t>("do_rotary", 0);
  test.AddAttribute<int64_t>("rotary_interleaved", 0);

  const int64_t hidden_size = 16;
  const int64_t kv_num_heads = 2;
  const int64_t head_size = hidden_size / kv_num_heads;
  const int64_t max_cache_sequence_length = total_sequence_length;

  test.AddInput<float>("query", {batch_size, sequence_length, hidden_size},
                       std::vector<float>(static_cast<size_t>(batch_size * sequence_length * hidden_size), 0.0f));
  test.AddInput<float>("key", {batch_size, sequence_length, hidden_size},
                       std::vector<float>(static_cast<size_t>(batch_size * sequence_length * hidden_size), 0.0f));
  test.AddInput<float>("value", {batch_size, sequence_length, hidden_size},
                       std::vector<float>(static_cast<size_t>(batch_size * sequence_length * hidden_size), 0.0f));
  test.AddInput<float>("past_key", {batch_size, kv_num_heads, max_cache_sequence_length, head_size},
                       std::vector<float>(static_cast<size_t>(batch_size * kv_num_heads * max_cache_sequence_length * head_size), 0.0f));
  test.AddInput<float>("past_value", {batch_size, kv_num_heads, max_cache_sequence_length, head_size},
                       std::vector<float>(static_cast<size_t>(batch_size * kv_num_heads * max_cache_sequence_length * head_size), 0.0f));
  test.AddInput<int32_t>("block_row_indices", {1, 6}, {0, 1, 3, 6, 10, 15});
  test.AddInput<int32_t>("block_col_indices", {1, 15}, {0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4});
  test.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length});
  test.AddInput<int32_t>("key_total_sequence_lengths", {batch_size}, total_key_lengths_data);
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();

  test.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                        std::vector<float>(static_cast<size_t>(batch_size * sequence_length * hidden_size), 0.0f));
  test.AddOutput<float>("present_key", {batch_size, kv_num_heads, max_cache_sequence_length, head_size},
                        std::vector<float>(static_cast<size_t>(batch_size * kv_num_heads * max_cache_sequence_length * head_size), 0.0f));
  test.AddOutput<float>("present_value", {batch_size, kv_num_heads, max_cache_sequence_length, head_size},
                        std::vector<float>(static_cast<size_t>(batch_size * kv_num_heads * max_cache_sequence_length * head_size), 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

TEST(SparseAttentionTest, RejectsOutOfRangeKeyTotalSequenceLengths) {
  RunSparseAttentionInvalidInputTest({-5}, {1}, "key_total_sequence_lengths value -5 at batch index 0 is out of range [1, 4]");
}

TEST(SparseAttentionTest, RejectsKeyTotalSequenceLengthsShapeMismatch) {
  RunSparseAttentionInvalidInputTest({4, 4}, {2}, "key_total_sequence_lengths must have shape (batch_size)");
}

TEST(SparseAttentionTest, RejectsPromptKeyTotalSequenceLengthsShorterThanSequenceLength) {
  RunSparseAttentionInvalidInputTest({0}, {1},
                                     "key_total_sequence_lengths value 0 at batch index 0 is out of range [1, 1]",
                                     1);
}

TEST(SparseAttentionTest, AcceptsPromptKeyTotalSequenceLengthsForPaddedBatch) {
  RunSparseAttentionPromptInputTest({5, 2}, 2, 5, 5);
}

}  // namespace test
}  // namespace onnxruntime
