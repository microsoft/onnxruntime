// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

void RunSparseAttentionInvalidInputTest(const std::vector<int32_t>& total_key_lengths_data,
                                        const std::vector<int64_t>& total_key_lengths_dims,
                                        const std::string& expected_error) {
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
  test.AddInput<int32_t>("total_sequence_length", {1}, {4});
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

}  // namespace

TEST(SparseAttentionTest, RejectsOutOfRangeKeyTotalSequenceLengths) {
  RunSparseAttentionInvalidInputTest({-5}, {1}, "key_total_sequence_lengths value -5 at batch index 0 is out of range [1, 4]");
}

TEST(SparseAttentionTest, RejectsKeyTotalSequenceLengthsShapeMismatch) {
  RunSparseAttentionInvalidInputTest({4, 4}, {2}, "key_total_sequence_lengths must have shape (batch_size)");
}

}  // namespace test
}  // namespace onnxruntime
