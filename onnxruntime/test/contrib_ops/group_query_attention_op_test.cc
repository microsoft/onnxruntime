// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <optional>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// Helper to build a minimal GQA OpTester with given seqlens_k and total_seq_len.
// Uses num_heads=1, kv_num_heads=1, and head_size=8; past may be provided via
// provide_past/past_seq_len.
static void RunGQASeqlensKTest(
    const std::vector<int32_t>& seqlens_k_data,
    int32_t total_seq_len,
    int batch_size,
    int sequence_length,
    OpTester::ExpectResult expect,
    const std::string& expected_message,
    bool provide_past = false,
    int past_seq_len = 0,
    const std::optional<std::vector<int64_t>>& seqlens_k_shape = std::nullopt) {
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  std::vector<float> query_data(batch_size * sequence_length * hidden_size, 1.0f);
  tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size}, query_data);

  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size, 1.0f);
  tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size}, key_data);

  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size, 1.0f);
  tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size}, value_data);

  if (provide_past) {
    std::vector<float> past_k(batch_size * kv_num_heads * past_seq_len * head_size, 0.5f);
    std::vector<float> past_v(batch_size * kv_num_heads * past_seq_len * head_size, 0.5f);
    tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_seq_len, head_size}, past_k);
    tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_seq_len, head_size}, past_v);
  } else {
    tester.AddOptionalInputEdge<float>();  // past_key
    tester.AddOptionalInputEdge<float>();  // past_value
  }

  std::vector<int64_t> shape = seqlens_k_shape.has_value()
                                   ? *seqlens_k_shape
                                   : std::vector<int64_t>{batch_size};
  tester.AddInput<int32_t>("seqlens_k", shape, seqlens_k_data);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_seq_len});

  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  // For failure tests with invalid total_seq_len, clamp declared output shape to avoid
  // negative-sized vectors in test setup. The operator rejects these inputs before using outputs.
  int declared_present_seqlen = provide_past ? past_seq_len : std::max(1, static_cast<int>(total_seq_len));
  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(batch_size * sequence_length * hidden_size, 0.0f));
  tester.AddOutput<float>("present_key",
                          {batch_size, kv_num_heads, declared_present_seqlen, head_size},
                          std::vector<float>(batch_size * kv_num_heads * declared_present_seqlen * head_size, 0.0f));
  tester.AddOutput<float>("present_value",
                          {batch_size, kv_num_heads, declared_present_seqlen, head_size},
                          std::vector<float>(batch_size * kv_num_heads * declared_present_seqlen * head_size, 0.0f));

  // Tolerance is intentionally loose: these tests validate shape acceptance, not output values.
  if (expect == OpTester::ExpectResult::kExpectSuccess) {
    tester.SetOutputTolerance(1e6f);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(expect, expected_message, {}, nullptr, &execution_providers);
}

// Regression: negative seqlens_k wraps to huge size_t, causing GEMM OOB.
TEST(GroupQueryAttentionTest, NegativeSeqlensK_OOB) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{-5},
      /*total_seq_len=*/1,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "seqlens_k[0]");
}

// Regression: seqlens_k exceeding present KV cache buffer causes GEMM OOB.
TEST(GroupQueryAttentionTest, OversizedSeqlensK_OOB) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{100},
      /*total_seq_len=*/1,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "seqlens_k[0]");
}

// Multi-batch: one valid element, one bad — should still fail.
TEST(GroupQueryAttentionTest, MultiBatchOneBadSeqlensK_OOB) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0, -3},
      /*total_seq_len=*/1,
      /*batch_size=*/2,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "seqlens_k[1]");
}

// Boundary: seqlens_k == present_kv_seqlen - 1 is the maximum valid value.
// First prompt with seq=1, total_seq=1, present=1 → seqlens_k=0 should succeed.
TEST(GroupQueryAttentionTest, BoundaryValidSeqlensK) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0},
      /*total_seq_len=*/1,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectSuccess,
      "");
}

// Negative seqlens_k with past context: ensures the check fires even when
// past is provided (regression for the original OOB scenario with token generation).
TEST(GroupQueryAttentionTest, NegativeSeqlensKWithPast_OOB) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{-1},
      /*total_seq_len=*/2,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "seqlens_k[0]",
      /*provide_past=*/true,
      /*past_seq_len=*/1);
}

// Boundary: seqlens_k within range when present_kv_seqlen > total_sequence_length.
TEST(GroupQueryAttentionTest, BoundaryValidSeqlensKWithLargerPast) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{1},
      /*total_seq_len=*/2,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectSuccess,
      "",
      /*provide_past=*/true,
      /*past_seq_len=*/4);
}

// Non-first-prompt: seqlens_k valid for KV cache but too small for sequence_length.
// past_seqlen = total_seqlen - sequence_length underflows size_t, causing memcpy OOB.
TEST(GroupQueryAttentionTest, NonPromptSeqlensKUnderflow_OOB) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{1},
      /*total_seq_len=*/5,
      /*batch_size=*/1,
      /*sequence_length=*/3,
      OpTester::ExpectResult::kExpectFailure,
      "is too small for sequence_length",
      /*provide_past=*/true,
      /*past_seq_len=*/4);
}

// INT32_MAX seqlens_k: rejected by the >= present_kv_seqlen check.
TEST(GroupQueryAttentionTest, Int32MaxSeqlensK_OOB) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{std::numeric_limits<int32_t>::max()},
      /*total_seq_len=*/1,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "seqlens_k[0]");
}

// Boundary: seqlens_k == present_kv_seqlen - 1 is max valid value with larger present buffer.
TEST(GroupQueryAttentionTest, MaxBoundarySeqlensK) {
  // past_seq_len=4 → present_kv_seqlen=4; seqlens_k=3 → total_seqlen=4 == present_kv_seqlen
  // seqlens_k must be < present_kv_seqlen, so seqlens_k=3 is the max valid value.
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{3},
      /*total_seq_len=*/4,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectSuccess,
      "",
      /*provide_past=*/true,
      /*past_seq_len=*/4);
}

// Off-by-one: seqlens_k == present_kv_seqlen should be rejected (one past the valid range).
TEST(GroupQueryAttentionTest, OffByOneSeqlensK_OOB) {
  // past_seq_len=4 → present_kv_seqlen=4; seqlens_k=4 is out of range [0, 4).
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{4},
      /*total_seq_len=*/4,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "seqlens_k[0]",
      /*provide_past=*/true,
      /*past_seq_len=*/4);
}

// total_sequence_length <= 0 should be rejected in CheckInputs.
TEST(GroupQueryAttentionTest, TotalSeqLenZero) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0},
      /*total_seq_len=*/0,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "total_sequence_length must be positive");
}

TEST(GroupQueryAttentionTest, TotalSeqLenNegative) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0},
      /*total_seq_len=*/-5,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "total_sequence_length must be positive");
}

// Backward compat: seqlens_k shape {1, 1} accepted for batch_size=1.
// Older model builders (e.g. qwen3-0.6b) emit this instead of {1}.
TEST(GroupQueryAttentionTest, SeqlensKLegacy2DShape) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0},
      /*total_seq_len=*/1,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectSuccess,
      "",
      /*provide_past=*/false,
      /*past_seq_len=*/0,
      /*seqlens_k_shape=*/std::vector<int64_t>{1, 1});
}

// Backward compat: seqlens_k shape {2, 1} accepted for batch_size=2.
TEST(GroupQueryAttentionTest, SeqlensKLegacy2DShapeMultiBatch) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0, 0},
      /*total_seq_len=*/1,
      /*batch_size=*/2,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectSuccess,
      "",
      /*provide_past=*/false,
      /*past_seq_len=*/0,
      /*seqlens_k_shape=*/std::vector<int64_t>{2, 1});
}

// Backward compat: seqlens_k shape {1, 2} accepted for batch_size=2.
// Batch dimension in trailing position.
TEST(GroupQueryAttentionTest, SeqlensKLegacy2DShapeTrailingBatch) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0, 0},
      /*total_seq_len=*/1,
      /*batch_size=*/2,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectSuccess,
      "",
      /*provide_past=*/false,
      /*past_seq_len=*/0,
      /*seqlens_k_shape=*/std::vector<int64_t>{1, 2});
}

// Shape {2, 2} with batch_size=4: correct element count but invalid factored shape.
TEST(GroupQueryAttentionTest, SeqlensKInvalidFactoredShape) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0, 0, 0, 0},
      /*total_seq_len=*/1,
      /*batch_size=*/4,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "seqlens_k has unexpected shape",
      /*provide_past=*/false,
      /*past_seq_len=*/0,
      /*seqlens_k_shape=*/std::vector<int64_t>{2, 2});
}

// Wrong element count (1D): 2 elements for batch_size=1.
TEST(GroupQueryAttentionTest, SeqlensKWrongLength) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0, 0},
      /*total_seq_len=*/1,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "seqlens_k must have batch_size",
      /*provide_past=*/false,
      /*past_seq_len=*/0,
      /*seqlens_k_shape=*/std::vector<int64_t>{2});
}

// Wrong element count (2D): shape {2, 1} has 2 elements but batch_size=1.
TEST(GroupQueryAttentionTest, SeqlensKWrongElementCount2D) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0, 0},
      /*total_seq_len=*/1,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "seqlens_k must have batch_size",
      /*provide_past=*/false,
      /*past_seq_len=*/0,
      /*seqlens_k_shape=*/std::vector<int64_t>{2, 1});
}

// Scalar seqlens_k must be rejected even when batch_size=1.
TEST(GroupQueryAttentionTest, SeqlensKScalarRejected) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{0},
      /*total_seq_len=*/1,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "seqlens_k must be at least 1D",
      /*provide_past=*/false,
      /*past_seq_len=*/0,
      /*seqlens_k_shape=*/std::vector<int64_t>{});
}

}  // namespace test
}  // namespace onnxruntime
