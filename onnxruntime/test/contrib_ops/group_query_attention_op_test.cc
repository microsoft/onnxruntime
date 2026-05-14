// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
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

// Helper to compare two output vectors (non-zero check + element-wise tolerance).
static void ExpectOutputsMatch(const std::vector<float>& a, const std::vector<float>& b,
                               float tolerance, const char* label) {
  ASSERT_EQ(a.size(), b.size()) << label << ": output size mismatch";
  bool all_zero = true;
  for (size_t i = 0; i < a.size(); i++) {
    EXPECT_NEAR(a[i], b[i], tolerance) << label << " mismatch at index " << i;
    if (a[i] != 0.0f) all_zero = false;
  }
  EXPECT_FALSE(all_zero) << label << " output should not be all zeros";
}

// ---------------------------------------------------------------------------
// Tests for kv_sequence_length=0 with borrowed past_key/past_value
// (Gemma4 shared KV pattern: empty K/V inputs, all KV data in past buffer)
// ---------------------------------------------------------------------------

// Helper: run GQA with empty K/V and past_key/past_value (shared KV pattern).
// Returns the attention output.
static std::vector<float> RunGQASharedKV(
    int batch_size,
    int q_seq_len,
    int past_seq_len,
    const std::vector<float>& query_data,
    const std::vector<float>& past_key_data,
    const std::vector<float>& past_value_data,
    int num_heads,
    int kv_num_heads,
    int head_size,
    bool use_cuda = false) {
  const int hidden_size = num_heads * head_size;
  const int total_seq_len = past_seq_len;  // all KV data is in past

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  // Q: [batch, q_seq_len, hidden_size]
  tester.AddInput<float>("query", {batch_size, q_seq_len, hidden_size}, query_data);
  // K/V: empty [batch, 0, kv_hidden_size] — kv_sequence_length = 0
  const int kv_hidden_size = kv_num_heads * head_size;
  tester.AddInput<float>("key", {batch_size, 0, kv_hidden_size}, {});
  tester.AddInput<float>("value", {batch_size, 0, kv_hidden_size}, {});

  // past_key/past_value: [batch, kv_num_heads, past_seq_len, head_size] BNSH
  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_seq_len, head_size}, past_key_data);
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_seq_len, head_size}, past_value_data);

  std::vector<int32_t> seqlens_k_data(batch_size, static_cast<int32_t>(total_seq_len - 1));
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k_data);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {static_cast<int32_t>(total_seq_len)});

  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  const int output_size = batch_size * q_seq_len * hidden_size;
  tester.AddOutput<float>("output", {batch_size, q_seq_len, hidden_size},
                          std::vector<float>(output_size, 0.0f));

  // present_key/value: required when past is provided
  const int present_size = batch_size * kv_num_heads * past_seq_len * head_size;
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(present_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(present_size, 0.0f));

  tester.SetOutputTolerance(1e6f);  // We compare fetched outputs ourselves

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (use_cuda) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  } else {
    execution_providers.push_back(DefaultCpuExecutionProvider());
  }
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  auto fetches = tester.GetFetches();
  const float* out_data = fetches[0].Get<Tensor>().Data<float>();
  return std::vector<float>(out_data, out_data + output_size);
}

// Helper: run GQA with MLFloat16 tensors for actual CUDA kernel coverage.
// The CUDA GQA kernel only registers for MLFloat16/BFloat16, so float inputs
// fall back to CPU. This helper converts float inputs to fp16.
static std::vector<float> RunGQASharedKVFp16(
    int batch_size,
    int q_seq_len,
    int past_seq_len,
    const std::vector<float>& query_data,
    const std::vector<float>& past_key_data,
    const std::vector<float>& past_value_data,
    int num_heads,
    int kv_num_heads,
    int head_size) {
  const int hidden_size = num_heads * head_size;
  const int total_seq_len = past_seq_len;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  tester.AddInput<MLFloat16>("query", {batch_size, q_seq_len, hidden_size}, ToFloat16(query_data));
  const int kv_hidden_size = kv_num_heads * head_size;
  tester.AddInput<MLFloat16>("key", {batch_size, 0, kv_hidden_size}, {});
  tester.AddInput<MLFloat16>("value", {batch_size, 0, kv_hidden_size}, {});

  tester.AddInput<MLFloat16>("past_key", {batch_size, kv_num_heads, past_seq_len, head_size}, ToFloat16(past_key_data));
  tester.AddInput<MLFloat16>("past_value", {batch_size, kv_num_heads, past_seq_len, head_size}, ToFloat16(past_value_data));

  std::vector<int32_t> seqlens_k_data(batch_size, static_cast<int32_t>(total_seq_len - 1));
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k_data);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {static_cast<int32_t>(total_seq_len)});

  tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
  tester.AddOptionalInputEdge<int64_t>();    // position_ids
  tester.AddOptionalInputEdge<MLFloat16>();  // attention_bias
  tester.AddOptionalInputEdge<MLFloat16>();  // head_sink

  const int output_size = batch_size * q_seq_len * hidden_size;
  tester.AddOutput<MLFloat16>("output", {batch_size, q_seq_len, hidden_size},
                              std::vector<MLFloat16>(output_size, MLFloat16(0.0f)));

  const int present_size = batch_size * kv_num_heads * past_seq_len * head_size;
  tester.AddOutput<MLFloat16>("present_key", {batch_size, kv_num_heads, past_seq_len, head_size},
                              std::vector<MLFloat16>(present_size, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("present_value", {batch_size, kv_num_heads, past_seq_len, head_size},
                              std::vector<MLFloat16>(present_size, MLFloat16(0.0f)));

  tester.SetOutputTolerance(1e6f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  auto fetches = tester.GetFetches();
  // Convert fp16 output back to float for comparison
  const MLFloat16* out_fp16 = fetches[0].Get<Tensor>().Data<MLFloat16>();
  std::vector<float> result(output_size);
  for (int i = 0; i < output_size; i++) {
    result[i] = out_fp16[i].ToFloat();
  }
  return result;
}

// CPU: kv_sequence_length=0 with past_key/past_value (shared KV decode).
// Validates output is non-zero (attention over past KV produces valid output).
// Note: cannot compare against RunGQAAndGetOutput because the two paths have
// different causal masking semantics (past_seqlen differs).
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_CPU) {
  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int past_seq_len = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  // Verify non-zero and no NaN
  bool all_zero = true;
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
    if (output[i] != 0.0f) all_zero = false;
  }
  EXPECT_FALSE(all_zero) << "Output should not be all zeros";
}

// CPU: kv_sequence_length=0 with past, prompt phase (q_seq_len == total_seq_len).
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_Prompt_CPU) {
  constexpr int batch_size = 1;
  constexpr int q_seq_len = 8;
  constexpr int past_seq_len = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  bool all_zero = true;
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
    if (output[i] != 0.0f) all_zero = false;
  }
  EXPECT_FALSE(all_zero) << "Output should not be all zeros";
}

// CUDA: kv_sequence_length=0 with past, decode (q_seq=1).
// Cross-checks CUDA against CPU for correctness.
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int past_seq_len = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto cuda_output = RunGQASharedKVFp16(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size);

  auto cpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  ExpectOutputsMatch(cuda_output, cpu_output, 0.05f, "SharedKV_CUDA_vs_CPU");
}

// CPU: kv_sequence_length=0 with past, head_size=64.
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_LargeHeadSize_CPU) {
  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int past_seq_len = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 64;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 11 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 5 + 1);

  auto output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  bool all_zero = true;
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
    if (output[i] != 0.0f) all_zero = false;
  }
  EXPECT_FALSE(all_zero) << "Output should not be all zeros";
}

// CPU: GQA ratio num_heads=8, kv_num_heads=1 (matches Gemma4 config).
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_GQARatio8_CPU) {
  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int past_seq_len = 4;
  constexpr int num_heads = 8;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 13 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 5 + 1);

  auto output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  bool all_zero = true;
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
    if (output[i] != 0.0f) all_zero = false;
  }
  EXPECT_FALSE(all_zero) << "Output should not be all zeros";
}

// CPU: shared KV with batch_size > 1.
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_Batched_CPU) {
  constexpr int batch_size = 2;
  constexpr int q_seq_len = 1;
  constexpr int past_seq_len = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  bool all_zero = true;
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
    if (output[i] != 0.0f) all_zero = false;
  }
  EXPECT_FALSE(all_zero) << "Output should not be all zeros";
}

// Reject: kv_sequence_length=0 without past_key (shared KV requires past).
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_NoPast_Rejected) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size},
                         std::vector<float>(batch_size * sequence_length * hidden_size, 1.0f));
  // K/V: empty [B, 0, kv_hidden] — kv_sequence_length = 0
  tester.AddInput<float>("key", {batch_size, 0, kv_hidden_size}, {});
  tester.AddInput<float>("value", {batch_size, 0, kv_hidden_size}, {});
  // No past_key/past_value
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();

  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {static_cast<int32_t>(sequence_length - 1)});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {static_cast<int32_t>(sequence_length)});
  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(batch_size * sequence_length * hidden_size, 0.0f));
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, sequence_length, head_size},
                          std::vector<float>(batch_size * kv_num_heads * sequence_length * head_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, sequence_length, head_size},
                          std::vector<float>(batch_size * kv_num_heads * sequence_length * head_size, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "query and key must have the same sequence length",
             {}, nullptr, &execution_providers);
}

// CUDA: kv_sequence_length=0 with past, prompt phase. Cross-checks against CPU.
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_Prompt_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int q_seq_len = 8;
  constexpr int past_seq_len = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto cuda_output = RunGQASharedKVFp16(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size);
  auto cpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  ExpectOutputsMatch(cuda_output, cpu_output, 0.05f, "SharedKV_Prompt_CUDA_vs_CPU");
}

// CUDA: kv_sequence_length=0 with past, head_size=16 (different from default 8).
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_LargeHeadSize_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int past_seq_len = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 11 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 5 + 1);

  auto cuda_output = RunGQASharedKVFp16(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size);
  auto cpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  ExpectOutputsMatch(cuda_output, cpu_output, 0.05f, "SharedKV_LargeHead_CUDA_vs_CPU");
}

// CUDA: kv_sequence_length=0 with past, GQA ratio 8:1. Cross-checks against CPU.
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_GQARatio8_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int past_seq_len = 4;
  constexpr int num_heads = 8;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 13 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 5 + 1);

  auto cuda_output = RunGQASharedKVFp16(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size);
  auto cpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  ExpectOutputsMatch(cuda_output, cpu_output, 0.15f, "SharedKV_GQA8_CUDA_vs_CPU");
}

// ---------------------------------------------------------------------------
// Shared KV tests with do_rotary=1 (Gemma4 primary use case)
// ---------------------------------------------------------------------------

// Helper: run GQA with empty K/V, past_key/past_value, and do_rotary=1.
// Generates cos/sin caches and position_ids internally.
static std::vector<float> RunGQASharedKVWithRotary(
    int batch_size,
    int q_seq_len,
    int past_seq_len,
    const std::vector<float>& query_data,
    const std::vector<float>& past_key_data,
    const std::vector<float>& past_value_data,
    int num_heads,
    int kv_num_heads,
    int head_size,
    bool use_cuda = false) {
  const int hidden_size = num_heads * head_size;
  const int total_seq_len = past_seq_len;
  const int rotary_dim = head_size;           // full rotary
  const int max_seq_len = past_seq_len + 16;  // cos/sin cache length

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));

  // Q: [batch, q_seq_len, hidden_size]
  tester.AddInput<float>("query", {batch_size, q_seq_len, hidden_size}, query_data);
  // K/V: empty [batch, 0, kv_hidden_size]
  const int kv_hidden_size = kv_num_heads * head_size;
  tester.AddInput<float>("key", {batch_size, 0, kv_hidden_size}, {});
  tester.AddInput<float>("value", {batch_size, 0, kv_hidden_size}, {});

  // past_key/past_value: [batch, kv_num_heads, past_seq_len, head_size] BNSH
  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_seq_len, head_size}, past_key_data);
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_seq_len, head_size}, past_value_data);

  std::vector<int32_t> seqlens_k_data(batch_size, static_cast<int32_t>(total_seq_len - 1));
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k_data);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {static_cast<int32_t>(total_seq_len)});

  // cos_cache/sin_cache: [max_seq_len, rotary_dim / 2]
  const int half_rotary = rotary_dim / 2;
  std::vector<float> cos_cache(max_seq_len * half_rotary);
  std::vector<float> sin_cache(max_seq_len * half_rotary);
  for (int pos = 0; pos < max_seq_len; pos++) {
    for (int d = 0; d < half_rotary; d++) {
      float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(d) / static_cast<float>(rotary_dim));
      cos_cache[pos * half_rotary + d] = std::cos(static_cast<float>(pos) * freq);
      sin_cache[pos * half_rotary + d] = std::sin(static_cast<float>(pos) * freq);
    }
  }
  tester.AddInput<float>("cos_cache", {max_seq_len, half_rotary}, cos_cache);
  tester.AddInput<float>("sin_cache", {max_seq_len, half_rotary}, sin_cache);

  // position_ids: [batch, q_seq_len] — positions for the Q tokens
  std::vector<int64_t> position_ids(batch_size * q_seq_len);
  for (int b = 0; b < batch_size; b++) {
    int past_len = total_seq_len - q_seq_len;
    for (int s = 0; s < q_seq_len; s++) {
      position_ids[b * q_seq_len + s] = static_cast<int64_t>(past_len + s);
    }
  }
  tester.AddInput<int64_t>("position_ids", {batch_size, q_seq_len}, position_ids);

  tester.AddOptionalInputEdge<float>();  // attention_bias
  tester.AddOptionalInputEdge<float>();  // head_sink

  const int output_size = batch_size * q_seq_len * hidden_size;
  tester.AddOutput<float>("output", {batch_size, q_seq_len, hidden_size},
                          std::vector<float>(output_size, 0.0f));

  const int present_size = batch_size * kv_num_heads * past_seq_len * head_size;
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(present_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(present_size, 0.0f));

  tester.SetOutputTolerance(1e6f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (use_cuda) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  } else {
    execution_providers.push_back(DefaultCpuExecutionProvider());
  }
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  auto fetches = tester.GetFetches();
  const float* out_data = fetches[0].Get<Tensor>().Data<float>();
  return std::vector<float>(out_data, out_data + output_size);
}

// Helper: run GQA with MLFloat16 tensors + do_rotary=1 for actual CUDA kernel coverage.
static std::vector<float> RunGQASharedKVWithRotaryFp16(
    int batch_size,
    int q_seq_len,
    int past_seq_len,
    const std::vector<float>& query_data,
    const std::vector<float>& past_key_data,
    const std::vector<float>& past_value_data,
    int num_heads,
    int kv_num_heads,
    int head_size) {
  const int hidden_size = num_heads * head_size;
  const int total_seq_len = past_seq_len;
  const int rotary_dim = head_size;
  const int max_seq_len = past_seq_len + 16;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));

  tester.AddInput<MLFloat16>("query", {batch_size, q_seq_len, hidden_size}, ToFloat16(query_data));
  const int kv_hidden_size = kv_num_heads * head_size;
  tester.AddInput<MLFloat16>("key", {batch_size, 0, kv_hidden_size}, {});
  tester.AddInput<MLFloat16>("value", {batch_size, 0, kv_hidden_size}, {});

  tester.AddInput<MLFloat16>("past_key", {batch_size, kv_num_heads, past_seq_len, head_size}, ToFloat16(past_key_data));
  tester.AddInput<MLFloat16>("past_value", {batch_size, kv_num_heads, past_seq_len, head_size}, ToFloat16(past_value_data));

  std::vector<int32_t> seqlens_k_data(batch_size, static_cast<int32_t>(total_seq_len - 1));
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k_data);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {static_cast<int32_t>(total_seq_len)});

  const int half_rotary = rotary_dim / 2;
  std::vector<float> cos_cache(max_seq_len * half_rotary);
  std::vector<float> sin_cache(max_seq_len * half_rotary);
  for (int pos = 0; pos < max_seq_len; pos++) {
    for (int d = 0; d < half_rotary; d++) {
      float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(d) / static_cast<float>(rotary_dim));
      cos_cache[pos * half_rotary + d] = std::cos(static_cast<float>(pos) * freq);
      sin_cache[pos * half_rotary + d] = std::sin(static_cast<float>(pos) * freq);
    }
  }
  tester.AddInput<MLFloat16>("cos_cache", {max_seq_len, half_rotary}, ToFloat16(cos_cache));
  tester.AddInput<MLFloat16>("sin_cache", {max_seq_len, half_rotary}, ToFloat16(sin_cache));

  std::vector<int64_t> position_ids(batch_size * q_seq_len);
  for (int b = 0; b < batch_size; b++) {
    int past_len = total_seq_len - q_seq_len;
    for (int s = 0; s < q_seq_len; s++) {
      position_ids[b * q_seq_len + s] = static_cast<int64_t>(past_len + s);
    }
  }
  tester.AddInput<int64_t>("position_ids", {batch_size, q_seq_len}, position_ids);

  tester.AddOptionalInputEdge<MLFloat16>();  // attention_bias
  tester.AddOptionalInputEdge<MLFloat16>();  // head_sink

  const int output_size = batch_size * q_seq_len * hidden_size;
  tester.AddOutput<MLFloat16>("output", {batch_size, q_seq_len, hidden_size},
                              std::vector<MLFloat16>(output_size, MLFloat16(0.0f)));

  const int present_size = batch_size * kv_num_heads * past_seq_len * head_size;
  tester.AddOutput<MLFloat16>("present_key", {batch_size, kv_num_heads, past_seq_len, head_size},
                              std::vector<MLFloat16>(present_size, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("present_value", {batch_size, kv_num_heads, past_seq_len, head_size},
                              std::vector<MLFloat16>(present_size, MLFloat16(0.0f)));

  tester.SetOutputTolerance(1e6f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  auto fetches = tester.GetFetches();
  const MLFloat16* out_fp16 = fetches[0].Get<Tensor>().Data<MLFloat16>();
  std::vector<float> result(output_size);
  for (int i = 0; i < output_size; i++) {
    result[i] = out_fp16[i].ToFloat();
  }
  return result;
}

// CPU: shared KV with do_rotary=1 (Q-only RoPE path).
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_Rotary_CPU) {
  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int past_seq_len = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;  // must be multiple of 16 for rotary
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  // Output with rotary should differ from without rotary (RoPE changes Q projections)
  auto output_no_rotary = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  bool all_zero = true;
  bool differs_from_no_rotary = false;
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
    if (output[i] != 0.0f) all_zero = false;
    if (std::abs(output[i] - output_no_rotary[i]) > 1e-6f) differs_from_no_rotary = true;
  }
  EXPECT_FALSE(all_zero) << "Output should not be all zeros";
  EXPECT_TRUE(differs_from_no_rotary) << "Rotary output should differ from non-rotary output";
}

// CUDA: shared KV with do_rotary=1, cross-checked against CPU.
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_Rotary_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int past_seq_len = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto cuda_output = RunGQASharedKVWithRotaryFp16(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size);
  auto cpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  ExpectOutputsMatch(cuda_output, cpu_output, 0.05f, "SharedKV_Rotary_CUDA_vs_CPU");
}

// CUDA: shared KV + rotary, prompt phase (q_seq_len > 1).
TEST(GroupQueryAttentionTest, SharedKV_EmptyKV_WithPast_Rotary_Prompt_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int q_seq_len = 4;
  constexpr int past_seq_len = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto cuda_output = RunGQASharedKVWithRotaryFp16(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size);
  auto cpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false);

  ExpectOutputsMatch(cuda_output, cpu_output, 0.05f, "SharedKV_Rotary_Prompt_CUDA_vs_CPU");
}

}  // namespace test
}  // namespace onnxruntime
