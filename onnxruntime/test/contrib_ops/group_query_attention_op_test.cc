// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

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
    bool use_cuda = false,
    bool use_webgpu = false) {
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
  } else if (use_webgpu) {
    execution_providers.push_back(DefaultWebGpuExecutionProvider());
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
    bool use_cuda = false,
    bool use_webgpu = false) {
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
  } else if (use_webgpu) {
    execution_providers.push_back(DefaultWebGpuExecutionProvider());
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

// ---------------------------------------------------------------------------
// Quantized KV cache tests for CPU GroupQueryAttention
// ---------------------------------------------------------------------------

namespace {

// INT4 pack: two signed 4-bit values packed into one uint8_t.
// val_even goes into low nibble (+8 biased), val_odd goes into high nibble (+8 biased).
inline uint8_t PackInt4(int8_t even, int8_t odd) {
  return static_cast<uint8_t>(((even + 8) & 0x0F) | (((odd + 8) & 0x0F) << 4));
}

// INT4 unpack: returns (even, odd) from packed uint8_t.
inline std::pair<int8_t, int8_t> UnpackInt4(uint8_t packed) {
  int8_t even = static_cast<int8_t>((packed & 0x0F)) - 8;
  int8_t odd = static_cast<int8_t>((packed >> 4)) - 8;
  return {even, odd};
}

// Reference FP32 GQA: compute attention output = softmax(Q*K^T / sqrt(d)) * V
// This is a simplified reference for prompt-only (no past), single batch.
// Q: [B, num_heads, seq_len, head_size]
// K: [B, kv_num_heads, kv_seqlen, head_size]
// V: [B, kv_num_heads, kv_seqlen, head_size]
// output: [B, seq_len, num_heads * head_size]
void ReferenceGQA(const float* Q, const float* K, const float* V, float* output,
                  int batch_size, int num_heads, int kv_num_heads, int seq_len,
                  int kv_seqlen, int head_size, bool causal) {
  const int groups = num_heads / kv_num_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

  for (int b = 0; b < batch_size; b++) {
    for (int h = 0; h < num_heads; h++) {
      int kv_h = h / groups;
      for (int q_s = 0; q_s < seq_len; q_s++) {
        // Compute QK^T for this query position
        std::vector<float> logits(kv_seqlen, 0.0f);
        for (int k_s = 0; k_s < kv_seqlen; k_s++) {
          float dot = 0.0f;
          for (int d = 0; d < head_size; d++) {
            float q_val = Q[((b * num_heads + h) * seq_len + q_s) * head_size + d];
            float k_val = K[((b * kv_num_heads + kv_h) * kv_seqlen + k_s) * head_size + d];
            dot += q_val * k_val;
          }
          logits[k_s] = dot * scale;
        }
        // Apply causal mask
        if (causal) {
          for (int k_s = q_s + 1; k_s < kv_seqlen; k_s++) {
            logits[k_s] = -std::numeric_limits<float>::infinity();
          }
        }
        // Softmax
        float max_val = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        for (int k_s = 0; k_s < kv_seqlen; k_s++) {
          logits[k_s] = std::exp(logits[k_s] - max_val);
          sum_exp += logits[k_s];
        }
        for (int k_s = 0; k_s < kv_seqlen; k_s++) {
          logits[k_s] /= sum_exp;
        }
        // Compute output = attn * V
        for (int d = 0; d < head_size; d++) {
          float acc = 0.0f;
          for (int k_s = 0; k_s < kv_seqlen; k_s++) {
            float v_val = V[((b * kv_num_heads + kv_h) * kv_seqlen + k_s) * head_size + d];
            acc += logits[k_s] * v_val;
          }
          output[(b * seq_len + q_s) * num_heads * head_size + h * head_size + d] = acc;
        }
      }
    }
  }
}

// Run quantized GQA through OpTester: prompt phase with quantized KV cache.
// Compares quantized output against dequantized-reference GQA.
struct QuantGQAConfig {
  int batch_size;
  int seq_len;
  int num_heads;
  int kv_num_heads;
  int head_size;
  std::string quant_type;  // "PER_TENSOR" or "PER_CHANNEL"
  int bit_width;           // 4 or 8
};

void RunQuantizedGQAPromptTest(const QuantGQAConfig& cfg) {
  const int hidden_size = cfg.num_heads * cfg.head_size;
  const int kv_hidden_size = cfg.kv_num_heads * cfg.head_size;
  const int kv_elements = cfg.batch_size * cfg.kv_num_heads * cfg.seq_len * cfg.head_size;

  // Generate random input data with small magnitude to keep quantization error manageable
  std::default_random_engine gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> query_data(cfg.batch_size * cfg.seq_len * hidden_size);
  std::vector<float> key_data(cfg.batch_size * cfg.seq_len * kv_hidden_size);
  std::vector<float> value_data(cfg.batch_size * cfg.seq_len * kv_hidden_size);
  for (auto& v : query_data) v = dist(gen);
  for (auto& v : key_data) v = dist(gen);
  for (auto& v : value_data) v = dist(gen);

  // Create K/V in BNSH layout for quantization (matching what the op will produce in present_k/v)
  // The op will write the new K/V into present_k/v cache during prompt.
  // For testing, we need to provide empty past_k/v and let the op write to present.
  // Actually, for prompt-only, we provide K/V as inputs and past_k/v as empty.
  // The output (present_k/v) will be quantized.

  // Build OpTester
  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(cfg.num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(cfg.kv_num_heads));
  tester.AddAttribute<std::string>("k_quant_type", cfg.quant_type);
  tester.AddAttribute<std::string>("v_quant_type", cfg.quant_type);
  tester.AddAttribute<int64_t>("kv_cache_bit_width", static_cast<int64_t>(cfg.bit_width));

  tester.AddInput<float>("query", {cfg.batch_size, cfg.seq_len, hidden_size}, query_data);
  tester.AddInput<float>("key", {cfg.batch_size, cfg.seq_len, kv_hidden_size}, key_data);
  tester.AddInput<float>("value", {cfg.batch_size, cfg.seq_len, kv_hidden_size}, value_data);

  // Past cache: zero-filled buffer (prompt phase, share_buffer mode).
  // Must be provided so the type constraint T_CACHE can be resolved.
  const int packed_head_size = (cfg.bit_width == 4) ? ((cfg.head_size + 1) / 2) : cfg.head_size;
  const int past_elements = cfg.batch_size * cfg.kv_num_heads * cfg.seq_len * packed_head_size;

  if (cfg.bit_width == 4) {
    tester.AddInput<uint8_t>("past_key",
                             {cfg.batch_size, cfg.kv_num_heads, static_cast<int64_t>(cfg.seq_len), packed_head_size},
                             std::vector<uint8_t>(past_elements, 0));
    tester.AddInput<uint8_t>("past_value",
                             {cfg.batch_size, cfg.kv_num_heads, static_cast<int64_t>(cfg.seq_len), packed_head_size},
                             std::vector<uint8_t>(past_elements, 0));
  } else {
    tester.AddInput<int8_t>("past_key",
                            {cfg.batch_size, cfg.kv_num_heads, static_cast<int64_t>(cfg.seq_len), packed_head_size},
                            std::vector<int8_t>(past_elements, 0));
    tester.AddInput<int8_t>("past_value",
                            {cfg.batch_size, cfg.kv_num_heads, static_cast<int64_t>(cfg.seq_len), packed_head_size},
                            std::vector<int8_t>(past_elements, 0));
  }

  std::vector<int32_t> seqlens_k_data(cfg.batch_size, static_cast<int32_t>(cfg.seq_len - 1));
  tester.AddInput<int32_t>("seqlens_k", {cfg.batch_size}, seqlens_k_data);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {static_cast<int32_t>(cfg.seq_len)});

  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  // Scale inputs: one scale per (kv_head, d) channel for PER_CHANNEL; one scalar for PER_TENSOR.
  const int scale_size = (cfg.quant_type == "PER_CHANNEL")
                             ? (cfg.kv_num_heads * cfg.head_size)
                             : 1;
  float qmax = (cfg.bit_width == 4) ? 7.0f : 127.0f;

  std::vector<float> k_scale_data(scale_size);
  std::vector<float> v_scale_data(scale_size);
  // Keep a single per-tensor scalar in scope (used by the verifier for PER_TENSOR).
  float k_scale_val = 1.0f, v_scale_val = 1.0f;
  if (cfg.quant_type == "PER_CHANNEL") {
    // Compute true per-channel scales: distinct per (kv_head, d) pair.
    // key_data layout: [batch_size, seq_len, kv_num_heads * head_size] (BSNH).
    for (int ch = 0; ch < scale_size; ++ch) {
      float k_ch_max = 0.0f, v_ch_max = 0.0f;
      for (int b = 0; b < cfg.batch_size; ++b) {
        for (int s = 0; s < cfg.seq_len; ++s) {
          int idx = (b * cfg.seq_len + s) * kv_hidden_size + ch;
          k_ch_max = std::max(k_ch_max, std::fabs(key_data[idx]));
          v_ch_max = std::max(v_ch_max, std::fabs(value_data[idx]));
        }
      }
      k_scale_data[ch] = (k_ch_max > 1e-6f) ? (k_ch_max / qmax) : 1.0f;
      v_scale_data[ch] = (v_ch_max > 1e-6f) ? (v_ch_max / qmax) : 1.0f;
    }
  } else {
    float k_amax = 0.0f, v_amax = 0.0f;
    for (auto& v : key_data) k_amax = std::max(k_amax, std::fabs(v));
    for (auto& v : value_data) v_amax = std::max(v_amax, std::fabs(v));
    k_scale_val = (k_amax > 1e-6f) ? (k_amax / qmax) : 1.0f;
    v_scale_val = (v_amax > 1e-6f) ? (v_amax / qmax) : 1.0f;
    k_scale_data[0] = k_scale_val;
    v_scale_data[0] = v_scale_val;
  }

  std::vector<int64_t> scale_shape;
  if (cfg.quant_type == "PER_CHANNEL") {
    scale_shape = {static_cast<int64_t>(cfg.kv_num_heads * cfg.head_size)};
  } else {
    scale_shape = {1};
  }
  tester.AddInput<float>("k_scale", scale_shape, k_scale_data);
  tester.AddInput<float>("v_scale", scale_shape, v_scale_data);

  // Outputs - we use loose tolerance and verify non-zero
  const int output_size = cfg.batch_size * cfg.seq_len * hidden_size;
  tester.AddOutput<float>("output", {cfg.batch_size, cfg.seq_len, hidden_size},
                          std::vector<float>(output_size, 0.0f));

  const int present_size = cfg.batch_size * cfg.kv_num_heads * cfg.seq_len * packed_head_size;

  if (cfg.bit_width == 4) {
    tester.AddOutput<uint8_t>("present_key",
                              {cfg.batch_size, cfg.kv_num_heads, static_cast<int64_t>(cfg.seq_len), packed_head_size},
                              std::vector<uint8_t>(present_size, 0));
    tester.AddOutput<uint8_t>("present_value",
                              {cfg.batch_size, cfg.kv_num_heads, static_cast<int64_t>(cfg.seq_len), packed_head_size},
                              std::vector<uint8_t>(present_size, 0));
  } else {
    tester.AddOutput<int8_t>("present_key",
                             {cfg.batch_size, cfg.kv_num_heads, static_cast<int64_t>(cfg.seq_len), packed_head_size},
                             std::vector<int8_t>(present_size, 0));
    tester.AddOutput<int8_t>("present_value",
                             {cfg.batch_size, cfg.kv_num_heads, static_cast<int64_t>(cfg.seq_len), packed_head_size},
                             std::vector<int8_t>(present_size, 0));
  }

  tester.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches,
                                     const std::string& /*provider_type*/) {
    ASSERT_GE(fetches.size(), size_t{1});
    const float* out_data = fetches[0].Get<Tensor>().Data<float>();

    // Verify output is non-zero and no NaN
    bool all_zero = true;
    for (int i = 0; i < output_size; i++) {
      EXPECT_FALSE(std::isnan(out_data[i]))
          << "NaN in output at index " << i
          << " (quant=" << cfg.quant_type << " bit=" << cfg.bit_width << ")";
      if (out_data[i] != 0.0f) all_zero = false;
    }
    EXPECT_FALSE(all_zero)
        << "Output should not be all zeros (quant=" << cfg.quant_type << " bit=" << cfg.bit_width << ")";

    // --- Cross-check against FP32 reference with dequantized KV ---
    // Reshape K/V from [B, kv_seq, kv_hidden] to [B, kv_num_heads, kv_seq, head_size] (BNSH)
    std::vector<float> K_bnsh(kv_elements), V_bnsh(kv_elements);
    for (int b = 0; b < cfg.batch_size; b++) {
      for (int s = 0; s < cfg.seq_len; s++) {
        for (int n = 0; n < cfg.kv_num_heads; n++) {
          for (int d = 0; d < cfg.head_size; d++) {
            int bnsh_idx = ((b * cfg.kv_num_heads + n) * cfg.seq_len + s) * cfg.head_size + d;
            int bsnh_idx = (b * cfg.seq_len + s) * kv_hidden_size + n * cfg.head_size + d;
            K_bnsh[bnsh_idx] = key_data[bsnh_idx];
            V_bnsh[bnsh_idx] = value_data[bsnh_idx];
          }
        }
      }
    }

    // Helper: returns the channel scale for element at flat BNSH index.
    // K/V BNSH layout: [B, kv_num_heads, seq_len, head_size]; channel = kv_head * head_size + d.
    auto get_k_scale = [&](size_t flat) -> float {
      if (cfg.quant_type != "PER_CHANNEL") return k_scale_val;
      int d = static_cast<int>(flat) % cfg.head_size;
      int kv_head = (static_cast<int>(flat) / (cfg.seq_len * cfg.head_size)) % cfg.kv_num_heads;
      return k_scale_data[kv_head * cfg.head_size + d];
    };
    auto get_v_scale = [&](size_t flat) -> float {
      if (cfg.quant_type != "PER_CHANNEL") return v_scale_val;
      int d = static_cast<int>(flat) % cfg.head_size;
      int kv_head = (static_cast<int>(flat) / (cfg.seq_len * cfg.head_size)) % cfg.kv_num_heads;
      return v_scale_data[kv_head * cfg.head_size + d];
    };

    // Simulate quantization noise using the same per-channel (or per-tensor) scales.
    std::vector<float> K_deq(kv_elements), V_deq(kv_elements);
    if (cfg.bit_width == 8) {
      std::vector<int8_t> k_q(kv_elements), v_q(kv_elements);
      for (size_t i = 0; i < K_bnsh.size(); i++) {
        k_q[i] = static_cast<int8_t>(std::round(std::clamp(K_bnsh[i] / get_k_scale(i), -128.0f, 127.0f)));
      }
      for (size_t i = 0; i < V_bnsh.size(); i++) {
        v_q[i] = static_cast<int8_t>(std::round(std::clamp(V_bnsh[i] / get_v_scale(i), -128.0f, 127.0f)));
      }
      for (size_t i = 0; i < K_deq.size(); i++) K_deq[i] = k_q[i] * get_k_scale(i);
      for (size_t i = 0; i < V_deq.size(); i++) V_deq[i] = v_q[i] * get_v_scale(i);
    } else {
      std::vector<uint8_t> k_p((K_bnsh.size() + 1) / 2), v_p((V_bnsh.size() + 1) / 2);
      for (size_t i = 0; i < K_bnsh.size(); i += 2) {
        int8_t q0 = static_cast<int8_t>(std::round(std::clamp(K_bnsh[i] / get_k_scale(i), -8.0f, 7.0f)));
        int8_t q1 = static_cast<int8_t>(std::round(std::clamp(K_bnsh[i + 1] / get_k_scale(i + 1), -8.0f, 7.0f)));
        k_p[i / 2] = PackInt4(q0, q1);
      }
      for (size_t i = 0; i < V_bnsh.size(); i += 2) {
        int8_t q0 = static_cast<int8_t>(std::round(std::clamp(V_bnsh[i] / get_v_scale(i), -8.0f, 7.0f)));
        int8_t q1 = static_cast<int8_t>(std::round(std::clamp(V_bnsh[i + 1] / get_v_scale(i + 1), -8.0f, 7.0f)));
        v_p[i / 2] = PackInt4(q0, q1);
      }
      // Dequantize using per-element scale (unpack nibbles back).
      for (size_t i = 0; i < K_bnsh.size(); i += 2) {
        auto [q0, q1] = UnpackInt4(k_p[i / 2]);
        K_deq[i] = q0 * get_k_scale(i);
        K_deq[i + 1] = q1 * get_k_scale(i + 1);
      }
      for (size_t i = 0; i < V_bnsh.size(); i += 2) {
        auto [q0, q1] = UnpackInt4(v_p[i / 2]);
        V_deq[i] = q0 * get_v_scale(i);
        V_deq[i + 1] = q1 * get_v_scale(i + 1);
      }
    }

    // Reshape Q to BNSH
    std::vector<float> Q_bnsh(cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.head_size);
    for (int b = 0; b < cfg.batch_size; b++) {
      for (int s = 0; s < cfg.seq_len; s++) {
        for (int n = 0; n < cfg.num_heads; n++) {
          for (int d = 0; d < cfg.head_size; d++) {
            int bnsh_idx = ((b * cfg.num_heads + n) * cfg.seq_len + s) * cfg.head_size + d;
            int bsnh_idx = (b * cfg.seq_len + s) * hidden_size + n * cfg.head_size + d;
            Q_bnsh[bnsh_idx] = query_data[bsnh_idx];
          }
        }
      }
    }

    // Compute reference output
    std::vector<float> ref_output(output_size, 0.0f);
    ReferenceGQA(Q_bnsh.data(), K_deq.data(), V_deq.data(), ref_output.data(),
                 cfg.batch_size, cfg.num_heads, cfg.kv_num_heads, cfg.seq_len,
                 cfg.seq_len, cfg.head_size, /*causal=*/true);

    // Compare with tolerance
    float atol = (cfg.bit_width == 4) ? 0.15f : 0.05f;
    for (int i = 0; i < output_size; i++) {
      float diff = std::fabs(out_data[i] - ref_output[i]);
      EXPECT_LE(diff, atol)
          << "Quantized vs reference mismatch at index " << i
          << ", got=" << out_data[i] << " ref=" << ref_output[i]
          << " (quant=" << cfg.quant_type << " bit=" << cfg.bit_width << ")";
    }
  });

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // anonymous namespace

// INT8 per-tensor prompt test
TEST(GroupQueryAttentionTest, QuantizedKV_INT8_PerTensor_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/8, /*quant_type=*/"PER_TENSOR", /*bit_width=*/8});
}

// INT8 per-channel prompt test
TEST(GroupQueryAttentionTest, QuantizedKV_INT8_PerChannel_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/8, /*quant_type=*/"PER_CHANNEL", /*bit_width=*/8});
}

// INT4 per-tensor prompt test
TEST(GroupQueryAttentionTest, QuantizedKV_INT4_PerTensor_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/8, /*quant_type=*/"PER_TENSOR", /*bit_width=*/4});
}

// INT4 per-channel prompt test
TEST(GroupQueryAttentionTest, QuantizedKV_INT4_PerChannel_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/8, /*quant_type=*/"PER_CHANNEL", /*bit_width=*/4});
}

// Multi-batch INT8 prompt test
TEST(GroupQueryAttentionTest, QuantizedKV_INT8_MultiBatch_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/2, /*seq_len=*/4, /*num_heads=*/4, /*kv_num_heads=*/2,
                             /*head_size=*/16, /*quant_type=*/"PER_TENSOR", /*bit_width=*/8});
}

// Larger head_size INT8 prompt test
TEST(GroupQueryAttentionTest, QuantizedKV_INT8_LargeHead_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/8, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/64, /*quant_type=*/"PER_TENSOR", /*bit_width=*/8});
}

// GQA ratio 4:1 INT4 prompt test
TEST(GroupQueryAttentionTest, QuantizedKV_INT4_GQARatio4_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/4, /*kv_num_heads=*/1,
                             /*head_size=*/16, /*quant_type=*/"PER_TENSOR", /*bit_width=*/4});
}

// Error: MLFloat16 Q with quantized KV should fail at model construction or runtime.
TEST(GroupQueryAttentionTest, QuantizedKV_RejectMLFloat16) {
  constexpr int batch_size = 1;
  constexpr int seq_len = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<std::string>("k_quant_type", "PER_TENSOR");
  tester.AddAttribute<std::string>("v_quant_type", "PER_TENSOR");
  tester.AddAttribute<int64_t>("kv_cache_bit_width", 8);

  std::vector<MLFloat16> query_data(batch_size * seq_len * hidden_size, MLFloat16(0.1f));
  tester.AddInput<MLFloat16>("query", {batch_size, seq_len, hidden_size}, query_data);
  std::vector<MLFloat16> key_data(batch_size * seq_len * kv_hidden_size, MLFloat16(0.1f));
  tester.AddInput<MLFloat16>("key", {batch_size, seq_len, kv_hidden_size}, key_data);
  std::vector<MLFloat16> value_data(batch_size * seq_len * kv_hidden_size, MLFloat16(0.1f));
  tester.AddInput<MLFloat16>("value", {batch_size, seq_len, kv_hidden_size}, value_data);

  // Past with matching T_CACHE type (int8)
  tester.AddInput<int8_t>("past_key", {batch_size, kv_num_heads, seq_len, head_size},
                          std::vector<int8_t>(batch_size * kv_num_heads * seq_len * head_size, 0));
  tester.AddInput<int8_t>("past_value", {batch_size, kv_num_heads, seq_len, head_size},
                          std::vector<int8_t>(batch_size * kv_num_heads * seq_len * head_size, 0));
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {seq_len - 1});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {seq_len});
  tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
  tester.AddOptionalInputEdge<int64_t>();    // position_ids
  tester.AddOptionalInputEdge<MLFloat16>();  // attention_bias
  tester.AddOptionalInputEdge<MLFloat16>();  // head_sink
  tester.AddInput<float>("k_scale", {1}, {0.01f});
  tester.AddInput<float>("v_scale", {1}, {0.01f});

  tester.AddOutput<MLFloat16>("output", {batch_size, seq_len, hidden_size},
                              std::vector<MLFloat16>(batch_size * seq_len * hidden_size, MLFloat16(0.0f)));
  tester.AddOutput<int8_t>("present_key", {batch_size, kv_num_heads, seq_len, head_size},
                           std::vector<int8_t>(batch_size * kv_num_heads * seq_len * head_size, 0));
  tester.AddOutput<int8_t>("present_value", {batch_size, kv_num_heads, seq_len, head_size},
                           std::vector<int8_t>(batch_size * kv_num_heads * seq_len * head_size, 0));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "only supports float Q dtype",
             {}, nullptr, &execution_providers);
}

// Error: Missing k_scale with quantized KV cache
TEST(GroupQueryAttentionTest, QuantizedKV_MissingScale) {
  constexpr int batch_size = 1;
  constexpr int seq_len = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<std::string>("k_quant_type", "PER_TENSOR");
  tester.AddAttribute<std::string>("v_quant_type", "PER_TENSOR");
  tester.AddAttribute<int64_t>("kv_cache_bit_width", 8);

  tester.AddInput<float>("query", {batch_size, seq_len, hidden_size},
                         std::vector<float>(batch_size * seq_len * hidden_size, 0.1f));
  tester.AddInput<float>("key", {batch_size, seq_len, kv_hidden_size},
                         std::vector<float>(batch_size * seq_len * kv_hidden_size, 0.1f));
  tester.AddInput<float>("value", {batch_size, seq_len, kv_hidden_size},
                         std::vector<float>(batch_size * seq_len * kv_hidden_size, 0.1f));

  tester.AddInput<int8_t>("past_key", {batch_size, kv_num_heads, seq_len, head_size},
                          std::vector<int8_t>(batch_size * kv_num_heads * seq_len * head_size, 0));
  tester.AddInput<int8_t>("past_value", {batch_size, kv_num_heads, seq_len, head_size},
                          std::vector<int8_t>(batch_size * kv_num_heads * seq_len * head_size, 0));
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {seq_len - 1});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {seq_len});
  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink
  // No k_scale or v_scale provided

  tester.AddOutput<float>("output", {batch_size, seq_len, hidden_size},
                          std::vector<float>(batch_size * seq_len * hidden_size, 0.0f));
  tester.AddOutput<int8_t>("present_key", {batch_size, kv_num_heads, seq_len, head_size},
                           std::vector<int8_t>(batch_size * kv_num_heads * seq_len * head_size, 0));
  tester.AddOutput<int8_t>("present_value", {batch_size, kv_num_heads, seq_len, head_size},
                           std::vector<int8_t>(batch_size * kv_num_heads * seq_len * head_size, 0));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "k_scale must be provided",
             {}, nullptr, &execution_providers);
}

// Regression: seqlens_k valid for KV cache but exceeding cos_cache.shape[0] must be rejected
// when do_rotary is enabled. Without this check, the position ID derived from seqlens_k
// would index out of bounds in the cos/sin cache, leaking heap memory into output.
TEST(GroupQueryAttentionTest, SeqlensKExceedsCosCache_OOB) {
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;  // must be multiple of 16 for rotary
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int rotary_half_dim = head_size / 2;  // cos/sin cache dim-1 = 8

  constexpr int cos_cache_max_seq = 4;  // small rotary cache
  constexpr int past_seq_len = 16;      // large KV cache
  constexpr int seqlens_k_val = 10;     // valid for KV (10 < 16) but OOB for cos (10 >= 4)
  constexpr int total_seq_len = 4;      // passes CheckRotaryCaches (4 <= cos_cache_max_seq)

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));

  tester.AddInput<float>("query", {1, 1, hidden_size}, std::vector<float>(hidden_size, 1.0f));
  tester.AddInput<float>("key", {1, 1, kv_hidden_size}, std::vector<float>(kv_hidden_size, 1.0f));
  tester.AddInput<float>("value", {1, 1, kv_hidden_size}, std::vector<float>(kv_hidden_size, 1.0f));

  // Past KV cache is large enough for seqlens_k=10
  tester.AddInput<float>("past_key", {1, kv_num_heads, past_seq_len, head_size},
                         std::vector<float>(kv_num_heads * past_seq_len * head_size, 0.5f));
  tester.AddInput<float>("past_value", {1, kv_num_heads, past_seq_len, head_size},
                         std::vector<float>(kv_num_heads * past_seq_len * head_size, 0.5f));

  tester.AddInput<int32_t>("seqlens_k", {1}, {seqlens_k_val});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_seq_len});

  // cos/sin cache with only 4 rows — seqlens_k=10 exceeds this
  tester.AddInput<float>("cos_cache", {cos_cache_max_seq, rotary_half_dim},
                         std::vector<float>(cos_cache_max_seq * rotary_half_dim, 1.0f));
  tester.AddInput<float>("sin_cache", {cos_cache_max_seq, rotary_half_dim},
                         std::vector<float>(cos_cache_max_seq * rotary_half_dim, 0.0f));

  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  tester.AddOutput<float>("output", {1, 1, hidden_size}, std::vector<float>(hidden_size, 0.0f));
  tester.AddOutput<float>("present_key", {1, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(kv_num_heads * past_seq_len * head_size, 0.0f));
  tester.AddOutput<float>("present_value", {1, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(kv_num_heads * past_seq_len * head_size, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure, "is out of range for rotary cache dimension 0",
             {}, nullptr, &execution_providers);
}

// Positive test: seqlens_k within cos/sin cache bounds with do_rotary enabled should succeed.
TEST(GroupQueryAttentionTest, SeqlensKWithinCosCache_Rotary) {
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;  // must be multiple of 16 for rotary
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int rotary_half_dim = head_size / 2;

  constexpr int cos_cache_max_seq = 16;  // rotary cache large enough
  constexpr int past_seq_len = 16;
  constexpr int seqlens_k_val = 3;  // valid: 3 < 16 (cos cache) and 3 < 16 (KV cache)
  constexpr int total_seq_len = 4;  // seqlens_k + 1

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));

  tester.AddInput<float>("query", {1, 1, hidden_size}, std::vector<float>(hidden_size, 1.0f));
  tester.AddInput<float>("key", {1, 1, kv_hidden_size}, std::vector<float>(kv_hidden_size, 1.0f));
  tester.AddInput<float>("value", {1, 1, kv_hidden_size}, std::vector<float>(kv_hidden_size, 1.0f));

  tester.AddInput<float>("past_key", {1, kv_num_heads, past_seq_len, head_size},
                         std::vector<float>(kv_num_heads * past_seq_len * head_size, 0.5f));
  tester.AddInput<float>("past_value", {1, kv_num_heads, past_seq_len, head_size},
                         std::vector<float>(kv_num_heads * past_seq_len * head_size, 0.5f));

  tester.AddInput<int32_t>("seqlens_k", {1}, {seqlens_k_val});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_seq_len});

  tester.AddInput<float>("cos_cache", {cos_cache_max_seq, rotary_half_dim},
                         std::vector<float>(cos_cache_max_seq * rotary_half_dim, 1.0f));
  tester.AddInput<float>("sin_cache", {cos_cache_max_seq, rotary_half_dim},
                         std::vector<float>(cos_cache_max_seq * rotary_half_dim, 0.0f));

  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  tester.AddOutput<float>("output", {1, 1, hidden_size}, std::vector<float>(hidden_size, 0.0f));
  tester.AddOutput<float>("present_key", {1, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(kv_num_heads * past_seq_len * head_size, 0.0f));
  tester.AddOutput<float>("present_value", {1, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(kv_num_heads * past_seq_len * head_size, 0.0f));

  tester.SetOutputTolerance(1e6f);  // shape acceptance test, not numerical correctness

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "",
             {}, nullptr, &execution_providers);
}

// Multi-batch test: one valid and one OOB seqlens_k value.
// Verifies the validation loop correctly identifies the offending batch index.
TEST(GroupQueryAttentionTest, SeqlensKExceedsCosCache_MultiBatch) {
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int rotary_half_dim = head_size / 2;

  constexpr int cos_cache_max_seq = 4;
  constexpr int past_seq_len = 16;
  constexpr int total_seq_len = 4;
  constexpr int batch_size = 2;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));

  tester.AddInput<float>("query", {batch_size, 1, hidden_size},
                         std::vector<float>(batch_size * hidden_size, 1.0f));
  tester.AddInput<float>("key", {batch_size, 1, kv_hidden_size},
                         std::vector<float>(batch_size * kv_hidden_size, 1.0f));
  tester.AddInput<float>("value", {batch_size, 1, kv_hidden_size},
                         std::vector<float>(batch_size * kv_hidden_size, 1.0f));

  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_seq_len, head_size},
                         std::vector<float>(batch_size * kv_num_heads * past_seq_len * head_size, 0.5f));
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_seq_len, head_size},
                         std::vector<float>(batch_size * kv_num_heads * past_seq_len * head_size, 0.5f));

  // seqlens_k: batch 0 is valid (3 < 4), batch 1 is OOB (10 >= 4)
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {3, 10});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_seq_len});

  tester.AddInput<float>("cos_cache", {cos_cache_max_seq, rotary_half_dim},
                         std::vector<float>(cos_cache_max_seq * rotary_half_dim, 1.0f));
  tester.AddInput<float>("sin_cache", {cos_cache_max_seq, rotary_half_dim},
                         std::vector<float>(cos_cache_max_seq * rotary_half_dim, 0.0f));

  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  tester.AddOutput<float>("output", {batch_size, 1, hidden_size},
                          std::vector<float>(batch_size * hidden_size, 0.0f));
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(batch_size * kv_num_heads * past_seq_len * head_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(batch_size * kv_num_heads * past_seq_len * head_size, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  // Error should reference batch index 1: seqlens_k[1] = 10
  tester.Run(OpTester::ExpectResult::kExpectFailure, "seqlens_k[1] = 10",
             {}, nullptr, &execution_providers);
}

// ---------------------------------------------------------------------------
// WebGPU: shared KV tests (Gemma4 kv_sequence_length=0 pattern)
// Each test cross-checks WebGPU against CPU for correctness.
// ---------------------------------------------------------------------------

// WebGPU: kv_sequence_length=0 with past, decode (q_seq=1).
TEST(GroupQueryAttentionTest, WebGPU_SharedKV_Decode) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
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

  auto webgpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false, /*use_webgpu=*/true);
  auto cpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false, /*use_webgpu=*/false);

  ExpectOutputsMatch(webgpu_output, cpu_output, 0.05f, "SharedKV_WebGPU_vs_CPU");
}

// WebGPU: kv_sequence_length=0 with past, prompt phase (q_seq_len > 1).
TEST(GroupQueryAttentionTest, WebGPU_SharedKV_Prefill) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
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

  auto webgpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false, /*use_webgpu=*/true);
  auto cpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false, /*use_webgpu=*/false);

  ExpectOutputsMatch(webgpu_output, cpu_output, 0.05f, "SharedKV_Prompt_WebGPU_vs_CPU");
}

// WebGPU: kv_sequence_length=0 with past and do_rotary=1 (Q-only RoPE path).
TEST(GroupQueryAttentionTest, WebGPU_SharedKV_Rotary) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
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

  auto webgpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false, /*use_webgpu=*/true);
  auto cpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false, /*use_webgpu=*/false);

  ExpectOutputsMatch(webgpu_output, cpu_output, 0.05f, "SharedKV_Rotary_WebGPU_vs_CPU");
}

// WebGPU: kv_sequence_length=0 with do_rotary=1 and q_seq_len > 1 (prefill).
// Validates position_offset + bsnh[1] arithmetic for multiple sequence positions.
TEST(GroupQueryAttentionTest, WebGPU_SharedKV_Rotary_Prefill) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int q_seq_len = 4;
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

  auto webgpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false, /*use_webgpu=*/true);
  auto cpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false, /*use_webgpu=*/false);

  ExpectOutputsMatch(webgpu_output, cpu_output, 0.05f, "SharedKV_Rotary_Prefill_WebGPU_vs_CPU");
}

// WebGPU: kv_sequence_length=0 with do_rotary=1 and batch_size > 1.
// Validates batch stride calculations in the rotary embedding path.
TEST(GroupQueryAttentionTest, WebGPU_SharedKV_Rotary_MultiBatch) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 2;
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

  auto webgpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false, /*use_webgpu=*/true);
  auto cpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, /*use_cuda=*/false, /*use_webgpu=*/false);

  ExpectOutputsMatch(webgpu_output, cpu_output, 0.05f, "SharedKV_Rotary_MultiBatch_WebGPU_vs_CPU");
}

// WebGPU: kv_sequence_length=0 with sliding window active (total_seq > local_window_size).
// Regression test: sliding window must not block flash attention for kv_empty layers.
TEST(GroupQueryAttentionTest, WebGPU_SharedKV_SlidingWindow) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int q_seq_len = 4;
  constexpr int past_seq_len = 32;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int local_window_size = 16;  // < past_seq_len to trigger sliding window
  constexpr int total_seq_len = past_seq_len;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<int64_t>("local_window_size", static_cast<int64_t>(local_window_size));

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  tester.AddInput<float>("query", {batch_size, q_seq_len, hidden_size}, query_data);
  tester.AddInput<float>("key", {batch_size, 0, kv_hidden_size}, {});
  tester.AddInput<float>("value", {batch_size, 0, kv_hidden_size}, {});
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
  const int present_size = batch_size * kv_num_heads * past_seq_len * head_size;
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(present_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, past_seq_len, head_size},
                          std::vector<float>(present_size, 0.0f));

  tester.SetOutputTolerance(1e6f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
