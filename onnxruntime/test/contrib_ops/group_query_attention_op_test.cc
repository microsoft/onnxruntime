// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

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
    int past_seq_len = 0) {
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

  tester.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k_data);
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

  // For success tests, we only care that validation passes without crash;
  // exact output values are not the focus of these security regression tests.
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

// Shape validation: seqlens_k with wrong rank (2D instead of 1D) must be rejected.
TEST(GroupQueryAttentionTest, SeqlensKWrongRank) {
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  tester.AddInput<float>("query", {1, 1, hidden_size}, std::vector<float>(hidden_size, 1.0f));
  tester.AddInput<float>("key", {1, 1, kv_hidden_size}, std::vector<float>(kv_hidden_size, 1.0f));
  tester.AddInput<float>("value", {1, 1, kv_hidden_size}, std::vector<float>(kv_hidden_size, 1.0f));
  tester.AddOptionalInputEdge<float>();  // past_key
  tester.AddOptionalInputEdge<float>();  // past_value
  // 2D shape {1, 1} instead of {1}
  tester.AddInput<int32_t>("seqlens_k", {1, 1}, {0});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {1});
  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  tester.AddOutput<float>("output", {1, 1, hidden_size}, std::vector<float>(hidden_size, 0.0f));
  tester.AddOutput<float>("present_key", {1, kv_num_heads, 1, head_size},
                          std::vector<float>(kv_num_heads * head_size, 0.0f));
  tester.AddOutput<float>("present_value", {1, kv_num_heads, 1, head_size},
                          std::vector<float>(kv_num_heads * head_size, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure, "seqlens_k must be shape (batch_size)",
             {}, nullptr, &execution_providers);
}

// Shape validation: seqlens_k with wrong length (2 elements for batch_size=1) must be rejected.
TEST(GroupQueryAttentionTest, SeqlensKWrongLength) {
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  tester.AddInput<float>("query", {1, 1, hidden_size}, std::vector<float>(hidden_size, 1.0f));
  tester.AddInput<float>("key", {1, 1, kv_hidden_size}, std::vector<float>(kv_hidden_size, 1.0f));
  tester.AddInput<float>("value", {1, 1, kv_hidden_size}, std::vector<float>(kv_hidden_size, 1.0f));
  tester.AddOptionalInputEdge<float>();  // past_key
  tester.AddOptionalInputEdge<float>();  // past_value
  // Length 2 instead of 1 for batch_size=1
  tester.AddInput<int32_t>("seqlens_k", {2}, {0, 0});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {1});
  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  tester.AddOutput<float>("output", {1, 1, hidden_size}, std::vector<float>(hidden_size, 0.0f));
  tester.AddOutput<float>("present_key", {1, kv_num_heads, 1, head_size},
                          std::vector<float>(kv_num_heads * head_size, 0.0f));
  tester.AddOutput<float>("present_value", {1, kv_num_heads, 1, head_size},
                          std::vector<float>(kv_num_heads * head_size, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure, "seqlens_k must be shape (batch_size)",
             {}, nullptr, &execution_providers);
}

// ============================================================================
// Optional present_key/present_value output tests
// ============================================================================

// Run GQA with the given inputs and return the output tensor as a vector.
// Supports separate Q and K/V sequence lengths for KV-shared decode scenarios.
// When use_cuda=true, runs on CUDA EP instead of CPU EP.
static std::vector<float> RunGQAAndGetOutput(
    int batch_size,
    int q_seq_len,
    int kv_seq_len,
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& value_data,
    int num_heads,
    int kv_num_heads,
    int head_size,
    bool omit_present,
    bool use_cuda = false) {
  const int hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int total_seq_len = kv_seq_len;  // no past: total = kv length

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  tester.AddInput<float>("query", {batch_size, q_seq_len, hidden_size}, query_data);
  tester.AddInput<float>("key", {batch_size, kv_seq_len, kv_hidden_size}, key_data);
  tester.AddInput<float>("value", {batch_size, kv_seq_len, kv_hidden_size}, value_data);

  tester.AddOptionalInputEdge<float>();  // past_key
  tester.AddOptionalInputEdge<float>();  // past_value

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

  if (omit_present) {
    tester.AddOptionalOutputEdge<float>();  // present_key
    tester.AddOptionalOutputEdge<float>();  // present_value
  } else {
    const int present_size = batch_size * kv_num_heads * total_seq_len * head_size;
    tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, total_seq_len, head_size},
                            std::vector<float>(present_size, 0.0f));
    tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, total_seq_len, head_size},
                            std::vector<float>(present_size, 0.0f));
  }
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

// Helper: compare two output vectors element-wise and check non-zero.
static void ExpectOutputsMatch(const std::vector<float>& a, const std::vector<float>& b,
                               float tol, const std::string& label) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); i++) {
    EXPECT_NEAR(a[i], b[i], tol) << label << " mismatch at index " << i;
  }
  bool all_zero = true;
  for (float v : a) {
    if (v != 0.0f) {
      all_zero = false;
      break;
    }
  }
  EXPECT_FALSE(all_zero) << label << " output should not be all zeros";
}

// Regression: omitting optional present outputs must not change the attention output
// compared to when present outputs are connected (first-prompt, no past KV).
TEST(GroupQueryAttentionTest, OptionalPresent_OmittingDoesNotChangeOutput) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  // Deterministic non-trivial inputs
  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < key_data.size(); i++) key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < value_data.size(); i++) value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto output_with_present = RunGQAAndGetOutput(
      batch_size, sequence_length, sequence_length, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/false);

  auto output_without_present = RunGQAAndGetOutput(
      batch_size, sequence_length, sequence_length, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/true);

  ExpectOutputsMatch(output_with_present, output_without_present, 1e-5f, "OptionalPresent");
}

// Regression (batched): same equivalence check with batch_size > 1
TEST(GroupQueryAttentionTest, OptionalPresent_BatchedOmitMatchesConnected) {
  constexpr int batch_size = 2;
  constexpr int sequence_length = 3;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.15f * static_cast<float>(i % 11 + 1);
  for (size_t i = 0; i < key_data.size(); i++) key_data[i] = 0.25f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < value_data.size(); i++) value_data[i] = 0.35f * static_cast<float>(i % 5 + 1);

  auto output_with = RunGQAAndGetOutput(
      batch_size, sequence_length, sequence_length, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/false);

  auto output_without = RunGQAAndGetOutput(
      batch_size, sequence_length, sequence_length, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/true);

  ExpectOutputsMatch(output_with, output_without, 1e-5f, "BatchedOptionalPresent");
}

// KV-shared first-prompt: longer sequence with no past, present omitted.
TEST(GroupQueryAttentionTest, OptionalPresent_KVSharedFirstPrompt) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < key_data.size(); i++) key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < value_data.size(); i++) value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto output_with = RunGQAAndGetOutput(
      batch_size, sequence_length, sequence_length, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/false);
  auto output_without = RunGQAAndGetOutput(
      batch_size, sequence_length, sequence_length, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/true);

  ExpectOutputsMatch(output_with, output_without, 1e-5f, "KVSharedFirstPrompt");
}

// Reject: omitting present outputs when past_key is provided (KV cache concatenation needed)
TEST(GroupQueryAttentionTest, OptionalPresent_RejectWithPastKey) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;
  constexpr int past_seq_len = 4;
  constexpr int total_seq_len = past_seq_len + sequence_length;
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
  tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size},
                         std::vector<float>(batch_size * sequence_length * kv_hidden_size, 0.5f));
  tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size},
                         std::vector<float>(batch_size * sequence_length * kv_hidden_size, 0.5f));

  // Provide past_key/past_value — this triggers the rejection when present is omitted
  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_seq_len, head_size},
                         std::vector<float>(batch_size * kv_num_heads * past_seq_len * head_size, 0.3f));
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_seq_len, head_size},
                         std::vector<float>(batch_size * kv_num_heads * past_seq_len * head_size, 0.3f));

  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {static_cast<int32_t>(total_seq_len - 1)});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {static_cast<int32_t>(total_seq_len)});

  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(batch_size * sequence_length * hidden_size, 0.0f));
  tester.AddOptionalOutputEdge<float>();  // present_key — omitted
  tester.AddOptionalOutputEdge<float>();  // present_value — omitted

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "present_key and present_value outputs are required when past_key is provided",
             {}, nullptr, &execution_providers);
}

// Regression (CUDA): omitting present outputs on CUDA EP must produce the same
// attention output as when present outputs are connected. The CUDA path allocates
// internal scratch buffers to serve as KV workspace for flash/MEA/unfused kernels.
TEST(GroupQueryAttentionTest, OptionalPresent_CudaOmitMatchesConnected) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < key_data.size(); i++) key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < value_data.size(); i++) value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto output_with = RunGQAAndGetOutput(
      batch_size, sequence_length, sequence_length, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/false, /*use_cuda=*/true);
  auto output_without = RunGQAAndGetOutput(
      batch_size, sequence_length, sequence_length, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/true, /*use_cuda=*/true);

  ExpectOutputsMatch(output_with, output_without, 1e-5f, "CudaOptionalPresent");
}

// KV-shared decode: Q_seq=1, KV_seq=8, no past, present omitted.
TEST(GroupQueryAttentionTest, OptionalPresent_KVSharedDecode) {
  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int kv_seq_len = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> key_data(batch_size * kv_seq_len * kv_hidden_size);
  std::vector<float> value_data(batch_size * kv_seq_len * kv_hidden_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < key_data.size(); i++) key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < value_data.size(); i++) value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  auto output_with = RunGQAAndGetOutput(
      batch_size, q_seq_len, kv_seq_len, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/false);
  auto output_without = RunGQAAndGetOutput(
      batch_size, q_seq_len, kv_seq_len, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/true);

  ExpectOutputsMatch(output_with, output_without, 1e-5f, "KVSharedDecode");
}

// CUDA KV-shared decode: Q_seq=1, KV_seq=8, no past, present omitted.
// Exercises the CUDA split path (Transpose_BSNH_to_BNSH for K/V, Q used directly).
// CUDA KV-shared decode: Q_seq=1, KV_seq=8, exercises the CUDA Transpose_BSNH_to_BNSH path.
TEST(GroupQueryAttentionTest, OptionalPresent_CudaKVSharedDecode) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int kv_seq_len = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> key_data(batch_size * kv_seq_len * kv_hidden_size);
  std::vector<float> value_data(batch_size * kv_seq_len * kv_hidden_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < key_data.size(); i++) key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < value_data.size(); i++) value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  // CUDA with vs without present
  auto cuda_with = RunGQAAndGetOutput(
      batch_size, q_seq_len, kv_seq_len, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/false, /*use_cuda=*/true);
  auto cuda_without = RunGQAAndGetOutput(
      batch_size, q_seq_len, kv_seq_len, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/true, /*use_cuda=*/true);

  ExpectOutputsMatch(cuda_with, cuda_without, 1e-4f, "CudaKVSharedDecode");

  // Cross-check: CUDA should match CPU
  auto cpu_with = RunGQAAndGetOutput(
      batch_size, q_seq_len, kv_seq_len, query_data, key_data, value_data,
      num_heads, kv_num_heads, head_size, /*omit_present=*/false, /*use_cuda=*/false);

  ASSERT_EQ(cuda_with.size(), cpu_with.size());
  for (size_t i = 0; i < cuda_with.size(); i++) {
    EXPECT_NEAR(cuda_with[i], cpu_with[i], 1e-4f)
        << "CUDA vs CPU KV-shared decode mismatch at index " << i;
  }
}

}  // namespace test
}  // namespace onnxruntime
