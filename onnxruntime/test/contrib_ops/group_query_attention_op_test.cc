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

// Selects which EP backs a GQA test helper. Modeled as a single enum (rather
// than two bools) so adding a new EP later does not silently fall through to
// CPU and callers cannot accidentally select two backends at once.
enum class GqaTargetEp { kCpu,
                         kCuda,
                         kWebGpu };

// Builds the default EP for the chosen backend. Centralized so that adding a
// new enumerator only requires updating one switch; the `ORT_THROW` default
// turns a missed update into a loud runtime failure instead of a silent
// empty-provider fallback inside OpTester.
static std::unique_ptr<IExecutionProvider> MakeExecutionProviderForGqaTest(GqaTargetEp target_ep) {
  switch (target_ep) {
    case GqaTargetEp::kCuda:
      return DefaultCudaExecutionProvider();
    case GqaTargetEp::kWebGpu:
      return DefaultWebGpuExecutionProvider();
    case GqaTargetEp::kCpu:
      return DefaultCpuExecutionProvider();
  }
  ORT_THROW("Unhandled GqaTargetEp");
}

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

// CPU GroupQueryAttention does not implement the WebGPU-only fused Q/K RMS-norm prologue
// inputs (q_norm_weight/k_norm_weight at indices 14/15). Ensure we reject these explicitly.
TEST(GroupQueryAttentionTest, CpuRejectsQKNormWeightInputs) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size},
                         std::vector<float>(batch_size * sequence_length * hidden_size, 0.1f));
  tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size},
                         std::vector<float>(batch_size * sequence_length * kv_hidden_size, 0.1f));
  tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size},
                         std::vector<float>(batch_size * sequence_length * kv_hidden_size, 0.1f));

  tester.AddOptionalInputEdge<float>();  // past_key
  tester.AddOptionalInputEdge<float>();  // past_value
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {0});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {1});

  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink
  tester.AddOptionalInputEdge<float>();    // k_scale
  tester.AddOptionalInputEdge<float>();    // v_scale

  tester.AddInput<float>("q_norm_weight", {head_size}, std::vector<float>(head_size, 1.0f));
  tester.AddInput<float>("k_norm_weight", {head_size}, std::vector<float>(head_size, 1.0f));

  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(batch_size * sequence_length * hidden_size, 0.0f));
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, sequence_length, head_size},
                          std::vector<float>(batch_size * kv_num_heads * sequence_length * head_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, sequence_length, head_size},
                          std::vector<float>(batch_size * kv_num_heads * sequence_length * head_size, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "q_norm_weight / k_norm_weight inputs are not supported",
             {}, nullptr, &execution_providers);
}

// CUDA GroupQueryAttention also does not implement the WebGPU-only fused Q/K RMS-norm
// prologue inputs (q_norm_weight/k_norm_weight at indices 14/15). Ensure the guard is covered.
TEST(GroupQueryAttentionTest, CudaRejectsQKNormWeightInputs) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  tester.AddInput<MLFloat16>("query", {batch_size, sequence_length, hidden_size},
                             std::vector<MLFloat16>(batch_size * sequence_length * hidden_size, MLFloat16(0.1f)));
  tester.AddInput<MLFloat16>("key", {batch_size, sequence_length, kv_hidden_size},
                             std::vector<MLFloat16>(batch_size * sequence_length * kv_hidden_size, MLFloat16(0.1f)));
  tester.AddInput<MLFloat16>("value", {batch_size, sequence_length, kv_hidden_size},
                             std::vector<MLFloat16>(batch_size * sequence_length * kv_hidden_size, MLFloat16(0.1f)));

  tester.AddOptionalInputEdge<MLFloat16>();  // past_key
  tester.AddOptionalInputEdge<MLFloat16>();  // past_value
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {0});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {1});

  tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
  tester.AddOptionalInputEdge<int64_t>();    // position_ids
  tester.AddOptionalInputEdge<MLFloat16>();  // attention_bias
  tester.AddOptionalInputEdge<MLFloat16>();  // head_sink
  tester.AddOptionalInputEdge<MLFloat16>();  // k_scale
  tester.AddOptionalInputEdge<MLFloat16>();  // v_scale

  tester.AddInput<MLFloat16>("q_norm_weight", {head_size}, std::vector<MLFloat16>(head_size, MLFloat16(1.0f)));
  tester.AddInput<MLFloat16>("k_norm_weight", {head_size}, std::vector<MLFloat16>(head_size, MLFloat16(1.0f)));

  tester.AddOutput<MLFloat16>("output", {batch_size, sequence_length, hidden_size},
                              std::vector<MLFloat16>(batch_size * sequence_length * hidden_size, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("present_key", {batch_size, kv_num_heads, sequence_length, head_size},
                              std::vector<MLFloat16>(batch_size * kv_num_heads * sequence_length * head_size, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("present_value", {batch_size, kv_num_heads, sequence_length, head_size},
                              std::vector<MLFloat16>(batch_size * kv_num_heads * sequence_length * head_size, MLFloat16(0.0f)));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "q_norm_weight / k_norm_weight inputs are not supported",
             {}, nullptr, &execution_providers);
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

// Applies per-head simplified RMS norm to BSNH input in-place.
static void ApplyPerHeadRmsNormBSNH(std::vector<float>& data,
                                    int batch_size,
                                    int sequence_length,
                                    int num_heads,
                                    int head_size,
                                    const std::vector<float>& weight,
                                    float epsilon) {
  ASSERT_EQ(static_cast<int>(weight.size()), head_size);
  const int hidden_size = num_heads * head_size;
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < sequence_length; ++s) {
      const int token_offset = (b * sequence_length + s) * hidden_size;
      for (int h = 0; h < num_heads; ++h) {
        const int head_offset = token_offset + h * head_size;
        float mean_square = 0.0f;
        for (int d = 0; d < head_size; ++d) {
          const float v = data[head_offset + d];
          mean_square += v * v;
        }
        mean_square /= static_cast<float>(head_size);
        const float inv_rms = 1.0f / std::sqrt(mean_square + epsilon);
        for (int d = 0; d < head_size; ++d) {
          data[head_offset + d] = data[head_offset + d] * inv_rms * weight[d];
        }
      }
    }
  }
}

// Runs GroupQueryAttention with do_rotary=1 and optional q/k norm weights.
// If q_norm_weight/k_norm_weight are provided, this exercises the WebGPU-only
// q/k norm input contract. CPU callers should pass nullptr for both and feed
// pre-normalized Q/K values instead.
static std::vector<float> RunGQARotaryWithOptionalQKNorm(
    bool use_webgpu,
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& value_data,
    const std::vector<float>& past_key_data,
    const std::vector<float>& past_value_data,
    const std::vector<float>* q_norm_weight,
    const std::vector<float>* k_norm_weight,
    int batch_size,
    int sequence_length,
    int past_seq_len,
    int num_heads,
    int kv_num_heads,
    int head_size,
    float qk_norm_epsilon) {
  const int hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int total_sequence_length = past_seq_len + sequence_length;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));
  tester.AddAttribute<float>("qk_norm_epsilon", qk_norm_epsilon);

  tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size}, query_data);
  tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size}, key_data);
  tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size}, value_data);

  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_seq_len, head_size}, past_key_data);
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_seq_len, head_size}, past_value_data);

  // For prompt/decode this is the index of the last valid KV token.
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {total_sequence_length - 1});
  // Marked as initializer so shape inference (ctx.getInputData) can read the value at graph-build
  // time and compute present_seq_len = max(past_seq, total_seq) = total_seq, matching the runtime
  // allocation. The real fix (emit dynamic dim in fallback) is tracked for a separate upstream PR.
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length}, /*is_initializer=*/true);

  const int max_seq_len = total_sequence_length + 8;
  const int half_rotary = head_size / 2;
  std::vector<float> cos_cache(max_seq_len * half_rotary);
  std::vector<float> sin_cache(max_seq_len * half_rotary);
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int d = 0; d < half_rotary; ++d) {
      const float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(d) / static_cast<float>(head_size));
      cos_cache[pos * half_rotary + d] = std::cos(static_cast<float>(pos) * freq);
      sin_cache[pos * half_rotary + d] = std::sin(static_cast<float>(pos) * freq);
    }
  }
  tester.AddInput<float>("cos_cache", {max_seq_len, half_rotary}, cos_cache);
  tester.AddInput<float>("sin_cache", {max_seq_len, half_rotary}, sin_cache);

  // Position IDs are contiguous token positions for the current query span.
  std::vector<int64_t> position_ids(batch_size * sequence_length);
  const int64_t base_position = static_cast<int64_t>(total_sequence_length - sequence_length);
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < sequence_length; ++s) {
      position_ids[b * sequence_length + s] = base_position + static_cast<int64_t>(s);
    }
  }
  tester.AddInput<int64_t>("position_ids", {batch_size, sequence_length}, position_ids);

  tester.AddOptionalInputEdge<float>();  // attention_bias
  tester.AddOptionalInputEdge<float>();  // head_sink
  tester.AddOptionalInputEdge<float>();  // k_scale
  tester.AddOptionalInputEdge<float>();  // v_scale

  if (q_norm_weight && k_norm_weight) {
    tester.AddInput<float>("q_norm_weight", {head_size}, *q_norm_weight);
    tester.AddInput<float>("k_norm_weight", {head_size}, *k_norm_weight);
  }

  const int output_size = batch_size * sequence_length * hidden_size;
  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(output_size, 0.0f));

  // Shape inference computes present_seq = max(past_seq, total_seq) = total_seq (always, since
  // total_seq >= past_seq). Declaring total_sequence_length matches both inferred and actual runtime shape.
  const int present_seq_len = total_sequence_length;
  const int present_size = batch_size * kv_num_heads * present_seq_len * head_size;
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, present_seq_len, head_size},
                          std::vector<float>(present_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, present_seq_len, head_size},
                          std::vector<float>(present_size, 0.0f));

  // This helper compares fetched outputs explicitly against a CPU reference.
  // Keep OpTester from enforcing exact match with the zero-filled placeholders above.
  tester.SetOutputTolerance(1e6f);
  tester.SetCustomOutputVerifier([output_size](const std::vector<OrtValue>& fetches,
                                               const std::string& /*provider_type*/) {
    ASSERT_FALSE(fetches.empty());
    ASSERT_TRUE(fetches[0].IsTensor());
    const auto& out_tensor = fetches[0].Get<Tensor>();
    EXPECT_EQ(out_tensor.Shape().Size(), output_size);
  });

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (use_webgpu) {
    execution_providers.push_back(DefaultWebGpuExecutionProvider());
  } else {
    execution_providers.push_back(DefaultCpuExecutionProvider());
  }
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  auto fetches = tester.GetFetches();
  const float* out_data = fetches[0].Get<Tensor>().Data<float>();
  return std::vector<float>(out_data, out_data + output_size);
}

TEST(GroupQueryAttentionTest, WebGpuQKNormWeightRotaryDecodeFunctional) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;  // decode path
  constexpr int past_seq_len = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr float qk_norm_epsilon = 1e-5f;

  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);

  for (size_t i = 0; i < query_data.size(); ++i) query_data[i] = 0.03f * static_cast<float>(i + 1);
  for (size_t i = 0; i < key_data.size(); ++i) key_data[i] = 0.025f * static_cast<float>(i + 2);
  for (size_t i = 0; i < value_data.size(); ++i) value_data[i] = 0.02f * static_cast<float>((i % 7) + 1);
  for (size_t i = 0; i < past_key_data.size(); ++i) past_key_data[i] = 0.015f * static_cast<float>((i % 13) + 1);
  for (size_t i = 0; i < past_value_data.size(); ++i) past_value_data[i] = 0.01f * static_cast<float>((i % 11) + 1);

  std::vector<float> q_norm_weight(head_size);
  std::vector<float> k_norm_weight(head_size);
  for (int i = 0; i < head_size; ++i) {
    q_norm_weight[i] = 0.8f + 0.01f * static_cast<float>(i);
    k_norm_weight[i] = 0.9f + 0.008f * static_cast<float>(i);
  }

  std::vector<float> ref_query = query_data;
  std::vector<float> ref_key = key_data;
  ApplyPerHeadRmsNormBSNH(ref_query, batch_size, sequence_length, num_heads, head_size, q_norm_weight, qk_norm_epsilon);
  ApplyPerHeadRmsNormBSNH(ref_key, batch_size, sequence_length, kv_num_heads, head_size, k_norm_weight, qk_norm_epsilon);

  // CPU does not accept q_norm_weight/k_norm_weight directly. Build an equivalent
  // reference path by explicitly applying per-head RMSNorm to Q/K first, then run
  // CPU GQA without q/k norm inputs.
  const auto cpu_expected_output = RunGQARotaryWithOptionalQKNorm(
      /*use_webgpu=*/false,
      ref_query,
      ref_key,
      value_data,
      past_key_data,
      past_value_data,
      /*q_norm_weight=*/nullptr,
      /*k_norm_weight=*/nullptr,
      batch_size,
      sequence_length,
      past_seq_len,
      num_heads,
      kv_num_heads,
      head_size,
      qk_norm_epsilon);

  const auto webgpu_output = RunGQARotaryWithOptionalQKNorm(
      /*use_webgpu=*/true,
      query_data,
      key_data,
      value_data,
      past_key_data,
      past_value_data,
      &q_norm_weight,
      &k_norm_weight,
      batch_size,
      sequence_length,
      past_seq_len,
      num_heads,
      kv_num_heads,
      head_size,
      qk_norm_epsilon);

  ExpectOutputsMatch(webgpu_output, cpu_expected_output, 1e-3f, "WebGpuQKNormWeightRotaryDecodeFunctional");
}

TEST(GroupQueryAttentionTest, WebGpuQKNormWeightRotaryPrefillFunctional) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 3;  // prompt/prefill path
  constexpr int past_seq_len = 0;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr float qk_norm_epsilon = 1e-5f;

  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);

  for (size_t i = 0; i < query_data.size(); ++i) query_data[i] = 0.02f * static_cast<float>(i + 1);
  for (size_t i = 0; i < key_data.size(); ++i) key_data[i] = 0.017f * static_cast<float>(i + 3);
  for (size_t i = 0; i < value_data.size(); ++i) value_data[i] = 0.013f * static_cast<float>((i % 9) + 1);

  std::vector<float> q_norm_weight(head_size);
  std::vector<float> k_norm_weight(head_size);
  for (int i = 0; i < head_size; ++i) {
    q_norm_weight[i] = 0.85f + 0.005f * static_cast<float>(i);
    k_norm_weight[i] = 0.92f + 0.004f * static_cast<float>(i);
  }

  std::vector<float> ref_query = query_data;
  std::vector<float> ref_key = key_data;
  ApplyPerHeadRmsNormBSNH(ref_query, batch_size, sequence_length, num_heads, head_size, q_norm_weight, qk_norm_epsilon);
  ApplyPerHeadRmsNormBSNH(ref_key, batch_size, sequence_length, kv_num_heads, head_size, k_norm_weight, qk_norm_epsilon);

  const auto cpu_expected_output = RunGQARotaryWithOptionalQKNorm(
      /*use_webgpu=*/false,
      ref_query,
      ref_key,
      value_data,
      past_key_data,
      past_value_data,
      /*q_norm_weight=*/nullptr,
      /*k_norm_weight=*/nullptr,
      batch_size,
      sequence_length,
      past_seq_len,
      num_heads,
      kv_num_heads,
      head_size,
      qk_norm_epsilon);

  const auto webgpu_output = RunGQARotaryWithOptionalQKNorm(
      /*use_webgpu=*/true,
      query_data,
      key_data,
      value_data,
      past_key_data,
      past_value_data,
      &q_norm_weight,
      &k_norm_weight,
      batch_size,
      sequence_length,
      past_seq_len,
      num_heads,
      kv_num_heads,
      head_size,
      qk_norm_epsilon);

  ExpectOutputsMatch(webgpu_output, cpu_expected_output, 1e-3f, "WebGpuQKNormWeightRotaryPrefillFunctional");
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
    GqaTargetEp target_ep = GqaTargetEp::kCpu) {
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
  execution_providers.push_back(MakeExecutionProviderForGqaTest(target_ep));
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
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size);

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
    GqaTargetEp target_ep = GqaTargetEp::kCpu) {
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
  execution_providers.push_back(MakeExecutionProviderForGqaTest(target_ep));
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
      num_heads, kv_num_heads, head_size);

  // Output with rotary should differ from without rotary (RoPE changes Q projections)
  auto output_no_rotary = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size);

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
      num_heads, kv_num_heads, head_size, GqaTargetEp::kWebGpu);
  auto cpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, GqaTargetEp::kCpu);

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
      num_heads, kv_num_heads, head_size, GqaTargetEp::kWebGpu);
  auto cpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, GqaTargetEp::kCpu);

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
      num_heads, kv_num_heads, head_size, GqaTargetEp::kWebGpu);
  auto cpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, GqaTargetEp::kCpu);

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
      num_heads, kv_num_heads, head_size, GqaTargetEp::kWebGpu);
  auto cpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, GqaTargetEp::kCpu);

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
      num_heads, kv_num_heads, head_size, GqaTargetEp::kWebGpu);
  auto cpu_output = RunGQASharedKVWithRotary(
      batch_size, q_seq_len, past_seq_len, query_data, past_key_data, past_value_data,
      num_heads, kv_num_heads, head_size, GqaTargetEp::kCpu);

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

// ---------------------------------------------------------------------------
// Batched right-padded packed-QKV prefill with do_rotary.
//
// In a multi-batch prefill where individual prompts have different real lengths,
// GenAI right-pads short prompts up to the max sequence_length and reports each
// batch's real length via seqlens_k[b] = real_len[b] - 1. The property under
// test: each batch's real-last-token output (the one used to predict the next
// token) must equal what we get from running that prompt singly as a batch=1
// prefill. This is a generic correctness check that any GQA-supporting EP
// should satisfy.
// ---------------------------------------------------------------------------

// Builds a packed QKV tensor with deterministic values at real positions and
// zeros at right-padded positions. Layout per token: [Q(hidden), K(kv), V(kv)].
// Uses values of order ~1.0 (well above the 5e-3 mismatch tolerance) so the
// rotated-vs-unrotated divergence is unambiguously detectable.
static void FillBatchedRightPaddedPackedQKV(int batch_size,
                                            int sequence_length,
                                            int num_heads,
                                            int kv_num_heads,
                                            int head_size,
                                            const std::vector<int>& real_lens,
                                            std::vector<float>& packed_out) {
  const int hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int token_size = hidden_size + 2 * kv_hidden_size;
  packed_out.assign(batch_size * sequence_length * token_size, 0.0f);
  for (int b = 0; b < batch_size; ++b) {
    const int real_len = real_lens[b];
    for (int s = 0; s < real_len; ++s) {
      float* token = &packed_out[(b * sequence_length + s) * token_size];
      for (int c = 0; c < hidden_size; ++c) {
        token[c] = 0.1f + 0.3f * static_cast<float>(((b * 7 + s * 3 + c) % 13) + 1);
      }
      for (int c = 0; c < kv_hidden_size; ++c) {
        token[hidden_size + c] =
            0.1f + 0.25f * static_cast<float>(((b * 5 + s * 2 + c) % 11) + 1);
        token[hidden_size + kv_hidden_size + c] =
            0.1f + 0.2f * static_cast<float>(((b * 3 + s + c) % 9) + 1);
      }
    }
  }
}

// Runs a packed-QKV GQA prefill with do_rotary=1 and the given per-batch
// seqlens_k. Returns the output tensor [batch_size, sequence_length, hidden_size].
static std::vector<float> RunGQAPackedQKVRotaryPrefill(
    int batch_size,
    int sequence_length,
    int num_heads,
    int kv_num_heads,
    int head_size,
    const std::vector<int32_t>& seqlens_k_data,
    const std::vector<float>& packed_qkv_data,
    GqaTargetEp target_ep = GqaTargetEp::kCpu) {
  const int hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int qkv_hidden = hidden_size + 2 * kv_hidden_size;
  const int total_sequence_length = sequence_length;  // prefill: no past
  const int half_rotary = head_size / 2;
  const int max_seq_len = sequence_length + 8;

  // The CUDA GQA kernel only registers for MLFloat16/BFloat16, so float inputs
  // silently fall back to the CPU EP. Feed fp16 tensors when targeting CUDA so
  // the *_CUDA test genuinely exercises the CUDA kernel. The CPU and WebGPU
  // kernels both support float (WebGpuSupportedFloatTypes = {float, MLFloat16}),
  // so those paths keep fp32 for tighter numeric comparison.
  const bool use_fp16 = target_ep == GqaTargetEp::kCuda;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));

  // Packed QKV: pass through `query` input, leave key/value as optional edges.
  if (use_fp16) {
    tester.AddInput<MLFloat16>("query", {batch_size, sequence_length, qkv_hidden}, ToFloat16(packed_qkv_data));
    tester.AddOptionalInputEdge<MLFloat16>();  // key (signals packed)
    tester.AddOptionalInputEdge<MLFloat16>();  // value (signals packed)
    tester.AddOptionalInputEdge<MLFloat16>();  // past_key
    tester.AddOptionalInputEdge<MLFloat16>();  // past_value
  } else {
    tester.AddInput<float>("query", {batch_size, sequence_length, qkv_hidden}, packed_qkv_data);
    tester.AddOptionalInputEdge<float>();  // key (signals packed)
    tester.AddOptionalInputEdge<float>();  // value (signals packed)
    tester.AddOptionalInputEdge<float>();  // past_key
    tester.AddOptionalInputEdge<float>();  // past_value
  }

  tester.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k_data);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length},
                           /*is_initializer=*/true);

  std::vector<float> cos_cache(max_seq_len * half_rotary);
  std::vector<float> sin_cache(max_seq_len * half_rotary);
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int d = 0; d < half_rotary; ++d) {
      const float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(d) /
                                                       static_cast<float>(head_size));
      cos_cache[pos * half_rotary + d] = std::cos(static_cast<float>(pos) * freq);
      sin_cache[pos * half_rotary + d] = std::sin(static_cast<float>(pos) * freq);
    }
  }
  if (use_fp16) {
    tester.AddInput<MLFloat16>("cos_cache", {max_seq_len, half_rotary}, ToFloat16(cos_cache));
    tester.AddInput<MLFloat16>("sin_cache", {max_seq_len, half_rotary}, ToFloat16(sin_cache));
  } else {
    tester.AddInput<float>("cos_cache", {max_seq_len, half_rotary}, cos_cache);
    tester.AddInput<float>("sin_cache", {max_seq_len, half_rotary}, sin_cache);
  }

  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  if (use_fp16) {
    tester.AddOptionalInputEdge<MLFloat16>();  // attention_bias
    tester.AddOptionalInputEdge<MLFloat16>();  // head_sink
  } else {
    tester.AddOptionalInputEdge<float>();  // attention_bias
    tester.AddOptionalInputEdge<float>();  // head_sink
  }

  const int output_size = batch_size * sequence_length * hidden_size;
  const int present_size = batch_size * kv_num_heads * total_sequence_length * head_size;
  if (use_fp16) {
    tester.AddOutput<MLFloat16>("output", {batch_size, sequence_length, hidden_size},
                                std::vector<MLFloat16>(output_size, MLFloat16(0.0f)));
    tester.AddOutput<MLFloat16>("present_key", {batch_size, kv_num_heads, total_sequence_length, head_size},
                                std::vector<MLFloat16>(present_size, MLFloat16(0.0f)));
    tester.AddOutput<MLFloat16>("present_value", {batch_size, kv_num_heads, total_sequence_length, head_size},
                                std::vector<MLFloat16>(present_size, MLFloat16(0.0f)));
  } else {
    tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                            std::vector<float>(output_size, 0.0f));
    tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, total_sequence_length, head_size},
                            std::vector<float>(present_size, 0.0f));
    tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, total_sequence_length, head_size},
                            std::vector<float>(present_size, 0.0f));
  }

  tester.SetOutputTolerance(1e6f);  // We fetch and compare outputs ourselves.

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(MakeExecutionProviderForGqaTest(target_ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  auto fetches = tester.GetFetches();
  std::vector<float> result(output_size);
  if (use_fp16) {
    const MLFloat16* out_data = fetches[0].Get<Tensor>().Data<MLFloat16>();
    for (int i = 0; i < output_size; ++i) {
      result[i] = out_data[i].ToFloat();
    }
  } else {
    const float* out_data = fetches[0].Get<Tensor>().Data<float>();
    std::copy_n(out_data, output_size, result.begin());
  }
  return result;
}

// Inner helper: builds packed-QKV inputs, computes per-prompt references, runs
// the right-padded batched prefill, and asserts each batch's real-last-token
// output matches its single-prompt reference. Both reference and batched runs
// go through the same EP, so this validates per-batch consistency within each
// EP rather than cross-EP equivalence.
static void RunBatchedRightPaddedRotaryPrefillForEP(GqaTargetEp target_ep) {
  constexpr int batch_size = 3;
  constexpr int num_heads = 4;
  constexpr int kv_num_heads = 2;
  constexpr int head_size = 16;  // multiple of 4 for FlashAttention gate; rotary half = 8
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int qkv_hidden = hidden_size + 2 * kv_hidden_size;

  // Real prompt lengths per batch; max = sequence_length (right-padding extends
  // shorter batches up to this length). The bug only manifests when at least
  // one batch is shorter than sequence_length.
  const std::vector<int> real_lens = {4, 2, 6};
  const int sequence_length = *std::max_element(real_lens.begin(), real_lens.end());

  std::vector<float> packed_batched;
  FillBatchedRightPaddedPackedQKV(batch_size, sequence_length, num_heads, kv_num_heads,
                                  head_size, real_lens, packed_batched);

  // Build single-prompt references by extracting each batch's real-len slice
  // and running it as a batch_size=1 prefill (which is known correct).
  std::vector<std::vector<float>> ref_outputs(batch_size);
  for (int b = 0; b < batch_size; ++b) {
    const int real_len = real_lens[b];
    std::vector<float> packed_single(real_len * qkv_hidden);
    for (int s = 0; s < real_len; ++s) {
      std::copy_n(&packed_batched[(b * sequence_length + s) * qkv_hidden], qkv_hidden,
                  &packed_single[s * qkv_hidden]);
    }
    ref_outputs[b] = RunGQAPackedQKVRotaryPrefill(
        /*batch_size=*/1, /*sequence_length=*/real_len,
        num_heads, kv_num_heads, head_size,
        /*seqlens_k_data=*/{static_cast<int32_t>(real_len - 1)},
        packed_single, target_ep);
  }

  // Now run all batches together with right-padding.
  std::vector<int32_t> seqlens_k_data(batch_size);
  for (int b = 0; b < batch_size; ++b) {
    seqlens_k_data[b] = static_cast<int32_t>(real_lens[b] - 1);
  }
  const auto batched_output = RunGQAPackedQKVRotaryPrefill(
      batch_size, sequence_length, num_heads, kv_num_heads, head_size,
      seqlens_k_data, packed_batched, target_ep);

  // Each batch's real-last-token output (used to predict next token) must match
  // its single-prompt reference. The tolerance is loose enough for fp16 rounding
  // while still catching the underflow bug (which produces values that differ
  // by orders of magnitude or are NaN/Inf).
  constexpr float tolerance = 5e-3f;
  for (int b = 0; b < batch_size; ++b) {
    const int real_len = real_lens[b];
    const int q_last = real_len - 1;
    const float* batched_last =
        batched_output.data() + (b * sequence_length + q_last) * hidden_size;
    const float* ref_last = ref_outputs[b].data() + q_last * hidden_size;
    for (int c = 0; c < hidden_size; ++c) {
      EXPECT_NEAR(batched_last[c], ref_last[c], tolerance)
          << "batch " << b << " real_len=" << real_len
          << " channel " << c << " mismatch";
    }
  }
}

TEST(GroupQueryAttentionTest, BatchedRightPaddedRotaryPrefill_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }
  RunBatchedRightPaddedRotaryPrefillForEP(GqaTargetEp::kCuda);
}

TEST(GroupQueryAttentionTest, BatchedRightPaddedRotaryPrefill_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  RunBatchedRightPaddedRotaryPrefillForEP(GqaTargetEp::kWebGpu);
}

}  // namespace test
}  // namespace onnxruntime
