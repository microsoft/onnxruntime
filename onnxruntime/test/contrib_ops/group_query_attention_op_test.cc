// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#ifdef USE_WEBGPU
#include "contrib_ops/webgpu/bert/kv_cache_quantization.h"
#include "core/graph/model.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/inference_session.h"
#include "core/session/IOBinding.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#endif

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

template <typename T>
static void RunGQACausalMaskTest(
    GqaTargetEp target_ep,
    std::optional<int64_t> causal,
    const std::vector<float>& expected_output,
    OpTester::ExpectResult expect = OpTester::ExpectResult::kExpectSuccess,
    const std::string& expected_message = "",
    std::optional<int64_t> local_window_size = std::nullopt) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 2;
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;

  auto execution_provider = MakeExecutionProviderForGqaTest(target_ep);
  if (!execution_provider) {
    GTEST_SKIP() << "Requested execution provider is not available";
  }

  auto convert = [](const std::vector<float>& values) {
    std::vector<T> converted;
    converted.reserve(values.size());
    for (float value : values) {
      converted.emplace_back(value);
    }
    return converted;
  };

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", num_heads);
  tester.AddAttribute<int64_t>("kv_num_heads", kv_num_heads);
  if (causal.has_value()) {
    tester.AddAttribute<int64_t>("causal", *causal);
  }
  if (local_window_size.has_value()) {
    tester.AddAttribute<int64_t>("local_window_size", *local_window_size);
  }

  std::vector<float> query(sequence_length * hidden_size, 0.0f);
  std::vector<float> key(sequence_length * hidden_size, 0.0f);
  query[0] = 1.0f;
  query[hidden_size] = 1.0f;
  key[hidden_size] = std::sqrt(static_cast<float>(head_size)) * std::log(3.0f);
  std::vector<float> value(sequence_length * hidden_size, 1.0f);
  std::fill(value.begin() + hidden_size, value.end(), 3.0f);
  tester.AddInput<T>("query", {batch_size, sequence_length, hidden_size}, convert(query));
  tester.AddInput<T>("key", {batch_size, sequence_length, hidden_size}, convert(key));
  tester.AddInput<T>("value", {batch_size, sequence_length, hidden_size}, convert(value));
  tester.AddOptionalInputEdge<T>();  // past_key
  tester.AddOptionalInputEdge<T>();  // past_value
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {sequence_length - 1});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {sequence_length});
  tester.AddOptionalInputEdge<T>();        // cos_cache
  tester.AddOptionalInputEdge<T>();        // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<T>();        // attention_bias
  tester.AddOptionalInputEdge<T>();        // head_sink

  tester.AddOutput<T>("output", {batch_size, sequence_length, hidden_size}, convert(expected_output));
  tester.AddOutput<T>("present_key", {batch_size, kv_num_heads, sequence_length, head_size},
                      convert(key));
  tester.AddOutput<T>("present_value", {batch_size, kv_num_heads, sequence_length, head_size},
                      convert(value));
  tester.SetOutputTolerance(0.001f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(execution_provider));
  tester.Run(expect, expected_message, {}, nullptr, &execution_providers);
}

TEST(GroupQueryAttentionTest, CausalMaskDefaultsToEnabled_CPU) {
  std::vector<float> expected_output(16, 2.5f);
  std::fill(expected_output.begin(), expected_output.begin() + 8, 1.0f);
  RunGQACausalMaskTest<float>(GqaTargetEp::kCpu, std::nullopt, expected_output);
}

TEST(GroupQueryAttentionTest, BidirectionalMask_CPU) {
  RunGQACausalMaskTest<float>(GqaTargetEp::kCpu, 0, std::vector<float>(16, 2.5f));
}

TEST(GroupQueryAttentionTest, BidirectionalMask_CUDA) {
  RunGQACausalMaskTest<MLFloat16>(GqaTargetEp::kCuda, 0, std::vector<float>(16, 2.5f));
}

TEST(GroupQueryAttentionTest, BidirectionalLocalWindowRejected_CPU) {
  RunGQACausalMaskTest<float>(
      GqaTargetEp::kCpu, 0, std::vector<float>(16, 0.0f),
      OpTester::ExpectResult::kExpectFailure,
      "GroupQueryAttention (CPU): local_window_size must be -1 when causal is 0.", 1);
}

TEST(GroupQueryAttentionTest, BidirectionalLocalWindowRejected_CUDA) {
  RunGQACausalMaskTest<MLFloat16>(
      GqaTargetEp::kCuda, 0, std::vector<float>(16, 0.0f),
      OpTester::ExpectResult::kExpectFailure,
      "GroupQueryAttention (CUDA): local_window_size must be -1 when causal is 0.", 1);
}

TEST(GroupQueryAttentionTest, InvalidCausalValue_CPU) {
  RunGQACausalMaskTest<float>(
      GqaTargetEp::kCpu, 2, std::vector<float>(16, 0.0f),
      OpTester::ExpectResult::kExpectFailure, "causal must be 0 or 1.");
}

TEST(GroupQueryAttentionTest, InvalidCausalValue_CUDA) {
  RunGQACausalMaskTest<MLFloat16>(
      GqaTargetEp::kCuda, 2, std::vector<float>(16, 0.0f),
      OpTester::ExpectResult::kExpectFailure, "causal must be 0 or 1.");
}

#ifdef USE_WEBGPU
TEST(GroupQueryAttentionTest, BidirectionalMaskNotImplemented_WebGPU) {
  RunGQACausalMaskTest<float>(
      GqaTargetEp::kWebGpu, 0, std::vector<float>(16, 2.0f),
      OpTester::ExpectResult::kExpectFailure, "GroupQueryAttention (WebGPU): causal=0 is not implemented.");
}

TEST(GroupQueryAttentionTest, InvalidCausalValue_WebGPU) {
  RunGQACausalMaskTest<float>(
      GqaTargetEp::kWebGpu, 2, std::vector<float>(16, 0.0f),
      OpTester::ExpectResult::kExpectFailure, "causal must be 0 or 1.");
}
#endif

// CPU GroupQueryAttention does not implement the CUDA/WebGPU fused Q/K RMS-norm prologue
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

static void RunCudaQKNormInputContractTest(
    OpTester::ExpectResult expect,
    const std::string& expected_message,
    std::optional<float> qk_norm_epsilon = std::nullopt,
    bool include_q_norm_weight = true,
    bool include_k_norm_weight = true,
    int q_norm_weight_size = 8,
    int k_norm_weight_size = 8) {
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
  if (qk_norm_epsilon.has_value()) {
    tester.AddAttribute<float>("qk_norm_epsilon", *qk_norm_epsilon);
  }

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

  if (include_q_norm_weight) {
    tester.AddInput<MLFloat16>("q_norm_weight", {q_norm_weight_size},
                               std::vector<MLFloat16>(q_norm_weight_size, MLFloat16(1.0f)));
  } else if (include_k_norm_weight) {
    tester.AddOptionalInputEdge<MLFloat16>();
  }
  if (include_k_norm_weight) {
    tester.AddInput<MLFloat16>("k_norm_weight", {k_norm_weight_size},
                               std::vector<MLFloat16>(k_norm_weight_size, MLFloat16(1.0f)));
  }

  tester.AddOutput<MLFloat16>("output", {batch_size, sequence_length, hidden_size},
                              std::vector<MLFloat16>(batch_size * sequence_length * hidden_size, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("present_key", {batch_size, kv_num_heads, sequence_length, head_size},
                              std::vector<MLFloat16>(batch_size * kv_num_heads * sequence_length * head_size, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("present_value", {batch_size, kv_num_heads, sequence_length, head_size},
                              std::vector<MLFloat16>(batch_size * kv_num_heads * sequence_length * head_size, MLFloat16(0.0f)));

  if (expect == OpTester::ExpectResult::kExpectSuccess) {
    // This is an input-contract smoke test. The dedicated QK-Norm functional tests cover numerical equivalence.
    tester.SetOutputTolerance(1e6f);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  tester.Run(expect, expected_message, {}, nullptr, &execution_providers);
}

// CUDA GroupQueryAttention implements the fused Q/K RMS-norm prologue inputs
// (q_norm_weight/k_norm_weight at indices 14/15). Ensure the input contract is accepted.
TEST(GroupQueryAttentionTest, CudaAcceptsQKNormWeightInputs) {
  RunCudaQKNormInputContractTest(OpTester::ExpectResult::kExpectSuccess, "");
}

TEST(GroupQueryAttentionTest, CudaRejectsQKNormMissingKWeight) {
  RunCudaQKNormInputContractTest(OpTester::ExpectResult::kExpectFailure,
                                 "q_norm_weight and k_norm_weight must be provided together",
                                 std::nullopt,
                                 /*include_q_norm_weight=*/true,
                                 /*include_k_norm_weight=*/false);
}

TEST(GroupQueryAttentionTest, CudaRejectsQKNormWrongKWeightShape) {
  RunCudaQKNormInputContractTest(OpTester::ExpectResult::kExpectFailure,
                                 "k_norm_weight must be a 1D tensor of shape",
                                 std::nullopt,
                                 /*include_q_norm_weight=*/true,
                                 /*include_k_norm_weight=*/true,
                                 /*q_norm_weight_size=*/8,
                                 /*k_norm_weight_size=*/9);
}

TEST(GroupQueryAttentionTest, CudaRejectsZeroQKNormEpsilon) {
  RunCudaQKNormInputContractTest(OpTester::ExpectResult::kExpectFailure,
                                 "qk_norm_epsilon must be finite and positive",
                                 0.0f);
}

TEST(GroupQueryAttentionTest, CudaRejectsNegativeQKNormEpsilon) {
  RunCudaQKNormInputContractTest(OpTester::ExpectResult::kExpectFailure,
                                 "qk_norm_epsilon must be finite and positive",
                                 -1.0e-6f);
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

// Regression: present buffer large enough (total_seq_len passes the present-buffer check),
// but the past buffer is much smaller. ConcatStateChunkGQA would copy
// (seqlens_k + 1 - sequence_length) rows out of the small past buffer, reading past its end.
TEST(GroupQueryAttentionTest, SeqlensKExceedsPastBuffer_OOBRead) {
  // present_kv_seqlen = max(total_seq_len=100, past_seq_len=2) = 100, so seqlens_k=50 passes the
  // present-buffer check, but past_seqlen = 51 - 1 = 50 rows >> past buffer (2 rows) => OOB read.
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{50},
      /*total_seq_len=*/100,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "exceeds the past buffer sequence length",
      /*provide_past=*/true,
      /*past_seq_len=*/2);
}

TEST(GroupQueryAttentionTest, SeqlensKExceedsEmptyPastBuffer_OOBRead) {
  RunGQASeqlensKTest(
      /*seqlens_k_data=*/{50},
      /*total_seq_len=*/100,
      /*batch_size=*/1,
      /*sequence_length=*/1,
      OpTester::ExpectResult::kExpectFailure,
      "exceeds the past buffer sequence length",
      /*provide_past=*/true,
      /*past_seq_len=*/0);
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
// If q_norm_weight/k_norm_weight are provided, this exercises the EP-specific
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

// CUDA: out-of-range (negative) seqlens_k must not drive an out-of-bounds KV-cache write.
// On the CUDA EP seqlens_k is device-resident, so the host-side range check in the operator is
// skipped and the derived append offset is clamped on the device instead. With sequence_length > 1
// the non-fast-decode path is taken, exercising both the derived-length clamp and the cache-store
// bound. The run must complete and yield finite outputs. This is a memory-safety regression that is
// most precisely observed under compute-sanitizer, where the pre-clamp code reported an invalid
// device write at this site.
TEST(GroupQueryAttentionTest, NegativeSeqlensK_CacheAppend_NoOOB_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 2;  // > 1 forces the non-fast-decode path
  constexpr int past_seq_len = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;  // must be a multiple of 16 for rotary
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int total_sequence_length = past_seq_len + sequence_length;
  constexpr int present_seq_len = total_sequence_length;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));

  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); ++i) query_data[i] = 0.05f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < key_data.size(); ++i) key_data[i] = 0.04f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < value_data.size(); ++i) value_data[i] = 0.03f * static_cast<float>(i % 3 + 1);
  for (size_t i = 0; i < past_key_data.size(); ++i) past_key_data[i] = 0.02f * static_cast<float>(i % 11 + 1);
  for (size_t i = 0; i < past_value_data.size(); ++i) past_value_data[i] = 0.01f * static_cast<float>(i % 13 + 1);

  tester.AddInput<MLFloat16>("query", {batch_size, sequence_length, hidden_size}, ToFloat16(query_data));
  tester.AddInput<MLFloat16>("key", {batch_size, sequence_length, kv_hidden_size}, ToFloat16(key_data));
  tester.AddInput<MLFloat16>("value", {batch_size, sequence_length, kv_hidden_size}, ToFloat16(value_data));
  tester.AddInput<MLFloat16>("past_key", {batch_size, kv_num_heads, past_seq_len, head_size}, ToFloat16(past_key_data));
  tester.AddInput<MLFloat16>("past_value", {batch_size, kv_num_heads, past_seq_len, head_size},
                             ToFloat16(past_value_data));

  // seqlens_k is negative, so the derived past length, (max(seqlens_k, 0) + 1) - sequence_length, is
  // negative (here 0 + 1 - 2 = -1). The device-side derivation must neutralize this so the cache append
  // for the new tokens stays within the present buffer instead of indexing before its start.
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {-1});
  // Marked as an initializer so shape inference can read the value at graph-build time and size
  // present_kv to max(past_seq_len, total_sequence_length), matching the declared present outputs below.
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
  tester.AddInput<MLFloat16>("cos_cache", {max_seq_len, half_rotary}, ToFloat16(cos_cache));
  tester.AddInput<MLFloat16>("sin_cache", {max_seq_len, half_rotary}, ToFloat16(sin_cache));

  // Valid position_ids so the rotary index path is well-formed and only the cache-store bound is stressed.
  std::vector<int64_t> position_ids(batch_size * sequence_length);
  for (int s = 0; s < sequence_length; ++s) {
    position_ids[s] = static_cast<int64_t>(past_seq_len + s);
  }
  tester.AddInput<int64_t>("position_ids", {batch_size, sequence_length}, position_ids);

  const int output_size = batch_size * sequence_length * hidden_size;
  tester.AddOutput<MLFloat16>("output", {batch_size, sequence_length, hidden_size},
                              std::vector<MLFloat16>(output_size, MLFloat16(0.0f)));
  const int present_size = batch_size * kv_num_heads * present_seq_len * head_size;
  tester.AddOutput<MLFloat16>("present_key", {batch_size, kv_num_heads, present_seq_len, head_size},
                              std::vector<MLFloat16>(present_size, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("present_value", {batch_size, kv_num_heads, present_seq_len, head_size},
                              std::vector<MLFloat16>(present_size, MLFloat16(0.0f)));

  // The malformed seqlens_k drives the derived past length negative, which is the condition under test.
  // That leaves the KV length under-specified for the query, so the attention is degenerate and its
  // outputs may be non-finite; this is expected and intentionally not asserted. The regression point is
  // that the cache append and attention complete without indexing outside their buffers (which a
  // sanitizer build would otherwise flag), so only the output shape is verified.
  tester.SetOutputTolerance(1e6f);
  tester.SetCustomOutputVerifier([](const std::vector<OrtValue>& fetches,
                                    const std::string& /*provider*/) {
    ASSERT_FALSE(fetches.empty());
    ASSERT_TRUE(fetches[0].IsTensor());
    EXPECT_EQ(fetches[0].Get<Tensor>().Shape().Size(), static_cast<int64_t>(output_size));
  });

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
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
  bool use_fp16 = false;
  bool disable_flash = false;
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

  if (cfg.use_fp16) {
    auto to_fp16 = [](const std::vector<float>& data) {
      std::vector<MLFloat16> result;
      result.reserve(data.size());
      for (float value : data) {
        result.emplace_back(value);
      }
      return result;
    };
    tester.AddInput<MLFloat16>("query", {cfg.batch_size, cfg.seq_len, hidden_size}, to_fp16(query_data));
    tester.AddInput<MLFloat16>("key", {cfg.batch_size, cfg.seq_len, kv_hidden_size}, to_fp16(key_data));
    tester.AddInput<MLFloat16>("value", {cfg.batch_size, cfg.seq_len, kv_hidden_size}, to_fp16(value_data));
  } else {
    tester.AddInput<float>("query", {cfg.batch_size, cfg.seq_len, hidden_size}, query_data);
    tester.AddInput<float>("key", {cfg.batch_size, cfg.seq_len, kv_hidden_size}, key_data);
    tester.AddInput<float>("value", {cfg.batch_size, cfg.seq_len, kv_hidden_size}, value_data);
  }

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

  if (cfg.use_fp16) {
    tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
    tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
  } else {
    tester.AddOptionalInputEdge<float>();  // cos_cache
    tester.AddOptionalInputEdge<float>();  // sin_cache
  }
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  if (cfg.use_fp16) {
    tester.AddOptionalInputEdge<MLFloat16>();  // attention_bias
    tester.AddOptionalInputEdge<MLFloat16>();  // head_sink
  } else {
    tester.AddOptionalInputEdge<float>();  // attention_bias
    tester.AddOptionalInputEdge<float>();  // head_sink
  }

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
  if (cfg.use_fp16) {
    tester.AddOutput<MLFloat16>("output", {cfg.batch_size, cfg.seq_len, hidden_size},
                                std::vector<MLFloat16>(output_size, MLFloat16(0.0f)));
  } else {
    tester.AddOutput<float>("output", {cfg.batch_size, cfg.seq_len, hidden_size},
                            std::vector<float>(output_size, 0.0f));
  }

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
    const float* out_data = nullptr;
    std::vector<float> fp16_output;
    if (cfg.use_fp16) {
      const MLFloat16* output_fp16 = fetches[0].Get<Tensor>().Data<MLFloat16>();
      fp16_output.reserve(output_size);
      for (int i = 0; i < output_size; ++i) {
        fp16_output.push_back(static_cast<float>(output_fp16[i]));
      }
      out_data = fp16_output.data();
    } else {
      out_data = fetches[0].Get<Tensor>().Data<float>();
    }

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
  if (cfg.disable_flash) {
    ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_GQA_DISABLE_FLASH_ATTENTION", "1"}}};
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  } else {
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

void RunQuantizedGQAFp16DecodeTest(int bit_width) {
  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;
  constexpr int cache_length = 2;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  const int packed_head_size = bit_width == 4 ? head_size / 2 : head_size;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<std::string>("k_quant_type", "PER_TENSOR");
  tester.AddAttribute<std::string>("v_quant_type", "PER_TENSOR");
  tester.AddAttribute<int64_t>("kv_cache_bit_width", bit_width);

  tester.AddInput<MLFloat16>("query", {batch_size, sequence_length, hidden_size},
                             std::vector<MLFloat16>(hidden_size, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {batch_size, sequence_length, kv_hidden_size},
                             std::vector<MLFloat16>(kv_hidden_size, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {batch_size, sequence_length, kv_hidden_size},
                             std::vector<MLFloat16>(kv_hidden_size, MLFloat16(0.3f)));

  if (bit_width == 4) {
    tester.AddInput<uint8_t>("past_key", {batch_size, kv_num_heads, cache_length, packed_head_size},
                             std::vector<uint8_t>(cache_length * packed_head_size, 0x88));
    tester.AddInput<uint8_t>("past_value", {batch_size, kv_num_heads, cache_length, packed_head_size},
                             {0xAA, 0xAA, 0xAA, 0xAA, 0x88, 0x88, 0x88, 0x88});
  } else {
    tester.AddInput<int8_t>("past_key", {batch_size, kv_num_heads, cache_length, packed_head_size},
                            std::vector<int8_t>(cache_length * packed_head_size, 0));
    std::vector<int8_t> past_value(cache_length * packed_head_size, 0);
    std::fill_n(past_value.begin(), head_size, 10);
    tester.AddInput<int8_t>("past_value", {batch_size, kv_num_heads, cache_length, packed_head_size}, past_value);
  }

  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {cache_length - 1});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {cache_length});
  tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
  tester.AddOptionalInputEdge<int64_t>();    // position_ids
  tester.AddOptionalInputEdge<MLFloat16>();  // attention_bias
  tester.AddOptionalInputEdge<MLFloat16>();  // head_sink
  const float scale = bit_width == 4 ? 0.05f : 0.01f;
  tester.AddInput<float>("k_scale", {1}, {scale});
  tester.AddInput<float>("v_scale", {1}, {scale});

  tester.AddOutput<MLFloat16>("output", {batch_size, sequence_length, hidden_size},
                              std::vector<MLFloat16>(hidden_size, MLFloat16(0.2f)));
  if (bit_width == 4) {
    tester.AddOutput<uint8_t>("present_key", {batch_size, kv_num_heads, cache_length, packed_head_size},
                              std::vector<uint8_t>(cache_length * packed_head_size, 0x88));
    tester.AddOutput<uint8_t>("present_value", {batch_size, kv_num_heads, cache_length, packed_head_size},
                              {0xAA, 0xAA, 0xAA, 0xAA, 0xEE, 0xEE, 0xEE, 0xEE});
  } else {
    tester.AddOutput<int8_t>("present_key", {batch_size, kv_num_heads, cache_length, packed_head_size},
                             std::vector<int8_t>(cache_length * packed_head_size, 0));
    std::vector<int8_t> present_value(cache_length * packed_head_size, 30);
    std::fill_n(present_value.begin(), head_size, 10);
    tester.AddOutput<int8_t>("present_value", {batch_size, kv_num_heads, cache_length, packed_head_size},
                             present_value);
  }

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

TEST(GroupQueryAttentionTest, QuantizedKV_FP16_INT8_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/8, /*quant_type=*/"PER_TENSOR", /*bit_width=*/8,
                             /*use_fp16=*/true});
}

TEST(GroupQueryAttentionTest, QuantizedKV_FP16_INT4_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/8, /*quant_type=*/"PER_TENSOR", /*bit_width=*/4,
                             /*use_fp16=*/true});
}

TEST(GroupQueryAttentionTest, QuantizedKV_FP16_INT8_PerChannel_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/8, /*quant_type=*/"PER_CHANNEL", /*bit_width=*/8,
                             /*use_fp16=*/true});
}

TEST(GroupQueryAttentionTest, QuantizedKV_FP16_INT4_PerChannel_Prompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/8, /*quant_type=*/"PER_CHANNEL", /*bit_width=*/4,
                             /*use_fp16=*/true});
}

TEST(GroupQueryAttentionTest, QuantizedKV_FP16_INT8_NaivePrompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/8, /*quant_type=*/"PER_TENSOR", /*bit_width=*/8,
                             /*use_fp16=*/true, /*disable_flash=*/true});
}

TEST(GroupQueryAttentionTest, QuantizedKV_FP16_INT4_NaivePrompt) {
  RunQuantizedGQAPromptTest({/*batch_size=*/1, /*seq_len=*/4, /*num_heads=*/2, /*kv_num_heads=*/1,
                             /*head_size=*/8, /*quant_type=*/"PER_TENSOR", /*bit_width=*/4,
                             /*use_fp16=*/true, /*disable_flash=*/true});
}

TEST(GroupQueryAttentionTest, QuantizedKV_FP16_INT8_Decode) {
  RunQuantizedGQAFp16DecodeTest(8);
}

TEST(GroupQueryAttentionTest, QuantizedKV_FP16_INT4_Decode) {
  RunQuantizedGQAFp16DecodeTest(4);
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
// attention_bias parity (CUDA vs CPU).
//
// The CUDA GQA kernel routes attention_bias-carrying nodes to the unfused
// fallback; the CPU EP implements attention_bias directly, so it is the
// reference. This runs the SAME prompt case on CUDA (fp16, the only float type
// the CUDA kernel registers) and on CPU (fp32) and compares the outputs,
// covering the three bias shapes that drive broadcast_attn_bias_dim_0/1:
//   [batch, 1,    S, S]  -> dim0=false, dim1=true  (the default)
//   [1,     1,    S, S]  -> dim0=true              (batch broadcast)
//   [batch, heads, S, S] -> dim1=false             (per-head)
// These run on real GPU in PR CI (C++ ctest), unlike the Python parity tests
// which need a CUDA-enabled torch the PR agents don't have.
// ---------------------------------------------------------------------------
template <typename T>
static std::vector<float> RunGQAPromptWithBias(
    int batch_size, int seq_len, int num_heads, int kv_num_heads, int head_size,
    int bias_dim0, int bias_dim1,
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& value_data,
    const std::vector<float>& bias_data,
    GqaTargetEp target_ep) {
  const int hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int total_seq_len = seq_len;  // prompt: no past

  auto cvt = [](const std::vector<float>& v) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      return ToFloat16(v);
    } else {
      return v;
    }
  };

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  tester.AddInput<T>("query", {batch_size, seq_len, hidden_size}, cvt(query_data));
  tester.AddInput<T>("key", {batch_size, seq_len, kv_hidden_size}, cvt(key_data));
  tester.AddInput<T>("value", {batch_size, seq_len, kv_hidden_size}, cvt(value_data));

  tester.AddOptionalInputEdge<T>();  // past_key
  tester.AddOptionalInputEdge<T>();  // past_value
  std::vector<int32_t> seqlens_k(batch_size, total_seq_len - 1);
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_seq_len}, /*is_initializer=*/true);

  tester.AddOptionalInputEdge<T>();        // cos_cache
  tester.AddOptionalInputEdge<T>();        // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddInput<T>("attention_bias", {bias_dim0, bias_dim1, seq_len, total_seq_len}, cvt(bias_data));

  const int output_size = batch_size * seq_len * hidden_size;
  tester.AddOutput<T>("output", {batch_size, seq_len, hidden_size}, std::vector<T>(output_size, T(0.0f)));
  const int present_size = batch_size * kv_num_heads * total_seq_len * head_size;
  tester.AddOutput<T>("present_key", {batch_size, kv_num_heads, total_seq_len, head_size},
                      std::vector<T>(present_size, T(0.0f)));
  tester.AddOutput<T>("present_value", {batch_size, kv_num_heads, total_seq_len, head_size},
                      std::vector<T>(present_size, T(0.0f)));

  tester.SetOutputTolerance(1e6f);  // outputs are compared explicitly by the caller

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(MakeExecutionProviderForGqaTest(target_ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  auto fetches = tester.GetFetches();
  const T* out = fetches[0].Get<Tensor>().Data<T>();
  std::vector<float> result(output_size);
  for (int i = 0; i < output_size; ++i) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      result[i] = out[i].ToFloat();
    } else {
      result[i] = out[i];
    }
  }
  return result;
}

TEST(GroupQueryAttentionTest, CudaAttentionBiasParityVsCpu) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  constexpr int batch_size = 2;
  constexpr int seq_len = 3;
  constexpr int num_heads = 4;
  constexpr int kv_num_heads = 2;
  constexpr int head_size = 8;
  const int hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;

  std::vector<float> query(batch_size * seq_len * hidden_size);
  std::vector<float> key(batch_size * seq_len * kv_hidden_size);
  std::vector<float> value(batch_size * seq_len * kv_hidden_size);
  for (size_t i = 0; i < query.size(); ++i) query[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < key.size(); ++i) key[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < value.size(); ++i) value[i] = 0.3f * static_cast<float>(i % 3 + 1);

  struct BiasShape {
    int dim0;
    int dim1;
    const char* label;
  };
  const std::vector<BiasShape> shapes = {
      {batch_size, 1, "batch_head_broadcast"},  // dim0=false, dim1=true (default)
      {1, 1, "batch_broadcast"},                // dim0=true
      {batch_size, num_heads, "per_head"},      // dim1=false
  };

  for (const auto& shape : shapes) {
    std::vector<float> bias(static_cast<size_t>(shape.dim0) * shape.dim1 * seq_len * seq_len);
    for (size_t i = 0; i < bias.size(); ++i) {
      bias[i] = 0.5f * std::sin(0.7f * static_cast<float>(i));
    }

    auto cuda_output = RunGQAPromptWithBias<MLFloat16>(
        batch_size, seq_len, num_heads, kv_num_heads, head_size, shape.dim0, shape.dim1,
        query, key, value, bias, GqaTargetEp::kCuda);
    auto cpu_output = RunGQAPromptWithBias<float>(
        batch_size, seq_len, num_heads, kv_num_heads, head_size, shape.dim0, shape.dim1,
        query, key, value, bias, GqaTargetEp::kCpu);

    ExpectOutputsMatch(cuda_output, cpu_output, 0.02f, shape.label);
  }
}

#ifdef USE_WEBGPU
// WebGPU graph capture test for kv_empty (Gemma4 shared-KV) layers.
//
// When graph capture is enabled, total_seqlen is GPU-resident and
// PrepareIndirectDispatchProgram must compute dispatch sizes on GPU.
// This test exercises the full ORT graph capture path by allocating all
// inputs as GPU tensors via IOBinding, running capture then replay, and
// verifying the replay output matches the CPU reference.
TEST(GroupQueryAttentionTest, WebGPU_SharedKV_IndirectDispatchForGraphCapture) {
  constexpr int batch_size = 1;
  constexpr int q_seq_len = 1;
  constexpr int past_seq_len = 32;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  // Build a GQA model directly using the Graph API so graph inputs are
  // properly declared (OpTester::BuildModel doesn't call graph.Resolve or
  // SetInputs, producing a proto that InferenceSession::Load rejects).
  std::unique_ptr<onnxruntime::Model> p_model;
  {
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[kOnnxDomain] = 17;
    domain_to_version[kMSDomain] = 1;
    p_model = std::make_unique<onnxruntime::Model>(
        "gqa_gc_test", true, ModelMetaData(), PathString(),
        IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
        std::vector<ONNX_NAMESPACE::FunctionProto>{},
        DefaultLoggingManager().DefaultLogger(),
        ModelOptions(true, true));
    onnxruntime::Graph& graph = p_model->MainGraph();

    ONNX_NAMESPACE::TypeProto tp_float, tp_int32;
    tp_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tp_int32.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

    std::vector<onnxruntime::NodeArg*> in_defs = {
        &graph.GetOrCreateNodeArg("query", &tp_float),
        &graph.GetOrCreateNodeArg("key", &tp_float),
        &graph.GetOrCreateNodeArg("value", &tp_float),
        &graph.GetOrCreateNodeArg("past_key", &tp_float),
        &graph.GetOrCreateNodeArg("past_value", &tp_float),
        &graph.GetOrCreateNodeArg("seqlens_k", &tp_int32),
        &graph.GetOrCreateNodeArg("total_sequence_length", &tp_int32),
        // optional inputs 8-12 are absent — use empty NodeArgs
        &graph.GetOrCreateNodeArg("", nullptr),
        &graph.GetOrCreateNodeArg("", nullptr),
        &graph.GetOrCreateNodeArg("", nullptr),
        &graph.GetOrCreateNodeArg("", nullptr),
        &graph.GetOrCreateNodeArg("", nullptr),
    };
    std::vector<onnxruntime::NodeArg*> out_defs = {
        &graph.GetOrCreateNodeArg("output", &tp_float),
    };

    auto& node = graph.AddNode("gqa", "GroupQueryAttention", "GQA",
                               in_defs, out_defs, nullptr, kMSDomain);
    node.AddAttribute("num_heads", static_cast<int64_t>(num_heads));
    node.AddAttribute("kv_num_heads", static_cast<int64_t>(kv_num_heads));

    ORT_THROW_IF_ERROR(graph.Resolve());
  }

  std::string model_data;
  p_model->ToProto().SerializeToString(&model_data);

  // Create InferenceSession with graph capture EP.
  SessionOptions so;
  InferenceSession session{so, GetEnvironment()};
  ConfigOptions config_options{};
  ORT_THROW_IF_ERROR(config_options.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                                   webgpu::options::kEnableGraphCapture_ON));
  auto webgpu_ep = WebGpuExecutionProviderWithOptions(config_options);
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  IExecutionProvider* ep_ptr = webgpu_ep.get();
  ORT_THROW_IF_ERROR(session.RegisterExecutionProvider(std::move(webgpu_ep)));
  std::istringstream model_stream(model_data);
  ORT_THROW_IF_ERROR(session.Load(model_stream));
  ORT_THROW_IF_ERROR(session.Initialize());

  // Get GPU allocator from session.
  OrtMemoryInfo gpu_mem_info(WEBGPU_BUFFER, OrtAllocatorType::OrtDeviceAllocator,
                             OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0));
  auto gpu_alloc = session.GetAllocator(gpu_mem_info);
  AllocatorPtr cpu_alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  // Capture-run input data (set A).
  std::vector<float> query_data(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data.size(); i++) query_data[i] = 0.1f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_key_data.size(); i++) past_key_data[i] = 0.2f * static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < past_value_data.size(); i++) past_value_data[i] = 0.3f * static_cast<float>(i % 3 + 1);

  // Replay-run input data (set B) — different values to verify replay actually uses new data.
  std::vector<float> query_data2(batch_size * q_seq_len * hidden_size);
  std::vector<float> past_key_data2(batch_size * kv_num_heads * past_seq_len * head_size);
  std::vector<float> past_value_data2(batch_size * kv_num_heads * past_seq_len * head_size);
  for (size_t i = 0; i < query_data2.size(); i++) query_data2[i] = 0.5f * static_cast<float>(i % 11 + 1);
  for (size_t i = 0; i < past_key_data2.size(); i++) past_key_data2[i] = 0.4f * static_cast<float>(i % 7 + 1);
  for (size_t i = 0; i < past_value_data2.size(); i++) past_value_data2[i] = 0.6f * static_cast<float>(i % 4 + 1);

  // Helper: create GPU OrtValue by copying from CPU data (nullptr = zero-size tensor).
  auto make_gpu_value = [&](const void* data, MLDataType dtype, TensorShape shape) -> OrtValue {
    Tensor gpu_tensor(dtype, shape, gpu_alloc);
    if (data != nullptr && shape.Size() > 0) {
      Tensor cpu_tensor(dtype, shape, const_cast<void*>(data), cpu_alloc->Info());
      ORT_THROW_IF_ERROR(ep_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    }
    OrtValue val;
    Tensor::InitOrtValue(std::move(gpu_tensor), val);
    return val;
  };

  // Helper: overwrite an existing GPU tensor's data in-place from CPU (same buffer, new values).
  // Under graph capture the WGPUBuffer pointer is baked in; rebinding is not allowed.
  auto update_gpu_value = [&](const OrtValue& gpu_val, const void* data, MLDataType dtype, TensorShape shape) {
    if (shape.Size() > 0) {
      Tensor cpu_tensor(dtype, shape, const_cast<void*>(data), cpu_alloc->Info());
      ORT_THROW_IF_ERROR(ep_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, const_cast<Tensor&>(gpu_val.Get<Tensor>())));
    }
  };

  auto q_val = make_gpu_value(query_data.data(), DataTypeImpl::GetType<float>(),
                              TensorShape{batch_size, q_seq_len, hidden_size});
  auto key_val = make_gpu_value(nullptr, DataTypeImpl::GetType<float>(),
                                TensorShape{batch_size, 0, kv_hidden_size});
  auto value_val = make_gpu_value(nullptr, DataTypeImpl::GetType<float>(),
                                  TensorShape{batch_size, 0, kv_hidden_size});
  auto pk_val = make_gpu_value(past_key_data.data(), DataTypeImpl::GetType<float>(),
                               TensorShape{batch_size, kv_num_heads, past_seq_len, head_size});
  auto pv_val = make_gpu_value(past_value_data.data(), DataTypeImpl::GetType<float>(),
                               TensorShape{batch_size, kv_num_heads, past_seq_len, head_size});
  std::vector<int32_t> seqlens_k_data = {past_seq_len - 1};
  auto sk_val = make_gpu_value(seqlens_k_data.data(), DataTypeImpl::GetType<int32_t>(),
                               TensorShape{batch_size});
  std::vector<int32_t> total_seqlen_data = {past_seq_len};
  auto ts_val = make_gpu_value(total_seqlen_data.data(), DataTypeImpl::GetType<int32_t>(),
                               TensorShape{1});

  // Preallocate GPU output.
  Tensor gpu_out_tensor(DataTypeImpl::GetType<float>(),
                        TensorShape{batch_size, q_seq_len, hidden_size}, gpu_alloc);
  OrtValue out_val;
  Tensor::InitOrtValue(std::move(gpu_out_tensor), out_val);

  // Bind inputs and output.
  std::unique_ptr<IOBinding> io_binding;
  ORT_THROW_IF_ERROR(session.NewIOBinding(&io_binding));
  ORT_THROW_IF_ERROR(io_binding->BindInput("query", q_val));
  ORT_THROW_IF_ERROR(io_binding->BindInput("key", key_val));
  ORT_THROW_IF_ERROR(io_binding->BindInput("value", value_val));
  ORT_THROW_IF_ERROR(io_binding->BindInput("past_key", pk_val));
  ORT_THROW_IF_ERROR(io_binding->BindInput("past_value", pv_val));
  ORT_THROW_IF_ERROR(io_binding->BindInput("seqlens_k", sk_val));
  ORT_THROW_IF_ERROR(io_binding->BindInput("total_sequence_length", ts_val));
  ORT_THROW_IF_ERROR(io_binding->BindOutput("output", out_val));
  ORT_THROW_IF_ERROR(io_binding->SynchronizeInputs());

  // Run 1: capture.
  RunOptions run_options;
  ORT_THROW_IF_ERROR(session.Run(run_options, *io_binding));

  // Run 2: replay with different inputs (set B) written into the same GPU buffers.
  // Rebinding is not allowed under graph capture — the WGPUBuffer pointers are baked
  // in at capture time, so new data must be copied into the existing buffers.
  update_gpu_value(q_val, query_data2.data(), DataTypeImpl::GetType<float>(),
                   TensorShape{batch_size, q_seq_len, hidden_size});
  update_gpu_value(pk_val, past_key_data2.data(), DataTypeImpl::GetType<float>(),
                   TensorShape{batch_size, kv_num_heads, past_seq_len, head_size});
  update_gpu_value(pv_val, past_value_data2.data(), DataTypeImpl::GetType<float>(),
                   TensorShape{batch_size, kv_num_heads, past_seq_len, head_size});
  ORT_THROW_IF_ERROR(session.Run(run_options, *io_binding));

  // Copy replay output GPU -> CPU and compare against CPU reference for set B.
  const int output_size = batch_size * q_seq_len * hidden_size;
  auto& gpu_out = io_binding->GetOutputs()[0].Get<Tensor>();
  Tensor cpu_out_tensor(DataTypeImpl::GetType<float>(), gpu_out.Shape(), cpu_alloc);
  ORT_THROW_IF_ERROR(ep_ptr->GetDataTransfer()->CopyTensor(gpu_out, cpu_out_tensor));
  std::vector<float> webgpu_output(cpu_out_tensor.Data<float>(),
                                   cpu_out_tensor.Data<float>() + output_size);

  auto cpu_output = RunGQASharedKV(
      batch_size, q_seq_len, past_seq_len, query_data2, past_key_data2, past_value_data2,
      num_heads, kv_num_heads, head_size, GqaTargetEp::kCpu);

  ExpectOutputsMatch(webgpu_output, cpu_output, 0.05f, "SharedKV_IndirectDispatchForGraphCapture_vs_CPU");
}
#endif  // USE_WEBGPU

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
    GqaTargetEp target_ep = GqaTargetEp::kCpu,
    bool smooth_softmax = false) {
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
  if (smooth_softmax) {
    // smooth_softmax disqualifies the WebGPU FlashAttention path via the outer
    // gating in GroupQueryAttention::ComputeInternal, routing this case through
    // ApplyAttention instead.
    tester.AddAttribute<int64_t>("smooth_softmax", static_cast<int64_t>(1));
  }

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
static void RunBatchedRightPaddedRotaryPrefillForEP(GqaTargetEp target_ep,
                                                    const std::vector<int>& real_lens = {4, 2, 6},
                                                    bool smooth_softmax = false) {
  constexpr int batch_size = 3;
  constexpr int num_heads = 4;
  constexpr int kv_num_heads = 2;
  constexpr int head_size = 16;  // multiple of 4 for FlashAttention gate; rotary half = 8
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int qkv_hidden = hidden_size + 2 * kv_hidden_size;

  // Per-batch real prompt lengths; max = sequence_length (right-padding extends
  // shorter batches up to this length). The bug only manifests when at least
  // one batch is shorter than sequence_length.
  ASSERT_EQ(static_cast<int>(real_lens.size()), batch_size);
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
        packed_single, target_ep, smooth_softmax);
  }

  // Now run all batches together with right-padding.
  std::vector<int32_t> seqlens_k_data(batch_size);
  for (int b = 0; b < batch_size; ++b) {
    seqlens_k_data[b] = static_cast<int32_t>(real_lens[b] - 1);
  }
  const auto batched_output = RunGQAPackedQKVRotaryPrefill(
      batch_size, sequence_length, num_heads, kv_num_heads, head_size,
      seqlens_k_data, packed_batched, target_ep, smooth_softmax);

  // Guard the regression deterministically: every element of the batched output
  // (including padding rows) must be finite. The CPU root cause is uninitialized
  // attention-probs memory, so a NaN/Inf at any padding position would otherwise
  // depend on the allocator returning non-zero pages.
  for (size_t i = 0; i < batched_output.size(); ++i) {
    ASSERT_TRUE(std::isfinite(batched_output[i]))
        << "non-finite value at index " << i << " in batched GQA output";
  }

  // Each batch's real-last-token output (used to predict next token) must match
  // its single-prompt reference. Tolerance is loose enough for fp16 rounding,
  // tight enough to catch the right-padding regressions across EPs:
  //   - CPU: uninitialized attention-probs reads at padding positions -> NaN.
  //   - WebGPU: u32 underflow on rotary past_seqlen -> out-of-range cos/sin
  //     index -> garbage Q/K (see PR #29002).
  // Both manifest as NaN/Inf or values differing by orders of magnitude.
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

TEST(GroupQueryAttentionTest, BatchedRightPaddedRotaryPrefill_CPU) {
  RunBatchedRightPaddedRotaryPrefillForEP(GqaTargetEp::kCpu);
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

// Same property as BatchedRightPaddedRotaryPrefill_WebGPU, but with per-batch
// real_lens whose max crosses the prefill threshold (sequence_length >= 32) so
// the WebGPU EP picks FlashAttentionProgram (single-kernel prefill path with
// subgroup shuffles) instead of the split-reduce decode path. This exercises
// the prefill flash-attention kernel under right-padded batches with do_rotary,
// which is the path used by Phi-style models during batched prefill.
TEST(GroupQueryAttentionTest, BatchedRightPaddedRotaryPrefillFlashAttention_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  // sequence_length = max(real_lens) = 33 > 32 -> FlashAttentionProgram path.
  // Mixed shorter batches (12, 20) ensure right-padding is non-trivial.
  RunBatchedRightPaddedRotaryPrefillForEP(GqaTargetEp::kWebGpu, {20, 12, 33});
}

// Stress the FlashAttention prefill path with a per-batch spread that exceeds
// the indirect-dispatch tile size (64). batch 0 has the SHORTEST real length;
// batch 2 has the LONGEST. This is the data pattern that would surface the
// indirect-dispatch undersizing bug when graph capture is enabled (where the
// dispatch grid is sized from a GPU buffer rather than the host scalar).
// OpTester does not toggle graph capture, so this test exercises the new
// total_sequence_length_input shader plumbing on the non-graph-capture path;
// the graph-capture path is covered end-to-end by phi4-graph-prune verification.
TEST(GroupQueryAttentionTest, BatchedRightPaddedRotaryPrefillFlashAttentionLargeSpread_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  // spread = 96 - 20 = 76 > tile_size(64), batch 0 is not the max.
  RunBatchedRightPaddedRotaryPrefillForEP(GqaTargetEp::kWebGpu, {20, 12, 96});
}

// Same property as BatchedRightPaddedRotaryPrefill_WebGPU, but with
// smooth_softmax=1 so the WebGPU EP bypasses CanApplyFlashAttention and routes
// through ApplyAttention (non-flash path). Covers right-padded batched prefill
// on the non-flash attention path (used by e.g. Phi-4 attention variants).
TEST(GroupQueryAttentionTest, BatchedRightPaddedRotaryPrefillNonFlashAttention_WebGPU) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  RunBatchedRightPaddedRotaryPrefillForEP(GqaTargetEp::kWebGpu, {4, 2, 6}, /*smooth_softmax=*/true);
}

#ifdef USE_WEBGPU
// ---------------------------------------------------------------------------
// WebGPU graph-capture and KV-cache quantization tests.
// Tests exercise static-cache preprocessing, Q4 TurboQuant, and Q8 block quantization in
// GroupQueryAttention + FlashAttention.
// The helpers below reference webgpu::options::* constants, which are only
// available when USE_WEBGPU is defined; guard the whole section so non-WebGPU
// test builds (CPU/CUDA) still compile the rest of this file.
// ---------------------------------------------------------------------------

static std::unique_ptr<IExecutionProvider> WebGpuEPForGqaOptions(bool enable_graph_capture,
                                                                 uint32_t kv_cache_quant_bits,
                                                                 uint32_t multi_rotary_cache_concat_offset = 0) {
  ORT_ENFORCE(kv_cache_quant_bits == 0 || kv_cache_quant_bits == 4 || kv_cache_quant_bits == 8,
              "KV cache quantization bit width must be 0, 4, or 8, got ", kv_cache_quant_bits);
  ConfigOptions config_options{};
  ORT_THROW_IF_ERROR(config_options.AddConfigEntry(webgpu::options::kStorageBufferCacheMode,
                                                   webgpu::options::kBufferCacheMode_Disabled));
  if (kv_cache_quant_bits != 0) {
    const char* option_value = kv_cache_quant_bits == 8
                                   ? webgpu::options::kKvCacheQuantizationBits_8Bit
                                   : webgpu::options::kKvCacheQuantizationBits_4Bit;
    ORT_THROW_IF_ERROR(config_options.AddConfigEntry(webgpu::options::kKvCacheQuantizationBits,
                                                     option_value));
  }
  if (enable_graph_capture) {
    ORT_THROW_IF_ERROR(config_options.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                                     webgpu::options::kEnableGraphCapture_ON));
  }
  if (multi_rotary_cache_concat_offset > 0) {
    ORT_THROW_IF_ERROR(config_options.AddConfigEntry(
        webgpu::options::kMultiRotaryCacheConcatOffset,
        std::to_string(multi_rotary_cache_concat_offset).c_str()));
  }
  return WebGpuExecutionProviderWithOptions(config_options);
}

static std::unique_ptr<IExecutionProvider> WebGpuEPWithKVCacheQuantization(
    uint32_t bit_width,
    bool enable_graph_capture = false) {
  return WebGpuEPForGqaOptions(enable_graph_capture, bit_width);
}

// Graph capture requires the indirect-dispatch dimensions to be prepared on the GPU.
// Verify that static-cache preprocessing uses the batch-wide total_sequence_length input
// instead of deriving the dispatch width from batch 0's (possibly shorter) seqlens_k value. The
// four-token input also makes batch 0's logical total shorter than kv_sequence_length,
// covering the right-padding underflow clamp with true static-cache aliasing.
static void RunIndirectDispatchGraphCapture(bool do_rotary,
                                            uint32_t kv_cache_quant_bits,
                                            bool enable_multi_rotary_cache) {
  constexpr int batch_size = 2;
  constexpr int sequence_length = 4;
  constexpr int short_total_sequence_length = 2;
  constexpr int cache_sequence_length = 130;  // Three 64-token attention tiles.
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int packed_hidden_size = hidden_size + 2 * kv_hidden_size;
  constexpr uint32_t multi_rotary_cache_concat_offset = 4;
  const int cache_head_size = kv_cache_quant_bits == 0
                                  ? head_size
                                  : (head_size * static_cast<int>(kv_cache_quant_bits) + 32) / 32;

  std::unique_ptr<onnxruntime::Model> model;
  {
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[kOnnxDomain] = 17;
    domain_to_version[kMSDomain] = 1;
    model = std::make_unique<onnxruntime::Model>(
        do_rotary ? "tq_gc_rotary_test" : "tq_gc_test", true, ModelMetaData(), PathString(),
        IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
        std::vector<ONNX_NAMESPACE::FunctionProto>{},
        DefaultLoggingManager().DefaultLogger(),
        ModelOptions(true, true));
    onnxruntime::Graph& graph = model->MainGraph();

    ONNX_NAMESPACE::TypeProto tp_float, tp_int32;
    tp_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tp_int32.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

    auto& empty = graph.GetOrCreateNodeArg("", nullptr);
    std::vector<onnxruntime::NodeArg*> inputs = {
        &graph.GetOrCreateNodeArg("query", &tp_float),
        do_rotary ? &empty : &graph.GetOrCreateNodeArg("key", &tp_float),
        do_rotary ? &empty : &graph.GetOrCreateNodeArg("value", &tp_float),
        &graph.GetOrCreateNodeArg("past_key", &tp_float),
        &graph.GetOrCreateNodeArg("past_value", &tp_float),
        &graph.GetOrCreateNodeArg("seqlens_k", &tp_int32),
        &graph.GetOrCreateNodeArg("total_sequence_length", &tp_int32),
        do_rotary ? &graph.GetOrCreateNodeArg("cos_cache", &tp_float) : &empty,
        do_rotary ? &graph.GetOrCreateNodeArg("sin_cache", &tp_float) : &empty,
        &empty,  // position_ids
        &empty,  // attention_bias
        &empty,  // head_sink
    };
    std::vector<onnxruntime::NodeArg*> outputs = {
        &graph.GetOrCreateNodeArg("output", &tp_float),
        &graph.GetOrCreateNodeArg("present_key", &tp_float),
        &graph.GetOrCreateNodeArg("present_value", &tp_float),
    };

    auto& node = graph.AddNode("gqa", "GroupQueryAttention", "GQA", inputs, outputs, nullptr, kMSDomain);
    node.AddAttribute("num_heads", static_cast<int64_t>(num_heads));
    node.AddAttribute("kv_num_heads", static_cast<int64_t>(kv_num_heads));
    if (do_rotary) {
      node.AddAttribute("do_rotary", int64_t{1});
    }
    ORT_THROW_IF_ERROR(graph.Resolve());
  }

  std::string model_data;
  model->ToProto().SerializeToString(&model_data);

  SessionOptions session_options;
  InferenceSession session{session_options, GetEnvironment()};
  auto webgpu_ep = WebGpuEPForGqaOptions(
      /*enable_graph_capture=*/true,
      kv_cache_quant_bits,
      enable_multi_rotary_cache ? multi_rotary_cache_concat_offset : 0);
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  IExecutionProvider* ep = webgpu_ep.get();
  ORT_THROW_IF_ERROR(session.RegisterExecutionProvider(std::move(webgpu_ep)));
  std::istringstream model_stream(model_data);
  ORT_THROW_IF_ERROR(session.Load(model_stream));
  ORT_THROW_IF_ERROR(session.Initialize());

  OrtMemoryInfo gpu_memory_info(WEBGPU_BUFFER, OrtAllocatorType::OrtDeviceAllocator,
                                OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                          OrtDevice::VendorIds::NONE, 0));
  auto gpu_allocator = session.GetAllocator(gpu_memory_info);
  AllocatorPtr cpu_allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  auto make_data = [](size_t size, float scale, int period) {
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = scale * static_cast<float>(i % static_cast<size_t>(period) + 1);
    }
    return data;
  };
  auto swap_batches = [](const std::vector<float>& input) {
    const size_t elements_per_batch = input.size() / batch_size;
    std::vector<float> output(input.size());
    std::copy_n(input.data() + elements_per_batch, elements_per_batch, output.data());
    std::copy_n(input.data(), elements_per_batch, output.data() + elements_per_batch);
    return output;
  };

  const int query_width = do_rotary ? packed_hidden_size : hidden_size;
  auto query_data = make_data(batch_size * sequence_length * query_width, 0.01f, 31);
  auto key_data = make_data(batch_size * sequence_length * kv_hidden_size, 0.02f, 29);
  auto value_data = make_data(batch_size * sequence_length * kv_hidden_size, 0.03f, 23);
  auto past_key_data = make_data(batch_size * kv_num_heads * cache_sequence_length * cache_head_size,
                                 0.001f, 19);
  auto past_value_data = make_data(batch_size * kv_num_heads * cache_sequence_length * cache_head_size,
                                   0.002f, 17);
  auto query_data_swapped = swap_batches(query_data);
  auto key_data_swapped = swap_batches(key_data);
  auto value_data_swapped = swap_batches(value_data);
  auto past_key_data_swapped = swap_batches(past_key_data);
  auto past_value_data_swapped = swap_batches(past_value_data);

  constexpr int half_rotary_dim = head_size / 2;
  const int large_rotary_cache_length = cache_sequence_length + 1;
  auto cos_cache_data = make_data(large_rotary_cache_length * half_rotary_dim, 0.001f, 37);
  auto sin_cache_data = make_data(large_rotary_cache_length * half_rotary_dim, 0.001f, 41);
  int rotary_cache_length = large_rotary_cache_length;
  if (enable_multi_rotary_cache) {
    const size_t small_cache_size = multi_rotary_cache_concat_offset * half_rotary_dim;
    cos_cache_data.insert(cos_cache_data.begin(), small_cache_size,
                          std::numeric_limits<float>::quiet_NaN());
    sin_cache_data.insert(sin_cache_data.begin(), small_cache_size,
                          std::numeric_limits<float>::quiet_NaN());
    rotary_cache_length += multi_rotary_cache_concat_offset;
  }

  auto make_gpu_value = [&](const void* data, MLDataType data_type, const TensorShape& shape) {
    Tensor gpu_tensor(data_type, shape, gpu_allocator);
    Tensor cpu_tensor(data_type, shape, const_cast<void*>(data), cpu_allocator->Info());
    ORT_THROW_IF_ERROR(ep->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue value;
    Tensor::InitOrtValue(std::move(gpu_tensor), value);
    return value;
  };
  auto update_gpu_value = [&](const OrtValue& gpu_value, const void* data, MLDataType data_type,
                              const TensorShape& shape) {
    Tensor cpu_tensor(data_type, shape, const_cast<void*>(data), cpu_allocator->Info());
    ORT_THROW_IF_ERROR(ep->GetDataTransfer()->CopyTensor(
        cpu_tensor, const_cast<Tensor&>(gpu_value.Get<Tensor>())));
  };

  const TensorShape query_shape{batch_size, sequence_length, query_width};
  const TensorShape kv_shape{batch_size, sequence_length, kv_hidden_size};
  const TensorShape cache_shape{batch_size, kv_num_heads, cache_sequence_length, cache_head_size};
  const TensorShape seqlens_shape{batch_size};
  const TensorShape total_sequence_length_shape{1};
  const TensorShape rotary_cache_shape{rotary_cache_length, half_rotary_dim};
  auto query_value = make_gpu_value(query_data.data(), DataTypeImpl::GetType<float>(), query_shape);
  auto key_value = make_gpu_value(key_data.data(), DataTypeImpl::GetType<float>(), kv_shape);
  auto value_value = make_gpu_value(value_data.data(), DataTypeImpl::GetType<float>(), kv_shape);
  auto past_key_value = make_gpu_value(past_key_data.data(), DataTypeImpl::GetType<float>(), cache_shape);
  auto past_value_value = make_gpu_value(past_value_data.data(), DataTypeImpl::GetType<float>(), cache_shape);
  std::vector<int32_t> seqlens_data{short_total_sequence_length - 1, cache_sequence_length - 1};
  auto seqlens_value = make_gpu_value(seqlens_data.data(), DataTypeImpl::GetType<int32_t>(), seqlens_shape);
  std::vector<int32_t> total_sequence_length_data{cache_sequence_length};
  auto total_sequence_length_value = make_gpu_value(total_sequence_length_data.data(),
                                                    DataTypeImpl::GetType<int32_t>(),
                                                    total_sequence_length_shape);
  auto cos_cache_value = make_gpu_value(cos_cache_data.data(), DataTypeImpl::GetType<float>(), rotary_cache_shape);
  auto sin_cache_value = make_gpu_value(sin_cache_data.data(), DataTypeImpl::GetType<float>(), rotary_cache_shape);

  Tensor output_tensor(DataTypeImpl::GetType<float>(),
                       TensorShape{batch_size, sequence_length, hidden_size}, gpu_allocator);
  OrtValue output_value;
  Tensor::InitOrtValue(std::move(output_tensor), output_value);

  std::unique_ptr<IOBinding> io_binding;
  ORT_THROW_IF_ERROR(session.NewIOBinding(&io_binding));
  ORT_THROW_IF_ERROR(io_binding->BindInput("query", query_value));
  if (!do_rotary) {
    ORT_THROW_IF_ERROR(io_binding->BindInput("key", key_value));
    ORT_THROW_IF_ERROR(io_binding->BindInput("value", value_value));
  }
  ORT_THROW_IF_ERROR(io_binding->BindInput("past_key", past_key_value));
  ORT_THROW_IF_ERROR(io_binding->BindInput("past_value", past_value_value));
  ORT_THROW_IF_ERROR(io_binding->BindInput("seqlens_k", seqlens_value));
  ORT_THROW_IF_ERROR(io_binding->BindInput("total_sequence_length", total_sequence_length_value));
  if (do_rotary) {
    ORT_THROW_IF_ERROR(io_binding->BindInput("cos_cache", cos_cache_value));
    ORT_THROW_IF_ERROR(io_binding->BindInput("sin_cache", sin_cache_value));
  }
  ORT_THROW_IF_ERROR(io_binding->BindOutput("output", output_value));
  // Alias past and present buffers so packed rotary uses the static-cache fused path.
  ORT_THROW_IF_ERROR(io_binding->BindOutput("present_key", past_key_value));
  ORT_THROW_IF_ERROR(io_binding->BindOutput("present_value", past_value_value));
  ORT_THROW_IF_ERROR(io_binding->SynchronizeInputs());

  auto read_output = [&]() {
    auto& gpu_output = io_binding->GetOutputs()[0].Get<Tensor>();
    Tensor cpu_output(DataTypeImpl::GetType<float>(), gpu_output.Shape(), cpu_allocator);
    ORT_THROW_IF_ERROR(ep->GetDataTransfer()->CopyTensor(gpu_output, cpu_output));
    return std::vector<float>(cpu_output.Data<float>(), cpu_output.Data<float>() + cpu_output.Shape().Size());
  };
  auto read_gpu_bytes = [&](const OrtValue& gpu_value) {
    const auto& gpu_tensor = gpu_value.Get<Tensor>();
    Tensor cpu_tensor(gpu_tensor.DataType(), gpu_tensor.Shape(), cpu_allocator);
    ORT_THROW_IF_ERROR(ep->GetDataTransfer()->CopyTensor(gpu_tensor, cpu_tensor));
    const auto* begin = static_cast<const uint8_t*>(cpu_tensor.DataRaw());
    return std::vector<uint8_t>(begin, begin + cpu_tensor.SizeInBytes());
  };

  RunOptions run_options;
  ORT_THROW_IF_ERROR(session.Run(run_options, *io_binding));
  auto first_output = read_output();

  // Batch 0 has only two logical tokens in a four-token input. TurboQuant static-cache
  // slots for its two padded tokens must retain their original contents. The standard
  // path currently writes padding slots, which is unrelated to cache-bank selection.
  auto expect_padding_unchanged = [&](const std::vector<uint8_t>& actual,
                                      const std::vector<float>& initial,
                                      const char* cache_name) {
    ASSERT_EQ(actual.size(), initial.size() * sizeof(float));
    const size_t bytes_per_token = kv_num_heads * cache_head_size * sizeof(float);
    const size_t padding_offset = short_total_sequence_length * bytes_per_token;
    const size_t padding_size = (sequence_length - short_total_sequence_length) * bytes_per_token;
    const auto* initial_bytes = reinterpret_cast<const uint8_t*>(initial.data());
    EXPECT_TRUE(std::equal(actual.begin() + padding_offset,
                           actual.begin() + padding_offset + padding_size,
                           initial_bytes + padding_offset))
        << cache_name << " padded static-cache slots were overwritten";
  };
  if (kv_cache_quant_bits != 0) {
    expect_padding_unchanged(read_gpu_bytes(past_key_value), past_key_data, "key");
    expect_padding_unchanged(read_gpu_bytes(past_value_value), past_value_data, "value");
  }
  if (kv_cache_quant_bits == 8 && do_rotary) {
    const auto key_bytes = read_gpu_bytes(past_key_value);
    const auto value_bytes = read_gpu_bytes(past_value_value);
    std::vector<uint32_t> key_words(key_bytes.size() / sizeof(uint32_t));
    std::vector<uint32_t> value_words(value_bytes.size() / sizeof(uint32_t));
    std::memcpy(key_words.data(), key_bytes.data(), key_bytes.size());
    std::memcpy(value_words.data(), value_bytes.data(), value_bytes.size());

    constexpr int batch = 0;
    constexpr int seq = 1;
    const size_t cache_base =
        ((batch * kv_num_heads) * cache_sequence_length + seq) * cache_head_size;
    float key_scale;
    float value_scale;
    std::memcpy(&key_scale, &key_words[cache_base], sizeof(key_scale));
    std::memcpy(&value_scale, &value_words[cache_base], sizeof(value_scale));
    ASSERT_GT(key_scale, 0.0f);
    ASSERT_GT(value_scale, 0.0f);

    const size_t packed_token_base =
        (batch * sequence_length + seq) * packed_hidden_size;
    for (int dim = 0; dim < head_size; ++dim) {
      const int rotary_dim = dim % half_rotary_dim;
      const float cos_value = cos_cache_data[seq * half_rotary_dim + rotary_dim];
      const float sin_value = sin_cache_data[seq * half_rotary_dim + rotary_dim];
      const float first = query_data[packed_token_base + hidden_size + rotary_dim];
      const float second =
          query_data[packed_token_base + hidden_size + rotary_dim + half_rotary_dim];
      const float expected_key = dim < half_rotary_dim
                                     ? first * cos_value - second * sin_value
                                     : first * sin_value + second * cos_value;
      const float expected_value =
          query_data[packed_token_base + hidden_size + kv_hidden_size + dim];
      const int shift = (dim % 4) * 8;
      const int key_quantized =
          static_cast<int>((key_words[cache_base + 1 + dim / 4] >> shift) & 0xffu) - 128;
      const int value_quantized =
          static_cast<int>((value_words[cache_base + 1 + dim / 4] >> shift) & 0xffu) - 128;
      EXPECT_NEAR(static_cast<float>(key_quantized) * key_scale, expected_key,
                  key_scale * 0.51f + 1e-6f);
      EXPECT_NEAR(static_cast<float>(value_quantized) * value_scale, expected_value,
                  value_scale * 0.51f + 1e-6f);
    }
  }

  update_gpu_value(query_value, query_data_swapped.data(), DataTypeImpl::GetType<float>(), query_shape);
  if (!do_rotary) {
    update_gpu_value(key_value, key_data_swapped.data(), DataTypeImpl::GetType<float>(), kv_shape);
    update_gpu_value(value_value, value_data_swapped.data(), DataTypeImpl::GetType<float>(), kv_shape);
  }
  update_gpu_value(past_key_value, past_key_data_swapped.data(), DataTypeImpl::GetType<float>(), cache_shape);
  update_gpu_value(past_value_value, past_value_data_swapped.data(), DataTypeImpl::GetType<float>(), cache_shape);
  seqlens_data = {cache_sequence_length - 1, short_total_sequence_length - 1};
  update_gpu_value(seqlens_value, seqlens_data.data(), DataTypeImpl::GetType<int32_t>(), seqlens_shape);
  ORT_THROW_IF_ERROR(session.Run(run_options, *io_binding));
  auto second_output = read_output();

  ASSERT_EQ(first_output.size(), second_output.size());
  EXPECT_TRUE(std::all_of(first_output.begin(), first_output.end(),
                          [](float value) { return std::isfinite(value); }))
      << "first graph-capture output contains a non-finite value";
  EXPECT_TRUE(std::all_of(second_output.begin(), second_output.end(),
                          [](float value) { return std::isfinite(value); }))
      << "second graph-capture output contains a non-finite value";
  constexpr size_t output_elements_per_batch = sequence_length * hidden_size;
  for (int second_batch = 0; second_batch < batch_size; ++second_batch) {
    const int first_batch = batch_size - 1 - second_batch;
    const auto* second_begin = second_output.data() + second_batch * output_elements_per_batch;
    const auto* first_begin = first_output.data() + first_batch * output_elements_per_batch;
    EXPECT_TRUE(std::equal(second_begin, second_begin + output_elements_per_batch, first_begin))
        << "indirect-dispatch output mismatch after swapping batch " << first_batch
        << " into slot " << second_batch;
  }
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_IndirectDispatch_UsesGlobalLength_NoRotary) {
  RunIndirectDispatchGraphCapture(/*do_rotary=*/false,
                                  /*kv_cache_quant_bits=*/4,
                                  /*enable_multi_rotary_cache=*/false);
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_IndirectDispatch_UsesGlobalLength_Rotary) {
  RunIndirectDispatchGraphCapture(/*do_rotary=*/true,
                                  /*kv_cache_quant_bits=*/4,
                                  /*enable_multi_rotary_cache=*/false);
}

TEST(GroupQueryAttentionTest, WebGPU_IndirectDispatch_MultiRotaryCache_UsesGlobalLength) {
  RunIndirectDispatchGraphCapture(/*do_rotary=*/true,
                                  /*kv_cache_quant_bits=*/0,
                                  /*enable_multi_rotary_cache=*/true);
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_IndirectDispatch_MultiRotaryCache_UsesGlobalLength) {
  RunIndirectDispatchGraphCapture(/*do_rotary=*/true,
                                  /*kv_cache_quant_bits=*/4,
                                  /*enable_multi_rotary_cache=*/true);
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_IndirectDispatch_FusedRotary) {
  RunIndirectDispatchGraphCapture(/*do_rotary=*/true,
                                  /*kv_cache_quant_bits=*/8,
                                  /*enable_multi_rotary_cache=*/false);
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_IndirectDispatch_NoRotary) {
  RunIndirectDispatchGraphCapture(/*do_rotary=*/false,
                                  /*kv_cache_quant_bits=*/8,
                                  /*enable_multi_rotary_cache=*/false);
}

// The non-static packed-QKV path uses split_packed_qkv_with_rotary_embedding.
// A batch-wide total above the concat offset must select the long RoPE cache for
// every batch, including batches whose individual total remains below the offset.
TEST(GroupQueryAttentionTest, WebGPU_MultiRotaryCache_UsesGlobalLength_NonStaticCache) {
  constexpr int batch_size = 2;
  constexpr int sequence_length = 1;
  constexpr int past_sequence_length = 4;
  constexpr int total_sequence_length = past_sequence_length + sequence_length;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int packed_hidden_size = hidden_size + 2 * kv_hidden_size;
  constexpr int half_rotary_dim = head_size / 2;
  constexpr uint32_t multi_rotary_cache_concat_offset = 4;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", num_heads);
  tester.AddAttribute<int64_t>("kv_num_heads", kv_num_heads);
  tester.AddAttribute<int64_t>("do_rotary", 1);

  std::vector<float> packed_qkv(batch_size * sequence_length * packed_hidden_size);
  for (size_t i = 0; i < packed_qkv.size(); ++i) {
    packed_qkv[i] = 0.01f * static_cast<float>(i % 17 + 1);
  }
  tester.AddInput<float>("query", {batch_size, sequence_length, packed_hidden_size}, packed_qkv);
  tester.AddOptionalInputEdge<float>();  // key
  tester.AddOptionalInputEdge<float>();  // value

  const int past_cache_size = batch_size * kv_num_heads * past_sequence_length * head_size;
  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_sequence_length, head_size},
                         std::vector<float>(past_cache_size, 0.05f));
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_sequence_length, head_size},
                         std::vector<float>(past_cache_size, 0.07f));

  // Per-batch totals are {3, 5}; only the batch-global total crosses offset 4.
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {2, 4});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length}, /*is_initializer=*/true);

  const int large_cache_length = total_sequence_length;
  const size_t small_cache_size = multi_rotary_cache_concat_offset * half_rotary_dim;
  std::vector<float> cos_cache(small_cache_size, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> sin_cache(small_cache_size, std::numeric_limits<float>::quiet_NaN());
  cos_cache.insert(cos_cache.end(), large_cache_length * half_rotary_dim, 1.0f);
  sin_cache.insert(sin_cache.end(), large_cache_length * half_rotary_dim, 0.0f);
  tester.AddInput<float>("cos_cache", {multi_rotary_cache_concat_offset + large_cache_length, half_rotary_dim},
                         cos_cache);
  tester.AddInput<float>("sin_cache", {multi_rotary_cache_concat_offset + large_cache_length, half_rotary_dim},
                         sin_cache);
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(batch_size * sequence_length * hidden_size, 0.0f));
  const int present_cache_size = batch_size * kv_num_heads * total_sequence_length * head_size;
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, total_sequence_length, head_size},
                          std::vector<float>(present_cache_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, total_sequence_length, head_size},
                          std::vector<float>(present_cache_size, 0.0f));
  tester.SetCustomOutputVerifier([](const std::vector<OrtValue>& fetches, const std::string&) {
    const auto& output = fetches[0].Get<Tensor>();
    const float* output_data = output.Data<float>();
    EXPECT_TRUE(std::all_of(output_data, output_data + output.Shape().Size(),
                            [](float value) { return std::isfinite(value); }))
        << "multi-RoPE output contains a non-finite value";
  });

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(WebGpuEPForGqaOptions(
      /*enable_graph_capture=*/false,
      /*kv_cache_quant_bits=*/0,
      multi_rotary_cache_concat_offset));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Helper to run a GQA op with TurboQuant enabled and separate Q/K/V with rotary.
// past_seq_len controls total KV cache depth; sequence_length controls prefill vs decode.
// Returns the output tensor data on success.
template <typename T = float>
static std::vector<float> RunGQATurboQuant(
    int batch_size,
    int sequence_length,
    int past_seq_len,
    int num_heads,
    int kv_num_heads,
    int head_size,
    bool do_rotary,
    bool is_packed_qkv,
    uint32_t bit_width = 4,
    OpTester::ExpectResult expect = OpTester::ExpectResult::kExpectSuccess,
    const std::string& expected_error = "") {
  const int hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int total_sequence_length = past_seq_len + sequence_length;

  const int kv_head_dim = ((head_size * static_cast<int>(bit_width) + 32) / 32) *
                          static_cast<int>(sizeof(uint32_t) / sizeof(T));

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  if (do_rotary) {
    tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));
  }

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  if (is_packed_qkv) {
    const int packed_dim = hidden_size + 2 * kv_hidden_size;
    std::vector<T> packed_data(batch_size * sequence_length * packed_dim);
    for (auto& v : packed_data) v = T(dist(rng));
    tester.AddInput<T>("query", {batch_size, sequence_length, packed_dim}, packed_data);
    tester.AddOptionalInputEdge<T>();  // key
    tester.AddOptionalInputEdge<T>();  // value
  } else {
    std::vector<T> query_data(batch_size * sequence_length * hidden_size);
    std::vector<T> key_data(batch_size * sequence_length * kv_hidden_size);
    std::vector<T> value_data(batch_size * sequence_length * kv_hidden_size);
    for (auto& v : query_data) v = T(dist(rng));
    for (auto& v : key_data) v = T(dist(rng));
    for (auto& v : value_data) v = T(dist(rng));
    tester.AddInput<T>("query", {batch_size, sequence_length, hidden_size}, query_data);
    tester.AddInput<T>("key", {batch_size, sequence_length, kv_hidden_size}, key_data);
    tester.AddInput<T>("value", {batch_size, sequence_length, kv_hidden_size}, value_data);
  }

  // Past KV is an element-typed payload whose raw bits are interpreted as packed u32 data.
  const int past_kv_size = batch_size * kv_num_heads * past_seq_len * kv_head_dim;
  std::vector<T> past_key_data(past_kv_size);
  std::vector<T> past_value_data(past_kv_size);
  for (auto& v : past_key_data) v = T(dist(rng));
  for (auto& v : past_value_data) v = T(dist(rng));
  tester.AddInput<T>("past_key", {batch_size, kv_num_heads, past_seq_len, kv_head_dim}, past_key_data);
  tester.AddInput<T>("past_value", {batch_size, kv_num_heads, past_seq_len, kv_head_dim}, past_value_data);

  std::vector<int32_t> tq_seqlens_k(batch_size, total_sequence_length - 1);
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, tq_seqlens_k);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length}, /*is_initializer=*/true);

  if (do_rotary) {
    const int max_seq_len = total_sequence_length + 8;
    const int half_rotary = head_size / 2;
    std::vector<T> cos_cache(max_seq_len * half_rotary);
    std::vector<T> sin_cache(max_seq_len * half_rotary);
    for (int pos = 0; pos < max_seq_len; ++pos) {
      for (int d = 0; d < half_rotary; ++d) {
        float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(d) / static_cast<float>(head_size));
        cos_cache[pos * half_rotary + d] = T(std::cos(static_cast<float>(pos) * freq));
        sin_cache[pos * half_rotary + d] = T(std::sin(static_cast<float>(pos) * freq));
      }
    }
    tester.AddInput<T>("cos_cache", {max_seq_len, half_rotary}, cos_cache);
    tester.AddInput<T>("sin_cache", {max_seq_len, half_rotary}, sin_cache);
  } else {
    tester.AddOptionalInputEdge<T>();  // cos_cache
    tester.AddOptionalInputEdge<T>();  // sin_cache
  }

  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<T>();        // attention_bias
  tester.AddOptionalInputEdge<T>();        // head_sink

  const int output_size = batch_size * sequence_length * hidden_size;
  tester.AddOutput<T>("output", {batch_size, sequence_length, hidden_size},
                      std::vector<T>(output_size, T(0.0f)));
  const int present_seq_len = total_sequence_length;
  const int present_size = batch_size * kv_num_heads * present_seq_len * kv_head_dim;
  tester.AddOutput<T>("present_key", {batch_size, kv_num_heads, present_seq_len, kv_head_dim},
                      std::vector<T>(present_size, T(0.0f)));
  tester.AddOutput<T>("present_value", {batch_size, kv_num_heads, present_seq_len, kv_head_dim},
                      std::vector<T>(present_size, T(0.0f)));

  // TurboQuant present_key/present_value are u32-packed quantized data reinterpreted as float.
  // Values can be astronomically large, so skip value checks via custom verifier.
  tester.SetOutputTolerance(1e6f);
  tester.SetCustomOutputVerifier([batch_size, sequence_length, hidden_size, kv_num_heads, present_seq_len, kv_head_dim](
                                     const std::vector<OrtValue>& fetches,
                                     const std::string& /*provider_type*/) {
    ASSERT_EQ(fetches.size(), 3u);
    ASSERT_TRUE(fetches[0].IsTensor());
    ASSERT_TRUE(fetches[1].IsTensor());
    ASSERT_TRUE(fetches[2].IsTensor());

    const auto& out_tensor = fetches[0].Get<Tensor>();
    EXPECT_EQ(out_tensor.Shape().NumDimensions(), 3);
    EXPECT_EQ(out_tensor.Shape()[0], batch_size);
    EXPECT_EQ(out_tensor.Shape()[1], sequence_length);
    EXPECT_EQ(out_tensor.Shape()[2], hidden_size);

    const auto& pk = fetches[1].Get<Tensor>();
    EXPECT_EQ(pk.Shape().NumDimensions(), 4);
    EXPECT_EQ(pk.Shape()[0], batch_size);
    EXPECT_EQ(pk.Shape()[1], kv_num_heads);
    EXPECT_EQ(pk.Shape()[2], present_seq_len);
    EXPECT_EQ(pk.Shape()[3], kv_head_dim);

    const auto& pv = fetches[2].Get<Tensor>();
    EXPECT_EQ(pv.Shape().NumDimensions(), 4);
    EXPECT_EQ(pv.Shape()[0], batch_size);
    EXPECT_EQ(pv.Shape()[1], kv_num_heads);
    EXPECT_EQ(pv.Shape()[2], present_seq_len);
    EXPECT_EQ(pv.Shape()[3], kv_head_dim);
  });

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  auto ep = WebGpuEPWithKVCacheQuantization(bit_width);
  if (!ep) {
    // GTEST_SKIP() cannot be used in a value-returning helper (it expands to a
    // void `return`). Callers already GTEST_SKIP() when the EP is unavailable, so
    // this branch is unreachable in practice; return empty to keep the helper valid.
    return {};
  }
  execution_providers.push_back(std::move(ep));
  tester.Run(expect, expected_error, {}, nullptr, &execution_providers);

  if (expect == OpTester::ExpectResult::kExpectSuccess) {
    auto fetches = tester.GetFetches();
    const T* out_data = fetches[0].Get<Tensor>().Data<T>();
    std::vector<float> output(output_size);
    std::transform(out_data, out_data + output_size, output.begin(), [](T value) {
      if constexpr (std::is_same_v<T, float>) {
        return value;
      } else {
        return value.ToFloat();
      }
    });
    return output;
  }
  return {};
}

static void ExpectFiniteNonzeroOutput(const std::vector<float>& output, const char* test_case) {
  EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); }))
      << test_case << " output contains a non-finite value";
  EXPECT_TRUE(std::any_of(output.begin(), output.end(), [](float value) { return value != 0.0f; }))
      << test_case << " output should not be all zeros";
}

// --- Error path: TurboQuant with smooth_softmax (non-flash attention) ---
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_RejectsNonFlashAttention) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;
  constexpr int past_seq_len = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int kv_head_dim = (head_size * 4 + 32) / 32;
  constexpr int total_sequence_length = past_seq_len + sequence_length;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  tester.AddAttribute<int64_t>("smooth_softmax", static_cast<int64_t>(1));  // Forces non-flash path

  std::vector<float> query_data(batch_size * sequence_length * hidden_size, 0.1f);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size, 0.1f);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size, 0.1f);
  tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size}, query_data);
  tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size}, key_data);
  tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size}, value_data);

  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * kv_head_dim, 0.0f);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * kv_head_dim, 0.0f);
  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_seq_len, kv_head_dim}, past_key_data);
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_seq_len, kv_head_dim}, past_value_data);

  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {total_sequence_length - 1});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length}, /*is_initializer=*/true);
  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  const int output_size = batch_size * sequence_length * hidden_size;
  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(output_size, 0.0f));
  const int present_size = batch_size * kv_num_heads * total_sequence_length * kv_head_dim;
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, total_sequence_length, kv_head_dim},
                          std::vector<float>(present_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, total_sequence_length, kv_head_dim},
                          std::vector<float>(present_size, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "KV cache quantization requires flash attention",
             {}, nullptr, &execution_providers);
}

// --- Error path: TurboQuant with invalid head_size (not power of 2) ---
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_RejectsNonPowerOf2HeadSize) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;
  constexpr int past_seq_len = 8;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 96;  // Not a power of 2
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  // Even with invalid head_size, we still need a valid past shape for the test to reach the check.
  // Use the would-be compressed dim (though it won't actually be used since the op errors out).
  constexpr int kv_head_dim = (head_size * 4 + 32) / 32;
  constexpr int total_sequence_length = past_seq_len + sequence_length;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));

  std::vector<float> query_data(batch_size * sequence_length * hidden_size, 0.1f);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size, 0.1f);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size, 0.1f);
  tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size}, query_data);
  tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size}, key_data);
  tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size}, value_data);

  std::vector<float> past_key_data(batch_size * kv_num_heads * past_seq_len * kv_head_dim, 0.0f);
  std::vector<float> past_value_data(batch_size * kv_num_heads * past_seq_len * kv_head_dim, 0.0f);
  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_seq_len, kv_head_dim}, past_key_data);
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_seq_len, kv_head_dim}, past_value_data);

  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {total_sequence_length - 1});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length}, /*is_initializer=*/true);
  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  const int output_size = batch_size * sequence_length * hidden_size;
  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(output_size, 0.0f));
  const int present_size = batch_size * kv_num_heads * total_sequence_length * kv_head_dim;
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, total_sequence_length, kv_head_dim},
                          std::vector<float>(present_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, total_sequence_length, kv_head_dim},
                          std::vector<float>(present_size, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "KV cache quantization requires head_size >= 8 and a power of 2",
             {}, nullptr, &execution_providers);
}

// --- Success paths: TurboQuant with flash attention at various K sizes ---
// K=1 (decode with minimal past), K=24 (moderate), K=128 (large)
// These exercise the split-reduce decode path (QKV + VxReduce kernels) for seq_len=1,
// and the prefill path (single FlashAttentionProgram kernel) for seq_len>1.

// Decode (sequence_length=1) with separate K/V, no rotary. past_seq_len controls k_size.
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Decode_K1) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/1, /*past_seq_len=*/1,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/false, /*is_packed_qkv=*/false);
  bool all_zero = std::all_of(output.begin(), output.end(), [](float v) { return v == 0.0f; });
  EXPECT_FALSE(all_zero) << "TurboQuant decode K=1 output should not be all zeros";
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Decode_K24) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/1, /*past_seq_len=*/24,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/false, /*is_packed_qkv=*/false);
  bool all_zero = std::all_of(output.begin(), output.end(), [](float v) { return v == 0.0f; });
  EXPECT_FALSE(all_zero) << "TurboQuant decode K=24 output should not be all zeros";
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Decode_K128) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/1, /*past_seq_len=*/128,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/false, /*is_packed_qkv=*/false);
  bool all_zero = std::all_of(output.begin(), output.end(), [](float v) { return v == 0.0f; });
  EXPECT_FALSE(all_zero) << "TurboQuant decode K=128 output should not be all zeros";
}

// Prefill (sequence_length > 1) with separate K/V, no rotary.
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Prefill_K1) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/4, /*past_seq_len=*/1,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/false, /*is_packed_qkv=*/false);
  bool all_zero = std::all_of(output.begin(), output.end(), [](float v) { return v == 0.0f; });
  EXPECT_FALSE(all_zero) << "TurboQuant prefill K=1 output should not be all zeros";
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Prefill_K24) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/4, /*past_seq_len=*/24,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/false, /*is_packed_qkv=*/false);
  bool all_zero = std::all_of(output.begin(), output.end(), [](float v) { return v == 0.0f; });
  EXPECT_FALSE(all_zero) << "TurboQuant prefill K=24 output should not be all zeros";
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Prefill_K128) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/4, /*past_seq_len=*/128,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/false, /*is_packed_qkv=*/false);
  bool all_zero = std::all_of(output.begin(), output.end(), [](float v) { return v == 0.0f; });
  EXPECT_FALSE(all_zero) << "TurboQuant prefill K=128 output should not be all zeros";
}

// Decode with rotary embedding (separate K/V path).
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Decode_Rotary_K24) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/1, /*past_seq_len=*/24,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/true, /*is_packed_qkv=*/false);
  bool all_zero = std::all_of(output.begin(), output.end(), [](float v) { return v == 0.0f; });
  EXPECT_FALSE(all_zero) << "TurboQuant decode rotary K=24 output should not be all zeros";
}

// Decode with packed QKV + rotary (fused split+rotary+Hadamard+quantize path).
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Decode_PackedRotary_K24) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/1, /*past_seq_len=*/24,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/true, /*is_packed_qkv=*/true);
  bool all_zero = std::all_of(output.begin(), output.end(), [](float v) { return v == 0.0f; });
  EXPECT_FALSE(all_zero) << "TurboQuant decode packed+rotary K=24 output should not be all zeros";
}

// Prefill with packed QKV + rotary (fused path, sequence_length > 1).
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Prefill_PackedRotary_K24) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/4, /*past_seq_len=*/24,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/true, /*is_packed_qkv=*/true);
  bool all_zero = std::all_of(output.begin(), output.end(), [](float v) { return v == 0.0f; });
  EXPECT_FALSE(all_zero) << "TurboQuant prefill packed+rotary K=24 output should not be all zeros";
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_Prefill_Regular_UsesQ8CacheShape) {
  auto ep = WebGpuEPWithKVCacheQuantization(8);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/4, /*past_seq_len=*/24,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/false, /*is_packed_qkv=*/false,
                                 /*bit_width=*/8);
  ExpectFiniteNonzeroOutput(output, "Q8 regular prefill");
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_Decode_PackedRotary_UsesQ8CacheShape) {
  auto ep = WebGpuEPWithKVCacheQuantization(8);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant(/*batch_size=*/1, /*sequence_length=*/1, /*past_seq_len=*/24,
                                 /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
                                 /*do_rotary=*/true, /*is_packed_qkv=*/true,
                                 /*bit_width=*/8);
  ExpectFiniteNonzeroOutput(output, "Q8 packed rotary decode");
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_Prefill_Float16) {
  auto ep = WebGpuEPWithKVCacheQuantization(8);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  auto output = RunGQATurboQuant<MLFloat16>(
      /*batch_size=*/1, /*sequence_length=*/4, /*past_seq_len=*/24,
      /*num_heads=*/2, /*kv_num_heads=*/1, /*head_size=*/128,
      /*do_rotary=*/false, /*is_packed_qkv=*/false,
      /*bit_width=*/8);
  ExpectFiniteNonzeroOutput(output, "Q8 float16 prefill");
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_UsesOffsetBinaryStorage) {
  auto ep = WebGpuEPWithKVCacheQuantization(8);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int head_size = 128;
  constexpr int compressed_head_size = 33;
  std::vector<float> query(head_size, 0.0f);
  std::vector<float> key(head_size);
  std::vector<float> value(head_size);
  std::array<uint32_t, compressed_head_size> expected_key{};
  std::array<uint32_t, compressed_head_size> expected_value{};
  const float expected_scale = 1.0f / 127.0f;
  std::memcpy(&expected_key[0], &expected_scale, sizeof(expected_scale));
  std::memcpy(&expected_value[0], &expected_scale, sizeof(expected_scale));
  for (int i = 0; i < head_size; ++i) {
    const int key_q = i - 127;
    const int value_q = 127 - i;
    key[i] = static_cast<float>(key_q) / 127.0f;
    value[i] = static_cast<float>(value_q) / 127.0f;
    const int word = 1 + i / 4;
    const int shift = (i % 4) * 8;
    expected_key[word] |= static_cast<uint32_t>(key_q + 128) << shift;
    expected_value[word] |= static_cast<uint32_t>(value_q + 128) << shift;
  }

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<float>("query", {1, 1, head_size}, query);
  tester.AddInput<float>("key", {1, 1, head_size}, key);
  tester.AddInput<float>("value", {1, 1, head_size}, value);
  tester.AddInput<float>("past_key", {1, 1, 0, compressed_head_size}, {});
  tester.AddInput<float>("past_value", {1, 1, 0, compressed_head_size}, {});
  tester.AddInput<int32_t>("seqlens_k", {1}, {0});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {1}, /*is_initializer=*/true);
  tester.AddOptionalInputEdge<float>();    // cos_cache
  tester.AddOptionalInputEdge<float>();    // sin_cache
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink
  tester.AddOutput<float>("output", {1, 1, head_size}, std::vector<float>(head_size));
  tester.AddOutput<float>("present_key", {1, 1, 1, compressed_head_size},
                          std::vector<float>(compressed_head_size));
  tester.AddOutput<float>("present_value", {1, 1, 1, compressed_head_size},
                          std::vector<float>(compressed_head_size));
  tester.SetOutputTolerance(1e6f);
  tester.SetCustomOutputVerifier([expected_key, expected_value](const std::vector<OrtValue>& fetches,
                                                                const std::string&) {
    ASSERT_EQ(fetches.size(), 3u);
    const auto* actual_key =
        static_cast<const uint32_t*>(fetches[1].Get<Tensor>().DataRaw());
    const auto* actual_value =
        static_cast<const uint32_t*>(fetches[2].Get<Tensor>().DataRaw());
    for (size_t i = 0; i < expected_key.size(); ++i) {
      EXPECT_EQ(actual_key[i], expected_key[i]) << "key word " << i;
      EXPECT_EQ(actual_value[i], expected_value[i]) << "value word " << i;
    }
  });

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// --- Decode test helper: multi-batch with per-batch seqlens_k using TurboQuant ---
// Before the fix, the TurboQuant copy-to-quantized-KV-cache kernels read seqlen_k[0]
// for EVERY batch, so batches 1..N-1 used the wrong past length. This helper proves the
// kernels now read seqlen_k[batch] via "swap invariance":
//
//   Run 1: batch data [A, B], seqlens_k [sA, sB]
//   Run 2: batch data [B, A], seqlens_k [sB, sA]   (both batches physically swapped)
//
// If each batch correctly uses its own seqlen, then the valid cache prefixes swap exactly
// between runs. If the kernel wrongly used seqlen_k[0] for all batches, the two runs would
// copy from or write to different cache positions for the same data and the swapped caches
// would NOT match.
//
// Both variants exercise turbo_quant_hadamard. The rotary variant additionally covers
// the separate Q/K rotary preprocessing used when past/present buffers are not aliased.
static void RunTurboQuantMultiBatchSwapInvariance(bool do_rotary) {
  if (!WebGpuEPWithKVCacheQuantization(4)) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 2;
  constexpr int sequence_length = 1;
  constexpr int past_seq_len = 24;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;

  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int total_sequence_length = past_seq_len + sequence_length;
  constexpr int kv_head_dim = (head_size * 4 + 32) / 32;  // TQ4 compressed dim
  constexpr int output_size = batch_size * sequence_length * hidden_size;
  constexpr int present_size = batch_size * kv_num_heads * total_sequence_length * kv_head_dim;

  // Distinct per-batch past lengths (right-padded prompts of different lengths).
  constexpr int32_t seqA = 20;  // batch slot 0 "A"
  constexpr int32_t seqB = 24;  // batch slot 1 "B"

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  // Per-batch Q/K/V and (already-packed) past-KV blocks for the two logical prompts A and B.
  const int q_per_batch = sequence_length * hidden_size;
  const int kv_per_batch = sequence_length * kv_hidden_size;
  const int past_per_batch = kv_num_heads * past_seq_len * kv_head_dim;

  auto make_vec = [&](int n) {
    std::vector<float> v(n);
    for (auto& e : v) e = dist(rng);
    return v;
  };
  const std::vector<float> qA = make_vec(q_per_batch), qB = make_vec(q_per_batch);
  const std::vector<float> kA = make_vec(kv_per_batch), kB = make_vec(kv_per_batch);
  const std::vector<float> vA = make_vec(kv_per_batch), vB = make_vec(kv_per_batch);
  const std::vector<float> pkA = make_vec(past_per_batch), pkB = make_vec(past_per_batch);
  const std::vector<float> pvA = make_vec(past_per_batch), pvB = make_vec(past_per_batch);

  const int max_seq_len = total_sequence_length + 8;
  const int half_rotary = head_size / 2;
  std::vector<float> cos_cache(max_seq_len * half_rotary);
  std::vector<float> sin_cache(max_seq_len * half_rotary);
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int d = 0; d < half_rotary; ++d) {
      float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(d) / static_cast<float>(head_size));
      cos_cache[pos * half_rotary + d] = std::cos(static_cast<float>(pos) * freq);
      sin_cache[pos * half_rotary + d] = std::sin(static_cast<float>(pos) * freq);
    }
  }

  auto concat = [](const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a);
    out.insert(out.end(), b.begin(), b.end());
    return out;
  };

  struct PackedPresentCache {
    std::vector<uint8_t> key;
    std::vector<uint8_t> value;
  };

  auto copy_tensor_bytes = [](const Tensor& tensor) {
    const auto* begin = static_cast<const uint8_t*>(tensor.DataRaw());
    return std::vector<uint8_t>(begin, begin + tensor.SizeInBytes());
  };

  // Runs GQA with the two batch slots holding (first, second) prompts and the given seqlens.
  // Returns owned copies of the packed present caches.
  auto run = [&](const std::vector<float>& q0, const std::vector<float>& q1,
                 const std::vector<float>& k0, const std::vector<float>& k1,
                 const std::vector<float>& v0, const std::vector<float>& v1,
                 const std::vector<float>& pk0, const std::vector<float>& pk1,
                 const std::vector<float>& pv0, const std::vector<float>& pv1,
                 int32_t s0, int32_t s1) {
    OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
    tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
    if (do_rotary) {
      tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));
    }

    tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size}, concat(q0, q1));
    tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size}, concat(k0, k1));
    tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size}, concat(v0, v1));
    tester.AddInput<float>("past_key", {batch_size, kv_num_heads, past_seq_len, kv_head_dim}, concat(pk0, pk1));
    tester.AddInput<float>("past_value", {batch_size, kv_num_heads, past_seq_len, kv_head_dim}, concat(pv0, pv1));

    std::vector<int32_t> seqlens_k{s0, s1};
    tester.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k);
    tester.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length}, /*is_initializer=*/true);

    if (do_rotary) {
      tester.AddInput<float>("cos_cache", {max_seq_len, half_rotary}, cos_cache);
      tester.AddInput<float>("sin_cache", {max_seq_len, half_rotary}, sin_cache);
    } else {
      tester.AddOptionalInputEdge<float>();  // cos_cache
      tester.AddOptionalInputEdge<float>();  // sin_cache
    }
    tester.AddOptionalInputEdge<int64_t>();  // position_ids
    tester.AddOptionalInputEdge<float>();    // attention_bias
    tester.AddOptionalInputEdge<float>();    // head_sink

    tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                            std::vector<float>(output_size, 0.0f));
    tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, total_sequence_length, kv_head_dim},
                            std::vector<float>(present_size, 0.0f));
    tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, total_sequence_length, kv_head_dim},
                            std::vector<float>(present_size, 0.0f));

    // Only the attention output is numerically meaningful; present_key/value are packed
    // quantized bytes reinterpreted as float, so skip their value checks.
    tester.SetOutputTolerance(1e6f);
    tester.SetCustomOutputVerifier([](const std::vector<OrtValue>&, const std::string&) {});

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(WebGpuEPWithKVCacheQuantization(4));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

    auto fetches = tester.GetFetches();
    return PackedPresentCache{copy_tensor_bytes(fetches[1].Get<Tensor>()),
                              copy_tensor_bytes(fetches[2].Get<Tensor>())};
  };

  // Run 1: slot0=A (seqA), slot1=B (seqB).
  auto out1 = run(qA, qB, kA, kB, vA, vB, pkA, pkB, pvA, pvB, seqA, seqB);
  // Run 2: swap both the data AND the seqlens so slot0=B (seqB), slot1=A (seqA).
  auto out2 = run(qB, qA, kB, kA, vB, vA, pkB, pkA, pvB, pvA, seqB, seqA);

  auto expect_batches_swapped = [&](const std::vector<uint8_t>& first,
                                    const std::vector<uint8_t>& second,
                                    const char* cache_name) {
    ASSERT_EQ(first.size(), second.size());
    ASSERT_EQ(first.size(), static_cast<size_t>(present_size) * sizeof(float));
    const size_t bytes_per_batch = first.size() / batch_size;
    constexpr size_t bytes_per_token = kv_num_heads * kv_head_dim * sizeof(float);
    for (int second_batch = 0; second_batch < batch_size; ++second_batch) {
      const int first_batch = batch_size - 1 - second_batch;
      const auto* second_begin = second.data() + second_batch * bytes_per_batch;
      const auto* first_begin = first.data() + first_batch * bytes_per_batch;
      const int32_t logical_sequence_length = first_batch == 0 ? seqA + 1 : seqB + 1;
      const size_t valid_bytes = logical_sequence_length * bytes_per_token;
      EXPECT_TRUE(std::equal(second_begin, second_begin + valid_bytes, first_begin))
          << cache_name << " cache mismatch after swapping batch " << first_batch
          << " into slot " << second_batch
          << "; per-batch seqlens_k not honored in quantized KV cache path";
    }
  };

  expect_batches_swapped(out1.key, out2.key, "key");
  expect_batches_swapped(out1.value, out2.value, "value");
}

// Rotary variant: exercises the fused rotary+Hadamard copy kernel.
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Decode_MultiBatch_UsesPerBatchSeqlensK) {
  RunTurboQuantMultiBatchSwapInvariance(/*do_rotary=*/true);
}

// Non-rotary variant: exercises the plain Hadamard copy kernel (turbo_quant_hadamard).
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Decode_MultiBatch_NoRotary_UsesPerBatchSeqlensK) {
  RunTurboQuantMultiBatchSwapInvariance(/*do_rotary=*/false);
}

// Right-padded first prompts can have a per-batch total sequence length smaller than the
// padded K/V sequence length. Exercise the dynamic-cache path for turbo_quant_hadamard;
// the graph-capture tests above cover the static-cache and fused rotary variants.
static void RunTurboQuantRightPaddedPrefill(bool do_rotary) {
  if (!WebGpuEPWithKVCacheQuantization(4)) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 2;
  constexpr int sequence_length = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int packed_hidden_size = hidden_size + 2 * kv_hidden_size;
  constexpr int kv_head_dim = (head_size * 4 + 32) / 32;

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", num_heads);
  tester.AddAttribute<int64_t>("kv_num_heads", kv_num_heads);
  if (do_rotary) {
    tester.AddAttribute<int64_t>("do_rotary", 1);
  }

  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  auto make_data = [&](int size) {
    std::vector<float> data(size);
    for (float& value : data) {
      value = dist(rng);
    }
    return data;
  };

  if (do_rotary) {
    tester.AddInput<float>("query", {batch_size, sequence_length, packed_hidden_size},
                           make_data(batch_size * sequence_length * packed_hidden_size));
    tester.AddOptionalInputEdge<float>();  // key
    tester.AddOptionalInputEdge<float>();  // value
  } else {
    tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size},
                           make_data(batch_size * sequence_length * hidden_size));
    tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size},
                           make_data(batch_size * sequence_length * kv_hidden_size));
    tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size},
                           make_data(batch_size * sequence_length * kv_hidden_size));
  }

  const int cache_size = batch_size * kv_num_heads * sequence_length * kv_head_dim;
  tester.AddInput<float>("past_key", {batch_size, kv_num_heads, sequence_length, kv_head_dim},
                         std::vector<float>(cache_size, 0.0f));
  tester.AddInput<float>("past_value", {batch_size, kv_num_heads, sequence_length, kv_head_dim},
                         std::vector<float>(cache_size, 0.0f));

  // seqlens_k is total_sequence_length - 1. Batch 0 therefore has only two valid
  // tokens in the four-token padded prompt, while batch 1 uses the full prompt.
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {1, 3});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {sequence_length}, /*is_initializer=*/true);

  if (do_rotary) {
    constexpr int max_seq_len = sequence_length + 8;
    constexpr int half_rotary = head_size / 2;
    std::vector<float> cos_cache(max_seq_len * half_rotary);
    std::vector<float> sin_cache(max_seq_len * half_rotary);
    for (int pos = 0; pos < max_seq_len; ++pos) {
      for (int dim = 0; dim < half_rotary; ++dim) {
        const float frequency = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(dim) / head_size);
        cos_cache[pos * half_rotary + dim] = std::cos(static_cast<float>(pos) * frequency);
        sin_cache[pos * half_rotary + dim] = std::sin(static_cast<float>(pos) * frequency);
      }
    }
    tester.AddInput<float>("cos_cache", {max_seq_len, half_rotary}, cos_cache);
    tester.AddInput<float>("sin_cache", {max_seq_len, half_rotary}, sin_cache);
  } else {
    tester.AddOptionalInputEdge<float>();  // cos_cache
    tester.AddOptionalInputEdge<float>();  // sin_cache
  }
  tester.AddOptionalInputEdge<int64_t>();  // position_ids
  tester.AddOptionalInputEdge<float>();    // attention_bias
  tester.AddOptionalInputEdge<float>();    // head_sink

  const int output_size = batch_size * sequence_length * hidden_size;
  tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                          std::vector<float>(output_size, 0.0f));
  tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, sequence_length, kv_head_dim},
                          std::vector<float>(cache_size, 0.0f));
  tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, sequence_length, kv_head_dim},
                          std::vector<float>(cache_size, 0.0f));
  tester.SetOutputTolerance(1e6f);
  tester.SetCustomOutputVerifier([](const std::vector<OrtValue>&, const std::string&) {});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(WebGpuEPWithKVCacheQuantization(4));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  auto fetches = tester.GetFetches();
  const auto& output = fetches[0].Get<Tensor>();
  const float* output_data = output.Data<float>();
  const int output_per_batch = sequence_length * hidden_size;
  for (int batch = 0; batch < batch_size; ++batch) {
    const float* batch_begin = output_data + batch * output_per_batch;
    const bool all_zero = std::all_of(batch_begin, batch_begin + output_per_batch,
                                      [](float value) { return value == 0.0f; });
    EXPECT_FALSE(all_zero) << "TurboQuant output is all zero for right-padded batch " << batch;
  }
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Prefill_MultiBatch_RightPadding_NoRotary) {
  RunTurboQuantRightPaddedPrefill(/*do_rotary=*/false);
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_Prefill_MultiBatch_RightPadding_Rotary) {
  RunTurboQuantRightPaddedPrefill(/*do_rotary=*/true);
}

// ---------------------------------------------------------------------------
// TurboQuant cross-validation tests: compare TQ output vs non-TQ reference.
// With past_seq_len=0 (no past KV), both versions receive identical Q/K/V.
// The Hadamard transform is orthogonal (preserves dot products), so TQ output
// should approximate non-TQ output within 4-bit quantization error bounds.
// ---------------------------------------------------------------------------

// Helper: runs GQA without TurboQuant (standard uncompressed KV cache) and returns the output.
static std::vector<float> RunGQAReference(
    int batch_size,
    int sequence_length,
    int num_heads,
    int kv_num_heads,
    int head_size,
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& value_data,
    bool do_rotary,
    bool use_fp16 = false) {
  const int hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int total_sequence_length = sequence_length;  // no past

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  if (do_rotary) {
    tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));
  }

  if (use_fp16) {
    tester.AddInput<MLFloat16>("query", {batch_size, sequence_length, hidden_size}, ToFloat16(query_data));
    tester.AddInput<MLFloat16>("key", {batch_size, sequence_length, kv_hidden_size}, ToFloat16(key_data));
    tester.AddInput<MLFloat16>("value", {batch_size, sequence_length, kv_hidden_size}, ToFloat16(value_data));
    tester.AddOptionalInputEdge<MLFloat16>();  // past_key
    tester.AddOptionalInputEdge<MLFloat16>();  // past_value
  } else {
    tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size}, query_data);
    tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size}, key_data);
    tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size}, value_data);
    tester.AddOptionalInputEdge<float>();  // past_key
    tester.AddOptionalInputEdge<float>();  // past_value
  }

  std::vector<int32_t> seqlens_k(batch_size, total_sequence_length - 1);
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length}, /*is_initializer=*/true);

  if (do_rotary) {
    const int max_seq_len = total_sequence_length + 8;
    const int half_rotary = head_size / 2;
    std::vector<float> cos_cache(max_seq_len * half_rotary);
    std::vector<float> sin_cache(max_seq_len * half_rotary);
    for (int pos = 0; pos < max_seq_len; ++pos) {
      for (int d = 0; d < half_rotary; ++d) {
        float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(d) / static_cast<float>(head_size));
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
  } else {
    if (use_fp16) {
      tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
      tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
    } else {
      tester.AddOptionalInputEdge<float>();  // cos_cache
      tester.AddOptionalInputEdge<float>();  // sin_cache
    }
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
                                std::vector<MLFloat16>(output_size));
    tester.AddOutput<MLFloat16>("present_key", {batch_size, kv_num_heads, total_sequence_length, head_size},
                                std::vector<MLFloat16>(present_size));
    tester.AddOutput<MLFloat16>("present_value", {batch_size, kv_num_heads, total_sequence_length, head_size},
                                std::vector<MLFloat16>(present_size));
  } else {
    tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                            std::vector<float>(output_size, 0.0f));
    tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, total_sequence_length, head_size},
                            std::vector<float>(present_size, 0.0f));
    tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, total_sequence_length, head_size},
                            std::vector<float>(present_size, 0.0f));
  }

  tester.SetOutputTolerance(1e6f);
  tester.SetCustomOutputVerifier([](const std::vector<OrtValue>&, const std::string&) {});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  auto fetches = tester.GetFetches();
  if (use_fp16) {
    const MLFloat16* out_data = fetches[0].Get<Tensor>().Data<MLFloat16>();
    std::vector<float> output;
    output.reserve(output_size);
    for (int i = 0; i < output_size; ++i) {
      output.push_back(out_data[i].ToFloat());
    }
    return output;
  }

  const float* out_data = fetches[0].Get<Tensor>().Data<float>();
  return std::vector<float>(out_data, out_data + output_size);
}

// Helper: runs GQA with TurboQuant, past_seq_len=0, and returns the output.
static std::vector<float> RunGQATurboQuantNoPast(
    int batch_size,
    int sequence_length,
    int num_heads,
    int kv_num_heads,
    int head_size,
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& value_data,
    bool do_rotary,
    uint32_t bit_width = 4,
    bool use_fp16 = false) {
  const int hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int total_sequence_length = sequence_length;  // no past
  const size_t element_size = use_fp16 ? sizeof(MLFloat16) : sizeof(float);
  const int kv_head_dim =
      static_cast<int>(contrib::webgpu::KvCacheQuantizedHeadSize(head_size, bit_width, element_size));

  OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("kv_num_heads", static_cast<int64_t>(kv_num_heads));
  if (do_rotary) {
    tester.AddAttribute<int64_t>("do_rotary", static_cast<int64_t>(1));
  }

  if (use_fp16) {
    tester.AddInput<MLFloat16>("query", {batch_size, sequence_length, hidden_size}, ToFloat16(query_data));
    tester.AddInput<MLFloat16>("key", {batch_size, sequence_length, kv_hidden_size}, ToFloat16(key_data));
    tester.AddInput<MLFloat16>("value", {batch_size, sequence_length, kv_hidden_size}, ToFloat16(value_data));
    tester.AddInput<MLFloat16>("past_key",
                               {batch_size, kv_num_heads, static_cast<int64_t>(0), kv_head_dim}, {});
    tester.AddInput<MLFloat16>("past_value",
                               {batch_size, kv_num_heads, static_cast<int64_t>(0), kv_head_dim}, {});
  } else {
    tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size}, query_data);
    tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size}, key_data);
    tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size}, value_data);
    tester.AddInput<float>("past_key", {batch_size, kv_num_heads, static_cast<int64_t>(0), kv_head_dim}, {});
    tester.AddInput<float>("past_value", {batch_size, kv_num_heads, static_cast<int64_t>(0), kv_head_dim}, {});
  }

  std::vector<int32_t> seqlens_k(batch_size, total_sequence_length - 1);
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, seqlens_k);
  tester.AddInput<int32_t>("total_sequence_length", {1}, {total_sequence_length}, /*is_initializer=*/true);

  if (do_rotary) {
    const int max_seq_len = total_sequence_length + 8;
    const int half_rotary = head_size / 2;
    std::vector<float> cos_cache(max_seq_len * half_rotary);
    std::vector<float> sin_cache(max_seq_len * half_rotary);
    for (int pos = 0; pos < max_seq_len; ++pos) {
      for (int d = 0; d < half_rotary; ++d) {
        float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(d) / static_cast<float>(head_size));
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
  } else {
    if (use_fp16) {
      tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
      tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
    } else {
      tester.AddOptionalInputEdge<float>();  // cos_cache
      tester.AddOptionalInputEdge<float>();  // sin_cache
    }
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
  const int present_size = batch_size * kv_num_heads * total_sequence_length * kv_head_dim;
  if (use_fp16) {
    tester.AddOutput<MLFloat16>("output", {batch_size, sequence_length, hidden_size},
                                std::vector<MLFloat16>(output_size));
    tester.AddOutput<MLFloat16>("present_key",
                                {batch_size, kv_num_heads, total_sequence_length, kv_head_dim},
                                std::vector<MLFloat16>(present_size));
    tester.AddOutput<MLFloat16>("present_value",
                                {batch_size, kv_num_heads, total_sequence_length, kv_head_dim},
                                std::vector<MLFloat16>(present_size));
  } else {
    tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                            std::vector<float>(output_size, 0.0f));
    tester.AddOutput<float>("present_key", {batch_size, kv_num_heads, total_sequence_length, kv_head_dim},
                            std::vector<float>(present_size, 0.0f));
    tester.AddOutput<float>("present_value", {batch_size, kv_num_heads, total_sequence_length, kv_head_dim},
                            std::vector<float>(present_size, 0.0f));
  }

  tester.SetOutputTolerance(1e6f);
  tester.SetCustomOutputVerifier([](const std::vector<OrtValue>&, const std::string&) {});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(WebGpuEPWithKVCacheQuantization(bit_width));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  auto fetches = tester.GetFetches();
  if (use_fp16) {
    const MLFloat16* out_data = fetches[0].Get<Tensor>().Data<MLFloat16>();
    std::vector<float> output;
    output.reserve(output_size);
    for (int i = 0; i < output_size; ++i) {
      output.push_back(out_data[i].ToFloat());
    }
    return output;
  }

  const float* out_data = fetches[0].Get<Tensor>().Data<float>();
  return std::vector<float>(out_data, out_data + output_size);
}

// Cross-validate TQ vs non-TQ: Prefill with 4 tokens, no past, no rotary.
// With 4-bit quantization (16 centroids), expect bounded error.
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_CrossValidate_Prefill_NoRotary) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  // Deterministic data
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (auto& v : query_data) v = dist(rng);
  for (auto& v : key_data) v = dist(rng);
  for (auto& v : value_data) v = dist(rng);

  auto ref_output = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                    query_data, key_data, value_data, /*do_rotary=*/false);
  auto tq_output = RunGQATurboQuantNoPast(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                          query_data, key_data, value_data, /*do_rotary=*/false);

  ASSERT_EQ(ref_output.size(), tq_output.size());
  float max_abs_err = 0.0f;
  float sum_sq_err = 0.0f;
  float sum_sq_ref = 0.0f;
  for (size_t i = 0; i < ref_output.size(); i++) {
    float err = std::abs(ref_output[i] - tq_output[i]);
    max_abs_err = std::max(max_abs_err, err);
    sum_sq_err += (ref_output[i] - tq_output[i]) * (ref_output[i] - tq_output[i]);
    sum_sq_ref += ref_output[i] * ref_output[i];
  }
  float rmse = std::sqrt(sum_sq_err / static_cast<float>(ref_output.size()));
  float relative_rmse = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : rmse;

  // 4-bit quantization with 16 MSE-optimal centroids: expect relative RMSE < 20%
  // and max absolute error bounded (values are in [-0.5, 0.5] range).
  EXPECT_LT(relative_rmse, 0.2f) << "TurboQuant relative RMSE too large: " << relative_rmse;
  EXPECT_LT(max_abs_err, 0.3f) << "TurboQuant max absolute error too large: " << max_abs_err;
}

// Cross-validate TQ vs non-TQ: Prefill with rotary embedding.
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_CrossValidate_Prefill_Rotary) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  std::mt19937 rng(456);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (auto& v : query_data) v = dist(rng);
  for (auto& v : key_data) v = dist(rng);
  for (auto& v : value_data) v = dist(rng);

  auto ref_output = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                    query_data, key_data, value_data, /*do_rotary=*/true);
  auto tq_output = RunGQATurboQuantNoPast(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                          query_data, key_data, value_data, /*do_rotary=*/true);

  ASSERT_EQ(ref_output.size(), tq_output.size());
  float max_abs_err = 0.0f;
  float sum_sq_err = 0.0f;
  float sum_sq_ref = 0.0f;
  for (size_t i = 0; i < ref_output.size(); i++) {
    float err = std::abs(ref_output[i] - tq_output[i]);
    max_abs_err = std::max(max_abs_err, err);
    sum_sq_err += (ref_output[i] - tq_output[i]) * (ref_output[i] - tq_output[i]);
    sum_sq_ref += ref_output[i] * ref_output[i];
  }
  float rmse = std::sqrt(sum_sq_err / static_cast<float>(ref_output.size()));
  float relative_rmse = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : rmse;

  EXPECT_LT(relative_rmse, 0.2f) << "TurboQuant+rotary relative RMSE too large: " << relative_rmse;
  EXPECT_LT(max_abs_err, 0.3f) << "TurboQuant+rotary max absolute error too large: " << max_abs_err;
}

// Cross-validate: single decode token (sequence_length=1, past_seq_len=0).
// This exercises the split-reduce decode kernel path.
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_CrossValidate_Decode) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  std::mt19937 rng(789);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (auto& v : query_data) v = dist(rng);
  for (auto& v : key_data) v = dist(rng);
  for (auto& v : value_data) v = dist(rng);

  auto ref_output = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                    query_data, key_data, value_data, /*do_rotary=*/false);
  auto tq_output = RunGQATurboQuantNoPast(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                          query_data, key_data, value_data, /*do_rotary=*/false);

  ASSERT_EQ(ref_output.size(), tq_output.size());
  float max_abs_err = 0.0f;
  float sum_sq_err = 0.0f;
  float sum_sq_ref = 0.0f;
  for (size_t i = 0; i < ref_output.size(); i++) {
    float err = std::abs(ref_output[i] - tq_output[i]);
    max_abs_err = std::max(max_abs_err, err);
    sum_sq_err += (ref_output[i] - tq_output[i]) * (ref_output[i] - tq_output[i]);
    sum_sq_ref += ref_output[i] * ref_output[i];
  }
  float relative_rmse = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);

  // Single token: attention is just softmax(Q*K^T/sqrt(d)) * V with a single KV pair.
  // Quantization error on one pair should be small.
  EXPECT_LT(relative_rmse, 0.15f) << "TurboQuant decode relative RMSE too large: " << relative_rmse;
  EXPECT_LT(max_abs_err, 0.25f) << "TurboQuant decode max absolute error too large: " << max_abs_err;
}

// Cross-validate: longer prefill (8 tokens) with multiple KV heads.
TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_CrossValidate_Prefill8_MultiKVHead) {
  auto ep = WebGpuEPWithKVCacheQuantization(4);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 8;
  constexpr int num_heads = 4;
  constexpr int kv_num_heads = 2;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  std::mt19937 rng(1001);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (auto& v : query_data) v = dist(rng);
  for (auto& v : key_data) v = dist(rng);
  for (auto& v : value_data) v = dist(rng);

  auto ref_output = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                    query_data, key_data, value_data, /*do_rotary=*/false);
  auto tq_output = RunGQATurboQuantNoPast(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                          query_data, key_data, value_data, /*do_rotary=*/false);

  ASSERT_EQ(ref_output.size(), tq_output.size());
  float max_abs_err = 0.0f;
  float sum_sq_err = 0.0f;
  float sum_sq_ref = 0.0f;
  for (size_t i = 0; i < ref_output.size(); i++) {
    float err = std::abs(ref_output[i] - tq_output[i]);
    max_abs_err = std::max(max_abs_err, err);
    sum_sq_err += (ref_output[i] - tq_output[i]) * (ref_output[i] - tq_output[i]);
    sum_sq_ref += ref_output[i] * ref_output[i];
  }
  float relative_rmse = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : std::sqrt(sum_sq_err);

  EXPECT_LT(relative_rmse, 0.2f) << "TurboQuant 8-token multi-head relative RMSE too large: " << relative_rmse;
  EXPECT_LT(max_abs_err, 0.3f) << "TurboQuant 8-token multi-head max absolute error too large: " << max_abs_err;
}

static void ExpectBlockQuantInt8Close(const std::vector<float>& reference,
                                      const std::vector<float>& actual,
                                      float max_relative_rmse,
                                      float max_absolute_error) {
  ASSERT_EQ(reference.size(), actual.size());
  float max_abs_err = 0.0f;
  float sum_sq_err = 0.0f;
  float sum_sq_ref = 0.0f;
  for (size_t i = 0; i < reference.size(); ++i) {
    const float error = reference[i] - actual[i];
    max_abs_err = std::max(max_abs_err, std::abs(error));
    sum_sq_err += error * error;
    sum_sq_ref += reference[i] * reference[i];
  }
  const float relative_rmse = sum_sq_ref > 0.0f ? std::sqrt(sum_sq_err / sum_sq_ref)
                                                : std::sqrt(sum_sq_err);
  EXPECT_LT(relative_rmse, max_relative_rmse);
  EXPECT_LT(max_abs_err, max_absolute_error);
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_CrossValidate_Prefill_Rotary) {
  auto ep = WebGpuEPWithKVCacheQuantization(8);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  std::mt19937 rng(8008);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (float& value : query_data) value = dist(rng);
  for (float& value : key_data) value = dist(rng);
  for (float& value : value_data) value = dist(rng);

  const auto reference = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                         query_data, key_data, value_data, /*do_rotary=*/true);
  const auto actual = RunGQATurboQuantNoPast(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                             query_data, key_data, value_data, /*do_rotary=*/true,
                                             /*bit_width=*/8);
  ExpectBlockQuantInt8Close(reference, actual, /*max_relative_rmse=*/0.02f,
                            /*max_absolute_error=*/0.03f);
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_CrossValidate_FlashPrefill) {
  auto ep = WebGpuEPWithKVCacheQuantization(8);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 40;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  std::mt19937 rng(8040);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (float& value : query_data) value = dist(rng);
  for (float& value : key_data) value = dist(rng);
  for (float& value : value_data) value = dist(rng);

  const auto reference = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                         query_data, key_data, value_data, /*do_rotary=*/false);
  const auto actual = RunGQATurboQuantNoPast(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                             query_data, key_data, value_data, /*do_rotary=*/false,
                                             /*bit_width=*/8);
  ExpectBlockQuantInt8Close(reference, actual, /*max_relative_rmse=*/0.02f,
                            /*max_absolute_error=*/0.03f);
}

static void RunFp16HighMagnitudeAttention(int sequence_length, uint32_t bit_width) {
  auto ep = bit_width == 0 ? DefaultWebGpuExecutionProvider() : WebGpuEPWithKVCacheQuantization(bit_width);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  std::mt19937 rng(1234);
  std::bernoulli_distribution sign_dist(0.5);
  std::vector<float> qk_signs(head_size);
  std::vector<float> value_signs(head_size);
  for (int d = 0; d < head_size; ++d) {
    qk_signs[d] = sign_dist(rng) ? 1.0f : -1.0f;
    value_signs[d] = sign_dist(rng) ? 1.0f : -1.0f;
  }

  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (int s = 0; s < sequence_length; ++s) {
    const float key_value = s % 2 == 0 ? 90.0f : 100.0f;
    const float value_value = s % 2 == 0 ? 0.25f : 0.75f;
    for (int h = 0; h < num_heads; ++h) {
      for (int d = 0; d < head_size; ++d) {
        query_data[(s * num_heads + h) * head_size + d] = 100.0f * qk_signs[d];
      }
    }
    for (int d = 0; d < head_size; ++d) {
      key_data[s * kv_hidden_size + d] = key_value * qk_signs[d];
      value_data[s * kv_hidden_size + d] = value_value * value_signs[d];
    }
  }

  const auto reference = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                         query_data, key_data, value_data, /*do_rotary=*/false);
  const auto actual =
      bit_width == 0
          ? RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                            query_data, key_data, value_data, /*do_rotary=*/false, /*use_fp16=*/true)
          : RunGQATurboQuantNoPast(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                   query_data, key_data, value_data, /*do_rotary=*/false,
                                   bit_width, /*use_fp16=*/true);
  const float relative_rmse_tolerance = bit_width == 4 ? 0.2f : 0.01f;
  const float absolute_error_tolerance = bit_width == 4 ? 0.5f : 0.01f;
  ExpectBlockQuantInt8Close(reference, actual, relative_rmse_tolerance, absolute_error_tolerance);
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_HighMagnitude_FlashPrefill) {
  RunFp16HighMagnitudeAttention(/*sequence_length=*/40, /*bit_width=*/0);
}

TEST(GroupQueryAttentionTest, WebGPU_FP16_HighMagnitude_SplitReduce) {
  RunFp16HighMagnitudeAttention(/*sequence_length=*/4, /*bit_width=*/0);
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_FP16_HighMagnitude_FlashPrefill) {
  RunFp16HighMagnitudeAttention(/*sequence_length=*/40, /*bit_width=*/4);
}

TEST(GroupQueryAttentionTest, WebGPU_TurboQuant_FP16_HighMagnitude_SplitReduce) {
  RunFp16HighMagnitudeAttention(/*sequence_length=*/4, /*bit_width=*/4);
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_FP16_HighMagnitude_FlashPrefill) {
  RunFp16HighMagnitudeAttention(/*sequence_length=*/40, /*bit_width=*/8);
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_FP16_HighMagnitude_SplitReduce) {
  RunFp16HighMagnitudeAttention(/*sequence_length=*/4, /*bit_width=*/8);
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_FP16_TinyScale) {
  auto ep = WebGpuEPWithKVCacheQuantization(8);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;

  std::vector<float> query_data(batch_size * sequence_length * hidden_size, 1.0f);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (int s = 0; s < sequence_length; ++s) {
    for (int d = 0; d < head_size; ++d) {
      const float sign = d % 2 == 0 ? 1.0f : -1.0f;
      key_data[s * kv_hidden_size + d] = sign * (2.0e-6f + static_cast<float>(s) * 2.0e-7f);
      value_data[s * kv_hidden_size + d] = sign * (1.5e-6f + static_cast<float>(s) * 2.0e-7f);
    }
  }

  const auto reference = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                         query_data, key_data, value_data, /*do_rotary=*/false);
  const auto actual = RunGQATurboQuantNoPast(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                             query_data, key_data, value_data, /*do_rotary=*/false,
                                             /*bit_width=*/8, /*use_fp16=*/true);
  ExpectBlockQuantInt8Close(reference, actual, /*max_relative_rmse=*/0.1f,
                            /*max_absolute_error=*/5.0e-7f);
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_CrossValidate_Decode) {
  auto ep = WebGpuEPWithKVCacheQuantization(8);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 1;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  std::mt19937 rng(808);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (float& value : query_data) value = dist(rng);
  for (float& value : key_data) value = dist(rng);
  for (float& value : value_data) value = dist(rng);

  const auto reference = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                         query_data, key_data, value_data, /*do_rotary=*/false);
  const auto actual = RunGQATurboQuantNoPast(batch_size, sequence_length, num_heads, kv_num_heads, head_size,
                                             query_data, key_data, value_data, /*do_rotary=*/false,
                                             /*bit_width=*/8);
  ExpectBlockQuantInt8Close(reference, actual, /*max_relative_rmse=*/0.01f,
                            /*max_absolute_error=*/0.01f);
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_CrossValidate_PrefillThenDecode) {
  auto ep = WebGpuEPWithKVCacheQuantization(8);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int prefill_length = 4;
  constexpr int decode_length = 1;
  constexpr int total_sequence_length = prefill_length + decode_length;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  constexpr int compressed_head_size = (head_size * 8 + 32) / 32;

  struct Q8RunResult {
    std::vector<float> output;
    std::vector<float> present_key;
    std::vector<float> present_value;
  };
  auto run_q8 = [&](int sequence_length,
                    int past_sequence_length,
                    const std::vector<float>& query,
                    const std::vector<float>& key,
                    const std::vector<float>& value,
                    const std::vector<float>& past_key,
                    const std::vector<float>& past_value) {
    const int present_sequence_length = past_sequence_length + sequence_length;
    OpTester tester("GroupQueryAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", num_heads);
    tester.AddAttribute<int64_t>("kv_num_heads", kv_num_heads);
    tester.AddInput<float>("query", {batch_size, sequence_length, hidden_size}, query);
    tester.AddInput<float>("key", {batch_size, sequence_length, kv_hidden_size}, key);
    tester.AddInput<float>("value", {batch_size, sequence_length, kv_hidden_size}, value);
    tester.AddInput<float>("past_key",
                           {batch_size, kv_num_heads, past_sequence_length, compressed_head_size},
                           past_key);
    tester.AddInput<float>("past_value",
                           {batch_size, kv_num_heads, past_sequence_length, compressed_head_size},
                           past_value);
    tester.AddInput<int32_t>("seqlens_k", {batch_size}, {present_sequence_length - 1});
    tester.AddInput<int32_t>("total_sequence_length", {1}, {present_sequence_length},
                             /*is_initializer=*/true);
    tester.AddOptionalInputEdge<float>();    // cos_cache
    tester.AddOptionalInputEdge<float>();    // sin_cache
    tester.AddOptionalInputEdge<int64_t>();  // position_ids
    tester.AddOptionalInputEdge<float>();    // attention_bias
    tester.AddOptionalInputEdge<float>();    // head_sink

    tester.AddOutput<float>("output", {batch_size, sequence_length, hidden_size},
                            std::vector<float>(batch_size * sequence_length * hidden_size));
    tester.AddOutput<float>(
        "present_key", {batch_size, kv_num_heads, present_sequence_length, compressed_head_size},
        std::vector<float>(batch_size * kv_num_heads * present_sequence_length * compressed_head_size));
    tester.AddOutput<float>(
        "present_value", {batch_size, kv_num_heads, present_sequence_length, compressed_head_size},
        std::vector<float>(batch_size * kv_num_heads * present_sequence_length * compressed_head_size));
    tester.SetOutputTolerance(1e6f);
    tester.SetCustomOutputVerifier([](const std::vector<OrtValue>&, const std::string&) {});

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(WebGpuEPWithKVCacheQuantization(8));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

    const auto fetches = tester.GetFetches();
    const auto copy_float_tensor = [](const Tensor& tensor) {
      return std::vector<float>(tensor.Data<float>(), tensor.Data<float>() + tensor.Shape().Size());
    };
    return Q8RunResult{copy_float_tensor(fetches[0].Get<Tensor>()),
                       copy_float_tensor(fetches[1].Get<Tensor>()),
                       copy_float_tensor(fetches[2].Get<Tensor>())};
  };

  std::mt19937 rng(8108);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> all_query(total_sequence_length * hidden_size);
  std::vector<float> all_key(total_sequence_length * kv_hidden_size);
  std::vector<float> all_value(total_sequence_length * kv_hidden_size);
  for (float& element : all_query) element = dist(rng);
  for (float& element : all_key) element = dist(rng);
  for (float& element : all_value) element = dist(rng);

  const std::vector<float> prefill_query(all_query.begin(), all_query.begin() + prefill_length * hidden_size);
  const std::vector<float> prefill_key(all_key.begin(), all_key.begin() + prefill_length * kv_hidden_size);
  const std::vector<float> prefill_value(all_value.begin(), all_value.begin() + prefill_length * kv_hidden_size);
  const std::vector<float> decode_query(all_query.begin() + prefill_length * hidden_size, all_query.end());
  const std::vector<float> decode_key(all_key.begin() + prefill_length * kv_hidden_size, all_key.end());
  const std::vector<float> decode_value(all_value.begin() + prefill_length * kv_hidden_size, all_value.end());

  const auto prefill = run_q8(prefill_length, 0, prefill_query, prefill_key, prefill_value, {}, {});
  const auto decode = run_q8(decode_length, prefill_length, decode_query, decode_key, decode_value,
                             prefill.present_key, prefill.present_value);
  const auto reference = RunGQAReference(batch_size, total_sequence_length, num_heads, kv_num_heads,
                                         head_size, all_query, all_key, all_value, /*do_rotary=*/false);
  const std::vector<float> reference_decode(reference.end() - hidden_size, reference.end());
  ExpectBlockQuantInt8Close(reference_decode, decode.output, /*max_relative_rmse=*/0.01f,
                            /*max_absolute_error=*/0.01f);

  ASSERT_EQ(decode.present_key.size(),
            static_cast<size_t>(total_sequence_length * compressed_head_size));
  ASSERT_EQ(decode.present_value.size(),
            static_cast<size_t>(total_sequence_length * compressed_head_size));
  EXPECT_EQ(std::memcmp(prefill.present_key.data(), decode.present_key.data(),
                        prefill.present_key.size() * sizeof(float)),
            0);
  EXPECT_EQ(std::memcmp(prefill.present_value.data(), decode.present_value.data(),
                        prefill.present_value.size() * sizeof(float)),
            0);
}

TEST(GroupQueryAttentionTest, WebGPU_BlockQuantInt8_NonPowerOfTwoHeadSize) {
  auto ep = WebGpuEPWithKVCacheQuantization(8);
  if (!ep) {
    GTEST_SKIP() << "WebGPU EP not available";
  }

  constexpr int batch_size = 1;
  constexpr int sequence_length = 4;
  constexpr int num_heads = 2;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 96;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int kv_hidden_size = kv_num_heads * head_size;
  std::mt19937 rng(8096);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> query_data(batch_size * sequence_length * hidden_size);
  std::vector<float> key_data(batch_size * sequence_length * kv_hidden_size);
  std::vector<float> value_data(batch_size * sequence_length * kv_hidden_size);
  for (float& value : query_data) value = dist(rng);
  for (float& value : key_data) value = dist(rng);
  for (float& value : value_data) value = dist(rng);

  const auto reference = RunGQAReference(batch_size, sequence_length, num_heads, kv_num_heads,
                                         head_size, query_data, key_data, value_data,
                                         /*do_rotary=*/false);
  const auto actual = RunGQATurboQuantNoPast(batch_size, sequence_length, num_heads,
                                             kv_num_heads, head_size, query_data, key_data,
                                             value_data, /*do_rotary=*/false,
                                             /*bit_width=*/8);
  ExpectBlockQuantInt8Close(reference, actual, /*max_relative_rmse=*/0.02f,
                            /*max_absolute_error=*/0.03f);
}

#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime
