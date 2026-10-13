// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <sstream>
#include <type_traits>

#include "core/common/narrow.h"
#include "core/graph/model.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/test_environment.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "test/contrib_ops/attention_op_test_helper.h"

namespace onnxruntime {
using contrib::AttentionMaskType;
namespace test {

namespace {

class ScopedStdoutCapture {
 public:
  explicit ScopedStdoutCapture(bool enabled) : capturing_(enabled) {
    if (capturing_) {
      testing::internal::CaptureStdout();
    }
  }

  ~ScopedStdoutCapture() {
    if (capturing_) {
      (void)testing::internal::GetCapturedStdout();
    }
  }

  std::string Stop() {
    capturing_ = false;
    return testing::internal::GetCapturedStdout();
  }

 private:
  bool capturing_;
};

template <typename Run>
void RunAndVerifyAttentionRoute(const char* expected_route, Run&& run) {
  ScopedStdoutCapture capture(true);
  run();
  const std::string debug_output = capture.Stop();
  EXPECT_NE(debug_output.find(expected_route), std::string::npos) << debug_output;
}

bool IsTrtFusedAttentionRouteObservable(int head_size, int sequence_length) {
#if USE_TRT_FUSED_ATTENTION
  if (!HasCudaEnvironment(0)) {
    return false;
  }

  // These tests use the non-flash branch of FusedMHARunnerFP16v2::IsSupported.
  // Keep its SM allowlist, head-size rules, and sequence cap in sync with mha_runner.cu.
  const int sm = GetCudaArchitecture() / 10;
  const bool supported_sm = sm == 70 || sm == 75 || sm == 80 ||
                            sm == 86 || sm == 89;
  return supported_sm &&
         (head_size == 32 || head_size == 64) &&
         !(sm == 70 && head_size == 32) &&
         sequence_length <= 384;
#else
  ORT_UNUSED_PARAMETER(head_size);
  ORT_UNUSED_PARAMETER(sequence_length);
  return false;
#endif
}

bool IsMemoryEfficientAttentionRouteObservable(int head_size) {
#if USE_MEMORY_EFFICIENT_ATTENTION
  return HasCudaEnvironment(530) &&
         head_size % 8 == 0 &&
         head_size <= 1024;
#else
  ORT_UNUSED_PARAMETER(head_size);
  return false;
#endif
}

enum class PackedAttentionRoute {
  Trt,
  MemoryEfficient,
  Unfused,
};

}  // namespace

static void RunPackedAttentionTest(
    const std::vector<float>& input_data,                    // input:      [token_count, hidden_size]
    const std::vector<float>& weights_data,                  // weights:    [hidden_size, 3 * hidden_size]
    const std::vector<float>& bias_data,                     // bias:       [3 * hidden_size]
    const std::vector<int32_t>& token_offset,                // token_offset: [batch_size, sequence_length]
    const std::vector<int32_t>& cumulative_sequence_length,  // cum_seq_len: [batch_size + 1]
    const std::vector<float>& output_data,                   // output:     [token_count, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    int token_count,
    bool use_float16,
    bool use_scale,
    std::vector<int32_t> qkv_sizes,
    const std::vector<float>& attention_bias_data) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);

  if (enable_cuda) {
    OpTester tester("PackedAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));

    int32_t qkv_hidden_size_sum;
    int32_t v_hidden_size;
    int32_t head_size_k;
    if (qkv_sizes.size() != 0) {
      qkv_hidden_size_sum = qkv_sizes[0] + qkv_sizes[1] + qkv_sizes[2];
      std::vector<int64_t> sizes_attribute{qkv_sizes[0], qkv_sizes[1], qkv_sizes[2]};
      tester.AddAttribute<std::vector<int64_t>>("qkv_hidden_sizes", sizes_attribute);
      v_hidden_size = qkv_sizes[2];
      head_size_k = qkv_sizes[1] / number_of_heads;
    } else {
      qkv_hidden_size_sum = 3 * hidden_size;
      v_hidden_size = hidden_size;
      head_size_k = hidden_size / number_of_heads;
    }

    if (use_scale) {
      tester.AddAttribute<float>("scale", static_cast<float>(1.f / sqrt(head_size_k)));
    }

    std::vector<int64_t> input_dims = {token_count, hidden_size};
    std::vector<int64_t> weights_dims = {hidden_size, qkv_hidden_size_sum};
    std::vector<int64_t> bias_dims = {qkv_hidden_size_sum};
    std::vector<int64_t> token_offset_dims = {batch_size, sequence_length};
    std::vector<int64_t> cum_seq_len_dims = {batch_size + 1};
    std::vector<int64_t> attention_bias_data_dims = {batch_size, number_of_heads, sequence_length, sequence_length};
    std::vector<int64_t> output_dims = {token_count, v_hidden_size};
    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      tester.AddInput<MLFloat16>("weight", weights_dims, ToFloat16(weights_data));
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      tester.AddInput<int32_t>("token_offset", token_offset_dims, token_offset);
      tester.AddInput<int32_t>("cumulative_sequence_length", cum_seq_len_dims, cumulative_sequence_length);
      if (attention_bias_data.size() > 0) {
        tester.AddInput<MLFloat16>("attention_bias", attention_bias_data_dims, ToFloat16(attention_bias_data));
      }

      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      tester.AddInput<float>("weight", weights_dims, weights_data);
      tester.AddInput<float>("bias", bias_dims, bias_data);
      tester.AddInput<int32_t>("token_offset", token_offset_dims, token_offset);
      tester.AddInput<int32_t>("cumulative_sequence_length", cum_seq_len_dims, cumulative_sequence_length);
      if (attention_bias_data.size() > 0) {
        tester.AddInput<float>("attention_bias", attention_bias_data_dims, attention_bias_data);
      }

      tester.AddOutput<float>("output", output_dims, output_data);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

static void RunPackedAttentionRouteTest(PackedAttentionRoute route) {
  constexpr int kBatchSize = 2;
  constexpr int kSequenceLength = 2;
  constexpr int kTokenCount = 3;
  constexpr int kHiddenSize = 32;
  constexpr int kNumHeads = 1;
  constexpr int kHeadSize = kHiddenSize / kNumHeads;
  static_assert(kTokenCount < kBatchSize * kSequenceLength);

  // Each token has input [1, x, 0, ...]. The block projection produces
  // Q=x+0.25, K=1, and V=2x-0.75 in every head coordinate. Since K is
  // constant within each sequence, attention is uniform: the one-token
  // sequence returns 1.25 and the two-token sequence averages 3.25 and 7.25.
  const std::vector<float> token_values{1.0f, 2.0f, 4.0f};
  std::vector<float> input_data(kTokenCount * kHiddenSize, 0.0f);
  std::vector<float> weight_data(kHiddenSize * 3 * kHiddenSize, 0.0f);
  std::vector<float> bias_data(3 * kHiddenSize, 0.0f);
  std::vector<float> output_data(kTokenCount * kHiddenSize);
  const std::vector<float> expected_token_values{1.25f, 5.25f, 5.25f};

  for (int token = 0; token < kTokenCount; ++token) {
    input_data[token * kHiddenSize] = 1.0f;
    input_data[token * kHiddenSize + 1] = token_values[token];
    for (int hidden = 0; hidden < kHiddenSize; ++hidden) {
      output_data[token * kHiddenSize + hidden] = expected_token_values[token];
    }
  }

  constexpr int kProjectionSize = 3 * kHiddenSize;
  for (int hidden = 0; hidden < kHiddenSize; ++hidden) {
    weight_data[kProjectionSize + hidden] = 1.0f;
    weight_data[kHiddenSize + hidden] = 0.5f;
    weight_data[kProjectionSize + 2 * kHiddenSize + hidden] = 2.0f;
    bias_data[hidden] = 0.25f;
    bias_data[kHiddenSize + hidden] = 0.5f;
    bias_data[2 * kHiddenSize + hidden] = -0.75f;
  }

  const std::vector<int32_t> token_offset{0, 2, 3, 1};
  const std::vector<int32_t> cumulative_sequence_length{0, 1, 3};

  if (route == PackedAttentionRoute::Trt) {
    if (!IsTrtFusedAttentionRouteObservable(kHeadSize, kSequenceLength)) {
      GTEST_SKIP() << "PackedAttention TRT route is unavailable in this build or CUDA environment.";
    }

    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};
    RunAndVerifyAttentionRoute("SdpaKernel=TRT_FUSED_ATTENTION", [&]() {
      RunPackedAttentionTest(
          input_data, weight_data, bias_data, token_offset, cumulative_sequence_length,
          output_data, kBatchSize, kSequenceLength, kHiddenSize, kNumHeads, kTokenCount,
          true, true, {}, {});
    });
    return;
  }

  if (route == PackedAttentionRoute::MemoryEfficient) {
    if (!IsMemoryEfficientAttentionRouteObservable(kHeadSize)) {
      GTEST_SKIP() << "PackedAttention MEA route is unavailable in this build or CUDA environment.";
    }

    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
            {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};
    RunAndVerifyAttentionRoute("SdpaKernel=EFFICIENT_ATTENTION", [&]() {
      RunPackedAttentionTest(
          input_data, weight_data, bias_data, token_offset, cumulative_sequence_length,
          output_data, kBatchSize, kSequenceLength, kHiddenSize, kNumHeads, kTokenCount,
          true, true, {}, {});
    });
    return;
  }

  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "PackedAttention MATH route requires a CUDA device.";
  }

  // PackedAttention does not honor the MEA-disable option. FP32 is deliberately
  // MEA-ineligible here, so this invocation deterministically
  // observes the unfused route without changing runtime behavior.
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};
  RunAndVerifyAttentionRoute("SdpaKernel=MATH", [&]() {
    RunPackedAttentionTest(
        input_data, weight_data, bias_data, token_offset, cumulative_sequence_length,
        output_data, kBatchSize, kSequenceLength, kHiddenSize, kNumHeads, kTokenCount,
        false, true, {}, {});
  });
}

static void RunPackedAttentionTest(
    const std::vector<float>& input_data,                    // input:      [token_count, hidden_size]
    const std::vector<float>& weights_data,                  // weights:    [hidden_size, 3 * hidden_size]
    const std::vector<float>& bias_data,                     // bias:       [3 * hidden_size]
    const std::vector<int32_t>& token_offset,                // token_offset: [batch_size, sequence_length]
    const std::vector<int32_t>& cumulative_sequence_length,  // cum_seq_len: [batch_size + 1]
    const std::vector<float>& output_data,                   // output:     [token_count, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    int token_count,
    std::vector<int32_t> qkv_sizes = {},
    const std::vector<float>& attention_bias_data = {}) {
#define InvokePackedAttentionTest(use_float16, use_scale) \
  RunPackedAttentionTest(                                 \
      input_data,                                         \
      weights_data,                                       \
      bias_data,                                          \
      token_offset,                                       \
      cumulative_sequence_length,                         \
      output_data,                                        \
      batch_size,                                         \
      sequence_length,                                    \
      hidden_size,                                        \
      number_of_heads,                                    \
      token_count,                                        \
      use_float16,                                        \
      use_scale,                                          \
      qkv_sizes,                                          \
      attention_bias_data);

  InvokePackedAttentionTest(true, true);
  InvokePackedAttentionTest(true, false);
  InvokePackedAttentionTest(false, true);
  InvokePackedAttentionTest(false, false);
}

TEST(PackedAttentionTest, PackedRouteObservedTrtWithPadding) {
  RunPackedAttentionRouteTest(PackedAttentionRoute::Trt);
}

TEST(PackedAttentionTest, PackedRouteObservedMemoryEfficientWithPadding) {
  RunPackedAttentionRouteTest(PackedAttentionRoute::MemoryEfficient);
}

TEST(PackedAttentionTest, PackedRouteObservedUnfusedWithPadding) {
  RunPackedAttentionRouteTest(PackedAttentionRoute::Unfused);
}

TEST(PackedAttentionTest, EmptyTokensAndSequence_CUDA) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "PackedAttention empty-output test requires a CUDA device.";
  }

  constexpr int kBatchSize = 2;
  constexpr int kHiddenSize = 32;
  constexpr int kNumHeads = 2;

  OpTester tester("PackedAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", kNumHeads);
  tester.AddInput<float>("input", {0, kHiddenSize}, {});
  tester.AddInput<float>("weight", {kHiddenSize, 3 * kHiddenSize},
                         std::vector<float>(kHiddenSize * 3 * kHiddenSize, 1.0f));
  tester.AddInput<float>("bias", {3 * kHiddenSize},
                         std::vector<float>(3 * kHiddenSize, 0.5f));
  tester.AddInput<int32_t>("token_offset", {kBatchSize, 0}, {});
  tester.AddInput<int32_t>("cumulative_sequence_length", {kBatchSize + 1}, {0, 0, 0});
  tester.AddOutput<float>("output", {0, kHiddenSize}, {});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(PackedAttentionTest, NoPack) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> token_offset{0, 1};
  std::vector<int32_t> cum_seq_len{0, 2};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunPackedAttentionTest(
      input_data,
      weight_data,
      bias_data,
      token_offset,
      cum_seq_len,
      output_data,
      batch_size,
      sequence_length,
      hidden_size,
      number_of_heads,
      batch_size * sequence_length);
}

TEST(PackedAttentionTest, NoPackWithAttentionBias) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> token_offset{0, 1, 2, 3};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  std::vector<float> attention_bias = {
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f,
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f};

  std::vector<float> output_data = {
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f,
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f};

  RunPackedAttentionTest(
      input_data,
      weight_data,
      bias_data,
      token_offset,
      cum_seq_len,
      output_data,
      batch_size,
      sequence_length,
      hidden_size,
      number_of_heads,
      batch_size * sequence_length,
      {},
      attention_bias);
}

TEST(PackedAttentionTest, PackedWithAttentionBias) {
  int batch_size = 2;
  int sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,   // b0:s0
      0.5f, 0.2f, 0.3f, -0.6f,  // b0:s1
      0.8f, -0.5f, 0.0f, 1.f,   // b1:s0
      0.5f, 0.2f, 0.3f, -0.6f   // b1:s1
  };

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> token_offset{0, 1, 4, 5, 2, 3, 6, 7};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  std::vector<float> attention_bias = {
      0.2f, -0.1f, 0.f, 0.f, 0.4f, 2.5f, 0.f, 0.f,
      1.6f, -1.1f, 0.f, 0.f, 0.4f, -2.5f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,

      0.2f, -0.1f, 0.f, 0.f, 0.4f, 2.5f, 0.f, 0.f,
      1.6f, -1.1f, 0.f, 0.f, 0.4f, -2.5f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  std::vector<float> output_data = {
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f,
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f};

  RunPackedAttentionTest(
      input_data,
      weight_data,
      bias_data,
      token_offset,
      cum_seq_len,
      output_data,
      batch_size,
      sequence_length,
      hidden_size,
      number_of_heads,
      4,
      {},
      attention_bias);
}

TEST(PackedAttentionTest, PackedBatch) {
  int batch_size = 2;
  int sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,   // b0:s0
      0.5f, 0.2f, 0.3f, -0.6f,  // b0:s1
      0.8f, -0.5f, 0.0f, 1.f,   // b1:s0
      0.5f, 0.2f, 0.3f, -0.6f   // b1:s1
  };

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> token_offset{0, 1, 4, 5, 2, 3, 6, 7};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunPackedAttentionTest(
      input_data,
      weight_data,
      bias_data,
      token_offset,
      cum_seq_len,
      output_data,
      batch_size,
      sequence_length,
      hidden_size,
      number_of_heads,
      4);
}

TEST(PackedAttentionTest, PackedBatchWithQKV) {
  int batch_size = 2;
  int sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,

      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<int32_t> qkv_sizes = {6, 6, 4};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f,

      0.3f, 0.2f, 4.0f, 2.2f, 2.4f, 3.3f, 2.1f, 4.2f, 0.5f, 0.1f, 0.4f, 1.6f,
      0.4f, 0.8f, 0.9f, 0.1f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f,
      0.5f, 0.7f, 0.2f, 1.2f};

  std::vector<int32_t> token_offset{0, 1, 4, 5, 2, 3, 6, 7};
  std::vector<int32_t> cum_seq_len{0, 2, 4};

  std::vector<float> output_data = {
      3.1967618465423584f, 0.51903456449508667f, 0.63051539659500122f, 2.9394614696502686f,
      0.65332180261611938f, 1.000949501991272f, 0.74175024032592773f, 2.8231701850891113f,

      3.1967618465423584f, 0.51903456449508667f, 0.63051539659500122f, 2.9394614696502686f,
      0.65332180261611938f, 1.000949501991272f, 0.74175024032592773f, 2.8231701850891113f};

  RunPackedAttentionTest(
      input_data,
      weight_data,
      bias_data,
      token_offset,
      cum_seq_len,
      output_data,
      batch_size,
      sequence_length,
      hidden_size,
      number_of_heads,
      4,
      qkv_sizes);
}

TEST(PackedAttentionTest, InvalidTokenOffset_Unfused_CUDA) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "PackedAttention token offset validation requires a CUDA device.";
  }

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"}}};

  constexpr int kTokenCount = 1;
  constexpr int kBatchSize = 1;
  constexpr int kSequenceLength = 2;
  constexpr int kHiddenSize = 8;

  for (const std::vector<int32_t>& token_offset : {
           std::vector<int32_t>{-1, 1},
           std::vector<int32_t>{0, kBatchSize * kSequenceLength}}) {
    SCOPED_TRACE(token_offset[0]);
    SCOPED_TRACE(token_offset[1]);

    OpTester tester("PackedAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", 1);
    tester.AddInput<float>("input", {kTokenCount, kHiddenSize},
                           std::vector<float>(kTokenCount * kHiddenSize));
    tester.AddInput<float>("weight", {kHiddenSize, 3 * kHiddenSize},
                           std::vector<float>(3 * kHiddenSize * kHiddenSize));
    tester.AddInput<float>("bias", {3 * kHiddenSize},
                           std::vector<float>(3 * kHiddenSize));
    tester.AddInput<int32_t>("token_offset", {kBatchSize, kSequenceLength}, token_offset);
    tester.AddInput<int32_t>("cumulative_sequence_length", {kBatchSize + 1}, {0, kTokenCount});
    tester.AddOutput<float>("output", {kTokenCount, kHiddenSize},
                            std::vector<float>(kTokenCount * kHiddenSize));

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(
        OpTester::ExpectResult::kExpectFailure,
        "PackedAttention token_offset values must be in [0, B * S).",
        {}, nullptr, &execution_providers);
  }
}

TEST(PackedAttentionTest, InvalidTokenOffset_UnfusedCudaGraphReplay) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "PackedAttention CUDA graph test requires a CUDA device.";
  }

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableTrtFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedSelfAttention, "1"},
          {onnxruntime::contrib::attention::kDisableFusedCrossAttention, "1"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"}}};

  constexpr int64_t kTokenCount = 2;
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kSequenceLength = 2;
  constexpr int64_t kHiddenSize = 8;

  auto make_tensor_type = [](int32_t element_type, std::initializer_list<int64_t> shape) {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor_type = type.mutable_tensor_type();
    tensor_type->set_elem_type(element_type);
    for (int64_t dimension : shape) {
      tensor_type->mutable_shape()->add_dim()->set_dim_value(dimension);
    }
    return type;
  };

  auto input_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {kTokenCount, kHiddenSize});
  auto weights_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {kHiddenSize, 3 * kHiddenSize});
  auto bias_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {3 * kHiddenSize});
  auto token_offset_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_INT32, {kBatchSize, kSequenceLength});
  auto cumulative_sequence_length_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_INT32, {kBatchSize + 1});
  auto output_type = make_tensor_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {kTokenCount, kHiddenSize});

  Model model("packed_attention_cuda_graph", true, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}, {kMSDomain, 1}},
              {}, DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  auto& graph = model.MainGraph();
  std::vector<NodeArg*> inputs{
      &graph.GetOrCreateNodeArg("input", &input_type),
      &graph.GetOrCreateNodeArg("weights", &weights_type),
      &graph.GetOrCreateNodeArg("bias", &bias_type),
      &graph.GetOrCreateNodeArg("token_offset", &token_offset_type),
      &graph.GetOrCreateNodeArg("cumulative_sequence_length", &cumulative_sequence_length_type)};
  std::vector<NodeArg*> outputs{
      &graph.GetOrCreateNodeArg("output", &output_type)};
  auto& node = graph.AddNode(
      "packed_attention", "PackedAttention", "", inputs, outputs, nullptr, kMSDomain);
  node.AddAttribute("num_heads", int64_t{1});
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.enable_cuda_graph = true;
  auto cuda_ep = CudaExecutionProviderWithOptions(&provider_options);
  ASSERT_NE(cuda_ep, nullptr);
  IExecutionProvider* cuda_ep_ptr = cuda_ep.get();

  SessionOptions session_options;
  InferenceSession session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
  std::istringstream model_stream(model_data);
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());

  auto gpu_allocators = cuda_ep_ptr->CreatePreferredAllocators();
  auto gpu_allocator = std::find_if(gpu_allocators.begin(), gpu_allocators.end(), [](const auto& allocator) {
    return allocator->Info().device.Type() == OrtDevice::GPU &&
           allocator->Info().alloc_type != OrtAllocatorType::OrtReadOnlyAllocator &&
           allocator->Info().mem_type == OrtMemTypeDefault;
  });
  ASSERT_NE(gpu_allocator, gpu_allocators.end());
  auto allocator = session.GetAllocator((*gpu_allocator)->Info());
  ASSERT_NE(allocator, nullptr);
  auto cpu_allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  auto make_gpu_value = [&](const auto& values, const TensorShape& shape) {
    using Element = typename std::decay_t<decltype(values)>::value_type;
    Tensor cpu_tensor(
        DataTypeImpl::GetType<Element>(), shape,
        const_cast<Element*>(values.data()), cpu_allocator->Info());
    Tensor gpu_tensor(DataTypeImpl::GetType<Element>(), shape, allocator);
    ORT_THROW_IF_ERROR(cuda_ep_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue result;
    Tensor::InitOrtValue(std::move(gpu_tensor), result);
    return result;
  };

  auto input = make_gpu_value(
      std::vector<float>(kTokenCount * kHiddenSize),
      TensorShape{kTokenCount, kHiddenSize});
  auto weights = make_gpu_value(
      std::vector<float>(kHiddenSize * 3 * kHiddenSize),
      TensorShape{kHiddenSize, 3 * kHiddenSize});
  std::vector<float> bias_data(3 * kHiddenSize);
  std::fill(bias_data.begin() + 2 * kHiddenSize, bias_data.end(), 1.0f);
  auto bias = make_gpu_value(bias_data, TensorShape{3 * kHiddenSize});
  auto token_offset = make_gpu_value(
      std::vector<int32_t>{0, 1}, TensorShape{kBatchSize, kSequenceLength});
  auto cumulative_sequence_length = make_gpu_value(
      std::vector<int32_t>{0, 2}, TensorShape{kBatchSize + 1});
  Tensor output_tensor(
      DataTypeImpl::GetType<float>(),
      TensorShape{kTokenCount, kHiddenSize}, allocator);
  OrtValue output;
  Tensor::InitOrtValue(std::move(output_tensor), output);

  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("input", input));
  ASSERT_STATUS_OK(binding->BindInput("weights", weights));
  ASSERT_STATUS_OK(binding->BindInput("bias", bias));
  ASSERT_STATUS_OK(binding->BindInput("token_offset", token_offset));
  ASSERT_STATUS_OK(binding->BindInput("cumulative_sequence_length", cumulative_sequence_length));
  ASSERT_STATUS_OK(binding->BindOutput("output", output));
  ASSERT_STATUS_OK(binding->SynchronizeInputs());

  RunOptions run_options;
  ASSERT_STATUS_OK(run_options.config_options.AddConfigEntry("gpu_graph_id", "1"));
  for (int i = 0; i < 3; ++i) {
    ASSERT_STATUS_OK(session.Run(run_options, *binding));
  }
  ASSERT_TRUE(cuda_ep_ptr->IsGraphCaptured(1));

  const std::vector<int32_t> invalid_offsets{-1, kBatchSize * kSequenceLength};
  Tensor invalid_offsets_tensor(
      DataTypeImpl::GetType<int32_t>(), TensorShape{kBatchSize, kSequenceLength},
      const_cast<int32_t*>(invalid_offsets.data()), cpu_allocator->Info());
  ASSERT_STATUS_OK(cuda_ep_ptr->GetDataTransfer()->CopyTensor(
      invalid_offsets_tensor, *token_offset.GetMutable<Tensor>()));
  ASSERT_STATUS_OK(binding->SynchronizeInputs());
  ASSERT_STATUS_OK(session.Run(run_options, *binding));
  ASSERT_STATUS_OK(binding->SynchronizeOutputs());

  Tensor cpu_output(
      DataTypeImpl::GetType<float>(),
      TensorShape{kTokenCount, kHiddenSize}, cpu_allocator);
  ASSERT_STATUS_OK(cuda_ep_ptr->GetDataTransfer()->CopyTensor(
      output.Get<Tensor>(), cpu_output));
  for (float value : cpu_output.DataAsSpan<float>()) {
    EXPECT_EQ(value, 0.0f);
  }
}

static void RunModelWithRandomInput(
    int64_t batch_size,
    int64_t sequence_length,
    std::string& onnx_model,
    bool is_float16,
    bool has_rbp = false) {
  RandomValueGenerator random{234};

  constexpr int hidden_size = 768;
  constexpr int num_heads = 12;

  int64_t token_count = 0;
  std::vector<int32_t> cum_seq_len(batch_size + 1);
  cum_seq_len[0] = 0;

  int64_t original_offset = 0;
  int64_t token_offset_idx = 0;
  std::vector<int32_t> token_offset(batch_size * sequence_length);
  for (int64_t b = 0; b < batch_size; b++) {
    int64_t actual_seq_len = (sequence_length / (b + 1));
    token_count += actual_seq_len;
    cum_seq_len[b + 1] = narrow<int32_t>(token_count);

    original_offset = b * sequence_length;
    for (int64_t s = 0; s < actual_seq_len; s++) {
      token_offset[token_offset_idx++] = narrow<int32_t>(original_offset++);
    }
  }

  for (int64_t b = 0; b < batch_size; b++) {
    int64_t actual_seq_len = (sequence_length / (b + 1));
    original_offset = b * sequence_length + actual_seq_len;
    for (int64_t s = actual_seq_len; s < sequence_length; s++) {
      token_offset[token_offset_idx++] = narrow<int32_t>(original_offset++);
    }
  }

  assert(token_offset_idx == batch_size * sequence_length);

  std::vector<int64_t> input_dims{token_count, hidden_size};
  std::vector<float> input_data = random.Gaussian<float>(input_dims, 0.0f, 0.3f);

  std::vector<int64_t> weight_dims{hidden_size, 3 * hidden_size};
  std::vector<float> weight_data = random.Gaussian<float>(weight_dims, 0.0f, 0.3f);

  std::vector<int64_t> bias_dims{3 * hidden_size};
  std::vector<float> bias_data = random.Gaussian<float>(bias_dims, 0.0f, 0.1f);

  std::vector<int64_t> token_offset_dims{batch_size, sequence_length};
  std::vector<int64_t> cum_seq_len_dims{batch_size + 1};

  float gpu_threshold = is_float16 ? 0.15f : 0.005f;
  gpu_threshold *= sequence_length > 1024 ? 4.0f : 1.0f;  // threshold should increase with sequence length
  bool enable_cuda = HasCudaEnvironment(is_float16 ? 530 : 0);
  if (enable_cuda) {
    OpTester test("PackedAttention", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    if (is_float16) {
      test.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      test.AddInput<MLFloat16>("weight", weight_dims, ToFloat16(weight_data));
      test.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    } else {
      test.AddInput<float>("input", input_dims, input_data);
      test.AddInput<float>("weight", weight_dims, weight_data);
      test.AddInput<float>("bias", bias_dims, bias_data);
    }
    test.AddInput<int32_t>("token_offset", token_offset_dims, token_offset);
    test.AddInput<int32_t>("cumulative_sequence_length", cum_seq_len_dims, cum_seq_len);

    if (has_rbp) {
      std::vector<int64_t> rbp_dims{1, num_heads, sequence_length, sequence_length};
      std::vector<float> rbp_data = random.Gaussian<float>(rbp_dims, 0.0f, 0.1f);
      if (is_float16) {
        test.AddInput<MLFloat16>("rbp", rbp_dims, ToFloat16(rbp_data));
      } else {
        test.AddInput<float>("rbp", rbp_dims, rbp_data);
      }
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    test.AddReferenceOutputs(onnx_model, gpu_threshold, DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(PackedAttentionTest, TestWithRandomData) {
  std::string onnx_model = "testdata/packed_attention_fp32.onnx";
  std::string onnx_model_fp16 = "testdata/packed_attention_fp16.onnx";
  for (int batch_size : std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8})) {
    for (int sequence_length : std::vector<int>({32, 48, 64, 95, 128})) {
      RunModelWithRandomInput(
          batch_size,
          sequence_length,
          onnx_model,
          false);
      RunModelWithRandomInput(
          batch_size,
          sequence_length,
          onnx_model_fp16,
          true);
    }
  }
}

TEST(PackedAttentionTest, TestWithRandomDataWithRBP) {
  std::string onnx_model_fp16 = "testdata/packed_attention_fp16.rbp.onnx";  // mainly for cutlass
  for (int batch_size : std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8})) {
    for (int sequence_length : std::vector<int>({32, 48, 64, 95, 128})) {
      RunModelWithRandomInput(
          batch_size,
          sequence_length,
          onnx_model_fp16,
          true /*is_float16*/,
          true /*has_rbp*/);
    }
  }
}

TEST(PackedAttentionTest, TestWithRandomDataLargeSeq) {
  int batch_size = 2;
  int sequence_length = 1152;  // > 1024
  std::string onnx_model = "testdata/packed_attention_fp32.onnx";
  std::string onnx_model_fp16 = "testdata/packed_attention_fp16.onnx";
  RunModelWithRandomInput(
      batch_size,
      sequence_length,
      onnx_model,
      false);
  RunModelWithRandomInput(
      batch_size,
      sequence_length,
      onnx_model_fp16,
      true);
}

}  // namespace test
}  // namespace onnxruntime
