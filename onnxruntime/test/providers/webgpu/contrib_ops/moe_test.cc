// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/common/float8.h"
#endif
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#if defined(USE_WEBGPU)
#include "contrib_ops/webgpu/moe/block_fp8_expert_matmul.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/webgpu_context.h"
#endif

namespace onnxruntime {
namespace test {

#ifndef ENABLE_TRAINING
#if defined(USE_WEBGPU)
// This must be used before constructing OpTester: an unexecuted OpTester traps in Debug builds.
#define GET_WEBGPU_EP_OR_SKIP(name)                               \
  auto name = DefaultWebGpuExecutionProvider();                   \
  if (!name) {                                                    \
    GTEST_SKIP() << "WebGPU execution provider is not available"; \
  }

static void RunWebGpuOnly(OpTester& tester, std::unique_ptr<IExecutionProvider> webgpu_ep,
                          OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                          const std::string& expected_error = {}) {
  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(webgpu_ep));
  tester.Run(session_options, expect_result, expected_error, {}, nullptr, &providers);
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_MatrixDispatchLimit) {
  constexpr uint32_t limit = 65535;
  EXPECT_TRUE(contrib::webgpu::BlockFp8MatrixDispatchFits(128, limit * 16, limit));
  EXPECT_FALSE(contrib::webgpu::BlockFp8MatrixDispatchFits(128, limit * 16 + 1, limit));
  EXPECT_LT((limit * 16 + 1 + 63) / 64, limit);
  EXPECT_FALSE(contrib::webgpu::BlockFp8MatrixDispatchFits(limit * 64 + 1, 256, limit));
  EXPECT_FALSE(contrib::webgpu::BlockFp8MatrixDispatchFits(128, UINT32_MAX, limit));
}

// Packed QMoE does not need cumulative sequence lengths: every token is routed and evaluated
// independently. This fixture represents three requests of lengths [2, 1, 2] concatenated into a
// single 2D token-major input. Distinct token values and nonzero weights make each output depend on
// the corresponding hidden-state row, while expert-specific biases expose the selected expert.
TEST(MoETest, QMoETest_WebGPU_PackedRaggedBatch) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_rows = 5;
  constexpr int num_experts = 2;
  constexpr int hidden_size = 64;
  constexpr int inter_size = 64;

  const std::array<float, num_rows> token_values = {0.05f, 0.1f, -0.05f, 0.2f, -0.1f};
  std::vector<float> input;
  input.reserve(num_rows * hidden_size);
  for (float token_value : token_values) {
    input.insert(input.end(), hidden_size, token_value);
  }
  const std::vector<float> router_probs = {
      10.0f,
      0.0f,
      0.0f,
      10.0f,
      0.0f,
      10.0f,
      10.0f,
      0.0f,
      0.0f,
      10.0f,
  };

  // 0x99 encodes two INT4 values one above the default zero point, so both projections are nonzero.
  std::vector<uint8_t> fc1_experts_weights(num_experts * 2 * inter_size * hidden_size / 2, 0x99);
  std::vector<uint8_t> fc2_experts_weights(num_experts * hidden_size * inter_size / 2, 0x99);
  std::vector<float> fc1_scales(num_experts * 2 * inter_size, 0.01f);
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.01f);
  std::vector<float> fc2_bias(hidden_size, 1.0f);
  fc2_bias.insert(fc2_bias.end(), hidden_size, 2.0f);

  std::vector<float> expected_output;
  expected_output.reserve(num_rows * hidden_size);
  const std::array<float, num_rows> expert_biases = {1.0f, 2.0f, 2.0f, 1.0f, 2.0f};
  for (int row = 0; row < num_rows; ++row) {
    const float projection = hidden_size * token_values[row] * 0.01f;
    const float swiglu = projection / (1.0f + std::exp(-projection)) * projection;
    const float expert_output = inter_size * swiglu * 0.01f + expert_biases[row];
    expected_output.insert(expected_output.end(), hidden_size, expert_output);
  }

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 4);

  tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
  tester.AddInput<uint8_t>("fc1_experts_weights",
                           {num_experts, 2 * inter_size, hidden_size / 2},
                           fc1_experts_weights);
  tester.AddInput<MLFloat16>("fc1_scales",
                             {num_experts, 2 * inter_size},
                             ToFloat16(fc1_scales));
  tester.AddOptionalInputEdge<MLFloat16>();  // fc1_experts_bias
  tester.AddInput<uint8_t>("fc2_experts_weights",
                           {num_experts, hidden_size, inter_size / 2},
                           fc2_experts_weights);
  tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size}, ToFloat16(fc2_scales));
  tester.AddInput<MLFloat16>("fc2_experts_bias", {num_experts, hidden_size}, ToFloat16(fc2_bias));
  tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights
  tester.AddOptionalInputEdge<MLFloat16>();  // fc3_scales
  tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
  tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, ToFloat16(expected_output));
  tester.SetOutputTolerance(0.01f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

// Test QMoE with num_rows=1 on WebGPU to exercise the fused 1-token decode path.
// Uses SwiGLU activation without FC3 (2-gate fused in FC1), which is the configuration
// used by real MoE models on WebGPU (e.g., gpt-oss-20b).
TEST(MoETest, QMoETest_WebGPU_SingleToken) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  int num_rows = 1;
  int num_experts = 2;
  int hidden_size = 64;
  int inter_size = 64;
  int top_k = 2;

  // Simple input — first row from QMoETest_Mixtral_Int4
  const std::vector<float> input = {
      -0.8477f, -0.0746f, 1.606f, -0.3242f, 0.4028f, 0.2384f, -0.0359f, -1.667f, -1.265f, -0.3035f, 0.5327f,
      1.109f, 1.111f, 0.533f, -0.5947f, -0.2009f, 0.4224f, -0.576f, 0.825f, 1.038f, -0.2722f, 0.0497f,
      1.963f, -1.075f, -0.8374f, 1.055f, 0.448f, -0.602f, -0.2874f, -1.311f, -0.0609f, -1.991f, -0.0732f,
      -1.49f, 0.6636f, -0.4053f, -1.603f, -1.088f, 0.09534f, -0.6807f, -0.3958f, 1.205f, -0.4275f, 0.82f,
      1.029f, 0.2693f, 1.229f, 1.116f, 0.718f, -0.827f, 2.527f, -1.041f, 1.042f, -2.771f, -0.654f,
      0.7144f, 0.6255f, -0.00957f, -0.2313f, 0.4663f, 2.803f, 0.0655f, 1.232f, 1.557f};
  const std::vector<float> router_probs = {-0.579f, -0.07007f};

  // Zero weights (0x88 for 4-bit = signed 0,0) produce zero output through the SwiGLU path:
  // FC1 output = 0 → SwiGLU(0, 0) = silu(0) * 0 = 0 → FC2 output = 0 → FinalMix = 0
  // FC1 weights: {num_experts, 2*inter_size, hidden_size/2} for 4-bit SwiGLU fusion
  std::vector<uint8_t> fc1_experts_weights(num_experts * 2 * inter_size * hidden_size / 2, 0x88);
  // FC2 weights: {num_experts, hidden_size, inter_size/2} for 4-bit
  std::vector<uint8_t> fc2_experts_weights(num_experts * hidden_size * inter_size / 2, 0x88);

  std::vector<float> fc1_scales(num_experts * 2 * inter_size, 0.01f);
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.01f);

  std::vector<float> expected_output(num_rows * hidden_size, 0.0f);

  OpTester webgpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  webgpu_tester.AddAttribute<int64_t>("k", static_cast<int64_t>(top_k));
  webgpu_tester.AddAttribute<std::string>("activation_type", "swiglu");
  webgpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  webgpu_tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  webgpu_tester.AddAttribute<int64_t>("expert_weight_bits", 4);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, 2 * inter_size, hidden_size / 2};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size / 2};
  std::vector<int64_t> fc1_scales_dims = {num_experts, 2 * inter_size};
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  webgpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  webgpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  webgpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  webgpu_tester.AddInput<MLFloat16>("fc1_scales", fc1_scales_dims, ToFloat16(fc1_scales));
  webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc1_experts_bias
  webgpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  webgpu_tester.AddInput<MLFloat16>("fc2_scales", fc2_scales_dims, ToFloat16(fc2_scales));
  webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc2_experts_bias
  webgpu_tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights
  webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_scales
  webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
  webgpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(expected_output));
  webgpu_tester.SetOutputTolerance(0.01f);

  RunWebGpuOnly(webgpu_tester, std::move(webgpu_ep));
}

// Regression test for issue where large router logits (e.g. one-hot @ 100) caused
// the WebGPU QMoE gate shader's softmax to overflow (exp(100) = inf, inf/inf = NaN),
// turning the entire output into NaN. CPU stays finite because it uses the
// log-sum-exp trick. After the fix, the gate shaders subtract max_val before exp(),
// so the output should match the CPU result (here, all-zero from zeroed weights).
TEST(MoETest, QMoETest_WebGPU_SingleToken_LargeLogits) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  int num_rows = 1;
  int num_experts = 2;
  int hidden_size = 64;
  int inter_size = 64;
  int top_k = 1;

  std::vector<float> input(num_rows * hidden_size, 0.1f);

  // One-hot router logit at 100 — pre-fix this overflows exp() and yields NaN.
  std::vector<float> router_probs = {100.0f, 0.0f};

  // 0x88 weights decode to signed 0 in 4-bit, so FC1 output is 0 regardless of input.
  // SwiGLU(0,0) * scale = 0, FC2 output = 0, FinalMix output = 0 — independent of
  // the router probability, as long as the router probabilities are finite.
  std::vector<uint8_t> fc1_experts_weights(num_experts * 2 * inter_size * hidden_size / 2, 0x88);
  std::vector<uint8_t> fc2_experts_weights(num_experts * hidden_size * inter_size / 2, 0x88);

  std::vector<float> fc1_scales(num_experts * 2 * inter_size, 0.01f);
  std::vector<float> fc2_scales(num_experts * hidden_size, 0.01f);

  std::vector<float> expected_output(num_rows * hidden_size, 0.0f);

  OpTester webgpu_tester("QMoE", 1, onnxruntime::kMSDomain);
  webgpu_tester.AddAttribute<int64_t>("k", static_cast<int64_t>(top_k));
  webgpu_tester.AddAttribute<std::string>("activation_type", "swiglu");
  webgpu_tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  webgpu_tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  webgpu_tester.AddAttribute<int64_t>("expert_weight_bits", 4);

  std::vector<int64_t> input_dims = {num_rows, hidden_size};
  std::vector<int64_t> router_probs_dims = {num_rows, num_experts};
  std::vector<int64_t> fc1_experts_weights_dims = {num_experts, 2 * inter_size, hidden_size / 2};
  std::vector<int64_t> fc2_experts_weights_dims = {num_experts, hidden_size, inter_size / 2};
  std::vector<int64_t> fc1_scales_dims = {num_experts, 2 * inter_size};
  std::vector<int64_t> fc2_scales_dims = {num_experts, hidden_size};
  std::vector<int64_t> output_dims = {num_rows, hidden_size};

  webgpu_tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
  webgpu_tester.AddInput<MLFloat16>("router_probs", router_probs_dims, ToFloat16(router_probs));
  webgpu_tester.AddInput<uint8_t>("fc1_experts_weights", fc1_experts_weights_dims, fc1_experts_weights);
  webgpu_tester.AddInput<MLFloat16>("fc1_scales", fc1_scales_dims, ToFloat16(fc1_scales));
  webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc1_experts_bias
  webgpu_tester.AddInput<uint8_t>("fc2_experts_weights", fc2_experts_weights_dims, fc2_experts_weights);
  webgpu_tester.AddInput<MLFloat16>("fc2_scales", fc2_scales_dims, ToFloat16(fc2_scales));
  webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc2_experts_bias
  webgpu_tester.AddOptionalInputEdge<uint8_t>();    // fc3_experts_weights
  webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_scales
  webgpu_tester.AddOptionalInputEdge<MLFloat16>();  // fc3_experts_bias
  webgpu_tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(expected_output));
  webgpu_tester.SetOutputTolerance(0.01f);

  RunWebGpuOnly(webgpu_tester, std::move(webgpu_ep));
}

TEST(MoETest, MoETest_WebGPU_PackedDenseActivationsAndFusion) {
  constexpr int num_rows = 2;
  constexpr int num_experts = 2;
  constexpr int hidden_size = 2;
  constexpr int inter_size = 2;

  const std::vector<float> input = {-1.0f, 0.5f, 1.5f, -2.0f};
  const std::vector<float> router_probs = {10.0f, 0.0f, 0.0f, 10.0f};

  const auto run_case = [&](const std::string& activation_type, int64_t swiglu_fusion,
                            bool has_fc3, bool use_fp16, bool dense_3d) {
    GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

    const int fc1_size = activation_type == "swiglu" && !has_fc3 ? 2 * inter_size : inter_size;
    std::vector<float> fc1_weights(num_experts * fc1_size * hidden_size, 0.0f);
    std::vector<float> fc2_weights(num_experts * hidden_size * inter_size, 0.0f);
    std::vector<float> fc3_weights(num_experts * inter_size * hidden_size, 0.0f);
    for (int expert = 0; expert < num_experts; ++expert) {
      const int fc1_base = expert * fc1_size * hidden_size;
      if (activation_type == "swiglu" && !has_fc3) {
        const std::array<int, 2> gate_rows = swiglu_fusion == 1 ? std::array<int, 2>{0, 2}
                                                                : std::array<int, 2>{0, 1};
        const std::array<int, 2> linear_rows = swiglu_fusion == 1 ? std::array<int, 2>{1, 3}
                                                                  : std::array<int, 2>{2, 3};
        for (int col = 0; col < hidden_size; ++col) {
          fc1_weights[fc1_base + gate_rows[col] * hidden_size + col] = 1.0f;
          fc1_weights[fc1_base + linear_rows[col] * hidden_size + col] = 2.0f;
        }
      } else {
        for (int col = 0; col < hidden_size; ++col) {
          fc1_weights[fc1_base + col * hidden_size + col] = 1.0f;
        }
      }
      for (int col = 0; col < hidden_size; ++col) {
        fc2_weights[expert * hidden_size * inter_size + col * inter_size + col] = 1.0f;
        fc3_weights[expert * inter_size * hidden_size + col * hidden_size + col] = 2.0f;
      }
    }

    const auto activate = [&](float value) {
      if (activation_type == "relu") {
        return std::max(value, 0.0f);
      }
      if (activation_type == "gelu") {
        return 0.5f * value *
               (1.0f + std::tanh(0.7978845608f * (value + 0.044715f * value * value * value)));
      }
      if (activation_type == "silu" || activation_type == "swiglu") {
        return value / (1.0f + std::exp(-value));
      }
      return value;
    };
    std::vector<float> expected;
    expected.reserve(input.size());
    for (float value : input) {
      float result = activate(value);
      if (has_fc3 || activation_type == "swiglu") {
        result *= 2.0f * value;
      }
      expected.push_back(result);
    }
    const std::vector<int64_t> input_dims = dense_3d ? std::vector<int64_t>{1, num_rows, hidden_size}
                                                     : std::vector<int64_t>{num_rows, hidden_size};

    OpTester tester("MoE", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("k", 1);
    tester.AddAttribute<std::string>("activation_type", activation_type);
    tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
    tester.AddAttribute<int64_t>("swiglu_fusion", swiglu_fusion);
    if (use_fp16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
      tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
      tester.AddInput<MLFloat16>("fc1_experts_weights", {num_experts, fc1_size, hidden_size},
                                 ToFloat16(fc1_weights));
      tester.AddOptionalInputEdge<MLFloat16>();
      tester.AddInput<MLFloat16>("fc2_experts_weights", {num_experts, hidden_size, inter_size},
                                 ToFloat16(fc2_weights));
      tester.AddOptionalInputEdge<MLFloat16>();
      if (has_fc3) {
        tester.AddInput<MLFloat16>("fc3_experts_weights", {num_experts, inter_size, hidden_size},
                                   ToFloat16(fc3_weights));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }
      tester.AddOptionalInputEdge<MLFloat16>();
      tester.AddOutput<MLFloat16>("output", input_dims, ToFloat16(expected));
      tester.SetOutputTolerance(0.01f);
    } else {
      tester.AddInput<float>("input", input_dims, input);
      tester.AddInput<float>("router_probs", {num_rows, num_experts}, router_probs);
      tester.AddInput<float>("fc1_experts_weights", {num_experts, fc1_size, hidden_size}, fc1_weights);
      tester.AddOptionalInputEdge<float>();
      tester.AddInput<float>("fc2_experts_weights", {num_experts, hidden_size, inter_size}, fc2_weights);
      tester.AddOptionalInputEdge<float>();
      if (has_fc3) {
        tester.AddInput<float>("fc3_experts_weights", {num_experts, inter_size, hidden_size}, fc3_weights);
      } else {
        tester.AddOptionalInputEdge<float>();
      }
      tester.AddOptionalInputEdge<float>();
      tester.AddOutput<float>("output", input_dims, expected);
      tester.SetOutputTolerance(0.001f);
    }
    RunWebGpuOnly(tester, std::move(webgpu_ep));
  };

  run_case("relu", 0, false, false, false);
  run_case("gelu", 0, false, true, true);
  run_case("silu", 0, true, true, false);
  run_case("identity", 0, false, false, false);
  run_case("swiglu", 0, true, true, false);
  run_case("swiglu", 1, false, true, false);
  run_case("swiglu", 2, false, true, false);
}

TEST(MoETest, MoETest_WebGPU_NonzeroExpertMatMuls) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int hidden_size = 8;
  constexpr int inter_size = 8;
  const std::vector<float> input = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
  std::vector<float> identity(hidden_size * hidden_size, 0.0f);
  for (int i = 0; i < hidden_size; ++i) {
    identity[i * hidden_size + i] = 1.0f;
  }

  OpTester tester("MoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddInput<float>("input", {1, hidden_size}, input);
  tester.AddInput<float>("router_probs", {1, 1}, {0.0f});
  tester.AddInput<float>("fc1_experts_weights", {1, inter_size, hidden_size}, identity);
  tester.AddOptionalInputEdge<float>();
  tester.AddInput<float>("fc2_experts_weights", {1, hidden_size, inter_size}, identity);
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOutput<float>("output", {1, hidden_size}, input);
  tester.SetOutputTolerance(0.001f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, MoETest_WebGPU_NonAlignedGatherMultipleRows) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_rows = 2;
  constexpr int num_experts = 2;
  constexpr int hidden_size = 65;
  constexpr int inter_size = 65;

  std::vector<float> input(num_rows * hidden_size);
  for (int index = 0; index < num_rows * hidden_size; ++index) {
    input[index] = static_cast<float>((index % 17) - 8) / 8.0f;
  }
  std::vector<float> weights(num_experts * hidden_size * hidden_size, 0.0f);
  for (int expert = 0; expert < num_experts; ++expert) {
    for (int index = 0; index < hidden_size; ++index) {
      weights[expert * hidden_size * hidden_size + index * hidden_size + index] = 1.0f;
    }
  }

  OpTester tester("MoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddInput<float>("input", {num_rows, hidden_size}, input);
  tester.AddInput<float>("router_probs", {num_rows, num_experts}, {10.0f, 0.0f, 10.0f, 0.0f});
  tester.AddInput<float>("fc1_experts_weights", {num_experts, inter_size, hidden_size}, weights);
  tester.AddOptionalInputEdge<float>();
  tester.AddInput<float>("fc2_experts_weights", {num_experts, hidden_size, inter_size}, weights);
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOutput<float>("output", {num_rows, hidden_size}, input);
  tester.SetOutputTolerance(0.001f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, MoETest_WebGPU_ChunkBoundary) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_rows = 2049;
  constexpr int hidden_size = 1;
  // The first 2,048-token chunk requests 131,072 FC1 workgroups and 65,536
  // activation workgroups. This covers normalized dispatch indexing when ORT
  // reshapes those logical 1-D counts into legal physical dimensions.
  constexpr int inter_size = 4096;

  std::vector<float> input(num_rows * hidden_size);
  for (int index = 0; index < num_rows * hidden_size; ++index) {
    input[index] = static_cast<float>((index % 13) - 6) / 8.0f;
  }
  const std::vector<float> fc1_weights(inter_size * hidden_size, 1.0f);
  const std::vector<float> fc2_weights(hidden_size * inter_size, 1.0f / inter_size);

  OpTester tester("MoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddInput<float>("input", {num_rows, hidden_size}, input);
  tester.AddInput<float>("router_probs", {num_rows, 1}, std::vector<float>(num_rows, 0.0f));
  tester.AddInput<float>("fc1_experts_weights", {1, inter_size, hidden_size}, fc1_weights);
  tester.AddOptionalInputEdge<float>();
  tester.AddInput<float>("fc2_experts_weights", {1, hidden_size, inter_size}, fc2_weights);
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOutput<float>("output", {num_rows, hidden_size}, input);
  tester.SetOutputTolerance(0.001f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

static void RunWebGpuExpertLimitTest(bool quantized, bool empty) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_experts = 1025;
  constexpr int hidden_size = 64;
  constexpr int inter_size = 64;
  const int num_rows = empty ? 0 : 1;
  const std::vector<float> input(static_cast<size_t>(num_rows * hidden_size), 0.0f);
  const std::vector<float> router_probs(static_cast<size_t>(num_rows * num_experts), 0.0f);
  const std::vector<float> expected(static_cast<size_t>(num_rows * hidden_size), 0.0f);

  OpTester tester(quantized ? "QMoE" : "MoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  if (quantized) {
    tester.AddAttribute<int64_t>("expert_weight_bits", 4);
    tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, ToFloat16(input));
    tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
    tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, inter_size, hidden_size / 2},
                             std::vector<uint8_t>(num_experts * inter_size * hidden_size / 2, 0x88));
    tester.AddInput<MLFloat16>("fc1_scales", {num_experts, inter_size},
                               ToFloat16(std::vector<float>(num_experts * inter_size, 0.1f)));
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, hidden_size, inter_size / 2},
                             std::vector<uint8_t>(num_experts * hidden_size * inter_size / 2, 0x88));
    tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size},
                               ToFloat16(std::vector<float>(num_experts * hidden_size, 0.1f)));
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<uint8_t>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, ToFloat16(expected));
  } else {
    tester.AddInput<float>("input", {num_rows, hidden_size}, input);
    tester.AddInput<float>("router_probs", {num_rows, num_experts}, router_probs);
    tester.AddInput<float>("fc1_experts_weights", {num_experts, inter_size, hidden_size},
                           std::vector<float>(num_experts * inter_size * hidden_size, 0.0f));
    tester.AddOptionalInputEdge<float>();
    tester.AddInput<float>("fc2_experts_weights", {num_experts, hidden_size, inter_size},
                           std::vector<float>(num_experts * hidden_size * inter_size, 0.0f));
    tester.AddOptionalInputEdge<float>();
    tester.AddOptionalInputEdge<float>();
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>("output", {num_rows, hidden_size}, expected);
  }

  RunWebGpuOnly(tester, std::move(webgpu_ep),
                empty ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
                empty ? "" : "requires num_experts to fit in one workgroup");
}

TEST(MoETest, MoETest_WebGPU_ExpertLimit) {
  RunWebGpuExpertLimitTest(false, false);
}

TEST(MoETest, MoETest_WebGPU_EmptyInputAboveExpertLimit) {
  RunWebGpuExpertLimitTest(false, true);
}

TEST(MoETest, QMoETest_WebGPU_ExpertLimit) {
  RunWebGpuExpertLimitTest(true, false);
}

TEST(MoETest, QMoETest_WebGPU_EmptyInputAboveExpertLimit) {
  RunWebGpuExpertLimitTest(true, true);
}

TEST(MoETest, QMoETest_WebGPU_WideTileExpertZeroPointsAndFC3) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_rows = 4;
  constexpr int num_experts = 2;
  constexpr int hidden_size = 64;
  constexpr int inter_size = 64;
  constexpr int pack_size = 2;

  const std::vector<float> input(num_rows * hidden_size, 0.25f);
  const std::vector<float> router_probs = {
      0.0f, 10.0f,
      0.0f, 10.0f,
      0.0f, 10.0f,
      0.0f, 10.0f};
  const std::vector<uint8_t> fc1_weights(num_experts * inter_size * hidden_size / pack_size, 0x99);
  const std::vector<uint8_t> fc2_weights(num_experts * hidden_size * inter_size / pack_size, 0x99);
  const std::vector<uint8_t> fc3_weights(num_experts * inter_size * hidden_size / pack_size, 0x99);
  const std::vector<float> fc1_scales(num_experts * inter_size * 2, 0.1f);
  const std::vector<float> fc2_scales(num_experts * hidden_size * 2, 0.1f);
  const std::vector<float> fc3_scales(num_experts * inter_size * 2, 0.1f);
  std::vector<uint8_t> fc1_zero_points(num_experts * inter_size, 0x88);
  std::vector<uint8_t> fc2_zero_points(num_experts * hidden_size, 0x88);
  std::vector<uint8_t> fc3_zero_points(num_experts * inter_size, 0x88);
  std::fill(fc1_zero_points.begin() + inter_size, fc1_zero_points.end(), 0x99);
  std::fill(fc2_zero_points.begin() + hidden_size, fc2_zero_points.end(), 0x99);
  std::fill(fc3_zero_points.begin() + inter_size, fc3_zero_points.end(), 0x99);
  const std::vector<float> fc2_bias(num_experts * hidden_size, 0.0f);
  const std::vector<float> expected(num_rows * hidden_size, 0.0f);

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "silu");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 4);
  tester.AddAttribute<int64_t>("block_size", 32);
  tester.AddAttribute<int64_t>("weights_prepacked", 0);
  tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
  tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, inter_size, hidden_size / pack_size}, fc1_weights);
  tester.AddInput<MLFloat16>("fc1_scales", {num_experts, inter_size, 2}, ToFloat16(fc1_scales));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, hidden_size, inter_size / pack_size}, fc2_weights);
  tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size, 2}, ToFloat16(fc2_scales));
  tester.AddInput<MLFloat16>("fc2_experts_bias", {num_experts, hidden_size}, ToFloat16(fc2_bias));
  tester.AddInput<uint8_t>("fc3_experts_weights", {num_experts, inter_size, hidden_size / pack_size}, fc3_weights);
  tester.AddInput<MLFloat16>("fc3_scales", {num_experts, inter_size, 2}, ToFloat16(fc3_scales));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc1_zero_points", {num_experts, inter_size, 1}, fc1_zero_points);
  tester.AddInput<uint8_t>("fc2_zero_points", {num_experts, hidden_size, 1}, fc2_zero_points);
  tester.AddInput<uint8_t>("fc3_zero_points", {num_experts, inter_size, 1}, fc3_zero_points);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, ToFloat16(expected));
  tester.SetOutputTolerance(0.05f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(MoETest, QMoETest_WebGPU_BlockFp8_RejectLegacyTranspose) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<MLFloat16>("input", {1, 2}, ToFloat16({1.0f, 2.0f}));
  tester.AddInput<MLFloat16>("router_probs", {1, 1}, ToFloat16({1.0f}));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {1, 2, 3},
                                std::vector<Float8E4M3FN>(6, Float8E4M3FN(0.5f)));
  tester.AddInput<float>("fc1_scales", {1, 1, 1}, {1.0f});
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {1, 3, 2},
                                std::vector<Float8E4M3FN>(6, Float8E4M3FN(0.5f)));
  tester.AddInput<float>("fc2_scales", {1, 1, 1}, {1.0f});
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {1, 2}, ToFloat16({0.0f, 0.0f}));

  RunWebGpuOnly(tester, std::move(webgpu_ep),
                OpTester::ExpectResult::kExpectFailure, "canonical [E,N,K]");
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_RejectOrphanFc3Scales) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<MLFloat16>("input", {1, 2}, ToFloat16({1.0f, 2.0f}));
  tester.AddInput<MLFloat16>("router_probs", {1, 1}, ToFloat16({1.0f}));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {1, 2, 2},
                                std::vector<Float8E4M3FN>(4, Float8E4M3FN(0.5f)));
  tester.AddInput<float>("fc1_scales", {1, 1, 1}, {1.0f});
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {1, 2, 2},
                                std::vector<Float8E4M3FN>(4, Float8E4M3FN(0.5f)));
  tester.AddInput<float>("fc2_scales", {1, 1, 1}, {1.0f});
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddInput<float>("fc3_scales", {1, 1, 1}, {1.0f});
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {1, 2}, ToFloat16({0.0f, 0.0f}));

  RunWebGpuOnly(tester, std::move(webgpu_ep),
                OpTester::ExpectResult::kExpectFailure, "fc3_scales requires fc3_experts_weights");
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_RejectActivationScales) {
  constexpr const char* scale_names[] = {
      "fc1_act_scale", "fc2_act_scale", "fc1_act_block_scale", "fc2_act_block_scale"};
  for (int scale_index = 0; scale_index < 4; ++scale_index) {
    SCOPED_TRACE(scale_names[scale_index]);
    GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

    OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("k", 1);
    tester.AddAttribute<std::string>("activation_type", "identity");
    tester.AddAttribute<int64_t>("expert_weight_bits", 8);
    tester.AddAttribute<int64_t>("block_size", 128);
    tester.AddAttribute<std::string>("quant_type", "fp8");
    tester.AddInput<MLFloat16>("input", {1, 2}, ToFloat16({1.0f, 2.0f}));
    tester.AddInput<MLFloat16>("router_probs", {1, 1}, ToFloat16({1.0f}));
    tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {1, 2, 2},
                                  std::vector<Float8E4M3FN>(4, Float8E4M3FN(0.5f)));
    tester.AddInput<float>("fc1_scales", {1, 1, 1}, {1.0f});
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {1, 2, 2},
                                  std::vector<Float8E4M3FN>(4, Float8E4M3FN(0.5f)));
    tester.AddInput<float>("fc2_scales", {1, 1, 1}, {1.0f});
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<Float8E4M3FN>();
    tester.AddOptionalInputEdge<float>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<uint8_t>();
    tester.AddOptionalInputEdge<uint8_t>();
    tester.AddOptionalInputEdge<uint8_t>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<float>();
    tester.AddOptionalInputEdge<float>();
    for (int prior = 0; prior < scale_index; ++prior) {
      tester.AddOptionalInputEdge<float>();
    }
    tester.AddInput<float>(scale_names[scale_index],
                           scale_index < 2 ? std::vector<int64_t>{1} : std::vector<int64_t>{1, 1, 1},
                           {1.0f});
    tester.AddOutput<MLFloat16>("output", {1, 2}, ToFloat16({0.0f, 0.0f}));

    RunWebGpuOnly(tester, std::move(webgpu_ep),
                  OpTester::ExpectResult::kExpectFailure, "does not support activation scales");
  }
}

void RunWebGpuBlockFp8NumericalReferenceAndBiases(bool fp32) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_experts = 2;
  constexpr int hidden_size = 2;
  constexpr int inter_size = 3;
  const std::vector<Float8E4M3FN> fc1_weights{
      Float8E4M3FN(0.0f), Float8E4M3FN(0.0f),
      Float8E4M3FN(0.0f), Float8E4M3FN(0.0f),
      Float8E4M3FN(0.0f), Float8E4M3FN(0.0f),
      Float8E4M3FN(uint8_t{0x01}, Float8E4M3FN::FromBits()), Float8E4M3FN(-0.5f),
      Float8E4M3FN(0.75f), Float8E4M3FN(-1.0f),
      Float8E4M3FN(0.25f), Float8E4M3FN(0.5625f)};
  const std::vector<Float8E4M3FN> fc2_weights{
      Float8E4M3FN(0.0f), Float8E4M3FN(0.0f), Float8E4M3FN(0.0f),
      Float8E4M3FN(0.0f), Float8E4M3FN(0.0f), Float8E4M3FN(0.0f),
      Float8E4M3FN(-0.25f), Float8E4M3FN(0.5f), Float8E4M3FN(0.125f),
      Float8E4M3FN(0.75f), Float8E4M3FN(uint8_t{0x81}, Float8E4M3FN::FromBits()),
      Float8E4M3FN(-0.5f)};
  const std::vector<float> fc1_bias{0.0f, 0.0f, 0.0f, 0.125f, -0.25f, 0.5f};
  const std::vector<float> fc2_bias{0.0f, 0.0f, 0.375f, -0.125f};
  const std::vector<float> input{2.0f, 1.0f};

  std::vector<float> fc1_result;
  for (int col = 0; col < inter_size; ++col) {
    float sum = fc1_bias[inter_size + col];
    for (int k = 0; k < hidden_size; ++k) {
      sum += input[k] * fc1_weights[inter_size * hidden_size + col * hidden_size + k].ToFloat() * 0.5f;
    }
    fc1_result.push_back(fp32 ? sum : MLFloat16(sum).ToFloat());
  }
  std::vector<float> expected;
  for (int col = 0; col < hidden_size; ++col) {
    float sum = fc2_bias[hidden_size + col];
    for (int k = 0; k < inter_size; ++k) {
      sum += fc1_result[k] * fc2_weights[hidden_size * inter_size + col * inter_size + k].ToFloat() * 0.25f;
    }
    expected.push_back(fp32 ? sum : MLFloat16(sum).ToFloat());
  }

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  if (fp32) {
    tester.AddInput<float>("input", {1, hidden_size}, input);
    tester.AddInput<float>("router_probs", {1, num_experts}, {0.0f, 10.0f});
  } else {
    tester.AddInput<MLFloat16>("input", {1, hidden_size}, ToFloat16(input));
    tester.AddInput<MLFloat16>("router_probs", {1, num_experts}, ToFloat16({0.0f, 10.0f}));
  }
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {num_experts, inter_size, hidden_size}, fc1_weights);
  tester.AddInput<float>("fc1_scales", {num_experts, 1, 1}, {1.0f, 0.5f});
  if (fp32) {
    tester.AddInput<float>("fc1_experts_bias", {num_experts, inter_size}, fc1_bias);
  } else {
    tester.AddInput<MLFloat16>("fc1_experts_bias", {num_experts, inter_size}, ToFloat16(fc1_bias));
  }
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {num_experts, hidden_size, inter_size}, fc2_weights);
  tester.AddInput<float>("fc2_scales", {num_experts, 1, 1}, {1.0f, 0.25f});
  if (fp32) {
    tester.AddInput<float>("fc2_experts_bias", {num_experts, hidden_size}, fc2_bias);
  } else {
    tester.AddInput<MLFloat16>("fc2_experts_bias", {num_experts, hidden_size}, ToFloat16(fc2_bias));
  }
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddOptionalInputEdge<float>();
  if (fp32) {
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>("output", {1, hidden_size}, expected);
    tester.SetOutputTolerance(0.001f);
  } else {
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOutput<MLFloat16>("output", {1, hidden_size}, ToFloat16(expected));
    tester.SetOutputTolerance(0.01f);
  }

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_NumericalReferenceAndBiases) {
  RunWebGpuBlockFp8NumericalReferenceAndBiases(false);
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_Fp32NumericalReferenceAndBiases) {
  RunWebGpuBlockFp8NumericalReferenceAndBiases(true);
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_RejectNormalizedDispatch) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int inter_size = 65535 * 64 + 1;
  constexpr int scale_blocks = (inter_size - 1) / 128 + 1;
  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<float>("input", {1, 1}, {1.0f});
  tester.AddInput<float>("router_probs", {1, 1}, {1.0f});
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {1, inter_size, 1},
                                std::vector<Float8E4M3FN>(inter_size, Float8E4M3FN(0.0f)));
  tester.AddInput<float>("fc1_scales", {1, scale_blocks, 1}, std::vector<float>(scale_blocks, 1.0f));
  tester.AddOptionalInputEdge<float>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {1, 1, inter_size},
                                std::vector<Float8E4M3FN>(inter_size, Float8E4M3FN(0.0f)));
  tester.AddInput<float>("fc2_scales", {1, 1, scale_blocks}, std::vector<float>(scale_blocks, 1.0f));
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOutput<float>("output", {1, 1}, {0.0f});

  RunWebGpuOnly(tester, std::move(webgpu_ep),
                OpTester::ExpectResult::kExpectFailure,
                "FC1 block-FP8 dispatch exceeds WebGPU's per-dimension workgroup limit");
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_SeparateFc3SingleToken) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_experts = 2;
  constexpr int size = 3;
  const std::vector<float> input{0.5f, 1.0f, 1.5f};
  std::vector<Float8E4M3FN> weights(num_experts * size * size, Float8E4M3FN(0.0f));
  for (int expert = 0; expert < num_experts; ++expert) {
    for (int index = 0; index < size; ++index) {
      weights[expert * size * size + index * size + index] = Float8E4M3FN(1.0f);
    }
  }

  std::vector<float> expected;
  for (float value : input) {
    float result = 0.0f;
    for (int expert = 0; expert < num_experts; ++expert) {
      const float fc1 = MLFloat16(value * (expert == 0 ? 1.0f : 2.0f)).ToFloat();
      const float fc3 = MLFloat16(value * (expert == 0 ? 3.0f : 4.0f)).ToFloat();
      const float activated = MLFloat16((fc1 / (1.0f + std::exp(-fc1))) * fc3).ToFloat();
      result += MLFloat16(activated * (expert == 0 ? 5.0f : 6.0f)).ToFloat() * 0.5f;
    }
    expected.push_back(result);
  }

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 2);
  tester.AddAttribute<std::string>("activation_type", "silu");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<MLFloat16>("input", {1, size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {1, num_experts}, ToFloat16({10.0f, 10.0f}));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {num_experts, size, size}, weights);
  tester.AddInput<float>("fc1_scales", {num_experts, 1, 1}, {1.0f, 2.0f});
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {num_experts, size, size}, weights);
  tester.AddInput<float>("fc2_scales", {num_experts, 1, 1}, {5.0f, 6.0f});
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc3_experts_weights", {num_experts, size, size}, weights);
  tester.AddInput<float>("fc3_scales", {num_experts, 1, 1}, {3.0f, 4.0f});
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {1, size}, ToFloat16(expected));
  tester.SetOutputTolerance(0.02f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_128x128) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_rows = 2;
  constexpr int num_experts = 2;
  constexpr int hidden_size = 4;
  constexpr int inter_size = 4;

  const std::vector<float> input = {
      1.0f, 2.0f, 3.0f, 4.0f,
      -1.0f, 0.5f, 2.0f, -3.0f};
  const std::vector<float> router_probs = {10.0f, 0.0f, 0.0f, 10.0f};
  std::vector<Float8E4M3FN> fc1_weights(num_experts * inter_size * hidden_size, Float8E4M3FN(0.0f));
  std::vector<Float8E4M3FN> fc2_weights(num_experts * hidden_size * inter_size, Float8E4M3FN(0.0f));
  for (int index = 0; index < hidden_size; ++index) {
    fc1_weights[inter_size * hidden_size + index * hidden_size + index] = Float8E4M3FN(1.0f);
    fc2_weights[hidden_size * inter_size + index * inter_size + index] = Float8E4M3FN(1.0f);
  }
  const std::vector<float> fc1_scales = {1.0f, 2.0f};
  const std::vector<float> fc2_scales = {1.0f, 3.0f};
  std::vector<float> expected(hidden_size, 0.0f);
  for (int index = hidden_size; index < static_cast<int>(input.size()); ++index) {
    expected.push_back(input[index] * 6.0f);
  }

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {num_experts, inter_size, hidden_size}, fc1_weights);
  tester.AddInput<float>("fc1_scales", {num_experts, 1, 1}, fc1_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {num_experts, hidden_size, inter_size}, fc2_weights);
  tester.AddInput<float>("fc2_scales", {num_experts, 1, 1}, fc2_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, ToFloat16(expected));
  tester.SetOutputTolerance(0.01f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_TiledPrefill) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int rows = 128;
  constexpr int experts = 2;
  constexpr int size = 257;
  std::vector<float> input(rows * size), expected(rows * size);
  for (int i = 0; i < rows * size; ++i) {
    input[i] = static_cast<float>(i % 13 - 6) * 0.25f;
    expected[i] = input[i] * (i % size < 128 ? 1.0f : i % size < 256 ? 0.5f
                                                                     : 0.25f) *
                      (i % size % 3 == 0 ? -1.0f : 1.0f) +
                  1.0f;
  }
  std::vector<float> router_probs(rows * experts, 0.0f);
  for (int row = 0; row < rows; ++row) {
    router_probs[row * experts + 1] = 10.0f;
  }
  std::vector<Float8E4M3FN> fc1_weights(experts * size * size, Float8E4M3FN(0.0f));
  std::vector<Float8E4M3FN> fc2_weights = fc1_weights;
  for (int i = 0; i < size; ++i) {
    fc1_weights[size * size + i * size + i] = Float8E4M3FN(1.0f);
    fc2_weights[size * size + i * size + i] = Float8E4M3FN(i % 3 == 0 ? -1.0f : 1.0f);
  }
  std::vector<float> fc2_bias(experts * size, 0.0f);
  std::fill(fc2_bias.begin() + size, fc2_bias.end(), 1.0f);

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<MLFloat16>("input", {rows, size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {rows, experts}, ToFloat16(router_probs));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {experts, size, size}, fc1_weights);
  std::vector<float> fc1_scales(experts * 3 * 3, 1.0f);
  std::vector<float> fc2_scales = fc1_scales;
  fc1_scales[9] = 0.5f;
  fc1_scales[13] = 2.0f;
  fc1_scales[17] = 0.5f;
  fc2_scales[9] = 2.0f;
  fc2_scales[13] = 0.25f;
  fc2_scales[17] = 0.5f;
  tester.AddInput<float>("fc1_scales", {experts, 3, 3}, fc1_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {experts, size, size}, fc2_weights);
  tester.AddInput<float>("fc2_scales", {experts, 3, 3}, fc2_scales);
  tester.AddInput<MLFloat16>("fc2_experts_bias", {experts, size}, ToFloat16(fc2_bias));
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {rows, size}, ToFloat16(expected));
  tester.SetOutputTolerance(0.01f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_MixedRoutingSeparateFc3) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);
  const auto& gpu_context = webgpu::WebGpuContextFactory::GetContext(0);
  const auto& adapter = gpu_context.AdapterInfo();
  const auto& configs = gpu_context.SubgroupMatrixConfigs();
  const bool matrix_supported =
      gpu_context.DeviceHasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix) &&
      webgpu::detail::SelectSubgroupMatrixConfigFromAdapterConfigs(
          {configs.configs, configs.configCount}, adapter.subgroupMinSize, adapter.subgroupMaxSize,
          gpu_context.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl),
          {{wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F32,
            16, 16, 16, 32, false}})
          .has_value();
  const auto dispatches_before = contrib::webgpu::BlockFp8MatrixDispatchCount();

  constexpr int rows = 132;
  constexpr int experts = 2;
  constexpr int size = 256;
  std::vector<float> input(rows * size, 0.0f);
  std::vector<float> router_probs(rows * experts, 0.0f);
  std::vector<float> expected(rows * size);
  std::vector<Float8E4M3FN> fc1_weights(experts * size * size, Float8E4M3FN(0.0f));
  std::vector<Float8E4M3FN> fc2_weights = fc1_weights;
  std::vector<Float8E4M3FN> fc3_weights = fc1_weights;
  const std::vector<float> fc1_scales{1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  const std::vector<float> fc2_scales{0.5f, 0.5f, 0.5f, 0.5f, 0.25f, 0.25f, 0.25f, 0.25f};
  const std::vector<float> fc3_scales{3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f};
  std::vector<float> fc1_bias(experts * size), fc2_bias(experts * size), fc3_bias(experts * size);
  for (int expert = 0; expert < experts; ++expert) {
    for (int col = 0; col < size; ++col) {
      const int index = expert * size * size + col * size + col;
      fc1_weights[index] = Float8E4M3FN(expert == 0 ? 1.0f : -1.0f);
      fc2_weights[index] = Float8E4M3FN(expert == 0 ? 1.0f : -1.0f);
      fc3_weights[index] = Float8E4M3FN(expert == 0 ? -1.0f : 1.0f);
      fc1_bias[expert * size + col] = expert == 0 ? 0.25f : -0.5f;
      fc2_bias[expert * size + col] = expert == 0 ? 0.75f : -1.0f;
      fc3_bias[expert * size + col] = expert == 0 ? -0.125f : 0.375f;
    }
  }
  for (int row = 0; row < rows; ++row) {
    const int expert = row < 4 ? 0 : 1;
    router_probs[row * experts + expert] = 10.0f;
    for (int col = 0; col < size; ++col) {
      const float value = static_cast<float>((row + col) % 7 - 3) * 0.125f;
      input[row * size + col] = value;
      const float fc1 = MLFloat16(value * (expert == 0 ? 1.0f : -2.0f) +
                                  fc1_bias[expert * size + col])
                            .ToFloat();
      const float fc3 = MLFloat16(value * (expert == 0 ? -3.0f : 4.0f) +
                                  fc3_bias[expert * size + col])
                            .ToFloat();
      const float activated = MLFloat16((fc1 / (1.0f + std::exp(-fc1))) * fc3).ToFloat();
      expected[row * size + col] = MLFloat16(activated * (expert == 0 ? 0.5f : -0.25f) +
                                             fc2_bias[expert * size + col])
                                       .ToFloat();
    }
  }

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "silu");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<MLFloat16>("input", {rows, size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {rows, experts}, ToFloat16(router_probs));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {experts, size, size}, fc1_weights);
  tester.AddInput<float>("fc1_scales", {experts, 2, 2}, fc1_scales);
  tester.AddInput<MLFloat16>("fc1_experts_bias", {experts, size}, ToFloat16(fc1_bias));
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {experts, size, size}, fc2_weights);
  tester.AddInput<float>("fc2_scales", {experts, 2, 2}, fc2_scales);
  tester.AddInput<MLFloat16>("fc2_experts_bias", {experts, size}, ToFloat16(fc2_bias));
  tester.AddInput<Float8E4M3FN>("fc3_experts_weights", {experts, size, size}, fc3_weights);
  tester.AddInput<float>("fc3_scales", {experts, 2, 2}, fc3_scales);
  tester.AddInput<MLFloat16>("fc3_experts_bias", {experts, size}, ToFloat16(fc3_bias));
  tester.AddOutput<MLFloat16>("output", {rows, size}, ToFloat16(expected));
  tester.SetOutputTolerance(0.02f);
  RunWebGpuOnly(tester, std::move(webgpu_ep));
  if (!matrix_supported) {
    GTEST_SKIP() << "Adapter does not support the FP8 expert subgroup-matrix configuration";
  }
  EXPECT_EQ(contrib::webgpu::BlockFp8MatrixDispatchCount() - dispatches_before, 3u);
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_LargeScaleCancellation) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int rows = 128;
  constexpr int size = 256;
  std::vector<float> input(rows * size, 0.0f);
  for (int row = 0; row < rows; ++row) {
    input[row * size] = 1.0f;
    input[row * size + 1] = 1.0f;
  }
  std::vector<Float8E4M3FN> fc1_weights(size * size, Float8E4M3FN(0.0f));
  fc1_weights[0] = Float8E4M3FN(448.0f);
  fc1_weights[1] = Float8E4M3FN(-448.0f);
  std::vector<Float8E4M3FN> fc2_weights(size * size, Float8E4M3FN(0.0f));
  fc2_weights[0] = Float8E4M3FN(1.0f);
  std::vector<float> fc1_scales(4, 200.0f);
  std::vector<float> fc2_scales(4, 1.0f);

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<MLFloat16>("input", {rows, size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {rows, 1}, ToFloat16(std::vector<float>(rows, 1.0f)));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {1, size, size}, fc1_weights);
  tester.AddInput<float>("fc1_scales", {1, 2, 2}, fc1_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {1, size, size}, fc2_weights);
  tester.AddInput<float>("fc2_scales", {1, 2, 2}, fc2_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {rows, size}, ToFloat16(std::vector<float>(rows * size, 0.0f)));
  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_UnalignedExpertBoundary) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_experts = 3;
  constexpr int hidden_size = 3;
  constexpr int inter_size = 3;
  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  const std::vector<float> router_probs = {
      10.0f, 0.0f, 0.0f,
      0.0f, 10.0f, 0.0f,
      0.0f, 0.0f, 10.0f};
  std::vector<Float8E4M3FN> weights(num_experts * inter_size * hidden_size, Float8E4M3FN(0.0f));
  for (int expert = 0; expert < num_experts; ++expert) {
    for (int index = 0; index < hidden_size; ++index) {
      weights[expert * inter_size * hidden_size + index * hidden_size + index] = Float8E4M3FN(1.0f);
    }
  }
  const std::vector<float> fc1_scales = {1.0f, 2.0f, 4.0f};
  const std::vector<float> fc2_scales = {1.0f, 3.0f, 5.0f};
  const std::vector<float> expected = {1.0f, 2.0f, 3.0f, 24.0f, 30.0f, 36.0f, 140.0f, 160.0f, 180.0f};

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<MLFloat16>("input", {3, hidden_size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {3, num_experts}, ToFloat16(router_probs));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {num_experts, inter_size, hidden_size}, weights);
  tester.AddInput<float>("fc1_scales", {num_experts, 1, 1}, fc1_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {num_experts, hidden_size, inter_size}, weights);
  tester.AddInput<float>("fc2_scales", {num_experts, 1, 1}, fc2_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {3, hidden_size}, ToFloat16(expected));
  tester.SetOutputTolerance(0.01f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_PartialScaleTiles) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_experts = 2;
  constexpr int hidden_size = 129;
  constexpr int inter_size = 129;
  const std::vector<float> router_probs = {10.0f, 0.0f, 0.0f, 10.0f};
  std::vector<float> input(2 * hidden_size, 0.0f);
  input[hidden_size] = 1.0f;
  input[2 * hidden_size - 1] = 2.0f;
  std::vector<Float8E4M3FN> weights(num_experts * inter_size * hidden_size, Float8E4M3FN(0.0f));
  for (int row : {0, 128}) {
    for (int col : {0, 128}) {
      weights[inter_size * hidden_size + row * hidden_size + col] = Float8E4M3FN(1.0f);
    }
  }
  const std::vector<float> fc1_scales = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> fc2_scales = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 7.0f, 11.0f};
  std::vector<float> expected(2 * hidden_size, 0.0f);
  expected[hidden_size] = 60.0f;
  expected[2 * hidden_size - 1] = 156.0f;

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<MLFloat16>("input", {2, hidden_size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {2, num_experts}, ToFloat16(router_probs));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {num_experts, inter_size, hidden_size}, weights);
  tester.AddInput<float>("fc1_scales", {num_experts, 2, 2}, fc1_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {num_experts, hidden_size, inter_size}, weights);
  tester.AddInput<float>("fc2_scales", {num_experts, 2, 2}, fc2_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {2, hidden_size}, ToFloat16(expected));
  tester.SetOutputTolerance(0.01f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, QMoETest_WebGPU_BlockFp8_SingleTokenMultipleExperts) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_experts = 2;
  constexpr int hidden_size = 4;
  constexpr int inter_size = 4;
  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> router_probs = {10.0f, 10.0f};
  std::vector<Float8E4M3FN> fc1_weights(num_experts * inter_size * hidden_size, Float8E4M3FN(0.0f));
  std::vector<Float8E4M3FN> fc2_weights(num_experts * hidden_size * inter_size, Float8E4M3FN(0.0f));
  for (int expert = 0; expert < num_experts; ++expert) {
    for (int index = 0; index < hidden_size; ++index) {
      fc1_weights[expert * inter_size * hidden_size + index * hidden_size + index] = Float8E4M3FN(1.0f);
      fc2_weights[expert * hidden_size * inter_size + index * inter_size + index] = Float8E4M3FN(1.0f);
    }
  }
  const std::vector<float> fc1_scales = {1.0f, 2.0f};
  const std::vector<float> fc2_scales = {3.0f, 4.0f};
  const std::vector<float> expected = {5.5f, 11.0f, 16.5f, 22.0f};

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 2);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<int64_t>("block_size", 128);
  tester.AddAttribute<std::string>("quant_type", "fp8");
  tester.AddInput<MLFloat16>("input", {1, hidden_size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {1, num_experts}, ToFloat16(router_probs));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {num_experts, inter_size, hidden_size}, fc1_weights);
  tester.AddInput<float>("fc1_scales", {num_experts, 1, 1}, fc1_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {num_experts, hidden_size, inter_size}, fc2_weights);
  tester.AddInput<float>("fc2_scales", {num_experts, 1, 1}, fc2_scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {1, hidden_size}, ToFloat16(expected));
  tester.SetOutputTolerance(0.01f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, QMoETest_WebGPU_RejectFp8WeightsWithIntegerQuantization) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  const std::vector<Float8E4M3FN> weights(16, Float8E4M3FN(1.0f));
  const std::vector<float> scales(4, 1.0f);
  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 8);
  tester.AddAttribute<std::string>("quant_type", "int");
  tester.AddInput<MLFloat16>("input", {1, 4}, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f}));
  tester.AddInput<MLFloat16>("router_probs", {1, 1}, ToFloat16({1.0f}));
  tester.AddInput<Float8E4M3FN>("fc1_experts_weights", {1, 4, 4}, weights);
  tester.AddInput<float>("fc1_scales", {1, 4}, scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<Float8E4M3FN>("fc2_experts_weights", {1, 4, 4}, weights);
  tester.AddInput<float>("fc2_scales", {1, 4}, scales);
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<Float8E4M3FN>();
  tester.AddOptionalInputEdge<float>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {1, 4}, ToFloat16({0.0f, 0.0f, 0.0f, 0.0f}));

  RunWebGpuOnly(tester, std::move(webgpu_ep),
                OpTester::ExpectResult::kExpectFailure, "Integer QMoE weights must use uint8 elements.");
}
#endif

static void RunQMoEWebGpuNormalizedRouterWeightsTest(int64_t num_rows) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int num_experts = 2;
  constexpr int hidden_size = 64;
  constexpr int inter_size = 64;
  constexpr int pack_size = 2;

  const std::vector<uint8_t> fc1_weights(num_experts * inter_size * hidden_size / pack_size, 0x99);
  const std::vector<uint8_t> fc2_weights(num_experts * hidden_size * inter_size / pack_size, 0x99);
  const std::vector<float> fc1_scales(num_experts * inter_size, 0.1f);
  const std::vector<float> fc2_scales(num_experts * hidden_size, 0.1f);

  const std::vector<float> input(static_cast<size_t>(num_rows * hidden_size), 0.25f);
  std::vector<float> router_probs;
  std::vector<float> router_weights;
  for (int64_t row = 0; row < num_rows; ++row) {
    router_probs.insert(router_probs.end(), {10.0f, 0.0f});
    router_weights.insert(router_weights.end(), {3.0f, 1.0f});
  }
  const float fc1_output = hidden_size * 0.25f * 0.1f;
  const float fc2_output = inter_size * fc1_output * 0.1f;
  const std::vector<float> expected(static_cast<size_t>(num_rows * hidden_size), fc2_output);

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 2);
  tester.AddAttribute<std::string>("activation_type", "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", 4);
  tester.AddInput<MLFloat16>("input", {num_rows, hidden_size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
  tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, inter_size, hidden_size / pack_size}, fc1_weights);
  tester.AddInput<MLFloat16>("fc1_scales", {num_experts, inter_size}, ToFloat16(fc1_scales));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, hidden_size, inter_size / pack_size}, fc2_weights);
  tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size}, ToFloat16(fc2_scales));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddInput<MLFloat16>("router_weights", {num_rows, num_experts}, ToFloat16(router_weights));
  tester.AddOutput<MLFloat16>("output", {num_rows, hidden_size}, ToFloat16(expected));
  tester.SetOutputTolerance(0.02f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, QMoETest_WebGPU_RouterWeights_SingleToken) {
  RunQMoEWebGpuNormalizedRouterWeightsTest(1);
}

TEST(MoETest, QMoETest_WebGPU_RouterWeights_MultiToken) {
  RunQMoEWebGpuNormalizedRouterWeightsTest(2);
}

TEST(MoETest, QMoETest_WebGPU_ActivationsAndSwiGLUFusion) {
  constexpr int num_experts = 2;
  constexpr int hidden_size = 64;
  constexpr int inter_size = 64;
  constexpr int pack_size = 2;
  const std::vector<float> input(hidden_size, 0.25f);
  const std::vector<float> router_probs = {10.0f, 0.0f};
  std::vector<float> fc2_bias(hidden_size, 1.0f);
  fc2_bias.insert(fc2_bias.end(), hidden_size, 2.0f);
  const std::vector<float> expected(hidden_size, 1.0f);

  const auto run_case = [&](const std::string& activation_type, int64_t swiglu_fusion) {
    GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

    const int fc1_size = activation_type == "swiglu" ? 2 * inter_size : inter_size;
    const std::vector<uint8_t> fc1_weights(num_experts * fc1_size * hidden_size / pack_size, 0x88);
    const std::vector<uint8_t> fc2_weights(num_experts * hidden_size * inter_size / pack_size, 0x88);
    const std::vector<float> fc1_scales(num_experts * fc1_size, 0.1f);
    const std::vector<float> fc2_scales(num_experts * hidden_size, 0.1f);

    OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("k", 1);
    tester.AddAttribute<std::string>("activation_type", activation_type);
    tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
    tester.AddAttribute<int64_t>("swiglu_fusion", swiglu_fusion);
    tester.AddAttribute<int64_t>("expert_weight_bits", 4);
    tester.AddAttribute<int64_t>("weights_prepacked", 0);
    tester.AddInput<MLFloat16>("input", {1, hidden_size}, ToFloat16(input));
    tester.AddInput<MLFloat16>("router_probs", {1, num_experts}, ToFloat16(router_probs));
    tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, fc1_size, hidden_size / pack_size}, fc1_weights);
    tester.AddInput<MLFloat16>("fc1_scales", {num_experts, fc1_size}, ToFloat16(fc1_scales));
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, hidden_size, inter_size / pack_size}, fc2_weights);
    tester.AddInput<MLFloat16>("fc2_scales", {num_experts, hidden_size}, ToFloat16(fc2_scales));
    tester.AddInput<MLFloat16>("fc2_experts_bias", {num_experts, hidden_size}, ToFloat16(fc2_bias));
    tester.AddOptionalInputEdge<uint8_t>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOutput<MLFloat16>("output", {1, hidden_size}, ToFloat16(expected));
    tester.SetOutputTolerance(0.01f);
    RunWebGpuOnly(tester, std::move(webgpu_ep));
  };

  run_case("relu", 0);
  run_case("gelu", 0);
  run_case("silu", 0);
  run_case("identity", 0);
  run_case("swiglu", 1);
  run_case("swiglu", 2);
}

static void RunQMoEWebGpuIntegerWidths(int num_rows, int num_experts, int k,
                                       int base_bits, int fc1_bits, int fc2_bits, int fc3_bits = 0) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);

  constexpr int size = 64;
  const bool has_fc3 = fc3_bits != 0;
  const std::vector<uint8_t> fc1_weights(num_experts * size * size * fc1_bits / 8,
                                         fc1_bits == 2 ? 0xFF : 0x99);
  const std::vector<uint8_t> fc2_weights(num_experts * size * size * fc2_bits / 8,
                                         fc2_bits == 2 ? 0xFF : 0x99);
  const std::vector<uint8_t> fc3_weights(has_fc3 ? num_experts * size * size * fc3_bits / 8 : 0,
                                         fc3_bits == 2 ? 0xFF : 0x99);
  std::vector<float> input(num_rows * size), router_probs(num_rows * num_experts, 0.0f);
  std::vector<float> fc1_scales(num_experts * size), fc2_scales(num_experts * size);
  std::vector<float> fc3_scales(has_fc3 ? num_experts * size : 0);
  std::vector<float> expected(num_rows * size);
  for (int expert = 0; expert < num_experts; ++expert) {
    std::fill_n(fc1_scales.begin() + expert * size, size, expert == 0 ? 0.5f : 1.0f);
    std::fill_n(fc2_scales.begin() + expert * size, size, expert == 0 ? 0.25f : 0.5f);
    if (has_fc3) {
      std::fill_n(fc3_scales.begin() + expert * size, size, expert == 0 ? 0.125f : 0.25f);
    }
  }
  for (int row = 0; row < num_rows; ++row) {
    const float value = 0.25f * (row + 1);
    std::fill_n(input.begin() + row * size, size, value);
    float result = 0.0f;
    for (int expert = 0; expert < num_experts; ++expert) {
      if (k == 1 && expert != row % num_experts) {
        continue;
      }
      router_probs[row * num_experts + expert] = 10.0f;
      const float fc1 = MLFloat16(size * value * fc1_scales[expert * size]).ToFloat();
      float activated = fc1;
      if (has_fc3) {
        const float fc3 = MLFloat16(size * value * fc3_scales[expert * size]).ToFloat();
        activated = MLFloat16(fc1 / (1.0f + std::exp(-fc1)) * fc3).ToFloat();
      }
      result += MLFloat16(size * activated * fc2_scales[expert * size]).ToFloat() / k;
    }
    std::fill_n(expected.begin() + row * size, size, result);
  }

  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", k);
  tester.AddAttribute<std::string>("activation_type", has_fc3 ? "swiglu" : "identity");
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", base_bits);
  if (fc1_bits != base_bits) tester.AddAttribute<int64_t>("fc1_expert_weight_bits", fc1_bits);
  if (fc2_bits != base_bits) tester.AddAttribute<int64_t>("fc2_expert_weight_bits", fc2_bits);
  if (has_fc3 && fc3_bits != base_bits) tester.AddAttribute<int64_t>("fc3_expert_weight_bits", fc3_bits);
  tester.AddInput<MLFloat16>("input", {num_rows, size}, ToFloat16(input));
  tester.AddInput<MLFloat16>("router_probs", {num_rows, num_experts}, ToFloat16(router_probs));
  tester.AddInput<uint8_t>("fc1_experts_weights", {num_experts, size, size * fc1_bits / 8}, fc1_weights);
  tester.AddInput<MLFloat16>("fc1_scales", {num_experts, size}, ToFloat16(fc1_scales));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", {num_experts, size, size * fc2_bits / 8}, fc2_weights);
  tester.AddInput<MLFloat16>("fc2_scales", {num_experts, size}, ToFloat16(fc2_scales));
  tester.AddOptionalInputEdge<MLFloat16>();
  if (has_fc3) {
    tester.AddInput<uint8_t>("fc3_experts_weights", {num_experts, size, size * fc3_bits / 8}, fc3_weights);
    tester.AddInput<MLFloat16>("fc3_scales", {num_experts, size}, ToFloat16(fc3_scales));
  } else {
    tester.AddOptionalInputEdge<uint8_t>();
    tester.AddOptionalInputEdge<MLFloat16>();
  }
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {num_rows, size}, ToFloat16(expected));
  tester.SetOutputTolerance(0.02f);

  RunWebGpuOnly(tester, std::move(webgpu_ep));
}

TEST(MoETest, QMoETest_WebGPU_MixedWidth_SingleToken) {
  RunQMoEWebGpuIntegerWidths(1, 1, 1, 4, 2, 4);
}

TEST(MoETest, QMoETest_WebGPU_Int2InheritedWidths) {
  RunQMoEWebGpuIntegerWidths(1, 1, 1, 2, 2, 2);
}

TEST(MoETest, QMoETest_WebGPU_MixedWidth_TwoExperts) {
  RunQMoEWebGpuIntegerWidths(1, 2, 2, 4, 2, 4);
}

TEST(MoETest, QMoETest_WebGPU_MixedWidth_MultiToken) {
  RunQMoEWebGpuIntegerWidths(2, 2, 1, 4, 2, 4);
}

TEST(MoETest, QMoETest_WebGPU_MixedWidth_SeparateFc3) {
  RunQMoEWebGpuIntegerWidths(1, 2, 2, 4, 4, 2, 2);
}

TEST(MoETest, QMoETest_WebGPU_Int2RejectUnalignedRows) {
  const auto run_case = [](int hidden_size, int inter_size) {
    GET_WEBGPU_EP_OR_SKIP(webgpu_ep);
    OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("k", 1);
    tester.AddAttribute<std::string>("activation_type", "identity");
    tester.AddAttribute<int64_t>("expert_weight_bits", 2);
    tester.AddInput<MLFloat16>("input", {1, hidden_size},
                               ToFloat16(std::vector<float>(hidden_size, 0.25f)));
    tester.AddInput<MLFloat16>("router_probs", {1, 1}, ToFloat16({1.0f}));
    tester.AddInput<uint8_t>("fc1_experts_weights", {1, inter_size, hidden_size / 4},
                             std::vector<uint8_t>(inter_size * hidden_size / 4, 0xFF));
    tester.AddInput<MLFloat16>("fc1_scales", {1, inter_size},
                               ToFloat16(std::vector<float>(inter_size, 0.5f)));
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddInput<uint8_t>("fc2_experts_weights", {1, hidden_size, inter_size / 4},
                             std::vector<uint8_t>(hidden_size * inter_size / 4, 0xFF));
    tester.AddInput<MLFloat16>("fc2_scales", {1, hidden_size},
                               ToFloat16(std::vector<float>(hidden_size, 0.25f)));
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<uint8_t>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOptionalInputEdge<MLFloat16>();
    tester.AddOutput<MLFloat16>("output", {1, hidden_size},
                                ToFloat16(std::vector<float>(hidden_size, 0.0f)));
    RunWebGpuOnly(tester, std::move(webgpu_ep), OpTester::ExpectResult::kExpectFailure, "32-bit aligned");
  };
  run_case(12, 64);
  run_case(64, 12);
}

TEST(MoETest, QMoETest_WebGPU_Int2RejectUnalignedFc3) {
  GET_WEBGPU_EP_OR_SKIP(webgpu_ep);
  constexpr int hidden_size = 8;
  constexpr int inter_size = 64;
  OpTester tester("QMoE", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("k", 1);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("expert_weight_bits", 4);
  tester.AddAttribute<int64_t>("fc3_expert_weight_bits", 2);
  tester.AddInput<MLFloat16>("input", {1, hidden_size},
                             ToFloat16(std::vector<float>(hidden_size, 0.25f)));
  tester.AddInput<MLFloat16>("router_probs", {1, 1}, ToFloat16({1.0f}));
  tester.AddInput<uint8_t>("fc1_experts_weights", {1, inter_size, hidden_size / 2},
                           std::vector<uint8_t>(inter_size * hidden_size / 2, 0x99));
  tester.AddInput<MLFloat16>("fc1_scales", {1, inter_size},
                             ToFloat16(std::vector<float>(inter_size, 0.5f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2_experts_weights", {1, hidden_size, inter_size / 2},
                           std::vector<uint8_t>(hidden_size * inter_size / 2, 0x99));
  tester.AddInput<MLFloat16>("fc2_scales", {1, hidden_size},
                             ToFloat16(std::vector<float>(hidden_size, 0.25f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc3_experts_weights", {1, inter_size, hidden_size / 4},
                           std::vector<uint8_t>(inter_size * hidden_size / 4, 0xFF));
  tester.AddInput<MLFloat16>("fc3_scales", {1, inter_size},
                             ToFloat16(std::vector<float>(inter_size, 0.125f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("output", {1, hidden_size},
                              ToFloat16(std::vector<float>(hidden_size, 0.0f)));
  RunWebGpuOnly(tester, std::move(webgpu_ep), OpTester::ExpectResult::kExpectFailure, "FC3 packed rows");
}

#undef GET_WEBGPU_EP_OR_SKIP
#endif
#endif  // ENABLE_TRAINING

}  // namespace test
}  // namespace onnxruntime