// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "gtest/gtest.h"
#include "core/common/cpuid_info.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/cpu/activation/activation_op_test.h"
#include <random>
#include <test/util/include/default_providers.h>

#if defined(USE_DNNL)
#include "test/common/dnnl_op_test_utils.h"
#endif

using namespace onnxruntime::test;

namespace onnxruntime {

namespace test {

TEST_F(ActivationOpTest, ThresholdedRelu_version_1_to_9) {
  float alpha = 0.1f;
  TestActivationOp<float>(
      "ThresholdedRelu", input_values, [alpha](float x) { return (x >= alpha) ? x : 0; }, {{"alpha", alpha}}, {},
      true, 1);
}

TEST_F(ActivationOpTest, ScaledTanh) {
  static constexpr float alpha = 2.0f;
  static constexpr float beta = 1.5f;

  TestActivationOp<float>("ScaledTanh", input_values, [](float x) { return alpha * tanh(beta * x); },
                          {{"alpha", alpha}, {"beta", beta}});
}

TEST_F(ActivationOpTest, ParametricSoftplus) {
  static constexpr float alpha = 2.0f;
  static constexpr float beta = 1.5f;

  TestActivationOp<float>(
      "ParametricSoftplus", input_values,
      [](float x) {
        float bx = beta * x;
        if (bx > 0)
          return alpha * (bx + logf(expf(-bx) + 1));
        else
          return alpha * logf(expf(bx) + 1);
      },
      {{"alpha", alpha}, {"beta", beta}}, {}, false);  // Disable TensorRT due to result mismatch
}

// [TODO] Temporarily ignore this test for OpenVINO
// Fails due to accuracy mismatch
#if !defined(USE_OPENVINO)
TEST_F(ActivationOpTest, Gelu) {
  TestActivationOp<float>(
      "Gelu", input_values, [](float x) { return x * 0.5f * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2))); }, {},
      {}, false, 1, kMSDomain);
}
#endif

#if defined(USE_DNNL)
std::vector<BFloat16> expected_output_bfloat16(const std::vector<float>& input_data) {
  std::vector<float> output;
  for (size_t i = 0; i < input_data.size(); i++) {
    float x = input_data[i];
    float y = erf(x * static_cast<float>(M_SQRT1_2));
    y = x * 0.5f * (y + 1.0f);
    output.push_back(y);
  }
  std::vector<BFloat16> output_bf16 = FloatsToBFloat16s(output);
  return output_bf16;
}

TEST_F(ActivationOpTest, Gelu_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  std::vector<float> input_values_temp{
      -1.0f, 0.0f, 1.0f,                                          // normal input values for activation
      100.0f, -100.0f, 1000.0f, -1000.0f,                         // input values that leads to exp() overflow
      FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,                       // min, denorm, -denorm
      FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity(),  // max, -max, inf
      -0.5f, 0.2f                                                 // inputs values that leads to exceed the original threshold
  };

  std::vector<BFloat16> output_bf16 = expected_output_bfloat16(input_values_temp);
  std::vector<BFloat16> input_bf16 = FloatsToBFloat16s(input_values_temp);
  OpTester tester("Gelu", 1, onnxruntime::kMSDomain);
  std::vector<int64_t> input_dims = {1, 1, 15};
  std::vector<int64_t> output_dims = input_dims;

  tester.AddInput<BFloat16>("X", input_dims, input_bf16);
  tester.AddOutput<BFloat16>("Y", output_dims, output_bf16);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#endif  // USE_DNNL

TEST_F(ActivationOpTest, QuickGelu) {
  // QuickGelu is not a single activation, some corner values in input_values will not work.
  std::vector<std::vector<float>> quick_gelu_input_values{{-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f}};

  // Positive alpha.
  {
    float alpha = 1.702f;
    TestActivationOp<float>(
        "QuickGelu", quick_gelu_input_values,
        [alpha](float x) {
          auto tmp = x * alpha;
          auto y = 1.f / (1.f + std::exp(-std::abs(tmp)));  // safe sigmoid
          y = tmp >= 0 ? y : 1 - y;
          return x * y;
        },
        {{"alpha", alpha}}, {}, false, 1, kMSDomain);
  }

  // Silu = x*sigmoid(x), i.e., alpha = 1.0f.
  {
    float alpha = 1.0f;
    TestActivationOp<float>(
        "QuickGelu", quick_gelu_input_values,
        [alpha](float x) {
          auto tmp = x * alpha;
          auto y = 1.f / (1.f + std::exp(-std::abs(tmp)));  // safe sigmoid
          y = tmp >= 0 ? y : 1 - y;
          return x * y;
        },
        {{"alpha", alpha}}, {}, false, 1, kMSDomain);
  }

  // Negative alpha.
  {
    float alpha = -1.702f;
    TestActivationOp<float>(
        "QuickGelu", quick_gelu_input_values,
        [alpha](float x) {
          auto tmp = x * alpha;
          auto y = 1.f / (1.f + std::exp(-std::abs(tmp)));  // safe sigmoid
          y = tmp >= 0 ? y : 1 - y;
          return x * y;
        },
        {{"alpha", alpha}}, {}, false, 1, kMSDomain);
  }
}

}  // namespace test
}  // namespace onnxruntime
