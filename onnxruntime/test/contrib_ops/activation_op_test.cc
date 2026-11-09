// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "gtest/gtest.h"
#include "core/common/cpuid_info.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/cpu/activation/activation_op_test.h"
#include <cmath>
#include <random>
#include <type_traits>
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

TEST_F(ActivationOpTest, Gelu_half) {
  const std::vector<MLFloat16>& X = input_values_fp16[0];
  std::vector<MLFloat16> Y;
  Y.reserve(X.size());
  for (const MLFloat16& x_half : X) {
    const float x = x_half.ToFloat();
    Y.push_back(MLFloat16(x * 0.5f * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2)))));
  }

  OpTester tester("Gelu", 1, onnxruntime::kMSDomain);
  const std::vector<int64_t> dims{1, 1, static_cast<int64_t>(X.size())};
  tester.AddInput<MLFloat16>("X", dims, X);
  tester.AddOutput<MLFloat16>("Y", dims, Y);
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

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

#if defined(USE_CUDA)
template <typename T>
static void CheckCudaQuickGeluPrecision() {
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP unavailable";
  for (float alpha : {1.0f, 1.702f, -1.702f, 0.0f}) {
    SCOPED_TRACE(alpha);
    std::vector<T> input;
    std::vector<T> expected;
    // Include float exp's subnormal/underflow region while the fused product is representable.
    for (float value : {-128.0f, -108.0f, -104.0f, -100.0f, -96.0f, -88.0f, -87.5f, -87.0f,
                        -80.0f, -64.0f, -60.0f, -52.0f, -32.0f, -24.0f, -20.0f, -16.0f, -12.0f, -8.0f, -4.0f,
                        -1.0f, -0.125f, 0.0f, 0.125f, 1.0f, 4.0f, 8.0f,
                        12.0f, 16.0f, 20.0f, 24.0f, 32.0f, 52.0f, 60.0f, 64.0f, 80.0f,
                        87.0f, 87.5f, 88.0f, 96.0f, 100.0f, 104.0f, 108.0f, 128.0f}) {
      input.emplace_back(value);
      const double scaled = static_cast<double>(value) * static_cast<double>(alpha);
      const double exponent = std::exp(-std::abs(scaled));
      const double sigmoid = scaled >= 0 ? 1.0 / (1.0 + exponent) : exponent / (1.0 + exponent);
      const double result = static_cast<double>(value) * sigmoid;
      if constexpr (std::is_same_v<T, double>) {
        expected.emplace_back(result);
      } else {
        expected.emplace_back(static_cast<float>(result));
      }
    }
    OpTester tester("QuickGelu", 1, kMSDomain);
    tester.AddAttribute("alpha", alpha);
    const std::vector<int64_t> dimensions{static_cast<int64_t>(input.size())};
    tester.AddInput<T>("X", dimensions, input);
    tester.AddOutput<T>("Y", dimensions, expected);
    tester.SetOutputAbsErr("Y", 0.0f);
    tester.SetOutputRelErr("Y", std::is_same_v<T, double> ? 2e-14f : 2e-6f);
    SessionOptions options;
    ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

TEST_F(ActivationOpTest, CudaQuickGeluFloatPrecision) {
  CheckCudaQuickGeluPrecision<float>();
}

TEST_F(ActivationOpTest, CudaQuickGeluDoublePrecision) {
  CheckCudaQuickGeluPrecision<double>();
}

TEST_F(ActivationOpTest, CudaQuickGeluFloat16Precision) {
  CheckCudaQuickGeluPrecision<MLFloat16>();
}

TEST_F(ActivationOpTest, CudaQuickGeluBFloat16Precision) {
  CheckCudaQuickGeluPrecision<BFloat16>();
}
#endif

}  // namespace test
}  // namespace onnxruntime
