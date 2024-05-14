// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"

#include <math.h>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {
void TestElementwiseGradientOp(
    const char* op,
    const std::vector<std::pair<std::string, std::vector<float>>>& inputs,
    std::function<float(const std::vector<float>&)> expected_func,
    const std::unordered_map<std::string, float> attrs = {},
    int opset_version = 7, const char* domain = kOnnxDomain) {
  const auto first_input = inputs.begin();
  ASSERT_NE(first_input, inputs.end());
  for (auto input = first_input; input != inputs.end(); ++input) {
    if (input == first_input) continue;
    ASSERT_EQ(first_input->second.size(), input->second.size());
  }

  OpTester test(op, opset_version, domain);

  for (auto attr : attrs) {
    test.AddAttribute(attr.first, attr.second);
  }

  const auto input_size = first_input->second.size();
  std::vector<int64_t> dims{static_cast<int64_t>(input_size)};

  std::vector<float> expected_vals;
  for (size_t i = 0; i < input_size; i++) {
    std::vector<float> params(inputs.size());
    std::transform(
        inputs.begin(), inputs.end(), params.begin(),
        [i](const std::pair<std::string, std::vector<float>>& input) {
          return input.second[i];
        });
    expected_vals.push_back(expected_func(params));
  }

  for (const auto& input : inputs) {
    test.AddInput<float>(input.first.c_str(), dims, input.second);
  }
  test.AddOutput<float>("dX", dims, expected_vals);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {});
}

float GeluGrad(float dy, float x) {
  return dy * (0.5f * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2))) +
               x * std::exp(-0.5f * x * x) * static_cast<float>(M_2_SQRTPI) * static_cast<float>(M_SQRT1_2) * 0.5f);
}

float GeluApproximationGrad(float dy, float x) {
  static constexpr float kAlpha = static_cast<float>(M_2_SQRTPI * M_SQRT1_2);
  static constexpr float kGamma = 0.044715f;
  static constexpr float kBeta = kAlpha * kGamma * 3.0f;

  float x_cube = x * x * x;
  float tanh_value = std::tanh(kAlpha * (x + kGamma * x_cube));
  float sech_sqr_value = 1 - tanh_value * tanh_value;
  float result = dy * 0.5f * (tanh_value + (sech_sqr_value * (kAlpha * x + kBeta * x_cube)) + 1.0f);
  return result;
}

constexpr float ReluGrad(float dy, float x) {
  return x > 0 ? dy : 0;
}

constexpr float SigmoidGrad(float dy, float y) {
  return dy * y * (1 - y);
}

constexpr float TanhGrad(float dy, float y) {
  return dy * (1 - y * y);
}

float QuickGeluGrad(float dy, float x, float alpha) {
  float v = x * alpha;
  float sigmoid = v >= 0 ? 1.f / (1.f + std::exp(-v)) : 1.f - 1.f / (1 + std::exp(v));
  return dy * sigmoid * (1 + v * (1 - sigmoid));
}

constexpr float LeakyReluGrad(float dy, float y, float alpha) {
  return dy * (y > 0.0f ? 1.0f : alpha);
}
}  // namespace

TEST(GeluGradTest, Basic) {
  const std::vector<float> x_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "GeluGrad",
      {{"dY", dY}, {"X", x_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], x = params[1];

        return GeluGrad(dy, x);
      },
      {}, 1, kMSDomain);
}

TEST(FastGeluGradTest, Basic) {
  const std::vector<float> x_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "FastGeluGrad",
      {{"dY", dY}, {"X", x_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], x = params[1];

        return GeluApproximationGrad(dy, x);
      },
      {}, 1, kMSDomain);
}

TEST(BiasGeluGradDxTest, Basic) {
  const std::vector<float> x_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);
  const std::vector<float> bias(7, 2.0f);

  TestElementwiseGradientOp(
      "BiasGeluGrad_dX",
      {{"dY", dY}, {"X", x_vals}, {"B", bias}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 3);
        const auto dy = params[0], x = params[1], b = params[2];

        return GeluGrad(dy, x + b);
      },
      {}, 1, kMSDomain);
}

TEST(BiasFastGeluGradDxTest, Basic) {
  const std::vector<float> x_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);
  const std::vector<float> bias(7, 2.0f);

  TestElementwiseGradientOp(
      "BiasFastGeluGrad_dX",
      {{"dY", dY}, {"X", x_vals}, {"B", bias}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 3);
        const auto dy = params[0], x = params[1], b = params[2];

        return GeluApproximationGrad(dy, x + b);
      },
      {}, 1, kMSDomain);
}

TEST(ReluGradTest, Basic) {
  const std::vector<float> x_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "ReluGrad",
      {{"dY", dY}, {"X", x_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], x = params[1];

        return ReluGrad(dy, x);
      },
      {}, 1, kMSDomain);
}

TEST(SigmoidGradTest, Basic) {
  const std::vector<float> y_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "SigmoidGrad",
      {{"dY", dY}, {"Y", y_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], y = params[1];

        return SigmoidGrad(dy, y);
      },
      {}, 1, kMSDomain);
}

TEST(TanhGradTest, Basic) {
  const std::vector<float> y_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "TanhGrad",
      {{"dY", dY}, {"Y", y_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], y = params[1];

        return TanhGrad(dy, y);
      },
      {}, 1, kMSDomain);
}

TEST(QuickGeluGradTest, Basic) {
  const std::vector<float> x_vals = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
  const std::vector<float> dY(5, 1.0f);

  // Positive alpha.
  {
    constexpr float alpha = 1.702f;
    TestElementwiseGradientOp(
        "QuickGeluGrad", {{"dY", dY}, {"X", x_vals}},
    // The ifdef is to suppress a warning: "lambda capture 'alpha' is not required to be captured for this use."
    // But on Windows it is required.
#ifdef __clang__
        [](const std::vector<float>& params) {
#else
        [alpha](const std::vector<float>& params) {
#endif
          ORT_ENFORCE(params.size() == 2);
          const auto dy = params[0], x = params[1];
          return QuickGeluGrad(dy, x, alpha);
        },
        {{"alpha", alpha}}, 1, kMSDomain);
  }

  // Silu = x*sigmoid(x), i.e., alpha = 1.0f.
  {
    constexpr float alpha = 1.0f;
    TestElementwiseGradientOp(
        "QuickGeluGrad", {{"dY", dY}, {"X", x_vals}},
#ifdef __clang__
        [](const std::vector<float>& params) {
#else
        [alpha](const std::vector<float>& params) {
#endif
          ORT_ENFORCE(params.size() == 2);
          const auto dy = params[0], x = params[1];
          return QuickGeluGrad(dy, x, alpha);
        },
        {{"alpha", alpha}}, 1, kMSDomain);
  }

  // Negative alpha.
  {
    constexpr float alpha = -1.702f;
    TestElementwiseGradientOp(
        "QuickGeluGrad", {{"dY", dY}, {"X", x_vals}},
#ifdef __clang__
        [](const std::vector<float>& params) {
#else
        [alpha](const std::vector<float>& params) {
#endif
          ORT_ENFORCE(params.size() == 2);
          const auto dy = params[0], x = params[1];
          return QuickGeluGrad(dy, x, alpha);
        },
        {{"alpha", alpha}}, 1, kMSDomain);
  }
}

TEST(LeakyReluGradTest, Basic) {
  const std::vector<float> y_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);
  float alpha = 0.5f;

  TestElementwiseGradientOp(
      "LeakyReluGrad",
      {{"dY", dY}, {"Y", y_vals}},
      [alpha](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], y = params[1];

        return LeakyReluGrad(dy, y, alpha);
      },
      {{"alpha", alpha}}, 1, kMSDomain);
}

namespace {
template <typename TComputeGeluGradScalarFn>
void TestBiasGeluGradBroadcastBias(const std::string& op, int opset_version, const std::string& domain,
                                   const TensorShape& input_shape,
                                   TComputeGeluGradScalarFn compute_gelu_grad_scalar_fn) {
  OpTester test(op.c_str(), opset_version, domain.c_str());

  ASSERT_TRUE(input_shape.NumDimensions() > 0 && input_shape.Size() > 0);

  const TensorShape bias_shape = input_shape.Slice(input_shape.NumDimensions() - 1);
  const auto input_size = input_shape.Size(), bias_size = bias_shape.Size();

  const std::vector<float> X = ValueRange(input_size, static_cast<float>(-input_size / 2));
  const std::vector<float> dY(input_size, 1.0f);
  const std::vector<float> B = ValueRange(bias_size, 1.0f);

  test.AddInput<float>("dY", input_shape.AsShapeVector(), dY);
  test.AddInput<float>("X", input_shape.AsShapeVector(), X);
  test.AddInput<float>("B", bias_shape.AsShapeVector(), B);

  std::vector<float> expected_dX{};
  for (int64_t i = 0; i < input_size; ++i) {
    expected_dX.push_back(compute_gelu_grad_scalar_fn(dY[i], X[i] + B[i % bias_size]));
  }

  test.AddOutput("dX", input_shape.AsShapeVector(), expected_dX);

  test.Run();
}
}  // namespace

TEST(BiasGeluGradDxTest, BroadcastBias) {
  TestBiasGeluGradBroadcastBias("BiasGeluGrad_dX", 1, kMSDomain, {2, 3, 4, 5}, GeluGrad);
  TestBiasGeluGradBroadcastBias("BiasGeluGrad_dX", 1, kMSDomain, {2, 4, 3072}, GeluGrad);
  TestBiasGeluGradBroadcastBias("BiasGeluGrad_dX", 1, kMSDomain, {2, 16384}, GeluGrad);
  TestBiasGeluGradBroadcastBias("BiasGeluGrad_dX", 1, kMSDomain, {2, 2333}, GeluGrad);
}

TEST(BiasFastGeluGradDxTest, BroadcastBias) {
  TestBiasGeluGradBroadcastBias("BiasFastGeluGrad_dX", 1, kMSDomain, {2, 3, 4, 5}, GeluApproximationGrad);
  TestBiasGeluGradBroadcastBias("BiasFastGeluGrad_dX", 1, kMSDomain, {2, 4, 3072}, GeluApproximationGrad);
  TestBiasGeluGradBroadcastBias("BiasFastGeluGrad_dX", 1, kMSDomain, {2, 16384}, GeluApproximationGrad);
  TestBiasGeluGradBroadcastBias("BiasFastGeluGrad_dX", 1, kMSDomain, {2, 2333}, GeluApproximationGrad);
}

}  // namespace test
}  // namespace onnxruntime
