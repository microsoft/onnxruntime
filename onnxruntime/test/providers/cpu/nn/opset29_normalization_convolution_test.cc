// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

#include "core/framework/customregistry.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

#if !defined(ORT_MINIMAL_BUILD)
namespace onnxruntime::test {
namespace {

// Session-local signatures exercise the production kernel registrations without
// changing the pinned ONNX dependency or globally registering unreleased schemas.
// Contract: onnx/onnx@94d0e6056931a8c1dad84104c2e567e0011d04f2.
std::shared_ptr<CustomRegistry> Opset29TestSchemas() {
  using ONNX_NAMESPACE::OpSchema;
  auto registry = std::make_shared<CustomRegistry>();
  const std::vector<std::string> types{"tensor(float)", "tensor(double)",
                                       "tensor(float16)", "tensor(bfloat16)"};
  auto mvn = OpSchema("MeanVarianceNormalization", __FILE__, __LINE__)
                 .SetDomain(kOnnxDomain)
                 .SinceVersion(29)
                 .Input(0, "X", "", "T")
                 .Output(0, "Y", "", "T")
                 .Attr("axes", "", ONNX_NAMESPACE::AttributeProto::INTS, std::vector<int64_t>{0, 2, 3})
                 .Attr("epsilon", "", ONNX_NAMESPACE::AttributeProto::FLOAT, 1e-9f)
                 .TypeConstraint("T", types, "");
  auto conv = OpSchema("CausalConvWithState", __FILE__, __LINE__)
                  .SetDomain(kOnnxDomain)
                  .SinceVersion(29)
                  .Input(0, "input", "", "T")
                  .Input(1, "weight", "", "T")
                  .Input(2, "bias", "", "T", OpSchema::Optional)
                  .Input(3, "past_state", "", "T", OpSchema::Optional)
                  .Output(0, "output", "", "T")
                  .Output(1, "present_state", "", "T")
                  .Attr("dilation", "", ONNX_NAMESPACE::AttributeProto::INT, int64_t{1})
                  .Attr("activation", "", ONNX_NAMESPACE::AttributeProto::STRING, std::string("none"))
                  .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "");
  std::vector<OpSchema> schemas{mvn, conv};
  ORT_THROW_IF_ERROR(registry->RegisterOpSet(schemas, kOnnxDomain, 28, 29));
  return registry;
}

void RunNative29(OpTester& test, bool cuda, const std::string& failure = "") {
  test.AddCustomOpRegistry(Opset29TestSchemas());
  test.SetAllowUnreleasedOnnxOpset();
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider());
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  if (cuda) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  }
  test.Run(options, failure.empty() ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
           failure, {}, nullptr, &providers);
}

template <typename T>
std::vector<T> Convert(gsl::span<const double> values) {
  std::vector<T> result;
  result.reserve(values.size());
  using SourceT = std::conditional_t<std::is_same_v<T, double>, double, float>;
  for (auto v : values) result.push_back(static_cast<T>(static_cast<SourceT>(v)));
  return result;
}

template <typename T>
void MvnCase(bool cuda, const std::vector<int64_t>& shape, const std::vector<int64_t>& axes,
             gsl::span<const double> x, gsl::span<const double> expected,
             std::optional<float> epsilon = std::nullopt) {
  OpTester test("MeanVarianceNormalization", 29);
  test.AddAttribute("axes", axes);
  if (epsilon) test.AddAttribute("epsilon", *epsilon);
  test.AddInput<T>("X", shape, Convert<T>(x));
  test.AddOutput<T>("Y", shape, Convert<T>(expected));
  test.SetOutputTolerance(1e-6f, 0.0f);
  RunNative29(test, cuda);
}

template <typename T>
void MvnCases(bool cuda) {
  // Distinguishes sqrt(var)+eps from sqrt(var+eps) and catches low-precision accumulation.
  MvnCase<T>(cuda, {2}, {0}, std::array<double, 2>{0, 2}, std::array<double, 2>{-0.5, 0.5}, 1.0f);
  MvnCase<T>(cuda, {2}, {0}, std::array<double, 2>{1, 1}, std::array<double, 2>{0, 0});
  MvnCase<T>(cuda, {2}, {}, std::array<double, 2>{992, 1008}, std::array<double, 2>{-1, 1});
  MvnCase<T>(cuda, {2}, {0}, std::array<double, 2>{1, 1.0078125}, std::array<double, 2>{-1, 1}, 0.0f);
  MvnCase<T>(cuda, {2, 2}, {-1}, std::array<double, 4>{0, 2, 4, 6},
             std::array<double, 4>{-1, 1, -1, 1});
  MvnCase<T>(cuda, {2, 2}, {0}, std::array<double, 4>{0, 2, 4, 6},
             std::array<double, 4>{-1, -1, 1, 1});
  const double denominator = std::sqrt(4.25) + 1.0;
  MvnCase<T>(cuda, {2, 2, 2}, {0, -1}, std::array<double, 8>{0, 1, 2, 3, 4, 5, 6, 7},
             std::array<double, 8>{-2.5 / denominator, -1.5 / denominator, -2.5 / denominator, -1.5 / denominator,
                                   1.5 / denominator, 2.5 / denominator, 1.5 / denominator, 2.5 / denominator},
             1.0f);
  std::vector<double> arbitrary(16), arbitrary_expected(16);
  for (size_t i = 0; i < arbitrary.size(); ++i) {
    arbitrary[i] = static_cast<double>(i);
    arbitrary_expected[i] = (i % 8 < 4 ? (i % 2 == 0 ? -2.5 : -1.5) : (i % 2 == 0 ? 1.5 : 2.5)) / denominator;
  }
  MvnCase<T>(cuda, {2, 2, 2, 2}, {1, 3}, arbitrary, arbitrary_expected, 1.0f);
  MvnCase<T>(cuda, {}, {}, std::array<double, 1>{3}, std::array<double, 1>{0});
  MvnCase<T>(cuda, {0, 3}, {1}, {}, {});
  std::vector<double> large(2 * 3 * 257), normalized(large.size());
  for (size_t i = 0; i < large.size(); ++i) {
    large[i] = i % 2 == 0 ? -1 : 1;
    normalized[i] = large[i] * 0.5;
  }
  MvnCase<T>(cuda, {2, 3, 257}, {0, 2}, large, normalized, 1.0f);
  large.resize(2 * 3 * 8193);
  normalized.resize(large.size());
  for (size_t i = 0; i < large.size(); ++i) {
    large[i] = i % 2 == 0 ? -1 : 1;
    normalized[i] = large[i] * 0.5;
  }
  MvnCase<T>(cuda, {2, 3, 8193}, {0, 2}, large, normalized, 1.0f);
}

template <typename T>
void ConvCase(bool cuda, int64_t length, int64_t kernel, int64_t dilation,
              const std::vector<double>& x, const std::vector<double>& w,
              const std::optional<std::vector<double>>& previous,
              const std::vector<double>& expected, const std::vector<double>& next,
              std::string activation = "none", std::optional<double> bias = std::nullopt) {
  OpTester test("CausalConvWithState", 29);
  test.AddAttribute("dilation", dilation);
  test.AddAttribute("activation", activation);
  test.AddInput<T>("input", {1, 1, length}, Convert<T>(x));
  test.AddInput<T>("weight", {1, 1, kernel}, Convert<T>(w));
  if (bias) {
    test.AddInput<T>("bias", {1}, Convert<T>(std::array<double, 1>{*bias}));
  } else {
    test.AddOptionalInputEdge<T>();
  }
  if (previous) test.AddInput<T>("past_state", {1, 1, (kernel - 1) * dilation}, Convert<T>(*previous));
  test.AddOutput<T>("output", {1, 1, length}, Convert<T>(expected));
  test.AddOutput<T>("present_state", {1, 1, (kernel - 1) * dilation}, Convert<T>(next));
  test.SetOutputTolerance(1e-6f, 0.0f);
  RunNative29(test, cuda);
}

template <typename T>
void ConvCases(bool cuda) {
  ConvCase<T>(cuda, 3, 3, 2, {5, 6, 7}, {1, 10, 100}, std::vector<double>{1, 2, 3, 4},
              {531, 642, 753}, {4, 5, 6, 7});
  ConvCase<T>(cuda, 1, 3, 2, {8}, {1, 10, 100}, std::vector<double>{4, 5, 6, 7},
              {864}, {5, 6, 7, 8});
  ConvCase<T>(cuda, 1, 3, 2, {8}, {1, 10, 100}, std::nullopt, {800}, {0, 0, 0, 8});
  ConvCase<T>(cuda, 0, 3, 2, {}, {1, 10, 100}, std::vector<double>{1, 2, 3, 4}, {}, {1, 2, 3, 4});
  ConvCase<T>(cuda, 0, 3, 2, {}, {1, 10, 100}, std::nullopt, {}, {0, 0, 0, 0});
  ConvCase<T>(cuda, 3, 1, 4, {1, 2, 3}, {2}, std::nullopt, {3, 5, 7}, {}, "none", 1);
  ConvCase<T>(cuda, 0, 1, 7, {}, {2}, std::nullopt, {}, {});
  // Compute SiLU from the rounded Conv result (noticeable for fp16/bf16).
  const float rounded = static_cast<float>(static_cast<T>(0.703125f * 0.703125f + 0.0078125f));
  const double activated = rounded / (1.0f + std::exp(-rounded));
  ConvCase<T>(cuda, 1, 1, 1, {0.703125}, {0.703125}, std::nullopt, {activated}, {},
              "silu", 0.0078125);
  ConvCase<T>(cuda, 1, 1, 1, {0.703125}, {0.703125}, std::nullopt, {activated}, {},
              "swish", 0.0078125);
}

template <typename T>
void ConvPrefill(bool cuda, int64_t length = 257, int64_t kernel = 5, int64_t dilation = 3) {
  constexpr int64_t batch = 2, channels = 3;
  const int64_t history = (kernel - 1) * dilation;
  std::vector<T> input(batch * channels * length), weight(channels * kernel),
      past(batch * channels * history), expected(input.size()), next(past.size());
  for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<T>((static_cast<int>(i % 17) - 8) * 0.125f);
  for (size_t i = 0; i < weight.size(); ++i) weight[i] = static_cast<T>((static_cast<int>(i % 7) - 3) * 0.125f);
  for (size_t i = 0; i < past.size(); ++i) past[i] = static_cast<T>((static_cast<int>(i % 11) - 5) * 0.125f);
  for (int64_t row = 0; row < batch * channels; ++row) {
    std::vector<T> concatenated(past.begin() + row * history, past.begin() + (row + 1) * history);
    concatenated.insert(concatenated.end(), input.begin() + row * length, input.begin() + (row + 1) * length);
    for (int64_t t = 0; t < length; ++t) {
      float sum = 0;
      for (int64_t k = 0; k < kernel; ++k) {
        sum += static_cast<float>(concatenated[t + k * dilation]) *
               static_cast<float>(weight[(row % channels) * kernel + k]);
      }
      expected[row * length + t] = static_cast<T>(sum);
    }
    std::copy(concatenated.end() - history, concatenated.end(), next.begin() + row * history);
  }
  OpTester test("CausalConvWithState", 29);
  test.AddAttribute("dilation", dilation);
  test.AddInput<T>("input", {batch, channels, length}, input);
  test.AddInput<T>("weight", {channels, 1, kernel}, weight);
  test.AddOptionalInputEdge<T>();
  test.AddInput<T>("past_state", {batch, channels, history}, past);
  test.AddOutput<T>("output", {batch, channels, length}, expected);
  test.AddOutput<T>("present_state", {batch, channels, history}, next);
  test.SetOutputTolerance(1e-6f, 0.0f);
  RunNative29(test, cuda);
}

TEST(Opset29NormalizationConvolution, Cpu) {
  MvnCases<float>(false);
  MvnCases<double>(false);
  MvnCases<MLFloat16>(false);
  MvnCases<BFloat16>(false);
  ConvCases<float>(false);
  ConvCases<MLFloat16>(false);
  ConvCases<BFloat16>(false);
  ConvPrefill<float>(false);
  ConvPrefill<MLFloat16>(false);
  ConvPrefill<BFloat16>(false);
  ConvPrefill<float>(false, 1, 3, 2);
  ConvPrefill<float>(false, 193, 4, 1);
}

TEST(Opset29NormalizationConvolution, DefaultAxesAndDoublePrecision) {
  OpTester default_axes("MeanVarianceNormalization", 29);
  default_axes.AddInput<float>("X", {2, 2, 1, 1}, {0, 2, 4, 6});
  default_axes.AddOutput<float>("Y", {2, 2, 1, 1}, {-1, -1, 1, 1});
  RunNative29(default_axes, false);

  OpTester precise("MeanVarianceNormalization", 29);
  precise.AddAttribute("axes", std::vector<int64_t>{0});
  precise.AddAttribute("epsilon", 0.1f);
  const double result = 1.0 / (1.0 + static_cast<double>(0.1f));
  precise.AddInput<double>("X", {2}, {1e12, 1e12 + 2});
  precise.AddOutput<double>("Y", {2}, {-result, result});
  precise.SetOutputTolerance(1e-13f, 0.0f);
  RunNative29(precise, false);
}

TEST(Opset29NormalizationConvolution, Cuda) {
#ifdef USE_CUDA
  if (!DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP is unavailable";
  MvnCases<float>(true);
  MvnCases<double>(true);
  MvnCases<MLFloat16>(true);
  MvnCases<BFloat16>(true);
  ConvCases<float>(true);
  ConvCases<MLFloat16>(true);
  ConvCases<BFloat16>(true);
  ConvPrefill<float>(true);
  ConvPrefill<MLFloat16>(true);
  ConvPrefill<BFloat16>(true);
  ConvPrefill<float>(true, 1, 3, 2);
  ConvPrefill<float>(true, 193, 4, 1);
#else
  GTEST_SKIP() << "CUDA EP is not built";
#endif
}

TEST(Opset29NormalizationConvolution, InvalidAxes) {
  OpTester test("MeanVarianceNormalization", 29);
  test.AddAttribute("axes", std::vector<int64_t>{2});
  test.AddInput<float>("X", {2}, {0, 2});
  test.AddOutput<float>("Y", {2}, {0, 0});
  RunNative29(test, false, "axis 2 is invalid for rank 1");
}

TEST(Opset29NormalizationConvolution, EmptyBatch) {
  OpTester test("CausalConvWithState", 29);
  test.AddAttribute("dilation", int64_t{4});
  test.AddInput<float>("input", {0, 3, 7}, {});
  test.AddInput<float>("weight", {3, 1, 2}, {1, 1, 1, 1, 1, 1});
  test.AddOutput<float>("output", {0, 3, 7}, {});
  test.AddOutput<float>("present_state", {0, 3, 4}, {});
  RunNative29(test, false);
}

TEST(Opset29NormalizationConvolution, InvalidDilation) {
  OpTester test("CausalConvWithState", 29);
  test.AddAttribute("dilation", int64_t{0});
  test.AddInput<float>("input", {1, 1, 1}, {1});
  test.AddInput<float>("weight", {1, 1, 1}, {1});
  test.AddOutput<float>("output", {1, 1, 1}, {1});
  test.AddOutput<float>("present_state", {1, 1, 0}, {});
  RunNative29(test, false, "requires dilation >= 1");
}

TEST(Opset29NormalizationConvolution, InvalidStateShape) {
  OpTester test("CausalConvWithState", 29);
  test.AddAttribute("dilation", int64_t{2});
  test.AddInput<float>("input", {1, 1, 1}, {1});
  test.AddInput<float>("weight", {1, 1, 3}, {1, 1, 1});
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("past_state", {1, 1, 2}, {0, 0});
  test.AddOutput<float>("output", {1, 1, 1}, {1});
  test.AddOutput<float>("present_state", {1, 1, 4}, {0, 0, 0, 1});
  RunNative29(test, false, "past_state must have shape");
}

TEST(Opset29NormalizationConvolution, HistoryOverflow) {
  OpTester test("CausalConvWithState", 29);
  test.AddAttribute("dilation", std::numeric_limits<int64_t>::max());
  test.AddInput<float>("input", {1, 1, 1}, {1});
  test.AddInput<float>("weight", {1, 1, 2}, {1, 1});
  test.AddOutput<float>("output", {1, 1, 1}, {1});
  test.AddOutput<float>("present_state", {1, 1, 0}, {});
  RunNative29(test, false, "requires dilation >= 1 and (K-1)*dilation+1 to fit int64");
}

}  // namespace
}  // namespace onnxruntime::test
#endif
