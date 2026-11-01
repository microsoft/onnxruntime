// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/framework/int4.h"
#include "core/framework/to_tensor_proto_element_type.h"
#include "core/graph/graph.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/webgpu/graph_capture_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

class DequantizeLinearCacheTester final : public OpTester {
 public:
  explicit DequantizeLinearCacheTester(bool scalar_first)
      : OpTester("DequantizeLinear", 19), scalar_first_(scalar_first) {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DequantizeLinearCacheTester);

  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>&) override {
    for (int i : {0, 1}) {
      const int index = scalar_first_ ? i : 1 - i;
      graph.AddNode(index == 0 ? "scalar_dq" : "vector_dq", "DequantizeLinear", "",
                    {inputs[index], inputs[2], inputs[3]}, {outputs[index]});
    }
  }

 private:
  const bool scalar_first_;
};

template <typename QuantizedType, typename FloatType>
void TestPackedDequantizeLinearCache() {
  const std::array<QuantizedType, 6> scalar_input{std::numeric_limits<QuantizedType>::min(), 1, 2, 3, 4,
                                                  std::numeric_limits<QuantizedType>::max()};
  const std::array<QuantizedType, 4> vector_input{std::numeric_limits<QuantizedType>::max(), 4, 2,
                                                  std::numeric_limits<QuantizedType>::min()};
  constexpr QuantizedType zero_point = 3;
  auto dequantize = [](const auto& input) {
    std::array<FloatType, std::tuple_size_v<std::decay_t<decltype(input)>>> output;
    for (size_t i = 0; i < input.size(); ++i) {
      output[i] = FloatType((static_cast<int32_t>(input[i]) - zero_point) * 0.5f);
    }
    return output;
  };
  const auto scalar_output = dequantize(scalar_input);
  const auto vector_output = dequantize(vector_input);

  for (bool scalar_first : {true, false}) {
    SCOPED_TRACE(scalar_first);
    auto provider = DefaultWebGpuExecutionProvider();
    if (!provider) {
      GTEST_SKIP() << "WebGPU EP is not available";
    }

    // Both kernels have packed u32 inputs and rank-1 output views, but only the
    // four-element output uses vec4. They must not share a cached shader.
    DequantizeLinearCacheTester test(scalar_first);
    test.AddInput<QuantizedType>("scalar_input", {6}, scalar_input.data(), scalar_input.size());
    test.AddInput<QuantizedType>("vector_input", {1, 4}, vector_input.data(), vector_input.size());
    test.AddInput<FloatType>("scale", {}, {FloatType(0.5f)});
    test.AddInput<QuantizedType>("zero_point", {}, {zero_point});
    test.AddOutput<FloatType>("scalar_output", {6}, scalar_output.data(), scalar_output.size());
    test.AddOutput<FloatType>("vector_output", {1, 4}, vector_output.data(), vector_output.size());

    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
  }
}

TEST(WebGpuDequantizeLinearTest, Uint8FloatScalarAndVectorCache) {
  TestPackedDequantizeLinearCache<uint8_t, float>();
}

TEST(WebGpuDequantizeLinearTest, Int8FloatScalarAndVectorCache) {
  TestPackedDequantizeLinearCache<int8_t, float>();
}

TEST(WebGpuDequantizeLinearTest, Uint8Float16ScalarAndVectorCache) {
  TestPackedDequantizeLinearCache<uint8_t, MLFloat16>();
}

TEST(WebGpuDequantizeLinearTest, Int8Float16ScalarAndVectorCache) {
  TestPackedDequantizeLinearCache<int8_t, MLFloat16>();
}

template <typename T>
void TestEmptyDequantization(const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& scale_shape,
                             const std::vector<int64_t>* zero_point_shape,
                             int64_t block_size, const char* error = "") {
  ConfigOptions config;
  ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kValidationMode,
                                         webgpu::options::kValidationMode_full));
  ASSERT_STATUS_OK(config.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                         webgpu::options::kEnableGraphCapture_ON));
  auto provider = WebGpuExecutionProviderWithOptions(config);
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }

  WebGpuGraphCaptureTester test("DequantizeLinear", 21);
  // Exercise kernel validation rather than model-load shape inference.
  test.AddShapeToTensorData(false);
  test.AddAttribute("axis", int64_t{0});
  test.AddAttribute("block_size", block_size);
  test.AddInput<int8_t>("x", input_shape, {});
  test.AddInput<T>("x_scale", scale_shape,
                   std::vector<T>(TensorShape(scale_shape).Size(), T{0.5f}));
  if (zero_point_shape) {
    test.AddInput<int8_t>("x_zero_point", *zero_point_shape,
                          std::vector<int8_t>(TensorShape(*zero_point_shape).Size(), 0));
  }
  test.AddOutput<T>("y", input_shape, {});
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  options.graph_optimization_level = TransformerLevel::Default;
  const bool expect_failure = error[0] != '\0';
  if (expect_failure) {
    test.Config(options)
        .Config(OpTester::ExpectResult::kExpectFailure, error)
        .ConfigEp(std::move(provider))
        .RunWithConfig();
  } else {
    test.RunWithBoundCapture<T>(std::move(provider), options, TensorShape(input_shape), {});
  }
}

TEST(WebGpuDequantizeLinearTest, EmptyBlockedShapesAreValidatedBeforeNoDispatch) {
  for (const auto& shape : {std::vector<int64_t>{35, 0}, std::vector<int64_t>{0, 16}}) {
    const std::vector<int64_t> scale_shape{(shape[0] + 31) / 32, shape[1]};
    TestEmptyDequantization<float>(shape, scale_shape, nullptr, 32);
    TestEmptyDequantization<MLFloat16>(shape, scale_shape, &scale_shape, 32);
  }
  TestEmptyDequantization<float>({35, 0}, {3, 0}, nullptr, 32,
                                 "x_scale must be ceil(Di/block_size)");
  TestEmptyDequantization<float>({0, 16}, {0, 15}, nullptr, 32,
                                 "same shape on non-quantize axes");
  TestEmptyDequantization<float>({0, 16}, {0, 16, 1}, nullptr, 32,
                                 "same rank for blocked quantization");
  TestEmptyDequantization<float>({35, 0}, {0}, nullptr, 32,
                                 "same rank for blocked quantization");
  const std::vector<int64_t> wrong_zero_shape{2, 1};
  TestEmptyDequantization<float>({35, 0}, {2, 0}, &wrong_zero_shape, 32,
                                 "x_zero_point and x_scale must have the same shape");
  const std::vector<int64_t> wrong_zero_rank{0};
  TestEmptyDequantization<float>({35, 0}, {2, 0}, &wrong_zero_rank, 32,
                                 "x_zero_point and x_scale must have the same rank");
}

TEST(WebGpuDequantizeLinearTest, EmptyShapeBlockInferenceIsSafe) {
  TestEmptyDequantization<float>({32, 0}, {2, 0}, nullptr, 0);
  TestEmptyDequantization<MLFloat16>({0, 16}, {0, 16}, nullptr, 0);
  TestEmptyDequantization<float>({35, 0}, {0, 0}, nullptr, 0,
                                 "x_scale must be nonempty on a nonempty quantize axis");
  TestEmptyDequantization<float>({35, 0}, {0}, nullptr, 0,
                                 "x_scale must be nonempty on a nonempty quantize axis");
}

TEST(WebGpuDequantizeLinearTest, EmptyPerTensorAndPerAxisShapesAreValidated) {
  const std::vector<int64_t> scalar{};
  const std::vector<int64_t> empty_axis{0};
  TestEmptyDequantization<float>({0, 16}, {}, &scalar, 0);
  TestEmptyDequantization<MLFloat16>({0, 16}, {0}, &empty_axis, 0);
  TestEmptyDequantization<float>({0, 16}, {}, &empty_axis, 0,
                                 "x_zero_point must be a scalar or size-1 vector");
  TestEmptyDequantization<float>({0, 16}, {2}, nullptr, 0,
                                 "x_scale must match the quantize axis");
  const std::vector<int64_t> wrong_zero_shape{2};
  TestEmptyDequantization<float>({0, 16}, {0}, &wrong_zero_shape, 0,
                                 "x_zero_point must match x_scale");
}

template <typename T>
std::vector<T> FromFloats(const std::vector<float>& values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (float v : values) {
    result.push_back(T(v));
  }
  return result;
}

SessionOptions WebGpuOnlySessionOptions() {
  SessionOptions options;
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  return options;
}

// WebNN conformance: dequantizeLinear uint8 1D tensor with 1D scale, implicit block_size = 2
// (issue #32730). Covers both an explicit block_size and one inferred from the shapes.
template <typename FloatType>
void TestDequantizeLinearUint8OneDimensionBlocked(int64_t block_size, float abs_error) {
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  const std::vector<uint8_t> x{12, 24, 35, 123};
  const std::vector<FloatType> scale = FromFloats<FloatType>({9.343092918395996f, 4.617084980010986f});
  const std::vector<uint8_t> zero_point{128, 110};
  std::vector<float> expected;
  for (size_t i = 0; i < x.size(); ++i) {
    expected.push_back((static_cast<float>(x[i]) - zero_point[i / 2]) * static_cast<float>(scale[i / 2]));
  }

  OpTester test("DequantizeLinear", 21);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddInput<uint8_t>("x", {4}, x);
  test.AddInput<FloatType>("x_scale", {2}, scale, true);
  test.AddInput<uint8_t>("x_zero_point", {2}, zero_point, true);
  test.AddOutput<FloatType>("y", {4}, FromFloats<FloatType>(expected));
  test.SetOutputAbsErr("y", abs_error);
  test.Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(provider)).RunWithConfig();
}

TEST(WebGpuDequantizeLinearTest, Uint8OneDimensionBlockSize2) {
  for (int64_t block_size : {0, 2}) {
    SCOPED_TRACE(block_size);
    TestDequantizeLinearUint8OneDimensionBlocked<float>(block_size, 1e-3f);
    TestDequantizeLinearUint8OneDimensionBlocked<MLFloat16>(block_size, 1.0f);
  }
}

// Runs QuantizeLinear on WebGPU only. The scale has the input's float type; the zero point (if
// any) has the scale's shape.
template <typename FloatType, typename QuantType>
void RunQuantizeLinear(int opset,
                       const std::vector<int64_t>& x_shape, const std::vector<float>& x,
                       const std::vector<int64_t>& scale_shape, const std::vector<float>& scale,
                       const std::vector<QuantType>* zero_point,
                       const std::vector<QuantType>& expected,
                       int64_t axis = 1, int64_t block_size = 0) {
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  OpTester test("QuantizeLinear", opset);
  if (opset >= 13) {
    test.AddAttribute<int64_t>("axis", axis);
  }
  if (block_size != 0) {
    test.AddAttribute<int64_t>("block_size", block_size);
  }
  test.AddInput<FloatType>("x", x_shape, FromFloats<FloatType>(x));
  test.AddInput<FloatType>("y_scale", scale_shape, FromFloats<FloatType>(scale));
  if (zero_point != nullptr) {
    test.AddInput<QuantType>("y_zero_point", scale_shape, *zero_point);
  }
  test.AddOutput<QuantType>("y", x_shape, expected);
  test.Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(provider)).RunWithConfig();
}

template <typename FloatType>
void TestQuantizeLinearWebNNConformance() {
  // quantizeLinear float32 0D tensor with int8 0D zeroPoint.
  const std::vector<int8_t> int8_zero_point{1};
  RunQuantizeLinear<FloatType, int8_t>(21, {}, {10.794857501983643f}, {}, {1.1202747821807861f},
                                       &int8_zero_point, {11});

  const std::vector<float> x{-2.549168109893799f, -4.794857501983643f, 8.413617134094238f, 6.108623504638672f};
  // 1D tensor with uint8 1D zeroPoint (per-axis).
  const std::vector<uint8_t> zero_point_4(4, 128);
  RunQuantizeLinear<FloatType, uint8_t>(
      21, {4}, x, {4}, {9.343092918395996f, 0.2800687253475189f, 4.617084980010986f, 1.1202747821807861f},
      &zero_point_4, {128, 111, 130, 133}, /*axis*/ 0);
  // 4D tensor broadcasting scale and zeroPoint: one blocked axis.
  const std::vector<uint8_t> zero_point_2(2, 128);
  RunQuantizeLinear<FloatType, uint8_t>(21, {1, 1, 2, 2}, x, {1, 1, 2, 1},
                                        {0.2800687253475189f, 4.617084980010986f}, &zero_point_2,
                                        {119, 111, 130, 129}, /*axis*/ 3, /*block_size*/ 2);
  // Scale with the input's shape, emitted by Chromium as axis = 0, block_size = 1.
  RunQuantizeLinear<FloatType, uint8_t>(
      21, {1, 1, 2, 2}, x, {1, 1, 2, 2},
      {0.2800687253475189f, 4.617084980010986f, 0.2800687253475189f, 4.617084980010986f},
      &zero_point_4, {119, 127, 158, 129}, /*axis*/ 0, /*block_size*/ 1);
  // 3D tensor with implicit block_size = [1, 2, 1].
  const std::vector<uint8_t> blocked_zero_point{128, 189};
  RunQuantizeLinear<FloatType, uint8_t>(21, {1, 4, 1}, x, {1, 2, 1},
                                        {0.2800687253475189f, 4.617084980010986f}, &blocked_zero_point,
                                        {119, 111, 191, 190}, /*axis*/ 1, /*block_size*/ 2);
}

TEST(WebGpuQuantizeLinearTest, WebNNConformance_Float) {
  TestQuantizeLinearWebNNConformance<float>();
}

TEST(WebGpuQuantizeLinearTest, WebNNConformance_Float16) {
  TestQuantizeLinearWebNNConformance<MLFloat16>();
}

template <typename FloatType>
void TestQuantizeLinearRoundingAndSaturation() {
  // Ties round to even. Nine elements, so the last packed word is partial.
  const std::vector<int8_t> int8_zero{0};
  RunQuantizeLinear<FloatType, int8_t>(21, {9}, {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 3.5f, -3.5f, 0.0f},
                                       {}, {1.0f}, &int8_zero, {0, 2, 2, 0, -2, -2, 4, -4, 0});
  RunQuantizeLinear<FloatType, int8_t>(21, {3}, {0.25f, 0.75f, 1.25f}, {}, {0.5f}, &int8_zero, {0, 2, 2});

  // Saturation, with and without a zero point offset.
  RunQuantizeLinear<FloatType, int8_t>(21, {6}, {1000.0f, -1000.0f, 127.25f, -128.75f, 300.0f, -0.25f},
                                       {}, {1.0f}, &int8_zero, {127, -128, 127, -128, 127, 0});
  const std::vector<int8_t> int8_offset{10};
  RunQuantizeLinear<FloatType, int8_t>(21, {3}, {120.0f, -140.0f, -5.0f}, {}, {1.0f}, &int8_offset,
                                       {127, -128, 5});
  const std::vector<uint8_t> uint8_zero{0};
  RunQuantizeLinear<FloatType, uint8_t>(21, {5}, {-5.0f, 300.0f, 255.25f, 254.5f, 0.5f}, {}, {1.0f},
                                        &uint8_zero, {0, 255, 255, 254, 0});
  const std::vector<uint8_t> uint8_offset{128};
  RunQuantizeLinear<FloatType, uint8_t>(21, {4}, {-200.0f, 200.0f, -128.0f, 127.0f}, {}, {1.0f},
                                        &uint8_offset, {0, 255, 0, 255});
}

TEST(WebGpuQuantizeLinearTest, RoundingAndSaturation_Float) {
  TestQuantizeLinearRoundingAndSaturation<float>();
}

TEST(WebGpuQuantizeLinearTest, RoundingAndSaturation_Float16) {
  TestQuantizeLinearRoundingAndSaturation<MLFloat16>();
}

// Reference quantization; the scales used with it are powers of two so x / scale is exact.
template <typename QuantType>
QuantType ReferenceQuantize(float x, float scale, QuantType zero_point) {
  const float q = std::nearbyint(x / scale) + static_cast<float>(zero_point);
  return static_cast<QuantType>(std::clamp(q, static_cast<float>(std::numeric_limits<QuantType>::min()),
                                           static_cast<float>(std::numeric_limits<QuantType>::max())));
}

std::vector<float> QuarterSteps(size_t count) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = 0.25f * static_cast<float>(static_cast<int64_t>(i % 97) - 48);
  }
  return values;
}

template <typename FloatType, typename QuantType>
void TestQuantizeLinearPerAxisAndBlocked() {
  // Per-axis along the middle axis of a [3, 5, 2] tensor (30 elements, partial last word).
  {
    const std::vector<int64_t> shape{3, 5, 2};
    const auto x = QuarterSteps(30);
    const std::vector<float> scale{0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
    const std::vector<QuantType> zero_point{0, 3, 5, 7, 9};
    std::vector<QuantType> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      const size_t c = (i / 2) % 5;
      expected[i] = ReferenceQuantize(x[i], scale[c], zero_point[c]);
    }
    RunQuantizeLinear<FloatType, QuantType>(21, shape, x, {5}, scale, &zero_point, expected, /*axis*/ 1);
    RunQuantizeLinear<FloatType, QuantType>(21, shape, x, {5}, scale, &zero_point, expected, /*axis*/ -2);
  }
  // Blocked along the last axis with a partial final block: [2, 5] input, block_size 2 -> [2, 3] scale.
  {
    const std::vector<int64_t> shape{2, 5};
    const auto x = QuarterSteps(10);
    const std::vector<float> scale{0.25f, 0.5f, 1.0f, 2.0f, 0.125f, 0.5f};
    const std::vector<QuantType> zero_point{1, 2, 3, 4, 5, 6};
    std::vector<QuantType> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      const size_t s = (i / 5) * 3 + (i % 5) / 2;
      expected[i] = ReferenceQuantize(x[i], scale[s], zero_point[s]);
    }
    RunQuantizeLinear<FloatType, QuantType>(21, shape, x, {2, 3}, scale, &zero_point, expected,
                                            /*axis*/ 1, /*block_size*/ 2);
  }
  // Blocked along the first axis.
  {
    const std::vector<int64_t> shape{4, 3};
    const auto x = QuarterSteps(12);
    const std::vector<float> scale{0.25f, 0.5f, 1.0f, 2.0f, 0.125f, 0.5f};
    const std::vector<QuantType> zero_point{6, 5, 4, 3, 2, 1};
    std::vector<QuantType> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      const size_t s = ((i / 3) / 2) * 3 + i % 3;
      expected[i] = ReferenceQuantize(x[i], scale[s], zero_point[s]);
    }
    RunQuantizeLinear<FloatType, QuantType>(21, shape, x, {2, 3}, scale, &zero_point, expected,
                                            /*axis*/ 0, /*block_size*/ 2);
  }
}

TEST(WebGpuQuantizeLinearTest, PerAxisAndBlocked_Float_Int8) {
  TestQuantizeLinearPerAxisAndBlocked<float, int8_t>();
}

TEST(WebGpuQuantizeLinearTest, PerAxisAndBlocked_Float_Uint8) {
  TestQuantizeLinearPerAxisAndBlocked<float, uint8_t>();
}

TEST(WebGpuQuantizeLinearTest, PerAxisAndBlocked_Float16_Int8) {
  TestQuantizeLinearPerAxisAndBlocked<MLFloat16, int8_t>();
}

TEST(WebGpuQuantizeLinearTest, PerAxisAndBlocked_Float16_Uint8) {
  TestQuantizeLinearPerAxisAndBlocked<MLFloat16, uint8_t>();
}

// Quotients that are not exactly representable, including inputs that sit within rounding error of
// a .5 tie, must match the CPU kernel, which rounds a correctly rounded f32 quotient. GPU f32
// division alone is not accurate enough (e.g. -5 / 0.039215686 is -127.4999..., not -127.5).
template <typename FloatType, typename QuantType>
void TestQuantizeLinearInexactQuotients(float scale) {
  const float s = static_cast<float>(FloatType(scale));
  std::vector<float> x{5.0f, -5.0f};
  for (int k = -140; k <= 140; ++k) {
    x.push_back(static_cast<float>(FloatType((static_cast<float>(k) + 0.5f) * s)));
    x.push_back(static_cast<float>(FloatType(static_cast<float>(k) * 0.0137f)));
  }
  const QuantType zero_point = std::is_signed_v<QuantType> ? QuantType(-3) : QuantType(120);
  std::vector<QuantType> expected(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    expected[i] = ReferenceQuantize(x[i], s, zero_point);
  }
  const std::vector<QuantType> zero_point_vec{zero_point};
  RunQuantizeLinear<FloatType, QuantType>(21, {static_cast<int64_t>(x.size())}, x, {}, {s}, &zero_point_vec,
                                          expected);
}

TEST(WebGpuQuantizeLinearTest, InexactQuotients) {
  for (float scale : {0.039215686f, 0.1f, 1.0f / 3.0f, 0.0137f, 7.3f}) {
    SCOPED_TRACE(scale);
    TestQuantizeLinearInexactQuotients<float, int8_t>(scale);
    TestQuantizeLinearInexactQuotients<float, uint8_t>(scale);
    TestQuantizeLinearInexactQuotients<MLFloat16, int8_t>(scale);
    TestQuantizeLinearInexactQuotients<MLFloat16, uint8_t>(scale);
  }
}

TEST(WebGpuQuantizeLinearTest, OpsetVersions) {
  // Large enough for multiple workgroups.
  const auto x = QuarterSteps(1001);
  std::vector<int8_t> expected_int8(x.size());
  std::vector<uint8_t> expected_uint8(x.size());
  std::vector<uint8_t> expected_no_zero_point(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    expected_int8[i] = ReferenceQuantize<int8_t>(x[i], 0.125f, -3);
    expected_uint8[i] = ReferenceQuantize<uint8_t>(x[i], 0.125f, 100);
    expected_no_zero_point[i] = ReferenceQuantize<uint8_t>(x[i], 0.125f, 0);
  }
  const std::vector<int8_t> int8_zero_point{-3};
  const std::vector<uint8_t> uint8_zero_point{100};
  for (int opset : {10, 13, 19, 21, 23, 24, 25, 28}) {
    SCOPED_TRACE(opset);
    RunQuantizeLinear<float, int8_t>(opset, {1001}, x, {}, {0.125f}, &int8_zero_point, expected_int8);
    RunQuantizeLinear<float, uint8_t>(opset, {1001}, x, {}, {0.125f}, &uint8_zero_point, expected_uint8);
    RunQuantizeLinear<float, uint8_t>(opset, {1001}, x, {}, {0.125f}, nullptr, expected_no_zero_point);
  }
}

// From opset 23, x / y_scale is computed in the `precision` type, which defaults to the y_scale type.
// The WebGPU kernel divides in f32, so a float16 y_scale or an explicit float16 precision must be left
// to the CPU EP. CPU fallback stays enabled; an f32 division on WebGPU would give 5 and 3 below.
// Opset 28 is omitted because the CPU EP has no QuantizeLinear-28 kernel to fall back to.
TEST(WebGpuQuantizeLinearTest, Opset23PrecisionOtherThanFloat) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  for (int opset : {23, 24, 25}) {
    SCOPED_TRACE(opset);
    {
      // 1.5 / 0.333251953125 is 4.5011 in f32 but 4.5 in float16, which rounds half to even to 4.
      OpTester test("QuantizeLinear", opset);
      test.AddInput<MLFloat16>("x", {1}, {MLFloat16(1.5f)});
      test.AddInput<MLFloat16>("y_scale", {}, {MLFloat16(0.333251953125f)});
      test.AddInput<int8_t>("y_zero_point", {}, {0});
      test.AddOutput<int8_t>("y", {1}, {4});
      test.ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
    }
    {
      // 2.5001 is 2.5 in float16, which rounds half to even to 2.
      OpTester test("QuantizeLinear", opset);
      test.AddAttribute<int64_t>("precision", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
      test.AddInput<float>("x", {1}, {2.5001f});
      test.AddInput<float>("y_scale", {}, {1.0f});
      test.AddInput<int8_t>("y_zero_point", {}, {0});
      test.AddOutput<int8_t>("y", {1}, {2});
      test.ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
    }
  }
}

// An explicit f32 precision matches the kernel's division and stays on WebGPU.
TEST(WebGpuQuantizeLinearTest, Opset23ExplicitFloatPrecision) {
  for (int opset : {23, 24, 25, 28}) {
    SCOPED_TRACE(opset);
    auto provider = DefaultWebGpuExecutionProvider();
    if (!provider) {
      GTEST_SKIP() << "WebGPU EP is not available";
    }
    OpTester test("QuantizeLinear", opset);
    test.AddAttribute<int64_t>("precision", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    test.AddInput<float>("x", {3}, {2.5001f, -2.5f, 7.75f});
    test.AddInput<float>("y_scale", {}, {1.0f});
    test.AddInput<int8_t>("y_zero_point", {}, {0});
    test.AddOutput<int8_t>("y", {3}, {3, -2, 8});
    test.Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(provider)).RunWithConfig();
  }
}

// QuantizeLinear followed by DequantizeLinear, as in WebNN's "quantizeLinear then
// dequantizeLinear" conformance tests.
template <typename QuantType>
class QuantizeDequantizeTester final : public OpTester {
 public:
  QuantizeDequantizeTester() : OpTester("QuantizeLinear", 21) {}

  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>&) override {
    ONNX_NAMESPACE::TypeProto quantized_type;
    quantized_type.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<QuantType>());
    auto& quantized = graph.GetOrCreateNodeArg("quantized", &quantized_type);
    graph.AddNode("quantize", "QuantizeLinear", "", {inputs[0], inputs[1], inputs[2]}, {&quantized});
    graph.AddNode("dequantize", "DequantizeLinear", "", {&quantized, inputs[1], inputs[2]}, {outputs[0]});
  }
};

template <typename FloatType, typename QuantType>
void TestQuantizeDequantizeRoundTrip() {
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  const std::vector<float> x{-2.549168109893799f, -4.794857501983643f, 8.413617134094238f, 6.108623504638672f,
                             0.0f, 100.0f, -100.0f};
  constexpr float kScale = 0.125f;
  constexpr QuantType kZeroPoint = 7;
  std::vector<float> expected;
  for (float v : x) {
    expected.push_back((static_cast<float>(ReferenceQuantize(v, kScale, kZeroPoint)) - kZeroPoint) * kScale);
  }

  QuantizeDequantizeTester<QuantType> test;
  test.template AddInput<FloatType>("x", {7}, FromFloats<FloatType>(x));
  test.template AddInput<FloatType>("scale", {}, {FloatType(kScale)});
  test.template AddInput<QuantType>("zero_point", {}, {kZeroPoint});
  test.template AddOutput<FloatType>("y", {7}, FromFloats<FloatType>(expected));
  SessionOptions options = WebGpuOnlySessionOptions();
  // Keep graph optimizers from folding away the Q -> DQ pair.
  options.graph_optimization_level = TransformerLevel::Default;
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}

TEST(WebGpuQuantizeLinearTest, QuantizeDequantizeRoundTrip) {
  TestQuantizeDequantizeRoundTrip<float, uint8_t>();
  TestQuantizeDequantizeRoundTrip<float, int8_t>();
  TestQuantizeDequantizeRoundTrip<MLFloat16, uint8_t>();
  TestQuantizeDequantizeRoundTrip<MLFloat16, int8_t>();
}

// Outputs larger than maxStorageBufferBindingSize are bound as several buffer segments and written
// through a generated setter, which must pack byte lanes like the unsegmented path. A 2 KiB test limit
// splits the 3001-element float16 input into 3 segments and the byte output into 2.
template <typename QuantType>
void TestQuantizeLinearSegmentedOutput() {
  WebGpuExecutionProviderTestOptions test_options{};
  test_options.max_storage_buffer_binding_size = 2048;
  auto provider = WebGpuExecutionProviderWithTestOptions(test_options);
  if (!provider) {
    GTEST_SKIP() << "Artificially small storage binding limits require the native test factory.";
  }
  constexpr float kScale = 0.125f;
  const auto zero_point = static_cast<QuantType>(std::is_signed_v<QuantType> ? -3 : 100);
  const auto x = QuarterSteps(3001);
  std::vector<QuantType> expected(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    expected[i] = ReferenceQuantize(x[i], kScale, zero_point);
  }
  OpTester test("QuantizeLinear", 21);
  test.AddInput<MLFloat16>("x", {static_cast<int64_t>(x.size())}, FromFloats<MLFloat16>(x));
  test.AddInput<MLFloat16>("y_scale", {}, {MLFloat16(kScale)});
  test.AddInput<QuantType>("y_zero_point", {}, {zero_point});
  test.AddOutput<QuantType>("y", {static_cast<int64_t>(x.size())}, expected);
  test.Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(provider)).RunWithConfig();
}

TEST(WebGpuQuantizeLinearTest, SegmentedOutput) {
  TestQuantizeLinearSegmentedOutput<int8_t>();
  TestQuantizeLinearSegmentedOutput<uint8_t>();
}

// Packs 4-bit values two per byte, the first element in the low nibble.
template <typename T>
std::vector<T> Pack4Bit(const std::vector<int>& values) {
  std::vector<T> packed(T::CalcNumInt4Pairs(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    packed[i / 2].SetElem(i % 2, static_cast<typename T::UnpackedType>(values[i]));
  }
  return packed;
}

// Reference 4-bit quantization; the scales used with it are powers of two so x / scale is exact.
template <typename T>
int ReferenceQuantize4Bit(float x, float scale, int zero_point) {
  const float q = std::nearbyint(x / scale) + static_cast<float>(zero_point);
  return static_cast<int>(std::clamp(q, static_cast<float>(T::min_val), static_cast<float>(T::max_val)));
}

// The value a FloatType input holds for `value`.
template <typename FloatType>
float RoundTo(float value) {
  return static_cast<float>(FloatType(value));
}

// RunQuantizeLinear with unpacked 4-bit zero point and expected values.
template <typename FloatType, typename QuantType>
void RunQuantizeLinear4Bit(int opset,
                           const std::vector<int64_t>& x_shape, const std::vector<float>& x,
                           const std::vector<int64_t>& scale_shape, const std::vector<float>& scale,
                           const std::vector<int>& zero_point, const std::vector<int>& expected,
                           int64_t axis = 1, int64_t block_size = 0) {
  const auto packed_zero_point = Pack4Bit<QuantType>(zero_point);
  RunQuantizeLinear<FloatType, QuantType>(opset, x_shape, x, scale_shape, scale, &packed_zero_point,
                                          Pack4Bit<QuantType>(expected), axis, block_size);
}

// Chromium's WebNN backend (ONNX opset 21) reshapes a broadcastable scale and zero point to the scalar or
// 1-D shape that (De)QuantizeLinear takes. The CPU Reshape kernel has no int4/uint4 types, so constant
// folding leaves an int4/uint4 zero point Reshape in the graph and, with CPU fallback disabled, it must run
// on WebGPU. Inputs are x, scale, zero point and the shape both are reshaped to.
class ReshapedQuantizationParamsTester final : public OpTester {
 public:
  explicit ReshapedQuantizationParamsTester(const char* op_type) : OpTester(op_type, 21), op_type_(op_type) {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReshapedQuantizationParamsTester);

  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>& add_attribute_funcs) override {
    std::vector<NodeArg*> node_inputs{inputs[0]};
    for (size_t i = 1; i <= 2; ++i) {
      ONNX_NAMESPACE::TypeProto type;
      type.mutable_tensor_type()->set_elem_type(inputs[i]->TypeAsProto()->tensor_type().elem_type());
      auto& reshaped = graph.GetOrCreateNodeArg(inputs[i]->Name() + "_reshaped", &type);
      graph.AddNode(inputs[i]->Name() + "_reshape", "Reshape", "", {inputs[i], inputs[3]}, {&reshaped});
      node_inputs.push_back(&reshaped);
    }
    auto& node = graph.AddNode("node", op_type_, "", node_inputs, outputs);
    for (auto& add_attribute : add_attribute_funcs) {
      add_attribute(node);
    }
  }

 private:
  const char* op_type_;
};

// WebNN conformance (WPT quantizeLinear.https.any.js) int4/uint4 vectors, in the form Chromium's backend
// passes them to ONNX Runtime.
template <typename FloatType>
void TestQuantizeLinear4BitWebNNConformance() {
  constexpr float kScale = 1.1202747821807861f;
  constexpr float kScale2 = 2.1202747821807861f;
  const std::vector<float> x{4.794857501983643f, 3.23434354545f};
  // "tensor with int4 zeroPoint which has odd size": 0-D, per-tensor.
  RunQuantizeLinear4Bit<FloatType, Int4x2>(21, {}, {4.794857501983643f}, {}, {kScale}, {-4}, {0});
  // "tensor with int4 zeroPoint which has even size" and the uint4 version: per-axis.
  RunQuantizeLinear4Bit<FloatType, Int4x2>(21, {2}, x, {2}, {kScale, kScale}, {-6, -5}, {-2, -2}, /*axis*/ 0);
  RunQuantizeLinear4Bit<FloatType, UInt4x2>(21, {2}, x, {2}, {kScale, kScale}, {1, 5}, {5, 8}, /*axis*/ 0);
  // "1D tensor with uint4 zeroPoint with block_size = 3".
  RunQuantizeLinear4Bit<FloatType, UInt4x2>(
      21, {6},
      {4.794857501983643f, 3.23434354545f, 1.794857501983643f, 2.23434354545f, 4.794857501983643f, 3.23434354545f},
      {2}, {kScale, kScale}, {1, 5}, {5, 4, 3, 7, 9, 8}, /*axis*/ 0, /*block_size*/ 3);
  // "2D tensor with int4 zeroPoint with block_size = [3, 2]": Chromium expands the [1, 2] scale and zero point
  // to the input shape (Cast/Resize/Cast, constant folded on CPU) and uses axis = 0, block_size = 1.
  RunQuantizeLinear4Bit<FloatType, Int4x2>(
      21, {3, 4},
      {4.794857501983643f, 3.23434354545f, 2.794857501983643f, 5.794857501983643f, 0.0f, 7.23434354545f,
       4.794857501983643f, 3.23434354545f, 2.794857501983643f, 5.794857501983643f, 0.0f, 7.23434354545f},
      {3, 4}, {kScale, kScale, kScale2, kScale2, kScale, kScale, kScale2, kScale2, kScale, kScale, kScale2, kScale2},
      {-6, -6, -5, -5, -6, -6, -5, -5, -6, -6, -5, -5}, {-2, -3, -4, -2, -6, 0, -3, -3, -4, -1, -5, -2},
      /*axis*/ 0, /*block_size*/ 1);

  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  {
    // "2D tensor with int4 zeroPoint which has even size": [1, 2] scale and zero point, per-axis on axis 1.
    ReshapedQuantizationParamsTester test("QuantizeLinear");
    test.AddAttribute<int64_t>("axis", 1);
    test.AddInput<FloatType>("x", {3, 2},
                             FromFloats<FloatType>({4.794857501983643f, 3.23434354545f, 2.794857501983643f,
                                                    5.794857501983643f, 0.0f, 7.23434354545f}));
    test.AddInput<FloatType>("y_scale", {1, 2}, FromFloats<FloatType>({kScale, kScale2}), true);
    test.AddInput<Int4x2>("y_zero_point", {1, 2}, Pack4Bit<Int4x2>({-6, -5}), true);
    test.AddInput<int64_t>("param_shape", {1}, {2}, true);
    test.AddOutput<Int4x2>("y", {3, 2}, Pack4Bit<Int4x2>({-2, -3, -4, -2, -6, -2}));
    test.Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(provider)).RunWithConfig();
  }
  {
    // "tensor with uint4 zeroPoint which has odd size": [1] scale and zero point, per-tensor.
    ReshapedQuantizationParamsTester test("QuantizeLinear");
    test.AddInput<FloatType>("x", {5},
                             FromFloats<FloatType>({4.794857501983643f, 2.794857501983643f, 1.794857501983643f, 0.0f,
                                                    3.794857501983643f}));
    test.AddInput<FloatType>("y_scale", {1}, FromFloats<FloatType>({kScale}), true);
    test.AddInput<UInt4x2>("y_zero_point", {1}, Pack4Bit<UInt4x2>({10}), true);
    test.AddInput<int64_t>("param_shape", {0}, {}, true);
    test.AddOutput<UInt4x2>("y", {5}, Pack4Bit<UInt4x2>({14, 12, 12, 10, 13}));
    test.Config(WebGpuOnlySessionOptions()).ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
  }
}

TEST(WebGpuQuantizeLinearTest, Int4WebNNConformance_Float) {
  TestQuantizeLinear4BitWebNNConformance<float>();
}

TEST(WebGpuQuantizeLinearTest, Int4WebNNConformance_Float16) {
  TestQuantizeLinear4BitWebNNConformance<MLFloat16>();
}

template <typename FloatType>
void TestQuantizeLinear4BitRoundingAndSaturation() {
  // Ties round to even and results saturate to [-8, 7]. 13 elements, so the second packed word is partial.
  RunQuantizeLinear4Bit<FloatType, Int4x2>(
      21, {13}, {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 6.5f, 7.5f, -8.5f, -7.5f, 100.0f, -100.0f, 7.25f},
      {}, {1.0f}, {0}, {0, 2, 2, 0, -2, -2, 6, 7, -8, -8, 7, -8, 7});
  RunQuantizeLinear4Bit<FloatType, Int4x2>(21, {3}, {5.0f, -12.0f, -0.25f}, {}, {1.0f}, {3}, {7, -8, 3});
  // [0, 15] for uint4.
  RunQuantizeLinear4Bit<FloatType, UInt4x2>(21, {5}, {-1.0f, 15.5f, 16.0f, 14.5f, 0.5f}, {}, {1.0f}, {0},
                                            {0, 15, 15, 14, 0});
  RunQuantizeLinear4Bit<FloatType, UInt4x2>(21, {5}, {-9.0f, -8.0f, 7.0f, 8.0f, 0.5f}, {}, {1.0f}, {8},
                                            {0, 0, 15, 15, 8});
}

TEST(WebGpuQuantizeLinearTest, Int4RoundingAndSaturation_Float) {
  TestQuantizeLinear4BitRoundingAndSaturation<float>();
}

TEST(WebGpuQuantizeLinearTest, Int4RoundingAndSaturation_Float16) {
  TestQuantizeLinear4BitRoundingAndSaturation<MLFloat16>();
}

// Zero points that span several packed words, and odd element counts.
template <typename FloatType, typename QuantType>
void TestQuantizeLinear4BitPerAxisAndBlocked() {
  constexpr bool kSigned = std::is_same_v<QuantType, Int4x2>;
  const auto zero_points = [&](size_t count) {
    std::vector<int> values(count);
    for (size_t i = 0; i < count; ++i) {
      values[i] = static_cast<int>(i % 12) + (kSigned ? -6 : 2);
    }
    return values;
  };
  // Per-axis along the middle axis of a [2, 11, 3] tensor.
  {
    const std::vector<int64_t> shape{2, 11, 3};
    const auto x = QuarterSteps(66);
    const std::vector<float> scale{0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
    const auto zero_point = zero_points(11);
    std::vector<int> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      const size_t c = (i / 3) % 11;
      expected[i] = ReferenceQuantize4Bit<QuantType>(x[i], scale[c], zero_point[c]);
    }
    RunQuantizeLinear4Bit<FloatType, QuantType>(21, shape, x, {11}, scale, zero_point, expected, /*axis*/ 1);
    RunQuantizeLinear4Bit<FloatType, QuantType>(21, shape, x, {11}, scale, zero_point, expected, /*axis*/ -2);
  }
  // Blocked along the last axis with a partial final block: [3, 10] input, block_size 4 -> [3, 3] scale.
  {
    const std::vector<int64_t> shape{3, 10};
    const auto x = QuarterSteps(30);
    const std::vector<float> scale{0.25f, 0.5f, 1.0f, 2.0f, 0.125f, 0.5f, 1.0f, 0.25f, 2.0f};
    const auto zero_point = zero_points(9);
    std::vector<int> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      const size_t s = (i / 10) * 3 + (i % 10) / 4;
      expected[i] = ReferenceQuantize4Bit<QuantType>(x[i], scale[s], zero_point[s]);
    }
    RunQuantizeLinear4Bit<FloatType, QuantType>(21, shape, x, {3, 3}, scale, zero_point, expected,
                                                /*axis*/ 1, /*block_size*/ 4);
  }
  // Blocked along the first axis: [5, 3] input, block_size 2 -> [3, 3] scale.
  {
    const std::vector<int64_t> shape{5, 3};
    const auto x = QuarterSteps(15);
    const std::vector<float> scale{0.25f, 0.5f, 1.0f, 2.0f, 0.125f, 0.5f, 1.0f, 0.25f, 2.0f};
    const auto zero_point = zero_points(9);
    std::vector<int> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      const size_t s = ((i / 3) / 2) * 3 + i % 3;
      expected[i] = ReferenceQuantize4Bit<QuantType>(x[i], scale[s], zero_point[s]);
    }
    RunQuantizeLinear4Bit<FloatType, QuantType>(21, shape, x, {3, 3}, scale, zero_point, expected,
                                                /*axis*/ 0, /*block_size*/ 2);
  }
}

TEST(WebGpuQuantizeLinearTest, Int4PerAxisAndBlocked) {
  TestQuantizeLinear4BitPerAxisAndBlocked<float, Int4x2>();
  TestQuantizeLinear4BitPerAxisAndBlocked<float, UInt4x2>();
  TestQuantizeLinear4BitPerAxisAndBlocked<MLFloat16, Int4x2>();
  TestQuantizeLinear4BitPerAxisAndBlocked<MLFloat16, UInt4x2>();
}

TEST(WebGpuQuantizeLinearTest, Int4OpsetVersions) {
  // Odd and large enough for multiple workgroups.
  const auto x = QuarterSteps(1017);
  std::vector<int> expected_int4(x.size());
  std::vector<int> expected_uint4(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    expected_int4[i] = ReferenceQuantize4Bit<Int4x2>(x[i], 1.0f, -3);
    expected_uint4[i] = ReferenceQuantize4Bit<UInt4x2>(x[i], 1.0f, 6);
  }
  for (int opset : {21, 23, 24, 25, 28}) {
    SCOPED_TRACE(opset);
    RunQuantizeLinear4Bit<float, Int4x2>(opset, {1017}, x, {}, {1.0f}, {-3}, expected_int4);
    RunQuantizeLinear4Bit<float, UInt4x2>(opset, {1017}, x, {}, {1.0f}, {6}, expected_uint4);
  }
  RunQuantizeLinear4Bit<MLFloat16, Int4x2>(21, {1017}, x, {}, {1.0f}, {-3}, expected_int4);
  RunQuantizeLinear4Bit<MLFloat16, UInt4x2>(21, {1017}, x, {}, {1.0f}, {6}, expected_uint4);
}

// Without a zero point, `output_dtype` selects the 4-bit output type.
TEST(WebGpuQuantizeLinearTest, Int4NoZeroPoint) {
  const std::vector<float> x{-9.0f, -0.5f, 1.5f, 20.0f, 3.0f};
  for (int opset : {21, 23}) {
    SCOPED_TRACE(opset);
    for (bool is_signed : {true, false}) {
      auto provider = DefaultWebGpuExecutionProvider();
      if (!provider) {
        GTEST_SKIP() << "WebGPU EP is not available";
      }
      OpTester test("QuantizeLinear", opset);
      test.AddAttribute<int64_t>("output_dtype", is_signed ? ONNX_NAMESPACE::TensorProto_DataType_INT4
                                                           : ONNX_NAMESPACE::TensorProto_DataType_UINT4);
      test.AddInput<float>("x", {5}, x);
      test.AddInput<float>("y_scale", {}, {1.0f});
      if (is_signed) {
        test.AddOutput<Int4x2>("y", {5}, Pack4Bit<Int4x2>({-8, 0, 2, 7, 3}));
      } else {
        test.AddOutput<UInt4x2>("y", {5}, Pack4Bit<UInt4x2>({0, 0, 2, 15, 3}));
      }
      test.Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(provider)).RunWithConfig();
    }
  }
}

template <typename FloatType, typename QuantType>
void TestQuantizeDequantize4BitRoundTrip() {
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  const std::vector<float> x{-2.549168109893799f, -4.794857501983643f, 8.413617134094238f, 6.108623504638672f,
                             0.0f, 100.0f, -100.0f};
  constexpr float kScale = 0.5f;
  constexpr int kZeroPoint = std::is_same_v<QuantType, Int4x2> ? 1 : 7;
  std::vector<float> expected;
  for (float v : x) {
    expected.push_back(static_cast<float>(ReferenceQuantize4Bit<QuantType>(v, kScale, kZeroPoint) - kZeroPoint) *
                       kScale);
  }

  QuantizeDequantizeTester<QuantType> test;
  test.template AddInput<FloatType>("x", {7}, FromFloats<FloatType>(x));
  test.template AddInput<FloatType>("scale", {}, {FloatType(kScale)});
  test.template AddInput<QuantType>("zero_point", {}, Pack4Bit<QuantType>({kZeroPoint}));
  test.template AddOutput<FloatType>("y", {7}, FromFloats<FloatType>(expected));
  SessionOptions options = WebGpuOnlySessionOptions();
  // Keep graph optimizers from folding away the Q -> DQ pair.
  options.graph_optimization_level = TransformerLevel::Default;
  test.Config(options).ConfigEp(std::move(provider)).RunWithConfig();
}

TEST(WebGpuQuantizeLinearTest, Int4QuantizeDequantizeRoundTrip) {
  TestQuantizeDequantize4BitRoundTrip<float, Int4x2>();
  TestQuantizeDequantize4BitRoundTrip<float, UInt4x2>();
  TestQuantizeDequantize4BitRoundTrip<MLFloat16, Int4x2>();
  TestQuantizeDequantize4BitRoundTrip<MLFloat16, UInt4x2>();
}

// Runs DequantizeLinear-21 on WebGPU with a 4-bit input and zero point (given unpacked). With
// `reshaped_param_shape`, the scale and zero point are reshaped to it first, as Chromium's WebNN backend does.
template <typename FloatType, typename QuantType>
void RunDequantizeLinear4Bit(const std::vector<int64_t>& x_shape, const std::vector<int>& x,
                             const std::vector<int64_t>& param_shape, const std::vector<float>& scale,
                             const std::vector<int>& zero_point, const std::vector<float>& expected,
                             const std::vector<int64_t>* reshaped_param_shape = nullptr,
                             std::optional<int64_t> axis = std::nullopt) {
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  std::unique_ptr<OpTester> test;
  if (reshaped_param_shape != nullptr) {
    test = std::make_unique<ReshapedQuantizationParamsTester>("DequantizeLinear");
  } else {
    test = std::make_unique<OpTester>("DequantizeLinear", 21);
  }
  if (axis.has_value()) {
    test->AddAttribute<int64_t>("axis", *axis);
  }
  test->AddInput<QuantType>("x", x_shape, Pack4Bit<QuantType>(x));
  test->AddInput<FloatType>("x_scale", param_shape, FromFloats<FloatType>(scale), true);
  test->AddInput<QuantType>("x_zero_point", param_shape, Pack4Bit<QuantType>(zero_point), true);
  if (reshaped_param_shape != nullptr) {
    test->AddInput<int64_t>("param_shape", {static_cast<int64_t>(reshaped_param_shape->size())},
                            *reshaped_param_shape, true);
  }
  test->AddOutput<FloatType>("y", x_shape, FromFloats<FloatType>(expected));
  test->Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(provider)).RunWithConfig();
}

// WebNN conformance (WPT dequantizeLinear.https.any.js) int4/uint4 vectors. Chromium reshapes the [1] scale
// and zero point to scalars and the [1, 1, 2, 1] ones to [2] (per-axis on axis 2).
template <typename FloatType>
void TestDequantizeLinear4BitWebNNConformance() {
  const float s = RoundTo<FloatType>(1.1202747821807861f);
  const float s0 = RoundTo<FloatType>(9.343092918395996f);
  const float s1 = RoundTo<FloatType>(4.617084980010986f);
  const std::vector<int64_t> scalar_shape{};
  const std::vector<int64_t> axis_shape{2};
  for (bool reshape : {false, true}) {
    SCOPED_TRACE(reshape);
    // "uint4 1D tensor of odd size": expected {8.962198257446289, 11.202747344970703, 13.443297386169434}.
    RunDequantizeLinear4Bit<FloatType, UInt4x2>({3}, {10, 12, 14}, reshape ? std::vector<int64_t>{1} : scalar_shape,
                                                {1.1202747821807861f}, {2}, {8 * s, 10 * s, 12 * s},
                                                reshape ? &scalar_shape : nullptr);
    // "int4 1D tensor of odd size": expected {2.2405495643615723, 11.202747344970703, 3.3608243465423584}.
    RunDequantizeLinear4Bit<FloatType, Int4x2>({3}, {-1, 7, 0}, reshape ? std::vector<int64_t>{1} : scalar_shape,
                                               {1.1202747821807861f}, {-3}, {2 * s, 10 * s, 3 * s},
                                               reshape ? &scalar_shape : nullptr);
    // "uint4 4D tensor with broadcasting 4D scale and uint4 4D zeroPoint": expected
    // {-18.686185836791992, -9.343092918395996, 32.31959533691406, 55.40502166748047}.
    RunDequantizeLinear4Bit<FloatType, UInt4x2>(
        {1, 1, 2, 2}, {0, 1, 10, 15}, reshape ? std::vector<int64_t>{1, 1, 2, 1} : axis_shape,
        {9.343092918395996f, 4.617084980010986f}, {2, 3}, {-2 * s0, -1 * s0, 7 * s1, 12 * s1},
        reshape ? &axis_shape : nullptr, /*axis*/ 2);
  }
}

TEST(WebGpuDequantizeLinearTest, Int4WebNNConformance_Float) {
  TestDequantizeLinear4BitWebNNConformance<float>();
}

TEST(WebGpuDequantizeLinearTest, Int4WebNNConformance_Float16) {
  TestDequantizeLinear4BitWebNNConformance<MLFloat16>();
}

// Odd element counts with per-axis zero points that span several packed words.
TEST(WebGpuDequantizeLinearTest, Int4PerAxisMultiWordZeroPoint) {
  const std::vector<float> scale{0.5f, 0.25f, 2.0f, 1.0f, 0.125f, 4.0f, 0.5f, 0.25f, 2.0f, 1.0f, 0.125f};
  const std::vector<int> int4_zero_point{-8, -5, -1, 0, 1, 3, 7, -2, 4, -6, 2};
  const std::vector<int> uint4_zero_point{0, 3, 7, 8, 9, 11, 15, 6, 12, 2, 10};
  std::vector<int> int4_x(33);
  std::vector<int> uint4_x(33);
  std::vector<float> int4_expected(33);
  std::vector<float> uint4_expected(33);
  for (size_t i = 0; i < int4_x.size(); ++i) {
    const size_t c = (i / 3) % 11;
    int4_x[i] = static_cast<int>(i % 16) - 8;
    uint4_x[i] = static_cast<int>((i * 7) % 16);
    int4_expected[i] = static_cast<float>(int4_x[i] - int4_zero_point[c]) * scale[c];
    uint4_expected[i] = static_cast<float>(uint4_x[i] - uint4_zero_point[c]) * scale[c];
  }
  RunDequantizeLinear4Bit<float, Int4x2>({11, 3}, int4_x, {11}, scale, int4_zero_point, int4_expected,
                                         nullptr, /*axis*/ 0);
  RunDequantizeLinear4Bit<float, UInt4x2>({11, 3}, uint4_x, {11}, scale, uint4_zero_point, uint4_expected,
                                          nullptr, /*axis*/ 0);
  RunDequantizeLinear4Bit<MLFloat16, Int4x2>({11, 3}, int4_x, {11}, scale, int4_zero_point, int4_expected,
                                             nullptr, /*axis*/ 0);
  RunDequantizeLinear4Bit<MLFloat16, UInt4x2>({11, 3}, uint4_x, {11}, scale, uint4_zero_point, uint4_expected,
                                              nullptr, /*axis*/ 0);
}

}  // namespace
}  // namespace onnxruntime::test
