// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
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
  for (int opset : {10, 13, 19, 21, 23, 24}) {
    SCOPED_TRACE(opset);
    RunQuantizeLinear<float, int8_t>(opset, {1001}, x, {}, {0.125f}, &int8_zero_point, expected_int8);
    RunQuantizeLinear<float, uint8_t>(opset, {1001}, x, {}, {0.125f}, &uint8_zero_point, expected_uint8);
    RunQuantizeLinear<float, uint8_t>(opset, {1001}, x, {}, {0.125f}, nullptr, expected_no_zero_point);
  }
}

// From opset 23, x / y_scale is computed in the `precision` type, which defaults to the y_scale type.
// The WebGPU kernel divides in f32, so a float16 y_scale or an explicit float16 precision must be left
// to the CPU EP. CPU fallback stays enabled; an f32 division on WebGPU would give 5 and 3 below.
TEST(WebGpuQuantizeLinearTest, Opset23PrecisionOtherThanFloat) {
  {
    auto provider = DefaultWebGpuExecutionProvider();
    if (!provider) {
      GTEST_SKIP() << "WebGPU EP is not available";
    }
    // 1.5 / 0.333251953125 is 4.5011 in f32 but 4.5 in float16, which rounds half to even to 4.
    OpTester test("QuantizeLinear", 23);
    test.AddInput<MLFloat16>("x", {1}, {MLFloat16(1.5f)});
    test.AddInput<MLFloat16>("y_scale", {}, {MLFloat16(0.333251953125f)});
    test.AddInput<int8_t>("y_zero_point", {}, {0});
    test.AddOutput<int8_t>("y", {1}, {4});
    test.ConfigEp(std::move(provider)).RunWithConfig();
  }
  {
    auto provider = DefaultWebGpuExecutionProvider();
    // 2.5001 is 2.5 in float16, which rounds half to even to 2.
    OpTester test("QuantizeLinear", 23);
    test.AddAttribute<int64_t>("precision", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    test.AddInput<float>("x", {1}, {2.5001f});
    test.AddInput<float>("y_scale", {}, {1.0f});
    test.AddInput<int8_t>("y_zero_point", {}, {0});
    test.AddOutput<int8_t>("y", {1}, {2});
    test.ConfigEp(std::move(provider)).RunWithConfig();
  }
}

// An explicit f32 precision matches the kernel's division and stays on WebGPU.
TEST(WebGpuQuantizeLinearTest, Opset23ExplicitFloatPrecision) {
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  OpTester test("QuantizeLinear", 23);
  test.AddAttribute<int64_t>("precision", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  test.AddInput<float>("x", {3}, {2.5001f, -2.5f, 7.75f});
  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddOutput<int8_t>("y", {3}, {3, -2, 8});
  test.Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(provider)).RunWithConfig();
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
  test.AddInput<FloatType>("x", {7}, FromFloats<FloatType>(x));
  test.AddInput<FloatType>("scale", {}, {FloatType(kScale)});
  test.AddInput<QuantType>("zero_point", {}, {kZeroPoint});
  test.AddOutput<FloatType>("y", {7}, FromFloats<FloatType>(expected));
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

}  // namespace
}  // namespace onnxruntime::test
