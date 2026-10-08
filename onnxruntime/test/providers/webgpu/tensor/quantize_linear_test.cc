// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/graph/graph.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
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

  OpTester test("DequantizeLinear", 21);
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
  test.SetNumRunCalls(expect_failure ? 1 : 4);
  test.Config(options)
      .Config(expect_failure ? OpTester::ExpectResult::kExpectFailure : OpTester::ExpectResult::kExpectSuccess,
              error)
      .ConfigEp(std::move(provider))
      .RunWithConfig();
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

}  // namespace
}  // namespace onnxruntime::test
