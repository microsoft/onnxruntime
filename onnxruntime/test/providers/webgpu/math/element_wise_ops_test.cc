// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/graph/graph.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/providers/webgpu/math/binary_elementwise_broadcast_utils.h"
#endif

namespace onnxruntime {
namespace test {

namespace {
std::vector<MLFloat16> MakeMLFloat16(const std::initializer_list<float>& input) {
  std::vector<MLFloat16> output;
  std::transform(input.begin(), input.end(), std::back_inserter(output),
                 [](float fl) { return MLFloat16(fl); });
  return output;
}
}  // namespace

#if defined(USE_WEBGPU)
// Deviceless regression for issue #28969: the shared-trailing-dimension count must never exceed
// either operand's rank. When operands have unequal ranks, an exhausted operand's implicit size-1
// dimension is a broadcast, not a shared dimension, and previously over-extended the shared run,
// underflowing the downstream reshape math (size_t wrap to SIZE_MAX).
TEST(MathOpTest, WebGpu_CountSharedTrailingDimensions) {
  int64_t shared_product = -1;

  // The crashing corner: lhs=[1,1,6,6], rhs=[6,6]. Only the two trailing 6s are shared; once rhs
  // is exhausted, lhs's leading unit dims are broadcasts and must not be counted.
  const TensorShape lhs_4d({1, 1, 6, 6});
  const TensorShape rhs_2d({6, 6});
  size_t num_shared = onnxruntime::webgpu::CountSharedTrailingDimensions(
      lhs_4d, rhs_2d, /*output_rank=*/4, shared_product);
  EXPECT_EQ(num_shared, static_cast<size_t>(2));
  EXPECT_EQ(shared_product, static_cast<int64_t>(36));
  // The core invariant: the count never exceeds the smaller operand's rank (derived from the
  // shapes, not hard-coded), which is exactly what keeps the downstream reshape from underflowing.
  EXPECT_LE(num_shared, std::min(lhs_4d.NumDimensions(), rhs_2d.NumDimensions()));

  // Operand-order symmetry: shorter operand on the left ([6,6] vs [1,1,6,6]) exercises the
  // ns == lhs_rank branch and must yield the same count/product and respect the same invariant.
  num_shared = onnxruntime::webgpu::CountSharedTrailingDimensions(
      rhs_2d, lhs_4d, /*output_rank=*/4, shared_product);
  EXPECT_EQ(num_shared, static_cast<size_t>(2));
  EXPECT_EQ(shared_product, static_cast<int64_t>(36));
  EXPECT_LE(num_shared, std::min(rhs_2d.NumDimensions(), lhs_4d.NumDimensions()));

  // Equal ranks: counting stops at output_rank - 1, leaving at least one outer dimension.
  num_shared = onnxruntime::webgpu::CountSharedTrailingDimensions(
      TensorShape({2, 3, 4}), TensorShape({2, 3, 4}), /*output_rank=*/3, shared_product);
  EXPECT_EQ(num_shared, static_cast<size_t>(2));
  EXPECT_EQ(shared_product, static_cast<int64_t>(12));

  // A genuine mismatch stops the run immediately.
  num_shared = onnxruntime::webgpu::CountSharedTrailingDimensions(
      TensorShape({2, 3, 4}), TensorShape({1, 4}), /*output_rank=*/3, shared_product);
  EXPECT_EQ(num_shared, static_cast<size_t>(1));
  EXPECT_EQ(shared_product, static_cast<int64_t>(4));
}

// End-to-end regression for issue #28969 on the WebGPU EP. Pre-fix this crashed with an
// ORT_ENFORCE in TensorShape::SizeFromDimension: the trailing product 36 is divisible by 4
// (taking the vectorized shared-dim path) while the last dim 6 is not, and the unequal ranks plus
// leading unit dims caused num_shared_dimension to exceed rhs's rank, underflowing the reshape.
TEST(MathOpTest, Add_Broadcast_WebGpu_UnequalRank_LeadingUnitDims) {
  OpTester test("Add");
  const std::vector<int64_t> lhs_dims{1, 1, 6, 6};
  const std::vector<int64_t> rhs_dims{6, 6};
  std::vector<float> lhs_values(36);
  std::vector<float> rhs_values(36);
  std::vector<float> out_values(36);
  for (int i = 0; i < 36; ++i) {
    lhs_values[i] = static_cast<float>(i);
    rhs_values[i] = static_cast<float>(2 * i);
    out_values[i] = static_cast<float>(3 * i);
  }
  test.AddInput<float>("A", lhs_dims, lhs_values);
  test.AddInput<float>("B", rhs_dims, rhs_values);
  test.AddOutput<float>("C", lhs_dims, out_values);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Companion to the regression above: the leading dim is REAL (2), not a unit dim. It must survive
// the trailing reshape (the shared run is only the two trailing 6s), guarding against
// over-collapsing the outer dimension when it is not a broadcast. Trailing product 36 still hits
// the divisible-by-4 vectorized reshape path.
TEST(MathOpTest, Add_Broadcast_WebGpu_UnequalRank_LeadingNonUnitDim) {
  OpTester test("Add");
  const std::vector<int64_t> lhs_dims{2, 1, 6, 6};  // 72 elements
  const std::vector<int64_t> rhs_dims{6, 6};        // 36 elements, broadcast over the leading [2,1]
  std::vector<float> lhs_values(72);
  std::vector<float> rhs_values(36);
  std::vector<float> out_values(72);
  for (int i = 0; i < 36; ++i) {
    rhs_values[i] = static_cast<float>(2 * i);
  }
  for (int i = 0; i < 72; ++i) {
    lhs_values[i] = static_cast<float>(i);
    out_values[i] = lhs_values[i] + rhs_values[i % 36];
  }
  test.AddInput<float>("A", lhs_dims, lhs_values);
  test.AddInput<float>("B", rhs_dims, rhs_values);
  test.AddOutput<float>("C", lhs_dims, out_values);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Operand-order symmetry for the regression: the shorter operand is on the LHS ([6,6] + [1,1,6,6]),
// exercising the ns == lhs_rank branch of the reshape. Must produce correct results without
// underflowing, mirroring Add_Broadcast_WebGpu_UnequalRank_LeadingUnitDims.
TEST(MathOpTest, Add_Broadcast_WebGpu_UnequalRank_ShorterLhs) {
  OpTester test("Add");
  const std::vector<int64_t> lhs_dims{6, 6};        // 36 elements
  const std::vector<int64_t> rhs_dims{1, 1, 6, 6};  // 36 elements
  const std::vector<int64_t> out_dims{1, 1, 6, 6};
  std::vector<float> lhs_values(36);
  std::vector<float> rhs_values(36);
  std::vector<float> out_values(36);
  for (int i = 0; i < 36; ++i) {
    lhs_values[i] = static_cast<float>(i);
    rhs_values[i] = static_cast<float>(2 * i);
    out_values[i] = static_cast<float>(3 * i);
  }
  test.AddInput<float>("A", lhs_dims, lhs_values);
  test.AddInput<float>("B", rhs_dims, rhs_values);
  test.AddOutput<float>("C", out_dims, out_values);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// WebGPU EP currently handles a special case for supporting Pow op:
// A Pow followed by a Cast to int64 type.
TEST(MathOpTest, Pow_float_sqrt) {
  class PowCastTester : public OpTester {
   public:
    PowCastTester() : OpTester("Pow", 22) {}

    void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                  std::vector<std::function<void(Node&)>>& /*add_attribute_funcs*/) override {
      ONNX_NAMESPACE::TypeProto intermediate_type = *inputs[0]->TypeAsProto();
      auto& intermediate = graph.GetOrCreateNodeArg("pow_output", &intermediate_type);
      graph.AddNode("Pow", "Pow", "", inputs, {&intermediate});
      auto& cast = graph.AddNode("Cast", "Cast", "", {&intermediate}, outputs);
      cast.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_INT64});
    }
  } test;

  test.AddInput<float>("x", {1}, {576.0f});
  test.AddInput<float>("y", {1}, {0.5f});
  test.AddOutput<int64_t>("out", {1}, {24});
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
}
#endif

// The shared Max/Min NaN tests are pinned to CPU/CUDA. These WebGPU-specific cases lock in
// the ONNX opset-12+ requirement that Max/Min propagate NaN (a plain WGSL max/min builtin does
// not guarantee this). The broadcast case exercises the vec4 broadcast shader path; the scalar
// case exercises the element-wise scalar-operand path.
TEST(MathOpTest, Max_12_Float_Nan_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not enabled in this build.";
  }
  OpTester test("Max", 12);
  test.AddInput<float>("data_0", {3, 3},
                       {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        -0.5f, 0.0f, -2.0f,
                        0.5f, 0.0f, 2.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {0.0f, -1.0f, 1.0f});
  test.AddOutput<float>("max", {3, 3},
                        {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         -0.5f, 0.0f, -1.0f,
                         1.0f, 1.0f, 2.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Min_12_Float_Nan_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not enabled in this build.";
  }
  OpTester test("Min", 12);
  test.AddInput<float>("data_0", {3, 3},
                       {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        -0.5f, 0.0f, -2.0f,
                        0.5f, 0.0f, 2.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {0.0f, -1.0f, 1.0f});
  test.AddOutput<float>("min", {3, 3},
                        {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         -1.0f, -1.0f, -2.0f,
                         0.5f, 0.0f, 1.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// A scalar NaN operand must turn every output element into NaN (element-wise scalar path).
TEST(MathOpTest, Max_12_Float_with_scalar_Nan_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not enabled in this build.";
  }
  OpTester test("Max", 12);
  test.AddInput<float>("data_0", {2, 2},
                       {0.25f, -0.25f, -0.5f, 0.5f});
  test.AddInput<float>("data_1", {1}, {std::numeric_limits<float>::quiet_NaN()});
  test.AddOutput<float>("max", {2, 2},
                        {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN()});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Variadic (3-input) fold must also propagate a NaN that appears in a middle operand.
TEST(MathOpTest, Min_12_Float_Variadic_Nan_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not enabled in this build.";
  }
  OpTester test("Min", 12);
  test.AddInput<float>("data_0", {1, 3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_1", {1, 3},
                       {std::numeric_limits<float>::quiet_NaN(), 0.0f, 5.0f});
  test.AddInput<float>("data_2", {1, 3}, {-1.0f, -2.0f, 4.0f});
  test.AddOutput<float>("min", {1, 3},
                        {std::numeric_limits<float>::quiet_NaN(), -2.0f, 3.0f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Float16 exercises the distinct NaN-detection path in the shader: f16 is widened to f32 before the
// integer bitcast, since a vec4<f16> is too narrow for the vec4<u32> bitcast the check relies on.
TEST(MathOpTest, Max_12_Float16_Nan_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not enabled in this build.";
  }
  OpTester test("Max", 12);
  test.AddInput<MLFloat16>("data_0", {3, 1},
                           MakeMLFloat16({-1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f}));
  test.AddInput<MLFloat16>("data_1", {3, 1},
                           MakeMLFloat16({0.5f, 1.0f, std::numeric_limits<float>::quiet_NaN()}));
  test.AddOutput<MLFloat16>("max", {3, 1},
                            MakeMLFloat16({0.5f,
                                           std::numeric_limits<float>::quiet_NaN(),
                                           std::numeric_limits<float>::quiet_NaN()}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
