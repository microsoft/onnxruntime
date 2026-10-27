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

namespace {
template <typename Base, typename Exponent>
void TestPowBroadcast(const std::vector<int64_t>& base_shape, const std::vector<Base>& base,
                      const std::vector<int64_t>& exponent_shape, const std::vector<Exponent>& exponent,
                      const std::vector<int64_t>& output_shape, const std::vector<Base>& expected) {
  for (bool exponent_is_initializer : {false, true}) {
    SCOPED_TRACE(exponent_is_initializer);
    auto webgpu_ep = DefaultWebGpuExecutionProvider();
    if (!webgpu_ep) {
      GTEST_SKIP() << "WebGPU execution provider is not enabled in this build.";
    }
    OpTester test("Pow", 15);
    test.AddInput<Base>("X", base_shape, base);
    test.AddInput<Exponent>("Y", exponent_shape, exponent, exponent_is_initializer);
    test.AddOutput<Base>("Z", output_shape, expected);
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
  }
}
}  // namespace

TEST(MathOpTest, Pow_BroadcastExponent_WebGpu) {
  // Issue #32842: a rank-4 singleton uses vectorized broadcasting rather than the scalar path.
  const std::vector<int64_t> shape{3, 2, 64, 64};
  std::vector<float> base(3 * 2 * 64 * 64);
  std::vector<float> expected(base.size());
  for (size_t i = 0; i < base.size(); ++i) {
    base[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
    expected[i] = base[i] * base[i] * base[i];
  }
  TestPowBroadcast<float, float>(shape, base, {1, 1, 1, 1}, {3.0f}, shape, expected);
}

TEST(MathOpTest, Pow_BroadcastBase_WebGpu) {
  TestPowBroadcast<float, float>({1, 1, 1, 1}, {2.0f}, {1, 1, 2, 4},
                                 {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 2.0f, 1.0f},
                                 {1, 1, 2, 4}, {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 8.0f, 4.0f, 2.0f});
}

TEST(MathOpTest, Pow_BroadcastExponent_MixedTypes_WebGpu) {
  TestPowBroadcast<float, int32_t>({2, 4}, {-2.0f, -1.0f, 0.0f, 2.0f, -3.0f, -2.0f, 1.0f, 3.0f},
                                   {2, 1}, {3, 2}, {2, 4}, {-8.0f, -1.0f, 0.0f, 8.0f, 9.0f, 4.0f, 1.0f, 9.0f});
}

TEST(MathOpTest, Pow_BroadcastBase_MixedTypes_WebGpu) {
  TestPowBroadcast<int32_t, float>({2, 1}, {-2, 3}, {2, 4}, {0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f},
                                   {2, 4}, {1, -2, 4, -8, 1, 3, 9, 27});
}

TEST(MathOpTest, Pow_BroadcastExponent_Float16_WebGpu) {
  TestPowBroadcast<MLFloat16, MLFloat16>({2, 4}, MakeMLFloat16({-2.0f, -1.0f, 0.0f, 2.0f, -3.0f, -2.0f, 1.0f, 3.0f}),
                                         {2, 1}, MakeMLFloat16({3.0f, 2.0f}), {2, 4},
                                         MakeMLFloat16({-8.0f, -1.0f, 0.0f, 8.0f, 9.0f, 4.0f, 1.0f, 9.0f}));
}

TEST(MathOpTest, Pow_BroadcastBase_Float16_WebGpu) {
  TestPowBroadcast<MLFloat16, MLFloat16>({2, 1}, MakeMLFloat16({-2.0f, 3.0f}),
                                         {2, 4}, MakeMLFloat16({0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f}),
                                         {2, 4}, MakeMLFloat16({1.0f, -2.0f, 4.0f, -8.0f, 1.0f, 3.0f, 9.0f, 27.0f}));
}

namespace {
// Runs a bool binary op (And/Or/Xor) on WebGPU only, with CPU fallback disabled so a missing
// kernel fails the test. Expected values are computed on the host with NumPy-style broadcasting.
void RunLogicalBinaryWebGpu(const char* op_type, bool (*fn)(bool, bool),
                            const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not enabled in this build.";
  }

  const size_t rank = std::max(a_shape.size(), b_shape.size());
  const auto padded = [rank](const std::vector<int64_t>& shape) {
    std::vector<int64_t> result(rank - shape.size(), 1);
    result.insert(result.end(), shape.begin(), shape.end());
    return result;
  };
  const std::vector<int64_t> a_dims = padded(a_shape);
  const std::vector<int64_t> b_dims = padded(b_shape);
  std::vector<int64_t> out_shape(rank);
  for (size_t i = 0; i < rank; ++i) {
    out_shape[i] = std::max(a_dims[i], b_dims[i]);
  }

  const auto size_of = [](const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto d : shape) size *= d;
    return static_cast<size_t>(size);
  };
  const size_t a_size = size_of(a_shape);
  const size_t b_size = size_of(b_shape);
  const size_t out_size = size_of(out_shape);

  auto a_values = std::make_unique<bool[]>(a_size);
  auto b_values = std::make_unique<bool[]>(b_size);
  auto expected = std::make_unique<bool[]>(out_size);
  for (size_t i = 0; i < a_size; ++i) a_values[i] = (i % 2) == 1;
  for (size_t i = 0; i < b_size; ++i) b_values[i] = (i % 3) != 0;

  for (size_t out_idx = 0; out_idx < out_size; ++out_idx) {
    size_t remaining = out_idx;
    size_t a_offset = 0, b_offset = 0, a_stride = 1, b_stride = 1;
    for (size_t d = rank; d-- > 0;) {
      const size_t coord = remaining % static_cast<size_t>(out_shape[d]);
      remaining /= static_cast<size_t>(out_shape[d]);
      if (a_dims[d] != 1) a_offset += coord * a_stride;
      if (b_dims[d] != 1) b_offset += coord * b_stride;
      a_stride *= static_cast<size_t>(a_dims[d]);
      b_stride *= static_cast<size_t>(b_dims[d]);
    }
    expected[out_idx] = fn(a_values[a_offset], b_values[b_offset]);
  }

  OpTester test(op_type, 7);
  test.AddInput<bool>("A", a_shape, a_values.get(), a_size);
  test.AddInput<bool>("B", b_shape, b_values.get(), b_size);
  test.AddOutput<bool>("C", out_shape, expected.get(), out_size);
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// Shape pairs covering every BinaryElementwiseProgram mode for Boolx4 operands: scalar operands,
// same-shape element-wise with a partial trailing vec4, non-vectorized broadcast (including the
// 4-D to 5-D broadcast WebNN conformance tests use), and both vectorized broadcast sub-paths.
const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> kLogicalBinaryShapes = {
    {{}, {}},
    {{}, {2, 2, 2, 3}},
    {{2, 3}, {}},
    {{1, 2, 3, 1, 3}, {1, 2, 3, 1, 3}},
    {{2, 1, 3, 1}, {1, 2, 2, 3, 5}},
    {{1, 2, 2, 3, 5}, {2, 1, 3, 1}},
    {{2, 3, 1}, {1, 1, 4}},
    {{3, 4}, {2, 3, 4}},
};
}  // namespace

TEST(MathOpTest, And_Broadcast_WebGpu) {
  for (const auto& [a_shape, b_shape] : kLogicalBinaryShapes) {
    SCOPED_TRACE(::testing::Message() << "A rank " << a_shape.size() << ", B rank " << b_shape.size());
    RunLogicalBinaryWebGpu("And", [](bool a, bool b) { return a && b; }, a_shape, b_shape);
  }
}

TEST(MathOpTest, Or_Broadcast_WebGpu) {
  for (const auto& [a_shape, b_shape] : kLogicalBinaryShapes) {
    SCOPED_TRACE(::testing::Message() << "A rank " << a_shape.size() << ", B rank " << b_shape.size());
    RunLogicalBinaryWebGpu("Or", [](bool a, bool b) { return a || b; }, a_shape, b_shape);
  }
}

TEST(MathOpTest, Xor_Broadcast_WebGpu) {
  for (const auto& [a_shape, b_shape] : kLogicalBinaryShapes) {
    SCOPED_TRACE(::testing::Message() << "A rank " << a_shape.size() << ", B rank " << b_shape.size());
    RunLogicalBinaryWebGpu("Xor", [](bool a, bool b) { return a != b; }, a_shape, b_shape);
  }
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
