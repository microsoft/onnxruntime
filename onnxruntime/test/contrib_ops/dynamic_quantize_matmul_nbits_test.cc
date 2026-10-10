// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "onnx/shape_inference/implementation.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime::test {
namespace {
ONNX_NAMESPACE::ModelProto ShapeModel(int64_t k = 33) {
  using namespace ONNX_NAMESPACE;
  ModelProto model;
  model.set_ir_version(8);
  auto* opset = model.add_opset_import();
  opset->set_domain(kMSDomain);
  opset->set_version(1);
  auto* graph = model.mutable_graph();
  graph->set_name("dynamic_quantize_matmul_nbits");
  auto* node = graph->add_node();
  node->set_op_type("DynamicQuantizeMatMulNBits");
  node->set_domain(kMSDomain);
  const auto input = [&](const char* name, int type, std::initializer_list<int64_t> shape) {
    node->add_input(name);
    auto* value = graph->add_input();
    value->set_name(name);
    auto* tensor = value->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(type);
    for (int64_t dim : shape) tensor->mutable_shape()->add_dim()->set_dim_value(dim);
  };
  input("A", TensorProto_DataType_FLOAT16, {2, k});
  input("B", TensorProto_DataType_UINT8, {5, k / 128 + (k % 128 != 0), 128});
  input("scales", TensorProto_DataType_FLOAT, {5, k / 128 + (k % 128 != 0)});
  node->add_output("Y");
  graph->add_output()->set_name("Y");
  return model;
}

void IntAttribute(ONNX_NAMESPACE::ModelProto& model, const char* name, int64_t value) {
  auto* attr = model.mutable_graph()->mutable_node(0)->add_attribute();
  attr->set_name(name);
  attr->set_type(ONNX_NAMESPACE::AttributeProto::INT);
  attr->set_i(value);
}

void Infer(ONNX_NAMESPACE::ModelProto& model) {
  ONNX_NAMESPACE::shape_inference::InferShapes(model, ONNX_NAMESPACE::OpSchemaRegistry::Instance(),
                                               ONNX_NAMESPACE::ShapeInferenceOptions(true, 1, true));
}
}  // namespace

TEST(DynamicQuantizeMatMulNBitsShapeTest, SymbolicAndRankOne) {
  auto model = ShapeModel();
  auto* a = model.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape();
  a->mutable_dim(0)->clear_dim_value();
  a->mutable_dim(0)->set_dim_param("tokens");
  Infer(model);
  const auto& output = model.graph().output(0).type().tensor_type();
  EXPECT_EQ(output.elem_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  EXPECT_EQ(output.shape().dim(0).dim_param(), "tokens");
  EXPECT_EQ(output.shape().dim(1).dim_value(), 5);

  model = ShapeModel();
  a = model.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape();
  a->clear_dim();
  a->add_dim()->set_dim_value(33);
  Infer(model);
  EXPECT_EQ(model.graph().output(0).type().tensor_type().shape().dim_size(), 1);
  EXPECT_EQ(model.graph().output(0).type().tensor_type().shape().dim(0).dim_value(), 5);
}

TEST(DynamicQuantizeMatMulNBitsShapeTest, RejectInvalidAttributesAndShapes) {
  for (auto [name, value] : {std::pair{"bits", 3}, {"block_size", 15}, {"activation_block_size", 16}}) {
    auto model = ShapeModel();
    IntAttribute(model, name, value);
    EXPECT_THROW(Infer(model), ONNX_NAMESPACE::InferenceError);
  }
  auto model = ShapeModel();
  IntAttribute(model, "bits", 2);
  IntAttribute(model, "block_size", 512);
  EXPECT_THROW(Infer(model), ONNX_NAMESPACE::InferenceError);
  model = ShapeModel();
  model.mutable_graph()->mutable_input(1)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(2)->set_dim_value(127);
  EXPECT_THROW(Infer(model), ONNX_NAMESPACE::InferenceError);
  model = ShapeModel();
  model.mutable_graph()->mutable_input(2)->mutable_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  EXPECT_THROW(Infer(model), ONNX_NAMESPACE::InferenceError);
}

TEST(DynamicQuantizeMatMulNBitsShapeTest, AccumulatorBound) {
  auto model = ShapeModel(132104);
  IntAttribute(model, "block_size", 262144);
  auto* shape = model.mutable_graph()->mutable_input(1)->mutable_type()->mutable_tensor_type()->mutable_shape();
  shape->mutable_dim(1)->set_dim_value(1);
  shape->mutable_dim(2)->set_dim_value(262144);
  model.mutable_graph()->mutable_input(2)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1)->set_dim_value(1);
  EXPECT_NO_THROW(Infer(model));
  model.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1)->set_dim_value(132105);
  EXPECT_THROW(Infer(model), ONNX_NAMESPACE::InferenceError);
}

TEST(DynamicQuantizeMatMulNBitsShapeTest, ValidateScalesWithUnknownK) {
  auto model = ShapeModel();
  auto* k = model.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(1);
  k->clear_dim_value();
  k->set_dim_param("K");
  auto* scales = model.mutable_graph()->mutable_input(2)->mutable_type()->mutable_tensor_type()->mutable_shape();
  scales->clear_dim();
  scales->add_dim()->set_dim_value(5);
  EXPECT_NO_THROW(Infer(model));
  scales->mutable_dim(0)->set_dim_value(6);
  EXPECT_THROW(Infer(model), ONNX_NAMESPACE::InferenceError);
}

#if defined(USE_CUDA)
namespace {
float RoundedMultiply(float a, float b) {
  volatile float result = a * b;
  return result;
}
float RoundedAdd(float a, float b) {
  volatile float result = a + b;
  return result;
}
float ReferenceScale(const std::vector<float>& values) {
  float maximum = 0;
  for (float v : values) {
    if (!std::isfinite(v)) return std::numeric_limits<float>::quiet_NaN();
    maximum = std::max(maximum, std::abs(v));
  }
  return maximum == 0 ? 1.0f : std::max(maximum / 127.0f, std::numeric_limits<float>::denorm_min());
}
float ReferenceProduct(int64_t acc, float a, float b) {
  if (std::isnan(a)) return a;
  const float value = static_cast<float>(acc);
  const float first = RoundedMultiply(value, a);
  return acc == 0 || std::isnormal(first) ? RoundedMultiply(first, b)
                                          : static_cast<float>((static_cast<double>(value) * a) * b);
}

template <typename T, typename S = float>
void RunCase(const std::vector<int64_t>& shape, int64_t n, int bits, int64_t block,
             const std::string& mode, int64_t activation_block = 0,
             bool constant_scales = false, bool flattened_scales = false, bool with_bias = true,
             std::vector<float> input_values = {}, std::vector<float> scale_values = {}, bool identity = false) {
  const int64_t k = shape.back();
  const int64_t m = std::accumulate(shape.begin(), shape.end() - 1, int64_t{1}, std::multiplies<int64_t>());
  const int64_t groups = k / block + (k % block != 0);
  const int64_t blob = block / (8 / bits);
  std::vector<T> a(m * k);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = T(input_values.empty() ? static_cast<float>(int(i * 17 % 61) - 30) / 8.0f : input_values.at(i));
  }
  std::vector<S> scales(n * groups);
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] = S(scale_values.empty() ? static_cast<float>(int(i % 7) - 3) / 16.0f : scale_values.at(i));
  }
  std::vector<int> weights(n * k);
  std::vector<uint8_t> b(n * groups * blob, 0xff);  // Nonzero padding must be ignored.
  for (int64_t column = 0; column < n; ++column) {
    for (int64_t pos = 0; pos < k; ++pos) {
      const int code = identity ? (column == pos ? 1 : 0) : int((column * 7 + pos * 3) % (1 << bits)) - (1 << (bits - 1));
      weights[column * k + pos] = code;
      const int shift = int(pos % (8 / bits)) * bits;
      auto& byte = b[(column * groups + pos / block) * blob + (pos % block) / (8 / bits)];
      byte = static_cast<uint8_t>((byte & ~(((1 << bits) - 1) << shift)) | ((code + (1 << (bits - 1))) << shift));
    }
  }
  std::vector<T> bias(n);
  for (int64_t column = 0; column < n; ++column) bias[column] = T(float(column) / 4.0f);
  std::vector<int> quantized(m * k);
  std::vector<float> a_scales(m * k);
  float tensor_scale = 1;
  if (mode == "dynamic_per_tensor") {
    std::vector<float> values;
    values.reserve(a.size());
    for (T v : a) values.push_back(float(v));
    tensor_scale = ReferenceScale(values);
  }
  if (k && m) {
    const int64_t domain_size = mode == "dynamic_per_tensor"  ? m * k
                                : mode == "dynamic_per_token" ? k
                                                              : activation_block;
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t start = 0; start < k;) {
        const int64_t count = std::min(k - start, domain_size);
        std::vector<float> values;
        if (mode != "dynamic_per_tensor") {
          for (int64_t i = 0; i < count; ++i) values.push_back(float(a[row * k + start + i]));
        }
        const float scale = mode == "dynamic_per_tensor" ? tensor_scale : ReferenceScale(values);
        for (int64_t i = 0; i < count; ++i) {
          const int64_t index = row * k + start + i;
          a_scales[index] = scale;
          quantized[index] = std::isnan(scale) ? 0 : std::clamp(int(std::nearbyint(float(a[index]) / scale)), -127, 127);
        }
        start += count;
      }
    }
  }
  std::vector<T> expected(m * n);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t column = 0; column < n; ++column) {
      float total = 0;
      for (int64_t start = 0; start < k;) {
        int64_t end = std::min(k, start + (block - start % block));
        if (mode == "dynamic_per_block") end = std::min(end, start + (activation_block - start % activation_block));
        int64_t acc = 0;
        for (int64_t pos = start; pos < end; ++pos) acc += int64_t{quantized[row * k + pos]} * weights[column * k + pos];
        const float term = ReferenceProduct(acc, a_scales[row * k + start], float(scales[column * groups + start / block]));
        total = start == 0 ? term : RoundedAdd(total, term);
        start = end;
      }
      expected[row * n + column] = T(RoundedAdd(total, with_bias ? float(bias[column]) : 0.0f));
    }
  }
  auto output_shape = shape;
  output_shape.back() = n;
  OpTester test("DynamicQuantizeMatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("bits", bits);
  test.AddAttribute<int64_t>("block_size", block);
  test.AddAttribute<std::string>("activation_quantization", mode);
  if (mode == "dynamic_per_block") test.AddAttribute<int64_t>("activation_block_size", activation_block);
  test.AddInput<T>("A", shape, a);
  test.AddInput<uint8_t>("B", {n, groups, blob}, b);
  test.AddInput<S>("scales", flattened_scales ? std::vector<int64_t>{n * groups} : std::vector<int64_t>{n, groups}, scales, constant_scales);
  if (with_bias) test.AddInput<T>("bias", {n}, bias);
  test.AddOutput<T>("Y", output_shape, expected);
  test.SetCustomOutputVerifier([expected](const std::vector<OrtValue>& outputs, const std::string&) {
    const auto& tensor = outputs.at(0).Get<Tensor>();
    ASSERT_EQ(tensor.Shape().Size(), static_cast<int64_t>(expected.size()));
    const T* actual = tensor.Data<T>();
    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::isnan(float(expected[i])))
        EXPECT_TRUE(std::isnan(float(actual[i]))) << i;
      else
        EXPECT_EQ(actual[i].val, expected[i].val) << "output " << i;
    }
  });
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
}  // namespace

TEST(DynamicQuantizeMatMulNBitsTest, WidthsModesDtypesAndTails) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (int bits : {2, 4, 8}) {
    for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"}) {
      SCOPED_TRACE(std::to_string(bits) + " " + mode);
      RunCase<MLFloat16>({2, 3, 65}, 7, bits, 32, mode, 19);
      RunCase<BFloat16, MLFloat16>({3, 17}, 5, bits, 16, mode, 3, true, true, false);
      RunCase<MLFloat16, BFloat16>({65}, 3, bits, 128, mode, 96, true);
      for (int64_t k : {1, 2, 3, 4}) RunCase<MLFloat16>({2, k}, 4, bits, 16, mode, 1);
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, ExactQuantizationTiesAndZero) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  RunCase<MLFloat16>({16}, 16, 8, 16, "dynamic_per_token", 0, false, false, false,
                     {127, -127, 0.5f, 1.5f, 2.5f, 3.5f, -0.5f, -1.5f, -2.5f, -3.5f, 0, 0, 0, 0, 0, 0},
                     std::vector<float>(16, 1.0f), true);
  RunCase<MLFloat16>({2, 17}, 3, 4, 16, "dynamic_per_block", 3, false, false, true, std::vector<float>(34, 0));
}

TEST(DynamicQuantizeMatMulNBitsTest, LargeBlocksAndFullCodeRange) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (int bits : {2, 4, 8}) {
    for (int block : {16, 32, 64, 128, 256}) {
      RunCase<MLFloat16>({2, 257}, 5, bits, block, "dynamic_per_block", 96);
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, TensorCoreTilesAndCommonIntervals) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (int bits : {2, 4, 8}) {
    for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"}) {
      SCOPED_TRACE(std::to_string(bits) + " " + mode);
      RunCase<MLFloat16>({33, 64}, 37, bits, 128, mode, 32, true);
      RunCase<BFloat16, MLFloat16>({129, 257}, 73, bits, 128, mode, 96, true, true);
      RunCase<MLFloat16, BFloat16>({33, 65}, 37, bits, 32, mode, 19, true, false, false);
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, TensorCoreNonfiniteAndGradualUnderflow) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  constexpr int64_t m = 33;
  constexpr int64_t k = 64;
  constexpr int64_t n = 37;
  for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"}) {
    std::vector<float> values(m * k, 1.0f);
    values[3] = std::numeric_limits<float>::quiet_NaN();
    RunCase<BFloat16>({m, k}, n, 8, 128, mode, 32, true, false, false, values, std::vector<float>(n, 0.0f));
    RunCase<BFloat16>({m, k}, n, 8, 128, mode, 32, true, false, false,
                      std::vector<float>(m * k, float(BFloat16::FromBits(0x7f7f))), std::vector<float>(n, 1e-38f));
    RunCase<BFloat16>({m, k}, n, 8, 128, mode, 32, true, false, false,
                      std::vector<float>(m * k, float(BFloat16::FromBits(1))), std::vector<float>(n, 1.0f));
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, SingleIntervalVectorizedTails) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (int64_t k : {32, 48, 80}) {
    for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"}) {
      SCOPED_TRACE(std::to_string(k) + " " + mode);
      RunCase<MLFloat16>({17, k}, 137, 8, 128, mode, 128, true, false, false);
      RunCase<BFloat16, MLFloat16>({65, k}, 137, 8, 128, mode, 128, true, true);
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, PipelinedTilesAndScaleIntervals) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"}) {
    for (int64_t k : {144, 272, 528}) {
      // Exercise both pipeline buffers, zero-filled K tails and partial M/N tiles.
      RunCase<MLFloat16>({129, k}, 137, 8, 1024, mode, 1024, true);
      RunCase<BFloat16, MLFloat16>({129, k}, 137, 8, 1024, mode, 1024, true, true, false);
    }
    for (int bits : {2, 4, 8}) {
      // Unequal aligned boundaries must reset INT32 and add FP32 terms in K order.
      RunCase<MLFloat16, BFloat16>({65, 320}, 137, bits, 128, mode, 192, true);
      RunCase<BFloat16>({33, 256}, 73, bits, 64, mode, 128, true, false, false);
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, PipelinedLargeGroupedTile) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (int bits : {2, 4, 8}) {
    RunCase<MLFloat16>({961, 192}, 961, bits, 128, "dynamic_per_block", 64, true, false, false);
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, PipelinedNonfiniteAndExtremeScales) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  constexpr int64_t m = 129;
  constexpr int64_t k = 192;
  constexpr int64_t n = 137;
  for (int bits : {2, 4, 8}) {
    const int64_t block = bits == 8 ? 256 : 128;
    const int64_t groups = (k + block - 1) / block;
    for (float value : {std::numeric_limits<float>::quiet_NaN(), float(BFloat16::FromBits(1)),
                        float(BFloat16::FromBits(0x7f7f))}) {
      RunCase<BFloat16>({m, k}, n, bits, block, "dynamic_per_token", 0, true, false, false,
                        std::vector<float>(m * k, value), std::vector<float>(n * groups, 1e-38f));
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, NonfiniteDomainsAndExtremeScales) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"}) {
    for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
      std::vector<float> a(34, 1.0f);
      a[3] = value;
      RunCase<MLFloat16>({2, 17}, 3, 8, 32, mode, 3, false, false, false, a, {0, -1, 1});
    }
  }
  const float maximum = float(BFloat16::FromBits(0x7f7f));
  RunCase<BFloat16>({4}, 4, 8, 16, "dynamic_per_token", 0, false, false, false,
                    std::vector<float>(4, maximum), std::vector<float>(4, 1e-38f));
  RunCase<BFloat16>({4}, 4, 8, 16, "dynamic_per_token", 0, false, false, false,
                    std::vector<float>(4, float(BFloat16::FromBits(1))), std::vector<float>(4, 1.0f));
  RunCase<BFloat16>({4}, 4, 8, 16, "dynamic_per_token", 0, false, false, false,
                    std::vector<float>(4, 127), std::vector<float>(4, 1e-40f));
  RunCase<BFloat16>({65}, 4, 8, 32, "dynamic_per_token", 0, false, false, false,
                    std::vector<float>(65, maximum), std::vector<float>(12, 1.0f));
}

TEST(DynamicQuantizeMatMulNBitsTest, EmptyDimensions) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  RunCase<MLFloat16>({2, 0}, 3, 4, 16, "dynamic_per_token");
  RunCase<BFloat16>({0}, 3, 2, 16, "dynamic_per_block", 7, true, true, false);
  RunCase<MLFloat16>({0, 17}, 3, 8, 32, "dynamic_per_tensor");
  RunCase<MLFloat16>({3, 17}, 0, 8, 32, "dynamic_per_token");
}

TEST(DynamicQuantizeMatMulNBitsTest, PerTensorScaleSpansScratchTiles) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  // Slightly exceed the 16 MiB activation scratch budget. The last row changes
  // quantization in the first tile: 0.5 rounds to zero only with the global scale.
  constexpr int64_t m = 4097;
  constexpr int64_t k = 4096;
  std::vector<float> values(m * k, 0.5f);
  values.back() = 127.0f;
  RunCase<MLFloat16>({m, k}, 1, 8, 128, "dynamic_per_tensor", 0, true, false, false, std::move(values));
}

TEST(DynamicQuantizeMatMulNBitsTest, RejectNonfiniteScales) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (bool constant : {false, true}) {
    for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
      OpTester test("DynamicQuantizeMatMulNBits", 1, kMSDomain);
      test.AddInput<MLFloat16>("A", {1}, {MLFloat16(1.0f)});
      test.AddInput<uint8_t>("B", {1, 1, 128}, std::vector<uint8_t>(128, 129));
      test.AddInput<float>("scales", {1}, {value}, constant);
      test.AddOutput<MLFloat16>("Y", {1}, {MLFloat16(1.0f)});
      std::vector<std::unique_ptr<IExecutionProvider>> providers;
      providers.push_back(DefaultCudaExecutionProvider());
      test.Run(OpTester::ExpectResult::kExpectFailure, "scales must contain only finite values", {}, nullptr, &providers);
    }
  }
}
#endif

}  // namespace onnxruntime::test
