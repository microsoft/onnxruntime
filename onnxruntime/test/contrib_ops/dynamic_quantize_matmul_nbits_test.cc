// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstring>
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

// Applies the preprocess_weights_for_mixed_gemm SM80 stages: row permutation, column-major packing,
// column-interleaved tiles of 64 K x (16 / bits) columns, and the in-word pair interleave.
std::vector<uint8_t> PrepackSm80(const std::vector<int>& weights, int64_t n, int64_t k, int bits) {
  const int interleave = 16 / bits;
  const int permute_rows = 8 * interleave;
  const int elements_per_word = 32 / bits;
  const int words_per_tile = 64 / elements_per_word;
  const int64_t words_per_column = k / elements_per_word;
  std::vector<uint32_t> words(n * words_per_column, 0);
  for (int64_t column = 0; column < n; ++column) {
    for (int64_t row = 0; row < k; ++row) {
      // Destination row `r` of each permutation block holds source row 2g + 8j + h.
      const int64_t block_start = row / permute_rows * permute_rows;
      int64_t permuted = -1;
      for (int r = 0; r < permute_rows; ++r) {
        const int g = r / (2 * interleave);
        const int j = (r % (2 * interleave)) / 2;
        if (2 * g + 8 * j + r % 2 == row - block_start) permuted = block_start + r;
      }
      ORT_ENFORCE(permuted >= 0);
      const int64_t word = permuted / elements_per_word;
      const int element = static_cast<int>(permuted % elements_per_word);
      const int slot = element % 2 == 0 ? element / 2 : (element - 1) / 2 + elements_per_word / 2;
      const int64_t write_word = (column / interleave) * words_per_column * interleave +
                                 interleave * (word / words_per_tile * words_per_tile) +
                                 words_per_tile * (column % interleave) + word % words_per_tile;
      const uint32_t code = static_cast<uint32_t>(weights[column * k + row] + (1 << (bits - 1)));
      words[write_word] |= code << (slot * bits);
    }
  }
  std::vector<uint8_t> packed(words.size() * 4);
  std::memcpy(packed.data(), words.data(), packed.size());
  return packed;
}

template <typename T, typename S = float>
void RunCase(const std::vector<int64_t>& shape, int64_t n, int bits, int64_t block,
             const std::string& mode, int64_t activation_block = 0,
             bool constant_scales = false, bool flattened_scales = false, bool with_bias = true,
             std::vector<float> input_values = {}, std::vector<float> scale_values = {}, bool identity = false,
             bool constant_weights = false, int64_t weight_prepacked = 0) {
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
      // A hashed pattern lets every code position distinguish layout permutations.
      const uint32_t mixed = static_cast<uint32_t>((column * 0x9E3779B1u) ^ (pos * 0x85EBCA6Bu));
      const int code = identity ? (column == pos ? 1 : 0)
                       : weight_prepacked != 0
                           ? int(((mixed ^ (mixed >> 15)) * 0x2C1B3C6Du >> 12) % (1u << bits)) - (1 << (bits - 1))
                           : int((column * 7 + pos * 3) % (1 << bits)) - (1 << (bits - 1));
      weights[column * k + pos] = code;
      const int shift = int(pos % (8 / bits)) * bits;
      auto& byte = b[(column * groups + pos / block) * blob + (pos % block) / (8 / bits)];
      byte = static_cast<uint8_t>((byte & ~(((1 << bits) - 1) << shift)) | ((code + (1 << (bits - 1))) << shift));
    }
  }
  if (weight_prepacked == 1) {
    ORT_ENFORCE(block == (bits == 2 ? 64 : 32) && k % 128 == 0 && n % 64 == 0);
    b = PrepackSm80(weights, n, k, bits);
    ORT_ENFORCE(b.size() == static_cast<size_t>(n * groups * blob));
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
  test.AddAttribute<int64_t>("weight_prepacked", weight_prepacked);
  test.AddAttribute<std::string>("activation_quantization", mode);
  if (mode == "dynamic_per_block") test.AddAttribute<int64_t>("activation_block_size", activation_block);
  test.AddInput<T>("A", shape, a);
  test.AddInput<uint8_t>("B", {n, groups, blob}, b, constant_weights);
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

TEST(DynamicQuantizeMatMulNBitsTest, RegisterResidentActivationQuantizer) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  constexpr int64_t m = 5;
  constexpr int64_t n = 3;
  // Domain sizes cover each cooperative group width, the 8192-element limit, and the scalar fallback.
  for (const int64_t k : {8, 136, 256, 520, 1024, 1032, 2048, 4104, 8192, 8200}) {
    for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token"}) {
      SCOPED_TRACE(std::to_string(k) + " " + mode);
      RunCase<MLFloat16>({m, k}, n, 8, 16, mode, 0, true, false, true);
      std::vector<float> values(m * k);
      for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(static_cast<int>(i * 29 % 113) - 56) / 7.0f;
      values[k + 5] = std::numeric_limits<float>::infinity();
      values[3 * k + k - 1] = std::numeric_limits<float>::quiet_NaN();
      RunCase<BFloat16>({m, k}, n, 8, 16, mode, 0, true, false, false, values);
    }
  }
  for (const auto& [k, block] : std::vector<std::pair<int64_t, int64_t>>{{2048, 200}, {1040, 48}, {520, 64}, {1032, 1032}, {130, 24}}) {
    SCOPED_TRACE(std::to_string(k) + " " + std::to_string(block));
    RunCase<MLFloat16>({m, k}, n, 8, 16, "dynamic_per_block", block, true, false, true);
    RunCase<BFloat16>({m, k}, n, 8, 16, "dynamic_per_block", block, true, false, false);
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

TEST(DynamicQuantizeMatMulNBitsTest, Common128ScaleIntervals) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (int bits : {2, 4, 8}) {
    for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token"}) {
      RunCase<MLFloat16>({64, 256}, 64, bits, 128, mode, 0, true, false, true, {}, {}, false, true);
      RunCase<BFloat16, MLFloat16>({1024, 256}, 1024, bits, 128, mode, 0, true, true, false, {}, {}, false, true);
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, BlackwellPortableKernelDispatch) {
  if (!HasCudaEnvironment(1200)) GTEST_SKIP();
  for (int bits : {2, 4, 8}) {
    RunCase<MLFloat16>({8, 256}, 64, bits, 128, "dynamic_per_token", 0, true, false, true);
    RunCase<BFloat16>({8, 256}, 64, bits, 128, "dynamic_per_tensor", 0, true, false, false);
    RunCase<MLFloat16>({128, 128}, 4096, bits, 64, "dynamic_per_token", 0, true, false, false);
  }
  RunCase<MLFloat16>({16, 128}, 512, 8, 128, "dynamic_per_token", 0, true, false, false);
}

TEST(DynamicQuantizeMatMulNBitsTest, Sm80PrepackedInt4Dispatch) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  RunCase<MLFloat16>({8, 128}, 64, 4, 32, "dynamic_per_token", 0,
                     true, false, true, {}, {}, false, true, 1);
  RunCase<BFloat16, BFloat16>({9, 128}, 64, 4, 32, "dynamic_per_token", 0,
                              true, false, false, {}, {}, false, true, 1);
  RunCase<BFloat16, MLFloat16>({33, 128}, 64, 4, 32, "dynamic_per_tensor", 0,
                               true, false, true, {}, {}, false, true, 1);
  RunCase<MLFloat16>({17, 384}, 128, 4, 32, "dynamic_per_token", 0,
                     true, false, true, {}, {}, false, true, 1);
}

TEST(DynamicQuantizeMatMulNBitsTest, Sm80PrepackedLayoutMatchesMatMulNBits) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  constexpr int64_t n = 128, k = 256;
  for (const auto& [bits, block] : {std::pair{2, 64}, std::pair{4, 32}, std::pair{8, 32}}) {
    SCOPED_TRACE("bits=" + std::to_string(bits));
    std::vector<int> weights(n * k);
    std::vector<MLFloat16> a(k * k, MLFloat16(0.0f));
    std::vector<MLFloat16> expected(k * n);
    for (int64_t column = 0; column < n; ++column) {
      for (int64_t pos = 0; pos < k; ++pos) {
        const uint32_t mixed = static_cast<uint32_t>((column * 0x9E3779B1u) ^ (pos * 0x85EBCA6Bu));
        weights[column * k + pos] = int(((mixed ^ (mixed >> 15)) * 0x2C1B3C6Du >> 12) % (1u << bits)) - (1 << (bits - 1));
        expected[pos * n + column] = MLFloat16(static_cast<float>(weights[column * k + pos]));
      }
    }
    for (int64_t i = 0; i < k; ++i) a[i * k + i] = MLFloat16(1.0f);
    const std::vector<uint8_t> b = PrepackSm80(weights, n, k, bits);
    OpTester test("MatMulNBits", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", k);
    test.AddAttribute<int64_t>("N", n);
    test.AddAttribute<int64_t>("block_size", block);
    test.AddAttribute<int64_t>("bits", bits);
    test.AddAttribute<int64_t>("accuracy_level", 0);
    test.AddAttribute<int64_t>("weight_prepacked", 1);
    test.AddInput<MLFloat16>("A", {k, k}, a);
    test.AddInput<uint8_t>("B", {n, k / block, block * bits / 8}, b, true);
    test.AddInput<MLFloat16>("scales", {n, k / block}, std::vector<MLFloat16>(n * k / block, MLFloat16(1.0f)), true);
    test.AddOptionalInputEdge<uint8_t>();
    test.AddOptionalInputEdge<int32_t>();
    test.AddOptionalInputEdge<MLFloat16>();
    test.AddOutput<MLFloat16>("Y", {k, n}, expected);
    test.SetOutputAbsErr("Y", 0.0f);
    test.SetOutputRelErr("Y", 0.0f);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, Sm80PrepackedInt2Int8Dispatch) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (const auto& [bits, block] : {std::pair{2, 64}, std::pair{8, 32}}) {
    RunCase<MLFloat16>({8, 128}, 64, bits, block, "dynamic_per_token", 0,
                       true, false, true, {}, {}, false, true, 1);
    RunCase<BFloat16, BFloat16>({9, 128}, 64, bits, block, "dynamic_per_token", 0,
                                true, false, false, {}, {}, false, true, 1);
    RunCase<BFloat16, MLFloat16>({33, 128}, 64, bits, block, "dynamic_per_tensor", 0,
                                 true, false, true, {}, {}, false, true, 1);
    RunCase<MLFloat16>({17, 384}, 128, bits, block, "dynamic_per_token", 0,
                       true, false, true, {}, {}, false, true, 1);
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, ConstantWeightsAndFallbacks) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (int bits : {2, 4}) {
    for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"}) {
      for (int64_t k : {192, 256, 320, 257}) {
        RunCase<MLFloat16>({65, k}, 128, bits, 128, mode, 192, true, false, true, {}, {}, false, true);
        RunCase<BFloat16, BFloat16>({65, k}, 128, bits, 128, mode, 192, true, true, false, {}, {}, false, true);
      }
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, Common128NonfiniteAndScaleBounds) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  constexpr int64_t m = 64;
  constexpr int64_t k = 256;
  constexpr int64_t n = 64;
  std::vector<float> scales(n * 2);
  const float values[] = {0, -1, 1, 1e-38f, std::numeric_limits<float>::max()};
  for (size_t i = 0; i < scales.size(); ++i) scales[i] = values[i % 5];
  for (int bits : {2, 4, 8}) {
    for (float value : {std::numeric_limits<float>::quiet_NaN(), float(BFloat16::FromBits(1)),
                        float(BFloat16::FromBits(0x7f7f))}) {
      RunCase<BFloat16>({m, k}, n, bits, 128, "dynamic_per_token", 0, true, false, false,
                        std::vector<float>(m * k, value), scales);
    }
    for (float value : {float(MLFloat16::FromBits(1)), 65504.0f}) {
      RunCase<MLFloat16>({m, k}, n, bits, 128, "dynamic_per_token", 0, true, false, false,
                         std::vector<float>(m * k, value), scales);
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, HopperSingleIntervalWeights) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"}) {
    for (int64_t k : {144, 272}) {
      RunCase<MLFloat16>({17, k}, 136, 8, 512, mode, 512, true, false, true, {}, {}, false, true);
      RunCase<MLFloat16, MLFloat16>({129, k}, 512, 8, 512, mode, 512, true, true, false, {}, {}, false, true);
      RunCase<MLFloat16, BFloat16>({65, k}, 136, 8, 512, mode, 512, false, false, true, {}, {}, false, true);
    }
    RunCase<BFloat16>({65, 256}, 136, 8, 512, mode, 512, true, false, false, {}, {}, false, true);
    RunCase<MLFloat16>({15, 256}, 136, 8, 512, mode, 128, true, false, false, {}, {}, false, true);
    RunCase<MLFloat16>({65, 257}, 136, 8, 512, mode, 192, true, false, false, {}, {}, false, true);
  }
  RunCase<MLFloat16, MLFloat16>({129, 256}, 4096, 8, 512, "dynamic_per_token", 0,
                                true, false, true, {}, {}, false, true);
}

TEST(DynamicQuantizeMatMulNBitsTest, HopperExceptionalDomains) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  constexpr int64_t m = 129;
  constexpr int64_t k = 144;
  constexpr int64_t n = 512;
  std::vector<float> scales(n);
  const float scale_values[] = {0, -1, 1, std::numeric_limits<float>::denorm_min(),
                                std::numeric_limits<float>::max()};
  for (size_t i = 0; i < scales.size(); ++i) scales[i] = scale_values[i % 5];
  for (const std::string mode : {"dynamic_per_token", "dynamic_per_tensor"}) {
    for (float value : {float(MLFloat16::FromBits(1)), 65504.0f, std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
      RunCase<MLFloat16>({m, k}, n, 8, 512, mode, 0, true, false, false,
                         std::vector<float>(m * k, value), scales, false, true);
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, HopperGroupedScaleIntervals) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  for (int bits : {2, 4, 8}) {
    for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"}) {
      SCOPED_TRACE(std::to_string(bits) + " " + mode);
      // Covers each tile configuration, ragged rows, row tiles, and K from two to eight scale blocks.
      RunCase<MLFloat16>({17, 256}, 264, bits, 128, mode, 256, true, false, true, {}, {}, false, true);
      RunCase<MLFloat16, MLFloat16>({129, 384}, 264, bits, 128, mode, 384, true, true, false, {}, {}, false, true);
      RunCase<BFloat16, BFloat16>({64, 1024}, 256, bits, 128, mode, 1024, true, false, true, {}, {}, false, true);
      RunCase<BFloat16, MLFloat16>({300, 256}, 1032, bits, 128, mode, 4096, true, false, false, {}, {}, false, true);
      RunCase<MLFloat16>({161, 640}, 1032, bits, 128, mode, 640, false, false, true, {}, {}, false, true);
    }
  }
  // Decode-sized row counts use the same kernel; TMA zero-fills the missing rows.
  for (int64_t m : {1, 3, 15}) {
    RunCase<MLFloat16>({m, 256}, 264, 4, 128, "dynamic_per_token", 0, true, false, true, {}, {}, false, true);
    RunCase<BFloat16, MLFloat16>({m, 384}, 520, 2, 128, "dynamic_per_tensor", 0, true, false, false, {}, {}, false,
                                 true);
  }
  // Large enough for the 128x96 and 128x128 tiles and several K blocks.
  RunCase<MLFloat16>({1536, 512}, 1024, 4, 128, "dynamic_per_token", 0, true, false, true, {}, {}, false, true);
  RunCase<BFloat16, MLFloat16>({256, 256}, 2056, 2, 128, "dynamic_per_tensor", 0, true, false, false, {}, {}, false,
                               true);
  // More output tiles than SMs, so persistent CTAs walk several tiles with odd and even K-block counts.
  RunCase<MLFloat16>({1030, 128 * 3}, 2056, 4, 128, "dynamic_per_token", 0, true, false, true, {}, {}, false, true);
  RunCase<BFloat16, BFloat16>({1030, 256}, 2056, 2, 128, "dynamic_per_tensor", 0, true, false, false, {}, {}, false,
                              true);
}

TEST(DynamicQuantizeMatMulNBitsTest, HopperGroupedWeightBlocksAndActivationIntervals) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  // Weight blocks from 32 to 4096 (including a block wider than K), K tails that are multiples of 16 but not of the
  // block, and activation intervals that are, are not, or cannot be (1000, 24) multiples of the K tile.
  for (int bits : {2, 4, 8}) {
    for (int64_t wblock : {32, 64, 256, 512}) {
      if (bits == 2 && wblock > 256) continue;
      SCOPED_TRACE(std::to_string(bits) + " wblock " + std::to_string(wblock));
      RunCase<MLFloat16>({33, 1040}, 264, bits, wblock, "dynamic_per_token", 0, true, false, true, {}, {}, false,
                         true);
      RunCase<BFloat16, MLFloat16>({17, 1040}, 264, bits, wblock, "dynamic_per_tensor", 0, true, false, false, {}, {},
                                   false, true);
      for (int64_t ablock : {32, 96, 192, 256, 1000, 24}) {
        SCOPED_TRACE("ablock " + std::to_string(ablock));
        RunCase<MLFloat16>({33, 1040}, 264, bits, wblock, "dynamic_per_block", ablock, true, false, true, {}, {},
                           false, true);
      }
    }
    SCOPED_TRACE(bits);
    RunCase<BFloat16, BFloat16>({129, 400}, 520, bits, 128, "dynamic_per_block", 128, true, true, false, {}, {}, false,
                                true);
    RunCase<MLFloat16, BFloat16>({65, 384}, 264, bits, 128, "dynamic_per_block", 256, false, false, true, {}, {}, false,
                                 true);
  }
  // Blocks wider than K and a single weight block spanning all of K (W4 per channel).
  RunCase<MLFloat16>({33, 272}, 264, 4, 4096, "dynamic_per_token", 0, true, false, true, {}, {}, false, true);
  RunCase<BFloat16>({129, 1040}, 264, 8, 128, "dynamic_per_block", 96, true, false, true, {}, {}, false, true);
  // Large tiles with several activation intervals.
  RunCase<MLFloat16>({1030, 512}, 1032, 4, 128, "dynamic_per_block", 256, true, false, false, {}, {}, false, true);
  RunCase<BFloat16>({1030, 512}, 520, 4, 256, "dynamic_per_block", 64, true, false, true, {}, {}, false, true);
}

TEST(DynamicQuantizeMatMulNBitsTest, HopperGroupedExceptionalDomains) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  constexpr int64_t m = 64;
  constexpr int64_t k = 256;
  constexpr int64_t n = 256;
  std::vector<float> scales(n * 2);
  const float values[] = {0, -1, 1, 1e-38f, std::numeric_limits<float>::max()};
  for (size_t i = 0; i < scales.size(); ++i) scales[i] = values[i % 5];
  std::vector<float> mixed(m * k);
  const float row_values[] = {float(BFloat16::FromBits(1)), 1e-30f, 1.0f, 3e5f, 1e30f,
                              float(BFloat16::FromBits(0x7f7f)), std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity()};
  for (int64_t i = 0; i < m * k; ++i) mixed[i] = row_values[(i / k) % 8] * (static_cast<float>(i % 7) - 3.0f);
  for (int bits : {2, 4, 8}) {
    SCOPED_TRACE(bits);
    for (float value : row_values) {
      RunCase<BFloat16>({m, k}, n, bits, 128, "dynamic_per_token", 0, true, false, false,
                        std::vector<float>(m * k, value), scales, false, true);
    }
    RunCase<BFloat16>({m, k}, n, bits, 128, "dynamic_per_token", 0, true, false, true, mixed, scales, false, true);
    RunCase<BFloat16>({m, k}, n, bits, 128, "dynamic_per_tensor", 0, true, false, true, mixed, scales, false, true);
    for (float value : {float(MLFloat16::FromBits(1)), 65504.0f, std::numeric_limits<float>::quiet_NaN()}) {
      RunCase<MLFloat16>({m, k}, n, bits, 128, "dynamic_per_token", 0, true, false, false,
                         std::vector<float>(m * k, value), scales, false, true);
    }
  }
}

TEST(DynamicQuantizeMatMulNBitsTest, HopperBFloat16Domains) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  constexpr int64_t m = 129;
  constexpr int64_t k = 144;
  constexpr int64_t n = 512;
  std::vector<float> scales(n);
  const float scale_values[] = {0, -1, 1, std::numeric_limits<float>::denorm_min(),
                                std::numeric_limits<float>::max(), 1e-30f, 3e-3f};
  for (size_t i = 0; i < scales.size(); ++i) scales[i] = scale_values[i % 7];
  // Rows span tiny, ordinary, and near-overflow magnitudes so activation scales cover normal and non-normal products.
  const float row_values[] = {float(BFloat16::FromBits(1)), 1e-30f, 1.0f, 3e5f, 1e30f,
                              float(BFloat16::FromBits(0x7f7f)), std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity()};
  std::vector<float> mixed(m * k);
  for (int64_t i = 0; i < m * k; ++i) {
    mixed[i] = row_values[(i / k) % 8] * (static_cast<float>(i % 7) - 3.0f);
  }
  for (const std::string mode : {"dynamic_per_token", "dynamic_per_tensor"}) {
    SCOPED_TRACE(mode);
    for (float value : row_values) {
      RunCase<BFloat16>({m, k}, n, 8, 512, mode, 0, true, false, false,
                        std::vector<float>(m * k, value), scales, false, true);
    }
    RunCase<BFloat16>({m, k}, n, 8, 512, mode, 0, true, false, true, mixed, scales, false, true);
    RunCase<BFloat16, MLFloat16>({m, k}, n, 8, 512, mode, 0, true, false, true, mixed, {}, false, true);
    RunCase<BFloat16, BFloat16>({m, k}, n, 8, 512, mode, 0, true, true, false, {}, {}, false, true);
  }
  RunCase<BFloat16, MLFloat16>({1536, 1024}, 2048, 8, 1024, "dynamic_per_token", 0, true, false, true,
                               {}, {}, false, true);
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

TEST(DynamicQuantizeMatMulNBitsTest, Fp16ScaleProductBounds) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP();
  constexpr int64_t m = 33;
  constexpr int64_t k = 192;
  constexpr int64_t n = 37;
  for (int bits : {2, 4, 8}) {
    const int64_t block = bits == 8 ? 256 : 128;
    const int64_t groups = (k + block - 1) / block;
    std::vector<float> scales(n * groups);
    const float scale_values[] = {0, 1, -1, std::numeric_limits<float>::denorm_min(),
                                  std::numeric_limits<float>::max()};
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = scale_values[i % 5];
    for (const std::string mode : {"dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"}) {
      for (float value : {float(MLFloat16::FromBits(1)), 65504.0f, std::numeric_limits<float>::quiet_NaN()}) {
        SCOPED_TRACE(std::to_string(bits) + " " + mode + " " + std::to_string(value));
        RunCase<MLFloat16>({m, k}, n, bits, block, mode, 128, true, false, false,
                           std::vector<float>(m * k, value), scales);
      }
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
