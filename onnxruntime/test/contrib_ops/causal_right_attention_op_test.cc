// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/model.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

using ONNX_NAMESPACE::TensorProto;
using Shapes = std::array<std::vector<int64_t>, 3>;

Status ResolveCausalRightAttention(
    const Shapes& dimensions,
    const std::array<int, 3>& types = {TensorProto::FLOAT16, TensorProto::FLOAT16, TensorProto::FLOAT16},
    std::optional<float> scale = 0.0625f, bool symbolic_batch = false, bool omit_shapes = false,
    const std::function<void(const NodeArg&)>& verify = nullptr) {
  const std::unordered_map<std::string, int> versions{{kOnnxDomain, 18}, {kMSDomain, 1}};
  Model model("causal_right_attention_shape", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), versions, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<NodeArg*> inputs;
  for (size_t i = 0; i < dimensions.size(); ++i) {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor = type.mutable_tensor_type();
    tensor->set_elem_type(types[i]);
    if (!omit_shapes) {
      auto* shape = tensor->mutable_shape();
      for (size_t axis = 0; axis < dimensions[i].size(); ++axis) {
        auto* dim = shape->add_dim();
        if (symbolic_batch && axis == 0)
          dim->set_dim_param("batch");
        else
          dim->set_dim_value(dimensions[i][axis]);
      }
    }
    inputs.push_back(&graph.GetOrCreateNodeArg("input_" + std::to_string(i), &type));
  }
  auto& output = graph.GetOrCreateNodeArg("Y", nullptr);
  auto& node = graph.AddNode("attention", "CausalRightAttention", "", inputs, {&output}, nullptr, kMSDomain);
  if (scale) node.AddAttribute("scale", *scale);
  graph.SetOutputs({&output});
  auto status = graph.Resolve();
  if (status.IsOK() && verify) verify(output);
  return status;
}

TEST(CausalRightAttentionShapeTest, TypesShapeAndSymbolicDimensions) {
  const Shapes shapes{{{2, 3, 7, 256}, {2, 3, 30, 256}, {2, 3, 30, 256}}};
  for (int type : {TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    ASSERT_STATUS_OK(ResolveCausalRightAttention(shapes, {type, type, type}, .0625f, false, false,
                                                 [&](const NodeArg& output) {
                                                   EXPECT_EQ(output.TypeAsProto()->tensor_type().elem_type(), type);
                                                   ASSERT_NE(output.Shape(), nullptr);
                                                   ASSERT_EQ(output.Shape()->dim_size(), 4);
                                                   EXPECT_EQ(output.Shape()->dim(0).dim_value(), 2);
                                                   EXPECT_EQ(output.Shape()->dim(1).dim_value(), 3);
                                                   EXPECT_EQ(output.Shape()->dim(2).dim_value(), 7);
                                                   EXPECT_EQ(output.Shape()->dim(3).dim_value(), 256);
                                                 }));
    ASSERT_STATUS_OK(ResolveCausalRightAttention(shapes, {type, type, type}, .0625f, true, false,
                                                 [](const NodeArg& output) {
                                                   ASSERT_NE(output.Shape(), nullptr);
                                                   EXPECT_EQ(output.Shape()->dim(0).dim_param(), "batch");
                                                 }));
    ASSERT_STATUS_OK(ResolveCausalRightAttention(shapes, {type, type, type}, .0625f, false, true));
  }
}

TEST(CausalRightAttentionShapeTest, RejectsInvalidKnownDimensions) {
  const Shapes valid{{{1, 2, 8, 256}, {1, 2, 30, 256}, {1, 2, 30, 256}}};
  for (size_t input = 0; input < 3; ++input) {
    for (size_t axis = 0; axis < 4; ++axis) {
      for (int64_t invalid : {int64_t{0}, int64_t{257}}) {
        auto shapes = valid;
        shapes[input][axis] = invalid;
        EXPECT_FALSE(ResolveCausalRightAttention(shapes).IsOK());
      }
    }
    auto rank = valid;
    rank[input].pop_back();
    EXPECT_FALSE(ResolveCausalRightAttention(rank).IsOK());
    auto head_size = valid;
    head_size[input][3] = 128;
    EXPECT_FALSE(ResolveCausalRightAttention(head_size).IsOK());
  }
  auto shapes = valid;
  shapes[0][0] = 9;
  EXPECT_FALSE(ResolveCausalRightAttention(shapes).IsOK());
  shapes = valid;
  shapes[0][1] = 33;
  EXPECT_FALSE(ResolveCausalRightAttention(shapes).IsOK());
  shapes = valid;
  shapes[1][1] = 1;
  EXPECT_FALSE(ResolveCausalRightAttention(shapes).IsOK());
  shapes = valid;
  shapes[2][2] = 29;
  EXPECT_FALSE(ResolveCausalRightAttention(shapes).IsOK());
  shapes = valid;
  shapes[0][2] = 31;
  EXPECT_FALSE(ResolveCausalRightAttention(shapes).IsOK());
}

TEST(CausalRightAttentionShapeTest, RequiresScaleAndNativeMatchingTypes) {
  const Shapes shapes{{{1, 2, 1, 256}, {1, 2, 30, 256}, {1, 2, 30, 256}}};
  for (float scale : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_FALSE(ResolveCausalRightAttention(shapes, {TensorProto::FLOAT16, TensorProto::FLOAT16, TensorProto::FLOAT16},
                                             scale)
                     .IsOK());
  }
  EXPECT_FALSE(ResolveCausalRightAttention(shapes, {TensorProto::FLOAT16, TensorProto::FLOAT16, TensorProto::FLOAT16},
                                           std::nullopt)
                   .IsOK());
  EXPECT_FALSE(ResolveCausalRightAttention(shapes, {TensorProto::FLOAT, TensorProto::FLOAT, TensorProto::FLOAT}).IsOK());
  EXPECT_FALSE(ResolveCausalRightAttention(shapes, {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT16}).IsOK());
}

TEST(CausalRightAttentionTest, NoCpuFallback) {
  OpTester test("CausalRightAttention", 1, kMSDomain);
  test.AddAttribute("scale", .0625f);
  const std::vector<MLFloat16> values(256);
  for (const char* name : {"Q", "K", "V"}) test.AddInput<MLFloat16>(name, {1, 1, 1, 256}, values);
  test.AddOutput<MLFloat16>("Y", {1, 1, 1, 256}, values);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "Could not find an implementation", {}, nullptr, &providers);
}

template <typename T>
std::vector<T> RandomNative(size_t size, std::mt19937& rng) {
  std::uniform_real_distribution<float> distribution(-0.3f, 0.3f);
  std::vector<T> values;
  values.reserve(size);
  for (size_t i = 0; i < size; ++i) values.emplace_back(distribution(rng));
  return values;
}

template <typename T>
std::vector<T> Reference(const std::vector<T>& q, const std::vector<T>& k, const std::vector<T>& v,
                         int batch, int heads, int queries, int keys, float scale) {
  constexpr int d = 256;
  std::vector<T> output(q.size());
  std::vector<double> probabilities(keys);
  for (int bh = 0; bh < batch * heads; ++bh) {
    for (int row = 0; row < queries; ++row) {
      const int count = keys - queries + row + 1;
      double max_score = -std::numeric_limits<double>::infinity();
      for (int col = 0; col < count; ++col) {
        double score = 0;
        for (int x = 0; x < d; ++x) {
          score += static_cast<double>(q[(bh * queries + row) * d + x].ToFloat()) *
                   k[(bh * keys + col) * d + x].ToFloat();
        }
        probabilities[col] = score * scale;
        max_score = std::max(max_score, probabilities[col]);
      }
      double sum = 0;
      for (int col = 0; col < count; ++col) {
        probabilities[col] = std::exp(probabilities[col] - max_score);
        sum += probabilities[col];
      }
      for (int x = 0; x < d; ++x) {
        double value = 0;
        for (int col = 0; col < count; ++col) {
          value += probabilities[col] / sum * v[(bh * keys + col) * d + x].ToFloat();
        }
        output[(bh * queries + row) * d + x] = T(static_cast<float>(value));
      }
    }
  }
  return output;
}

#ifdef USE_CUDA
std::unique_ptr<IExecutionProvider> CausalRightCudaProvider() {
  auto ep = DefaultCudaExecutionProvider();
  if (ep == nullptr) return nullptr;
  auto registry = ep->GetKernelRegistry();
  if (registry == nullptr) return nullptr;
  const KernelCreateInfo* info = nullptr;
  KernelRegistry::TypeConstraintMap constraints;
  if (!registry->TryFindKernel(ep->Type(), "CausalRightAttention", kMSDomain, 1, constraints,
                               DefaultLoggingManager().DefaultLogger(), &info)
           .IsOK()) {
    return nullptr;
  }
  return ep;
}

template <typename T>
void RunCorrectness() {
  auto ep = CausalRightCudaProvider();
  if (!ep) GTEST_SKIP() << "CUDA memory-efficient attention is not compiled";
  if (GetCudaArchitecture() < 900 || GetCudaArchitecture() >= 1000) {
    GTEST_SKIP() << "CausalRightAttention currently supports compute capability 9.x only";
  }
  std::mt19937 rng(20261028);
  const std::array<std::array<int, 4>, 10> cases{{{1, 1, 1, 1}, {2, 3, 7, 30}, {1, 16, 30, 30}, {2, 2, 32, 35}, {1, 2, 33, 49}, {1, 2, 127, 129}, {1, 2, 129, 255}, {1, 1, 256, 256}, {8, 1, 2, 3}, {1, 32, 1, 256}}};
  auto run = [&](int batch, int heads, int queries, int keys) {
    SCOPED_TRACE(std::to_string(batch) + ":" + std::to_string(heads) + ":" +
                 std::to_string(queries) + ":" + std::to_string(keys));
    const auto q = RandomNative<T>(size_t(batch) * heads * queries * 256, rng);
    const auto k = RandomNative<T>(size_t(batch) * heads * keys * 256, rng);
    const auto v = RandomNative<T>(k.size(), rng);
    const auto expected = Reference(q, k, v, batch, heads, queries, keys, .0625f);
    OpTester test("CausalRightAttention", 1, kMSDomain);
    test.AddAttribute("scale", .0625f);
    test.AddInput<T>("Q", {batch, heads, queries, 256}, q);
    test.AddInput<T>("K", {batch, heads, keys, 256}, k);
    test.AddInput<T>("V", {batch, heads, keys, 256}, v);
    test.AddOutput<T>("Y", {batch, heads, queries, 256}, expected);
    constexpr bool bf16 = std::is_same_v<T, BFloat16>;
    test.SetOutputAbsErr("Y", bf16 ? .01f : .005f);
    test.SetOutputRelErr("Y", bf16 ? .05f : .005f);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  for (const auto& shape : cases) run(shape[0], shape[1], shape[2], shape[3]);
  for (int keys = 30; keys <= 49; ++keys) run(1, 16, 1, keys);
}

TEST(CausalRightAttentionTest, Float16SeededBounds) {
  RunCorrectness<MLFloat16>();
}

TEST(CausalRightAttentionTest, BFloat16SeededBounds) {
  RunCorrectness<BFloat16>();
}

TEST(CausalRightAttentionTest, SingleKeyPreservesNativeBits) {
  auto ep = CausalRightCudaProvider();
  if (!ep) GTEST_SKIP() << "CUDA memory-efficient attention is not compiled";
  if (GetCudaArchitecture() < 900 || GetCudaArchitecture() >= 1000) GTEST_SKIP() << "Requires compute capability 9.x";
  std::mt19937 rng(42);
  auto run = [&](auto type) {
    using T = decltype(type);
    const auto q = RandomNative<T>(2 * 3 * 256, rng);
    const auto k = RandomNative<T>(q.size(), rng);
    const auto v = RandomNative<T>(q.size(), rng);
    OpTester test("CausalRightAttention", 1, kMSDomain);
    test.AddAttribute("scale", .0625f);
    test.AddInput<T>("Q", {2, 3, 1, 256}, q);
    test.AddInput<T>("K", {2, 3, 1, 256}, k);
    test.AddInput<T>("V", {2, 3, 1, 256}, v);
    test.AddOutput<T>("Y", {2, 3, 1, 256}, v);
    test.SetOutputAbsErr("Y", 0);
    test.SetOutputRelErr("Y", 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  run(MLFloat16{});
  run(BFloat16{});
}

TEST(CausalRightAttentionTest, RuntimeRejectsUnknownInvalidDimensions) {
  auto ep = CausalRightCudaProvider();
  if (!ep) GTEST_SKIP() << "CUDA memory-efficient attention is not compiled";
  for (const Shapes& shapes : {
           Shapes{{{1, 2, 8, 256}, {1, 2, 7, 256}, {1, 2, 7, 256}}},
           Shapes{{{1, 2, 1, 128}, {1, 2, 30, 128}, {1, 2, 30, 128}}},
           Shapes{{{1, 2, 1, 256}, {1, 1, 30, 256}, {1, 1, 30, 256}}}}) {
    auto count = [](const std::vector<int64_t>& shape) {
      size_t size = 1;
      for (auto dim : shape) size *= static_cast<size_t>(dim);
      return size;
    };
    OpTester test("CausalRightAttention", 1, kMSDomain);
    test.AddShapeToTensorData(false);
    test.AddAttribute("scale", .0625f);
    test.AddInput<MLFloat16>("Q", shapes[0], std::vector<MLFloat16>(count(shapes[0])));
    test.AddInput<MLFloat16>("K", shapes[1], std::vector<MLFloat16>(count(shapes[1])));
    test.AddInput<MLFloat16>("V", shapes[2], std::vector<MLFloat16>(count(shapes[2])));
    test.AddOutput<MLFloat16>("Y", shapes[0], std::vector<MLFloat16>(count(shapes[0])));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure, "1<=S<=T<=256", {}, nullptr, &providers);
  }
}
#endif  // USE_CUDA

}  // namespace
}  // namespace onnxruntime::test
