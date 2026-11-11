// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <iostream>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {

struct RouterShapeCase {
  std::array<std::optional<std::vector<int64_t>>, 3> shapes{
      std::vector<int64_t>{3, 256}, std::vector<int64_t>{9, 256}, std::vector<int64_t>{9}};
  std::optional<int64_t> k = 2;
  int dtype = ONNX_NAMESPACE::TensorProto::FLOAT16;
};

Status ResolveRouter(const RouterShapeCase& spec, bool verify_outputs = true) {
  Model model("group_router_projection_shape_inference", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}, {kMSDomain, 1}}, {},
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<NodeArg*> inputs;
  for (size_t i = 0; i < spec.shapes.size(); ++i) {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor = type.mutable_tensor_type();
    tensor->set_elem_type(i == 2 ? ONNX_NAMESPACE::TensorProto::FLOAT : spec.dtype);
    if (spec.shapes[i]) {
      auto* shape = tensor->mutable_shape();
      for (int64_t value : *spec.shapes[i]) {
        auto* dim = shape->add_dim();
        if (value == -1) {
          dim->set_dim_param(i == 0 ? "rows_or_width" : "unknown");
        } else {
          dim->set_dim_value(value);
        }
      }
    }
    inputs.push_back(&graph.GetOrCreateNodeArg("input_" + std::to_string(i), &type));
  }
  auto& empty = graph.GetOrCreateNodeArg("", nullptr);
  auto& scores = graph.GetOrCreateNodeArg("selection_scores", nullptr);
  auto& weights = graph.GetOrCreateNodeArg("router_weights", nullptr);
  auto& node = graph.AddNode("router", "MoEGroupRouter", "",
                             {inputs[0], inputs[2], inputs[1]},
                             {&empty, &scores, &weights},
                             nullptr, kMSDomain);
  if (spec.k) node.AddAttribute("top_k", *spec.k);
  graph.SetOutputs({&scores, &weights});
  const auto status = graph.Resolve();
  if (status.IsOK() && verify_outputs) {
    EXPECT_EQ(scores.TypeAsProto()->tensor_type().elem_type(),
              ONNX_NAMESPACE::TensorProto::FLOAT);
    EXPECT_EQ(weights.TypeAsProto()->tensor_type().elem_type(), ONNX_NAMESPACE::TensorProto::FLOAT);
    for (const auto* output : {&scores, &weights}) {
      const auto* shape = output->Shape();
      EXPECT_NE(shape, nullptr);
      if (!shape) continue;
      EXPECT_EQ(shape->dim_size(), 2);
      if (shape->dim_size() != 2) continue;
      if (spec.shapes[1] && (*spec.shapes[1])[0] != -1) {
        EXPECT_EQ(shape->dim(1).dim_value(), (*spec.shapes[1])[0]);
      } else {
        EXPECT_FALSE(shape->dim(1).has_dim_value());
      }
      if (!spec.shapes[0]) {
        EXPECT_FALSE(shape->dim(0).has_dim_value());
      } else if ((*spec.shapes[0])[0] == -1) {
        EXPECT_EQ(shape->dim(0).dim_param(), "rows_or_width");
      } else {
        EXPECT_EQ(shape->dim(0).dim_value(), (*spec.shapes[0])[0]);
      }
    }
  }
  return status;
}

TEST(MoEGroupRouterProjectionShapeInferenceTest, StaticSymbolicAndUnknownShapes) {
  for (int dtype : {ONNX_NAMESPACE::TensorProto::FLOAT16, ONNX_NAMESPACE::TensorProto::BFLOAT16}) {
    RouterShapeCase spec;
    spec.dtype = dtype;
    ASSERT_STATUS_OK(ResolveRouter(spec));
    spec.shapes[0] = std::vector<int64_t>{0, 256};
    ASSERT_STATUS_OK(ResolveRouter(spec));
    spec.shapes = {std::vector<int64_t>{2048, 2048}, std::vector<int64_t>{512, 2048},
                   std::vector<int64_t>{512}};
    spec.k = 32;
    ASSERT_STATUS_OK(ResolveRouter(spec));
    spec.shapes[0] = std::vector<int64_t>{-1, -1};
    ASSERT_STATUS_OK(ResolveRouter(spec));
    spec.shapes = {std::nullopt, std::nullopt, std::nullopt};
    ASSERT_STATUS_OK(ResolveRouter(spec));
    spec.shapes[1] = std::vector<int64_t>{-1, 512};
    spec.shapes[2] = std::vector<int64_t>{-1};
    ASSERT_STATUS_OK(ResolveRouter(spec));
  }
}

TEST(MoEGroupRouterProjectionShapeInferenceTest, FullScoresUseExpertDimensionAndFloat32) {
  for (int dtype : {ONNX_NAMESPACE::TensorProto::FLOAT16, ONNX_NAMESPACE::TensorProto::BFLOAT16}) {
    RouterShapeCase spec;
    spec.dtype = dtype;
    ASSERT_STATUS_OK(ResolveRouter(spec));
    spec.shapes[0] = std::vector<int64_t>{-1, 256};
    ASSERT_STATUS_OK(ResolveRouter(spec));
    spec.shapes[0] = std::vector<int64_t>{0, 256};
    ASSERT_STATUS_OK(ResolveRouter(spec));
    spec.shapes = {std::nullopt, std::nullopt, std::nullopt};
    ASSERT_STATUS_OK(ResolveRouter(spec));
  }
}

#ifndef ORT_NO_EXCEPTIONS
TEST(MoEGroupRouterProjectionShapeInferenceTest, RejectsBadAttributesRanksAndKnownDimensions) {
  using Mutate = std::function<void(RouterShapeCase&)>;
  const std::vector<std::pair<Mutate, std::string>> cases{
      {[](auto& s) { s.k = std::nullopt; }, "k"},
      {[](auto& s) { s.k = 0; }, "top_k must be positive, got 0"},
      {[](auto& s) { s.k = 33; }, "k=33 projection_weight experts=9"},
      {[](auto& s) { s.shapes[0] = std::vector<int64_t>{}; }, "input must be rank 2, got 0"},
      {[](auto& s) { s.shapes[0] = std::vector<int64_t>{256}; }, "input must be rank 2, got 1"},
      {[](auto& s) { s.shapes[0] = std::vector<int64_t>{1, 3, 256}; }, "input must be rank 2, got 3"},
      {[](auto& s) { s.shapes[1] = std::vector<int64_t>{}; }, "weight must be rank 2, got 0"},
      {[](auto& s) { s.shapes[1] = std::vector<int64_t>{9}; }, "weight must be rank 2, got 1"},
      {[](auto& s) { s.shapes[1] = std::vector<int64_t>{1, 9, 256}; }, "weight must be rank 2, got 3"},
      {[](auto& s) { s.shapes[2] = std::vector<int64_t>{}; }, "bias must be rank 1, got 0"},
      {[](auto& s) { s.shapes[2] = std::vector<int64_t>{1, 9}; }, "bias must be rank 1, got 2"},
      {[](auto& s) { s.shapes[0] = std::vector<int64_t>{int64_t{1} << 31, 256}; },
       "rows must be in [0, INT_MAX], got 2147483648"},
      {[](auto& s) { s.shapes[0] = std::vector<int64_t>{3, 0}; }, "input width must be a positive multiple of 256"},
      {[](auto& s) { s.shapes[0] = std::vector<int64_t>{3, 255}; }, "got 255"},
      {[](auto& s) { s.shapes[0] = std::vector<int64_t>{3, int64_t{1} << 31}; }, "got 2147483648"},
      {[](auto& s) { s.k = 10; }, "k=10 projection_weight experts=9"},
      {[](auto& s) { s.shapes[1] = std::vector<int64_t>{513, 256}; }, "weight experts=513"},
      {[](auto& s) { s.shapes[1] = std::vector<int64_t>{9, 512}; },
       "weight width must match input width, got 512 and 256"},
      {[](auto& s) { s.shapes[2] = std::vector<int64_t>{8}; }, "bias length must match weight experts, got 8 and 9"},
      // Validate every available input even if activations or weights have unknown rank.
      {[](auto& s) { s.shapes[0] = std::nullopt; s.shapes[1] = std::vector<int64_t>{9}; }, "weight must be rank 2"},
      {[](auto& s) { s.shapes[0] = std::nullopt; s.shapes[1] = std::vector<int64_t>{9, 0}; }, "weight width"},
      {[](auto& s) { s.shapes[0] = std::nullopt; s.shapes[2] = std::vector<int64_t>{1, 9}; }, "bias must be rank 1"},
      {[](auto& s) { s.shapes[1] = std::nullopt; s.shapes[2] = std::vector<int64_t>{513}; }, "bias experts=513"},
      {[](auto& s) { s.shapes[1] = std::nullopt; s.shapes[2] = std::vector<int64_t>{1}; }, "k=2 expert_bias experts=1"},
      {[](auto& s) { s.shapes[0] = std::nullopt; s.shapes[2] = std::vector<int64_t>{8}; }, "got 8 and 9"}};
  for (const auto& [mutate, message] : cases) {
    SCOPED_TRACE(message);
    RouterShapeCase spec;
    mutate(spec);
    const auto status = ResolveRouter(spec, false);
    EXPECT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find(message), std::string::npos) << status.ErrorMessage();
  }
}
#endif

}  // namespace
}  // namespace onnxruntime::test

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace onnxruntime::test {
namespace {

struct RouterReference {
  std::vector<int32_t> indices;
  std::vector<float> weights;
  std::vector<float> full_scores;
  std::vector<float> mixing_values;
  float min_gap = std::numeric_limits<float>::infinity();
};

template <typename T>
RouterReference ReferenceRouter(int rows, int width, int experts, int k, float scale,
                                const std::vector<T>& input, const std::vector<T>& weight,
                                const std::vector<float>& bias) {
  RouterReference result;
  for (int row = 0; row < rows; ++row) {
    std::vector<float> scores(experts), choices(experts);
    std::vector<int32_t> order(experts);
    std::iota(order.begin(), order.end(), 0);
    for (int expert = 0; expert < experts; ++expert) {
      double dot = 0;
      for (int col = 0; col < width; ++col) {
        dot += static_cast<double>(input[row * width + col].ToFloat()) *
               weight[expert * width + col].ToFloat();
      }
      scores[expert] = static_cast<float>(1.0 / (1.0 + std::exp(-dot)));
      choices[expert] = scores[expert] + bias[expert];
    }
    result.full_scores.insert(result.full_scores.end(), choices.begin(), choices.end());
    result.mixing_values.insert(result.mixing_values.end(), scores.begin(), scores.end());
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      if (std::isnan(choices[a]) != std::isnan(choices[b])) return !std::isnan(choices[a]);
      if (choices[a] > choices[b]) return true;
      if (choices[a] < choices[b]) return false;
      return a < b;
    });
    float sum = 0.f;
    for (int rank = 0; rank < k; ++rank) {
      sum += scores[order[rank]];
      if (rank + 1 < experts) {
        result.min_gap = std::min(result.min_gap, choices[order[rank]] - choices[order[rank + 1]]);
      }
    }
    for (int rank = 0; rank < k; ++rank) {
      result.indices.push_back(order[rank]);
      result.weights.push_back(scores[order[rank]] / (sum + 1e-20f) * scale);
    }
  }
  return result;
}

template <typename T>
void CheckReferenceRouter(int rows, int width, int experts, int k,
                          const std::vector<T>& input, const std::vector<T>& weight,
                          const std::vector<float>& bias, const RouterReference& reference) {
  OpTester test("MoEGroupRouter", 1, kMSDomain);
  test.AddAttribute("top_k", static_cast<int64_t>(k));
  test.AddInput<T>("input", {rows, width}, input);
  test.AddInput<float>("bias", {experts}, bias, true);
  test.AddInput<T>("weight", {experts, width}, weight, true);
  test.AddOptionalOutputEdge<T>();
  test.AddOutput<float>("selection_scores", {rows, experts}, reference.full_scores);
  test.AddOutput<float>("router_weights", {rows, experts}, reference.mixing_values);
  test.SetOutputTolerance(1e-5f, 1e-5f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
void CheckRandomProjection(int width, int experts, int k, bool ties = false) {
  constexpr int rows = 3;
  constexpr float scale = 2.5f;
  constexpr float minimum_index_gap = 1e-5f;
  std::mt19937 generator(20261109 + width + experts);
  std::uniform_real_distribution<float> values(-.125f, .125f);
  std::uniform_real_distribution<float> biases(-.2f, .2f);
  std::vector<T> input(rows * width), weight(experts * width);
  std::vector<float> bias(experts);
  RouterReference reference;
  for (int attempt = 0; attempt < 100; ++attempt) {
    for (auto& value : input) value = T(values(generator));
    for (auto& value : weight) value = T(values(generator));
    for (auto& value : bias) value = biases(generator);
    if (ties) {
      // Experts 5 and 37 occupy different slots of the same top-k lane.
      std::copy_n(weight.begin() + 5 * width, width, weight.begin() + 37 * width);
      bias[5] = bias[37] = 4.f;
    }
    reference = ReferenceRouter(rows, width, experts, k, scale, input, weight, bias);
    // Exact index comparisons are meaningful only outside numerical near-ties.
    if (ties || reference.min_gap > minimum_index_gap) break;
  }
  if (!ties) {
    ASSERT_GT(reference.min_gap, minimum_index_gap);
  }
  if (ties) {
    for (int row = 0; row < rows; ++row) {
      ASSERT_EQ(reference.indices[row * k], 5);
      ASSERT_EQ(reference.indices[row * k + 1], 37);
    }
  }
  CheckReferenceRouter(rows, width, experts, k, input, weight, bias, reference);
}

template <typename T>
void CheckRoutes(int rows, bool tied) {
  constexpr int width = 256;
  constexpr int experts = 4;
  constexpr int k = 2;
  OpTester test("MoEGroupRouter", 1, kMSDomain);
  test.AddAttribute("top_k", int64_t{k});
  test.AddInput<T>("input", {rows, width}, std::vector<T>(rows * width, T(0.f)));
  const std::vector<float> bias = tied ? std::vector<float>(experts, 0.f)
                                       : std::vector<float>{0.f, 3.f, 2.f, 1.f};
  test.AddInput<float>("bias", {experts}, bias, true);
  test.AddInput<T>("weight", {experts, width}, std::vector<T>(experts * width, T(0.f)), true);
  std::vector<float> scores;
  for (int row = 0; row < rows; ++row) {
    for (float value : bias) scores.push_back(.5f + value);
  }
  test.AddOptionalOutputEdge<T>();
  test.AddOutput<float>("selection_scores", {rows, experts}, scores);
  test.AddOutput<float>("router_weights", {rows, experts}, std::vector<float>(rows * experts, .5f));
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
void CheckProjection() {
  constexpr int width = 256;
  constexpr int experts = 4;
  std::vector<T> input(width, T(0.f)), weight(experts * width, T(0.f));
  input[0] = T(1.f);
  for (int expert = 0; expert < experts; ++expert) weight[expert * width] = T(float(expert - 1));
  const float first = 1.f / (1.f + std::exp(1.f));
  OpTester test("MoEGroupRouter", 1, kMSDomain);
  test.AddAttribute("top_k", int64_t{2});
  test.AddInput<T>("input", {1, width}, input);
  test.AddInput<float>("bias", {experts}, {4.f, 3.f, 0.f, 0.f}, true);
  test.AddInput<T>("weight", {experts, width}, weight, true);
  test.AddOptionalOutputEdge<T>();
  const float third = 1.f / (1.f + std::exp(-1.f));
  const float fourth = 1.f / (1.f + std::exp(-2.f));
  test.AddOutput<float>("selection_scores", {1, experts}, {first + 4.f, 3.5f, third, fourth});
  test.AddOutput<float>("router_weights", {1, experts}, {first, .5f, third, fourth});
  test.SetOutputTolerance(1e-6f, 1e-6f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
void CheckMaximumK() {
  constexpr int width = 256;
  constexpr int experts = 512;
  constexpr int k = 32;
  OpTester test("MoEGroupRouter", 1, kMSDomain);
  test.AddAttribute("top_k", int64_t{k});
  test.AddInput<T>("input", {1, width}, std::vector<T>(width, T(0.f)));
  test.AddInput<float>("bias", {experts}, std::vector<float>(experts, 0.f), true);
  test.AddInput<T>("weight", {experts, width}, std::vector<T>(experts * width, T(0.f)), true);
  test.AddOptionalOutputEdge<T>();
  test.AddOutput<float>("selection_scores", {1, experts}, std::vector<float>(experts, .5f));
  test.AddOutput<float>("router_weights", {1, experts}, std::vector<float>(experts, .5f));
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(MoEGroupRouterProjectionTest, SelectionBiasDoesNotChangeMixtureWeights) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {0, 1, 8, 16}) {
    CheckRoutes<BFloat16>(rows, false);
    CheckRoutes<MLFloat16>(rows, false);
  }
}

TEST(MoEGroupRouterProjectionTest, PreservesTiedSelectionScores) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  CheckRoutes<BFloat16>(8, true);
  CheckRoutes<MLFloat16>(8, true);
}

TEST(MoEGroupRouterProjectionTest, ProjectionUsesUnbiasedAffinities) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  CheckProjection<BFloat16>();
  CheckProjection<MLFloat16>();
}

TEST(MoEGroupRouterProjectionTest, MaximumExpertsAndK) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  CheckMaximumK<BFloat16>();
  CheckMaximumK<MLFloat16>();
}

TEST(MoEGroupRouterProjectionTest, DistinctRandomRowsMatchIndependentReference) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int width : {256, 512, 2048}) {
    for (int experts : {9, 33, 257, 512}) {
      const int k = experts == 9 ? 1 : experts == 33 ? 2
                                   : experts == 257  ? 7
                                                     : 32;
      SCOPED_TRACE(testing::Message() << "width=" << width << " experts=" << experts << " k=" << k);
      CheckRandomProjection<MLFloat16>(width, experts, k);
      CheckRandomProjection<BFloat16>(width, experts, k);
    }
  }
}

TEST(MoEGroupRouterProjectionTest, TiesWithinLaneSlotsMatchIndependentReference) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  CheckRandomProjection<MLFloat16>(512, 257, 2, true);
  CheckRandomProjection<BFloat16>(2048, 512, 2, true);
}

template <typename T>
void CheckNonfiniteActivations() {
  constexpr int rows = 3, width = 512, experts = 33, k = 4;
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<T> input(rows * width, T(.0625f)), weight(experts * width, T(.03125f));
  input[0] = T(inf);
  input[width] = T(-inf);
  input[2 * width] = T(std::numeric_limits<float>::quiet_NaN());
  for (int expert = 0; expert < experts; ++expert) {
    weight[expert * width] = T(expert % 2 == 0 ? 1.f : -1.f);
  }
  const std::vector<float> bias(experts, 0.f);
  const auto reference = ReferenceRouter(rows, width, experts, k, 1.f, input, weight, bias);
  EXPECT_EQ(reference.indices, (std::vector<int32_t>{0, 2, 4, 6, 1, 3, 5, 7, 0, 1, 2, 3}));
  CheckReferenceRouter(rows, width, experts, k, input, weight, bias, reference);
}

TEST(MoEGroupRouterProjectionTest, NonfiniteActivationsAreNotSanitized) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  CheckNonfiniteActivations<MLFloat16>();
  CheckNonfiniteActivations<BFloat16>();
}

TEST(MoEGroupRouterProjectionTest, ZeroAffinitiesRemainUnnormalized) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  const std::vector<MLFloat16> input(256, MLFloat16(1.f)), weight(9 * 256, MLFloat16(-1.f));
  const std::vector<float> bias(9, 0.f);
  const auto reference = ReferenceRouter(1, 256, 9, 2, 1.f, input, weight, bias);
  CheckReferenceRouter(1, 256, 9, 2, input, weight, bias, reference);
}

TEST(MoEGroupRouterProjectionTest, NonfiniteBiasDoesNotChangeMixingValues) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int width = 256;
  const float inf = std::numeric_limits<float>::infinity();
  OpTester test("MoEGroupRouter", 1, kMSDomain);
  test.AddAttribute("top_k", int64_t{4});
  test.AddInput<BFloat16>("input", {1, width}, std::vector<BFloat16>(width, BFloat16(0.f)));
  test.AddInput<float>("bias", {4}, {-inf, inf, 0.f, std::numeric_limits<float>::quiet_NaN()}, true);
  test.AddInput<BFloat16>("weight", {4, width}, std::vector<BFloat16>(4 * width, BFloat16(0.f)), true);
  test.AddOptionalOutputEdge<BFloat16>();
  test.AddOutput<float>("selection_scores", {1, 4},
                        {-inf, inf, .5f, std::numeric_limits<float>::quiet_NaN()});
  test.AddOutput<float>("router_weights", {1, 4}, {.5f, .5f, .5f, .5f});
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
class ChainedRouterTester final : public OpTester {
 public:
  ChainedRouterTester() : OpTester("MoEPacked", 1, kMSDomain) {}

 private:
  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>&) override {
    auto& empty = graph.GetOrCreateNodeArg("", nullptr);
    auto& router = graph.AddNode("group_router", "MoEGroupRouter", "",
                                 {inputs[1], inputs[3], inputs[2]},
                                 {&empty, outputs[1], outputs[2]}, nullptr, kMSDomain);
    router.AddAttribute("top_k", int64_t{3});
    auto& expert_output = graph.GetOrCreateNodeArg("expert_output", nullptr);
    auto& chained = graph.AddNode("chained_moe", "MoEPacked", "",
                                  {inputs[0], outputs[1], inputs[4], &empty, inputs[5], outputs[2]},
                                  {&expert_output}, nullptr, kMSDomain);
    chained.AddAttribute("k", int64_t{3});
    chained.AddAttribute("activation_type", std::string("relu_squared"));
    chained.AddAttribute("gate_type", int64_t{16});
    chained.AddAttribute("down_type", int64_t{16});
    chained.AddAttribute("normalize_routing_weights", int64_t{1});
    graph.AddNode("external_routed_scale", "Mul", "", {&expert_output, inputs[6]}, {outputs[0]});
    graph.SetOutputs({outputs[0], outputs[1], outputs[2]});
  }
};

template <typename T>
void CheckChainedRouter(int rows) {
  constexpr int width = 512, hidden = 256, intermediate = 256, experts = 9, k = 3;
  std::mt19937 generator(20261109);
  std::uniform_real_distribution<float> random(-.125f, .125f);
  std::vector<T> router_input(rows * width), router_weight(experts * width), input(rows * hidden);
  std::vector<float> bias(experts);
  for (auto& value : router_input) value = T(random(generator));
  for (auto& value : router_weight) value = T(random(generator));
  for (auto& value : bias) value = random(generator);
  auto routes = ReferenceRouter(rows, width, experts, k, 2.5f, router_input, router_weight, bias);
  ASSERT_GT(routes.min_gap, 1e-4f);
  const auto fp32_routes = routes;
  for (int row = 0; row < rows; ++row) {
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return routes.full_scores[row * experts + a] > routes.full_scores[row * experts + b];
    });
    float sum = 0.f;
    for (int rank = 0; rank < k; ++rank) sum += routes.mixing_values[row * experts + order[rank]];
    const float inverse = sum == 0.f ? 0.f : 1.f / sum;
    for (int rank = 0; rank < k; ++rank) {
      routes.indices[row * k + rank] = order[rank];
      routes.weights[row * k + rank] = routes.mixing_values[row * experts + order[rank]] * inverse;
    }
  }
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < hidden; ++col) {
      input[row * hidden + col] = T(static_cast<float>((row + col) % 9 - 3) / 32.f);
    }
  }
  const auto gate_scale = [](int expert, int col) {
    return MLFloat16(static_cast<float>(1 + expert % 3 + col % 3) / 256.f).ToFloat();
  };
  const auto down_scale = [](int expert, int col) {
    return MLFloat16(static_cast<float>(1 + (expert + 2 * col) % 5) / 512.f).ToFloat();
  };
  const auto bank = [](int columns, const auto& scale) {
    std::vector<uint8_t> bytes(experts * columns * 66, 0);
    for (int expert = 0; expert < experts; ++expert) {
      for (int col = 0; col < columns; ++col) {
        const MLFloat16 d(scale(expert, col));
        auto* block = bytes.data() + (expert * columns + col) * 66;
        block[0] = static_cast<uint8_t>(d.val);
        block[1] = static_cast<uint8_t>(d.val >> 8);
        // IQ2_XXS grid 1 is [43, 8, ..., 8], with positive signs and scale d/8.
        for (int record = 0; record < 8; ++record) {
          for (int group = 0; group < 4; ++group) block[2 + record * 8 + group] = 1;
        }
      }
    }
    return bytes;
  };
  const auto decoded = [](float scale, int col) {
    return T(scale * (col % 8 == 0 ? 5.375f : 1.f)).ToFloat();
  };
  std::vector<T> expected(rows * hidden), fp32_expected(rows * hidden);
  for (int row = 0; row < rows; ++row) {
    for (bool previous : {false, true}) {
      std::vector<float> mixed(hidden, 0.f);
      for (int rank = 0; rank < k; ++rank) {
        const auto& reference = previous ? fp32_routes : routes;
        const int expert = reference.indices[row * k + rank];
        std::vector<T> activation(intermediate);
        for (int col = 0; col < intermediate; ++col) {
          double dot = 0;
          for (int in = 0; in < hidden; ++in) {
            dot += static_cast<double>(input[row * hidden + in].ToFloat()) * decoded(gate_scale(expert, col), in);
          }
          const float relu = std::max(0.f, T(static_cast<float>(dot)).ToFloat());
          activation[col] = T(relu * relu);
        }
        for (int col = 0; col < hidden; ++col) {
          double dot = 0;
          for (int in = 0; in < intermediate; ++in) {
            dot += static_cast<double>(activation[in].ToFloat()) * decoded(down_scale(expert, col), in);
          }
          const float fc2 = T(static_cast<float>(dot)).ToFloat();
          mixed[col] = std::fma(reference.weights[row * k + rank], fc2, mixed[col]);
        }
      }
      for (int col = 0; col < hidden; ++col) {
        if (previous)
          fp32_expected[row * hidden + col] = T(mixed[col]);
        else
          expected[row * hidden + col] = T(T(mixed[col]).ToFloat() * 2.5f);
      }
    }
  }
  ChainedRouterTester<T> test;
  test.template AddInput<T>("hidden", {rows, hidden}, input);
  test.template AddInput<T>("router_input", {rows, width}, router_input);
  test.template AddInput<T>("router_weight", {experts, width}, router_weight, true);
  test.template AddInput<float>("bias", {experts}, bias, true);
  test.template AddInput<uint8_t>("gate", {experts, intermediate, 66}, bank(intermediate, gate_scale), true);
  test.template AddInput<uint8_t>("down", {experts, hidden, 66}, bank(hidden, down_scale), true);
  test.template AddInput<T>("routed_scale", {}, {T(2.5f)}, true);
  test.template AddOutput<T>("chained", {rows, hidden}, expected);
  test.template AddOutput<float>("full_selection_scores", {rows, experts}, routes.full_scores);
  test.template AddOutput<float>("unbiased_mixing_values", {rows, experts}, routes.mixing_values);
  test.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string& provider) {
    ASSERT_EQ(provider, kCudaExecutionProvider);
    ASSERT_EQ(fetches.size(), 3u);
    const auto chained = fetches[0].Get<Tensor>().DataAsSpan<T>();
    const auto scores = fetches[1].Get<Tensor>().DataAsSpan<float>();
    const auto mixing = fetches[2].Get<Tensor>().DataAsSpan<float>();
    ASSERT_EQ(chained.size(), expected.size());
    ASSERT_EQ(scores.size(), routes.full_scores.size());
    ASSERT_EQ(mixing.size(), routes.mixing_values.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      const float ref = expected[i].ToFloat();
      const float tolerance = std::is_same_v<T, BFloat16> ? .01f : .002f;
      EXPECT_NEAR(chained[i].ToFloat(), ref, 1e-6f + tolerance * std::abs(ref)) << "element " << i;
    }
    for (size_t i = 0; i < routes.full_scores.size(); ++i) {
      EXPECT_NEAR(scores[i], routes.full_scores[i], 1e-5f) << "score " << i;
      EXPECT_NEAR(mixing[i], routes.mixing_values[i], 1e-5f) << "mixing value " << i;
    }
    double squared_error = 0, reference_norm = 0, actual_norm = 0, dot = 0;
    for (size_t i = 0; i < chained.size(); ++i) {
      const double a = chained[i].ToFloat(), b = fp32_expected[i].ToFloat();
      squared_error += (a - b) * (a - b);
      reference_norm += b * b;
      actual_norm += a * a;
      dot += a * b;
    }
    const double nrmse = std::sqrt(squared_error / reference_norm);
    const double cosine = dot / std::sqrt(actual_norm * reference_norm);
    std::cout << "CHAIN_FP32_COMPATIBILITY dtype=" << (std::is_same_v<T, BFloat16> ? "bf16" : "fp16")
              << " rows=" << rows << " nrmse=" << nrmse << " cosine=" << cosine << "\n";
    EXPECT_LE(nrmse, .03);
    EXPECT_GE(cosine, .999);
  });
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  ASSERT_EQ(test.GetFetches().size(), 3u) << "The complete CUDA chain must actually execute";
}

TEST(MoEGroupRouterProjectionTest, ChainedFullScoreMoeMatchesHostReference) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {3, 9}) {
    SCOPED_TRACE(testing::Message() << "rows=" << rows);
    CheckChainedRouter<MLFloat16>(rows);
    CheckChainedRouter<BFloat16>(rows);
  }
}

#ifndef ORT_NO_EXCEPTIONS
TEST(MoEGroupRouterProjectionTest, RejectsInvalidRuntimeGeometryWithObservedValues) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  struct InvalidCase {
    std::vector<int64_t> input, weight, bias;
    int64_t k;
    std::string message;
  };
  const std::vector<InvalidCase> cases{
      {{1, 255}, {9, 255}, {9}, 1, "width must be a positive multiple of 256 and <= INT_MAX, got 255"},
      {{1, 0}, {9, 0}, {9}, 1, "width must be a positive multiple of 256 and <= INT_MAX, got 0"},
      {{1, 256}, {9, 256}, {9}, 10, "k <= experts <= 512, got k=10 experts=9"},
      {{1, 256}, {513, 256}, {513}, 1, "k <= experts <= 512, got k=1 experts=513"},
      {{1, 256}, {9, 512}, {9}, 1, "weight must be [experts, 256], got {9,512}"},
      {{1, 256}, {9, 256}, {8}, 1, "bias must be [9], got {8}"},
      {{1, 256}, {9, 256}, {1, 9}, 1, "bias must be [9], got {1,9}"},
      {{256}, {9, 256}, {9}, 1, "input must be rank 2, got {256}"},
      {{1, 256}, {256}, {9}, 1, "weight must be rank 2, got {256}"}};
  for (const auto& spec : cases) {
    SCOPED_TRACE(spec.message);
    OpTester test("MoEGroupRouter", 1, kMSDomain);
    // Hide static geometry so these cases reach host-kernel validation.
    test.AddShapeToTensorData(false);
    test.AddAttribute("top_k", spec.k);
    const auto count = [](const auto& shape) {
      return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    };
    test.AddInput<MLFloat16>("input", spec.input, std::vector<MLFloat16>(count(spec.input), MLFloat16(0.f)));
    test.AddInput<float>("bias", spec.bias, std::vector<float>(count(spec.bias), 0.f));
    test.AddInput<MLFloat16>("weight", spec.weight, std::vector<MLFloat16>(count(spec.weight), MLFloat16(0.f)));
    test.AddOptionalOutputEdge<MLFloat16>();
    const int64_t experts = spec.weight[0];
    test.AddOutput<float>("selection_scores", {1, experts}, std::vector<float>(experts, .5f));
    test.AddOutput<float>("router_weights", {1, experts}, std::vector<float>(experts, .5f));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectFailure, spec.message, {}, nullptr, &providers);
  }
}
#endif

}  // namespace
}  // namespace onnxruntime::test
#endif
