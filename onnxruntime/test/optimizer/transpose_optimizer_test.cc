// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"

#include "core/graph/graph.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct NhwcWeightsRange {
  static constexpr T min_value = std::numeric_limits<T>::min();
  static constexpr T max_value = std::numeric_limits<T>::max();
};

template <>
struct NhwcWeightsRange<int8_t> {
  // Avoid saturation from u8s8 math.
  static constexpr int8_t min_value = -63;
  static constexpr int8_t max_value = +63;
};

template <typename T>
NodeArg* NhwcMakeInitializer(ModelTestBuilder& builder, const std::vector<int64_t>& shape) {
  return builder.MakeInitializer<T>(shape,
                                    NhwcWeightsRange<T>::min_value,
                                    NhwcWeightsRange<T>::max_value);
}

void SetNodeArgShape(NodeArg* node_arg, const std::optional<std::vector<int64_t>>& shape) {
  if (shape == std::nullopt) {
    node_arg->ClearShape();
  } else {
    ONNX_NAMESPACE::TensorShapeProto shape_proto;
    for (int64_t d : *shape) {
      auto dim = shape_proto.add_dim();
      if (d != -1) {
        dim->set_dim_value(d);
      }
    }
    node_arg->SetShape(shape_proto);
  }
}

template <typename T>
NodeArg* MakeInput(ModelTestBuilder& builder, const std::optional<std::vector<int64_t>>& input_shape, const std::vector<int64_t>& value_shape, T min, T max) {
  auto node_arg = builder.MakeInput<T>(value_shape, min, max);
  SetNodeArgShape(node_arg, input_shape);
  return node_arg;
}

template <typename T>
NodeArg* MakeInput(ModelTestBuilder& builder, const std::optional<std::vector<int64_t>>& input_shape, const std::vector<int64_t>& value_shape, const std::vector<T>& data) {
  auto node_arg = builder.MakeInput<T>(value_shape, data);
  SetNodeArgShape(node_arg, input_shape);
  return node_arg;
}

size_t EstimateTransposeCost(const Graph& graph) {
  size_t cost = 0;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Transpose") {
      auto arg = node.OutputDefs()[0];
      auto shape = arg->Shape();
      if (shape == nullptr) {
        cost += 5;
      } else {
        for (int i = 0; i < shape->dim_size(); ++i) {
          auto d = shape->dim(i);
          if (!d.has_dim_value() || d.dim_value() != 1) {
            cost += 1;
          }
        }
      }
    }
    if (node.ContainsSubgraph()) {
      for (auto& subgraph : node.GetSubgraphs()) {
        cost += EstimateTransposeCost(*subgraph);
      }
    }
  }
  return cost;
}

TEST(TransposeOptimizerTests, Sum) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, -1, 3, 4}}, {1, 2, 3, 4}, 0.0, 1.0);
    auto* input1_arg = MakeInput<float>(builder, {{-1, 3}}, {2, 3}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<float>({2, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();
    auto* sum_1_out_0 = builder.MakeOutput();
    auto* transpose_3_out_0 = builder.MakeOutput();
    auto* identity_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Sum", {transpose_1_out_0, const_1, input1_arg, const_1}, {sum_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {sum_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
    auto& transpose_3 = builder.AddNode("Transpose", {sum_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
    builder.AddNode("Identity", {const_1}, {identity_1_out_0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 6);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, Concat) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* input1_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 7, 2, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* concat_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& concat_1 = builder.AddNode("Concat", {transpose_1_out_0, transpose_2_out_0}, {concat_1_out_0});
    concat_1.AddAttribute("axis", (int64_t)2);
    auto& transpose_3 = builder.AddNode("Transpose", {concat_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, Split) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* split_1_out_0 = builder.MakeIntermediate();
    auto* split_1_out_1 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& split_1 = builder.AddNode("Split", {transpose_1_out_0}, {split_1_out_0, split_1_out_1});
    split_1.AddAttribute("axis", (int64_t)-1);
    auto& transpose_2 = builder.AddNode("Transpose", {split_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& transpose_3 = builder.AddNode("Transpose", {split_1_out_1}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_2 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* split_1_out_0 = builder.MakeIntermediate();
    auto* split_1_out_1 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Split", {transpose_1_out_0}, {split_1_out_0, split_1_out_1});
    auto& transpose_2 = builder.AddNode("Transpose", {split_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& transpose_3 = builder.AddNode("Transpose", {split_1_out_1}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_2 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_2,
                    check_optimized_graph_2,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_3 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* split_1_out_0 = builder.MakeIntermediate();
    auto* split_1_out_1 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& split_1 = builder.AddNode("Split", {transpose_1_out_0}, {split_1_out_0, split_1_out_1});
    split_1.AddAttribute("axis", (int64_t)1);
    auto& transpose_2 = builder.AddNode("Transpose", {split_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& transpose_3 = builder.AddNode("Transpose", {split_1_out_1}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_3 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_3,
                    check_optimized_graph_3,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, Transpose) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {1, 2, 3, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* relu_1_out_0 = builder.MakeIntermediate();
    auto* transpose_6_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();
    auto* transpose_3_out_0 = builder.MakeOutput();
    auto* transpose_4_out_0 = builder.MakeOutput();
    auto* transpose_5_out_0 = builder.MakeOutput();
    auto* transpose_7_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Transpose", {transpose_1_out_0}, {transpose_2_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {transpose_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
    auto& transpose_4 = builder.AddNode("Transpose", {transpose_1_out_0}, {transpose_4_out_0});
    transpose_4.AddAttribute("perm", std::vector<int64_t>{2, 0, 1, 3});
    auto& transpose_5 = builder.AddNode("Transpose", {transpose_3_out_0}, {transpose_5_out_0});
    transpose_5.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
    builder.AddNode("Relu", {input0_arg}, {relu_1_out_0});
    auto& transpose_6 = builder.AddNode("Transpose", {relu_1_out_0}, {transpose_6_out_0});
    transpose_6.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& transpose_7 = builder.AddNode("Transpose", {transpose_6_out_0}, {transpose_7_out_0});
    transpose_7.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 20);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, Add) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* add_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Add", {transpose_1_out_0, input1_arg}, {add_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {add_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 7);
}

TEST(TransposeOptimizerTests, Mul2) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* relu_1_out_0 = builder.MakeIntermediate();
    auto* relu_2_out_0 = builder.MakeIntermediate();
    auto* mul_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Relu", {transpose_1_out_0}, {relu_1_out_0});
    builder.AddNode("Relu", {transpose_2_out_0}, {relu_2_out_0});
    builder.AddNode("Mul", {relu_1_out_0, relu_2_out_0}, {mul_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {mul_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 7);
}

TEST(TransposeOptimizerTests, Pad) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* pad_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& pad_1 = builder.AddNode("Pad", {transpose_1_out_0}, {pad_1_out_0});
    pad_1.AddAttribute("mode", "constant");
    pad_1.AddAttribute("value", (float)2.3);
    pad_1.AddAttribute("pads", std::vector<int64_t>{1, -2, 3, 4, 5, 6, 7, 8});
    auto& transpose_2 = builder.AddNode("Transpose", {pad_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 10);
}

TEST(TransposeOptimizerTests, PadNonconst) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, {{8}}, {8}, {1, 0, 0, 1, 0, 0, -1, 1});
    auto* const_1 = builder.MakeInitializer<int64_t>({8}, {1, -2, 3, 4, 5, 6, 7, 8});
    auto* const_2 = builder.MakeInitializer<float>({}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* add_1_out_0 = builder.MakeIntermediate();
    auto* pad_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Add", {input1_arg, const_1}, {add_1_out_0});
    auto& pad_1 = builder.AddNode("Pad", {transpose_1_out_0, add_1_out_0, const_2}, {pad_1_out_0});
    pad_1.AddAttribute("mode", "constant");
    auto& transpose_2 = builder.AddNode("Transpose", {pad_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 11);
}

TEST(TransposeOptimizerTests, ReduceSum) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {0, -2});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_2 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_2 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_2,
                    check_optimized_graph_2,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_3 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)1);
  };

  auto check_optimized_graph_3 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 4);
  };

  TransformerTester(build_test_case_3,
                    check_optimized_graph_3,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_4 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_4 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_4,
                    check_optimized_graph_4,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_5 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)0);
  };

  auto check_optimized_graph_5 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_5,
                    check_optimized_graph_5,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_6 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)0);
  };

  auto check_optimized_graph_6 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_6,
                    check_optimized_graph_6,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_7 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {0, -2});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
  };

  auto check_optimized_graph_7 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_7,
                    check_optimized_graph_7,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_8 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
  };

  auto check_optimized_graph_8 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_8,
                    check_optimized_graph_8,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_9 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)1);
  };

  auto check_optimized_graph_9 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 4);
  };

  TransformerTester(build_test_case_9,
                    check_optimized_graph_9,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_10 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
  };

  auto check_optimized_graph_10 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_10,
                    check_optimized_graph_10,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_11 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)0);
  };

  auto check_optimized_graph_11 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_11,
                    check_optimized_graph_11,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_12 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)0);
  };

  auto check_optimized_graph_12 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_12,
                    check_optimized_graph_12,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_13 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {0, -2});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_13 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_13,
                    check_optimized_graph_13,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_14 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_14 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_14,
                    check_optimized_graph_14,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_15 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)1);
  };

  auto check_optimized_graph_15 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 4);
  };

  TransformerTester(build_test_case_15,
                    check_optimized_graph_15,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_16 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_16 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_16,
                    check_optimized_graph_16,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_17 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)0);
  };

  auto check_optimized_graph_17 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_17,
                    check_optimized_graph_17,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);

  auto build_test_case_18 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)0);
  };

  auto check_optimized_graph_18 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_18,
                    check_optimized_graph_18,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13,
                    /*per_sample_tolerance*/ 1e-08,
                    /*relative_per_sample_tolerance*/ 1e-07);
}

TEST(TransposeOptimizerTests, ReduceSumNonconst) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, {{}}, {}, {-1});
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {0, -1});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* add_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Add", {input1_arg, const_1}, {add_1_out_0});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, add_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 4);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_2 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, {{}}, {}, {-1});
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {0, -1});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* add_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Add", {input1_arg, const_1}, {add_1_out_0});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, add_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_2 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 4);
  };

  TransformerTester(build_test_case_2,
                    check_optimized_graph_2,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, Squeeze) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{1, -1, 1, 2}}, {1, 4, 1, 2}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {0, -1});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* squeeze_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Squeeze", {transpose_1_out_0, const_1}, {squeeze_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {squeeze_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, SqueezeEmpty) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{1, -1, 1, 2}}, {1, 4, 1, 2}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* squeeze_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Squeeze", {transpose_1_out_0}, {squeeze_1_out_0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, SqueezeNonconst) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{1, -1, 1, 2}}, {1, 4, 1, 2}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, {{-1}}, {2}, {0, -1});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* squeeze_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Squeeze", {transpose_1_out_0, input1_arg}, {squeeze_1_out_0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, Unsqueeze) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({4}, {0, -4, -1, 5});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* unsqueeze_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Unsqueeze", {transpose_1_out_0, const_1}, {unsqueeze_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {unsqueeze_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 3, 6, 2, 0, 4, 5, 7});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 4);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, UnsqueezeNonconst) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, {{4}}, {4}, {0, -4, -1, 5});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* unsqueeze_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Unsqueeze", {transpose_1_out_0, input1_arg}, {unsqueeze_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {unsqueeze_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 3, 6, 2, 0, 4, 5, 7});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 9);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, Slice) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({1}, {1});
    auto* const_2 = builder.MakeInitializer<int64_t>({1}, {-1});
    auto* const_3 = builder.MakeInitializer<int64_t>({1}, {2});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Slice", {transpose_1_out_0, const_1, const_2, const_3}, {slice_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_2 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {1, -3});
    auto* const_2 = builder.MakeInitializer<int64_t>({2}, {2, 4});
    auto* const_3 = builder.MakeInitializer<int64_t>({2}, {1, -2});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Slice", {transpose_1_out_0, const_1, const_2, const_3}, {slice_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_2 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_2,
                    check_optimized_graph_2,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_3 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {1, 4});
    auto* const_2 = builder.MakeInitializer<int64_t>({2}, {2, -3});
    auto* const_3 = builder.MakeInitializer<int64_t>({2}, {1, -2});
    auto* const_4 = builder.MakeInitializer<int64_t>({2}, {2, -1});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Slice", {transpose_1_out_0, const_1, const_2, const_3, const_4}, {slice_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_3 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_3,
                    check_optimized_graph_3,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, SliceNonconst) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, {{2}}, {2}, {1, -2});
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {1, -3});
    auto* const_2 = builder.MakeInitializer<int64_t>({2}, {2, 4});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Slice", {transpose_1_out_0, const_1, const_2, input1_arg}, {slice_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 9);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, Tile) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, std::nullopt, {4}, {1, 2, 1, 3});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* tile_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Tile", {transpose_1_out_0, input1_arg}, {tile_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {tile_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, Argmin) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmin_1 = builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    argmin_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_2 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmin_1 = builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    argmin_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_2 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_2,
                    check_optimized_graph_2,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_3 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
  };

  auto check_optimized_graph_3 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_3,
                    check_optimized_graph_3,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_4 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmin_1 = builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    argmin_1.AddAttribute("axis", (int64_t)-2);
    argmin_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_4 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_4,
                    check_optimized_graph_4,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_5 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmin_1 = builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    argmin_1.AddAttribute("axis", (int64_t)-2);
    argmin_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_5 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_5,
                    check_optimized_graph_5,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_6 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmin_1 = builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    argmin_1.AddAttribute("axis", (int64_t)-2);
  };

  auto check_optimized_graph_6 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_6,
                    check_optimized_graph_6,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, Argmax) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmax_1 = builder.AddNode("ArgMax", {transpose_1_out_0}, {argmax_1_out_0});
    argmax_1.AddAttribute("axis", (int64_t)-2);
    argmax_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, ReduceMax) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducemax_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_2 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_2 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_2,
                    check_optimized_graph_2,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_3 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducemax_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_3 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_3,
                    check_optimized_graph_3,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_4 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_4 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_4,
                    check_optimized_graph_4,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_5 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
  };

  auto check_optimized_graph_5 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_5,
                    check_optimized_graph_5,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_6 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
  };

  auto check_optimized_graph_6 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_6,
                    check_optimized_graph_6,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, ReduceOps) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducelogsum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducelogsum_1 = builder.AddNode("ReduceLogSum", {transpose_1_out_0}, {reducelogsum_1_out_0});
    reducelogsum_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducelogsum_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_2 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducelogsumexp_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducelogsumexp_1 = builder.AddNode("ReduceLogSumExp", {transpose_1_out_0}, {reducelogsumexp_1_out_0});
    reducelogsumexp_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducelogsumexp_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_2 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_2,
                    check_optimized_graph_2,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_3 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducemax_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_3 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_3,
                    check_optimized_graph_3,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_4 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemean_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemean_1 = builder.AddNode("ReduceMean", {transpose_1_out_0}, {reducemean_1_out_0});
    reducemean_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducemean_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_4 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_4,
                    check_optimized_graph_4,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_5 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemin_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemin_1 = builder.AddNode("ReduceMin", {transpose_1_out_0}, {reducemin_1_out_0});
    reducemin_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducemin_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_5 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_5,
                    check_optimized_graph_5,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_6 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reduceprod_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reduceprod_1 = builder.AddNode("ReduceProd", {transpose_1_out_0}, {reduceprod_1_out_0});
    reduceprod_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reduceprod_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_6 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_6,
                    check_optimized_graph_6,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_7 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesumsquare_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesumsquare_1 = builder.AddNode("ReduceSumSquare", {transpose_1_out_0}, {reducesumsquare_1_out_0});
    reducesumsquare_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducesumsquare_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_7 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_7,
                    check_optimized_graph_7,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_8 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducel1_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducel1_1 = builder.AddNode("ReduceL1", {transpose_1_out_0}, {reducel1_1_out_0});
    reducel1_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducel1_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_8 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_8,
                    check_optimized_graph_8,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);

  auto build_test_case_9 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducel2_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducel2_1 = builder.AddNode("ReduceL2", {transpose_1_out_0}, {reducel2_1_out_0});
    reducel2_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducel2_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_9 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_9,
                    check_optimized_graph_9,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, SoftHardMax) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* softmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Softmax", {transpose_1_out_0}, {softmax_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {softmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_2 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* softmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& softmax_1 = builder.AddNode("Softmax", {transpose_1_out_0}, {softmax_1_out_0});
    softmax_1.AddAttribute("axis", (int64_t)0);
    auto& transpose_2 = builder.AddNode("Transpose", {softmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_2 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_2,
                    check_optimized_graph_2,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_3 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5, 6}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* softmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
    auto& softmax_1 = builder.AddNode("Softmax", {transpose_1_out_0}, {softmax_1_out_0});
    softmax_1.AddAttribute("axis", (int64_t)-3);
    auto& transpose_2 = builder.AddNode("Transpose", {softmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
  };

  auto check_optimized_graph_3 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_3,
                    check_optimized_graph_3,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_4 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* softmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& softmax_1 = builder.AddNode("Softmax", {transpose_1_out_0}, {softmax_1_out_0});
    softmax_1.AddAttribute("axis", (int64_t)2);
    auto& transpose_2 = builder.AddNode("Transpose", {softmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_4 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_4,
                    check_optimized_graph_4,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_5 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5, 6}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* softmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
    auto& softmax_1 = builder.AddNode("Softmax", {transpose_1_out_0}, {softmax_1_out_0});
    softmax_1.AddAttribute("axis", (int64_t)-2);
    auto& transpose_2 = builder.AddNode("Transpose", {softmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
  };

  auto check_optimized_graph_5 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_5,
                    check_optimized_graph_5,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_6 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5, 6}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* softmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2, 4});
    auto& softmax_1 = builder.AddNode("Softmax", {transpose_1_out_0}, {softmax_1_out_0});
    softmax_1.AddAttribute("axis", (int64_t)-3);
    auto& transpose_2 = builder.AddNode("Transpose", {softmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1, 4});
  };

  auto check_optimized_graph_6 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_6,
                    check_optimized_graph_6,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_7 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* hardmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Hardmax", {transpose_1_out_0}, {hardmax_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {hardmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_7 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_7,
                    check_optimized_graph_7,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_8 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* hardmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& hardmax_1 = builder.AddNode("Hardmax", {transpose_1_out_0}, {hardmax_1_out_0});
    hardmax_1.AddAttribute("axis", (int64_t)0);
    auto& transpose_2 = builder.AddNode("Transpose", {hardmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_8 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_8,
                    check_optimized_graph_8,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_9 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5, 6}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* hardmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
    auto& hardmax_1 = builder.AddNode("Hardmax", {transpose_1_out_0}, {hardmax_1_out_0});
    hardmax_1.AddAttribute("axis", (int64_t)-3);
    auto& transpose_2 = builder.AddNode("Transpose", {hardmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
  };

  auto check_optimized_graph_9 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_9,
                    check_optimized_graph_9,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_10 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* hardmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& hardmax_1 = builder.AddNode("Hardmax", {transpose_1_out_0}, {hardmax_1_out_0});
    hardmax_1.AddAttribute("axis", (int64_t)2);
    auto& transpose_2 = builder.AddNode("Transpose", {hardmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_10 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_10,
                    check_optimized_graph_10,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_11 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5, 6}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* hardmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
    auto& hardmax_1 = builder.AddNode("Hardmax", {transpose_1_out_0}, {hardmax_1_out_0});
    hardmax_1.AddAttribute("axis", (int64_t)-2);
    auto& transpose_2 = builder.AddNode("Transpose", {hardmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
  };

  auto check_optimized_graph_11 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_11,
                    check_optimized_graph_11,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_12 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5, 6}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* hardmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2, 4});
    auto& hardmax_1 = builder.AddNode("Hardmax", {transpose_1_out_0}, {hardmax_1_out_0});
    hardmax_1.AddAttribute("axis", (int64_t)-3);
    auto& transpose_2 = builder.AddNode("Transpose", {hardmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1, 4});
  };

  auto check_optimized_graph_12 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_12,
                    check_optimized_graph_12,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_13 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* logsoftmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("LogSoftmax", {transpose_1_out_0}, {logsoftmax_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {logsoftmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_13 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_13,
                    check_optimized_graph_13,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_14 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* logsoftmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& logsoftmax_1 = builder.AddNode("LogSoftmax", {transpose_1_out_0}, {logsoftmax_1_out_0});
    logsoftmax_1.AddAttribute("axis", (int64_t)0);
    auto& transpose_2 = builder.AddNode("Transpose", {logsoftmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_14 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_14,
                    check_optimized_graph_14,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_15 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5, 6}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* logsoftmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
    auto& logsoftmax_1 = builder.AddNode("LogSoftmax", {transpose_1_out_0}, {logsoftmax_1_out_0});
    logsoftmax_1.AddAttribute("axis", (int64_t)-3);
    auto& transpose_2 = builder.AddNode("Transpose", {logsoftmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
  };

  auto check_optimized_graph_15 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_15,
                    check_optimized_graph_15,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_16 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* logsoftmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& logsoftmax_1 = builder.AddNode("LogSoftmax", {transpose_1_out_0}, {logsoftmax_1_out_0});
    logsoftmax_1.AddAttribute("axis", (int64_t)2);
    auto& transpose_2 = builder.AddNode("Transpose", {logsoftmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_16 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_16,
                    check_optimized_graph_16,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_17 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5, 6}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* logsoftmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
    auto& logsoftmax_1 = builder.AddNode("LogSoftmax", {transpose_1_out_0}, {logsoftmax_1_out_0});
    logsoftmax_1.AddAttribute("axis", (int64_t)-2);
    auto& transpose_2 = builder.AddNode("Transpose", {logsoftmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0, 3, 2, 4});
  };

  auto check_optimized_graph_17 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_17,
                    check_optimized_graph_17,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);

  auto build_test_case_18 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, std::nullopt, {2, 3, 4, 5, 6}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* logsoftmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2, 4});
    auto& logsoftmax_1 = builder.AddNode("LogSoftmax", {transpose_1_out_0}, {logsoftmax_1_out_0});
    logsoftmax_1.AddAttribute("axis", (int64_t)-3);
    auto& transpose_2 = builder.AddNode("Transpose", {logsoftmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1, 4});
  };

  auto check_optimized_graph_18 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_18,
                    check_optimized_graph_18,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9,
                    /*per_sample_tolerance*/ 1e-07);
}

TEST(TransposeOptimizerTests, BroadcastOps) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* add_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Add", {transpose_1_out_0, input1_arg}, {add_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {add_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_2 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* mul_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Mul", {transpose_1_out_0, input1_arg}, {mul_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {mul_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_2 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_2,
                    check_optimized_graph_2,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_3 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* sub_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Sub", {transpose_1_out_0, input1_arg}, {sub_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {sub_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_3 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_3,
                    check_optimized_graph_3,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_4 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* div_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Div", {transpose_1_out_0, input1_arg}, {div_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {div_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_4 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_4,
                    check_optimized_graph_4,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_5 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* prelu_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("PRelu", {transpose_1_out_0, input1_arg}, {prelu_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {prelu_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_5 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_5,
                    check_optimized_graph_5,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_6 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* greater_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Greater", {transpose_1_out_0, input1_arg}, {greater_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {greater_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_6 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_6,
                    check_optimized_graph_6,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_7 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* less_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Less", {transpose_1_out_0, input1_arg}, {less_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {less_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_7 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_7,
                    check_optimized_graph_7,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_8 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* pow_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Pow", {transpose_1_out_0, input1_arg}, {pow_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {pow_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_8 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_8,
                    check_optimized_graph_8,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_9 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* max_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Max", {transpose_1_out_0, input1_arg}, {max_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {max_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_9 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_9,
                    check_optimized_graph_9,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_10 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* min_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Min", {transpose_1_out_0, input1_arg}, {min_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {min_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_10 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_10,
                    check_optimized_graph_10,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_11 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* mean_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Mean", {transpose_1_out_0, input1_arg}, {mean_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {mean_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_11 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_11,
                    check_optimized_graph_11,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_12 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* sum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Sum", {transpose_1_out_0, input1_arg}, {sum_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {sum_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_12 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_12,
                    check_optimized_graph_12,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_13 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* greaterorequal_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("GreaterOrEqual", {transpose_1_out_0, input1_arg}, {greaterorequal_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {greaterorequal_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_13 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_13,
                    check_optimized_graph_13,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_14 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* lessorequal_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("LessOrEqual", {transpose_1_out_0, input1_arg}, {lessorequal_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {lessorequal_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_14 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_14,
                    check_optimized_graph_14,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_15 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<int32_t>(builder, {{4, 6, 10}}, {4, 6, 10}, -1, 5);
    auto* input1_arg = MakeInput<int32_t>(builder, {{10, 4}}, {10, 4}, -1, 5);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* equal_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Equal", {transpose_1_out_0, input1_arg}, {equal_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {equal_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_15 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_15,
                    check_optimized_graph_15,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_16 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<bool>(builder, {{4, 6, 10}}, {4, 6, 10}, false, true);
    auto* input1_arg = MakeInput<bool>(builder, {{10, 4}}, {10, 4}, false, true);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* and_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("And", {transpose_1_out_0, input1_arg}, {and_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {and_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_16 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_16,
                    check_optimized_graph_16,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_17 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<bool>(builder, {{4, 6, 10}}, {4, 6, 10}, false, true);
    auto* input1_arg = MakeInput<bool>(builder, {{10, 4}}, {10, 4}, false, true);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* or_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Or", {transpose_1_out_0, input1_arg}, {or_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {or_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_17 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_17,
                    check_optimized_graph_17,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_18 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<bool>(builder, {{4, 6, 10}}, {4, 6, 10}, false, true);
    auto* input1_arg = MakeInput<bool>(builder, {{10, 4}}, {10, 4}, false, true);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* xor_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Xor", {transpose_1_out_0, input1_arg}, {xor_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {xor_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_18 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_18,
                    check_optimized_graph_18,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_19 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* mod_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& mod_1 = builder.AddNode("Mod", {transpose_1_out_0, input1_arg}, {mod_1_out_0});
    mod_1.AddAttribute("fmod", (int64_t)1);
    auto& transpose_2 = builder.AddNode("Transpose", {mod_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_19 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_19,
                    check_optimized_graph_19,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);

  auto build_test_case_20 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<uint32_t>(builder, {{4, 6, 10}}, {4, 6, 10}, 0, 5);
    auto* input1_arg = MakeInput<uint32_t>(builder, {{10, 4}}, {10, 4}, 0, 5);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* bitshift_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& bitshift_1 = builder.AddNode("BitShift", {transpose_1_out_0, input1_arg}, {bitshift_1_out_0});
    bitshift_1.AddAttribute("direction", "LEFT");
    auto& transpose_2 = builder.AddNode("Transpose", {bitshift_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_20 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_20,
                    check_optimized_graph_20,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12);
}

TEST(TransposeOptimizerTests, Where) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<bool>(builder, {{10, 4}}, {10, 4}, false, true);
    auto* input1_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input2_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* where_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input1_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& transpose_2 = builder.AddNode("Transpose", {input2_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Where", {input0_arg, transpose_1_out_0, transpose_2_out_0}, {where_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {where_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    size_t transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 9);
}




}  // namespace test
}  // namespace onnxruntime
