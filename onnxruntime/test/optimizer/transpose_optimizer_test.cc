// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <optional>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/utils.h"
#include "core/optimizer/transpose_optimization/onnx_transpose_optimization.h"
#include "core/optimizer/transpose_optimization/optimizer_api.h"
#include "core/optimizer/transpose_optimization/ort_optimizer_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/test_environment.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/internal_testing/internal_testing_execution_provider.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_utils.h"

namespace onnxruntime {
namespace test {

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
NodeArg* MakeInput(ModelTestBuilder& builder, const std::optional<std::vector<int64_t>>& input_shape,
                   const std::vector<int64_t>& value_shape, T min, T max) {
  auto node_arg = builder.MakeInput<T>(value_shape, min, max);
  SetNodeArgShape(node_arg, input_shape);
  return node_arg;
}

NodeArg* MakeInputBool(ModelTestBuilder& builder, const std::optional<std::vector<int64_t>>& input_shape,
                       const std::vector<int64_t>& value_shape) {
  auto node_arg = builder.MakeInputBool(value_shape);
  SetNodeArgShape(node_arg, input_shape);
  return node_arg;
}

template <typename T>
NodeArg* MakeInput(ModelTestBuilder& builder, const std::optional<std::vector<int64_t>>& input_shape,
                   const std::vector<int64_t>& value_shape, const std::vector<T>& data) {
  auto node_arg = builder.MakeInput<T>(value_shape, data);
  SetNodeArgShape(node_arg, input_shape);
  return node_arg;
}

int EstimateTransposeCost(const Graph& graph) {
  int cost = 0;
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

TEST(TransposeOptimizerTests, TestSplit) {
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
    split_1.AddAttribute("axis", (int64_t)1);
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      split_1.AddAttribute("num_outputs", static_cast<int64_t>(2));
    }
    auto& transpose_2 = builder.AddNode("Transpose", {split_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
    auto& transpose_3 = builder.AddNode("Transpose", {split_1_out_1}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSplitDefaultAxis) {
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
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      split_1.AddAttribute("num_outputs", static_cast<int64_t>(2));
    }
    auto& transpose_2 = builder.AddNode("Transpose", {split_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
    auto& transpose_3 = builder.AddNode("Transpose", {split_1_out_1}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSplitNegativeAxis) {
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
    split_1.AddAttribute("axis", (int64_t)1);
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      split_1.AddAttribute("num_outputs", static_cast<int64_t>(2));
    }
    auto& transpose_2 = builder.AddNode("Transpose", {split_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
    auto& transpose_3 = builder.AddNode("Transpose", {split_1_out_1}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestConcat) {
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
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestPad) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* pad_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* value = builder.MakeInitializer<float>({1}, {(float)2.3});
      auto* pads = builder.MakeInitializer<int64_t>({8}, {1, -2, 3, 4, 5, 6, 7, 8});
      auto& pad_1 = builder.AddNode("Pad", {transpose_1_out_0, pads, value}, {pad_1_out_0});
      pad_1.AddAttribute("mode", "constant");
    } else {
      auto& pad_1 = builder.AddNode("Pad", {transpose_1_out_0}, {pad_1_out_0});
      pad_1.AddAttribute("mode", "constant");
      pad_1.AddAttribute("value", (float)2.3);
      pad_1.AddAttribute("pads", std::vector<int64_t>{1, -2, 3, 4, 5, 6, 7, 8});
    }
    auto& transpose_2 = builder.AddNode("Transpose", {pad_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {10, 18});
}

TEST(TransposeOptimizerTests, TestPadOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({8}, {1, -2, 3, 4, 5, 6, 7, 8});
    auto* const_2 = builder.MakeInitializer<float>({}, {2.3f});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* pad_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& pad_1 = builder.AddNode("Pad", {transpose_1_out_0, const_1, const_2}, {pad_1_out_0});
    pad_1.AddAttribute("mode", "constant");
    auto& transpose_2 = builder.AddNode("Transpose", {pad_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestPadNonconst) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, {{8}}, {8}, {1, 0, 0, 1, 0, 0, -1, 1});
    auto* const_1 = builder.MakeInitializer<int64_t>({8}, {1, -2, 3, 4, 5, 6, 7, 8});
    auto* const_2 = builder.MakeInitializer<float>({}, {2.3f});
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
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {11, 18});
}

TEST(TransposeOptimizerTests, TestResize) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<float>({4}, {0.3f, 2.5f, 1.0f, 0.7f});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* resize_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();
    auto empty_arg = NodeArg("", nullptr);

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 11) {
      builder.AddNode("Resize", {transpose_1_out_0, &empty_arg, const_1}, {resize_1_out_0});
    } else {
      builder.AddNode("Resize", {transpose_1_out_0, const_1}, {resize_1_out_0});
    }
    auto& transpose_2 = builder.AddNode("Transpose", {resize_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    // need the level 2 TransposeOptimizer as pushing a Transpose through a Resize requires it to be
                    // assigned to the CPU EP first
                    TransformerLevel::Level2,
                    /*opset_version*/ {10, 18});
}

TEST(TransposeOptimizerTests, TestResizeOpset11) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<float>({8}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    auto* const_2 = builder.MakeInitializer<float>({4}, {0.3f, 2.5f, 1.0f, 0.7f});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* resize_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Resize", {transpose_1_out_0, const_1, const_2}, {resize_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {resize_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    // need the level 2 TransposeOptimizer as pushing a Transpose through a Resize requires it to be
                    // assigned to the CPU EP first
                    TransformerLevel::Level2,
                    /*opset_version*/ {11, 18});
}

TEST(TransposeOptimizerTests, TestResizeOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<float>({4}, {0.3f, 2.5f, 1.0f, 0.7f});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* resize_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();
    auto empty_arg = NodeArg("", nullptr);

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Resize", {transpose_1_out_0, &empty_arg, const_1}, {resize_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {resize_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    // need the level 2 TransposeOptimizer as pushing a Transpose through a Resize requires it to be
                    // assigned to the CPU EP first
                    TransformerLevel::Level2,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestResizeSizeRoi) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<float>({8}, {0.1f, 0.2f, 0.3f, 0.4f, 0.9f, 0.8f, 0.7f, 0.6f});
    auto* const_2 = builder.MakeInitializer<int64_t>({4}, {10, 9, 8, 7});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* resize_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();
    auto empty_arg = NodeArg("", nullptr);

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& resize_1 = builder.AddNode("Resize", {transpose_1_out_0, const_1, &empty_arg, const_2}, {resize_1_out_0});
    resize_1.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
    auto& transpose_2 = builder.AddNode("Transpose", {resize_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    // need the level 2 TransposeOptimizer as pushing a Transpose through a Resize requires it to be
                    // assigned to the CPU EP first
                    TransformerLevel::Level2,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestResizeRoiScalesZeroRank0) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<uint8_t>({1, 512, 512, 3},
                                             std::numeric_limits<uint8_t>::min(),
                                             std::numeric_limits<uint8_t>::max());
    auto* resize_in_roi = builder.MakeInitializer<float>({0}, {});
    auto* resize_in_scales = builder.MakeInitializer<float>({0}, {});
    auto* resize_in_sizes = builder.MakeInitializer<int64_t>({4}, {1, 256, 32, 32});

    auto* transpose1_out_transposed = builder.MakeIntermediate();
    auto* resize_out_Y = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input}, {transpose1_out_transposed});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Resize",
                    {transpose1_out_transposed, resize_in_roi, resize_in_scales, resize_in_sizes},
                    {resize_out_Y});
    auto& transpose_2 = builder.AddNode("Transpose", {resize_out_Y}, {output});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    // need the level 2 TransposeOptimizer as pushing a Transpose through a Resize requires it to be
                    // assigned to the CPU EP first
                    TransformerLevel::Level2,
                    {12, 18});
}

TEST(TransposeOptimizerTests, TestResizeNonconst) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* input1_arg = MakeInput<float>(builder, {{8}}, {8}, {0.1f, 0.2f, 0.3f, 0.4f, 0.9f, 0.8f, 0.7f, 0.6f});
    auto* input2_arg = MakeInput<float>(builder, {{4}}, {4}, {0.3f, 2.5f, 1.0f, 0.7f});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* resize_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& resize_1 = builder.AddNode("Resize", {transpose_1_out_0, input1_arg, input2_arg}, {resize_1_out_0});
    resize_1.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
    auto& transpose_2 = builder.AddNode("Transpose", {resize_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    // need the level 2 TransposeOptimizer as pushing a Transpose through a Resize requires it to be
                    // assigned to the CPU EP first
                    TransformerLevel::Level2,
                    /*opset_version*/ {11, 18});
}

TEST(TransposeOptimizerTests, TestResizeNonconstOpset13) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* input1_arg = MakeInput<float>(builder, {{8}}, {8}, {0.1f, 0.2f, 0.3f, 0.4f, 0.9f, 0.8f, 0.7f, 0.6f});
    auto* input2_arg = MakeInput<float>(builder, {{4}}, {4}, {0.3f, 2.5f, 1.0f, 0.7f});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* resize_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& resize_1 = builder.AddNode("Resize", {transpose_1_out_0, input1_arg, input2_arg}, {resize_1_out_0});
    resize_1.AddAttribute("coordinate_transformation_mode", "tf_crop_and_resize");
    auto& transpose_2 = builder.AddNode("Transpose", {resize_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    // need the level 2 TransposeOptimizer as pushing a Transpose through a Resize requires it to be
                    // assigned to the CPU EP first
                    TransformerLevel::Level2,
                    /*opset_version*/ {13, 18});
}

TEST(TransposeOptimizerTests, TestAdd) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<float>({}, {3.2f});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* add_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Add", {transpose_1_out_0, const_1}, {add_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {add_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestShape) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 6, -1}}, {4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* shape_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Shape", {transpose_1_out_0}, {shape_1_out_0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {7, 18});
}

TEST(TransposeOptimizerTests, TestShapeOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 6, -1}}, {4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* shape_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Shape", {transpose_1_out_0}, {shape_1_out_0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestShapeSliceNoStart) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 6, -1}}, {4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* shape_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& shape_1 = builder.AddNode("Shape", {transpose_1_out_0}, {shape_1_out_0});
    shape_1.AddAttribute("end", (int64_t)5);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestShapeSliceNegativeEnd) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 6, -1}}, {4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* shape_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& shape_1 = builder.AddNode("Shape", {transpose_1_out_0}, {shape_1_out_0});
    shape_1.AddAttribute("end", (int64_t)-1);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestShapeSliceNegativeStartNoEnd) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 6, -1}}, {4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* shape_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& shape_1 = builder.AddNode("Shape", {transpose_1_out_0}, {shape_1_out_0});
    shape_1.AddAttribute("start", (int64_t)-30);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestShapeSliceStartAndEnd) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 6, -1}}, {4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* shape_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& shape_1 = builder.AddNode("Shape", {transpose_1_out_0}, {shape_1_out_0});
    shape_1.AddAttribute("start", (int64_t)1);
    shape_1.AddAttribute("end", (int64_t)2);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestShapeSliceEmptyResult) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 6, -1}}, {4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* shape_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& shape_1 = builder.AddNode("Shape", {transpose_1_out_0}, {shape_1_out_0});
    shape_1.AddAttribute("start", (int64_t)2);
    shape_1.AddAttribute("end", (int64_t)1);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceSumKeepdimsTrue) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* init = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, init}, {reducesum_1_out_0});
      reducesum_1.AddAttribute("keepdims", (int64_t)1);
    } else {
      auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
      reducesum_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducesum_1.AddAttribute("keepdims", (int64_t)1);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducesum_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {7, 18},
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsTrue) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {7, 18},
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumKeepdimsFalse) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* init = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, init}, {reducesum_1_out_0});
      reducesum_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
      reducesum_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducesum_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducesum_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {7, 18},
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsFalse) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {7, 18},
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumKeepdimsTrueOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {0, -2});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
    auto& transpose_2 = builder.AddNode("Transpose", {reducesum_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsTrueOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsTrueNoopEmptyTrueOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)1);
    auto& transpose_2 = builder.AddNode("Transpose", {reducesum_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsTrueNoopEmptyFalseOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumNoAxesInputOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)1);
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)0);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumKeepdimsFalseOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({2}, {0, -2});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
    auto& transpose_2 = builder.AddNode("Transpose", {reducesum_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsFalseOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsFalseNoopEmptyTrueOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({0}, {});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0, const_1}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)1);
    auto& transpose_2 = builder.AddNode("Transpose", {reducesum_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsFalseNoopEmptyFalseOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumNoAxesInput_2Opset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("keepdims", (int64_t)0);
    reducesum_1.AddAttribute("noop_with_empty_axes", (int64_t)0);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestReduceSumNonconstKeepdimsTrueNoOptOpset13) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {-1});
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
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13);
}

TEST(TransposeOptimizerTests, TestReduceSumNonconstKeepdimsFalseNoOptOpset13) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {-1});
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 13);
}

TEST(TransposeOptimizerTests, TestReduceMaxKeepdimsTrue) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0, axes}, {reducemax_1_out_0});
      reducemax_1.AddAttribute("keepdims", (int64_t)1);
    } else {
      auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
      reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducemax_1.AddAttribute("keepdims", (int64_t)1);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducemax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceMaxKeepdimsTrueDefaultAxes) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("keepdims", (int64_t)1);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceMaxKeepdimsFalse) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0, axes}, {reducemax_1_out_0});
      reducemax_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
      reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducemax_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducemax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceMaxKeepdimsFalseDefaultAxes) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("keepdims", (int64_t)0);
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceMax) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      builder.AddNode("ReduceMax", {transpose_1_out_0, axes}, {reducemax_1_out_0});
    } else {
      auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
      reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducemax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceMaxDefaultAxes) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceLogSum) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducelogsum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducelogsum_1 = builder.AddNode("ReduceLogSum", {transpose_1_out_0, axes}, {reducelogsum_1_out_0});
      reducelogsum_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducelogsum_1 = builder.AddNode("ReduceLogSum", {transpose_1_out_0}, {reducelogsum_1_out_0});
      reducelogsum_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducelogsum_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducelogsum_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceLogSumExp) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducelogsumexp_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducelogsumexp_1 = builder.AddNode("ReduceLogSumExp", {transpose_1_out_0, axes}, {reducelogsumexp_1_out_0});
      reducelogsumexp_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducelogsumexp_1 = builder.AddNode("ReduceLogSumExp", {transpose_1_out_0}, {reducelogsumexp_1_out_0});
      reducelogsumexp_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducelogsumexp_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducelogsumexp_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceMax) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0, axes}, {reducemax_1_out_0});
      reducemax_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
      reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducemax_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducemax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceMean) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemean_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducemean_1 = builder.AddNode("ReduceMean", {transpose_1_out_0, axes}, {reducemean_1_out_0});
      reducemean_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducemean_1 = builder.AddNode("ReduceMean", {transpose_1_out_0}, {reducemean_1_out_0});
      reducemean_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducemean_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducemean_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceMin) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemin_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducemin_1 = builder.AddNode("ReduceMin", {transpose_1_out_0, axes}, {reducemin_1_out_0});
      reducemin_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducemin_1 = builder.AddNode("ReduceMin", {transpose_1_out_0}, {reducemin_1_out_0});
      reducemin_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducemin_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducemin_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceProd) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reduceprod_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reduceprod_1 = builder.AddNode("ReduceProd", {transpose_1_out_0, axes}, {reduceprod_1_out_0});
      reduceprod_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reduceprod_1 = builder.AddNode("ReduceProd", {transpose_1_out_0}, {reduceprod_1_out_0});
      reduceprod_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reduceprod_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reduceprod_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceSumSquare) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesumsquare_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* init = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducesumsquare_1 = builder.AddNode("ReduceSumSquare", {transpose_1_out_0, init}, {reducesumsquare_1_out_0});
      reducesumsquare_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducesumsquare_1 = builder.AddNode("ReduceSumSquare", {transpose_1_out_0}, {reducesumsquare_1_out_0});
      reducesumsquare_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducesumsquare_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducesumsquare_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceL1) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducel1_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducel1_1 = builder.AddNode("ReduceL1", {transpose_1_out_0, axes}, {reducel1_1_out_0});
      reducel1_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducel1_1 = builder.AddNode("ReduceL1", {transpose_1_out_0}, {reducel1_1_out_0});
      reducel1_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducel1_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducel1_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceL2) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducel2_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* axes = builder.MakeInitializer<int64_t>({2}, {0, -2});
      auto& reducel2_1 = builder.AddNode("ReduceL2", {transpose_1_out_0, axes}, {reducel2_1_out_0});
      reducel2_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducel2_1 = builder.AddNode("ReduceL2", {transpose_1_out_0}, {reducel2_1_out_0});
      reducel2_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
      reducel2_1.AddAttribute("keepdims", (int64_t)0);
    }
    auto& transpose_2 = builder.AddNode("Transpose", {reducel2_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSqueezeOpset7) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{1, -1, 1, 2}}, {1, 4, 1, 2}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* squeeze_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& squeeze_1 = builder.AddNode("Squeeze", {transpose_1_out_0}, {squeeze_1_out_0});
    squeeze_1.AddAttribute("axes", std::vector<int64_t>{0, 3});
    auto& transpose_2 = builder.AddNode("Transpose", {squeeze_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 7);
}

TEST(TransposeOptimizerTests, TestSqueezeOpset11) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{1, -1, 1, 2}}, {1, 4, 1, 2}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* squeeze_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& squeeze_1 = builder.AddNode("Squeeze", {transpose_1_out_0}, {squeeze_1_out_0});
    squeeze_1.AddAttribute("axes", std::vector<int64_t>{0, -1});
    auto& transpose_2 = builder.AddNode("Transpose", {squeeze_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 11);
}

TEST(TransposeOptimizerTests, TestSqueezeOpset15) {
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
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestSqueezeEmptyNoOpt) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{1, -1, 1, 2}}, {1, 4, 1, 2}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* squeeze_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Squeeze", {transpose_1_out_0}, {squeeze_1_out_0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {7, 18});
}

TEST(TransposeOptimizerTests, TestSqueezeEmptyNoOptOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{1, -1, 1, 2}}, {1, 4, 1, 2}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* squeeze_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Squeeze", {transpose_1_out_0}, {squeeze_1_out_0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestSqueezeNonconstNoOpt) {
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
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestUnsqueezeOpset7) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* unsqueeze_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& unsqueeze_1 = builder.AddNode("Unsqueeze", {transpose_1_out_0}, {unsqueeze_1_out_0});
    unsqueeze_1.AddAttribute("axes", std::vector<int64_t>{0, 4, 7, 5});
    auto& transpose_2 = builder.AddNode("Transpose", {unsqueeze_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 1, 3, 6, 4, 5, 2, 7});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 7);
}

TEST(TransposeOptimizerTests, TestUnsqueezeOpset11) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* unsqueeze_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& unsqueeze_1 = builder.AddNode("Unsqueeze", {transpose_1_out_0}, {unsqueeze_1_out_0});
    unsqueeze_1.AddAttribute("axes", std::vector<int64_t>{0, -4, -1, 5});
    auto& transpose_2 = builder.AddNode("Transpose", {unsqueeze_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 1, 3, 6, 4, 5, 2, 7});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 11);
}

TEST(TransposeOptimizerTests, TestUnsqueezeOpset15) {
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
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 1, 3, 6, 4, 5, 2, 7});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestUnsqueezeNonconstNoOpt) {
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
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 14);
}

TEST(TransposeOptimizerTests, TestSlice) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& slice_1 = builder.AddNode("Slice", {transpose_1_out_0}, {slice_1_out_0});
    slice_1.AddAttribute("axes", std::vector<int64_t>{2});
    slice_1.AddAttribute("starts", std::vector<int64_t>{1});
    slice_1.AddAttribute("ends", std::vector<int64_t>{-1});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 7);
}

TEST(TransposeOptimizerTests, TestSliceNoAxes) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& slice_1 = builder.AddNode("Slice", {transpose_1_out_0}, {slice_1_out_0});
    slice_1.AddAttribute("starts", std::vector<int64_t>{1, -3, 0, 0});
    slice_1.AddAttribute("ends", std::vector<int64_t>{2, 4, 10, 10});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 7);
}

TEST(TransposeOptimizerTests, TestSliceOpset15) {
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
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceNoAxesOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({4}, {1, -3, 0, 0});
    auto* const_2 = builder.MakeInitializer<int64_t>({4}, {2, 4, 10, 10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Slice", {transpose_1_out_0, const_1, const_2}, {slice_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceNegativeAxesInt32) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int32_t>({2}, {1, -3});
    auto* const_2 = builder.MakeInitializer<int32_t>({2}, {2, 4});
    auto* const_3 = builder.MakeInitializer<int32_t>({2}, {1, -2});
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
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceStepsInt32) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int32_t>({2}, {1, 4});
    auto* const_2 = builder.MakeInitializer<int32_t>({2}, {2, -3});
    auto* const_3 = builder.MakeInitializer<int32_t>({2}, {1, -2});
    auto* const_4 = builder.MakeInitializer<int32_t>({2}, {2, -1});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Slice", {transpose_1_out_0, const_1, const_2, const_3, const_4}, {slice_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceNegativeAxes) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceSteps) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceNonconstNoOpt) {
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
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceNonconstInt32NoOpt) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int32_t>(builder, {{2}}, {2}, {1, -2});
    auto* const_1 = builder.MakeInitializer<int32_t>({2}, {1, -3});
    auto* const_2 = builder.MakeInitializer<int32_t>({2}, {2, 4});
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
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceDefaultAxesNonconstStarts) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, {{4}}, {4}, {1, -3, 0, 0});
    auto* input2_arg = MakeInput<int64_t>(builder, {{4}}, {4}, {2, 4, 10, 10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Slice", {transpose_1_out_0, input1_arg, input2_arg}, {slice_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceDefaultAxesNonconstStartsUnknownLengthNoOpt) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int64_t>(builder, std::nullopt, {4}, {1, -3, 0, 0});
    auto* input2_arg = MakeInput<int64_t>(builder, std::nullopt, {4}, {2, 4, 10, 10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Slice", {transpose_1_out_0, input1_arg, input2_arg}, {slice_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceDefaultAxesNonconstStartsInt32) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int32_t>(builder, {{4}}, {4}, {1, -3, 0, 0});
    auto* input2_arg = MakeInput<int32_t>(builder, {{4}}, {4}, {2, 4, 10, 10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Slice", {transpose_1_out_0, input1_arg, input2_arg}, {slice_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSliceDefaultAxesNonconstStartsUnknownLengthInt32NoOpt) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({2, 4, 6, 5}, 0.0, 1.0);
    auto* input1_arg = MakeInput<int32_t>(builder, std::nullopt, {4}, {1, -3, 0, 0});
    auto* input2_arg = MakeInput<int32_t>(builder, std::nullopt, {4}, {2, 4, 10, 10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* slice_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Slice", {transpose_1_out_0, input1_arg, input2_arg}, {slice_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {slice_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestTile) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({4}, {1, 2, 1, 3});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* tile_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Tile", {transpose_1_out_0, const_1}, {tile_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {tile_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestTileNonconstReps) {
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
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestArgMinNoAxisKeepdimsTrue) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmin_1 = builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    argmin_1.AddAttribute("keepdims", (int64_t)1);
    auto& transpose_2 = builder.AddNode("Transpose", {argmin_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestArgMinNoAxisKeepdimsFalse) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmin_1 = builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    argmin_1.AddAttribute("keepdims", (int64_t)0);
    auto& transpose_2 = builder.AddNode("Transpose", {argmin_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestArgMinNoAxis) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {argmin_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestArgMinKeepdimsTrue) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmin_1 = builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    argmin_1.AddAttribute("axis", (int64_t)-2);
    argmin_1.AddAttribute("keepdims", (int64_t)1);
    auto& transpose_2 = builder.AddNode("Transpose", {argmin_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestArgMinKeepdimsFalse) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmin_1 = builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    argmin_1.AddAttribute("axis", (int64_t)-2);
    argmin_1.AddAttribute("keepdims", (int64_t)0);
    auto& transpose_2 = builder.AddNode("Transpose", {argmin_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestArgMin) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmin_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmin_1 = builder.AddNode("ArgMin", {transpose_1_out_0}, {argmin_1_out_0});
    argmin_1.AddAttribute("axis", (int64_t)-2);
    auto& transpose_2 = builder.AddNode("Transpose", {argmin_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestArgMax) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, 4, -1, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* argmax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& argmax_1 = builder.AddNode("ArgMax", {transpose_1_out_0}, {argmax_1_out_0});
    argmax_1.AddAttribute("axis", (int64_t)-2);
    argmax_1.AddAttribute("keepdims", (int64_t)1);
    auto& transpose_2 = builder.AddNode("Transpose", {argmax_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSoftmax) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestSoftmaxNoAxis) {
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
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestSoftmax_2) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestSoftmaxNoOptimization) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestSoftmaxNoOptimization_2) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestSoftmaxNoOptimization_3) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 12,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestSoftmaxNoAxisOpset15) {
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
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestSoftmaxOpset15) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestHardmaxAndLogSoftmaxNoAxis) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestHardmaxAndLogSoftmaxNoAxis_2) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
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

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15,
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
}

TEST(TransposeOptimizerTests, TestBroadcastOpsAdd) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* add_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Add", {transpose_1_out_0, transpose_2_out_0}, {add_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {add_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsMul) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* mul_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Mul", {transpose_1_out_0, transpose_2_out_0}, {mul_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {mul_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsSub) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* sub_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Sub", {transpose_1_out_0, transpose_2_out_0}, {sub_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {sub_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsDiv) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* div_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Div", {transpose_1_out_0, transpose_2_out_0}, {div_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {div_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

#if defined(_M_ARM64) && _MSC_VER >= 1930
  // Slight difference in Windows ARM64 VS 2022:
  // expected 19.3678 (419af143), got 19.3678 (419af144), diff: 1.90735e-06, tol=0
  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18},
                    /*per_sample_tolerance*/ 1e-07,
                    /*relative_per_sample_tolerance*/ 1e-06);
#else
  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
#endif  // defined(_M_ARM64) && _MSC_VER >= 1930
}

TEST(TransposeOptimizerTests, TestBroadcastOpsPRelu) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* prelu_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("PRelu", {transpose_1_out_0, transpose_2_out_0}, {prelu_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {prelu_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsGreater) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* greater_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Greater", {transpose_1_out_0, transpose_2_out_0}, {greater_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {greater_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsLess) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* less_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Less", {transpose_1_out_0, transpose_2_out_0}, {less_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {less_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsPow) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* pow_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Pow", {transpose_1_out_0, transpose_2_out_0}, {pow_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {pow_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsMax) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* max_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Max", {transpose_1_out_0, transpose_2_out_0}, {max_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {max_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsMin) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* min_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Min", {transpose_1_out_0, transpose_2_out_0}, {min_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {min_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsMean) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* mean_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Mean", {transpose_1_out_0, transpose_2_out_0}, {mean_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {mean_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsSum) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* sum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Sum", {transpose_1_out_0, transpose_2_out_0}, {sum_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {sum_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsGreaterOrEqual) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* greaterorequal_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("GreaterOrEqual", {transpose_1_out_0, transpose_2_out_0}, {greaterorequal_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {greaterorequal_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsLessOrEqual) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* lessorequal_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("LessOrEqual", {transpose_1_out_0, transpose_2_out_0}, {lessorequal_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {lessorequal_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsEqual) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<int32_t>(builder, {{6, 4, 10}}, {6, 4, 10}, -1, 5);
    auto* input1_arg = MakeInput<int32_t>(builder, {{4, 10}}, {4, 10}, -1, 5);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* equal_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Equal", {transpose_1_out_0, transpose_2_out_0}, {equal_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {equal_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsAnd) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInputBool(builder, {{6, 4, 10}}, {6, 4, 10});
    auto* input1_arg = MakeInputBool(builder, {{4, 10}}, {4, 10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* and_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("And", {transpose_1_out_0, transpose_2_out_0}, {and_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {and_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsOr) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInputBool(builder, {{6, 4, 10}}, {6, 4, 10});
    auto* input1_arg = MakeInputBool(builder, {{4, 10}}, {4, 10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* or_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Or", {transpose_1_out_0, transpose_2_out_0}, {or_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {or_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsXor) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInputBool(builder, {{6, 4, 10}}, {6, 4, 10});
    auto* input1_arg = MakeInputBool(builder, {{4, 10}}, {4, 10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* xor_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Xor", {transpose_1_out_0, transpose_2_out_0}, {xor_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {xor_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsMod) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({6, 4, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* mod_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    auto& mod_1 = builder.AddNode("Mod", {transpose_1_out_0, transpose_2_out_0}, {mod_1_out_0});
    mod_1.AddAttribute("fmod", (int64_t)1);
    auto& transpose_3 = builder.AddNode("Transpose", {mod_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastOpsBitShift) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<uint32_t>(builder, {{6, 4, 10}}, {6, 4, 10}, 0, 5);
    auto* input1_arg = MakeInput<uint32_t>(builder, {{4, 10}}, {4, 10}, 0, 5);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* bitshift_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    auto& bitshift_1 = builder.AddNode("BitShift", {transpose_1_out_0, transpose_2_out_0}, {bitshift_1_out_0});
    bitshift_1.AddAttribute("direction", "LEFT");
    auto& transpose_3 = builder.AddNode("Transpose", {bitshift_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestWhere) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInputBool(builder, {{4, 6, 10}}, {4, 6, 10});
    auto* input1_arg = builder.MakeInput<float>(std::vector<int64_t>{}, 0.0, 1.0);
    auto* input2_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* where_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& transpose_2 = builder.AddNode("Transpose", {input2_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Where", {transpose_1_out_0, input1_arg, transpose_2_out_0}, {where_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {where_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

// Utility function that runs TransformerTester for the graph Transpose -> QuantizeLinear -> Transpose.
// Expects the Tranpose nodes to cancel.
template <typename QuantType>
static void RunQuantizeLinearTestCase(const std::optional<std::vector<int64_t>>& zp_input_shape,
                                      const std::vector<int64_t>& zp_value_shape,
                                      std::optional<ONNX_NAMESPACE::AttributeProto> axis,
                                      const std::string& q_domain = "") {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0.0, 1.0);

    NodeArg* scale_arg = nullptr;
    NodeArg* zero_point_arg = nullptr;

    if (zp_value_shape.empty()) {  // Per-tensor quantization
      QuantType zp = (qmax + qmin) / 2;
      scale_arg = MakeInput<float>(builder, zp_input_shape, zp_value_shape, {0.05f});
      zero_point_arg = MakeInput<QuantType>(builder, zp_input_shape, zp_value_shape, {zp});
    } else {  // Per-axis quantization
      scale_arg = MakeInput<float>(builder, zp_input_shape, zp_value_shape, 0.0f, 1.0f);
      zero_point_arg = MakeInput<QuantType>(builder, zp_input_shape, zp_value_shape, qmin, qmax);
    }
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* quantizelinear_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& quantizelinear_1 = builder.AddNode("QuantizeLinear", {transpose_1_out_0, scale_arg, zero_point_arg},
                                             {quantizelinear_1_out_0}, q_domain);

    if (axis.has_value()) {
      quantizelinear_1.AddAttributeProto(*axis);
    }

    auto& transpose_2 = builder.AddNode("Transpose", {quantizelinear_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph = [](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestQuantizeLinearScalar) {
  std::optional<std::vector<int64_t>> zp_input_shape = std::vector<int64_t>{};
  std::vector<int64_t> zp_value_shape{};
  std::optional<ONNX_NAMESPACE::AttributeProto> empty_axis;  // No axis value.

  RunQuantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, empty_axis, kOnnxDomain);

#if !defined(DISABLE_CONTRIB_OPS)
  // Use com.microsoft.QuantizeLinear op.
  RunQuantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, empty_axis, kMSDomain);
  RunQuantizeLinearTestCase<uint16_t>(zp_input_shape, zp_value_shape, empty_axis, kMSDomain);
  RunQuantizeLinearTestCase<int16_t>(zp_input_shape, zp_value_shape, empty_axis, kMSDomain);
#endif
}

TEST(TransposeOptimizerTests, TestQuantizeLinearScalarIgnoreAxis) {
  std::optional<std::vector<int64_t>> zp_input_shape = std::vector<int64_t>{};
  std::vector<int64_t> zp_value_shape{};
  auto ignored_axis = utils::MakeAttribute("axis", static_cast<int64_t>(10));  // Should be ignored for per-tensor Q

  RunQuantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, ignored_axis, kOnnxDomain);

#if !defined(DISABLE_CONTRIB_OPS)
  // Use com.microsoft.QuantizeLinear op.
  RunQuantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, ignored_axis, kMSDomain);
  RunQuantizeLinearTestCase<uint16_t>(zp_input_shape, zp_value_shape, ignored_axis, kMSDomain);
  RunQuantizeLinearTestCase<int16_t>(zp_input_shape, zp_value_shape, ignored_axis, kMSDomain);
#endif
}

TEST(TransposeOptimizerTests, TestQuantizeLinearVector) {
  std::optional<std::vector<int64_t>> zp_input_shape = std::vector<int64_t>{-1};
  std::vector<int64_t> zp_value_shape = {2};
  auto axis = utils::MakeAttribute("axis", static_cast<int64_t>(0));

  RunQuantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, axis, kOnnxDomain);

#if !defined(DISABLE_CONTRIB_OPS)
  // Use com.microsoft.QuantizeLinear op.
  RunQuantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, axis, kMSDomain);
  RunQuantizeLinearTestCase<uint16_t>(zp_input_shape, zp_value_shape, axis, kMSDomain);
  RunQuantizeLinearTestCase<int16_t>(zp_input_shape, zp_value_shape, axis, kMSDomain);
#endif
}

TEST(TransposeOptimizerTests, TestQuantizeLinearVectorUnknownRank) {
  std::optional<std::vector<int64_t>> zp_unknown_shape;  // Empty shape
  std::vector<int64_t> zp_value_shape = {3};
  auto axis = utils::MakeAttribute("axis", static_cast<int64_t>(1));

  RunQuantizeLinearTestCase<uint8_t>(zp_unknown_shape, zp_value_shape, axis, kOnnxDomain);

#if !defined(DISABLE_CONTRIB_OPS)
  // Use com.microsoft.QuantizeLinear op.
  RunQuantizeLinearTestCase<uint8_t>(zp_unknown_shape, zp_value_shape, axis, kMSDomain);
  RunQuantizeLinearTestCase<uint16_t>(zp_unknown_shape, zp_value_shape, axis, kMSDomain);
  RunQuantizeLinearTestCase<int16_t>(zp_unknown_shape, zp_value_shape, axis, kMSDomain);
#endif
}

TEST(TransposeOptimizerTests, TestQuantizeLinearScalarOpset10) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* input1_arg = MakeInput<float>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {2.3f});
    auto* input2_arg = MakeInput<uint8_t>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* quantizelinear_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("QuantizeLinear", {transpose_1_out_0, input1_arg, input2_arg}, {quantizelinear_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {quantizelinear_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 10);
}

// Utility function that runs TransformerTester for the graph Transpose -> DequantizeLinear -> Transpose.
// Expects the Tranpose nodes to cancel.
template <typename QuantType>
static void RunDequantizeLinearTestCase(const std::optional<std::vector<int64_t>>& zp_input_shape,
                                        const std::vector<int64_t>& zp_value_shape,
                                        std::optional<ONNX_NAMESPACE::AttributeProto> axis,
                                        const std::string& q_domain = "") {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    constexpr QuantType qmin = std::numeric_limits<QuantType>::min();
    constexpr QuantType qmax = std::numeric_limits<QuantType>::max();

    auto* input0_arg = MakeInput<QuantType>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, qmin, qmax);

    NodeArg* scale_arg = nullptr;
    NodeArg* zero_point_arg = nullptr;

    if (zp_value_shape.empty()) {  // Per-tensor quantization
      QuantType zp = (qmax + qmin) / 2;
      scale_arg = MakeInput<float>(builder, zp_input_shape, zp_value_shape, {0.05f});
      zero_point_arg = MakeInput<QuantType>(builder, zp_input_shape, zp_value_shape, {zp});
    } else {  // Per-axis quantization
      scale_arg = MakeInput<float>(builder, zp_input_shape, zp_value_shape, 0.0f, 1.0f);
      zero_point_arg = MakeInput<QuantType>(builder, zp_input_shape, zp_value_shape, qmin, qmax);
    }
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* dequantizelinear_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& dequantizelinear_1 = builder.AddNode("DequantizeLinear", {transpose_1_out_0, scale_arg, zero_point_arg},
                                               {dequantizelinear_1_out_0}, q_domain);

    if (axis.has_value()) {
      dequantizelinear_1.AddAttributeProto(*axis);
    }

    auto& transpose_2 = builder.AddNode("Transpose", {dequantizelinear_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph = [](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestDequantizeLinearScalarIgnoreAxis) {
  std::optional<std::vector<int64_t>> zp_input_shape = std::vector<int64_t>{};
  std::vector<int64_t> zp_value_shape{};
  auto ignored_axis = utils::MakeAttribute("axis", static_cast<int64_t>(10));  // Should be ignored for per-tensor Q

  RunDequantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, ignored_axis, kOnnxDomain);
#if !defined(DISABLE_CONTRIB_OPS)
  // Use com.microsoft.DequantizeLinear ops
  RunDequantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, ignored_axis, kMSDomain);
  RunDequantizeLinearTestCase<uint16_t>(zp_input_shape, zp_value_shape, ignored_axis, kMSDomain);
  RunDequantizeLinearTestCase<int16_t>(zp_input_shape, zp_value_shape, ignored_axis, kMSDomain);
#endif
}

TEST(TransposeOptimizerTests, TestDequantizeLinearVector) {
  std::optional<std::vector<int64_t>> zp_input_shape = std::vector<int64_t>{2};
  std::vector<int64_t> zp_value_shape = {2};
  auto axis = utils::MakeAttribute("axis", static_cast<int64_t>(-4));

  RunDequantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, axis, kOnnxDomain);
#if !defined(DISABLE_CONTRIB_OPS)
  // Use com.microsoft.DequantizeLinear ops
  RunDequantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, axis, kMSDomain);
  RunDequantizeLinearTestCase<uint16_t>(zp_input_shape, zp_value_shape, axis, kMSDomain);
  RunDequantizeLinearTestCase<int16_t>(zp_input_shape, zp_value_shape, axis, kMSDomain);
#endif
}

TEST(TransposeOptimizerTests, TestDequantizeLinearNoAxis) {
  std::optional<std::vector<int64_t>> zp_input_shape = std::vector<int64_t>{};
  std::vector<int64_t> zp_value_shape{};
  std::optional<ONNX_NAMESPACE::AttributeProto> no_axis;  // Empty axis value will not be set.

  RunDequantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, no_axis, kOnnxDomain);
#if !defined(DISABLE_CONTRIB_OPS)
  // Use com.microsoft.DequantizeLinear ops
  RunDequantizeLinearTestCase<uint8_t>(zp_input_shape, zp_value_shape, no_axis, kMSDomain);
  RunDequantizeLinearTestCase<uint16_t>(zp_input_shape, zp_value_shape, no_axis, kMSDomain);
  RunDequantizeLinearTestCase<int16_t>(zp_input_shape, zp_value_shape, no_axis, kMSDomain);
#endif
}

TEST(TransposeOptimizerTests, TestCast) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<int32_t>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, -1, 5);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* cast_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& cast_1 = builder.AddNode("Cast", {transpose_1_out_0}, {cast_1_out_0});
    cast_1.AddAttribute("to", (int64_t)1);
    auto& transpose_2 = builder.AddNode("Transpose", {cast_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestBroadcastReusedInputs) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, -1, 3, 4}}, {1, 2, 3, 4}, 0.0, 1.0);
    auto* input1_arg = MakeInput<float>(builder, {{-1, 3}}, {2, 3}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<float>({2, 3}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* sum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Sum", {transpose_1_out_0, const_1, input1_arg, const_1, transpose_1_out_0}, {sum_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {sum_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
    auto& transpose_3 = builder.AddNode("Transpose", {sum_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestTransposeGraphOutput) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, -1, -1, -1}}, {1, 2, 3, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeOutput();
    auto* transpose_2_out_0 = builder.MakeOutput();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& transpose_2 = builder.AddNode("Transpose", {transpose_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
    auto& transpose_3 = builder.AddNode("Transpose", {transpose_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1, 3});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    // One transpose is a graph output and cannot be removed. Other doesn't match perm. Cost 12 -> 8.
    EXPECT_EQ(transpose_cost, 8);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestSimpleReshapeAsTranspose) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({1, 1, 1, 3}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({1, 3}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({4}, {1, 1, 1, 3});
    auto* mul_1_out_0 = builder.MakeOutput();
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reshape_1_out_0 = builder.MakeIntermediate();
    auto* identity_1_out_0 = builder.MakeOutput();

    builder.AddNode("Mul", {input0_arg, input1_arg}, {mul_1_out_0});
    auto& transpose_1 = builder.AddNode("Transpose", {mul_1_out_0}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Reshape", {transpose_1_out_0, const_1}, {reshape_1_out_0});
    builder.AddNode("Identity", {reshape_1_out_0}, {identity_1_out_0});
  };

  auto check_optimized_graph = [&](InferenceSessionWrapper& session) {
    // Transpose cancels with the following reshape node so both the nodes
    // should be removed
    std::map<std::string, int> op_to_count = CountOpsInGraph(session.GetGraph());
    ASSERT_EQ(op_to_count["Mul"], 1);
    ASSERT_EQ(op_to_count["Transpose"], 0);
    ASSERT_EQ(op_to_count["Reshape"], 0);
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestReshapeAsTransposeGraphOutput) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({1, 1, 1, 3}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({1, 3}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({4}, {1, 1, 1, 3});
    auto* mul_1_out_0 = builder.MakeIntermediate();
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reshape_1_out_0 = builder.MakeOutput();

    builder.AddNode("Mul", {input0_arg, input1_arg}, {mul_1_out_0});
    auto& transpose_1 = builder.AddNode("Transpose", {mul_1_out_0}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Reshape", {transpose_1_out_0, const_1}, {reshape_1_out_0});
  };

  auto check_optimized_graph = [&](InferenceSessionWrapper& session) {
    // Transpose cancels with the following reshape node so both the nodes
    // should be removed
    std::map<std::string, int> op_to_count = CountOpsInGraph(session.GetGraph());
    ASSERT_EQ(op_to_count["Mul"], 1);
    ASSERT_EQ(op_to_count["Transpose"], 0);
    ASSERT_EQ(op_to_count["Reshape"], 0);
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

enum class TransposeReshapeResult {
  kUnchanged,  // nodes cannot be merged or removed
  kMerge,      // merge Transpose and Reshape into single Transpose
  kCancel      // Transpose and Reshape cancel each other and both can be dropped
};

static void TestTransposeReshape(const std::vector<int64_t>& input_shape,    // model and Transpose input shape
                                 const std::vector<int64_t>& perms,          // perms for Transpose node
                                 const std::vector<int64_t>& reshape_shape,  // shape for Reshape node
                                 TransposeReshapeResult expected_result,
                                 const std::vector<int64_t>& merged_perms = {},  // expected perms num_transpose == 1
                                 bool allow_zero = false) {                      // Reshape 'allowzero' value
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input_shape, 0.0, 1.0);
    auto* mul_arg1 = builder.MakeInput<float>({1}, 0.0, 1.0);
    auto* reshape_shape_value = builder.MakeInitializer<int64_t>({int64_t(reshape_shape.size())}, reshape_shape);

    auto* mul_out_0 = builder.MakeOutput();
    auto* transpose_out_0 = builder.MakeIntermediate();
    auto* reshape_out_0 = builder.MakeIntermediate();
    auto* identity_out_0 = builder.MakeOutput();

    builder.AddNode("Mul", {input_arg, mul_arg1}, {mul_out_0});  // node so Transpose isn't consuming graph input

    auto& transpose_1 = builder.AddNode("Transpose", {mul_out_0}, {transpose_out_0});
    transpose_1.AddAttribute("perm", perms);

    auto& reshape = builder.AddNode("Reshape", {transpose_out_0, reshape_shape_value}, {reshape_out_0});
    if (allow_zero) {
      reshape.AddAttribute("allowzero", int64_t(1));
    }

    builder.AddNode("Identity", {reshape_out_0}, {identity_out_0});  // node so Reshape isn't graph output
  };

  auto check_optimized_graph = [&](InferenceSessionWrapper& session) {
    const auto& graph = session.GetGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

    if (expected_result == TransposeReshapeResult::kCancel) {
      ASSERT_EQ(op_to_count["Transpose"], 0);
      ASSERT_EQ(op_to_count["Reshape"], 0);
    } else if (expected_result == TransposeReshapeResult::kMerge) {
      ASSERT_EQ(op_to_count["Transpose"], 1);
      ASSERT_EQ(op_to_count["Reshape"], 0);

      // find Transpose node and check perms
      const auto& nodes = graph.Nodes();
      const Node& transpose = *std::find_if(nodes.begin(), nodes.end(),
                                            [](const auto& node) { return node.OpType() == "Transpose"; });

      ProtoHelperNodeContext proto_helper_ctx(transpose);
      OpNodeProtoHelper<ProtoHelperNodeContext> proto_helper(&proto_helper_ctx);
      std::vector<int64_t> actual_perms;
      ASSERT_STATUS_OK(proto_helper.GetAttrs<int64_t>("perm", actual_perms));
      ASSERT_THAT(actual_perms, testing::ContainerEq(merged_perms));
    } else {
      ASSERT_EQ(op_to_count["Transpose"], 1);
      ASSERT_EQ(op_to_count["Reshape"], 1);
    }
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
}

// Transpose -> Reshape can be merged if the Reshape could also be expressed as a Transpose due to not changing the
// order or size of dims with value > 1.
TEST(TransposeOptimizerTests, TestReshapeCanMerge) {
  // 3D
  TestTransposeReshape({4, 1, 3}, {2, 0, 1},  // transpose input shape and perms. output shape {3, 4, 1}
                       {1, 3, 4},             // reshape 'shape'
                       TransposeReshapeResult::kMerge,
                       {1, 2, 0});  // perms required to merge

  // essentially the same test but with the 2 non-1 dims having the same value
  TestTransposeReshape({4, 1, 4}, {2, 0, 1},  // transpose input shape and perms. output shape {4, 4, 1}
                       {1, 4, 4},             // reshape 'shape'
                       TransposeReshapeResult::kMerge,
                       {1, 2, 0});  // perms required to merge

  // 4D - various combinations to exercise different paths in the perms calculation
  TestTransposeReshape({1, 6, 1, 5}, {3, 0, 1, 2},  // transpose input shape and perms. output shape {5, 1, 6, 1}
                       {1, 1, 5, 6},                // reshape 'shape'. equiv. transpose perms == 1,3,0,2
                       TransposeReshapeResult::kMerge,
                       {0, 2, 3, 1});  // merged perms of 3,0,1,2 followed by 1,3,0,2

  TestTransposeReshape({1, 6, 1, 5}, {2, 3, 1, 0},  // transpose input shape and perms. output shape {1, 5, 6, 1}
                       {5, 6, 1, 1},                // reshape 'shape'. equiv. transpose perms == 1,2,0,3
                       TransposeReshapeResult::kMerge,
                       {3, 1, 2, 0});  // merged perms of 2,3,1,0 followed by 1,2,0,3

  TestTransposeReshape({1, 6, 1, 5}, {3, 0, 2, 1},  // transpose input shape and perms. output shape {5, 1, 1, 6}
                       {1, 5, 6, 1},                // reshape 'shape'. equiv. transpose perms == 1,0,3,1
                       TransposeReshapeResult::kMerge,
                       {0, 3, 1, 2});  // merged perms of 3,0,2,1 followed by 1,0,3,1

  TestTransposeReshape({1, 6, 1, 5}, {2, 3, 0, 1},  // transpose input shape and perms. output shape {1, 5, 1, 6}
                       {5, 1, 6, 1},                // reshape 'shape'. equiv. transpose perms == 1,0,3,2
                       TransposeReshapeResult::kMerge,
                       {3, 2, 1, 0});  // merged perms of 2,3,0,1 followed by 1,0,3,2
}

// Transpose -> Reshape cancel each other out if Reshape can be expressed as a Transpose due to not changing the
// order or size of dims with value > 1, and the Transpose input shape is equal to the Reshape output shape.
TEST(TransposeOptimizerTests, TestReshapeCancelsTranspose) {
  // 2D
  TestTransposeReshape({1, 8}, {1, 0},  // transpose input shape and perms. output shape {8, 1}
                       {1, 8},          // reshape 'shape'
                       TransposeReshapeResult::kCancel);
  // 3D
  TestTransposeReshape({2, 1, 4}, {1, 0, 2},  // transpose input shape and perms. output shape {1, 2, 4}
                       {2, 1, 4},             // reshape 'shape'
                       TransposeReshapeResult::kCancel);

  // 4D
  TestTransposeReshape({1, 2, 1, 3}, {0, 2, 1, 3},  // transpose input shape and perms. output shape {1, 1, 2, 3}
                       {1, 2, 1, 3},                // reshape 'shape'
                       TransposeReshapeResult::kCancel);
}

// test some Reshape combinations that cannot be treated as a Transpose
TEST(TransposeOptimizerTests, TestReshapeCantMerge) {
  // rank mismatch
  TestTransposeReshape({2, 1, 4}, {1, 0, 2},  // transpose input shape and perms. output shape {1, 2, 4}
                       {2, 4},                // reshape 'shape'
                       TransposeReshapeResult::kUnchanged);

  // reshape can't be treated as a Transpose as a dim size changes
  TestTransposeReshape({1, 4, 4, 5}, {0, 3, 1, 2},  // transpose input shape and perms. output shape {1, 5, 4, 4}
                       {1, 5, 2, 8},                // reshape 'shape'
                       TransposeReshapeResult::kUnchanged);

  // reshape can't be treated as a Transpose as the order of dims with value > 1 changes
  TestTransposeReshape({1, 4, 8, 3}, {0, 3, 1, 2},  // transpose input shape and perms. output shape {1, 3, 4, 8}
                       {1, 3, 8, 4},                // reshape 'shape'
                       TransposeReshapeResult::kUnchanged);
}

// test Reshape with allow zero.
// If allow_zero is false, the dim value is inferred from the matching input dim.
// If allow_zero is true, a zero is maintained as-is (resulting in an output of empty data)
TEST(TransposeOptimizerTests, TestReshapeWithZero) {
  // First 2 tests have a 0 in the Reshape `shape` input. One can be merged, one can't.
  TestTransposeReshape({1, 5, 7, 1}, {0, 3, 2, 1},  // transpose input shape and perms. output shape {1, 1, 7, 5}
                       {1, 7, 1, 0},                // reshape 'shape'. the '0' should == 5 from the transpose output.
                                                    // equiv. transpose perms == 0,2,1,3
                       TransposeReshapeResult::kMerge,
                       {0, 2, 3, 1},  // result should be {1, 7, 1, 5}. merged perms of 0,3,2,1 followed by 0,2,1,3
                       false);        // allow_zero

  TestTransposeReshape({1, 4, 8, 3}, {0, 3, 1, 2},  // transpose input shape and perms. output shape {1, 3, 4, 8}
                       {1, 0, 8, 4},                // reshape 'shape'. '3' is inferred but last 2 dims swap
                       TransposeReshapeResult::kUnchanged,
                       {},
                       false);

  // Next test has a 0 in the Reshape `shape` input.
  TestTransposeReshape({1, 5, 7, 0}, {0, 2, 1, 3},  // transpose input shape and perms. output shape {1, 7, 5, 0}
                       {7, 5, 1, 0},                // reshape 'shape'. the '0' should be kept. we can't merge as a
                       TransposeReshapeResult::kMerge,
                       {2, 1, 0, 3},  // result should be {7, 5, 1, 0}. merged perms of 0,2,1,3 followed by 1,2,0,3
                       true);         // allow_zero

  TestTransposeReshape({1, 5, 7, 0}, {0, 2, 1, 3},  // transpose input shape and perms. output shape {1, 7, 5, 0}
                       {0, 7, 5, 1},                // reshape 'shape'. the '0' should be kept.
                       TransposeReshapeResult::kUnchanged,
                       {},
                       true);  // allow_zero
}

// test Reshape with an inferred dim due to value of -1
// test valid (inferred dim is 1:1 with existing) and invalid (inferred size differs)
TEST(TransposeOptimizerTests, TestReshapeWithMinusOne) {
  TestTransposeReshape({1, 5, 7, 1}, {0, 3, 2, 1},  // transpose input shape and perms. output shape {1, 1, 7, 5}
                       {1, 7, 1, -1},               // reshape 'shape'. -1 -> 5
                                                    // equiv. transpose perms == 0,2,1,3
                       TransposeReshapeResult::kMerge,
                       {0, 2, 3, 1});  // result should be {1, 7, 1, 5}. merged perms of 0,3,2,1 followed by 0,2,1,3

  // -1 dim changes size
  TestTransposeReshape({1, 8, 4, 1}, {0, 3, 2, 1},  // transpose input shape and perms. output shape {1, 1, 4, 8}
                       {1, -1, 2, 8},               // reshape 'shape'. -1 -> 2 which is incompatible
                       TransposeReshapeResult::kUnchanged);
}

TEST(TransposeOptimizerTests, TestCancelingNodesGraphOutputs) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({1, 1, 1, 3}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({1, 3}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<int64_t>({4}, {1, 1, 1, 3});
    auto* mul_1_out_0 = builder.MakeOutput();
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reshape_1_out_0 = builder.MakeOutput();

    builder.AddNode("Mul", {input0_arg, input1_arg}, {mul_1_out_0});
    auto& transpose_1 = builder.AddNode("Transpose", {mul_1_out_0}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Reshape", {transpose_1_out_0, const_1}, {reshape_1_out_0});
  };

  auto check_optimized_graph = [&](InferenceSessionWrapper& session) {
    // Transpose cancels with the following reshape node so both the nodes
    // should be removed
    std::map<std::string, int> op_to_count = CountOpsInGraph(session.GetGraph());
    ASSERT_EQ(op_to_count["Mul"], 1);
    ASSERT_EQ(op_to_count["Transpose"], 0);
    ASSERT_EQ(op_to_count["Reshape"], 0);
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestNonCancelingReshapeDueToNonConstShape) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({1, 1, 1, 3}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({1, 3}, 0.0, 1.0);
    auto* input3_arg = builder.MakeInput<int64_t>({4}, {1, 1, 1, 3});
    auto* mul_1_out_0 = builder.MakeOutput();
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reshape_1_out_0 = builder.MakeIntermediate();
    auto* identity_1_out_0 = builder.MakeOutput();

    builder.AddNode("Mul", {input0_arg, input1_arg}, {mul_1_out_0});
    auto& transpose_1 = builder.AddNode("Transpose", {mul_1_out_0}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Reshape", {transpose_1_out_0, input3_arg}, {reshape_1_out_0});
    builder.AddNode("Identity", {reshape_1_out_0}, {identity_1_out_0});
  };

  auto check_optimized_graph = [&](InferenceSessionWrapper& session) {
    // Transpose on mul output cannot be removed since reshape's shape input
    // is not const
    std::map<std::string, int> op_to_count = CountOpsInGraph(session.GetGraph());
    ASSERT_EQ(op_to_count["Mul"], 1);
    ASSERT_EQ(op_to_count["Transpose"], 1);
    ASSERT_EQ(op_to_count["Reshape"], 1);

    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 1);
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestPushBroadcastUnsqueezeTranspose) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* relu_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* relu_2_out_0 = builder.MakeIntermediate();
    auto* mul_1_out_0 = builder.MakeIntermediate();
    auto* transpose_3_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Relu", {transpose_1_out_0}, {relu_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 0});
    builder.AddNode("Relu", {transpose_2_out_0}, {relu_2_out_0});
    builder.AddNode("Mul", {relu_1_out_0, relu_2_out_0}, {mul_1_out_0});
    auto& transpose_3 = builder.AddNode("Transpose", {mul_1_out_0}, {transpose_3_out_0});
    transpose_3.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    // A transpose on rank 2 input of Mul can't be removed but is combined with the other transpose. Cost 8 -> 2.
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestOptimizeTowardsTranspose) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* mul_1_out_0 = builder.MakeIntermediate();
    auto* relu_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Mul", {transpose_1_out_0, input1_arg}, {mul_1_out_0});
    builder.AddNode("Relu", {mul_1_out_0}, {relu_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {relu_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{2, 0, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    // Pushing Transpose through Mul creates transpose on 2nd input. Cost 8 -> 2.
    EXPECT_EQ(transpose_cost, 2);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestOnlyOptimizeTowardsTranspose) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({10, 4}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* mul_1_out_0 = builder.MakeIntermediate();
    auto* relu_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Mul", {transpose_1_out_0, input1_arg}, {mul_1_out_0});
    builder.AddNode("Relu", {mul_1_out_0}, {relu_1_out_0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    // Pushing through Mul is skipped to avoid additional cost. Cast 3 remains.
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestDontOptimizeWrongInput) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<int32_t>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, -1, 5);
    auto* input1_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* castlike_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input1_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("CastLike", {input0_arg, transpose_1_out_0}, {castlike_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {castlike_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    ORT_UNUSED_PARAMETER(session);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestOptimizeBothInputs) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* input1_arg = builder.MakeInput<float>({4, 6, 10}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeIntermediate();
    auto* mul_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    auto& transpose_2 = builder.AddNode("Transpose", {input1_arg}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{1, 2, 0});
    builder.AddNode("Mul", {transpose_1_out_0, transpose_2_out_0}, {mul_1_out_0});
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    // No target transpose after Mul but both input transposes match and are pushed. Cost 6 -> 3.
    EXPECT_EQ(transpose_cost, 3);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

TEST(TransposeOptimizerTests, TestOmitIdentityTranspose) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    if (builder.DomainToVersionMap().find(kOnnxDomain)->second >= 18) {
      auto* init = builder.MakeInitializer<int64_t>({1}, {1});
      auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0, init}, {reducemax_1_out_0});
      reducemax_1.AddAttribute("keepdims", (int64_t)0);
    } else {
      auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
      reducemax_1.AddAttribute("axes", std::vector<int64_t>{1});
      reducemax_1.AddAttribute("keepdims", (int64_t)0);
    }
  };

  auto check_optimized_graph_1 = [&](InferenceSessionWrapper& session) {
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    // No transpose is placed on output since it would be an identity.
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ {15, 18});
}

// regression test for a model where the transpose optimizations were not completed in a single pass in level 1.
// fixed by
//   a) moving the RewriteRule level 1 optimizations so they run prior to the transpose optimizer; and
//   b) not returning `true` from TransposeOptimizer::ShouldOnlyApplyOnce as it should be safe to run the
//      transpose optimizer multiple times to ensure it completes in level 1.
// either of those changes would have fixed the issue.
// see https://github.com/microsoft/onnxruntime/issues/9671 for more details.
TEST(TransposeOptimizerTests, RegressionTest_GitHubIssue9671) {
  auto model_uri = ORT_TSTR("testdata/gh_issue_9671.onnx");

  SessionOptions so;
  so.session_logid = "TransposeOptimizerTests.RegressionTest_GitHubIssue9671";
  InferenceSession session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_uri));
  ASSERT_STATUS_OK(session.Initialize());  // optimizers run during initialization
}

// regression test for a model where the transpose optimizations incorrectly removed a node providing an implicit
// input to a subgraph. fixed by updating Graph::BuildConnections to add implicit inputs to node_arg_to_consumer_nodes_
// see https://github.com/microsoft/onnxruntime/issues/10305 for more details.
TEST(TransposeOptimizerTests, RegressionTest_GitHubIssue10305) {
  Status status;
  auto model_uri = ORT_TSTR("testdata/ort_github_issue_10305.onnx");

  SessionOptions so;
  so.session_logid = "TransposeOptimizerTests.RegressionTest_GitHubIssue10305";
  InferenceSession session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_uri));
  ASSERT_STATUS_OK(session.Initialize());  // optimizers run during initialization
}

// regression test for a model with DQ node with per-axis dequantization followed by a Transpose.
// the second phase can swap those around, but needs to use the correct perms for updating the 'axis'
// attribute in the DQ node.
// see https://github.com/microsoft/onnxruntime/issues/12151 for more details.
TEST(TransposeOptimizerTests, RegressionTest_GitHubIssue12151) {
  Status status;
  auto model_uri = ORT_TSTR("testdata/ort_github_issue_12151.onnx");

  NameMLValMap feeds;  // no inputs for this model
  std::vector<std::string> output_names{"Z"};
  std::vector<OrtValue> fetches_orig;
  std::vector<OrtValue> fetches;

  SessionOptions so;
  so.session_logid = "TransposeOptimizerTests.RegressionTest_GitHubIssue12151";

  {
    so.graph_optimization_level = TransformerLevel::Default;  // off
    InferenceSession session{so, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model_uri));
    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches_orig));
  }

  {
    so.graph_optimization_level = TransformerLevel::Level1;  // enable transpose optimizer
    InferenceSession session{so, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model_uri));
    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  }

  ASSERT_THAT(fetches_orig[0].Get<Tensor>().DataAsSpan<float>(),
              testing::ContainerEq(fetches[0].Get<Tensor>().DataAsSpan<float>()));
}

// These tests uses internal testing EP with static kernels which requires a full build,
// and the NHWC Conv which requires contrib ops
#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

// Test a Transpose node followed by a Reshape that is logically equivalent to an Transpose can be merged.
// The test graph was extracted from a model we were trying to use with the QNN EP.
TEST(TransposeOptimizerTests, QnnTransposeReshape) {
  Status status;
  auto model_uri = ORT_TSTR("testdata/layout_transform_reshape.onnx");

  SessionOptions so;

  // enable dumping graph so one test validates that infrastructure works. we don't want to do that in multiple
  // tests as the filenames for the debug output are hardcoded.
  // check the build output directory for files called `post_layout_transform_step_<step#>.onnx` to see how the graph
  // changes during the layout transformation process.
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kDebugLayoutTransformation, "1"));

  using InternalTestingEP = onnxruntime::internal_testing_ep::InternalTestingExecutionProvider;

  // set the test EP to support all ops in the model so that the layout transform applies to all nodes
  const std::unordered_set<std::string> empty_set;
  auto internal_testing_ep = std::make_unique<InternalTestingEP>(empty_set, empty_set, DataLayout::NHWC);
  internal_testing_ep->EnableStaticKernels().TakeAllNodes();

  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(internal_testing_ep)));
  ASSERT_STATUS_OK(session.Load(model_uri));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& graph = session.GetGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  // if we merge the Transpose -> Reshape the resulting Transpose node can be pushed down and will cancel out
  // all downstream Transpose nodes.
  // if the merge fails those two nodes remain, and the Transpose is blocked from being pushed down.
  // additionally, running the L1 transformers after the layout transform should constant fold the Transpose node that
  // gets inserted on the weights used by the Add node.
  // end result of everything working as expected is a single Transpose of the input data to NHWC as that can't be
  // avoided.
  ASSERT_EQ(op_to_count["Transpose"], 1) << "All layout transform Transpose ops should have been handled "
                                            "with the exception of the initial node prior to the Conv";

  // all nodes should be assigned to the internal testing EP, which also means they should be in NHWC layout
  std::string expected_ep(onnxruntime::utils::kInternalTestingExecutionProvider);
  for (const auto& node : graph.Nodes()) {
    EXPECT_TRUE(node.GetExecutionProviderType() == expected_ep) << node.OpType() << " node named '" << node.Name()
                                                                << "' was not assigned to the internal testing EP.";

    if (node.Name() == "Mul_212" || node.Name() == "Add_213") {
      // check that the special case in TransposeInputs for a single element input reconnects things back up correctly
      const auto& inputs = node.InputDefs();
      EXPECT_EQ(inputs.size(), size_t(2));
      EXPECT_TRUE(inputs[1]->Exists());
    }
  }
}

TEST(TransposeOptimizerTests, QnnTransposeReshapeQDQ) {
  Status status;
  auto model_uri = ORT_TSTR("testdata/layout_transform_reshape.qdq.onnx");

  SessionOptions so;

  using InternalTestingEP = onnxruntime::internal_testing_ep::InternalTestingExecutionProvider;

  // set the test EP to support all ops in the model so that the layout transform applies to all nodes
  const std::unordered_set<std::string> empty_set;
  auto internal_testing_ep = std::make_unique<InternalTestingEP>(empty_set, empty_set, DataLayout::NHWC);
  internal_testing_ep->EnableStaticKernels().TakeAllNodes();

  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(internal_testing_ep)));
  ASSERT_STATUS_OK(session.Load(model_uri));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& graph = session.GetGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  // if we merge the Transpose -> Reshape the resulting Transpose node can be pushed down and will cancel out
  // all downstream Transpose nodes.
  // if the merge fails those two nodes remain, and the Transpose is blocked from being pushed down.
  // additionally, running the L1 transformers after the layout transform should constant fold the Transpose node that
  // gets inserted on the weights used by the Add node.
  // end result of everything working as expected is a single Transpose of the input data to NHWC as that can't be
  // avoided.
  ASSERT_EQ(op_to_count["Transpose"], 1) << "All layout transform Transpose ops should have been handled "
                                            "with the exception of the initial node prior to the Conv";

  // all nodes should be assigned to the internal testing EP, which also means they should be in NHWC layout
  std::string expected_ep(onnxruntime::utils::kInternalTestingExecutionProvider);
  for (const auto& node : graph.Nodes()) {
    EXPECT_TRUE(node.GetExecutionProviderType() == expected_ep) << node.OpType() << " node named '" << node.Name()
                                                                << "' was not assigned to the internal testing EP.";
  }
}

// Validate handling for EP with layout specific Resize that prefers NHWC
TEST(TransposeOptimizerTests, QnnResizeOpset11) {
  Status status;
  auto model_uri = ORT_TSTR("testdata/nhwc_resize_scales_opset11.onnx");

  SessionOptions so;
  // Uncomment to debug
  // ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kDebugLayoutTransformation, "1"));

  using InternalTestingEP = onnxruntime::internal_testing_ep::InternalTestingExecutionProvider;

  // set the test EP to support all ops in the model so that the layout transform applies to all nodes
  const std::unordered_set<std::string> empty_set;
  auto internal_testing_ep = std::make_unique<InternalTestingEP>(empty_set, empty_set, DataLayout::NHWC);
  internal_testing_ep->EnableStaticKernels().TakeAllNodes();

  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(internal_testing_ep)));
  ASSERT_STATUS_OK(session.Load(model_uri));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& graph = session.GetGraph();
  // all nodes should be assigned to the internal testing EP, which also means they should be in NHWC layout
  std::string expected_ep(onnxruntime::utils::kInternalTestingExecutionProvider);
  for (const auto& node : graph.Nodes()) {
    EXPECT_TRUE(node.GetExecutionProviderType() == expected_ep) << node.OpType() << " node named '" << node.Name()
                                                                << "' was not assigned to the internal testing EP.";
    if (node.OpType() == "Resize") {
      EXPECT_EQ(node.Domain(), kMSInternalNHWCDomain) << "Resize was not converted to NHWC layout";
    }
  }

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 2) << "Resize should have been wrapped in 2 Transpose nodes to convert to NHWC";

  // And the post-Resize Transpose should have been pushed all the way to the end
  GraphViewer viewer(graph);
  EXPECT_EQ(graph.GetNode(viewer.GetNodesInTopologicalOrder().back())->OpType(), "Transpose");
}
#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

static void CheckSharedInitializerHandling(bool broadcast) {
  auto model_uri = broadcast ? ORT_TSTR("testdata/transpose_optimizer_shared_initializers_broadcast.onnx")
                             : ORT_TSTR("testdata/transpose_optimizer_shared_initializers.onnx");

  RandomValueGenerator random{123};
  std::vector<int64_t> input_dims{1, 2, 2, 3};
  std::vector<float> input_data = random.Gaussian<float>(input_dims, 0.0f, 1.0f);

  OrtValue input;
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], input_dims, input_data, &input);

  NameMLValMap feeds{{"input0", input}};

  std::vector<std::string> output_names{"output0"};
  std::vector<OrtValue> fetches_orig;
  std::vector<OrtValue> fetches;

  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableQuantQDQ, "1"));

  // get results with no modifications to the model
  {
    so.graph_optimization_level = TransformerLevel::Default;  // off
    InferenceSessionWrapper session{so, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model_uri));
    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches_orig));
  }

  {
    InferenceSessionWrapper session{so, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model_uri));

    // we call the ONNX transpose optimizer directly to simplify the model required to exercise the shared initializer
    // handling. this means we don't need to disable optimizers that might alter the graph before the
    // transpose optimizer runs (at a minimum ConstantFolding, CommonSubexpressionElimination and ConstantSharing).
    Graph& graph = session.GetMutableGraph();
    CPUAllocator allocator;

    using namespace onnx_transpose_optimization;
    auto api_graph = MakeApiGraph(graph, TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                                  /*new_node_ep*/ nullptr);

    // default optimization cost check
    OptimizeResult result = Optimize(*api_graph);

    ASSERT_EQ(result.error_msg, std::nullopt);
    ASSERT_TRUE(result.graph_modified);
    ASSERT_TRUE(graph.GraphResolveNeeded());
    ASSERT_STATUS_OK(graph.Resolve());

    // Use this hack to save model for viewing if needed
    // ASSERT_STATUS_OK(Model::Save(const_cast<Model&>(session.GetModel()), "updated_model.onnx"));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    EXPECT_EQ(op_to_count["Transpose"], 0) << "The Transpose nodes should have been pushed through or canceled out.";
    if (broadcast) {
      EXPECT_EQ(op_to_count["Unsqueeze"], 0) << "Any Unsqueeze nodes should have been canceled out.";
    }

    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  }

  ASSERT_THAT(fetches_orig[0].Get<Tensor>().DataAsSpan<float>(),
              testing::ContainerEq(fetches[0].Get<Tensor>().DataAsSpan<float>()));
}

// test we re-use a modified shared initializer wherever possible. model has one initializer that is used by 3 DQ nodes
// and one initializer that is used by 2 Add nodes. both cases should be handled with the initializer being
// modified in-place for the first usage, and the Transpose added to the second usage being cancelled out when the
// original Transpose at the start of the model is pushed down.
TEST(TransposeOptimizerTests, SharedInitializerHandling) {
  CheckSharedInitializerHandling(/*broadcast*/ false);
}

// same setup as the above test, however the initializer is broadcast to bring UnsqueezeInput into play.
// the in-place modification of the initializer for the first usage results in
//   <initializer> -> Transpose -> Squeeze -> {DQ | Add}
// the later usages of the initializer should attempt to cancel out the Squeeze in UnsqueezeInput,
// followed by canceling out the Transpose in TransposeInput.
TEST(TransposeOptimizerTests, SharedInitializerHandlingBroadcast) {
  CheckSharedInitializerHandling(/*broadcast*/ true);
}

// Unit test where EstimateTransposeValueCost must look past a DQ -> Squeeze to see the Transponse of a shared
// initializer for the overall cost of pushing the Transpose throught the second Where to be negative.
TEST(TransposeOptimizerTests, SharedInitializerHandlingBroadcast2) {
  auto model_uri = ORT_TSTR("testdata/transpose_optimizer_shared_initializers_broadcast2.onnx");

  RandomValueGenerator random{123};
  std::vector<int64_t> cond_input_0_dims{3, 2};
  std::vector<int64_t> cond_input_1_dims{2, 3};
  std::vector<bool> cond_input_data = {true, false, false, true, true, false};

  std::vector<int64_t> x_0_input_dims{3};
  std::vector<int64_t> x_1_input_dims{3};
  std::vector<float> x_input_data_0 = random.Gaussian<float>(x_0_input_dims, 0.0f, 1.0f);
  std::vector<float> x_input_data_1 = random.Gaussian<float>(x_1_input_dims, 0.0f, 1.0f);

  OrtValue cond_input_0, cond_input_1, x_input_0, x_input_1;
  CreateMLValue<bool>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], cond_input_0_dims, cond_input_data,
                      &cond_input_0);
  CreateMLValue<bool>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], cond_input_1_dims, cond_input_data,
                      &cond_input_1);
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], x_0_input_dims, x_input_data_0,
                       &x_input_0);
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], x_1_input_dims, x_input_data_1,
                       &x_input_1);

  NameMLValMap feeds{{"cond_in_0", cond_input_0},
                     {"cond_in_1", cond_input_1},
                     {"x_in_0", x_input_0},
                     {"x_in_1", x_input_1}};

  std::vector<std::string> output_names{"output0"};
  std::vector<OrtValue> fetches_orig;
  std::vector<OrtValue> fetches;

  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kDebugLayoutTransformation, "1"));
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableQuantQDQ, "1"));

  // get results with no modifications to the model
  {
    so.graph_optimization_level = TransformerLevel::Default;  // off
    InferenceSessionWrapper session{so, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model_uri));
    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches_orig));
  }

  {
    InferenceSessionWrapper session{so, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model_uri));

    // we call the ONNX transpose optimizer directly to simplify the model required to exercise the shared initializer
    // handling. this means we don't need to disable optimizers that might alter the graph before the
    // transpose optimizer runs (at a minimum ConstantFolding, CommonSubexpressionElimination and ConstantSharing).
    Graph& graph = session.GetMutableGraph();
    CPUAllocator allocator;

    using namespace onnx_transpose_optimization;
    auto api_graph = MakeApiGraph(graph, TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                                  /*new_node_ep*/ nullptr);

    // default optimization cost check
    OptimizeResult result = Optimize(*api_graph);

    ASSERT_EQ(result.error_msg, std::nullopt);
    ASSERT_TRUE(result.graph_modified);
    ASSERT_TRUE(graph.GraphResolveNeeded());
    ASSERT_STATUS_OK(graph.Resolve());

    // Use this hack to save model for viewing if needed
    // ASSERT_STATUS_OK(Model::Save(const_cast<Model&>(session.GetModel()), updated_model.onnx"));

    // Pushing the initial Transpose through the 2 Where nodes results in
    // - x_in_0 needs Transpose and Unsqueeze to broadcast correctly into the first Where
    // - y_quant is updated in-place to transposed layout and used in both Where nodes
    // - x_in_1 needs Transpose and Unsqueeze to broadcast correctly into the second Where
    // - cond_in_1 needs Transpose
    //   - as we're pushing a Transpose through the Add for one input, and undo-ing the Transpose on y_quant for
    //     the other input, we save 2 by adding 1 to cond_in_1
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    EXPECT_EQ(op_to_count["Transpose"], 3) << "The 2 X inputs and cond_in_1 should require transpose.";
    EXPECT_EQ(op_to_count["Unsqueeze"], 2) << "The 2 X inputs should require Unsqueeze.";

    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
  }

  ASSERT_THAT(fetches_orig[0].Get<Tensor>().DataAsSpan<float>(),
              testing::ContainerEq(fetches[0].Get<Tensor>().DataAsSpan<float>()));
}

// model where layout transform results in transposing a non-const input that is broadcast.
// this inserts Unsqueeze -> Transpose between the input and the node.
// test that QDQ node units are created for Unsqueeze and Transpose by inserting Q->DQ pairs after them
TEST(TransposeOptimizerTests, QnnTransposeNonConstBroadcastInput) {
  Status status;
  auto model_uri = ORT_TSTR("testdata/layout_transform_nonconst_broadcast_input.onnx");

  SessionOptions so;

  // ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kDebugLayoutTransformation, "1"));

  using InternalTestingEP = onnxruntime::internal_testing_ep::InternalTestingExecutionProvider;

  // set the test EP to support all ops in the model so that the layout transform applies to all nodes
  const std::unordered_set<std::string> empty_set;
  auto internal_testing_ep = std::make_unique<InternalTestingEP>(empty_set, empty_set, DataLayout::NHWC);
  internal_testing_ep->EnableStaticKernels().TakeAllNodes();

  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(internal_testing_ep)));
  ASSERT_STATUS_OK(session.Load(model_uri));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& graph = session.GetGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count["Transpose"], 3) << "Should have Transpose on 2 inputs and one on output.";

  // all nodes should be assigned to the internal testing EP, which also means they should be in NHWC layout
  std::string expected_ep(onnxruntime::utils::kInternalTestingExecutionProvider);
  for (const auto& node : graph.Nodes()) {
    EXPECT_EQ(node.GetExecutionProviderType(), expected_ep) << node.OpType() << " node named '" << node.Name()
                                                            << "' was not assigned to the internal testing EP.";
    // all nodes should be in QDQ node units except the Cast on an input which was not in a QDQ unit
    if (node.OpType() != "QuantizeLinear" && node.OpType() != "DequantizeLinear" && node.OpType() != "Cast") {
      for (auto cur_input = node.InputNodesBegin(), end = node.InputNodesEnd(); cur_input != end; ++cur_input) {
        EXPECT_EQ(cur_input->OpType(), "DequantizeLinear");
      }

      for (auto cur_output = node.OutputNodesBegin(), end = node.OutputNodesEnd(); cur_output != end; ++cur_output) {
        EXPECT_EQ(cur_output->OpType(), "QuantizeLinear");
      }
    }
  }
}
}  // namespace test
}  // namespace onnxruntime
