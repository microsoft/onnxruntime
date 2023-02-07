// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "graph_transform_test_builder.h"

#include "core/graph/graph.h"
#include "qdq_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

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
                    /*opset_version*/ 15);
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
    builder.AddNode("Split", {transpose_1_out_0}, {split_1_out_0, split_1_out_1});
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestPad) {
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
    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 0);
  };

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 10);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 11);
}

// The CUDA Resize kernel assumes that the input is NCHW and
// Resize can't be supported in ORT builds with CUDA enabled.
// TODO: Enable this once the CUDA Resize kernel is implemented
// "generically" (i.e.) aligning with the generic nature of the
// ONNX spec.
// See https://github.com/microsoft/onnxruntime/pull/10824 for
// a similar fix applied to the CPU Resize kernel.
// Per tests included in #10824, the ROCM EP also generates
// incorrect results when this handler is used, so the Resize
// handler is not enabled even for those builds.
#if !defined(USE_CUDA) && !defined(USE_ROCM)
TEST(TransposeOptimizerTests, TestResize) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{4, -1, 2, -1}}, {4, 6, 2, 10}, 0.0, 1.0);
    auto* const_1 = builder.MakeInitializer<float>({4}, {0.3f, 2.5f, 1.0f, 0.7f});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* resize_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("Resize", {transpose_1_out_0, const_1}, {resize_1_out_0});
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
                    TransformerLevel::Level1,
                    /*opset_version*/ 10);
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
                    TransformerLevel::Level1,
                    /*opset_version*/ 11);
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
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
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
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
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
                    TransformerLevel::Level1);
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
                    TransformerLevel::Level1,
                    /*opset_version*/ 11);
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
                    TransformerLevel::Level1,
                    /*opset_version*/ 13);
}

#endif
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 7);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceSumKeepdimsTrue) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
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
                    /*opset_version*/ 7,
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
                    /*opset_version*/ 7,
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
    auto& reducesum_1 = builder.AddNode("ReduceSum", {transpose_1_out_0}, {reducesum_1_out_0});
    reducesum_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
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
                    /*opset_version*/ 7,
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
                    /*opset_version*/ 7,
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

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsTrueNoopEmptyTrue) {
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

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsTrueNoopEmptyFalse) {
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

TEST(TransposeOptimizerTests, TestReduceSumNoAxesInput) {
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

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsFalseNoopEmptyTrue) {
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

TEST(TransposeOptimizerTests, TestReduceSumEmptyAxesKeepdimsFalseNoopEmptyFalse) {
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

TEST(TransposeOptimizerTests, TestReduceSumNoAxesInput_2) {
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

TEST(TransposeOptimizerTests, TestReduceSumNonconstKeepdimsTrueNoOpt) {
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

TEST(TransposeOptimizerTests, TestReduceSumNonconstKeepdimsFalseNoOpt) {
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
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducemax_1.AddAttribute("keepdims", (int64_t)1);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceMaxKeepdimsFalse) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducemax_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceMax) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceLogSum) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducelogsum_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducelogsum_1 = builder.AddNode("ReduceLogSum", {transpose_1_out_0}, {reducelogsum_1_out_0});
    reducelogsum_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducelogsum_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceLogSumExp) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducelogsumexp_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducelogsumexp_1 = builder.AddNode("ReduceLogSumExp", {transpose_1_out_0}, {reducelogsumexp_1_out_0});
    reducelogsumexp_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducelogsumexp_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceMax) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducemax_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceMean) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemean_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemean_1 = builder.AddNode("ReduceMean", {transpose_1_out_0}, {reducemean_1_out_0});
    reducemean_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducemean_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceMin) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemin_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemin_1 = builder.AddNode("ReduceMin", {transpose_1_out_0}, {reducemin_1_out_0});
    reducemin_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducemin_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceProd) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reduceprod_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reduceprod_1 = builder.AddNode("ReduceProd", {transpose_1_out_0}, {reduceprod_1_out_0});
    reduceprod_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reduceprod_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceSumSquare) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducesumsquare_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducesumsquare_1 = builder.AddNode("ReduceSumSquare", {transpose_1_out_0}, {reducesumsquare_1_out_0});
    reducesumsquare_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducesumsquare_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceL1) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducel1_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducel1_1 = builder.AddNode("ReduceL1", {transpose_1_out_0}, {reducel1_1_out_0});
    reducel1_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducel1_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestReduceOpsReduceL2) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducel2_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducel2_1 = builder.AddNode("ReduceL2", {transpose_1_out_0}, {reducel2_1_out_0});
    reducel2_1.AddAttribute("axes", std::vector<int64_t>{0, -2});
    reducel2_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestSqueeze) {
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
                    /*opset_version*/ 7);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestUnsqueeze) {
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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

  TransformerTester(build_test_case_1,
                    check_optimized_graph_1,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestQuantizeLinearScalar) {
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestQuantizeLinearScalarIgnoreAxis) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* input1_arg = MakeInput<float>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {2.3f});
    auto* input2_arg = MakeInput<uint8_t>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* quantizelinear_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& quantizelinear_1 = builder.AddNode("QuantizeLinear", {transpose_1_out_0, input1_arg, input2_arg}, {quantizelinear_1_out_0});
    quantizelinear_1.AddAttribute("axis", (int64_t)10);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestQuantizeLinearVector) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* input1_arg = MakeInput<float>(builder, {{-1}}, {2}, {2.3f, 2.4f});
    auto* input2_arg = MakeInput<uint8_t>(builder, {{-1}}, {2}, {10, 12});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* quantizelinear_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& quantizelinear_1 = builder.AddNode("QuantizeLinear", {transpose_1_out_0, input1_arg, input2_arg}, {quantizelinear_1_out_0});
    quantizelinear_1.AddAttribute("axis", (int64_t)0);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestQuantizeLinearVectorUnknownRank) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0.0, 1.0);
    auto* input1_arg = MakeInput<float>(builder, std::nullopt, {3}, {2.3f, 2.4f, 2.5f});
    auto* input2_arg = MakeInput<uint8_t>(builder, std::nullopt, {3}, {10, 12, 13});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* quantizelinear_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& quantizelinear_1 = builder.AddNode("QuantizeLinear", {transpose_1_out_0, input1_arg, input2_arg}, {quantizelinear_1_out_0});
    quantizelinear_1.AddAttribute("axis", (int64_t)1);
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
                    /*opset_version*/ 15);
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

TEST(TransposeOptimizerTests, TestDequantizeLinearScalarIgnoreAxis) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<uint8_t>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0, 5);
    auto* input1_arg = MakeInput<float>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {2.3f});
    auto* input2_arg = MakeInput<uint8_t>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* dequantizelinear_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& dequantizelinear_1 = builder.AddNode("DequantizeLinear", {transpose_1_out_0, input1_arg, input2_arg}, {dequantizelinear_1_out_0});
    dequantizelinear_1.AddAttribute("axis", (int64_t)10);
    auto& transpose_2 = builder.AddNode("Transpose", {dequantizelinear_1_out_0}, {transpose_2_out_0});
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestDequantizeLinearVector) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<uint8_t>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0, 5);
    auto* input1_arg = MakeInput<float>(builder, {{2}}, {2}, {2.3f, 2.4f});
    auto* input2_arg = MakeInput<uint8_t>(builder, {{2}}, {2}, {10, 12});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* dequantizelinear_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& dequantizelinear_1 = builder.AddNode("DequantizeLinear", {transpose_1_out_0, input1_arg, input2_arg}, {dequantizelinear_1_out_0});
    dequantizelinear_1.AddAttribute("axis", (int64_t)-4);
    auto& transpose_2 = builder.AddNode("Transpose", {dequantizelinear_1_out_0}, {transpose_2_out_0});
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestDequantizeLinearNoAxis) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<uint8_t>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0, 5);
    auto* input1_arg = MakeInput<float>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {2.3f});
    auto* input2_arg = MakeInput<uint8_t>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {10});
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* dequantizelinear_1_out_0 = builder.MakeIntermediate();
    auto* transpose_2_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    builder.AddNode("DequantizeLinear", {transpose_1_out_0, input1_arg, input2_arg}, {dequantizelinear_1_out_0});
    auto& transpose_2 = builder.AddNode("Transpose", {dequantizelinear_1_out_0}, {transpose_2_out_0});
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

TEST(TransposeOptimizerTests, TestDequantizeLinearTransposePropagation) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<uint8_t>(builder, {{2, -1, 6, 3}}, {2, 4, 6, 3}, 0, 5);
    auto* input1_arg = MakeInput<float>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {2.3f});
    auto* input2_arg = MakeInput<uint8_t>(builder, {std::vector<int64_t>{}}, std::vector<int64_t>{}, {10});
    auto* dequantizelinear_1_out_0 = builder.MakeIntermediate();
    auto* transpose_1_out_0 = builder.MakeOutput();
    auto* transpose_2_out_0 = builder.MakeOutput();

    builder.AddNode("DequantizeLinear", {input0_arg, input1_arg, input2_arg}, {dequantizelinear_1_out_0});

    auto& transpose_1 = builder.AddNode("Transpose", {dequantizelinear_1_out_0}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});

    auto& transpose_2 = builder.AddNode("Transpose", {dequantizelinear_1_out_0}, {transpose_2_out_0});
    transpose_2.AddAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    std::vector<std::string> expected_op_types_in_order{
        "DequantizeLinear",
        "Transpose",
        "Transpose"};

    const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph());
    EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
  };

  TransformerTester(build_test_case_1,
                    check_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 10);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
    ASSERT_TRUE(op_to_count["Mul"] == 1);
    ASSERT_TRUE(op_to_count["Transpose"] == 0);
    ASSERT_TRUE(op_to_count["Reshape"] == 0);
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
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
    ASSERT_TRUE(op_to_count["Mul"] == 1);
    ASSERT_TRUE(op_to_count["Transpose"] == 0);
    ASSERT_TRUE(op_to_count["Reshape"] == 0);
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
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
    ASSERT_TRUE(op_to_count["Mul"] == 1);
    ASSERT_TRUE(op_to_count["Transpose"] == 0);
    ASSERT_TRUE(op_to_count["Reshape"] == 0);
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestNonCancelingReshape) {
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
    ASSERT_TRUE(op_to_count["Mul"] == 1);
    ASSERT_TRUE(op_to_count["Transpose"] == 1);
    ASSERT_TRUE(op_to_count["Reshape"] == 1);

    int transpose_cost = EstimateTransposeCost(session.GetGraph());
    EXPECT_EQ(transpose_cost, 1);
  };

  TransformerTester(build_test_case,
                    check_optimized_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level1,
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
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
                    /*opset_version*/ 15);
}

TEST(TransposeOptimizerTests, TestOmitIdentityTranspose) {
  auto build_test_case_1 = [&](ModelTestBuilder& builder) {
    auto* input0_arg = MakeInput<float>(builder, {{-1, 4, -1, 5}}, {2, 4, 6, 5}, 0.0, 1.0);
    auto* transpose_1_out_0 = builder.MakeIntermediate();
    auto* reducemax_1_out_0 = builder.MakeOutput();

    auto& transpose_1 = builder.AddNode("Transpose", {input0_arg}, {transpose_1_out_0});
    transpose_1.AddAttribute("perm", std::vector<int64_t>{0, 3, 1, 2});
    auto& reducemax_1 = builder.AddNode("ReduceMax", {transpose_1_out_0}, {reducemax_1_out_0});
    reducemax_1.AddAttribute("axes", std::vector<int64_t>{1});
    reducemax_1.AddAttribute("keepdims", (int64_t)0);
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
                    /*opset_version*/ 15);
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
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());  // optimizers run during initialization
}

// regression test for a model where the transpose optimizations incorrectly removed a node providing an implicit
// input to a subgraph. fixed by updating Graph::BuildConnections to add implicit inputs to node_arg_to_consumer_nodes_
// see https://github.com/microsoft/onnxruntime/issues/10305 for more details.
TEST(TransposeOptimizerTests, RegressionTest_GitHubIssue10305) {
  Status status;
  auto model_uri = ORT_TSTR("testdata/ort_github_issue_10305.onnx");

  SessionOptions so;
  so.session_logid = "TransposeOptimizerTests.RegressionTest_GitHubIssue10305";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());  // optimizers run during initialization
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
    InferenceSession session_object{so, GetEnvironment()};
    ASSERT_STATUS_OK(session_object.Load(model_uri));
    ASSERT_STATUS_OK(session_object.Initialize());
    ASSERT_STATUS_OK(session_object.Run(feeds, output_names, &fetches_orig));
  }

  {
    so.graph_optimization_level = TransformerLevel::Level1;  // enable transpose optimizer
    InferenceSession session_object{so, GetEnvironment()};
    ASSERT_STATUS_OK(session_object.Load(model_uri));
    ASSERT_STATUS_OK(session_object.Initialize());
    ASSERT_STATUS_OK(session_object.Run(feeds, output_names, &fetches));
  }

  ASSERT_THAT(fetches_orig[0].Get<Tensor>().DataAsSpan<float>(),
              testing::ContainerEq(fetches[0].Get<Tensor>().DataAsSpan<float>()));
}
}  // namespace test
}  // namespace onnxruntime
