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

#ifndef DISABLE_CONTRIB_OPS

TEST(NhwcTransformerTests, Conv) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape, 0, 31);
      auto* output_arg = builder.MakeOutput();
      auto* weight_arg = NhwcMakeInitializer<uint8_t>(builder, weights_shape);

      builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                          weight_arg, .02f, 126,
                                          output_arg, .37f, 131);
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 2);
    };

    TransformerTester(build_test_case,
                      check_nhwc_graph,
                      TransformerLevel::Level2,
                      TransformerLevel::Level3);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

TEST(NhwcTransformerTests, ConvDequantizeLinear) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<uint8_t>({1, 12, 37}, 0, 31);
    auto* conv_output_arg = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();
    auto* weight_arg = NhwcMakeInitializer<uint8_t>(builder, {32, 12, 5});

    builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                        weight_arg, .02f, 126,
                                        conv_output_arg, .37f, 131);
    builder.AddDequantizeLinearNode<uint8_t>(conv_output_arg,
                                             .37f, 131,
                                             output_arg);
  };

  auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 0);
    EXPECT_EQ(op_to_count["Transpose"], 0);
  };

  // QLinearConv followed by only DequantizeLinear will remain as the ONNX
  // version of the operator to avoid adding unnecessary Transpose nodes to
  // the graph.
  TransformerTester(build_test_case,
                    check_nhwc_graph,
                    TransformerLevel::Level2,
                    TransformerLevel::Level3);
}

TEST(NhwcTransformerTests, ConvBlockBinary) {
  auto test_case = [&](const std::string& binary_op_type) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>({1, 23, 13, 13}, 0, 31);
      auto* conv1_output_arg = builder.MakeIntermediate();
      auto* conv2_output_arg = builder.MakeIntermediate();
      auto* output_arg = builder.MakeOutput();
      auto* conv1_weight_arg = builder.MakeInitializer<uint8_t>({30, 23, 3, 3},
                                                                NhwcWeightsRange<uint8_t>::min_value,
                                                                NhwcWeightsRange<uint8_t>::max_value);

      auto* conv2_weight_arg = builder.MakeInitializer<uint8_t>({30, 23, 1, 1},
                                                                NhwcWeightsRange<uint8_t>::min_value,
                                                                NhwcWeightsRange<uint8_t>::max_value);

      Node& conv1_node = builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                                             conv1_weight_arg, .02f, 126,
                                                             conv1_output_arg, .37f, 131);
      conv1_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                          conv2_weight_arg, .015f, 129,
                                          conv2_output_arg, .37f, 131);
      builder.AddQLinearBinaryNode(binary_op_type,
                                   conv1_output_arg, .37f, 131,
                                   conv2_output_arg, .37f, 131,
                                   output_arg, .43f, 126);
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
      EXPECT_EQ(op_to_count["Transpose"], 2);
    };

    TransformerTester(build_test_case,
                      check_nhwc_graph,
                      TransformerLevel::Level2,
                      TransformerLevel::Level3);
  };

  std::vector<std::string> activation_op_types{"QLinearAdd", "QLinearMul"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NhwcTransformerTests, ConvMaxPool) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape, 0, 31);
      auto* conv_output_arg = builder.MakeIntermediate();
      auto* output_arg = builder.MakeOutput();
      auto* conv_weight_arg = NhwcMakeInitializer<uint8_t>(builder, weights_shape);

      builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                          conv_weight_arg, .02f, 126,
                                          conv_output_arg, .37f, 131);
      Node& pool_node = builder.AddNode("MaxPool", {conv_output_arg}, {output_arg});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.NhwcMaxPool"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 2);
    };

    TransformerTester(build_test_case,
                      check_nhwc_graph,
                      TransformerLevel::Level2,
                      TransformerLevel::Level3);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({5, 12, 37}, {128, 12, 5});
  test_case({3, 14, 13, 13}, {64, 14, 3, 3});
  test_case({1, 15, 11, 13, 15}, {31, 15, 5, 3, 3});
}

TEST(NhwcTransformerTests, ConvMaxPoolIndexTensor) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<uint8_t>({1, 16, 17, 17}, 0, 31);
    auto* conv_output_arg = builder.MakeIntermediate();
    auto* index_output_arg = builder.MakeOutput();
    auto* output_arg = builder.MakeOutput();
    auto* conv_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {16, 16, 3, 3});

    builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                        conv_weight_arg, .02f, 126,
                                        conv_output_arg, .37f, 131);
    Node& pool_node = builder.AddNode("MaxPool", {conv_output_arg}, {output_arg, index_output_arg});
    pool_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
  };

  auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 1);
    EXPECT_EQ(op_to_count["MaxPool"], 1);
    EXPECT_EQ(op_to_count["Transpose"], 2);
  };

  // Test that MaxPool using the optional index tensor is not converted to NhwcMaxPool.
  TransformerTester(build_test_case,
                    check_nhwc_graph,
                    TransformerLevel::Level2,
                    TransformerLevel::Level3);
}

TEST(NhwcTransformerTests, ConvGlobalAveragePool) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<uint8_t>({1, 23, 13, 13}, 0, 31);
    auto* conv1_output_arg = builder.MakeIntermediate();
    auto* conv2_output_arg = builder.MakeIntermediate();
    auto* gavgpool1_output_arg = builder.MakeIntermediate();
    auto* gavgpool2_output_arg = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();
    auto* conv1_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {30, 23, 3, 3});
    auto* conv2_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {16, 30, 1, 1});

    Node& conv1_node = builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                                           conv1_weight_arg, .02f, 126,
                                                           conv1_output_arg, .37f, 131);
    conv1_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
    builder.AddQLinearActivationNode("QLinearGlobalAveragePool",
                                     conv1_output_arg, .37f, 131,
                                     gavgpool1_output_arg, .43f, 111);
    builder.AddQLinearConvNode<uint8_t>(gavgpool1_output_arg, .43f, 111,
                                        conv2_weight_arg, .015f, 129,
                                        conv2_output_arg, .37f, 131);
    builder.AddQLinearActivationNode("QLinearGlobalAveragePool",
                                     conv2_output_arg, .37f, 131,
                                     gavgpool2_output_arg, .37f, 131);
    builder.AddDequantizeLinearNode<uint8_t>(gavgpool2_output_arg, .37f, 131, output_arg);
  };

  auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
    EXPECT_EQ(op_to_count["Transpose"], 2);
  };

  TransformerTester(build_test_case,
                    check_nhwc_graph,
                    TransformerLevel::Level2,
                    TransformerLevel::Level3);
}

TEST(NhwcTransformerTests, ConvAveragePool) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<uint8_t>({1, 23, 13, 13}, 0, 31);
    auto* conv1_output_arg = builder.MakeIntermediate();
    auto* conv2_output_arg = builder.MakeIntermediate();
    auto* avgpool1_output_arg = builder.MakeIntermediate();
    auto* avgpool2_output_arg = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();
    auto* conv1_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {30, 23, 3, 3});
    auto* conv2_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {16, 30, 3, 3});

    Node& conv1_node = builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                                           conv1_weight_arg, .02f, 126,
                                                           conv1_output_arg, .37f, 131);
    conv1_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
    Node& avgpool_node1 = builder.AddQLinearActivationNode("QLinearAveragePool",
                                                           conv1_output_arg, .37f, 131,
                                                           avgpool1_output_arg, .43f, 111);
    avgpool_node1.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    avgpool_node1.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});

    builder.AddQLinearConvNode<uint8_t>(avgpool1_output_arg, .43f, 111,
                                        conv2_weight_arg, .015f, 129,
                                        conv2_output_arg, .37f, 131);
    Node& avgpool_node2 = builder.AddQLinearActivationNode("QLinearAveragePool",
                                                         conv2_output_arg, .37f, 131,
                                                         avgpool2_output_arg, .37f, 131);
    avgpool_node2.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    avgpool_node2.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});

    builder.AddDequantizeLinearNode<uint8_t>(avgpool2_output_arg, .37f, 131, output_arg);
  };

  auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
    EXPECT_EQ(op_to_count["Transpose"], 2);
  };

  TransformerTester(build_test_case,
                    check_nhwc_graph,
                    TransformerLevel::Level2,
                    TransformerLevel::Level3);
}

TEST(NhwcTransformerTests, ConvSplit) {
  for (int64_t axis = -4LL; axis < 4; axis++) {
    auto build_test_case = [&, axis](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>({2, 23, 16, 16}, 0, 31);
      auto* conv_output_arg = builder.MakeIntermediate();
      auto* split_output1_arg = builder.MakeIntermediate();
      auto* split_output2_arg = builder.MakeIntermediate();
      auto* qladd_output_arg = builder.MakeIntermediate();
      auto* output_arg = builder.MakeOutput();

      const int64_t conv1_output_channels = 32;
      auto* conv1_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {conv1_output_channels, 23, 3, 3});

      Node& conv_node = builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                                            conv1_weight_arg, .02f, 126,
                                                            conv_output_arg, .37f, 131);
      conv_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      Node& split_node = builder.AddNode("Split", {conv_output_arg}, {split_output1_arg, split_output2_arg});
      split_node.AddAttribute("axis", static_cast<int64_t>(axis));
      builder.AddQLinearBinaryNode("QLinearAdd",
                                   split_output1_arg, .37f, 131,
                                   split_output2_arg, .37f, 131,
                                   qladd_output_arg, .43f, 126);
      const int64_t channels_after_split =
          (axis == 1 || axis == -3) ? conv1_output_channels / 2 : conv1_output_channels;

      auto* conv2_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {17, channels_after_split, 3, 3});
      builder.AddQLinearConvNode<uint8_t>(qladd_output_arg, .43f, 126,
                                          conv2_weight_arg, .02f, 126,
                                          output_arg, .37f, 131);
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
      EXPECT_EQ(op_to_count["Transpose"], 2);
    };

    TransformerTester(build_test_case,
                      check_nhwc_graph,
                      TransformerLevel::Level2,
                      TransformerLevel::Level3);
  }
}

TEST(NhwcTransformerTests, ConvSplitQLinearConcat) {
  for (int64_t axis = -4LL; axis < 4; axis++) {
    auto build_test_case = [&, axis](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>({2, 23, 16, 16}, 0, 31);
      auto* conv_output_arg = builder.MakeIntermediate();
      auto* split_output1_arg = builder.MakeIntermediate();
      auto* split_output2_arg = builder.MakeIntermediate();
      auto* qlconcat_output_arg = builder.MakeIntermediate();
      auto* output_arg = builder.MakeOutput();

      const int64_t conv1_output_channels = 32;
      auto* conv1_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {conv1_output_channels, 23, 3, 3});
      Node& conv_node = builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                                            conv1_weight_arg, .02f, 126,
                                                            conv_output_arg, .37f, 131);
      conv_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});

      Node& split_node = builder.AddNode("Split", {conv_output_arg}, {split_output1_arg, split_output2_arg});
      split_node.AddAttribute("axis", static_cast<int64_t>(axis));

      Node& qlconcat_node = builder.AddQLinearConcatLike(
          "QLinearConcat", qlconcat_output_arg, .37f, 131,
          {std::make_tuple(split_output1_arg, .37f, uint8_t(131)), std::make_tuple(split_output2_arg, .37f, uint8_t(131))});
      qlconcat_node.AddAttribute("axis", static_cast<int64_t>(axis));

      auto* conv2_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {17, conv1_output_channels, 3, 3});
      builder.AddQLinearConvNode<uint8_t>(qlconcat_output_arg, .43f, 126,
                                          conv2_weight_arg, .02f, 126,
                                          output_arg, .37f, 131);
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
      EXPECT_EQ(op_to_count["Transpose"], 2);
    };

    TransformerTester(build_test_case,
                      check_nhwc_graph,
                      TransformerLevel::Level2,
                      TransformerLevel::Level3);
  }
}

TEST(NhwcTransformerTests, ConvPad) {
  std::vector<std::string> pad_modes{"constant", "reflect", "edge"};
  for (const auto& mode : pad_modes) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>({1, 23, 13, 13}, 0, 31);
      auto* conv1_output_arg = builder.MakeIntermediate();
      auto* pads_const = builder.MakeScalarInitializer<uint8_t>(131);
      auto* pads_arg = builder.Make1DInitializer<int64_t>({0, 0, 1, 2, 0, 0, 3, 4});
      auto* pad_output_arg = builder.MakeIntermediate();
      auto* conv2_output_arg = builder.MakeIntermediate();
      auto* output_arg = builder.MakeOutput();

      auto* conv1_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {30, 23, 3, 3});
      Node& conv1_node = builder.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                                             conv1_weight_arg, .02f, 126,
                                                             conv1_output_arg, .37f, 131);
      conv1_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      Node& pad_node = builder.AddNode("Pad", {conv1_output_arg, pads_arg, pads_const}, {pad_output_arg});
      pad_node.AddAttribute("mode", mode);

      auto* conv2_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {16, 30, 3, 3});
      builder.AddQLinearConvNode<uint8_t>(pad_output_arg, .37f, 131,
                                          conv2_weight_arg, .015f, 129,
                                          conv2_output_arg, .37f, 131);
      builder.AddDequantizeLinearNode<uint8_t>(conv2_output_arg, .37f, 131, output_arg);
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
      EXPECT_EQ(op_to_count["Transpose"], 2);
    };

    TransformerTester(build_test_case,
                      check_nhwc_graph,
                      TransformerLevel::Level2,
                      TransformerLevel::Level3);
  }
}

TEST(NhwcTransformerTests, ConvBlockActivation) {
  auto test_case = [&](uint32_t extra_edges) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<uint8_t>({1, 10, 13, 13}, 0, 31);
      auto* input2_arg = builder.MakeInput<uint8_t>({1, 13, 13, 13}, 0, 31);
      auto* concat_arg = builder.MakeIntermediate();
      auto* conv1_output_arg = builder.MakeIntermediate();
      auto* conv2_output_arg = builder.MakeIntermediate();
      auto* act1_output_arg = builder.MakeIntermediate();
      auto* act2_output_arg = builder.MakeIntermediate();
      auto* output_arg = builder.MakeOutput();

      // Create a convolution input that isn't directly a graph input.
      Node& concat_node = builder.AddNode("Concat", {input1_arg, input2_arg}, {concat_arg});
      concat_node.AddAttribute("axis", static_cast<int64_t>(1));

      auto* conv1_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {30, 23, 3, 3});
      Node& conv1_node = builder.AddQLinearConvNode<uint8_t>(concat_arg, .01f, 135,
                                                             conv1_weight_arg, .02f, 126,
                                                             conv1_output_arg, .37f, 131);
      conv1_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      builder.AddQLinearActivationNode("QLinearSigmoid",
                                       conv1_output_arg, .37f, 131,
                                       act1_output_arg, .37f, 131);

      auto* conv2_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {30, 23, 1, 1});
      builder.AddQLinearConvNode<uint8_t>(concat_arg, .01f, 135,
                                          conv2_weight_arg, .015f, 129,
                                          conv2_output_arg, .37f, 131);
      builder.AddQLinearActivationNode("QLinearLeakyRelu",
                                       conv2_output_arg, .37f, 131,
                                       act2_output_arg, .37f, 131);
      builder.AddQLinearBinaryNode("QLinearAdd",
                                   act1_output_arg, .37f, 131,
                                   act2_output_arg, .37f, 131,
                                   output_arg, .39f, 126);

      // Create extra uses of the various NodeArgs to exercise the transformer.
      if ((extra_edges & 1) != 0) {
        builder.AddDequantizeLinearNode<uint8_t>(concat_arg, .01f, 135, builder.MakeOutput());
      }
      if ((extra_edges & 2) != 0) {
        builder.AddDequantizeLinearNode<uint8_t>(conv1_output_arg, .37f, 131, builder.MakeOutput());
      }
      if ((extra_edges & 4) != 0) {
        builder.AddDequantizeLinearNode<uint8_t>(conv2_output_arg, .37f, 131, builder.MakeOutput());
      }
      if ((extra_edges & 8) != 0) {
        builder.AddDequantizeLinearNode<uint8_t>(act1_output_arg, .37f, 131, builder.MakeOutput());
      }
      if ((extra_edges & 16) != 0) {
        builder.AddDequantizeLinearNode<uint8_t>(act2_output_arg, .37f, 131, builder.MakeOutput());
      }
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
    };

    TransformerTester(build_test_case,
                      check_nhwc_graph,
                      TransformerLevel::Level2,
                      TransformerLevel::Level3);
  };

  // Add extra uses of the edges that cause the transformer to insert additional
  // Transpose operations.
  for (uint32_t extra_edges = 0; extra_edges < 32; extra_edges++) {
    test_case(extra_edges);
  }
}

TEST(NhwcTransformerTests, ConvMixTensorRanks) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<uint8_t>({1, 10, 7}, 0, 31);
    auto* input2_arg = builder.MakeInput<uint8_t>({1, 12, 7, 7}, 0, 31);
    auto* conv1_output_arg = builder.MakeIntermediate();
    auto* conv2_output_arg = builder.MakeIntermediate();
    auto* output_arg = builder.MakeOutput();

    auto* conv1_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {1, 10, 3});
    builder.AddQLinearConvNode<uint8_t>(input1_arg, .01f, 135,
                                        conv1_weight_arg, .02f, 126,
                                        conv1_output_arg, .37f, 131);

    auto* conv2_weight_arg = NhwcMakeInitializer<uint8_t>(builder, {1, 12, 3, 3});
    builder.AddQLinearConvNode<uint8_t>(input2_arg, .01f, 135,
                                        conv2_weight_arg, .02f, 126,
                                        conv2_output_arg, .37f, 131);
    // Broadcast add {1, 1, 5} to {1, 1, 5, 5}.
    builder.AddQLinearBinaryNode("QLinearAdd",
                                 conv1_output_arg, .37f, 131,
                                 conv2_output_arg, .37f, 131,
                                 output_arg, .39f, 126);
  };

  auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
    EXPECT_EQ(op_to_count["Transpose"], 4);
  };

  // Generate a graph with QLinearAdd that broadcasts adds a 1D tensor to a
  // 2D tensor and verify that the transformer handles the mixed tensor ranks.
  TransformerTester(build_test_case,
                    check_nhwc_graph,
                    TransformerLevel::Level2,
                    TransformerLevel::Level3);
}

#endif  // DISABLE_CONTRIB_OPS

}  // namespace test
}  // namespace onnxruntime
