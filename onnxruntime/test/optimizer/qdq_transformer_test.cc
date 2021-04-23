// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"

namespace onnxruntime {
namespace test {

template <typename T>
typename std::enable_if<IsTypeQuantLinearCompatible<T>::value, NodeArg*>::type
AddQDQNodePair(ModelTestBuilder& builder, NodeArg* q_input, float scale, T zp) {
  auto* q_output = builder.MakeIntermediate();
  auto* dq_output = builder.MakeIntermediate();
  builder.AddQuantizeLinearNode<T>(q_input, scale, zp, q_output);
  builder.AddDequantizeLinearNode<T>(q_output, scale, zp, dq_output);
  return dq_output;
}

template <typename T>
typename std::enable_if<IsTypeQuantLinearCompatible<T>::value, NodeArg*>::type
AddQDQNodePair(ModelTestBuilder& builder, NodeArg* q_input, float scale) {
  auto* q_output = builder.MakeIntermediate();
  auto* dq_output = builder.MakeIntermediate();
  builder.AddQuantizeLinearNode(q_input, scale, q_output);
  builder.AddDequantizeLinearNode<T>(q_output, scale, dq_output);
  return dq_output;
}

#ifndef DISABLE_CONTRIB_OPS

TEST(QDQTransformerTests, Conv) {
  // TODO: enable fully use_default_zp tests after fixing inference bug in QuantizeLinear
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape, bool use_default_zp) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      auto* conv_output = builder.MakeIntermediate();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      auto* dq_w_output = builder.MakeIntermediate();
      auto* dq_output = AddQDQNodePair<uint8_t>(builder, input_arg, .004f, 129);

      if (use_default_zp) {
        builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, dq_w_output);
      } else {
        builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      }

      builder.AddConvNode(dq_output, dq_w_output, conv_output);
      builder.AddQuantizeLinearNode<uint8_t>(conv_output, .0039f, 135, output_arg);
    };

    auto check_conv_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case, check_conv_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37}, {32, 12, 5}, true);
  test_case({1, 12, 37}, {32, 12, 5}, false);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, true);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, false);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, true);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, false);
}

TEST(QDQTransformerTests, ConvMaxPoolReshape_UInt8) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<uint8_t>(builder, input_arg, .004f, 129);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add QDQ + MaxPool
      auto* dq_maxpool_output = AddQDQNodePair<uint8_t>(builder, conv_output, .0039f, 135);
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {dq_maxpool_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<uint8_t>(builder, maxpool_output, .0039f, 135);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0039f, 135, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case, check_mp_reshape_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

TEST(QDQTransformerTests, ConvMaxPoolReshape_Int8) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add QDQ + MaxPool
      auto* dq_maxpool_output = AddQDQNodePair<int8_t>(builder, conv_output, .0039f, 7);
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {dq_maxpool_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<int8_t>(builder, maxpool_output, .0039f, 7);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0039f, 135, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case, check_mp_reshape_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

TEST(QDQTransformerTests, Add) {
  auto test_case = [&](const std::vector<int64_t>& input_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add QDQ + Add
      auto* add_output = builder.MakeIntermediate();
      auto* dq_add_output1 = AddQDQNodePair<uint8_t>(builder, input1_arg, .004f, 129);
      auto* dq_add_output2 = AddQDQNodePair<uint8_t>(builder, input2_arg, .004f, 129);
      builder.AddNode("Add", {dq_add_output1, dq_add_output2}, {add_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(add_output, .0039f, 135, output_arg);
    };

    auto check_add_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearAdd"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case, check_add_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37});
  test_case({1, 23, 13, 13});
  test_case({1, 22, 11, 13, 15});
}

TEST(QDQTransformerTests, Mul) {
  auto test_case = [&](const std::vector<int64_t>& input_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add QDQ + Mul
      auto* mul_output = builder.MakeIntermediate();
      auto* dq_mul_output1 = AddQDQNodePair<uint8_t>(builder, input1_arg, .004f, 129);
      auto* dq_mul_output2 = AddQDQNodePair<uint8_t>(builder, input2_arg, .004f, 129);
      builder.AddNode("Mul", {dq_mul_output1, dq_mul_output2}, {mul_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(mul_output, .0039f, 135, output_arg);
    };

    auto check_mul_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearMul"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case, check_mul_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37});
  test_case({1, 23, 13, 13});
  test_case({1, 22, 11, 13, 15});
}

TEST(QDQTransformerTests, MatMul) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add QDQ + MatMul
      auto* matmul_output = builder.MakeIntermediate();
      auto* dq_matmul_output1 = AddQDQNodePair<uint8_t>(builder, input1_arg, .004f, 129);
      auto* dq_matmul_output2 = AddQDQNodePair<uint8_t>(builder, input2_arg, .004f, 129);
      builder.AddNode("MatMul", {dq_matmul_output1, dq_matmul_output2}, {matmul_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(matmul_output, .0039f, 135, output_arg);
    };

    auto check_matmul_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearMatMul"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case, check_matmul_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({12, 37}, {37, 12});
  test_case({23, 13, 13}, {13, 13});
  test_case({22, 11, 13, 15}, {15, 13});
}

TEST(QDQTransformerTests, MatMul_No_Fusion) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add QDQ + MatMul
      auto* matmul_output = builder.MakeIntermediate();
      auto* dq_matmul_output1 = AddQDQNodePair<uint8_t>(builder, input1_arg, .004f, 129);
      builder.AddNode("MatMul", {dq_matmul_output1, input2_arg}, {matmul_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(matmul_output, .0039f, 135, output_arg);
    };

    auto check_matmul_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["MatMul"], 1);
      EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case, check_matmul_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({12, 37}, {37, 12});
  test_case({23, 13, 13}, {13, 13});
  test_case({22, 11, 13, 15}, {15, 13});
}

TEST(QDQTransformerTests, MatMul_1st_Input_Int8) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int8_t>(input1_shape, -128, 127);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add DQ with type int8
      auto* dq_output_1 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(input1_arg, .004f, 1, dq_output_1);

      // add QDQ + MatMul
      auto* matmul_output = builder.MakeIntermediate();
      auto* dq_matmul_output2 = AddQDQNodePair<uint8_t>(builder, input2_arg, .004f, 129);
      builder.AddNode("MatMul", {dq_output_1, dq_matmul_output2}, {matmul_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(matmul_output, .0039f, 135, output_arg);
    };

    auto check_matmul_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["MatMul"], 1);
      EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 2);
    };

    TransformerTester(build_test_case, check_matmul_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({12, 37}, {37, 12});
  test_case({23, 13, 13}, {13, 13});
  test_case({22, 11, 13, 15}, {15, 13});
}

TEST(QDQTransformerTests, ConvRelu) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape, bool is_zp_zero) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<uint8_t>(builder, input_arg, .004f, 129);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add Relu
      auto* relu_output = builder.MakeIntermediate();
      builder.AddNode("Relu", {conv_output}, {relu_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(relu_output, .0039f, is_zp_zero ? 0 : 1, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if (is_zp_zero) {
        EXPECT_EQ(op_to_count["QLinearConv"], 1);
        EXPECT_EQ(op_to_count["Conv"], 0);
        EXPECT_EQ(op_to_count["Relu"], 0);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
      } else {
        EXPECT_EQ(op_to_count["QLinearConv"], 0);
        EXPECT_EQ(op_to_count["Conv"], 0);
        EXPECT_EQ(op_to_count["Relu"], 0);
        EXPECT_EQ(op_to_count["com.microsoft.FusedConv"], 1);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 2);
      }
    };

    TransformerTester(build_test_case, check_mp_reshape_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37}, {32, 12, 5}, true);
  test_case({1, 12, 37}, {32, 12, 5}, false);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, true);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, false);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, true);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, false);
}

TEST(QDQTransformerTests, ConvAveragePoolReshape_UInt8) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<uint8_t>(builder, input_arg, .004f, 129);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add QDQ + AveragePool
      auto* dq_maxpool_output = AddQDQNodePair<uint8_t>(builder, conv_output, .0035f, 135);
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("AveragePool", {dq_maxpool_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<uint8_t>(builder, maxpool_output, .0035f, 135);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output);
      builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

TEST(QDQTransformerTests, ConvAveragePoolReshape_Int8) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add QDQ + AveragePool
      auto* dq_maxpool_output = AddQDQNodePair<int8_t>(builder, conv_output, .0035f, 7);
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("AveragePool", {dq_maxpool_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<int8_t>(builder, maxpool_output, .0035f, 7);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output);
      builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

TEST(QDQTransformerTests, ConvAveragePoolReshape_Int8_Fail) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<int8_t>(input_shape, -128, 127);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add DQ + Conv
      auto* dq_output = builder.MakeIntermediate();
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(input_arg, .004f, 1, dq_output);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      builder.AddConvNode(dq_output, dq_w_output, conv_output);

      // add QDQ + AveragePool
      auto* dq_maxpool_output = AddQDQNodePair<int8_t>(builder, conv_output, .0035f, 7);
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("AveragePool", {dq_maxpool_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<int8_t>(builder, maxpool_output, .0035f, 7);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output);
      builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["Conv"], 1);
      EXPECT_EQ(op_to_count["QLinearConv"], 0);
      EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 3);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

TEST(QDQTransformerTests, ConvTranspose_QBackward) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {conv_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Q
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(transpose_output, .0035f, 135, q_output);
      builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, {0, 3, 1, 2});
}

TEST(QDQTransformerTests, QBackward_MutilpleSteps) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add MaxPool
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {conv_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // Reshape
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {maxpool_output, reshape_shape}, {reshape_output});

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {reshape_output}, {transpose_output});
      transpose_node.AddAttribute("perm", std::vector<int64_t>({1, 0}));

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(transpose_output, .0035f, 135, q_output);
      builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
}

TEST(QDQTransformerTests, ConvTranspose_DQForward) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add QDQ
      auto* dq_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      builder.AddConvNode(transpose_output, dq_w_output, conv_output);

      // add Q
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(conv_output, .0035f, 135, q_output);
      builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 13, 13, 23}, {30, 23, 3, 3}, {0, 3, 1, 2});
}

TEST(QDQTransformerTests, DQForward_MutilpleSteps) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<uint8_t>(weights_shape, 0, 255);

      // add Transpose
      auto* qdq_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {qdq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add MaxPool
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {transpose_output}, {maxpool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(weight, .003f, 118, dq_w_output);
      builder.AddConvNode(maxpool_output, dq_w_output, conv_output);

      // Reshape
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {conv_output, reshape_shape}, {reshape_output});

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output);
      builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 13, 13, 23}, {30, 23, 3, 3}, {0, 3, 1, 2});
}

TEST(QDQTransformerTests, QDQPropagation_QDQCancelOut) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, size_t maxpool_dim, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add QDQ
      auto* qdq_output = AddQDQNodePair<uint8_t>(builder, input_arg, .004f, 129);

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {qdq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Q
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(transpose_output, .004f, 129, q_output);

      // add MaxPool
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {q_output}, {maxpool_output});
      std::vector<int64_t> pads((maxpool_dim - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(maxpool_dim - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // Reshape
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      builder.AddNode("Reshape", {maxpool_output, reshape_shape}, {output_arg});
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2});
}

TEST(QDQTransformerTests, QDQPropagation_QDQ_CancelOut_More) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, bool same_scale, bool same_zp) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add QDQ
      auto* qdq_output = AddQDQNodePair<uint8_t>(builder, input_arg, .004f, 129);

      // Reshape
      auto* reshape_output = builder.MakeIntermediate();
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      builder.AddNode("Reshape", {qdq_output, reshape_shape}, {reshape_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, same_scale ? .004f : .0039f, same_zp ? 129 : 128, output_arg);
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], same_scale && same_zp ? 1 : 2);
      EXPECT_EQ(op_to_count["DequantizeLinear"], same_scale && same_zp ? 0 : 1);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 13, 13, 23}, false, false);
  test_case({1, 13, 13, 23}, false, true);
  test_case({1, 13, 13, 23}, true, false);
  test_case({1, 13, 13, 23}, true, true);
}

#endif  // DISABLE_CONTRIB_OPS

}  // namespace test
}  // namespace onnxruntime
