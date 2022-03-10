// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/compute_capability.h"
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/mlas/inc/mlas.h"
#include "core/optimizer/qdq_transformer/qdq_final_cleanup.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"

#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"

#include "qdq_test_utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4127)
#endif  // #if defined(_MSC_VER)

#ifdef USE_NNAPI
#include "core/providers/shared/node_unit/node_unit.h"
#endif  // #ifdef USE_NNAPI

namespace onnxruntime {
namespace test {

static std::vector<std::string> GetNodeOpTypesInTopologicalOrder(const Graph& graph) {
  std::vector<std::string> op_types{};
  GraphViewer graph_viewer{graph};
  const auto& ordering = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : ordering) {
    op_types.push_back(graph.GetNode(node_idx)->OpType());
  }
  return op_types;
}

#if !defined(DISABLE_CONTRIB_OPS)

template <typename InputType, typename WeightType, typename BiasType, typename OutputType>
void QDQTransformerConvTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if constexpr (std::is_same<InputType, OutputType>::value &&
                    std::is_same<BiasType, int32_t>::value &&
                    (std::is_same<InputType, uint8_t>::value ||
                     QDQIsInt8Allowed() && std::is_same<WeightType, int8_t>::value)) {
        EXPECT_EQ(op_to_count["QLinearConv"], 1);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
      } else {
        EXPECT_EQ(op_to_count["Conv"], 1);
        EXPECT_EQ(op_to_count["QLinearConv"], 0);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 4);
      }
    };

    TransformerTester(BuildQDQConvTestCase<InputType, WeightType, BiasType, OutputType>(input_shape, weights_shape),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

TEST(QDQTransformerTests, Conv_U8X8U8) {
  QDQTransformerConvTests<uint8_t, uint8_t, int32_t, uint8_t>();
  QDQTransformerConvTests<uint8_t, int8_t, int32_t, uint8_t>();
}

TEST(QDQTransformerTests, Conv_U8X8U8_Bias_Not_i32) {
  // bias not int32_t
  QDQTransformerConvTests<uint8_t, uint8_t, int8_t, uint8_t>();
  QDQTransformerConvTests<uint8_t, int8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Conv_U8X8S8) {
  // output not uint8_t
  QDQTransformerConvTests<uint8_t, uint8_t, int32_t, int8_t>();
  QDQTransformerConvTests<uint8_t, int8_t, int32_t, int8_t>();
}

TEST(QDQTransformerTests, Conv_S8X8U8) {
  // input not uint8_t
  QDQTransformerConvTests<int8_t, uint8_t, int32_t, uint8_t>();
  QDQTransformerConvTests<int8_t, int8_t, int32_t, uint8_t>();
}

TEST(QDQTransformerTests, Conv_S8X8S8) {
  // input not uint8_t and output not uint8_t
  QDQTransformerConvTests<int8_t, uint8_t, int32_t, int8_t>();
  QDQTransformerConvTests<int8_t, int8_t, int32_t, int8_t>();
}

TEST(QDQTransformerTests, ConvMaxPoolReshape_UInt8) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape, int opset_version) {
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

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], opset_version < 12 ? 2 : 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], opset_version < 12 ? 1 : 0);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      opset_version);
  };

  test_case({1, 12, 37}, {32, 12, 5}, 11);
  test_case({1, 12, 37}, {32, 12, 5}, 12);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, 11);
  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, 12);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, 11);
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3}, 12);
}

TEST(QDQTransformerTests, ConvMaxPoolReshape_Int8) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output);
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
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(reshape_output, .0039f, 7, output_arg);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0039f, 135, output_arg);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const std::vector<std::string> expected_op_types_in_order{
          "QuantizeLinear",
          "QLinearConv",
          "MaxPool",
          "Reshape"};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph());
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

template <typename InputType, typename OutputType>
void QDQTransformerAveragePoolTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if constexpr (std::is_same<InputType, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
        EXPECT_EQ(op_to_count["AveragePool"], 0);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 0);
        EXPECT_EQ(op_to_count["AveragePool"], 1);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 2);
      }
    };

    TransformerTester(BuildQDQAveragePoolTestCase<InputType, OutputType>(input_shape),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37});
  test_case({1, 23, 13, 13});
  test_case({1, 22, 11, 13, 15});
}

TEST(QDQTransformerTests, AveragePool_S8S8) {
  QDQTransformerAveragePoolTests<int8_t, int8_t>();
}

TEST(QDQTransformerTests, AveragePool_U8U8) {
  QDQTransformerAveragePoolTests<uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, AveragePool_S8U8) {
  QDQTransformerAveragePoolTests<int8_t, uint8_t>();
}

TEST(QDQTransformerTests, AveragePool_U8S8) {
  QDQTransformerAveragePoolTests<uint8_t, int8_t>();
}

template <typename InputType, typename OutputType>
void QDQTransformerGlobalAveragePoolTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if constexpr (std::is_same<InputType, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearGlobalAveragePool"], 1);
        EXPECT_EQ(op_to_count["GlobalAveragePool"], 0);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearGlobalAveragePool"], 0);
        EXPECT_EQ(op_to_count["GlobalAveragePool"], 1);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 2);
      }
    };

    TransformerTester(BuildQDQGlobalAveragePoolTestCase<InputType, OutputType>(input_shape),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37});
  test_case({1, 23, 13, 13});
  test_case({1, 22, 11, 13, 15});
}

TEST(QDQTransformerTests, GlobalAveragePool_S8S8) {
  QDQTransformerAveragePoolTests<int8_t, int8_t>();
}

TEST(QDQTransformerTests, GlobalAveragePool_U8U8) {
  QDQTransformerAveragePoolTests<uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, GlobalAveragePool_S8U8) {
  QDQTransformerAveragePoolTests<int8_t, uint8_t>();
}

TEST(QDQTransformerTests, GlobalAveragePool_U8S8) {
  QDQTransformerAveragePoolTests<uint8_t, int8_t>();
}

template <typename Input1Type, typename Input2Type, typename OutputType>
void QDQTransformerBinaryOpTests(const std::string& op_type) {
  auto test_case = [&](const std::vector<int64_t>& input_shape) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if (std::is_same<Input1Type, Input2Type>::value &&
          std::is_same<Input1Type, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinear" + op_type], 1);
        EXPECT_EQ(op_to_count[op_type], 0);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinear" + op_type], 0);
        EXPECT_EQ(op_to_count[op_type], 1);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 3);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 3);
      }
    };

    TransformerTester(BuildBinaryOpTestCase<Input1Type, Input2Type, OutputType>(input_shape, op_type),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37});
  test_case({1, 23, 13, 13});
  test_case({1, 22, 11, 13, 15});
}

TEST(QDQTransformerTests, Add) {
  QDQTransformerBinaryOpTests<uint8_t, uint8_t, uint8_t>("Add");
  QDQTransformerBinaryOpTests<int8_t, int8_t, int8_t>("Add");
}

TEST(QDQTransformerTests, Add_Have_Different_Types) {
  QDQTransformerBinaryOpTests<uint8_t, int8_t, int8_t>("Add");
  QDQTransformerBinaryOpTests<uint8_t, uint8_t, int8_t>("Add");
  QDQTransformerBinaryOpTests<uint8_t, int8_t, uint8_t>("Add");
  QDQTransformerBinaryOpTests<int8_t, int8_t, int8_t>("Add");
  QDQTransformerBinaryOpTests<int8_t, uint8_t, int8_t>("Add");
  QDQTransformerBinaryOpTests<int8_t, int8_t, uint8_t>("Add");
}

TEST(QDQTransformerTests, Mul) {
  QDQTransformerBinaryOpTests<uint8_t, uint8_t, uint8_t>("Mul");
  QDQTransformerBinaryOpTests<int8_t, int8_t, int8_t>("Mul");
}

TEST(QDQTransformerTests, Mul_Have_Different_Types) {
  QDQTransformerBinaryOpTests<uint8_t, int8_t, int8_t>("Mul");
  QDQTransformerBinaryOpTests<uint8_t, uint8_t, int8_t>("Mul");
  QDQTransformerBinaryOpTests<uint8_t, int8_t, uint8_t>("Mul");
  QDQTransformerBinaryOpTests<int8_t, int8_t, int8_t>("Mul");
  QDQTransformerBinaryOpTests<int8_t, uint8_t, int8_t>("Mul");
  QDQTransformerBinaryOpTests<int8_t, int8_t, uint8_t>("Mul");
}

template <typename Input1Type, typename Input2Type, typename OutputType>
void QDQTransformerMatMulTests(bool has_output_q) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      typedef std::numeric_limits<Input1Type> Input1Limits;
      typedef std::numeric_limits<Input2Type> Input2Limits;
      typedef std::numeric_limits<OutputType> OutputTypeLimits;

      // add QDQ 1
      auto* q1_output = builder.MakeIntermediate();
      auto* dq1_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input1Type>(input1_arg,
                                                .039f,
                                                (Input1Limits::max() + Input1Limits::min()) / 2 + 1,
                                                q1_output);
      builder.AddDequantizeLinearNode<Input1Type>(q1_output,
                                                  .039f,
                                                  (Input2Limits::max() + Input1Limits::min()) / 2 + 1,
                                                  dq1_output);

      // add QDQ 2
      auto* q2_output = builder.MakeIntermediate();
      auto* dq2_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input2Type>(input2_arg,
                                                .04f,
                                                (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                q2_output);
      builder.AddDequantizeLinearNode<Input2Type>(q2_output,
                                                  .04f,
                                                  (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                  dq2_output);

      if (has_output_q) {
        // add binary operator
        auto* matmul_op_output = builder.MakeIntermediate();
        builder.AddNode("MatMul", {dq1_output, dq2_output}, {matmul_op_output});

        // add QDQ output
        auto* q3_output = builder.MakeIntermediate();
        builder.AddQuantizeLinearNode<OutputType>(matmul_op_output,
                                                  .039f,
                                                  (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                  q3_output);
        builder.AddDequantizeLinearNode<OutputType>(q3_output,
                                                    .039f,
                                                    (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                    output_arg);
      } else {
        builder.AddNode("MatMul", {dq1_output, dq2_output}, {output_arg});
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if (has_output_q) {
        if constexpr (std::is_same<Input1Type, OutputType>::value &&
                      (std::is_same<Input1Type, uint8_t>::value ||
                       QDQIsInt8Allowed() && std::is_same<Input2Type, int8_t>::value)) {
          EXPECT_EQ(op_to_count["QLinearMatMul"], 1);
          EXPECT_EQ(op_to_count["MatMul"], 0);
          EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
          EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
        } else {
          EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
          EXPECT_EQ(op_to_count["MatMul"], 1);
          EXPECT_EQ(op_to_count["QuantizeLinear"], 3);
          EXPECT_EQ(op_to_count["DequantizeLinear"], 3);
        }
      } else {
        if constexpr (std::is_same<Input1Type, uint8_t>::value ||
                      (QDQIsInt8Allowed() && std::is_same<Input2Type, int8_t>::value)) {
          EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
          EXPECT_EQ(op_to_count["MatMul"], 0);
          EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
          EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
        } else {
          EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 0);
          EXPECT_EQ(op_to_count["MatMul"], 1);
          EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
          EXPECT_EQ(op_to_count["DequantizeLinear"], 2);
        }
      }
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 2, 2}, {1, 2, 4});
  test_case({1, 23, 13, 13}, {13, 13});
  test_case({1, 22, 11, 13, 15}, {1, 22, 11, 15, 15});
}

TEST(QDQTransformerTests, MatMul_U8U8U8) {
  QDQTransformerMatMulTests<uint8_t, uint8_t, uint8_t>(false);
  QDQTransformerMatMulTests<uint8_t, uint8_t, uint8_t>(true);
}

TEST(QDQTransformerTests, MatMul_U8S8S8) {
  QDQTransformerMatMulTests<uint8_t, int8_t, int8_t>(false);
  QDQTransformerMatMulTests<uint8_t, int8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_U8U8S8) {
  QDQTransformerMatMulTests<uint8_t, uint8_t, int8_t>(false);
  QDQTransformerMatMulTests<uint8_t, uint8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_U8S8U8) {
  QDQTransformerMatMulTests<uint8_t, int8_t, uint8_t>(false);
  QDQTransformerMatMulTests<uint8_t, int8_t, uint8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8S8S8) {
  QDQTransformerMatMulTests<int8_t, int8_t, int8_t>(false);
  QDQTransformerMatMulTests<int8_t, int8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8U8U8) {
  QDQTransformerMatMulTests<int8_t, uint8_t, uint8_t>(false);
  QDQTransformerMatMulTests<int8_t, uint8_t, uint8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8U8S8) {
  QDQTransformerMatMulTests<int8_t, uint8_t, int8_t>(false);
  QDQTransformerMatMulTests<int8_t, uint8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8S8U8) {
  QDQTransformerMatMulTests<int8_t, int8_t, uint8_t>(false);
  QDQTransformerMatMulTests<int8_t, int8_t, uint8_t>(true);
}

template <typename Input1Type, typename Input2Type, typename OutputType, typename BiasType = int32_t>
void QDQTransformerGemmTests(bool has_output_q, bool has_bias, bool beta_not_one = false) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      typedef std::numeric_limits<Input1Type> Input1Limits;
      typedef std::numeric_limits<Input2Type> Input2Limits;
      typedef std::numeric_limits<OutputType> OutputTypeLimits;

      std::vector<NodeArg*> input_args;

      // add QDQ A
      auto* q1_output = builder.MakeIntermediate();
      auto* dq1_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input1Type>(input1_arg,
                                                .039f,
                                                (Input1Limits::max() + Input1Limits::min()) / 2 + 1,
                                                q1_output);
      builder.AddDequantizeLinearNode<Input1Type>(q1_output,
                                                  .039f,
                                                  (Input2Limits::max() + Input1Limits::min()) / 2 + 1,
                                                  dq1_output);

      input_args.push_back(dq1_output);

      // add QDQ B
      auto* q2_output = builder.MakeIntermediate();
      auto* dq2_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input2Type>(input2_arg,
                                                .04f,
                                                (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                q2_output);
      builder.AddDequantizeLinearNode<Input2Type>(q2_output,
                                                  .04f,
                                                  (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                  dq2_output);
      input_args.push_back(dq2_output);

      if (has_bias) {
        auto* dq_bias_output = builder.MakeIntermediate();
        auto* bias = builder.MakeInitializer<BiasType>({input2_shape[1]}, static_cast<BiasType>(0), static_cast<BiasType>(127));
        builder.AddDequantizeLinearNode<BiasType>(bias, 0.00156f,
                                                  0,
                                                  dq_bias_output);
        input_args.push_back(dq_bias_output);
      }

      Node* gemm_node = nullptr;

      if (has_output_q) {
        auto* gemm_op_output = builder.MakeIntermediate();
        gemm_node = &builder.AddNode("Gemm", input_args, {gemm_op_output});

        // add QDQ output
        auto* q3_output = builder.MakeIntermediate();
        builder.AddQuantizeLinearNode<OutputType>(gemm_op_output,
                                                  .039f,
                                                  (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                  q3_output);
        builder.AddDequantizeLinearNode<OutputType>(q3_output,
                                                    .039f,
                                                    (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                    output_arg);
      } else {
        gemm_node = &builder.AddNode("Gemm", input_args, {output_arg});
      }

      if (beta_not_one) {
        gemm_node->AddAttribute("beta", 2.0f);
      }
    };

    auto check_binary_op_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if ((!has_output_q || std::is_same_v<Input1Type, OutputType>)&&(!has_bias || (std::is_same_v<BiasType, int32_t> && !beta_not_one)) &&
          (std::is_same_v<Input1Type, uint8_t> || std::is_same_v<Input2Type, int8_t>)) {
        EXPECT_EQ(op_to_count["com.microsoft.QGemm"], 1);
        EXPECT_EQ(op_to_count["Gemm"], 0);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
        EXPECT_EQ(op_to_count["DequantizeLinear"], has_output_q ? 1 : 0);
      } else {
        int q_count = 2;   // Q for A and B
        int dq_count = 2;  // DQ for A and B
        if (has_bias) {
          dq_count++;
        }
        if (has_output_q) {
          q_count++;
          dq_count++;
        }
        EXPECT_EQ(op_to_count["com.microsoft.QGemm"], 0);
        EXPECT_EQ(op_to_count["Gemm"], 1);
        EXPECT_EQ(op_to_count["QuantizeLinear"], q_count);
        EXPECT_EQ(op_to_count["DequantizeLinear"], dq_count);
      }
    };

    TransformerTester(build_test_case,
                      check_binary_op_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({2, 2}, {2, 4});
  test_case({13, 15}, {15, 15});
}

template <typename Input1Type, typename Input2Type, typename OutputType, typename BiasType = int32_t>
void QDQTransformerGemmTests() {
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, false);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, false);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, false, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, true, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, false, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, true, true);
}

TEST(QDQTransformerTests, Gemm_U8U8U8) {
  QDQTransformerGemmTests<uint8_t, uint8_t, uint8_t>();
  QDQTransformerGemmTests<uint8_t, uint8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_U8S8S8) {
  QDQTransformerGemmTests<uint8_t, int8_t, int8_t>();
  QDQTransformerGemmTests<uint8_t, int8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_U8U8S8) {
  QDQTransformerGemmTests<uint8_t, uint8_t, int8_t>();
  QDQTransformerGemmTests<uint8_t, uint8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_U8S8U8) {
  QDQTransformerGemmTests<uint8_t, int8_t, uint8_t>();
  QDQTransformerGemmTests<uint8_t, int8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8S8S8) {
  QDQTransformerGemmTests<int8_t, int8_t, int8_t>();
  QDQTransformerGemmTests<int8_t, int8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8U8U8) {
  QDQTransformerGemmTests<int8_t, uint8_t, uint8_t>();
  QDQTransformerGemmTests<int8_t, uint8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8U8S8) {
  QDQTransformerGemmTests<int8_t, uint8_t, int8_t>();
  QDQTransformerGemmTests<int8_t, uint8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8S8U8) {
  QDQTransformerGemmTests<int8_t, int8_t, uint8_t>();
  QDQTransformerGemmTests<int8_t, int8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gather) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int64_t>(input1_shape, 0, weights_shape[0] - 1);
      auto* output_arg = builder.MakeOutput();

      // add Gather
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -128, 127);
      auto* dq_w_output = builder.MakeIntermediate();
      auto* gather_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, 1, dq_w_output);
      builder.AddNode("Gather", {dq_w_output, input1_arg}, {gather_output});

      // add Q
      builder.AddQuantizeLinearNode<int8_t>(gather_output, .003f, 1, output_arg);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["Gather"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  test_case({12, 37}, {24, 12});
}

TEST(QDQTransformerTests, Transpose) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& perms) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(BuildQDQTransposeTestCase<int8_t, int8_t>(input_shape, perms),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2);
  };

  test_case({2, 13, 12, 37}, {0, 3, 1, 2});
}

TEST(QDQTransformerTests, Transpose_No_Fusion) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int8_t>(input1_shape, -128, 127);
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(input1_arg, .003f, 1, dq_output);

      // add Transpose
      auto* transpose_output = builder.MakeOutput();  // transpose output is graph output
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Q
      builder.AddQuantizeLinearNode<int8_t>(transpose_output, .003f, 1, output_arg);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  test_case({2, 13, 12, 37}, {0, 3, 1, 2});
}

TEST(QDQTransformerTests, Resize) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape,
                       const std::vector<int64_t>& sizes_shape) {
    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["Resize"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(BuildQDQResizeTestCase(input1_shape, sizes_shape),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2);
  };

  RandomValueGenerator rand_gen{optional<RandomValueGenerator::RandomSeedType>{2345}};
  test_case({2, 13, 12, 37}, rand_gen.Uniform<int64_t>(std::vector<int64_t>{4}, 1, 16));
}

TEST(QDQTransformerTests, Resize_No_Fusion) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& sizes_shape,
                       const std::vector<int64_t>& concat_input2_shape,
                       const int64_t axis) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* roi = builder.MakeInitializer<float>({0}, {});
      auto* scales = builder.MakeInitializer<float>({0}, {});
      auto* sizes = builder.MakeInitializer<int64_t>(sizes_shape, {1, 8, 128, 128});
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input_arg, .003f, 1, dq_output);

      // add Resize
      auto* resize_output = builder.MakeIntermediate();
      builder.AddNode("Resize", {dq_output, roi, scales, sizes}, {resize_output});

      // add Concat
      std::vector<NodeArg*> concat_input_args;
      concat_input_args.push_back(resize_output);
      concat_input_args.push_back(builder.MakeInput<float>(concat_input2_shape,
                                                           std::numeric_limits<float>::min(),
                                                           std::numeric_limits<float>::max()));
      auto* concat_output = builder.MakeIntermediate();
      Node& concat_node = builder.AddNode("Concat", concat_input_args, {concat_output});
      concat_node.AddAttribute("axis", axis);

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(resize_output, .003f, 1, output_arg);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["Resize"], 1);
      EXPECT_EQ(op_to_count["Concat"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case, check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2);
  };

  test_case({1, 8, 64, 64}, {4}, {1, 4, 128, 128}, 1);
}

TEST(QDQTransformerTests, ResizeReshapeSqueezeUnsqueeze) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& sizes_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape,
                                                 std::numeric_limits<float>::min(),
                                                 std::numeric_limits<float>::max());
      auto* roi = builder.MakeInitializer<float>({0}, {});
      auto* scales = builder.MakeInitializer<float>({0}, {});
      auto* sizes = builder.MakeInitializer<int64_t>(sizes_shape, {1, 2, 52, 82});

      // add QDQ + Resize
      auto* qdq_input = AddQDQNodePair<uint8_t>(builder, input_arg, .003f, 1);
      auto* resize_output = builder.MakeIntermediate();
      builder.AddNode("Resize", {qdq_input, roi, scales, sizes}, {resize_output});

      // add QDQ + Reshape
      auto* qdq_resize_output = AddQDQNodePair<uint8_t>(builder, resize_output, .003f, 1);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({1, 2, 52, 82});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {qdq_resize_output, reshape_shape}, {reshape_output});

      // add QDQ + Squeeze
      auto* qdq_squeeze_output = AddQDQNodePair<uint8_t>(builder, reshape_output, .003f, 1);
      auto* squeeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* squeeze_output = builder.MakeIntermediate();
      builder.AddNode("Squeeze", {qdq_squeeze_output, squeeze_axes}, {squeeze_output});

      // add QDQ + Unsqueeze
      auto* qdq_unsqueeze_output = AddQDQNodePair<uint8_t>(builder, squeeze_output, .003f, 1);
      auto* unsqueeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* unsqueeze_output = builder.MakeIntermediate();
      builder.AddNode("Unsqueeze", {qdq_unsqueeze_output, unsqueeze_axes}, {unsqueeze_output});

      // add QDQ
      AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, unsqueeze_output, .003f, 1);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["Resize"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case, check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      13 /*opset_version*/);
  };

  test_case({1, 2, 26, 42}, {4});
}

TEST(QDQTransformerTests, ArgMax) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       int axis,
                       int keepdims,
                       int select_last_index) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input_arg, .003f, 1, dq_output);

      // add ArgMax
      Node& argmax_node = builder.AddNode("ArgMax", {dq_output}, {output_arg});
      argmax_node.AddAttribute("axis", static_cast<int64_t>(axis));
      argmax_node.AddAttribute("keepdims", static_cast<int64_t>(keepdims));
      argmax_node.AddAttribute("select_last_index", static_cast<int64_t>(select_last_index));
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["ArgMax"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case, check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      /* opset_version */ 13);
  };

  test_case({2, 13, 12, 37}, 1, 0, 0);
  test_case({2, 13, 12, 37}, 0, 1, 0);
  test_case({2, 13, 12, 37}, 0, 0, 1);
}

TEST(QDQTransformerTests, QLinearMatMul) {
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

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearMatMul"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

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

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["MatMul"], 1);
      EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

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

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["MatMul"], 1);
      EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 2);
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  test_case({12, 37}, {37, 12});
  test_case({23, 13, 13}, {13, 13});
  test_case({22, 11, 13, 15}, {15, 13});
}

TEST(QDQTransformerTests, MatMulIntegerToFloat) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<uint8_t>(input1_shape,
                                                    std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());
      auto* input2_arg = builder.MakeInput<uint8_t>(input2_shape,
                                                    std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output_1 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input1_arg, .0035f, 135, dq_output_1);

      auto* dq_output_2 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input2_arg, .0035f, 135, dq_output_2);

      builder.AddNode("MatMul", {dq_output_1, dq_output_2}, {output_arg});
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      1e-5 /*per_sample_tolerance*/,
                      1e-5 /*relative_per_sample_tolerance*/);
  };

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

    auto check_graph = [&](InferenceSessionWrapper& session) {
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

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

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
      auto* dq_averagepool_output = AddQDQNodePair<uint8_t>(builder, conv_output, .0035f, 135);
      auto* averagepool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("AveragePool", {dq_averagepool_output}, {averagepool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<uint8_t>(builder, averagepool_output, .0035f, 135);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output);
      builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
  };

  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

TEST(QDQTransformerTests, ConvAveragePoolReshape_Int8) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add QDQ + AveragePool
      auto* dq_averagepool_output = AddQDQNodePair<int8_t>(builder, conv_output, .0035f, 7);
      auto* averagepool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("AveragePool", {dq_averagepool_output}, {averagepool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<int8_t>(builder, averagepool_output, .0035f, 7);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(reshape_output, .0035f, 7, q_output);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
  };

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
      auto* dq_averagepool_output = AddQDQNodePair<int8_t>(builder, conv_output, .0035f, 7);
      auto* averagepool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("AveragePool", {dq_averagepool_output}, {averagepool_output});
      std::vector<int64_t> pads((weights_shape.size() - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(weights_shape.size() - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // add QDQ + Reshape
      auto* dq_reshape_output = AddQDQNodePair<int8_t>(builder, averagepool_output, .0035f, 7);
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {dq_reshape_output, reshape_shape}, {reshape_output});

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(reshape_output, .0035f, 7, q_output);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["Conv"], 1);
      EXPECT_EQ(op_to_count["QLinearConv"], 0);
      EXPECT_EQ(op_to_count["com.microsoft.QLinearAveragePool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 3);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/);
  };

  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

template <typename InputType, typename OutputType>
void QDQTransformerLeakyReluTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      // add QDQ + LeakyRelu
      auto* dq_output = AddQDQNodePair<InputType>(builder, input_arg, .0035f, 7);
      auto* leakyrelu_output = builder.MakeIntermediate();
      Node& leakyrelu_node = builder.AddNode("LeakyRelu", {dq_output}, {leakyrelu_output});
      leakyrelu_node.AddAttribute("alpha", 0.2f);

      // add QDQ output
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<OutputType>(leakyrelu_output,
                                                .0038f,
                                                std::numeric_limits<OutputType>::max() / 2,
                                                q_output);
      builder.AddDequantizeLinearNode<OutputType>(q_output,
                                                  .0039f,
                                                  std::numeric_limits<OutputType>::max() / 2,
                                                  output_arg);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if constexpr (std::is_same<InputType, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearLeakyRelu"], 1);
        EXPECT_EQ(op_to_count["LeakyRelu"], 0);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearLeakyRelu"], 0);
        EXPECT_EQ(op_to_count["LeakyRelu"], 1);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 2);
      }
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37});
  test_case({1, 23, 13, 13});
  test_case({1, 22, 11, 13, 15});
}

TEST(QDQTransformerTests, LeakyRelu_S8S8) {
  QDQTransformerLeakyReluTests<int8_t, int8_t>();
}

TEST(QDQTransformerTests, LeakyRelu_U8U8) {
  QDQTransformerLeakyReluTests<uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, LeakyRelu_S8U8) {
  QDQTransformerLeakyReluTests<int8_t, uint8_t>();
}

TEST(QDQTransformerTests, LeakyRelu_U8S8) {
  QDQTransformerLeakyReluTests<uint8_t, int8_t>();
}

template <typename InputType, typename OutputType>
void QDQTransformerSigmoidTests() {
  auto test_case = [&](const std::vector<int64_t>& input_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      // add QDQ + Sigmoid
      auto* dq_output = AddQDQNodePair<InputType>(builder, input_arg, .0035f, 7);
      auto* sigmoid_output = builder.MakeIntermediate();
      builder.AddNode("Sigmoid", {dq_output}, {sigmoid_output});

      // add QDQ output
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<OutputType>(sigmoid_output,
                                                .0038f,
                                                std::numeric_limits<OutputType>::max() / 2,
                                                q_output);
      builder.AddDequantizeLinearNode<OutputType>(q_output,
                                                  .0039f,
                                                  std::numeric_limits<OutputType>::max() / 2,
                                                  output_arg);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if constexpr (std::is_same<InputType, OutputType>::value) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearSigmoid"], 1);
        EXPECT_EQ(op_to_count["Sigmoid"], 0);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearSigmoid"], 0);
        EXPECT_EQ(op_to_count["Sigmoid"], 1);
        EXPECT_EQ(op_to_count["QuantizeLinear"], 2);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 2);
      }
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01 /*per_sample_tolerance*/,
                      0.01 /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({1, 12, 37});
  test_case({1, 23, 13, 13});
  test_case({1, 22, 11, 13, 15});
}

TEST(QDQTransformerTests, Sigmoid_S8S8) {
  QDQTransformerSigmoidTests<int8_t, int8_t>();
}

TEST(QDQTransformerTests, Sigmoid_U8U8) {
  QDQTransformerSigmoidTests<uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Sigmoid_S8U8) {
  QDQTransformerSigmoidTests<int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Sigmoid_U8S8) {
  QDQTransformerSigmoidTests<uint8_t, int8_t>();
}

TEST(QDQTransformerTests, ConvTranspose_QBackward) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output);
      builder.AddConvNode(dq_conv_output, dq_w_output, conv_output);

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {conv_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Q
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(transpose_output, .0035f, 7, q_output);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(transpose_output, .0035f, 135, q_output);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2);
  };

  test_case({1, 23, 13, 13}, {30, 23, 3, 3}, {0, 3, 1, 2});
}

TEST(QDQTransformerTests, QBackward_MutilpleSteps) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add QDQ + Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      auto* dq_conv_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output);
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

      // add Unsqueeze
      auto* unsqueeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* unsqueeze_output = builder.MakeIntermediate();
      builder.AddNode("Unsqueeze", {transpose_output, unsqueeze_axes}, {unsqueeze_output});

      // add Squeeze
      auto* squeeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* squeeze_output = builder.MakeIntermediate();
      builder.AddNode("Squeeze", {unsqueeze_output, squeeze_axes}, {squeeze_output});

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(squeeze_output, .0035f, 7, q_output);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(squeeze_output, .0035f, 135, q_output);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      13 /*opset_version*/);
  };

  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
}

TEST(QDQTransformerTests, ConvTranspose_DQForward) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

      // add QDQ
      auto* dq_output = AddQDQNodePair<int8_t>(builder, input_arg, .004f, 1);

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output);
      builder.AddConvNode(transpose_output, dq_w_output, conv_output);

      // add Q
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(conv_output, .0035f, 7, q_output);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(conv_output, .0035f, 135, q_output);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2);
  };

  test_case({1, 13, 13, 23}, {30, 23, 3, 3}, {0, 3, 1, 2});
}

TEST(QDQTransformerTests, DQForward_MutilpleSteps) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -64, 64);

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

      // add Unsqueeze
      auto* unsqueeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* unsqueeze_output = builder.MakeIntermediate();
      builder.AddNode("Unsqueeze", {maxpool_output, unsqueeze_axes}, {unsqueeze_output});

      // add Squeeze
      auto* squeeze_axes = builder.Make1DInitializer<int64_t>({0});
      auto* squeeze_output = builder.MakeIntermediate();
      builder.AddNode("Squeeze", {unsqueeze_output, squeeze_axes}, {squeeze_output});

      // add Conv
      auto* dq_w_output = builder.MakeIntermediate();
      auto* conv_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output);
      builder.AddConvNode(squeeze_output, dq_w_output, conv_output);

      // Reshape
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      auto* reshape_output = builder.MakeIntermediate();
      builder.AddNode("Reshape", {conv_output, reshape_shape}, {reshape_output});

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      if constexpr (QDQIsInt8Allowed()) {
        builder.AddQuantizeLinearNode<int8_t>(reshape_output, .0035f, 7, q_output);
        builder.AddDequantizeLinearNode<int8_t>(q_output, .0035f, 7, output_arg);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .0035f, 135, q_output);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, .0035f, 135, output_arg);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
      EXPECT_EQ(op_to_count["MaxPool"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 1);
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      13 /*opset_version*/);
  };

  test_case({1, 13, 13, 23}, {30, 23, 3, 3}, {0, 3, 1, 2});
}

TEST(QDQTransformerTests, Clip) {
  constexpr float epsilon = std::numeric_limits<float>::epsilon();

  auto test_case = [&](float scale, auto zero_point, int clip_count, int opset_version = 12) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<int8_t>({1, 32, 112, 112},
                                                  std::numeric_limits<int8_t>::min(),
                                                  std::numeric_limits<int8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(input_arg, .0035f, 7, dq_output);

      // add Clip
      auto* clip_output = builder.MakeIntermediate();
      constexpr float min = .0f;
      constexpr float max = 6.0f;
      if (opset_version >= 11) {
        auto* min_initializer = builder.MakeScalarInitializer<float>(min);
        auto* max_initializer = builder.MakeScalarInitializer<float>(max);
        builder.AddNode("Clip", {dq_output, min_initializer, max_initializer}, {clip_output});
      } else {
        Node& argmax_node = builder.AddNode("Clip", {dq_output}, {clip_output});
        argmax_node.AddAttribute("min", min);
        argmax_node.AddAttribute("max", max);
      }

      // add Q + DQ
      auto* q_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode(clip_output, scale, zero_point, q_output);
      builder.AddDequantizeLinearNode(q_output, scale, zero_point, output_arg);
    };

    auto check_clip_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
      EXPECT_EQ(op_to_count["Clip"], clip_count);
      EXPECT_EQ(op_to_count["DequantizeLinear"], 2);
    };

    TransformerTester(build_test_case, check_clip_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1,
                      opset_version,
                      epsilon,
                      epsilon);
  };

  test_case(.0235294122248888f, static_cast<int8_t>(-128), 0);  // [0, 6]
  test_case(.02f, static_cast<int8_t>(-128), 0);                // [0, 5.1]
  test_case(.03f, static_cast<int8_t>(-128), 1);                // [0, 7.65]
  test_case(.02f, static_cast<int8_t>(127), 1);                 // [-5.1 , 0]
  test_case(.02f, static_cast<int8_t>(0), 1);                   // [-2.56, 2.54]
  test_case(.04f, static_cast<int8_t>(-97), 1);                 // [-1.24, 8.96]
  test_case(.02352941176f, static_cast<uint8_t>(0), 0);         // [0, 6]
  test_case(.02f, static_cast<uint8_t>(0), 0);                  // [0, 5.1]
  test_case(.03f, static_cast<uint8_t>(0), 1);                  // [0, 7.65]
  test_case(.02f, static_cast<uint8_t>(255), 1);                // [-5.1, 0]
  test_case(.02f, static_cast<uint8_t>(128), 1);                // [-2.56, 2.54]
  test_case(.04f, static_cast<uint8_t>(31), 1);                 // [-1.24, 8.96]

  // opset_version = 10
  test_case(.02f, static_cast<int8_t>(-128), 0, 10);  // [0, 5.1]
  test_case(.03f, static_cast<int8_t>(-128), 1, 10);  // [0, 7.65]
  test_case(.02f, static_cast<uint8_t>(0), 0, 10);    // [0, 5.1]
  test_case(.03f, static_cast<uint8_t>(0), 1, 10);    // [0, 7.65]

  // difference between lower/upper and min/max are within epsilon
  test_case(epsilon, static_cast<int8_t>(-127), 0);              // [-epsilon, x] (x <= 6 + epsilon)
  test_case((6 + epsilon) / 255, static_cast<int8_t>(-128), 0);  // [0, 6 + epsilon]
  test_case(epsilon, static_cast<uint8_t>(1), 0);                // [-epsilon, x] (x <= 6 + epsilon)
  test_case((6 + epsilon) / 255, static_cast<uint8_t>(0), 0);    // [0, 6 + epsilon]
}

TEST(QDQTransformerTests, Concat) {
  auto test_case = [&](const std::vector<std::vector<int64_t>>& input_shapes,
                       int64_t axis,
                       bool has_input_float = false,
                       bool has_input_int8 = false,
                       bool has_output_int8 = false) {
    auto check_graph = [&input_shapes, &has_input_float, &has_input_int8, &has_output_int8](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if (has_input_float || has_input_int8 || has_output_int8) {
        EXPECT_EQ(op_to_count["com.microsoft.QLinearConcat"], 0);
      } else {
        EXPECT_EQ(op_to_count["QuantizeLinear"], static_cast<int>(input_shapes.size()));
        EXPECT_EQ(op_to_count["com.microsoft.QLinearConcat"], 1);
        EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
      }
    };

    TransformerTester(BuildQDQConcatTestCase(input_shapes,
                                             axis,
                                             has_input_float,
                                             has_input_int8,
                                             has_output_int8),
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.01f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()));
  };

  test_case({{1, 6, 36}, {1, 3, 36}}, 1);
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2);
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2, true);
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2, false, true);
  test_case({{1, 6, 36}, {1, 6, 8}, {1, 6, 2}}, 2, false, false, true);
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

TEST(QDQTransformerTests, QDQPropagation_QBackward) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       size_t maxpool_dim,
                       const std::vector<int64_t>& perms,
                       bool add_op_boundary,
                       bool include_zp) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      auto* transpose_input = add_op_boundary ? builder.MakeIntermediate() : input_arg;
      if (add_op_boundary) {
        // add Sign as boundary for QDQ propagation
        builder.AddNode("Sign", {input_arg}, {transpose_input});
      }

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {transpose_input}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add MaxPool
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {transpose_output}, {maxpool_output});
      std::vector<int64_t> pads((maxpool_dim - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(maxpool_dim - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // Reshape
      auto* reshape_output = builder.MakeIntermediate();
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      builder.AddNode("Reshape", {maxpool_output, reshape_shape}, {reshape_output});

      // add Q
      constexpr float qdq_scale = 0.004f;
      if (include_zp) {
        constexpr uint8_t qdq_zero_point = 129;
        builder.AddQuantizeLinearNode<uint8_t>(reshape_output, qdq_scale, qdq_zero_point, output_arg);
      } else {
        builder.AddQuantizeLinearNode(reshape_output, qdq_scale, output_arg);
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      std::vector<std::string> expected_op_types_in_order{};
      if (add_op_boundary) {
        expected_op_types_in_order.push_back("Sign");
      }
      expected_op_types_in_order.insert(
          expected_op_types_in_order.end(),
          {"QuantizeLinear", "DequantizeLinear",
           "Transpose",
           "QuantizeLinear", "DequantizeLinear",
           "MaxPool",
           "QuantizeLinear", "DequantizeLinear",
           "Reshape",
           "QuantizeLinear"});

      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph());
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, false);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, true);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, true, false);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, true, true);
}

TEST(QDQTransformerTests, QDQPropagation_DQForward) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       size_t maxpool_dim,
                       const std::vector<int64_t>& perms,
                       bool add_op_boundary,
                       bool include_zp) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      constexpr float qdq_scale = 0.004f;
      auto* dq_output = builder.MakeIntermediate();
      if (include_zp) {
        constexpr uint8_t qdq_zero_point = 129;
        builder.AddDequantizeLinearNode<uint8_t>(input_arg, qdq_scale, qdq_zero_point, dq_output);
      } else {
        builder.AddDequantizeLinearNode(input_arg, qdq_scale, dq_output);
      }

      // add Transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add MaxPool
      auto* maxpool_output = builder.MakeIntermediate();
      Node& pool_node = builder.AddNode("MaxPool", {transpose_output}, {maxpool_output});
      std::vector<int64_t> pads((maxpool_dim - 2) * 2, 1);
      pool_node.AddAttribute("pads", pads);
      std::vector<int64_t> kernel_shape(maxpool_dim - 2, 3);
      pool_node.AddAttribute("kernel_shape", kernel_shape);

      // Reshape
      auto* reshape_shape = builder.Make1DInitializer<int64_t>({-1, 0});
      auto* reshape_output = add_op_boundary ? builder.MakeIntermediate() : output_arg;
      builder.AddNode("Reshape", {maxpool_output, reshape_shape}, {reshape_output});

      if (add_op_boundary) {
        // add Sign as boundary for QDQ propagation
        builder.AddNode("Sign", {reshape_output}, {output_arg});
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      std::vector<std::string> expected_op_types_in_order{
          "DequantizeLinear",
          "Transpose",
          "QuantizeLinear", "DequantizeLinear",
          "MaxPool",
          "QuantizeLinear", "DequantizeLinear",
          "Reshape",
          "QuantizeLinear", "DequantizeLinear"};
      if (add_op_boundary) {
        expected_op_types_in_order.push_back("Sign");
      }

      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph());
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, false);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, false, true);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, true, false);
  test_case({1, 13, 13, 23}, 4, {0, 3, 1, 2}, true, true);
}

TEST(QDQTransformerTests, QDQPropagation_StopAtOtherQDQ) {
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

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const std::vector<std::string> expected_op_types_in_order{
          "QuantizeLinear", "DequantizeLinear",
          "Reshape",
          "QuantizeLinear"};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph());
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, false, false);
  test_case({1, 13, 13, 23}, false, true);
  test_case({1, 13, 13, 23}, true, false);
  test_case({1, 13, 13, 23}, true, true);
}

TEST(QDQTransformerTests, QDQPropagation_Q_No_Parent) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add transpose
      auto* transpose_output = builder.MakeIntermediate();
      Node& transpose_node = builder.AddNode("Transpose", {input_arg}, {transpose_output});
      transpose_node.AddAttribute("perm", perms);

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(transpose_output, .0035f, 135, output_arg);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const std::vector<std::string> expected_op_types_in_order{
          "QuantizeLinear", "DequantizeLinear",
          "Transpose",
          "QuantizeLinear"};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph());
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, {0, 2, 3, 1});
}

TEST(QDQTransformerTests, QDQPropagation_DQ_No_Children) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input_arg, .0035f, 135, dq_output);

      // add transpose
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {output_arg});
      transpose_node.AddAttribute("perm", perms);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const std::vector<std::string> expected_op_types_in_order{
          "DequantizeLinear",
          "Transpose",
          "QuantizeLinear", "DequantizeLinear"};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph());
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, {0, 2, 3, 1});
}

TEST(QDQTransformerTests, QDQPropagation_Per_Layer_No_Propagation) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& perms) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      auto* dq_scale = builder.Make1DInitializer(std::vector<float>(input_shape[1], 0.0035f));
      auto* dq_zp = builder.Make1DInitializer(std::vector<uint8_t>(input_shape[1], 135));
      builder.AddNode("DequantizeLinear", {input_arg, dq_scale, dq_zp}, {dq_output});

      // add transpose
      Node& transpose_node = builder.AddNode("Transpose", {dq_output}, {output_arg});
      transpose_node.AddAttribute("perm", perms);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const std::vector<std::string> expected_op_types_in_order{
          "DequantizeLinear",
          "Transpose"};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph());
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23}, {0, 2, 3, 1});
}

TEST(QDQTransformerTests, QDQPropagation_DQ_Q) {
  auto test_case = [&](const std::vector<int64_t>& input_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input_arg, .0035f, 135, dq_output);

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(dq_output, .0035f, 135, output_arg);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const std::vector<std::string> expected_op_types_in_order{
          "DequantizeLinear",
          "QuantizeLinear"};
      const auto op_types_in_order = GetNodeOpTypesInTopologicalOrder(session.GetGraph());
      EXPECT_EQ(op_types_in_order, expected_op_types_in_order);
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Default,
                      TransformerLevel::Level1);
  };

  test_case({1, 13, 13, 23});
}

TEST(QDQTransformerTests, QDQ_Selector_Test) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/transform/qdq_conv.onnx");

  SessionOptions so;
  // We want to keep the graph un-optimized to prevent QDQ transformer to kick in
  so.graph_optimization_level = TransformerLevel::Default;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_file_name));
  ASSERT_STATUS_OK(session_object.Initialize());
  const Graph& graph = session_object.GetGraph();
  const auto* conv_node = graph.GetNode(3);

  // Make sure node 3 is the conv node
  ASSERT_TRUE(nullptr != conv_node);
  ASSERT_EQ("Conv", conv_node->OpType());

  onnxruntime::QDQ::ConvNodeGroupSelector conv_selector;

  // Initialize SelectorManager
  QDQ::SelectorManager selector_mgr;

  // Create a GraphViewer covers the whole graph
  const GraphViewer whole_graph_viewer(graph);

  // Make sure the conv QDQ group is selected for the full graph
  {
    const auto result = conv_selector.GetQDQSelection(whole_graph_viewer, *conv_node);
    ASSERT_TRUE(result.has_value());
    const auto& qdq_group = *result;
    ASSERT_EQ(std::vector<NodeIndex>({0, 1, 2}), qdq_group.dq_nodes);
    ASSERT_EQ(NodeIndex(3), qdq_group.target_node);
    ASSERT_EQ(std::vector<NodeIndex>({4}), qdq_group.q_nodes);
  }

  // Check if SelectorManager get a conv qdq group selection as expected
  {
    const auto result = selector_mgr.GetQDQSelections(whole_graph_viewer);
    ASSERT_FALSE(result.empty());
    const auto& qdq_group = result.at(0);
    ASSERT_EQ(std::vector<NodeIndex>({0, 1, 2}), qdq_group.dq_nodes);
    ASSERT_EQ(NodeIndex(3), qdq_group.target_node);
    ASSERT_EQ(std::vector<NodeIndex>({4}), qdq_group.q_nodes);
  }

// The function GetAllNodeUnits is enabled for NNAPI EP only for now
#ifdef USE_NNAPI
  {
    // Get all the NodeUnits in the graph_viewer
    std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
    std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

    std::tie(node_unit_holder, node_unit_map) = GetAllNodeUnits(whole_graph_viewer);

    // We should get a single QDQ Node unit in the result
    ASSERT_EQ(1, node_unit_holder.size());
    ASSERT_EQ(5, node_unit_map.size());
    const auto& qdq_node_unit = *node_unit_holder[0];
    ASSERT_EQ(NodeUnit::Type::QDQGroup, qdq_node_unit.UnitType());

    ASSERT_EQ(3, qdq_node_unit.Inputs().size());
    ASSERT_EQ(1, qdq_node_unit.Outputs().size());
    ASSERT_EQ(conv_node, &qdq_node_unit.GetNode());

    const auto verify_io_def = [](const NodeUnitIODef& io_def, const Node& node) {
      const auto& op_type = node.OpType();
      const bool is_dq = op_type == "DequantizeLinear";
      const bool is_q = op_type == "QuantizeLinear";
      ASSERT_TRUE(is_dq || is_q);
      const auto input_defs = node.InputDefs();
      if (is_dq) {
        ASSERT_EQ(&io_def.node_arg, input_defs[0]);
      } else {  // is_q
        ASSERT_EQ(&io_def.node_arg, node.OutputDefs()[0]);
      }

      ASSERT_EQ(&io_def.quant_param->scale, input_defs[1]);

      // [optional] zero point should be consistent between NodeUnitIODef and Input/OutputDefs
      ASSERT_EQ(input_defs.size() == 3, !!io_def.quant_param->zero_point);
      if (input_defs.size() == 3)  // we have zero point
        ASSERT_EQ(io_def.quant_param->zero_point, input_defs[2]);
    };

    // We know the graph has 5 nodes, DQ_input, DQ_weight, DQ_bias, Conv, Q_output (index 0-4)
    verify_io_def(qdq_node_unit.Inputs()[0], *whole_graph_viewer.GetNode(0));   // DQ_input
    verify_io_def(qdq_node_unit.Inputs()[1], *whole_graph_viewer.GetNode(1));   // DQ_weight
    verify_io_def(qdq_node_unit.Inputs()[2], *whole_graph_viewer.GetNode(2));   // DQ_bias
    verify_io_def(qdq_node_unit.Outputs()[0], *whole_graph_viewer.GetNode(4));  // Q_output
  }
#endif  // #ifdef USE_NNAPI

  // Create a graph viewer covers part of the graph
  // Make sure the qdq conv selector will fail for the partial graph
  {
    // Get 3 nodes out of 5 nodes in the graph
    std::vector<const Node*> nodes{
        graph.GetNode(0),
        graph.GetNode(3),
        graph.GetNode(4),
    };

    // Generate the indexed subgraph
    const auto compute_capability = utils::MakeComputeCapability(
        whole_graph_viewer, nodes,
        []() { return "sub_graph"; },
        "Test Provider");

    const GraphViewer partial_graph_viewer(graph, *compute_capability->sub_graph);
    ASSERT_EQ(3, partial_graph_viewer.NumberOfNodes());

    // Check there is no qdq selection for the given nodes
    {
      const auto result = conv_selector.GetQDQSelection(partial_graph_viewer, *conv_node);
      ASSERT_FALSE(result.has_value());
    }

    // Check SelectorManager will get empty result
    {
      const auto result = selector_mgr.GetQDQSelections(partial_graph_viewer);
      ASSERT_TRUE(result.empty());
    }
  }
}

// test removal of Q->DQ pairs by QDQFinalCleanupTransformer
TEST(QDQTransformerTests, QDQFinalCleanupTransformer_Basic) {
  auto test_case = [&](const std::vector<std::vector<int64_t>>& input_shapes,
                       bool block_removal_of_last_dq = false,
                       bool block_removal_of_first_dq = false) {
    // create model with float input to multiple -> Q -> DQ -> Concat -> Q -> DQ -> output
    // If we enable cleanup and don't run the QDQ transformer we should drop all the Q->DQ pairs
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto input_count = input_shapes.size();
      std::vector<NodeArg*> input_args;
      std::vector<NodeArg*> q_input_args;
      for (size_t i = 0; i < input_count; i++) {
        input_args.push_back(builder.MakeInput<float>(input_shapes[i], -1.f, 1.f));
        q_input_args.push_back(AddQDQNodePair<uint8_t>(builder, input_args.back(), 0.05f, 128));

        if (i == 0 && block_removal_of_first_dq) {
          // add another edge to the DQ node
          auto* output = builder.MakeOutput();
          builder.AddNode("Identity", {q_input_args.back()}, {output});
        }
      }
      auto* concat_output = builder.MakeIntermediate();
      Node& concat_node = builder.AddNode("Concat", q_input_args, {concat_output});
      concat_node.AddAttribute("axis", int64_t(1));

      auto* q_concat_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(concat_output, 0.05f, 128, q_concat_output);

      auto* output_arg = builder.MakeOutput();
      Node& dq_node = builder.AddDequantizeLinearNode<uint8_t>(q_concat_output, 0.05f, 128, output_arg);

      if (block_removal_of_last_dq) {
        // add another edge to the DQ node
        auto* output = builder.MakeOutput();
        builder.AddNode("Identity", {dq_node.MutableOutputDefs()[0]}, {output});
      }
    };

    // if we block removal of the DQ node the Q node in the pair will not be removed either
    int expected_qdq_count = 0 + (block_removal_of_first_dq ? 1 : 0) + (block_removal_of_last_dq ? 1 : 0);

    auto check_graph = [expected_qdq_count](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QuantizeLinear"], expected_qdq_count);
      EXPECT_EQ(op_to_count["DequantizeLinear"], expected_qdq_count);
      EXPECT_EQ(op_to_count["Concat"], 1);
    };

    std::function<void(SessionOptions&)> func = [](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableQuantQDQCleanup, "1"));
    };

    // we increase the tolerance as removing the QDQ nodes means there's no round-trip to 8-bit and back
    // essentially rounding the input values.
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      0.025f /*per_sample_tolerance*/,
                      0.01f /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQFinalCleanupTransformer>(),
                      &func);
  };

  test_case({{1, 2, 4}, {1, 3, 4}});
  test_case({{1, 2, 4}, {1, 3, 4}}, true);         // block removal of first dq
  test_case({{1, 2, 4}, {1, 3, 4}}, false, true);  // block removal of last dq
  test_case({{1, 2, 4}, {1, 3, 4}}, true, true);   // block removal of first and last dq
}

// test removal when we have graph input -> Q -> DQ -> graph output
TEST(QDQTransformerTests, QDQFinalCleanupTransformer_GraphInputToOutput) {
  // create model with float input to -> Q -> DQ -> output
  auto build_test_case = [&](ModelTestBuilder& builder) {
    NodeArg* input = builder.MakeInput<float>({1, 2, 4}, -1.f, 1.f);
    NodeArg* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input, 0.05f, 128, q_output);
    auto* output_arg = builder.MakeOutput();
    builder.AddDequantizeLinearNode<uint8_t>(q_output, 0.05f, 128, output_arg);
  };

  // with the Q->DQ being dropped we should have inserted an Identity node
  // to connect the graph input to the graph output
  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    EXPECT_EQ(op_to_count["Identity"], 1);
  };

  std::function<void(SessionOptions&)> func = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableQuantQDQCleanup, "1"));
  };

  // we increase the tolerance as removing the QDQ nodes means there's no round-trip to 8-bit and back
  // essentially rounding the input values.
  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    12 /*opset_version*/,
                    0.025f /*per_sample_tolerance*/,
                    0.01f /*relative_per_sample_tolerance*/,
                    std::make_unique<QDQFinalCleanupTransformer>(),
                    &func);
}

}  // namespace test
}  // namespace onnxruntime
