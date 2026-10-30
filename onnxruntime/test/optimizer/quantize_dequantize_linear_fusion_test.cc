// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include "core/framework/int4.h"
#include "core/graph/constants.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/qdq_transformer/quantize_dequantize_linear_fusion.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

namespace {

constexpr const char* kFusedOp = "com.microsoft.QuantizeDequantizeLinear";

template <typename T>
T TestZeroPoint() {
  if constexpr (std::is_signed_v<T>) {
    return static_cast<T>(-5);
  } else {
    return static_cast<T>(32768);
  }
}

// X -> Mul -> Q -> DQ -> Add -> Y. The input range saturates part of the values.
template <typename T>
std::function<void(ModelTestBuilder&)> BuildInteriorPair(const std::vector<int64_t>& shape) {
  return [shape](ModelTestBuilder& builder) {
    NodeArg* x = builder.MakeInput<float>(shape, -20.0f, 20.0f);
    NodeArg* scaled = builder.MakeIntermediate();
    builder.AddNode("Mul", {x, builder.MakeScalarInitializer<float>(0.75f)}, {scaled});
    NodeArg* q = builder.MakeIntermediate();
    NodeArg* dq = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<T>(scaled, 0.0004f, TestZeroPoint<T>(), q);
    builder.AddDequantizeLinearNode<T>(q, 0.0004f, TestZeroPoint<T>(), dq);
    builder.AddNode("Add", {dq, builder.MakeScalarInitializer<float>(0.25f)}, {builder.MakeOutput()});
  };
}

void ExpectFused(const Graph& graph, int expected_fused) {
  const auto op_count = CountOpsInGraph(graph);
  EXPECT_EQ(OpCount(op_count, kFusedOp), expected_fused);
  EXPECT_EQ(OpCount(op_count, "QuantizeLinear"), 0);
  EXPECT_EQ(OpCount(op_count, "DequantizeLinear"), 0);
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "QuantizeDequantizeLinear") {
      EXPECT_EQ(node.GetExecutionProviderType(), kCpuExecutionProvider);
    }
  }
}

// Level2 keeps the Q/DQ nodes; Level3 adds the fusion. Outputs must be bitwise equal.
void RunPipeline(const std::function<void(ModelTestBuilder&)>& build,
                 const std::function<void(InferenceSessionWrapper&)>& check,
                 const std::function<void(SessionOptions&)>& add_session_options = {}) {
  TransformerTester(build, check, TransformerLevel::Level2, TransformerLevel::Level3, 21, 0.0, 0.0, nullptr,
                    add_session_options);
}

Status AssignAllNodesToCpu(Graph& graph) {
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  return Status::OK();
}

// Applies only the fusion and checks that the graph is unchanged.
void ExpectNotFused(const std::function<void(ModelTestBuilder&)>& build,
                    const std::function<Status(Graph&)>& pre = AssignAllNodesToCpu) {
  OpCountMap before;
  auto pre_checker = [&](Graph& graph) {
    ORT_RETURN_IF_ERROR(pre(graph));
    before = CountOpsInGraph(graph);
    return Status::OK();
  };
  auto post_checker = [&](Graph& graph) {
    const auto after = CountOpsInGraph(graph);
    EXPECT_EQ(OpCount(after, kFusedOp), 0);
    EXPECT_EQ(after, before);
    return Status::OK();
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, 21, DefaultLoggingManager().DefaultLogger(),
      std::make_unique<QuantizeDequantizeLinearFusion>(InlinedHashSet<std::string_view>{kCpuExecutionProvider}),
      TransformerLevel::Level3, 1, pre_checker, post_checker));
}

// X -> Mul -> Q16 -> {DQ, Transpose -> DQ, Slice -> Unsqueeze -> DQ, Split -> (DQ, DQ), Gather -> DQ,
// Reshape -> DQ}, each DQ followed by Relu -> output.
std::function<void(ModelTestBuilder&)> BuildDataMovementTree() {
  return [](ModelTestBuilder& builder) {
    NodeArg* x = builder.MakeInput<float>({2, 6, 40}, -15.0f, 15.0f);
    NodeArg* scaled = builder.MakeIntermediate();
    builder.AddNode("Mul", {x, builder.MakeScalarInitializer<float>(0.75f)}, {scaled});
    NodeArg* q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint16_t>(scaled, 0.0004f, static_cast<uint16_t>(30000), q);
    auto add_dq_relu = [&](NodeArg* quantized) {
      NodeArg* dq = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint16_t>(quantized, 0.0004f, static_cast<uint16_t>(30000), dq);
      builder.AddNode("Relu", {dq}, {builder.MakeOutput()});
    };
    auto add_op = [&](const std::string& op, std::vector<NodeArg*> inputs, size_t num_outputs = 1,
                      const NodeAttributes* attrs = nullptr) {
      std::vector<NodeArg*> outputs;
      for (size_t i = 0; i < num_outputs; ++i) {
        outputs.push_back(builder.MakeIntermediate());
      }
      builder.AddNode(op, inputs, outputs, "", attrs);
      return outputs;
    };

    add_dq_relu(q);

    NodeAttributes perm;
    utils::SetNodeAttribute(utils::MakeAttribute("perm", std::vector<int64_t>{0, 2, 1}), perm);
    add_dq_relu(add_op("Transpose", {q}, 1, &perm)[0]);

    NodeArg* sliced = add_op("Slice", {q, builder.Make1DInitializer<int64_t>({1}), builder.Make1DInitializer<int64_t>({5}),
                                       builder.Make1DInitializer<int64_t>({1})})[0];
    add_dq_relu(add_op("Unsqueeze", {sliced, builder.Make1DInitializer<int64_t>({0})})[0]);

    NodeAttributes split;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(2)), split);
    utils::SetNodeAttribute(utils::MakeAttribute("num_outputs", static_cast<int64_t>(2)), split);
    for (NodeArg* part : add_op("Split", {q}, 2, &split)) {
      add_dq_relu(part);
    }

    NodeAttributes gather;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(1)), gather);
    add_dq_relu(add_op("Gather", {q, builder.Make1DInitializer<int64_t>({0, 2})}, 1, &gather)[0]);

    add_dq_relu(add_op("Reshape", {q, builder.Make1DInitializer<int64_t>({2, 240})})[0]);
  };
}

}  // namespace

TEST(QuantizeDequantizeLinearFusionTests, ThroughDataMovementOps) {
  auto post = [](Graph& graph) {
    ExpectFused(graph, 1);
    const auto op_count = CountOpsInGraph(graph);
    for (const char* op : {"Transpose", "Slice", "Unsqueeze", "Split", "Gather", "Reshape"}) {
      EXPECT_EQ(OpCount(op_count, op), 1) << op;
    }
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "Relu" || node.OpType() == "Mul") {
        continue;
      }
      for (const NodeArg* output : node.OutputDefs()) {
        EXPECT_EQ(output->TypeAsProto()->tensor_type().elem_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
            << node.OpType() << " " << output->Name();
      }
    }
    return Status::OK();
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      BuildDataMovementTree(), 21, DefaultLoggingManager().DefaultLogger(),
      std::make_unique<QuantizeDequantizeLinearFusion>(InlinedHashSet<std::string_view>{kCpuExecutionProvider}),
      TransformerLevel::Level3, 1, AssignAllNodesToCpu, post));

  RunPipeline(BuildDataMovementTree(), [](InferenceSessionWrapper& session) { ExpectFused(session.GetGraph(), 1); });
}

TEST(QuantizeDequantizeLinearFusionTests, NotFused_DataMovementOutputsNotOnlyDQ) {
  // Q -> Transpose -> X, where X is a uint16 graph output, a DQ graph output, a DQ with another scale, or a Cast.
  for (const char* variant : {"quantized output", "dq output", "dq scale", "cast"}) {
    SCOPED_TRACE(variant);
    ExpectNotFused([variant](ModelTestBuilder& builder) {
      const std::string v = variant;
      NodeArg* q = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint16_t>(builder.MakeInput<float>({2, 8}, -1.0f, 1.0f), 0.001f,
                                              static_cast<uint16_t>(32768), q);
      NodeArg* dq = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint16_t>(q, 0.001f, static_cast<uint16_t>(32768), dq);
      builder.AddNode("Relu", {dq}, {builder.MakeOutput()});

      NodeArg* transposed = v == "quantized output" ? builder.MakeOutput() : builder.MakeIntermediate();
      builder.AddNode("Transpose", {q}, {transposed});
      if (v == "dq output" || v == "dq scale") {
        NodeArg* dq_out = v == "dq output" ? builder.MakeOutput() : builder.MakeIntermediate();
        builder.AddDequantizeLinearNode<uint16_t>(transposed, v == "dq scale" ? 0.002f : 0.001f,
                                                  static_cast<uint16_t>(32768), dq_out);
        if (v == "dq scale") {
          builder.AddNode("Relu", {dq_out}, {builder.MakeOutput()});
        }
      } else if (v == "cast") {
        NodeAttributes to;
        utils::SetNodeAttribute(utils::MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT)),
                                to);
        builder.AddNode("Cast", {transposed}, {builder.MakeOutput()}, "", &to);
      }
    });
  }
}

TEST(QuantizeDequantizeLinearFusionTests, InteriorPairIsBitExact) {
  // The larger shape is not a multiple of 128 elements and runs on several threads.
  for (const std::vector<int64_t>& shape : {std::vector<int64_t>{1, 7}, std::vector<int64_t>{3, 1037}}) {
    SCOPED_TRACE(::testing::Message() << "elements=" << shape[0] * shape[1]);
    auto check = [](InferenceSessionWrapper& session) { ExpectFused(session.GetGraph(), 1); };
    RunPipeline(BuildInteriorPair<uint16_t>(shape), check);
    RunPipeline(BuildInteriorPair<int16_t>(shape), check);
  }
}

TEST(QuantizeDequantizeLinearFusionTests, FanOutAndGraphOutputDQ) {
  // Q -> {DQ -> Relu -> Y0, DQ -> Sigmoid -> Y1, DQ -> Y2}: one fused node feeds both ops and produces Y2.
  auto build = [](ModelTestBuilder& builder) {
    NodeArg* x = builder.MakeInput<float>({4, 300}, -15.0f, 15.0f);
    NodeArg* q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint16_t>(x, 0.0004f, static_cast<uint16_t>(30000), q);
    for (const char* op : {"Relu", "Sigmoid"}) {
      NodeArg* dq = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint16_t>(q, 0.0004f, static_cast<uint16_t>(30000), dq);
      builder.AddNode(op, {dq}, {builder.MakeOutput()});
    }
    builder.AddDequantizeLinearNode<uint16_t>(q, 0.0004f, static_cast<uint16_t>(30000), builder.MakeOutput());
  };
  auto check = [](InferenceSessionWrapper& session) {
    const Graph& graph = session.GetGraph();
    ExpectFused(graph, 1);
    const Node* fused = nullptr;
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "QuantizeDequantizeLinear") {
        fused = &node;
      }
    }
    ASSERT_NE(fused, nullptr);
    EXPECT_TRUE(graph.IsOutput(fused->OutputDefs()[0]));
    EXPECT_EQ(fused->GetOutputEdgesCount(), 2u);
    EXPECT_TRUE(graph.IsInputsIncludingInitializers(fused->InputDefs()[0]));
  };
  RunPipeline(build, check);
}

TEST(QuantizeDequantizeLinearFusionTests, A16MatMulNBitsBoundaries) {
  // X -> Q16 -> DQ16 -> MatMul(DQ(W4)) -> Q16 -> DQ16 -> Y: both A16 boundaries of the MatMulNBits are fused.
  auto build = [](ModelTestBuilder& builder) {
    NodeArg* x = builder.MakeInput<float>({5, 64}, -1.0f, 1.0f);
    NodeArg* xq = builder.MakeIntermediate();
    NodeArg* xdq = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint16_t>(x, 0.0004f, static_cast<uint16_t>(32768), xq);
    builder.AddDequantizeLinearNode<uint16_t>(xq, 0.0004f, static_cast<uint16_t>(32768), xdq);

    NodeArg* w = builder.MakeInitializer<Int4x2>({64, 16}, Int4x2(Int4x2::min_val, 0), Int4x2(Int4x2::max_val, 0));
    NodeArg* w_dq = builder.MakeIntermediate();
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(1)), attrs);
    builder.AddNode("DequantizeLinear", {w, builder.MakeInitializer<float>({16}, 0.01f, 0.02f)}, {w_dq}, "", &attrs);

    NodeArg* mm = builder.MakeIntermediate();
    builder.AddNode("MatMul", {xdq, w_dq}, {mm});
    NodeArg* yq = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint16_t>(mm, 0.0005f, static_cast<uint16_t>(32768), yq);
    builder.AddDequantizeLinearNode<uint16_t>(yq, 0.0005f, static_cast<uint16_t>(32768), builder.MakeOutput());
  };
  auto check = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(OpCount(op_count, "com.microsoft.MatMulNBits"), 1);
    EXPECT_EQ(OpCount(op_count, "MatMul"), 0);
    ExpectFused(session.GetGraph(), 2);
  };
  RunPipeline(build, check);
}

TEST(QuantizeDequantizeLinearFusionTests, DisabledByDisableQuantQDQ) {
  auto check = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(OpCount(op_count, kFusedOp), 0);
    EXPECT_EQ(OpCount(op_count, "QuantizeLinear"), 1);
    EXPECT_EQ(OpCount(op_count, "DequantizeLinear"), 1);
  };
  auto options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableQuantQDQ, "1"));
  };
  RunPipeline(BuildInteriorPair<uint16_t>({2, 8}), check, options);
}

TEST(QuantizeDequantizeLinearFusionTests, NotFused_8Bit) {
  ExpectNotFused([](ModelTestBuilder& builder) {
    NodeArg* q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(builder.MakeInput<float>({2, 8}, -1.0f, 1.0f), 0.01f,
                                           static_cast<uint8_t>(128), q);
    NodeArg* dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint8_t>(q, 0.01f, static_cast<uint8_t>(128), dq);
    builder.AddNode("Relu", {dq}, {builder.MakeOutput()});
  });
}

TEST(QuantizeDequantizeLinearFusionTests, NotFused_DifferentDQParameters) {
  for (bool different_scale : {true, false}) {
    SCOPED_TRACE(different_scale ? "scale" : "zero point");
    ExpectNotFused([different_scale](ModelTestBuilder& builder) {
      NodeArg* q = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint16_t>(builder.MakeInput<float>({2, 8}, -1.0f, 1.0f), 0.001f,
                                              static_cast<uint16_t>(32768), q);
      NodeArg* dq = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint16_t>(q, different_scale ? 0.002f : 0.001f,
                                                static_cast<uint16_t>(different_scale ? 32768 : 32767), dq);
      builder.AddNode("Relu", {dq}, {builder.MakeOutput()});
    });
  }
}

TEST(QuantizeDequantizeLinearFusionTests, NotFused_QuantizedValueUsedElsewhere) {
  // The Q output is also a graph output, or feeds a non-DQ consumer.
  for (bool q_is_graph_output : {true, false}) {
    SCOPED_TRACE(q_is_graph_output ? "graph output" : "non-DQ consumer");
    ExpectNotFused([q_is_graph_output](ModelTestBuilder& builder) {
      NodeArg* q = q_is_graph_output ? builder.MakeOutput() : builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint16_t>(builder.MakeInput<float>({2, 8}, -1.0f, 1.0f), 0.001f,
                                              static_cast<uint16_t>(32768), q);
      if (!q_is_graph_output) {
        builder.AddNode("Identity", {q}, {builder.MakeOutput()});
      }
      NodeArg* dq = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint16_t>(q, 0.001f, static_cast<uint16_t>(32768), dq);
      builder.AddNode("Relu", {dq}, {builder.MakeOutput()});
    });
  }
}

TEST(QuantizeDequantizeLinearFusionTests, NotFused_TwoDQGraphOutputs) {
  ExpectNotFused([](ModelTestBuilder& builder) {
    NodeArg* q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<int16_t>(builder.MakeInput<float>({2, 8}, -1.0f, 1.0f), 0.001f,
                                           static_cast<int16_t>(0), q);
    builder.AddDequantizeLinearNode<int16_t>(q, 0.001f, static_cast<int16_t>(0), builder.MakeOutput());
    builder.AddDequantizeLinearNode<int16_t>(q, 0.001f, static_cast<int16_t>(0), builder.MakeOutput());
  });
}

TEST(QuantizeDequantizeLinearFusionTests, NotFused_NonConstantOrPerAxisParameters) {
  // Scale is a graph input.
  ExpectNotFused([](ModelTestBuilder& builder) {
    NodeArg* scale = builder.MakeInput<float>(std::vector<int64_t>{}, std::vector<float>{0.001f});
    NodeArg* zp = builder.MakeScalarInitializer<uint16_t>(32768);
    NodeArg* q = builder.MakeIntermediate();
    builder.AddNode("QuantizeLinear", {builder.MakeInput<float>({2, 8}, -1.0f, 1.0f), scale, zp}, {q});
    NodeArg* dq = builder.MakeIntermediate();
    builder.AddNode("DequantizeLinear", {q, scale, zp}, {dq});
    builder.AddNode("Relu", {dq}, {builder.MakeOutput()});
  });

  // Per-axis scale and zero point.
  ExpectNotFused([](ModelTestBuilder& builder) {
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(1)), attrs);
    const std::vector<float> scales{0.001f, 0.002f};
    const std::vector<uint16_t> zps{32768, 32768};
    NodeArg* q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint16_t>(builder.MakeInput<float>({4, 2}, -1.0f, 1.0f), scales, zps, q, &attrs);
    NodeArg* dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint16_t>(q, scales, zps, dq, &attrs);
    builder.AddNode("Relu", {dq}, {builder.MakeOutput()});
  });
}

TEST(QuantizeDequantizeLinearFusionTests, NotFused_NonCpuEp) {
  ExpectNotFused(BuildInteriorPair<uint16_t>({2, 8}), [](Graph& graph) {
    for (auto& node : graph.Nodes()) {
      node.SetExecutionProviderType(kCudaExecutionProvider);
    }
    return Status::OK();
  });
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
