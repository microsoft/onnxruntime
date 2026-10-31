// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// DQ(weight) -> MatMul/Gemm -> MatMulNBits fusion inside INT16/UINT16 activation QDQ boundaries (CPU EP).
// Only the constant weight DQ is fused. Activation DQ / output Q nodes must stay connected to the same values,
// and INT8/UINT8 activation boundaries must keep going through the existing QDQ rules.

#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/int4.h"
#include "core/graph/constants.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
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

enum class WeightQuant {
  kInt4PerChannel,
  kUInt4PerChannel,
  kInt4Block32,
  kInt4Block128,
  kInt8PerChannel,
  kUInt4PerTensor,
};

enum class BiasKind {
  kNone,
  kFloat,
  kDQ,
};

enum class ActivationType {
  kUInt16,
  kInt16,
  kUInt8,
};

struct A16Case {
  ActivationType activation{ActivationType::kUInt16};
  bool input_boundary{true};
  bool output_boundary{true};
  WeightQuant weight{WeightQuant::kInt4PerChannel};
  bool gemm{false};
  BiasKind bias{BiasKind::kNone};
  // The activation DQ output also feeds an Identity that produces a graph output.
  bool share_activation_dq{false};
  // Gemm attributes that make the node incompatible with MatMulNBits.
  bool gemm_trans_b{false};
  std::optional<float> gemm_alpha;
  std::optional<float> gemm_beta;
  int64_t M{7};
  int64_t K{256};
  int64_t N{32};
};

constexpr float kActScale = 0.0004f;
constexpr float kOutScale = 0.0005f;

NodeArg* AddWeightDQ(ModelTestBuilder& builder, WeightQuant weight, int64_t K, int64_t N) {
  NodeArg* dq_output = builder.MakeIntermediate();
  NodeAttributes attrs;
  switch (weight) {
    case WeightQuant::kInt4PerChannel:
    case WeightQuant::kUInt4PerChannel: {
      NodeArg* w = weight == WeightQuant::kInt4PerChannel
                       ? builder.MakeInitializer<Int4x2>({K, N}, Int4x2(Int4x2::min_val, 0), Int4x2(Int4x2::max_val, 0))
                       : builder.MakeInitializer<UInt4x2>({K, N}, UInt4x2(UInt4x2::min_val, 0),
                                                          UInt4x2(UInt4x2::max_val, 0));
      NodeArg* s = builder.MakeInitializer<float>({N}, 0.005f, 0.02f);
      utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(1)), attrs);
      if (weight == WeightQuant::kUInt4PerChannel) {
        NodeArg* zp = builder.MakeInitializer<UInt4x2>({N}, UInt4x2(6, 0), UInt4x2(10, 0));
        builder.AddNode("DequantizeLinear", {w, s, zp}, {dq_output}, "", &attrs);
      } else {
        builder.AddNode("DequantizeLinear", {w, s}, {dq_output}, "", &attrs);
      }
      break;
    }
    case WeightQuant::kInt4Block32:
    case WeightQuant::kInt4Block128: {
      const int64_t block_size = weight == WeightQuant::kInt4Block32 ? 32 : 128;
      const int64_t k_blocks = (K + block_size - 1) / block_size;
      NodeArg* w = builder.MakeInitializer<Int4x2>({K, N}, Int4x2(Int4x2::min_val, 0), Int4x2(Int4x2::max_val, 0));
      NodeArg* s = builder.MakeInitializer<float>({k_blocks, N}, 0.005f, 0.02f);
      NodeArg* zp = builder.MakeInitializer<Int4x2>({k_blocks, N}, Int4x2(-2, 0), Int4x2(2, 0));
      utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), attrs);
      utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);
      builder.AddNode("DequantizeLinear", {w, s, zp}, {dq_output}, "", &attrs);
      break;
    }
    case WeightQuant::kInt8PerChannel: {
      NodeArg* w = builder.MakeInitializer<int8_t>({K, N}, static_cast<int8_t>(-127), static_cast<int8_t>(127));
      NodeArg* s = builder.MakeInitializer<float>({N}, 0.0005f, 0.002f);
      NodeArg* zp = builder.MakeInitializer<int8_t>({N}, std::vector<int8_t>(static_cast<size_t>(N), 0));
      utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(1)), attrs);
      builder.AddNode("DequantizeLinear", {w, s, zp}, {dq_output}, "", &attrs);
      break;
    }
    case WeightQuant::kUInt4PerTensor: {
      NodeArg* w = builder.MakeInitializer<UInt4x2>({K, N}, UInt4x2(UInt4x2::min_val, 0),
                                                    UInt4x2(UInt4x2::max_val, 0));
      NodeArg* s = builder.MakeScalarInitializer<float>(0.01f);
      NodeArg* zp = builder.MakeInitializer<UInt4x2>({}, std::vector<UInt4x2>{UInt4x2(8, 0)});
      builder.AddNode("DequantizeLinear", {w, s, zp}, {dq_output});
      break;
    }
  }
  return dq_output;
}

// Adds Q -> DQ with the requested activation type and returns the DQ output.
NodeArg* AddActivationQDQ(ModelTestBuilder& builder, NodeArg* input, ActivationType type, float scale) {
  NodeArg* q_output = builder.MakeIntermediate();
  NodeArg* dq_output = builder.MakeIntermediate();
  switch (type) {
    case ActivationType::kUInt16:
      builder.AddQuantizeLinearNode<uint16_t>(input, scale, static_cast<uint16_t>(32768), q_output);
      builder.AddDequantizeLinearNode<uint16_t>(q_output, scale, static_cast<uint16_t>(32768), dq_output);
      break;
    case ActivationType::kInt16:
      builder.AddQuantizeLinearNode<int16_t>(input, scale, static_cast<int16_t>(0), q_output);
      builder.AddDequantizeLinearNode<int16_t>(q_output, scale, static_cast<int16_t>(0), dq_output);
      break;
    case ActivationType::kUInt8:
      builder.AddQuantizeLinearNode<uint8_t>(input, scale * 256.0f, static_cast<uint8_t>(128), q_output);
      builder.AddDequantizeLinearNode<uint8_t>(q_output, scale * 256.0f, static_cast<uint8_t>(128), dq_output);
      break;
  }
  return dq_output;
}

// X -> [Q -> DQ] -> MatMul/Gemm(DQ(W)[, bias]) -> [Q -> DQ] -> Y
std::function<void(ModelTestBuilder&)> BuildA16Case(const A16Case& c) {
  return [c](ModelTestBuilder& builder) {
    NodeArg* x = builder.MakeInput<float>({c.M, c.K}, -1.0f, 1.0f);
    NodeArg* activation = c.input_boundary ? AddActivationQDQ(builder, x, c.activation, kActScale) : x;

    if (c.share_activation_dq) {
      builder.AddNode("Identity", {activation}, {builder.MakeOutput()});
    }

    NodeArg* weight = AddWeightDQ(builder, c.weight, c.K, c.N);
    NodeArg* target_output = c.output_boundary ? builder.MakeIntermediate() : builder.MakeOutput();

    if (c.gemm) {
      std::vector<NodeArg*> inputs{activation, weight};
      if (c.bias == BiasKind::kFloat) {
        inputs.push_back(builder.MakeInitializer<float>({c.N}, -0.5f, 0.5f));
      } else if (c.bias == BiasKind::kDQ) {
        NodeArg* bias_q = builder.MakeInitializer<int32_t>({c.N}, -1000, 1000);
        NodeArg* bias_dq = builder.MakeIntermediate();
        builder.AddDequantizeLinearNode<int32_t>(bias_q, 0.0002f, 0, bias_dq);
        inputs.push_back(bias_dq);
      }
      NodeAttributes attrs;
      if (c.gemm_trans_b) {
        utils::SetNodeAttribute(utils::MakeAttribute("transB", static_cast<int64_t>(1)), attrs);
      }
      if (c.gemm_alpha.has_value()) {
        utils::SetNodeAttribute(utils::MakeAttribute("alpha", *c.gemm_alpha), attrs);
      }
      if (c.gemm_beta.has_value()) {
        utils::SetNodeAttribute(utils::MakeAttribute("beta", *c.gemm_beta), attrs);
      }
      builder.AddNode("Gemm", inputs, {target_output}, "", &attrs);
    } else {
      builder.AddNode("MatMul", {activation, weight}, {target_output});
    }

    if (c.output_boundary) {
      NodeArg* q_output = builder.MakeIntermediate();
      NodeArg* y = builder.MakeOutput();
      if (c.activation == ActivationType::kInt16) {
        builder.AddQuantizeLinearNode<int16_t>(target_output, kOutScale, static_cast<int16_t>(0), q_output);
        builder.AddDequantizeLinearNode<int16_t>(q_output, kOutScale, static_cast<int16_t>(0), y);
      } else if (c.activation == ActivationType::kUInt16) {
        builder.AddQuantizeLinearNode<uint16_t>(target_output, kOutScale, static_cast<uint16_t>(32768), q_output);
        builder.AddDequantizeLinearNode<uint16_t>(q_output, kOutScale, static_cast<uint16_t>(32768), y);
      } else {
        builder.AddQuantizeLinearNode<uint8_t>(target_output, kOutScale * 256.0f, static_cast<uint8_t>(128),
                                               q_output);
        builder.AddDequantizeLinearNode<uint8_t>(q_output, kOutScale * 256.0f, static_cast<uint8_t>(128), y);
      }
    }
  };
}

int32_t QuantizedElemType(const Node& node) {
  // Q: quantized type is the output; DQ: quantized type is input 0.
  const NodeArg* arg = node.OpType() == "QuantizeLinear" ? node.OutputDefs()[0] : node.InputDefs()[0];
  return arg->TypeAsProto()->tensor_type().elem_type();
}

bool IsActivation16Bit(int32_t type) {
  return type == ONNX_NAMESPACE::TensorProto_DataType_UINT16 || type == ONNX_NAMESPACE::TensorProto_DataType_INT16;
}

bool IsActivation8Bit(int32_t type) {
  return type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 || type == ONNX_NAMESPACE::TensorProto_DataType_INT8;
}

struct QDQCounts {
  int q16{0};
  int dq16{0};
  int q8{0};
  int dq8{0};
  int dq_other{0};  // weight / bias DQs
};

QDQCounts CountQDQ(const Graph& graph) {
  QDQCounts counts;
  for (const auto& node : graph.Nodes()) {
    const bool is_q = node.OpType() == "QuantizeLinear";
    const bool is_dq = node.OpType() == "DequantizeLinear";
    if (!is_q && !is_dq) {
      continue;
    }
    const int32_t type = QuantizedElemType(node);
    // Activation DQs consume a non-initializer; weight/bias DQs consume constant initializers.
    const bool is_constant_input = is_dq && graph.GetConstantInitializer(node.InputDefs()[0]->Name(), true);
    if (is_dq && is_constant_input) {
      ++counts.dq_other;
    } else if (IsActivation16Bit(type)) {
      ++(is_q ? counts.q16 : counts.dq16);
    } else if (IsActivation8Bit(type)) {
      ++(is_q ? counts.q8 : counts.dq8);
    } else {
      ++counts.dq_other;
    }
  }
  return counts;
}

const Node* FindSingleNode(const Graph& graph, const std::string& op_type) {
  const Node* found = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == op_type) {
      if (found != nullptr) {
        return nullptr;
      }
      found = &node;
    }
  }
  return found;
}

struct ActivationQParams {
  std::string scale_name;
  std::string zp_name;
  float scale{};
  int64_t zp{};
};

// Records the (name, value) of the scale / zero point of each 16-bit activation Q/DQ node.
std::unordered_map<std::string, ActivationQParams> CollectActivationQParams(const Graph& graph) {
  std::unordered_map<std::string, ActivationQParams> params;
  for (const auto& node : graph.Nodes()) {
    if ((node.OpType() != "QuantizeLinear" && node.OpType() != "DequantizeLinear") ||
        !IsActivation16Bit(QuantizedElemType(node))) {
      continue;
    }
    ActivationQParams p;
    p.scale_name = node.InputDefs()[1]->Name();
    p.zp_name = node.InputDefs()[2]->Name();
    const auto* scale_proto = graph.GetConstantInitializer(p.scale_name, true);
    const auto* zp_proto = graph.GetConstantInitializer(p.zp_name, true);
    if (scale_proto != nullptr && zp_proto != nullptr) {
      Initializer scale(graph, *scale_proto, graph.ModelPath());
      Initializer zp(graph, *zp_proto, graph.ModelPath());
      p.scale = scale.data<float>()[0];
      p.zp = zp.data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT16 ? zp.data<uint16_t>()[0]
                                                                           : zp.data<int16_t>()[0];
    }
    params.emplace(node.Name(), p);
  }
  return params;
}

std::function<Status(Graph&)> AssignAllNodesTo(const std::string& ep) {
  return [ep](Graph& graph) {
    for (auto& node : graph.Nodes()) {
      node.SetExecutionProviderType(ep);
    }
    return Status::OK();
  };
}

// Applies only the QDQ selector/action transformer to a graph whose nodes are assigned to `ep`.
void RunTransformerOnly(const A16Case& c, const std::string& ep, const std::function<void(Graph&)>& check) {
  std::unordered_map<std::string, ActivationQParams> before;
  QDQCounts counts_before;
  auto pre = [&](Graph& graph) {
    ORT_RETURN_IF_ERROR(AssignAllNodesTo(ep)(graph));
    before = CollectActivationQParams(graph);
    counts_before = CountQDQ(graph);
    return Status::OK();
  };
  auto post = [&](Graph& graph) {
    // Activation Q/DQ nodes and their quantization parameters are never touched.
    const QDQCounts counts_after = CountQDQ(graph);
    EXPECT_EQ(counts_after.q16, counts_before.q16);
    EXPECT_EQ(counts_after.dq16, counts_before.dq16);
    EXPECT_EQ(counts_after.q8, counts_before.q8);
    EXPECT_EQ(counts_after.dq8, counts_before.dq8);
    const auto after = CollectActivationQParams(graph);
    EXPECT_EQ(after.size(), before.size());
    for (const auto& [name, p] : before) {
      auto it = after.find(name);
      if (it == after.end()) {
        ADD_FAILURE() << "Activation Q/DQ node removed: " << name;
        continue;
      }
      EXPECT_EQ(it->second.scale_name, p.scale_name);
      EXPECT_EQ(it->second.zp_name, p.zp_name);
      EXPECT_EQ(it->second.scale, p.scale);
      EXPECT_EQ(it->second.zp, p.zp);
    }
    check(graph);
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(BuildA16Case(c), 21, DefaultLoggingManager().DefaultLogger(),
                                        std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                                        TransformerLevel::Level2, 1, pre, post));
}

// Asserts that the transformer produced exactly one MatMulNBits that consumes the same activation value and
// feeds the same output consumers as the original target.
void ExpectFusedInPlace(const Graph& graph, const A16Case& c) {
  const auto op_count = CountOpsInGraph(graph);
  EXPECT_EQ(OpCount(op_count, "MatMul"), 0);
  EXPECT_EQ(OpCount(op_count, "Gemm"), 0);
  ASSERT_EQ(OpCount(op_count, "com.microsoft.MatMulNBits"), 1);

  const Node* nbits = FindSingleNode(graph, "MatMulNBits");
  ASSERT_NE(nbits, nullptr);
  EXPECT_EQ(nbits->GetExecutionProviderType(), kCpuExecutionProvider);

  // Only the (optional) bias DQ remains among the constant-input DQs; the weight DQ was fused.
  const QDQCounts counts = CountQDQ(graph);
  EXPECT_EQ(counts.dq_other, c.bias == BiasKind::kDQ ? 1 : 0);

  const NodeArg* a = nbits->InputDefs()[0];
  if (c.input_boundary) {
    const Node* producer = graph.GetProducerNode(a->Name());
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer->OpType(), "DequantizeLinear");
    EXPECT_TRUE(IsActivation16Bit(QuantizedElemType(*producer)));
    bool has_edge_from_producer = false;
    for (auto it = nbits->InputEdgesBegin(); it != nbits->InputEdgesEnd(); ++it) {
      has_edge_from_producer |= it->GetNode().Index() == producer->Index() && it->GetDstArgIndex() == 0;
    }
    EXPECT_TRUE(has_edge_from_producer);
  } else {
    EXPECT_TRUE(graph.IsInputsIncludingInitializers(a));
  }

  const auto consumers = graph.GetConsumerNodes(nbits->OutputDefs()[0]->Name());
  if (c.output_boundary) {
    ASSERT_EQ(consumers.size(), 1u);
    EXPECT_EQ(consumers[0]->OpType(), "QuantizeLinear");
    EXPECT_TRUE(IsActivation16Bit(QuantizedElemType(*consumers[0])));
    EXPECT_EQ(nbits->GetOutputEdgesCount(), 1u);
  } else {
    EXPECT_TRUE(consumers.empty());
    EXPECT_TRUE(graph.IsOutput(nbits->OutputDefs()[0]));
  }

  if (c.bias != BiasKind::kNone) {
    ASSERT_GT(nbits->InputDefs().size(), 5u);
    const NodeArg* bias = nbits->InputDefs()[5];
    ASSERT_TRUE(bias->Exists());
    if (c.bias == BiasKind::kDQ) {
      const Node* producer = graph.GetProducerNode(bias->Name());
      ASSERT_NE(producer, nullptr);
      EXPECT_EQ(producer->OpType(), "DequantizeLinear");
    } else {
      EXPECT_NE(graph.GetConstantInitializer(bias->Name(), true), nullptr);
    }
  }
}

void ExpectNotFused(const Graph& graph, const A16Case& c) {
  const auto op_count = CountOpsInGraph(graph);
  EXPECT_EQ(OpCount(op_count, "com.microsoft.MatMulNBits"), 0);
  const std::string target = c.gemm ? "Gemm" : "MatMul";
  EXPECT_EQ(OpCount(op_count, target), 1);
  EXPECT_EQ(CountQDQ(graph).dq_other, c.bias == BiasKind::kDQ ? 2 : 1);
}

// Full optimization pipeline on the CPU EP; compares against the Level1 (unfused) result.
void RunPipeline(const A16Case& c, const std::function<void(InferenceSessionWrapper&)>& check,
                 double tolerance) {
  auto add_session_options = [](SessionOptions& so) {
    // FP32 compute inside MatMulNBits so the fused result matches the unfused float MatMul.
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel, "1"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableQuantQDQCleanup, "0"));
  };
  TransformerTester(BuildA16Case(c), check, TransformerLevel::Level1, TransformerLevel::Level2, 21,
                    tolerance, tolerance, nullptr, add_session_options);
}

// Within tolerance, a 16-bit output Q may round one step differently after a fused-vs-unfused FP32 reordering.
constexpr double kOneOutputStep = kOutScale * 1.01;

}  // namespace

TEST(QDQMatMulNBitsA16Tests, Transformer_FusesWeightAndKeepsBoundaries) {
  for (ActivationType activation : {ActivationType::kUInt16, ActivationType::kInt16}) {
    for (WeightQuant weight : {WeightQuant::kInt4PerChannel, WeightQuant::kUInt4PerChannel,
                               WeightQuant::kInt4Block32, WeightQuant::kInt4Block128,
                               WeightQuant::kInt8PerChannel, WeightQuant::kUInt4PerTensor}) {
      A16Case c;
      c.activation = activation;
      c.weight = weight;
      SCOPED_TRACE(::testing::Message() << "activation=" << static_cast<int>(activation)
                                        << " weight=" << static_cast<int>(weight));
      RunTransformerOnly(c, kCpuExecutionProvider, [&](Graph& graph) { ExpectFusedInPlace(graph, c); });
    }
  }
}

TEST(QDQMatMulNBitsA16Tests, Transformer_InputOnlyAndOutputOnlyBoundaries) {
  A16Case input_only;
  input_only.output_boundary = false;
  RunTransformerOnly(input_only, kCpuExecutionProvider,
                     [&](Graph& graph) { ExpectFusedInPlace(graph, input_only); });

  A16Case output_only;
  output_only.input_boundary = false;
  RunTransformerOnly(output_only, kCpuExecutionProvider,
                     [&](Graph& graph) { ExpectFusedInPlace(graph, output_only); });
}

TEST(QDQMatMulNBitsA16Tests, Transformer_Gemm) {
  for (BiasKind bias : {BiasKind::kNone, BiasKind::kFloat, BiasKind::kDQ}) {
    A16Case c;
    c.gemm = true;
    c.bias = bias;
    SCOPED_TRACE(::testing::Message() << "bias=" << static_cast<int>(bias));
    RunTransformerOnly(c, kCpuExecutionProvider, [&](Graph& graph) { ExpectFusedInPlace(graph, c); });
  }
}

TEST(QDQMatMulNBitsA16Tests, Transformer_SharedActivationDQIsRetained) {
  A16Case c;
  c.share_activation_dq = true;
  RunTransformerOnly(c, kCpuExecutionProvider, [&](Graph& graph) {
    ExpectFusedInPlace(graph, c);
    const Node* identity = FindSingleNode(graph, "Identity");
    const Node* nbits = FindSingleNode(graph, "MatMulNBits");
    ASSERT_NE(identity, nullptr);
    ASSERT_NE(nbits, nullptr);
    // The shared activation value still feeds both consumers.
    EXPECT_EQ(identity->InputDefs()[0], nbits->InputDefs()[0]);
  });
}

TEST(QDQMatMulNBitsA16Tests, Transformer_NotFused_8BitActivation) {
  // UINT8 activations with INT4 weights: no existing rule fuses this; the A16 rule must not either.
  A16Case c;
  c.activation = ActivationType::kUInt8;
  RunTransformerOnly(c, kCpuExecutionProvider, [&](Graph& graph) { ExpectNotFused(graph, c); });
}

TEST(QDQMatMulNBitsA16Tests, Transformer_NotFused_NonCpuEp) {
  for (const char* ep : {kCudaExecutionProvider, kDmlExecutionProvider}) {
    A16Case c;
    SCOPED_TRACE(ep);
    RunTransformerOnly(c, ep, [&](Graph& graph) { ExpectNotFused(graph, c); });
  }
}

TEST(QDQMatMulNBitsA16Tests, Transformer_NotFused_UnsupportedGemm) {
  A16Case c;
  c.gemm = true;
  c.gemm_trans_b = true;
  c.K = 32;
  c.N = 32;  // square so transB keeps the graph valid
  RunTransformerOnly(c, kCpuExecutionProvider, [&](Graph& graph) { ExpectNotFused(graph, c); });
}

// MatMulNBits drops alpha and beta, so they must be exactly 1: NaN and values within rounding distance of 1
// (1.0000005f differs from 1 by less than the old 1e-6 tolerance) are rejected.
TEST(QDQMatMulNBitsA16Tests, Transformer_NotFused_GemmAlphaNotExactlyOne) {
  for (float alpha : {1.0000005f, std::numeric_limits<float>::quiet_NaN()}) {
    SCOPED_TRACE(::testing::Message() << "alpha=" << alpha);
    A16Case c;
    c.gemm = true;
    c.gemm_alpha = alpha;
    RunTransformerOnly(c, kCpuExecutionProvider, [&](Graph& graph) { ExpectNotFused(graph, c); });
  }
}

TEST(QDQMatMulNBitsA16Tests, Transformer_NotFused_GemmBetaNotExactlyOne) {
  for (float beta : {1.0000005f, std::numeric_limits<float>::quiet_NaN()}) {
    SCOPED_TRACE(::testing::Message() << "beta=" << beta);
    A16Case c;
    c.gemm = true;
    c.bias = BiasKind::kFloat;
    c.gemm_beta = beta;
    RunTransformerOnly(c, kCpuExecutionProvider, [&](Graph& graph) { ExpectNotFused(graph, c); });
  }
}

TEST(QDQMatMulNBitsA16Tests, Transformer_NotFused_UnsupportedOutputConsumers) {
  auto build_case = [](const std::string& variant) {
    return [variant](ModelTestBuilder& builder) {
      NodeArg* x = builder.MakeInput<float>({7, 64}, -1.0f, 1.0f);
      NodeArg* a = AddActivationQDQ(builder, x, ActivationType::kUInt16, kActScale);
      NodeArg* w = AddWeightDQ(builder, WeightQuant::kInt4PerChannel, 64, 16);
      NodeArg* mm_out = builder.MakeIntermediate();
      builder.AddNode("MatMul", {a, w}, {mm_out});
      NodeArg* q_in = mm_out;
      if (variant == "relu") {
        q_in = builder.MakeIntermediate();
        builder.AddNode("Relu", {mm_out}, {q_in});
      } else if (variant == "fanout") {
        builder.AddNode("Identity", {mm_out}, {builder.MakeOutput()});
      }
      NodeArg* q_out = builder.MakeIntermediate();
      if (variant == "q8") {
        builder.AddQuantizeLinearNode<uint8_t>(q_in, 0.1f, static_cast<uint8_t>(128), q_out);
        builder.AddDequantizeLinearNode<uint8_t>(q_out, 0.1f, static_cast<uint8_t>(128), builder.MakeOutput());
      } else {
        builder.AddQuantizeLinearNode<uint16_t>(q_in, kOutScale, static_cast<uint16_t>(32768), q_out);
        builder.AddDequantizeLinearNode<uint16_t>(q_out, kOutScale, static_cast<uint16_t>(32768),
                                                  builder.MakeOutput());
      }
    };
  };

  for (const char* variant : {"relu", "fanout", "q8"}) {
    SCOPED_TRACE(variant);
    auto post = [&](Graph& graph) {
      const auto op_count = CountOpsInGraph(graph);
      EXPECT_EQ(OpCount(op_count, "com.microsoft.MatMulNBits"), 0);
      EXPECT_EQ(OpCount(op_count, "MatMul"), 1);
      if (std::string(variant) == "relu") {
        EXPECT_EQ(OpCount(op_count, "Relu"), 1);
      }
      return Status::OK();
    };
    ASSERT_STATUS_OK(TestGraphTransformer(build_case(variant), 21, DefaultLoggingManager().DefaultLogger(),
                                          std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                                          TransformerLevel::Level2, 1, AssignAllNodesTo(kCpuExecutionProvider),
                                          post));
  }
}

TEST(QDQMatMulNBitsA16Tests, Transformer_NotFused_NonConstantOrSharedWeight) {
  // Non-constant weight: the quantized weight is a graph input.
  auto non_constant = [](ModelTestBuilder& builder) {
    NodeArg* x = builder.MakeInput<float>({7, 64}, -1.0f, 1.0f);
    NodeArg* a = AddActivationQDQ(builder, x, ActivationType::kUInt16, kActScale);
    NodeArg* w = builder.MakeInput<int8_t>({64, 16}, static_cast<int8_t>(-8), static_cast<int8_t>(8));
    NodeArg* w_dq = builder.MakeIntermediate();
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(1)), attrs);
    builder.AddNode("DequantizeLinear", {w, builder.MakeInitializer<float>({16}, 0.01f, 0.02f)}, {w_dq}, "", &attrs);
    builder.AddNode("MatMul", {a, w_dq}, {builder.MakeOutput()});
  };

  // Shared weight initializer: two DQs read the same quantized weight.
  auto shared = [](ModelTestBuilder& builder) {
    NodeArg* w = builder.MakeInitializer<Int4x2>({64, 16}, Int4x2(Int4x2::min_val, 0), Int4x2(Int4x2::max_val, 0));
    NodeArg* s = builder.MakeInitializer<float>({16}, 0.01f, 0.02f);
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(1)), attrs);
    for (int i = 0; i < 2; ++i) {
      NodeArg* x = builder.MakeInput<float>({7, 64}, -1.0f, 1.0f);
      NodeArg* a = AddActivationQDQ(builder, x, ActivationType::kUInt16, kActScale);
      NodeArg* w_dq = builder.MakeIntermediate();
      builder.AddNode("DequantizeLinear", {w, s}, {w_dq}, "", &attrs);
      builder.AddNode("MatMul", {a, w_dq}, {builder.MakeOutput()});
    }
  };

  for (const auto& build : {std::function<void(ModelTestBuilder&)>(non_constant),
                            std::function<void(ModelTestBuilder&)>(shared)}) {
    auto post = [](Graph& graph) {
      const auto op_count = CountOpsInGraph(graph);
      EXPECT_EQ(OpCount(op_count, "com.microsoft.MatMulNBits"), 0);
      return Status::OK();
    };
    ASSERT_STATUS_OK(TestGraphTransformer(build, 21, DefaultLoggingManager().DefaultLogger(),
                                          std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                                          TransformerLevel::Level2, 1, AssignAllNodesTo(kCpuExecutionProvider),
                                          post));
  }
}

TEST(QDQMatMulNBitsA16Tests, Pipeline_FusesWeightAndKeepsBoundaries) {
  std::vector<A16Case> cases;
  for (ActivationType activation : {ActivationType::kUInt16, ActivationType::kInt16}) {
    for (WeightQuant weight : {WeightQuant::kInt4PerChannel, WeightQuant::kInt4Block32,
                               WeightQuant::kInt4Block128, WeightQuant::kInt8PerChannel,
                               WeightQuant::kUInt4PerTensor}) {
      A16Case c;
      c.activation = activation;
      c.weight = weight;
      cases.push_back(c);
    }
  }
  {
    A16Case c;
    c.output_boundary = false;
    cases.push_back(c);
  }
  {
    A16Case c;
    c.input_boundary = false;
    cases.push_back(c);
  }
  for (BiasKind bias : {BiasKind::kNone, BiasKind::kFloat, BiasKind::kDQ}) {
    A16Case c;
    c.gemm = true;
    c.bias = bias;
    cases.push_back(c);
  }
  {
    A16Case c;
    c.share_activation_dq = true;
    cases.push_back(c);
  }

  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "activation=" << static_cast<int>(c.activation)
                                      << " weight=" << static_cast<int>(c.weight) << " gemm=" << c.gemm
                                      << " bias=" << static_cast<int>(c.bias) << " in=" << c.input_boundary
                                      << " out=" << c.output_boundary << " shared=" << c.share_activation_dq);
    auto check = [&](InferenceSessionWrapper& session) {
      const Graph& graph = session.GetGraph();
      const auto op_count = CountOpsInGraph(graph);
      EXPECT_EQ(OpCount(op_count, "com.microsoft.MatMulNBits"), 1);
      EXPECT_EQ(OpCount(op_count, "MatMul"), 0);
      EXPECT_EQ(OpCount(op_count, "Gemm"), 0);
      const QDQCounts counts = CountQDQ(graph);
      // One Q/DQ pair per retained activation boundary. A shared activation DQ is duplicated per consumer by
      // EnsureUniqueDQForNodeUnit, so both copies remain.
      const int expected_in = c.input_boundary ? 1 : 0;
      const int expected_out = c.output_boundary ? 1 : 0;
      EXPECT_EQ(counts.q16, expected_in + expected_out);
      EXPECT_EQ(counts.dq16, expected_in * (c.share_activation_dq ? 2 : 1) + expected_out);
      EXPECT_EQ(counts.q8 + counts.dq8, 0);
    };
    RunPipeline(c, check, c.output_boundary ? kOneOutputStep : 1e-4);
  }
}

// Mixed graph: an A8 branch (UINT8 activations, INT8 weights) and an A16 branch share the input, and the A16
// output is requantized to UINT8 for a second A8 MatMul. The A8 MatMuls keep their QLinearMatMul fusion and all
// quantization grids; only the A16 weight DQ becomes MatMulNBits.
TEST(QDQMatMulNBitsA16Tests, Pipeline_MixedA8A16Graph) {
  constexpr int64_t M = 5, K = 64, N = 32;
  auto build = [](ModelTestBuilder& builder) {
    NodeArg* x = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);

    auto add_a8_matmul = [&](NodeArg* dq8_input, int64_t k, int64_t n) {
      NodeArg* w = builder.MakeInitializer<int8_t>({k, n}, static_cast<int8_t>(-64), static_cast<int8_t>(64));
      NodeArg* w_dq = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(w, 0.003f, static_cast<int8_t>(0), w_dq);
      NodeArg* mm = builder.MakeIntermediate();
      builder.AddNode("MatMul", {dq8_input, w_dq}, {mm});
      NodeArg* q = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<uint8_t>(mm, 0.05f, static_cast<uint8_t>(128), q);
      builder.AddDequantizeLinearNode<uint8_t>(q, 0.05f, static_cast<uint8_t>(128), builder.MakeOutput());
    };

    // A8 branch.
    NodeArg* x8 = AddActivationQDQ(builder, x, ActivationType::kUInt8, kActScale);
    add_a8_matmul(x8, K, N);

    // A16 branch.
    NodeArg* x16 = AddActivationQDQ(builder, x, ActivationType::kUInt16, kActScale);
    NodeArg* w16 = AddWeightDQ(builder, WeightQuant::kInt4PerChannel, K, N);
    NodeArg* mm16 = builder.MakeIntermediate();
    builder.AddNode("MatMul", {x16, w16}, {mm16});
    NodeArg* q16 = builder.MakeIntermediate();
    NodeArg* y16 = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint16_t>(mm16, kOutScale * 8, static_cast<uint16_t>(32768), q16);
    builder.AddDequantizeLinearNode<uint16_t>(q16, kOutScale * 8, static_cast<uint16_t>(32768), y16);
    builder.AddNode("Identity", {y16}, {builder.MakeOutput()});

    // A16 -> A8 boundary: requantize the shared A16 value to UINT8 for another A8 MatMul.
    NodeArg* y8 = AddActivationQDQ(builder, y16, ActivationType::kUInt8, 0.01f / 256.0f);
    add_a8_matmul(y8, N, N);
  };

  auto check = [](InferenceSessionWrapper& session) {
    const Graph& graph = session.GetGraph();
    const auto op_count = CountOpsInGraph(graph);
    EXPECT_EQ(OpCount(op_count, "com.microsoft.MatMulNBits"), 1);
    EXPECT_EQ(OpCount(op_count, "QLinearMatMul"), 2);
    EXPECT_EQ(OpCount(op_count, "MatMul"), 0);

    // A16 boundary pair around the NBits node is intact.
    const Node* nbits = FindSingleNode(graph, "MatMulNBits");
    ASSERT_NE(nbits, nullptr);
    const Node* a_producer = graph.GetProducerNode(nbits->InputDefs()[0]->Name());
    ASSERT_NE(a_producer, nullptr);
    EXPECT_EQ(a_producer->OpType(), "DequantizeLinear");
    EXPECT_EQ(QuantizedElemType(*a_producer), ONNX_NAMESPACE::TensorProto_DataType_UINT16);
    const auto consumers = graph.GetConsumerNodes(nbits->OutputDefs()[0]->Name());
    ASSERT_EQ(consumers.size(), 1u);
    EXPECT_EQ(QuantizedElemType(*consumers[0]), ONNX_NAMESPACE::TensorProto_DataType_UINT16);

    // Both A8 QLinearMatMuls still quantize with UINT8 inputs, and the A16 -> A8 requantization Q is kept.
    int uint8_q = 0;
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "QLinearMatMul") {
        EXPECT_EQ(node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type(),
                  ONNX_NAMESPACE::TensorProto_DataType_UINT8);
      }
      if (node.OpType() == "QuantizeLinear" && QuantizedElemType(node) == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
        ++uint8_q;
      }
    }
    EXPECT_EQ(uint8_q, 2);  // X -> Q8 and Y16 -> Q8
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel, "1"));
  };
  // Up to two UINT8 output steps (0.05) for rounding-boundary flips propagated through the requantization.
  TransformerTester(build, check, TransformerLevel::Level1, TransformerLevel::Level2, 21, 0.101, 0.0, nullptr,
                    add_session_options);
}

TEST(QDQMatMulNBitsA16Tests, Pipeline_A8KeepsExistingFusion) {
  // UINT8 activations with INT8 weights must still become QLinearMatMul, not MatMulNBits.
  auto build = [](ModelTestBuilder& builder) {
    NodeArg* x = builder.MakeInput<float>({5, 64}, -1.0f, 1.0f);
    NodeArg* x8 = AddActivationQDQ(builder, x, ActivationType::kUInt8, kActScale);
    NodeArg* w = builder.MakeInitializer<int8_t>({64, 32}, static_cast<int8_t>(-64), static_cast<int8_t>(64));
    NodeArg* w_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<int8_t>(w, 0.003f, static_cast<int8_t>(0), w_dq);
    NodeArg* mm = builder.MakeIntermediate();
    builder.AddNode("MatMul", {x8, w_dq}, {mm});
    NodeArg* q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(mm, 0.05f, static_cast<uint8_t>(128), q);
    builder.AddDequantizeLinearNode<uint8_t>(q, 0.05f, static_cast<uint8_t>(128), builder.MakeOutput());
  };
  auto check = [](InferenceSessionWrapper& session) {
    const auto op_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(OpCount(op_count, "QLinearMatMul"), 1);
    EXPECT_EQ(OpCount(op_count, "com.microsoft.MatMulNBits"), 0);
  };
  TransformerTester(build, check, TransformerLevel::Level1, TransformerLevel::Level2, 21, 0.051, 0.051);
}

TEST(QDQMatMulNBitsA16Tests, Pipeline_DefaultOptionsKeepA16Boundaries) {
  // No session options: fusion does not depend on any global activation Q/DQ removal option.
  A16Case c;
  auto check = [](InferenceSessionWrapper& session) {
    const Graph& graph = session.GetGraph();
    const auto op_count = CountOpsInGraph(graph);
    EXPECT_EQ(OpCount(op_count, "com.microsoft.MatMulNBits"), 1);
    const QDQCounts counts = CountQDQ(graph);
    EXPECT_EQ(counts.q16, 2);
    EXPECT_EQ(counts.dq16, 2);
    for (const auto& node : graph.Nodes()) {
      EXPECT_EQ(node.GetExecutionProviderType(), kCpuExecutionProvider);
    }
  };
  // Default accuracy level 4 quantizes A to int8 inside MatMulNBits, so allow a small relative error.
  TransformerTester(BuildA16Case(c), check, TransformerLevel::Level1, TransformerLevel::Level2, 21, 0.02, 0.02);
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
