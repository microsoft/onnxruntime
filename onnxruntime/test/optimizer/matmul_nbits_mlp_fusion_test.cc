// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/matmul_nbits_mlp_fusion.h"
#include "core/optimizer/utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

namespace {

constexpr const char* kExpectedActivation = "silu";

enum class NormAnchorKind {
  kSimplified,
  kSkipSimplified,
};

enum class SkipOutputKind {
  kNone,
  kGraphOutput,
};

void SetWebGpuProvider(Node& node) {
  node.SetExecutionProviderType(kWebGpuExecutionProvider);
}

NodeAttributes MakeMatMulNBitsAttrs(int64_t k, int64_t n, int64_t block_size, int64_t bits, int64_t accuracy_level) {
  NodeAttributes attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("K", k), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("N", n), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("bits", bits), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", accuracy_level), attrs);
  return attrs;
}

Status CheckMatMulNBitsMlpFusedGraphImpl(const Graph& graph, NormAnchorKind norm_anchor_kind) {
  const auto op_to_count = CountOpsInGraph(graph);
  if (OpCount(op_to_count, "com.microsoft.MatMulNBitsMlp") != 1 ||
      OpCount(op_to_count, "com.microsoft.MatMulNBits") != 0 ||
      OpCount(op_to_count, "SimplifiedLayerNormalization") != 0 ||
      OpCount(op_to_count, "com.microsoft.SkipSimplifiedLayerNormalization") != 0 ||
      OpCount(op_to_count, "Sigmoid") != 0 ||
      OpCount(op_to_count, "Mul") != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected operator counts after MatMulNBitsMlpFusion.");
  }

  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "MatMulNBitsMlp") {
      ORT_RETURN_IF_NOT(node.Domain() == kMSDomain, "Fused node must be in com.microsoft domain.");
      ORT_RETURN_IF_NOT(node.GetExecutionProviderType() == kWebGpuExecutionProvider,
                        "Fused node must be assigned to WebGPU EP.");
      ORT_RETURN_IF_NOT(node.InputDefs().size() == 9u, "Fused node must have 9 inputs.");
      const bool has_skip = node.InputDefs()[1] != nullptr && !node.InputDefs()[1]->Name().empty();
      const bool has_norm_scale = node.InputDefs()[2] != nullptr && !node.InputDefs()[2]->Name().empty();
      ORT_RETURN_IF_NOT(has_skip == (norm_anchor_kind == NormAnchorKind::kSkipSimplified),
                        "Unexpected skip input presence on fused node.");
      ORT_RETURN_IF_NOT(has_norm_scale,
                        "Expected norm_scale input on fused node.");
      ORT_RETURN_IF_NOT(node.OutputDefs().size() == 1u,
                        "Non-passthrough fusion should expose only the Y output.");

      const auto* activation_attr = graph_utils::GetNodeAttribute(node, "activation");
      ORT_RETURN_IF_NOT(activation_attr != nullptr && activation_attr->s() == kExpectedActivation,
                        "Fused node must carry activation='silu'.");
    }
  }

  return Status::OK();
}

Status CheckMatMulNBitsMlpSimplifiedFusedGraph(const Graph& graph) {
  return CheckMatMulNBitsMlpFusedGraphImpl(graph, NormAnchorKind::kSimplified);
}

Status CheckMatMulNBitsMlpSkipFusedGraph(const Graph& graph) {
  return CheckMatMulNBitsMlpFusedGraphImpl(graph, NormAnchorKind::kSkipSimplified);
}

Status CheckMatMulNBitsMlpSkipOutputPassthroughFusedGraph(const Graph& graph) {
  const auto op_to_count = CountOpsInGraph(graph);
  if (OpCount(op_to_count, "com.microsoft.MatMulNBitsMlp") != 1 ||
      OpCount(op_to_count, "com.microsoft.MatMulNBits") != 0 ||
      OpCount(op_to_count, "SimplifiedLayerNormalization") != 0 ||
      OpCount(op_to_count, "com.microsoft.SkipSimplifiedLayerNormalization") != 0 ||
      OpCount(op_to_count, "Sigmoid") != 0 ||
      OpCount(op_to_count, "Mul") != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Unexpected operator counts after MatMulNBitsMlpFusion with skip output passthrough.");
  }

  bool found_fused_node = false;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() != "MatMulNBitsMlp") {
      continue;
    }

    found_fused_node = true;
    ORT_RETURN_IF_NOT(node.Domain() == kMSDomain, "Fused node must be in com.microsoft domain.");
    ORT_RETURN_IF_NOT(node.GetExecutionProviderType() == kWebGpuExecutionProvider,
                      "Fused node must be assigned to WebGPU EP.");
    ORT_RETURN_IF_NOT(node.InputDefs().size() == 9u, "Fused node must have 9 inputs.");
    ORT_RETURN_IF_NOT(node.OutputDefs().size() == 2u,
                      "Fused node must expose Y and the passthrough residual output.");
    const bool has_skip = node.InputDefs()[1] != nullptr && !node.InputDefs()[1]->Name().empty();
    const bool has_norm_scale = node.InputDefs()[2] != nullptr && !node.InputDefs()[2]->Name().empty();
    ORT_RETURN_IF_NOT(has_skip && has_norm_scale,
                      "Skip output passthrough should remain fused into MatMulNBitsMlp.");
    ORT_RETURN_IF_NOT(node.OutputDefs()[1] != nullptr && !node.OutputDefs()[1]->Name().empty(),
                      "Expected fused node to preserve the residual passthrough output.");

    const auto* activation_attr = graph_utils::GetNodeAttribute(node, "activation");
    ORT_RETURN_IF_NOT(activation_attr != nullptr && activation_attr->s() == kExpectedActivation,
                      "Fused node must carry activation='silu'.");
  }

  ORT_RETURN_IF_NOT(found_fused_node, "Expected a MatMulNBitsMlp node in the transformed graph.");
  return Status::OK();
}

void BuildMatMulNBitsMlpWebGpuPatternImpl(ModelTestBuilder& builder,
                                          NormAnchorKind norm_anchor_kind,
                                          SkipOutputKind skip_output_kind = SkipOutputKind::kNone) {
  constexpr int64_t k = 32;
  constexpr int64_t n = 8;
  constexpr int64_t block_size = 32;
  constexpr int64_t bits = 4;
  constexpr int64_t accuracy_level = 4;
  constexpr int64_t blob_size = block_size * bits / 8;

  NodeArg* input = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{1, k},
      std::vector<MLFloat16>{
          MLFloat16(-1.0f), MLFloat16(-0.875f), MLFloat16(-0.75f), MLFloat16(-0.625f),
          MLFloat16(-0.5f), MLFloat16(-0.375f), MLFloat16(-0.25f), MLFloat16(-0.125f),
          MLFloat16(0.125f), MLFloat16(0.25f), MLFloat16(0.375f), MLFloat16(0.5f),
          MLFloat16(0.625f), MLFloat16(0.75f), MLFloat16(0.875f), MLFloat16(1.0f),
          MLFloat16(-1.0f), MLFloat16(-0.875f), MLFloat16(-0.75f), MLFloat16(-0.625f),
          MLFloat16(-0.5f), MLFloat16(-0.375f), MLFloat16(-0.25f), MLFloat16(-0.125f),
          MLFloat16(0.125f), MLFloat16(0.25f), MLFloat16(0.375f), MLFloat16(0.5f),
          MLFloat16(0.625f), MLFloat16(0.75f), MLFloat16(0.875f), MLFloat16(1.0f)});
  NodeArg* optional_tensor = builder.MakeOptionalTensor();

  NodeArg* gate_weight = builder.MakeInitializer<uint8_t>({n, 1, blob_size}, uint8_t{0}, uint8_t{15});
  NodeArg* gate_scale = builder.MakeInitializer<MLFloat16>({n, 1}, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* gate_bias = builder.MakeInitializer<MLFloat16>({n}, MLFloat16(0.0f), MLFloat16(0.0f));
  NodeArg* up_weight = builder.MakeInitializer<uint8_t>({n, 1, blob_size}, uint8_t{0}, uint8_t{15});
  NodeArg* up_scale = builder.MakeInitializer<MLFloat16>({n, 1}, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* up_bias = builder.MakeInitializer<MLFloat16>({n}, MLFloat16(0.0f), MLFloat16(0.0f));

  NodeArg* normalized_input = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, k});
  NodeArg* gate_out = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, n});
  NodeArg* up_out = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, n});
  NodeArg* sigmoid_out = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, n});
  NodeArg* silu_out = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, n});
  NodeArg* output = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{1, n});

  NodeAttributes matmul_attrs = MakeMatMulNBitsAttrs(k, n, block_size, bits, accuracy_level);
  Node* norm = nullptr;
  if (norm_anchor_kind == NormAnchorKind::kSkipSimplified) {
    NodeArg* skip_input = builder.MakeInput<MLFloat16>(
        std::vector<int64_t>{1, k},
        std::vector<MLFloat16>(static_cast<size_t>(k), MLFloat16(0.25f)));
    NodeArg* norm_scale = builder.MakeInitializer<MLFloat16>({k}, MLFloat16(1.0f), MLFloat16(1.0f));
    NodeArg* optional_norm_output_1 = builder.MakeOptionalTensor();
    NodeArg* optional_norm_output_2 = builder.MakeOptionalTensor();
    std::vector<NodeArg*> norm_outputs{normalized_input};
    if (skip_output_kind == SkipOutputKind::kGraphOutput) {
      NodeArg* residual_output = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{1, k});
      norm_outputs.push_back(optional_norm_output_1);
      norm_outputs.push_back(optional_norm_output_2);
      norm_outputs.push_back(residual_output);
    }
    norm = &builder.AddNode("SkipSimplifiedLayerNormalization", {input, skip_input, norm_scale}, norm_outputs,
                            kMSDomain);
  } else {
    NodeArg* norm_scale = builder.MakeInitializer<MLFloat16>({k}, MLFloat16(1.0f), MLFloat16(1.0f));
    norm = &builder.AddNode("SimplifiedLayerNormalization", {input, norm_scale}, {normalized_input});
  }

  Node& gate_matmul = builder.AddNode("MatMulNBits",
                                      {normalized_input, gate_weight, gate_scale, optional_tensor, optional_tensor,
                                       gate_bias},
                                      {gate_out}, kMSDomain, &matmul_attrs);
  Node& up_matmul = builder.AddNode("MatMulNBits",
                                    {normalized_input, up_weight, up_scale, optional_tensor, optional_tensor,
                                     up_bias},
                                    {up_out}, kMSDomain, &matmul_attrs);
  Node& sigmoid = builder.AddNode("Sigmoid", {gate_out}, {sigmoid_out});
  Node& silu_mul = builder.AddNode("Mul", {gate_out, sigmoid_out}, {silu_out});
  Node& final_mul = builder.AddNode("Mul", {silu_out, up_out}, {output});

  if (norm != nullptr) {
    SetWebGpuProvider(*norm);
  }
  SetWebGpuProvider(gate_matmul);
  SetWebGpuProvider(up_matmul);
  SetWebGpuProvider(sigmoid);
  SetWebGpuProvider(silu_mul);
  SetWebGpuProvider(final_mul);
}

void BuildMatMulNBitsMlpSimplifiedWebGpuPattern(ModelTestBuilder& builder) {
  BuildMatMulNBitsMlpWebGpuPatternImpl(builder, NormAnchorKind::kSimplified);
}

void BuildMatMulNBitsMlpSkipWebGpuPattern(ModelTestBuilder& builder) {
  BuildMatMulNBitsMlpWebGpuPatternImpl(builder, NormAnchorKind::kSkipSimplified);
}

void BuildMatMulNBitsMlpSkipOutputPassthroughWebGpuPattern(ModelTestBuilder& builder) {
  BuildMatMulNBitsMlpWebGpuPatternImpl(builder, NormAnchorKind::kSkipSimplified, SkipOutputKind::kGraphOutput);
}

}  // namespace

TEST_F(GraphTransformationTests, MatMulNBitsMlpFusionFusesSimplifiedWebGpuPattern) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      BuildMatMulNBitsMlpSimplifiedWebGpuPattern,
      21,
      *logger_,
      std::make_unique<MatMulNBitsMlpFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      TransformerLevel::Level2,
      1,
      nullptr,
      CheckMatMulNBitsMlpSimplifiedFusedGraph));
}

TEST_F(GraphTransformationTests, MatMulNBitsMlpFusionFusesSkipWebGpuPattern) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      BuildMatMulNBitsMlpSkipWebGpuPattern,
      21,
      *logger_,
      std::make_unique<MatMulNBitsMlpFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      TransformerLevel::Level2,
      1,
      nullptr,
      CheckMatMulNBitsMlpSkipFusedGraph));
}

TEST_F(GraphTransformationTests, MatMulNBitsMlpFusionFusesSkipWebGpuPatternWithResidualOutputPassthrough) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      BuildMatMulNBitsMlpSkipOutputPassthroughWebGpuPattern,
      21,
      *logger_,
      std::make_unique<MatMulNBitsMlpFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      TransformerLevel::Level2,
      1,
      nullptr,
      CheckMatMulNBitsMlpSkipOutputPassthroughFusedGraph));
}

TEST_F(GraphTransformationTests, MatMulNBitsMlpFusionMatchesUnfusedSimplifiedWebGpuResults) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    ASSERT_STATUS_OK(CheckMatMulNBitsMlpSimplifiedFusedGraph(session.GetGraph()));
  };

  TransformerTester(
      BuildMatMulNBitsMlpSimplifiedWebGpuPattern,
      check_transformed_graph,
      TransformerLevel::Level1,
      TransformerLevel::Level2,
      21,
      1e-3,
      1e-3,
      std::make_unique<MatMulNBitsMlpFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      {},
      {},
      std::move(webgpu_ep));
}

TEST_F(GraphTransformationTests, MatMulNBitsMlpFusionMatchesUnfusedSkipWebGpuResults) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    ASSERT_STATUS_OK(CheckMatMulNBitsMlpSkipFusedGraph(session.GetGraph()));
  };

  TransformerTester(
      BuildMatMulNBitsMlpSkipWebGpuPattern,
      check_transformed_graph,
      TransformerLevel::Level1,
      TransformerLevel::Level2,
      21,
      1e-3,
      1e-3,
      std::make_unique<MatMulNBitsMlpFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      {},
      {},
      std::move(webgpu_ep));
}

TEST_F(GraphTransformationTests, MatMulNBitsMlpFusionMatchesUnfusedSkipWebGpuResultsWithResidualOutputPassthrough) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableSpecifiedOptimizers,
                                                      "EliminateIdentity"));
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    ASSERT_STATUS_OK(CheckMatMulNBitsMlpSkipOutputPassthroughFusedGraph(session.GetGraph()));
  };

  TransformerTester(
      BuildMatMulNBitsMlpSkipOutputPassthroughWebGpuPattern,
      check_transformed_graph,
      TransformerLevel::Level1,
      TransformerLevel::Level2,
      21,
      1e-3,
      1e-3,
      std::make_unique<MatMulNBitsMlpFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      add_session_options,
      {},
      std::move(webgpu_ep));
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
