// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/node_attr_utils.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/matmul_nbits_qkv_fusion.h"
#include "core/optimizer/utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/optimizer/webgpu_fusion_test_util.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

namespace {

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

Status CheckMatMulNBitsQkvFusedGraphImpl(const Graph& graph, bool expect_skip_sln_output, bool expect_skip_input) {
  const auto op_to_count = CountOpsInGraph(graph);
  if (OpCount(op_to_count, "com.microsoft.MatMulNBitsQkv") != 1 ||
      OpCount(op_to_count, "SimplifiedLayerNormalization") != 0 ||
      OpCount(op_to_count, "com.microsoft.SkipSimplifiedLayerNormalization") != 0 ||
      OpCount(op_to_count, "com.microsoft.MatMulNBits") != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Unexpected operator counts after MatMulNBitsQkvFusion.");
  }

  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "MatMulNBitsQkv") {
      ORT_RETURN_IF_NOT(node.Domain() == kMSDomain, "Fused node must be in com.microsoft domain.");
      ORT_RETURN_IF_NOT(node.GetExecutionProviderType() == kWebGpuExecutionProvider,
                        "Fused node must be assigned to WebGPU EP.");
      ORT_RETURN_IF_NOT(node.InputDefs().size() == 9, "Fused node must expose the 9-input contract.");
      ORT_RETURN_IF_NOT(node.OutputDefs().size() == (expect_skip_sln_output ? 4u : 3u),
                        "Fused node outputs did not match the expected simplified vs skip-simplified contract.");
      // skip is at input index 1; for the SkipSimplifiedLayerNormalization-anchored pattern it
      // must be wired to a real NodeArg, otherwise it must be the empty optional.
      const auto* skip_def = node.InputDefs()[1];
      const bool skip_present = skip_def != nullptr && skip_def->Exists();
      ORT_RETURN_IF_NOT(skip_present == expect_skip_input,
                        "Fused node skip-input presence did not match the expected pattern variant.");
    }
  }

  return Status::OK();
}

Status CheckMatMulNBitsQkvFusedGraph(Graph& graph) {
  return CheckMatMulNBitsQkvFusedGraphImpl(static_cast<const Graph&>(graph),
                                           /*expect_skip_sln_output=*/false,
                                           /*expect_skip_input=*/false);
}

Status CheckMatMulNBitsQkvSkipFusedGraph(Graph& graph) {
  return CheckMatMulNBitsQkvFusedGraphImpl(static_cast<const Graph&>(graph),
                                           /*expect_skip_sln_output=*/false,
                                           /*expect_skip_input=*/true);
}

Status CheckMatMulNBitsQkvSkipOutputPassthroughFusedGraph(Graph& graph) {
  return CheckMatMulNBitsQkvFusedGraphImpl(static_cast<const Graph&>(graph),
                                           /*expect_skip_sln_output=*/true,
                                           /*expect_skip_input=*/true);
}

void BuildMatMulNBitsQkvWebGpuPatternImpl(ModelTestBuilder& builder, bool with_skip_input, bool with_skip_output) {
  constexpr int64_t k = 16;
  constexpr int64_t q_n = 8;
  constexpr int64_t kv_n = 4;
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
          MLFloat16(0.625f), MLFloat16(0.75f), MLFloat16(0.875f), MLFloat16(1.0f)});
  NodeArg* skip_input = with_skip_input
                            ? builder.MakeInput<MLFloat16>(
                                  std::vector<int64_t>{1, k},
                                  std::vector<MLFloat16>{
                                      MLFloat16(1.0f), MLFloat16(0.875f), MLFloat16(0.75f), MLFloat16(0.625f),
                                      MLFloat16(0.5f), MLFloat16(0.375f), MLFloat16(0.25f), MLFloat16(0.125f),
                                      MLFloat16(-0.125f), MLFloat16(-0.25f), MLFloat16(-0.375f), MLFloat16(-0.5f),
                                      MLFloat16(-0.625f), MLFloat16(-0.75f), MLFloat16(-0.875f), MLFloat16(-1.0f)})
                            : nullptr;

  NodeArg* norm_scale = builder.MakeInitializer<MLFloat16>({k}, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* q_weight = builder.MakeInitializer<uint8_t>({q_n, 1, blob_size}, uint8_t{0}, uint8_t{15});
  NodeArg* q_scale = builder.MakeInitializer<MLFloat16>({q_n, 1}, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* k_weight = builder.MakeInitializer<uint8_t>({kv_n, 1, blob_size}, uint8_t{0}, uint8_t{15});
  NodeArg* k_scale = builder.MakeInitializer<MLFloat16>({kv_n, 1}, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* v_weight = builder.MakeInitializer<uint8_t>({kv_n, 1, blob_size}, uint8_t{0}, uint8_t{15});
  NodeArg* v_scale = builder.MakeInitializer<MLFloat16>({kv_n, 1}, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* optional_tensor = builder.MakeOptionalTensor();

  NodeArg* norm_out = builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, k});
  NodeArg* optional_norm_output_1 = builder.MakeOptionalTensor();
  NodeArg* optional_norm_output_2 = builder.MakeOptionalTensor();
  NodeArg* residual_out = (with_skip_input && with_skip_output) ? builder.MakeIntermediate<MLFloat16>(std::vector<int64_t>{1, k}) : nullptr;
  NodeArg* q_output = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{1, q_n});
  NodeArg* k_output = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{1, kv_n});
  NodeArg* v_output = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{1, kv_n});
  NodeArg* residual_passthrough = (with_skip_input && with_skip_output) ? builder.MakeOutput<MLFloat16>(std::vector<int64_t>{1, k}) : nullptr;

  NodeAttributes q_attrs = MakeMatMulNBitsAttrs(k, q_n, block_size, bits, accuracy_level);
  NodeAttributes kv_attrs = MakeMatMulNBitsAttrs(k, kv_n, block_size, bits, accuracy_level);

  Node& norm = with_skip_input
                   ? builder.AddNode("SkipSimplifiedLayerNormalization",
                                     {input, skip_input, norm_scale},
                                     with_skip_output ? std::vector<NodeArg*>{norm_out, optional_norm_output_1, optional_norm_output_2, residual_out}
                                                      : std::vector<NodeArg*>{norm_out},
                                     kMSDomain)
                   : builder.AddNode("SimplifiedLayerNormalization", {input, norm_scale}, {norm_out});
  norm.AddAttribute("epsilon", 1e-6f);

  Node& q_matmul = builder.AddNode("MatMulNBits", {norm_out, q_weight, q_scale, optional_tensor, optional_tensor, optional_tensor}, {q_output}, kMSDomain, &q_attrs);
  Node& k_matmul = builder.AddNode("MatMulNBits", {norm_out, k_weight, k_scale, optional_tensor, optional_tensor, optional_tensor}, {k_output}, kMSDomain, &kv_attrs);
  Node& v_matmul = builder.AddNode("MatMulNBits", {norm_out, v_weight, v_scale, optional_tensor, optional_tensor, optional_tensor}, {v_output}, kMSDomain, &kv_attrs);

  SetWebGpuProvider(norm);
  SetWebGpuProvider(q_matmul);
  SetWebGpuProvider(k_matmul);
  SetWebGpuProvider(v_matmul);

  if (with_skip_output) {
    Node& residual_identity = builder.AddNode("Identity", {residual_out}, {residual_passthrough});
    SetWebGpuProvider(residual_identity);
  }
}

void BuildMatMulNBitsQkvWebGpuPattern(ModelTestBuilder& builder) {
  BuildMatMulNBitsQkvWebGpuPatternImpl(builder, false, false);
}

void BuildMatMulNBitsQkvSkipWebGpuPattern(ModelTestBuilder& builder) {
  BuildMatMulNBitsQkvWebGpuPatternImpl(builder, true, false);
}

void BuildMatMulNBitsQkvSkipOutputPassthroughWebGpuPattern(ModelTestBuilder& builder) {
  BuildMatMulNBitsQkvWebGpuPatternImpl(builder, true, true);
}

}  // namespace

TEST_F(GraphTransformationTests, MatMulNBitsQkvFusionFusesWebGpuPattern) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      BuildMatMulNBitsQkvWebGpuPattern,
      21,
      *logger_,
      std::make_unique<MatMulNBitsQkvFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      TransformerLevel::Level2,
      1,
      nullptr,
      CheckMatMulNBitsQkvFusedGraph));
}

TEST_F(GraphTransformationTests, MatMulNBitsQkvFusionMatchesUnfusedWebGpuResults) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    ASSERT_STATUS_OK(CheckMatMulNBitsQkvFusedGraphImpl(session.GetGraph(),
                                                       /*expect_skip_sln_output=*/false,
                                                       /*expect_skip_input=*/false));
  };

  RunWebGpuFusionTransformerTest(
      BuildMatMulNBitsQkvWebGpuPattern,
      check_transformed_graph,
      TransformerLevel::Level1,
      TransformerLevel::Level2,
      21,
      1e-3,
      1e-3,
      std::make_unique<MatMulNBitsQkvFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      []() { return DefaultWebGpuExecutionProvider(); });
}

TEST_F(GraphTransformationTests, MatMulNBitsQkvFusionFusesSkipWebGpuPattern) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      BuildMatMulNBitsQkvSkipWebGpuPattern,
      21,
      *logger_,
      std::make_unique<MatMulNBitsQkvFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      TransformerLevel::Level2,
      1,
      nullptr,
      CheckMatMulNBitsQkvSkipFusedGraph));
}

TEST_F(GraphTransformationTests, MatMulNBitsQkvFusionMatchesUnfusedSkipWebGpuResults) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    ASSERT_STATUS_OK(CheckMatMulNBitsQkvFusedGraphImpl(session.GetGraph(),
                                                       /*expect_skip_sln_output=*/false,
                                                       /*expect_skip_input=*/true));
  };

  RunWebGpuFusionTransformerTest(
      BuildMatMulNBitsQkvSkipWebGpuPattern,
      check_transformed_graph,
      TransformerLevel::Level1,
      TransformerLevel::Level2,
      21,
      1e-3,
      1e-3,
      std::make_unique<MatMulNBitsQkvFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      []() { return DefaultWebGpuExecutionProvider(); });
}

TEST_F(GraphTransformationTests, MatMulNBitsQkvFusionFusesSkipWebGpuPatternWithResidualOutputPassthrough) {
  ASSERT_STATUS_OK(TestGraphTransformer(
      BuildMatMulNBitsQkvSkipOutputPassthroughWebGpuPattern,
      21,
      *logger_,
      std::make_unique<MatMulNBitsQkvFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      TransformerLevel::Level2,
      1,
      nullptr,
      CheckMatMulNBitsQkvSkipOutputPassthroughFusedGraph));
}

TEST_F(GraphTransformationTests, MatMulNBitsQkvFusionMatchesUnfusedSkipWebGpuResultsWithResidualOutputPassthrough) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableSpecifiedOptimizers,
                                                      "EliminateIdentity"));
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    ASSERT_STATUS_OK(CheckMatMulNBitsQkvFusedGraphImpl(session.GetGraph(),
                                                       /*expect_skip_sln_output=*/true,
                                                       /*expect_skip_input=*/true));
  };

  RunWebGpuFusionTransformerTest(
      BuildMatMulNBitsQkvSkipOutputPassthroughWebGpuPattern,
      check_transformed_graph,
      TransformerLevel::Level1,
      TransformerLevel::Level2,
      21,
      1e-3,
      1e-3,
      std::make_unique<MatMulNBitsQkvFusion>(InlinedHashSet<std::string_view>{kWebGpuExecutionProvider}),
      []() { return DefaultWebGpuExecutionProvider(); },
      add_session_options);
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
