// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/graph/node_attr_utils.h"
#include "core/optimizer/group_query_attention_pre_norm_fusion.h"
#include "core/optimizer/utils.h"

#include "test/util/include/asserts.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

namespace {

// Small geometry that exercises the Q/K post-norm pattern without needing real GPU work.
constexpr int64_t kBatch = 1;
constexpr int64_t kSeq = 1;
constexpr int64_t kNumHeads = 2;
constexpr int64_t kKvNumHeads = 1;
constexpr int64_t kHeadSize = 4;
constexpr int64_t kQHidden = kNumHeads * kHeadSize;
constexpr int64_t kKvHidden = kKvNumHeads * kHeadSize;
constexpr int64_t kMaxSeq = 8;

void SetWebGpu(Node& node) { node.SetExecutionProviderType(kWebGpuExecutionProvider); }

// Builds: [Reshape -> SimplifiedLayerNormalization -> Reshape] on Q and K, feeding a
// GroupQueryAttention node. V goes straight into GQA. The function returns the GQA node
// so the caller can mutate it (e.g., flip k_norm's epsilon for negative tests).
struct BuildOptions {
  float q_epsilon = 1e-6f;
  float k_epsilon = 1e-6f;
  // If true, the inner reshape on the K side targets a different last-dim than head_size
  // so the matcher must reject it.
  bool break_k_inner_reshape_shape = false;
  // If true, the q_norm_weight initializer is given a non-1D shape so the matcher must
  // reject it.
  bool break_q_norm_weight_shape = false;
};

void BuildQwenQkPostNormPattern(ModelTestBuilder& builder, const BuildOptions& opts) {
  // Projection inputs (post linear projection, pre norm).
  NodeArg* q_proj = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kQHidden}, MLFloat16(-1.0f), MLFloat16(1.0f));
  NodeArg* k_proj = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kKvHidden}, MLFloat16(-1.0f), MLFloat16(1.0f));
  NodeArg* v_proj = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kKvHidden}, MLFloat16(-1.0f), MLFloat16(1.0f));

  // GQA cache + control inputs.
  NodeArg* past_key = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize}, MLFloat16(0.0f), MLFloat16(0.0f));
  NodeArg* past_value = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize}, MLFloat16(0.0f), MLFloat16(0.0f));
  NodeArg* seqlens_k = builder.MakeInput<int32_t>(std::vector<int64_t>{kBatch}, int32_t{0}, int32_t{0});
  NodeArg* total_seq_len = builder.MakeInput<int32_t>(std::vector<int64_t>{1}, int32_t{1}, int32_t{1});

  // Norm weight initializers: [head_size]. (Or non-1D when forcing a shape mismatch.)
  std::vector<int64_t> q_norm_weight_shape =
      opts.break_q_norm_weight_shape ? std::vector<int64_t>{1, kHeadSize} : std::vector<int64_t>{kHeadSize};
  NodeArg* q_norm_weight = builder.MakeInitializer<MLFloat16>(q_norm_weight_shape, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* k_norm_weight = builder.MakeInitializer<MLFloat16>({kHeadSize}, MLFloat16(1.0f), MLFloat16(1.0f));

  // Reshape "shape" initializers.
  NodeArg* reshape_to_per_head_q = builder.MakeInitializer<int64_t>({4}, {kBatch, kSeq, kNumHeads, kHeadSize});
  const int64_t k_inner_last_dim = opts.break_k_inner_reshape_shape ? (kHeadSize * 2) : kHeadSize;
  NodeArg* reshape_to_per_head_k =
      builder.MakeInitializer<int64_t>({4}, {kBatch, kSeq, kKvNumHeads, k_inner_last_dim});
  NodeArg* reshape_to_q_hidden = builder.MakeInitializer<int64_t>({3}, {kBatch, kSeq, kQHidden});
  NodeArg* reshape_to_kv_hidden = builder.MakeInitializer<int64_t>({3}, {kBatch, kSeq, kKvHidden});

  // Q-side chain.
  NodeArg* q_inner_reshape_out = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kNumHeads, kHeadSize});
  NodeArg* q_normed = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kNumHeads, kHeadSize});
  NodeArg* q_outer_reshape_out = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kQHidden});

  Node& q_inner_reshape = builder.AddNode("Reshape", {q_proj, reshape_to_per_head_q}, {q_inner_reshape_out});
  Node& q_sln = builder.AddNode("SimplifiedLayerNormalization", {q_inner_reshape_out, q_norm_weight}, {q_normed});
  q_sln.AddAttribute("axis", static_cast<int64_t>(-1));
  q_sln.AddAttribute("epsilon", opts.q_epsilon);
  Node& q_outer_reshape = builder.AddNode("Reshape", {q_normed, reshape_to_q_hidden}, {q_outer_reshape_out});

  // K-side chain.
  NodeArg* k_inner_reshape_out = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kKvNumHeads, k_inner_last_dim});
  NodeArg* k_normed = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kKvNumHeads, k_inner_last_dim});
  NodeArg* k_outer_reshape_out = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, kSeq, kKvHidden});

  Node& k_inner_reshape = builder.AddNode("Reshape", {k_proj, reshape_to_per_head_k}, {k_inner_reshape_out});
  Node& k_sln = builder.AddNode("SimplifiedLayerNormalization", {k_inner_reshape_out, k_norm_weight}, {k_normed});
  k_sln.AddAttribute("axis", static_cast<int64_t>(-1));
  k_sln.AddAttribute("epsilon", opts.k_epsilon);
  Node& k_outer_reshape = builder.AddNode("Reshape", {k_normed, reshape_to_kv_hidden}, {k_outer_reshape_out});

  // GQA outputs.
  NodeArg* gqa_out = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{kBatch, kSeq, kQHidden});
  NodeArg* present_key = builder.MakeOutput<MLFloat16>(
      std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize});
  NodeArg* present_value = builder.MakeOutput<MLFloat16>(
      std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize});

  Node& gqa = builder.AddNode("GroupQueryAttention",
                              {q_outer_reshape_out, k_outer_reshape_out, v_proj,
                               past_key, past_value, seqlens_k, total_seq_len},
                              {gqa_out, present_key, present_value},
                              kMSDomain);
  gqa.AddAttribute("num_heads", static_cast<int64_t>(kNumHeads));
  gqa.AddAttribute("kv_num_heads", static_cast<int64_t>(kKvNumHeads));

  SetWebGpu(q_inner_reshape);
  SetWebGpu(q_sln);
  SetWebGpu(q_outer_reshape);
  SetWebGpu(k_inner_reshape);
  SetWebGpu(k_sln);
  SetWebGpu(k_outer_reshape);
  SetWebGpu(gqa);
}

Status CheckFusedGraph(Graph& graph) {
  const auto op_to_count = CountOpsInGraph(graph);
  if (OpCount(op_to_count, "com.microsoft.GroupQueryAttention") != 1 ||
      OpCount(op_to_count, "SimplifiedLayerNormalization") != 0 ||
      OpCount(op_to_count, "Reshape") != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Unexpected op counts after GroupQueryAttentionPreNormFusion: ",
                           "GQA=", OpCount(op_to_count, "com.microsoft.GroupQueryAttention"),
                           " SLN=", OpCount(op_to_count, "SimplifiedLayerNormalization"),
                           " Reshape=", OpCount(op_to_count, "Reshape"));
  }

  for (const auto& node : graph.Nodes()) {
    if (node.OpType() != "GroupQueryAttention") continue;
    ORT_RETURN_IF_NOT(node.InputDefs().size() >= 16, "Fused GQA must expose 16 input slots.");

    const auto* q_norm = node.InputDefs()[14];
    const auto* k_norm = node.InputDefs()[15];
    ORT_RETURN_IF_NOT(q_norm != nullptr && q_norm->Exists(), "q_norm_weight (slot 14) missing.");
    ORT_RETURN_IF_NOT(k_norm != nullptr && k_norm->Exists(), "k_norm_weight (slot 15) missing.");

    const auto& attrs = node.GetAttributes();
    auto eps_it = attrs.find("qk_norm_epsilon");
    ORT_RETURN_IF_NOT(eps_it != attrs.end(), "qk_norm_epsilon attribute missing.");
    ORT_RETURN_IF_NOT(std::abs(eps_it->second.f() - 1e-6f) < 1e-9f, "qk_norm_epsilon value mismatch.");
  }
  return Status::OK();
}

Status CheckUnfusedGraph(Graph& graph) {
  const auto op_to_count = CountOpsInGraph(graph);
  if (OpCount(op_to_count, "com.microsoft.GroupQueryAttention") != 1 ||
      OpCount(op_to_count, "SimplifiedLayerNormalization") != 2 ||
      OpCount(op_to_count, "Reshape") != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Negative test: graph was fused unexpectedly.");
  }
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() != "GroupQueryAttention") continue;
    if (node.InputDefs().size() >= 15) {
      const auto* q_norm = node.InputDefs()[14];
      ORT_RETURN_IF_NOT(q_norm == nullptr || !q_norm->Exists(),
                        "Negative test: q_norm_weight should not be wired.");
    }
  }
  return Status::OK();
}

}  // namespace

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionFusesQwenPattern) {
  auto build = [](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, BuildOptions{}); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_,
      std::make_unique<GroupQueryAttentionPreNormFusion>(
          InlinedHashSet<std::string_view>{kWebGpuExecutionProvider, kJsExecutionProvider}),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckFusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionRejectsEpsilonMismatch) {
  BuildOptions opts;
  opts.q_epsilon = 1e-6f;
  opts.k_epsilon = 1e-5f;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_,
      std::make_unique<GroupQueryAttentionPreNormFusion>(
          InlinedHashSet<std::string_view>{kWebGpuExecutionProvider, kJsExecutionProvider}),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionRejectsBadInnerReshape) {
  BuildOptions opts;
  opts.break_k_inner_reshape_shape = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_,
      std::make_unique<GroupQueryAttentionPreNormFusion>(
          InlinedHashSet<std::string_view>{kWebGpuExecutionProvider, kJsExecutionProvider}),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionRejectsNon1DNormWeight) {
  BuildOptions opts;
  opts.break_q_norm_weight_shape = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_,
      std::make_unique<GroupQueryAttentionPreNormFusion>(
          InlinedHashSet<std::string_view>{kWebGpuExecutionProvider, kJsExecutionProvider}),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionSkipsUnmatchedEp) {
  // Build the pattern but assign all nodes to CPU EP. The fusion is gated to WebGPU/JS only,
  // so the graph must remain unfused.
  auto build = [](ModelTestBuilder& builder) {
    BuildQwenQkPostNormPattern(builder, BuildOptions{});
    for (auto& node : builder.graph_.Nodes()) {
      const_cast<Node&>(node).SetExecutionProviderType(kCpuExecutionProvider);
    }
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_,
      std::make_unique<GroupQueryAttentionPreNormFusion>(
          InlinedHashSet<std::string_view>{kWebGpuExecutionProvider, kJsExecutionProvider}),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
