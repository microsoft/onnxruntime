// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>
#include <vector>

#include "core/framework/resource_accountant.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/group_query_attention_pre_norm_fusion.h"
#include "core/optimizer/utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/optimizer/webgpu_fusion_test_util.h"

#include "gtest/gtest.h"
#include "onnx/defs/schema.h"

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

namespace {

// Small geometry that exercises the Q/K post-norm pattern without needing real GPU work.
constexpr int64_t kBatch = 1;
constexpr int64_t kSeq = 1;
constexpr int64_t kNumHeads = 2;
constexpr int64_t kKvNumHeads = 1;
constexpr int64_t kHeadSize = 16;
constexpr int64_t kQHidden = kNumHeads * kHeadSize;
constexpr int64_t kKvHidden = kKvNumHeads * kHeadSize;
constexpr int64_t kMaxSeq = 8;

void SetWebGpu(Node& node) { node.SetExecutionProviderType(kWebGpuExecutionProvider); }

// Builds: [Reshape -> SimplifiedLayerNormalization -> Reshape] on Q and K, feeding a
// GroupQueryAttention node. V goes straight into GQA. The pattern is configured via
// BuildOptions so individual tests can flip a single attribute / shape / epsilon to
// exercise each gate.
struct BuildOptions {
  float q_epsilon = 1e-6f;
  float k_epsilon = 1e-6f;
  float projection_scale = 1.0f;
  bool preexisting_epsilon_overrides = false;
  // Sequence geometry for the Q/K/V projections that feed GQA.
  int64_t sequence_length = kSeq;
  // GQA control scalars.
  int32_t seqlens_k = 0;
  int32_t total_sequence_length = 1;
  // If true, the inner reshape on the K side targets a different last-dim than head_size
  // so the matcher must reject it.
  bool break_k_inner_reshape_shape = false;
  // If true, the q_norm_weight initializer is given a non-1D shape so the matcher must
  // reject it.
  bool break_q_norm_weight_shape = false;
  // GQA do_rotary attribute. The WebGPU fused prologue only supports do_rotary=1, so the
  // optimizer must skip the rewrite when this is 0.
  int64_t do_rotary = 1;
  // If true, drop the K input from the GQA node (slot 1 empty), simulating the packed-QKV
  // form. The optimizer must skip the rewrite in that case.
  bool packed_qkv = false;
  // If true, pre-populate the GQA node's slot 14 with a q_norm_weight initializer so the
  // optimizer treats the node as already fused and skips it.
  bool pre_fused = false;
  // If true, wire a query_seqlens input in slot 16 (slots 14/15 stay empty). The fusion fills slots
  // 14/15 and must keep the input in slot 16.
  bool with_query_seqlens = false;
};

void BuildQwenQkPostNormPattern(ModelTestBuilder& builder, const BuildOptions& opts) {
  const int64_t seq = opts.sequence_length;

  // Projection inputs (post linear projection, pre norm).
  NodeArg* q_proj = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, seq, kQHidden}, MLFloat16(-opts.projection_scale), MLFloat16(opts.projection_scale));
  NodeArg* k_proj = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, seq, kKvHidden}, MLFloat16(-opts.projection_scale), MLFloat16(opts.projection_scale));
  NodeArg* v_proj = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, seq, kKvHidden}, MLFloat16(-1.0f), MLFloat16(1.0f));

  // GQA cache + control inputs.
  NodeArg* past_key = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize}, MLFloat16(0.0f), MLFloat16(0.0f));
  NodeArg* past_value = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize}, MLFloat16(0.0f), MLFloat16(0.0f));
  // Note: ModelTestBuilder::MakeInput<int>(shape, min, max) calls Uniform(min, max - 1)
  // internally, which asserts on min == max. Use the explicit-data overload instead.
  NodeArg* seqlens_k = builder.MakeInput<int32_t>(std::vector<int64_t>{kBatch}, std::vector<int32_t>{opts.seqlens_k});
  NodeArg* total_seq_len =
      builder.MakeInput<int32_t>(std::vector<int64_t>{1}, std::vector<int32_t>{opts.total_sequence_length});

  // Rotary caches + position ids (slots 7/8/9). Keep these explicitly wired so
  // runtime parity tests execute a valid do_rotary=1 configuration.
  const int64_t half_rotary = kHeadSize / 2;
  std::vector<MLFloat16> cos_cache_data(static_cast<size_t>(kMaxSeq * half_rotary), MLFloat16(1.0f));
  std::vector<MLFloat16> sin_cache_data(static_cast<size_t>(kMaxSeq * half_rotary), MLFloat16(0.0f));
  NodeArg* cos_cache = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kMaxSeq, half_rotary}, cos_cache_data);
  NodeArg* sin_cache = builder.MakeInput<MLFloat16>(
      std::vector<int64_t>{kMaxSeq, half_rotary}, sin_cache_data);

  std::vector<int64_t> position_ids_data(static_cast<size_t>(kBatch * seq));
  const int64_t base_position = opts.total_sequence_length - seq;
  for (int64_t s = 0; s < seq; ++s) {
    position_ids_data[static_cast<size_t>(s)] = base_position + s;
  }
  NodeArg* position_ids = builder.MakeInput<int64_t>(
      std::vector<int64_t>{kBatch, seq}, position_ids_data);

  // Norm weight initializers: [head_size]. (Or non-1D when forcing a shape mismatch.)
  std::vector<int64_t> q_norm_weight_shape =
      opts.break_q_norm_weight_shape ? std::vector<int64_t>{1, kHeadSize} : std::vector<int64_t>{kHeadSize};
  NodeArg* q_norm_weight = builder.MakeInitializer<MLFloat16>(q_norm_weight_shape, MLFloat16(1.0f), MLFloat16(1.0f));
  NodeArg* k_norm_weight = builder.MakeInitializer<MLFloat16>({kHeadSize}, MLFloat16(1.0f), MLFloat16(1.0f));

  // Reshape "shape" initializers.
  NodeArg* reshape_to_per_head_q = builder.MakeInitializer<int64_t>({4}, {kBatch, seq, kNumHeads, kHeadSize});
  const int64_t k_inner_last_dim = opts.break_k_inner_reshape_shape ? (kHeadSize * 2) : kHeadSize;
  NodeArg* reshape_to_per_head_k =
      builder.MakeInitializer<int64_t>({4}, {kBatch, seq, kKvNumHeads, k_inner_last_dim});
  NodeArg* reshape_to_q_hidden = builder.MakeInitializer<int64_t>({3}, {kBatch, seq, kQHidden});
  NodeArg* reshape_to_kv_hidden = builder.MakeInitializer<int64_t>({3}, {kBatch, seq, kKvHidden});

  // Q-side chain.
  NodeArg* q_inner_reshape_out = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, seq, kNumHeads, kHeadSize});
  NodeArg* q_normed = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, seq, kNumHeads, kHeadSize});
  NodeArg* q_outer_reshape_out = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, seq, kQHidden});

  Node& q_inner_reshape = builder.AddNode("Reshape", {q_proj, reshape_to_per_head_q}, {q_inner_reshape_out});
  Node& q_sln = builder.AddNode("SimplifiedLayerNormalization", {q_inner_reshape_out, q_norm_weight}, {q_normed});
  q_sln.AddAttribute("axis", static_cast<int64_t>(-1));
  q_sln.AddAttribute("epsilon", opts.q_epsilon);
  Node& q_outer_reshape = builder.AddNode("Reshape", {q_normed, reshape_to_q_hidden}, {q_outer_reshape_out});

  // K-side chain.
  NodeArg* k_inner_reshape_out = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, seq, kKvNumHeads, k_inner_last_dim});
  NodeArg* k_normed = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, seq, kKvNumHeads, k_inner_last_dim});
  NodeArg* k_outer_reshape_out = builder.MakeIntermediate<MLFloat16>(
      std::vector<int64_t>{kBatch, seq, kKvHidden});

  Node& k_inner_reshape = builder.AddNode("Reshape", {k_proj, reshape_to_per_head_k}, {k_inner_reshape_out});
  Node& k_sln = builder.AddNode("SimplifiedLayerNormalization", {k_inner_reshape_out, k_norm_weight}, {k_normed});
  k_sln.AddAttribute("axis", static_cast<int64_t>(-1));
  k_sln.AddAttribute("epsilon", opts.k_epsilon);
  Node& k_outer_reshape = builder.AddNode("Reshape", {k_normed, reshape_to_kv_hidden}, {k_outer_reshape_out});

  // GQA outputs.
  NodeArg* gqa_out = builder.MakeOutput<MLFloat16>(std::vector<int64_t>{kBatch, seq, kQHidden});
  NodeArg* present_key = builder.MakeOutput<MLFloat16>(
      std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize});
  NodeArg* present_value = builder.MakeOutput<MLFloat16>(
      std::vector<int64_t>{kBatch, kKvNumHeads, kMaxSeq, kHeadSize});

  // Build the GQA input list. The packed_qkv variant drops K (slot 1) and V (slot 2).
  NodeArg& empty_arg = builder.graph_.GetOrCreateNodeArg("", nullptr);
  std::vector<NodeArg*> gqa_inputs;
  gqa_inputs.push_back(q_outer_reshape_out);
  gqa_inputs.push_back(opts.packed_qkv ? &empty_arg : k_outer_reshape_out);
  gqa_inputs.push_back(opts.packed_qkv ? &empty_arg : v_proj);
  gqa_inputs.push_back(past_key);
  gqa_inputs.push_back(past_value);
  gqa_inputs.push_back(seqlens_k);
  gqa_inputs.push_back(total_seq_len);
  gqa_inputs.push_back(cos_cache);     // slot 7
  gqa_inputs.push_back(sin_cache);     // slot 8
  gqa_inputs.push_back(position_ids);  // slot 9
  gqa_inputs.push_back(&empty_arg);    // slot 10: attention_bias
  gqa_inputs.push_back(&empty_arg);    // slot 11: head_sink
  gqa_inputs.push_back(&empty_arg);    // slot 12: k_scale
  gqa_inputs.push_back(&empty_arg);    // slot 13: v_scale

  if (opts.pre_fused) {
    // Place a real norm weight in slot 14 to emulate an already-fused node.
    NodeArg* preexisting_q_norm =
        builder.MakeInitializer<MLFloat16>({kHeadSize}, MLFloat16(1.0f), MLFloat16(1.0f));
    gqa_inputs.push_back(preexisting_q_norm);
  }

  if (opts.with_query_seqlens) {
    gqa_inputs.push_back(&empty_arg);  // slot 14: q_norm_weight
    gqa_inputs.push_back(&empty_arg);  // slot 15: k_norm_weight
    NodeArg* query_seqlens = builder.MakeInput<int32_t>(std::vector<int64_t>{kBatch},
                                                        std::vector<int32_t>{static_cast<int32_t>(seq)});
    gqa_inputs.push_back(query_seqlens);  // slot 16
  }
  Node& gqa = builder.AddNode("GroupQueryAttention",
                              gqa_inputs,
                              {gqa_out, present_key, present_value},
                              kMSDomain);
  gqa.AddAttribute("num_heads", static_cast<int64_t>(kNumHeads));
  gqa.AddAttribute("kv_num_heads", static_cast<int64_t>(kKvNumHeads));
  gqa.AddAttribute("do_rotary", opts.do_rotary);
  if (opts.preexisting_epsilon_overrides) {
    gqa.AddAttribute("q_norm_epsilon", 0.1f);
    gqa.AddAttribute("k_norm_epsilon", 0.2f);
  }

  SetWebGpu(q_inner_reshape);
  SetWebGpu(q_sln);
  SetWebGpu(q_outer_reshape);
  SetWebGpu(k_inner_reshape);
  SetWebGpu(k_sln);
  SetWebGpu(k_outer_reshape);
  SetWebGpu(gqa);
}

Status CheckFusedGraphWithEpsilons(const Graph& graph, float q_epsilon, float k_epsilon) {
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
    ORT_RETURN_IF_NOT(eps_it->second.f() == q_epsilon, "qk_norm_epsilon value mismatch.");
    const auto q_eps_it = attrs.find("q_norm_epsilon");
    const auto k_eps_it = attrs.find("k_norm_epsilon");
    if (q_epsilon == k_epsilon) {
      ORT_RETURN_IF_NOT(q_eps_it == attrs.end() && k_eps_it == attrs.end(),
                        "Equal epsilons should use the shared attribute.");
    } else {
      ORT_RETURN_IF_NOT(q_eps_it != attrs.end() && q_eps_it->second.f() == q_epsilon,
                        "q_norm_epsilon value mismatch.");
      ORT_RETURN_IF_NOT(k_eps_it != attrs.end() && k_eps_it->second.f() == k_epsilon,
                        "k_norm_epsilon value mismatch.");
    }
  }
  return Status::OK();
}

Status CheckFusedGraph(const Graph& graph) {
  return CheckFusedGraphWithEpsilons(graph, 1e-6f, 1e-6f);
}

Status CheckUnfusedGraph(const Graph& graph) {
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

// Helper: build the transformer for the original WebGPU-only tests.
std::unique_ptr<GroupQueryAttentionPreNormFusion> MakeWebGpuTransformer() {
  return std::make_unique<GroupQueryAttentionPreNormFusion>(
      InlinedHashSet<std::string_view>{kWebGpuExecutionProvider});
}

// Helper: build the production-compatible transformer.
std::unique_ptr<GroupQueryAttentionPreNormFusion> MakeCudaWebGpuTransformer() {
  return std::make_unique<GroupQueryAttentionPreNormFusion>(
      InlinedHashSet<std::string_view>{kCudaExecutionProvider, kWebGpuExecutionProvider});
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionFusesQwenPattern) {
  auto build = [](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, BuildOptions{}); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeCudaWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckFusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionTransfersWorkspaceReservations) {
  auto build = [](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, BuildOptions{}); };
  NodeWorkspaceReservationMap reservations;
  size_t total_reserved_bytes = 0;
  auto install_reservation_callback = [&](Graph& graph) {
    for (const Node& node : graph.Nodes()) {
      reservations.insert_or_assign(
          node.Index(), WorkspaceEstimateSelection{10, WorkspaceEstimateSource::kFallback});
      total_reserved_bytes += 10;
    }
    graph.SetNodeReplacementCallback(
        [&reservations](const Graph&,
                        gsl::span<const NodeIndex> source_node_indices,
                        NodeIndex destination_node_index) {
          ConsolidateWorkspaceReservations(
              reservations, source_node_indices, destination_node_index);
        });
    return Status::OK();
  };
  auto check_reservations = [&](Graph& graph) {
    graph.SetNodeReplacementCallback({});
    ORT_RETURN_IF_ERROR(CheckFusedGraph(graph));
    ORT_RETURN_IF_NOT(reservations.size() == 1, "Expected one consolidated workspace reservation.");

    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "GroupQueryAttention") {
        ORT_RETURN_IF_NOT(
            reservations.at(node.Index()).bytes == total_reserved_bytes,
            "Fused GQA workspace reservation did not include all removed nodes.");
        return Status::OK();
      }
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Fused GQA node not found.");
  };

  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeCudaWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, install_reservation_callback, check_reservations));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionFusesCudaAssignedQwenPattern) {
  auto build = [](ModelTestBuilder& builder) {
    BuildQwenQkPostNormPattern(builder, BuildOptions{});
    for (auto& node : builder.graph_.Nodes()) {
      node.SetExecutionProviderType(kCudaExecutionProvider);
    }
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeCudaWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckFusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionFusesQwenPatternCurrentOpset) {
  // Uses the current max ONNX opset to catch version-list drift.
  // If this fails, update version lists in
  // onnxruntime/core/optimizer/group_query_attention_pre_norm_fusion.cc.
  const auto& map = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().Map();
  auto it = map.find(ONNX_NAMESPACE::ONNX_DOMAIN);
  ASSERT_TRUE(it != map.end()) << "ONNX domain not found in OpSchemaRegistry";
  int current_opset = it->second.second;
  auto build = [](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, BuildOptions{}); };
  // opset 27 is under development in ONNX 1.22 (released map-max 27 > last release 26), so strict legs
  // reject this *CurrentOpset model at load; allow the unreleased opset. Remove once opset 27 ships.
  // Tracked by #28966; this runs the CUDA+WebGPU attention-fusion path that surfaced #28969.
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/current_opset, *logger_, MakeCudaWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckFusedGraph,
      ModelOptions{kAllowReleasedOpsetsOnly, /*strict_shape_type_inference*/ false}));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionMatchesUnfusedWebGpuResults) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  // Baseline graph: explicit Reshape + SLN on Q/K feeding GQA.
  // Target graph: optimizer rewires the same math into GQA slot 14/15 q/k norm
  // weights and removes the explicit pre-norm subgraph.
  auto build = [](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, BuildOptions{}); };
  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    ASSERT_STATUS_OK(CheckFusedGraph(session.GetGraph()));
  };

  // Run the same random feeds through baseline (Level1) and fused (Level2) WebGPU
  // sessions and require output parity. This guards semantic equivalence beyond
  // pure graph-shape/op-count assertions.
  RunWebGpuFusionTransformerTest(
      build,
      check_transformed_graph,
      TransformerLevel::Level1,
      TransformerLevel::Level2,
      21,
      1e-3,
      1e-3,
      MakeWebGpuTransformer(),
      []() { return DefaultWebGpuExecutionProvider(); });
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionMatchesUnfusedWebGpuResultsPrefill) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU EP unavailable in this build.";
  }

  // Prompt/prefill geometry: sequence_length > 1 and seqlens_k points at the
  // last valid prompt token.
  BuildOptions opts;
  opts.sequence_length = 2;
  opts.seqlens_k = 1;
  opts.total_sequence_length = 2;

  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    ASSERT_STATUS_OK(CheckFusedGraph(session.GetGraph()));
  };

  RunWebGpuFusionTransformerTest(
      build,
      check_transformed_graph,
      TransformerLevel::Level1,
      TransformerLevel::Level2,
      21,
      1e-3,
      1e-3,
      MakeWebGpuTransformer(),
      []() { return DefaultWebGpuExecutionProvider(); });
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionFusesDifferentEpsilons) {
  BuildOptions opts;
  opts.q_epsilon = 1e-6f;
  opts.k_epsilon = 1e-5f;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  auto check = [opts](const Graph& graph) {
    return CheckFusedGraphWithEpsilons(graph, opts.q_epsilon, opts.k_epsilon);
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, check));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionFusesSmallDifferentEpsilons) {
  BuildOptions opts;
  opts.q_epsilon = 1e-9f;
  opts.k_epsilon = 1.9e-9f;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  auto check = [opts](const Graph& graph) {
    return CheckFusedGraphWithEpsilons(graph, opts.q_epsilon, opts.k_epsilon);
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeCudaWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, check));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionFusesAdjacentEpsilons) {
  BuildOptions opts;
  opts.k_epsilon = std::nextafter(opts.q_epsilon, 1.0f);
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  auto check = [opts](const Graph& graph) {
    return CheckFusedGraphWithEpsilons(graph, opts.q_epsilon, opts.k_epsilon);
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeCudaWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, check));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionReplacesExistingEpsilonOverrides) {
  for (bool different_epsilons : {false, true}) {
    SCOPED_TRACE(different_epsilons);
    BuildOptions opts;
    opts.preexisting_epsilon_overrides = true;
    if (different_epsilons) opts.k_epsilon = 1e-5f;
    auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
    auto check = [opts](const Graph& graph) {
      return CheckFusedGraphWithEpsilons(graph, opts.q_epsilon, opts.k_epsilon);
    };
    ASSERT_STATUS_OK(TestGraphTransformer(
        build, /*opset_version=*/21, *logger_, MakeCudaWebGpuTransformer(),
        TransformerLevel::Level2, /*steps=*/1, nullptr, check));
  }
}

static void RunEpsilonFusionTest(bool use_cuda, int64_t sequence_length, bool equal_epsilons = false) {
  auto ep_factory = [use_cuda]() {
    return use_cuda ? DefaultCudaExecutionProvider() : DefaultWebGpuExecutionProvider();
  };
  if (!ep_factory()) {
    GTEST_SKIP() << "Requested GPU EP is unavailable in this build.";
  }

  BuildOptions opts;
  opts.q_epsilon = 1e-9f;
  opts.k_epsilon = equal_epsilons ? opts.q_epsilon : 1.9e-9f;
  opts.projection_scale = 1e-4f;
  opts.sequence_length = sequence_length;
  opts.seqlens_k = static_cast<int32_t>(sequence_length);
  opts.total_sequence_length = opts.seqlens_k + 1;
  const auto provider_type = use_cuda ? kCudaExecutionProvider : kWebGpuExecutionProvider;
  auto build = [opts, provider_type](ModelTestBuilder& builder) {
    BuildQwenQkPostNormPattern(builder, opts);
    for (auto& node : builder.graph_.Nodes()) {
      node.SetExecutionProviderType(provider_type);
    }
  };
  auto check = [opts, provider_type](InferenceSessionWrapper& session) {
    ASSERT_STATUS_OK(CheckFusedGraphWithEpsilons(session.GetGraph(), opts.q_epsilon, opts.k_epsilon));
    for (const auto& node : session.GetGraph().Nodes()) {
      if (node.OpType() == "GroupQueryAttention") {
        EXPECT_EQ(node.GetExecutionProviderType(), provider_type);
      }
    }
  };
  RunWebGpuFusionTransformerTest(
      build, check, TransformerLevel::Level1, TransformerLevel::Level2, 21,
      1e-3, 1e-3, MakeCudaWebGpuTransformer(), ep_factory,
      [](SessionOptions& options) {
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
      });
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionDifferentEpsilonsWebGpuDecode) {
  RunEpsilonFusionTest(/*use_cuda=*/false, /*sequence_length=*/1);
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionDifferentEpsilonsWebGpuPrefill) {
  RunEpsilonFusionTest(/*use_cuda=*/false, /*sequence_length=*/2);
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionDifferentEpsilonsCudaDecode) {
  RunEpsilonFusionTest(/*use_cuda=*/true, /*sequence_length=*/1);
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionDifferentEpsilonsCudaPrefill) {
  RunEpsilonFusionTest(/*use_cuda=*/true, /*sequence_length=*/2);
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionEqualEpsilonsCudaDecode) {
  RunEpsilonFusionTest(/*use_cuda=*/true, /*sequence_length=*/1, /*equal_epsilons=*/true);
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionEqualEpsilonsCudaPrefill) {
  RunEpsilonFusionTest(/*use_cuda=*/true, /*sequence_length=*/2, /*equal_epsilons=*/true);
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionRejectsNonPositiveEpsilon) {
  BuildOptions opts;
  opts.q_epsilon = 0.0f;
  opts.k_epsilon = 0.0f;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionRejectsNonFiniteEpsilon) {
  BuildOptions opts;
  opts.q_epsilon = std::numeric_limits<float>::quiet_NaN();
  opts.k_epsilon = std::numeric_limits<float>::quiet_NaN();
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionRejectsBadInnerReshape) {
  BuildOptions opts;
  opts.break_k_inner_reshape_shape = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionRejectsNon1DNormWeight) {
  BuildOptions opts;
  opts.break_q_norm_weight_shape = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionSkipsCpuEp) {
  // Build the pattern but assign all nodes to CPU EP. The fusion is gated to CUDA/WebGPU only,
  // so the graph must remain unfused.
  auto build = [](ModelTestBuilder& builder) {
    BuildQwenQkPostNormPattern(builder, BuildOptions{});
    for (auto& node : builder.graph_.Nodes()) {
      const_cast<Node&>(node).SetExecutionProviderType(kCpuExecutionProvider);
    }
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeCudaWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionSkipsJsEp) {
  // JSEP does not implement the fused per-head Q/K RMSNorm prologue, so the optimizer
  // (which we now register for CUDA/WebGPU only) must leave JSEP-assigned graphs alone.
  auto build = [](ModelTestBuilder& builder) {
    BuildQwenQkPostNormPattern(builder, BuildOptions{});
    for (auto& node : builder.graph_.Nodes()) {
      const_cast<Node&>(node).SetExecutionProviderType(kJsExecutionProvider);
    }
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeCudaWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionSkipsWhenDoRotaryDisabled) {
  // The WebGPU fused prologue requires do_rotary=1; the optimizer must skip otherwise so
  // the runtime guard never trips.
  BuildOptions opts;
  opts.do_rotary = 0;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionSkipsPackedQkv) {
  // Packed-QKV form leaves slots 1 and 2 empty; the WebGPU fused prologue does not support
  // it, so the optimizer must skip the rewrite.
  BuildOptions opts;
  opts.packed_qkv = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, CheckUnfusedGraph));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionSkipsAlreadyFusedNode) {
  // If the GQA node already exposes a q_norm_weight (slot 14) input the optimizer must
  // treat it as already fused and leave the surrounding SLN/Reshape ops in place. The
  // standard CheckUnfusedGraph helper rejects any wiring at slot 14, so use a custom
  // checker that only verifies the surrounding ops weren't removed.
  BuildOptions opts;
  opts.pre_fused = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };
  auto check = [](Graph& graph) -> Status {
    const auto op_to_count = CountOpsInGraph(graph);
    ORT_RETURN_IF_NOT(OpCount(op_to_count, "com.microsoft.GroupQueryAttention") == 1,
                      "Already-fused test: GQA count changed.");
    ORT_RETURN_IF_NOT(OpCount(op_to_count, "SimplifiedLayerNormalization") == 2,
                      "Already-fused test: SLN ops were removed.");
    ORT_RETURN_IF_NOT(OpCount(op_to_count, "Reshape") == 4,
                      "Already-fused test: Reshape ops were removed.");
    return Status::OK();
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, nullptr, check));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionPreNormFusionPreservesQuerySeqlensInput) {
  // The rewrite rebuilds the GQA input list. An input past slot 15 (query_seqlens at slot 16) must
  // survive it, otherwise a right-padded chunk would silently be computed as an unpadded one.
  BuildOptions opts;
  opts.with_query_seqlens = true;
  auto build = [opts](ModelTestBuilder& builder) { BuildQwenQkPostNormPattern(builder, opts); };

  std::string query_seqlens_name;
  auto pre_check = [&query_seqlens_name](Graph& graph) -> Status {
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() != "GroupQueryAttention") continue;
      ORT_RETURN_IF_NOT(node.InputDefs().size() == 17 && node.InputDefs()[16]->Exists(),
                        "Test setup: GQA must have a query_seqlens input in slot 16.");
      query_seqlens_name = node.InputDefs()[16]->Name();
    }
    return Status::OK();
  };
  auto post_check = [&query_seqlens_name](Graph& graph) -> Status {
    ORT_RETURN_IF_ERROR(CheckFusedGraph(graph));
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() != "GroupQueryAttention") continue;
      ORT_RETURN_IF_NOT(node.InputDefs().size() == 17, "Fused GQA must keep the query_seqlens input slot.");
      ORT_RETURN_IF_NOT(node.InputDefs()[16]->Exists() && node.InputDefs()[16]->Name() == query_seqlens_name,
                        "Fused GQA lost or replaced the query_seqlens input.");
    }
    return Status::OK();
  };
  ASSERT_STATUS_OK(TestGraphTransformer(
      build, /*opset_version=*/21, *logger_, MakeCudaWebGpuTransformer(),
      TransformerLevel::Level2, /*steps=*/1, pre_check, post_check));
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
