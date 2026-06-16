// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/group_query_attention_pre_norm_fusion.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

constexpr const char* kQkNormEpsilonAttrName = "qk_norm_epsilon";
constexpr float kEpsilonTolerance = 1e-9f;

bool HasInput(const Node& node, size_t index) {
  return index < node.InputDefs().size() && node.InputDefs()[index] != nullptr &&
         !node.InputDefs()[index]->Name().empty();
}

bool HasProducedOutput(const Node& node, size_t index) {
  return index < node.OutputDefs().size() && node.OutputDefs()[index] != nullptr &&
         !node.OutputDefs()[index]->Name().empty();
}

bool IsGraphOutput(const Graph& graph, const NodeArg* arg) {
  if (arg == nullptr || arg->Name().empty()) {
    return false;
  }
  for (const auto* graph_output : graph.GetOutputs()) {
    if (graph_output != nullptr && graph_output->Name() == arg->Name()) {
      return true;
    }
  }
  return false;
}

// Walks back from `consumer` via input slot `consumer_input_index` and matches:
//     producer_proj -> Reshape(reshape_inner) -> SimplifiedLayerNormalization(sln) -> Reshape(reshape_outer) -> consumer
// (`reshape_inner` is the one closest to the projection: it reshapes the (batch, seq, hidden)
// tensor to (batch, seq, num_heads, head_size). `reshape_outer` is the one closest to the
// consumer: it folds back to (batch, seq, hidden).)
// On success returns true and fills the out-pointers. Each intermediate node must have a single
// consumer (the next op in the chain) and must not be a graph output.
bool MatchPreNormReshapeChain(Graph& graph,
                              Node& consumer,
                              int consumer_input_index,
                              int64_t expected_head_size,
                              int64_t expected_hidden_size,
                              Node*& reshape_outer_out,
                              Node*& sln_out,
                              Node*& reshape_inner_out,
                              NodeArg*& projection_arg_out,
                              NodeArg*& norm_weight_arg_out,
                              float& epsilon_out) {
  reshape_outer_out = nullptr;
  sln_out = nullptr;
  reshape_inner_out = nullptr;
  projection_arg_out = nullptr;
  norm_weight_arg_out = nullptr;
  epsilon_out = 0.0f;

  if (consumer_input_index < 0 ||
      static_cast<size_t>(consumer_input_index) >= consumer.InputDefs().size()) {
    return false;
  }

  NodeArg* consumer_input = consumer.MutableInputDefs()[consumer_input_index];
  if (consumer_input == nullptr || consumer_input->Name().empty()) {
    return false;
  }

  Node* reshape_outer = graph.GetMutableProducerNode(consumer_input->Name());
  if (reshape_outer == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*reshape_outer, "Reshape", {5, 13, 14, 19, 21, 23, 24, 25})) {
    return false;
  }
  if (reshape_outer->GetOutputEdgesCount() != 1) {
    return false;
  }
  if (IsGraphOutput(graph, reshape_outer->OutputDefs()[0])) {
    return false;
  }

  // Validate outer reshape output last dim equals hidden size (num_heads * head_size).
  const auto* reshape_outer_shape = reshape_outer->OutputDefs()[0]->Shape();
  if (reshape_outer_shape == nullptr || reshape_outer_shape->dim_size() < 1) {
    return false;
  }
  const auto& reshape_outer_last = reshape_outer_shape->dim(reshape_outer_shape->dim_size() - 1);
  if (!reshape_outer_last.has_dim_value() || reshape_outer_last.dim_value() != expected_hidden_size) {
    return false;
  }

  if (reshape_outer->InputDefs().empty() || reshape_outer->InputDefs()[0] == nullptr) {
    return false;
  }
  Node* sln = graph.GetMutableProducerNode(reshape_outer->InputDefs()[0]->Name());
  if (sln == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*sln, "SimplifiedLayerNormalization", {1})) {
    return false;
  }
  if (sln->GetOutputEdgesCount() != 1) {
    return false;
  }
  if (IsGraphOutput(graph, sln->OutputDefs()[0])) {
    return false;
  }
  // SLN may emit auxiliary outputs (mean / inv_std). They must not be consumed elsewhere.
  for (size_t i = 1; i < sln->OutputDefs().size(); ++i) {
    if (HasProducedOutput(*sln, i)) {
      return false;
    }
  }
  if (sln->InputDefs().size() < 2 || sln->InputDefs()[1] == nullptr ||
      sln->InputDefs()[1]->Name().empty()) {
    return false;
  }

  // SimplifiedLayerNormalization permits its input (T), scale (V) and output (T) to use different
  // element types. The fused GroupQueryAttention input slots reuse the projection's element type
  // (T), so we can only fuse when scale and output also use T -- otherwise the rewrite would
  // change the node's type constraints and produce a semantically different graph. Require all
  // three to match before fusing.
  auto get_elem_type = [](const NodeArg* arg) -> int32_t {
    if (arg == nullptr) {
      return ONNX_NAMESPACE::TensorProto::UNDEFINED;
    }
    const auto* type_proto = arg->TypeAsProto();
    if (type_proto == nullptr || !type_proto->has_tensor_type() ||
        !type_proto->tensor_type().has_elem_type()) {
      return ONNX_NAMESPACE::TensorProto::UNDEFINED;
    }
    return type_proto->tensor_type().elem_type();
  };
  const int32_t sln_input_elem_type = get_elem_type(sln->InputDefs()[0]);
  const int32_t sln_scale_elem_type = get_elem_type(sln->InputDefs()[1]);
  const int32_t sln_output_elem_type = get_elem_type(sln->OutputDefs()[0]);
  if (sln_input_elem_type == ONNX_NAMESPACE::TensorProto::UNDEFINED ||
      sln_input_elem_type != sln_scale_elem_type ||
      sln_input_elem_type != sln_output_elem_type) {
    return false;
  }

  // Norm weight must be an initializer of shape [head_size].
  NodeArg* norm_weight_arg = sln->MutableInputDefs()[1];
  const ONNX_NAMESPACE::TensorProto* norm_weight_tensor =
      graph_utils::GetConstantInitializer(graph, norm_weight_arg->Name());
  if (norm_weight_tensor == nullptr) {
    return false;
  }
  if (norm_weight_tensor->dims_size() != 1 || norm_weight_tensor->dims(0) != expected_head_size) {
    return false;
  }

  const auto* sln_axis_attr = graph_utils::GetNodeAttribute(*sln, "axis");
  const int64_t sln_axis = (sln_axis_attr == nullptr) ? -1 : sln_axis_attr->i();
  if (sln_axis != -1) {
    return false;
  }
  const auto* sln_eps_attr = graph_utils::GetNodeAttribute(*sln, "epsilon");
  const float sln_eps = (sln_eps_attr == nullptr) ? 1e-5f : sln_eps_attr->f();

  // Inner reshape (between projection and SLN).
  if (sln->InputDefs().empty() || sln->InputDefs()[0] == nullptr) {
    return false;
  }
  Node* reshape_inner = graph.GetMutableProducerNode(sln->InputDefs()[0]->Name());
  if (reshape_inner == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*reshape_inner, "Reshape", {5, 13, 14, 19, 21, 23, 24, 25})) {
    return false;
  }
  if (reshape_inner->GetOutputEdgesCount() != 1) {
    return false;
  }
  if (IsGraphOutput(graph, reshape_inner->OutputDefs()[0])) {
    return false;
  }
  const auto* reshape_inner_shape = reshape_inner->OutputDefs()[0]->Shape();
  if (reshape_inner_shape == nullptr || reshape_inner_shape->dim_size() < 1) {
    return false;
  }
  const auto& reshape_inner_last = reshape_inner_shape->dim(reshape_inner_shape->dim_size() - 1);
  if (!reshape_inner_last.has_dim_value() || reshape_inner_last.dim_value() != expected_head_size) {
    return false;
  }

  if (reshape_inner->InputDefs().empty() || reshape_inner->InputDefs()[0] == nullptr) {
    return false;
  }

  reshape_outer_out = reshape_outer;
  sln_out = sln;
  reshape_inner_out = reshape_inner;
  projection_arg_out = reshape_inner->MutableInputDefs()[0];
  norm_weight_arg_out = norm_weight_arg;
  epsilon_out = sln_eps;
  return true;
}

}  // namespace

Status GroupQueryAttentionPreNormFusion::ApplyImpl(Graph& graph,
                                                   bool& modified,
                                                   int graph_level,
                                                   const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr) {
      continue;
    }
    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GroupQueryAttention", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // Already fused?
    if (HasInput(node, 14) || HasInput(node, 15)) {
      continue;
    }

    // Need at least query (0), key (1), value (2), past_key (3) so we can read head_size.
    // Requiring K at slot 1 also excludes the packed-QKV form (Q occupies slot 0 and K/V
    // slots are empty), which the WebGPU fused prologue does not support.
    if (node.InputDefs().size() < 4 || !HasInput(node, 0) || !HasInput(node, 1) || !HasInput(node, 2)) {
      continue;
    }

    // The fused decode prologue only applies when rotary embedding is enabled (Qwen3-style
    // configuration). If the GQA node has do_rotary=0 the kernel will reject the rewritten
    // node, so skip the fusion here to avoid that regression.
    const auto& gqa_attrs = node.GetAttributes();
    auto do_rotary_it = gqa_attrs.find("do_rotary");
    const int64_t do_rotary = (do_rotary_it == gqa_attrs.end()) ? 0 : do_rotary_it->second.i();
    if (do_rotary != 1) {
      continue;
    }
    const NodeArg* past_key_arg = node.InputDefs()[3];
    if (past_key_arg == nullptr || past_key_arg->Shape() == nullptr ||
        past_key_arg->Shape()->dim_size() < 4) {
      continue;
    }
    const auto& head_size_dim = past_key_arg->Shape()->dim(3);
    if (!head_size_dim.has_dim_value()) {
      continue;
    }
    const int64_t head_size = head_size_dim.dim_value();

    auto num_heads_it = gqa_attrs.find("num_heads");
    auto kv_num_heads_it = gqa_attrs.find("kv_num_heads");
    if (num_heads_it == gqa_attrs.end() || kv_num_heads_it == gqa_attrs.end()) {
      continue;
    }
    const int64_t num_heads = num_heads_it->second.i();
    const int64_t kv_num_heads = kv_num_heads_it->second.i();
    const int64_t q_hidden_size = num_heads * head_size;
    const int64_t kv_hidden_size = kv_num_heads * head_size;

    // Match pre-norm Reshape -> SLN -> Reshape on Q (slot 0) and K (slot 1).
    Node* q_reshape_outer = nullptr;
    Node* q_sln = nullptr;
    Node* q_reshape_inner = nullptr;
    NodeArg* q_projection_arg = nullptr;
    NodeArg* q_norm_weight_arg = nullptr;
    float q_epsilon = 0.0f;
    if (!MatchPreNormReshapeChain(graph, node, /*consumer_input_index=*/0, head_size, q_hidden_size,
                                  q_reshape_outer, q_sln, q_reshape_inner,
                                  q_projection_arg, q_norm_weight_arg, q_epsilon)) {
      continue;
    }

    Node* k_reshape_outer = nullptr;
    Node* k_sln = nullptr;
    Node* k_reshape_inner = nullptr;
    NodeArg* k_projection_arg = nullptr;
    NodeArg* k_norm_weight_arg = nullptr;
    float k_epsilon = 0.0f;
    if (!MatchPreNormReshapeChain(graph, node, /*consumer_input_index=*/1, head_size, kv_hidden_size,
                                  k_reshape_outer, k_sln, k_reshape_inner,
                                  k_projection_arg, k_norm_weight_arg, k_epsilon)) {
      continue;
    }

    if (std::fabs(q_epsilon - k_epsilon) > kEpsilonTolerance) {
      continue;
    }

    LOGS(logger, VERBOSE) << "GroupQueryAttentionPreNormFusion: matched gqa='" << node.Name()
                          << "' q_sln='" << q_sln->Name() << "' k_sln='" << k_sln->Name()
                          << "' head_size=" << head_size
                          << " num_heads=" << num_heads << " kv_num_heads=" << kv_num_heads
                          << " epsilon=" << q_epsilon;

    // Build new GQA inputs: copy existing inputs, replace 0/1 with projection outputs,
    // pad up to slot 13 with empty NodeArgs, then add q/k norm weights at 14/15.
    NodeArg& empty_arg = graph.GetOrCreateNodeArg("", nullptr);
    InlinedVector<NodeArg*> new_inputs;
    new_inputs.reserve(16);
    for (size_t i = 0; i < 16; ++i) {
      if (i == 0) {
        new_inputs.push_back(q_projection_arg);
      } else if (i == 1) {
        new_inputs.push_back(k_projection_arg);
      } else if (i == 14) {
        new_inputs.push_back(q_norm_weight_arg);
      } else if (i == 15) {
        new_inputs.push_back(k_norm_weight_arg);
      } else if (i < node.InputDefs().size()) {
        NodeArg* existing = node.MutableInputDefs()[i];
        new_inputs.push_back((existing != nullptr && !existing->Name().empty()) ? existing : &empty_arg);
      } else {
        new_inputs.push_back(&empty_arg);
      }
    }

    // Outputs: keep the same NodeArgs so downstream consumers and graph outputs are preserved.
    InlinedVector<NodeArg*> new_outputs;
    new_outputs.reserve(node.OutputDefs().size());
    for (auto* out : node.OutputDefs()) {
      new_outputs.push_back(const_cast<NodeArg*>(out));
    }

    // Copy attributes and add qk_norm_epsilon.
    NodeAttributes new_attrs = node.GetAttributes();
    utils::SetNodeAttribute(utils::MakeAttribute(std::string(kQkNormEpsilonAttrName), q_epsilon), new_attrs);

    const std::string original_name = node.Name();
    const std::string original_ep = node.GetExecutionProviderType();

    // Snapshot the GQA's original input edges (we will rewire them, except for slots 0/1).
    auto gqa_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(node);
    auto gqa_output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(node);

    // Remove all involved nodes (their input edges from elsewhere drop with them).
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.Index());
    graph_utils::RemoveNodeOutputEdges(graph, *q_reshape_outer);
    graph.RemoveNode(q_reshape_outer->Index());
    graph_utils::RemoveNodeOutputEdges(graph, *q_sln);
    graph.RemoveNode(q_sln->Index());
    graph_utils::RemoveNodeOutputEdges(graph, *q_reshape_inner);
    graph.RemoveNode(q_reshape_inner->Index());
    graph_utils::RemoveNodeOutputEdges(graph, *k_reshape_outer);
    graph.RemoveNode(k_reshape_outer->Index());
    graph_utils::RemoveNodeOutputEdges(graph, *k_sln);
    graph.RemoveNode(k_sln->Index());
    graph_utils::RemoveNodeOutputEdges(graph, *k_reshape_inner);
    graph.RemoveNode(k_reshape_inner->Index());

    Node& fused = graph.AddNode(graph.GenerateNodeName(original_name + "_qknorm"),
                                "GroupQueryAttention",
                                "GroupQueryAttention with fused per-head Q/K RMSNorm",
                                new_inputs,
                                new_outputs,
                                &new_attrs,
                                kMSDomain);
    fused.SetExecutionProviderType(original_ep);

    // Rewire upstream edges that fed the original GQA. Skip slots 0 and 1 (now driven by
    // the projection outputs which are still produced by their upstream nodes; the
    // graph.AddNode + matching NodeArg name will let the graph's edge resolver re-attach
    // those producer edges automatically when Resolve() runs, but we add them explicitly
    // for safety).
    for (const auto& e : gqa_input_edges) {
      if (e.dst_arg_index == 0 || e.dst_arg_index == 1) {
        continue;
      }
      graph.AddEdge(e.src_node, fused.Index(), e.src_arg_index, e.dst_arg_index);
    }
    // Add explicit edges for the new query/key inputs from the projection nodes.
    if (Node* q_proj_node = graph.GetMutableProducerNode(q_projection_arg->Name())) {
      const int src_idx = graph_utils::GetNodeOutputIndexFromOutputName(*q_proj_node, q_projection_arg->Name());
      graph.AddEdge(q_proj_node->Index(), fused.Index(), src_idx, 0);
    }
    if (Node* k_proj_node = graph.GetMutableProducerNode(k_projection_arg->Name())) {
      const int src_idx = graph_utils::GetNodeOutputIndexFromOutputName(*k_proj_node, k_projection_arg->Name());
      graph.AddEdge(k_proj_node->Index(), fused.Index(), src_idx, 1);
    }
    // Rewire downstream edges from the original GQA outputs.
    for (const auto& e : gqa_output_edges) {
      graph.AddEdge(fused.Index(), e.dst_node, e.src_arg_index, e.dst_arg_index);
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
