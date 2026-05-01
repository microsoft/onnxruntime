// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_nbits_qkv_fusion.h"

#include <array>
#include <vector>

#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

bool HasInput(const Node& node, size_t index) {
  return index < node.InputDefs().size() && node.InputDefs()[index] != nullptr && !node.InputDefs()[index]->Name().empty();
}

bool IsSupportedSimplifiedLayerNormalization(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "SimplifiedLayerNormalization", {1});
}

bool IsSupportedSkipSimplifiedLayerNormalization(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "SkipSimplifiedLayerNormalization", {1}, kMSDomain);
}

bool IsSupportedNormForFusion(const Node& node) {
  return IsSupportedSimplifiedLayerNormalization(node) || IsSupportedSkipSimplifiedLayerNormalization(node);
}

bool HasProducedOutput(const Node& node, size_t index) {
  return index < node.OutputDefs().size() && node.OutputDefs()[index] != nullptr && !node.OutputDefs()[index]->Name().empty();
}

bool IsMatMulNBitsWithoutOptionalInputs(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMulNBits", {1}, kMSDomain) &&
         !HasInput(node, 3) && !HasInput(node, 4) && !HasInput(node, 5);
}

int64_t GetIntAttr(const Node& node, const char* name, int64_t default_value, bool required = false) {
  const auto* attr = graph_utils::GetNodeAttribute(node, name);
  if (attr == nullptr) {
    ORT_ENFORCE(!required, "Missing required attribute ", name, " on node ", node.Name());
    return default_value;
  }

  return attr->i();
}

float GetFloatAttr(const Node& node, const char* name, float default_value) {
  const auto* attr = graph_utils::GetNodeAttribute(node, name);
  return attr == nullptr ? default_value : attr->f();
}

struct QkvNodes {
  const Node* q = nullptr;
  const Node* k = nullptr;
  const Node* v = nullptr;
};

std::optional<QkvNodes> GetQkvNodes(const Graph& graph, const Node& norm) {
  if (!HasProducedOutput(norm, 0) || graph.NodeProducesGraphOutput(norm)) {
    return std::nullopt;
  }

  std::array<const Node*, 3> consumers{};
  size_t consumer_index = 0;
  for (auto edge_it = norm.OutputEdgesBegin(); edge_it != norm.OutputEdgesEnd(); ++edge_it) {
    if (edge_it->GetSrcArgIndex() != 0) {
      continue;
    }

    if (consumer_index >= consumers.size()) {
      return std::nullopt;
    }

    if (edge_it->GetDstArgIndex() != 0) {
      return std::nullopt;
    }

    const Node* consumer = graph.GetNode(edge_it->GetNode().Index());
    if (consumer == nullptr || !IsMatMulNBitsWithoutOptionalInputs(*consumer)) {
      return std::nullopt;
    }

    consumers[consumer_index++] = consumer;
  }

  if (consumer_index != consumers.size()) {
    return std::nullopt;
  }

  const int64_t n0 = GetIntAttr(*consumers[0], "N", -1, true);
  const int64_t n1 = GetIntAttr(*consumers[1], "N", -1, true);
  const int64_t n2 = GetIntAttr(*consumers[2], "N", -1, true);

  QkvNodes qkv;
  if (n0 != n1 && n1 == n2) {
    qkv = {consumers[0], consumers[1], consumers[2]};
  } else if (n1 != n0 && n0 == n2) {
    qkv = {consumers[1], consumers[0], consumers[2]};
  } else if (n2 != n0 && n0 == n1) {
    qkv = {consumers[2], consumers[0], consumers[1]};
  } else {
    return std::nullopt;
  }

  return qkv;
}

bool HasSupportedExecutionProvider(const Node& node) {
  const auto& node_ep = node.GetExecutionProviderType();
  return node_ep.empty() || node_ep == kWebGpuExecutionProvider;
}

bool IsFuseCandidate(const Node& norm, const QkvNodes& qkv) {
  if (!IsSupportedNormForFusion(norm) || qkv.q == nullptr || qkv.k == nullptr || qkv.v == nullptr) {
    return false;
  }

  if (!HasSupportedExecutionProvider(norm) || !HasSupportedExecutionProvider(*qkv.q) ||
      !HasSupportedExecutionProvider(*qkv.k) || !HasSupportedExecutionProvider(*qkv.v)) {
    return false;
  }

  const size_t min_norm_inputs = IsSupportedSkipSimplifiedLayerNormalization(norm) ? 3u : 2u;
  if (norm.InputDefs().size() < min_norm_inputs || qkv.q->InputDefs().empty() || qkv.k->InputDefs().empty() || qkv.v->InputDefs().empty()) {
    return false;
  }

  if (qkv.q->InputDefs()[0] != norm.OutputDefs()[0] || qkv.k->InputDefs()[0] != norm.OutputDefs()[0] ||
      qkv.v->InputDefs()[0] != norm.OutputDefs()[0]) {
    return false;
  }

  const int64_t q_k = GetIntAttr(*qkv.q, "K", -1, true);
  const int64_t k_k = GetIntAttr(*qkv.k, "K", -1, true);
  const int64_t v_k = GetIntAttr(*qkv.v, "K", -1, true);
  const int64_t q_bits = GetIntAttr(*qkv.q, "bits", 4);
  const int64_t k_bits = GetIntAttr(*qkv.k, "bits", 4);
  const int64_t v_bits = GetIntAttr(*qkv.v, "bits", 4);
  const int64_t q_block_size = GetIntAttr(*qkv.q, "block_size", -1, true);
  const int64_t k_block_size = GetIntAttr(*qkv.k, "block_size", -1, true);
  const int64_t v_block_size = GetIntAttr(*qkv.v, "block_size", -1, true);
  const int64_t q_accuracy_level = GetIntAttr(*qkv.q, "accuracy_level", 0);
  const int64_t k_accuracy_level = GetIntAttr(*qkv.k, "accuracy_level", 0);
  const int64_t v_accuracy_level = GetIntAttr(*qkv.v, "accuracy_level", 0);

  return q_k == k_k && q_k == v_k &&
         q_bits == k_bits && q_bits == v_bits && q_bits == 4 &&
         q_block_size == k_block_size && q_block_size == v_block_size && q_block_size == 32 &&
         q_accuracy_level == k_accuracy_level && q_accuracy_level == v_accuracy_level;
}

}  // namespace

Status MatMulNBitsQkvFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                       const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr) {
      continue;
    }

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!IsSupportedNormForFusion(node)) {
      continue;
    }

    const auto qkv_nodes = GetQkvNodes(graph, node);
    if (!qkv_nodes || !IsFuseCandidate(node, *qkv_nodes)) {
      continue;
    }

    const int64_t K = GetIntAttr(*qkv_nodes->q, "K", -1, true);
    const int64_t Nq = GetIntAttr(*qkv_nodes->q, "N", -1, true);
    const int64_t Nkv = GetIntAttr(*qkv_nodes->k, "N", -1, true);
    const int64_t bits = GetIntAttr(*qkv_nodes->q, "bits", 4);
    const int64_t block_size = GetIntAttr(*qkv_nodes->q, "block_size", -1, true);
    const int64_t accuracy_level = GetIntAttr(*qkv_nodes->q, "accuracy_level", 0);
    const float epsilon = GetFloatAttr(node, "epsilon", 1e-6f);

    const bool is_skip_sln = IsSupportedSkipSimplifiedLayerNormalization(node);

    LOGS(logger, VERBOSE) << "MatMulNBitsQkvFusion: matched norm='" << node.Name()
                          << "' q='" << qkv_nodes->q->Name() << "' k='" << qkv_nodes->k->Name()
                          << "' v='" << qkv_nodes->v->Name() << "' attrs={K=" << K
                          << ", Nq=" << Nq << ", Nkv=" << Nkv << ", bits=" << bits
                          << ", block_size=" << block_size << ", accuracy_level=" << accuracy_level
                          << ", epsilon=" << epsilon << ", skip_sln=" << is_skip_sln << "}";

    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("K", K), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("Nq", Nq), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("Nkv", Nkv), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("bits", bits), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", accuracy_level), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("epsilon", epsilon), attrs);

    NodeArg& empty_arg = graph.GetOrCreateNodeArg("", nullptr);

    InlinedVector<NodeArg*> fused_inputs{
        const_cast<NodeArg*>(node.InputDefs()[0]),
        is_skip_sln ? const_cast<NodeArg*>(node.InputDefs()[1]) : &empty_arg,
        const_cast<NodeArg*>(node.InputDefs()[is_skip_sln ? 2 : 1]),
        const_cast<NodeArg*>(qkv_nodes->q->InputDefs()[1]),
        const_cast<NodeArg*>(qkv_nodes->q->InputDefs()[2]),
        const_cast<NodeArg*>(qkv_nodes->k->InputDefs()[1]),
        const_cast<NodeArg*>(qkv_nodes->k->InputDefs()[2]),
        const_cast<NodeArg*>(qkv_nodes->v->InputDefs()[1]),
        const_cast<NodeArg*>(qkv_nodes->v->InputDefs()[2]),
    };

    InlinedVector<NodeArg*> fused_outputs{
        const_cast<NodeArg*>(qkv_nodes->q->OutputDefs()[0]),
        const_cast<NodeArg*>(qkv_nodes->k->OutputDefs()[0]),
        const_cast<NodeArg*>(qkv_nodes->v->OutputDefs()[0]),
    };
    if (is_skip_sln && HasProducedOutput(node, 3)) {
      fused_outputs.push_back(const_cast<NodeArg*>(node.OutputDefs()[3]));
    }

    const bool has_residual_output = is_skip_sln && HasProducedOutput(node, 3);
    const std::string norm_name = node.Name();
    const std::string q_name = qkv_nodes->q->Name();
    const std::string k_name = qkv_nodes->k->Name();
    const std::string v_name = qkv_nodes->v->Name();

    const auto norm_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(node);
    const auto q_output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(*qkv_nodes->q);
    const auto k_output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(*qkv_nodes->k);
    const auto v_output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(*qkv_nodes->v);
    const auto norm_output_edges = has_residual_output
                                       ? graph_utils::GraphEdge::GetNodeOutputEdges(node)
                                       : std::vector<graph_utils::GraphEdge>{};
    graph_utils::RemoveNodeOutputEdges(graph, const_cast<Node&>(*qkv_nodes->q));
    graph.RemoveNode(qkv_nodes->q->Index());
    graph_utils::RemoveNodeOutputEdges(graph, const_cast<Node&>(*qkv_nodes->k));
    graph.RemoveNode(qkv_nodes->k->Index());
    graph_utils::RemoveNodeOutputEdges(graph, const_cast<Node&>(*qkv_nodes->v));
    graph.RemoveNode(qkv_nodes->v->Index());
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.Index());

    Node& fused_node = graph.AddNode(graph.GenerateNodeName("MatMulNBitsQkv"),
                                     "MatMulNBitsQkv",
                                     "fused SimplifiedLayerNormalization with Q/K/V MatMulNBits projections",
                                     fused_inputs,
                                     fused_outputs,
                                     &attrs,
                                     kMSDomain);
    fused_node.SetExecutionProviderType(kWebGpuExecutionProvider);

    LOGS(logger, VERBOSE) << "MatMulNBitsQkvFusion: created fused node '" << fused_node.Name()
                          << "' from norm='" << norm_name << "' q='" << q_name
                          << "' k='" << k_name << "' v='" << v_name << "'";

    for (const auto& input_edge : norm_input_edges) {
      int fused_input_index = input_edge.dst_arg_index;
      if (!is_skip_sln && input_edge.dst_arg_index == 1) {
        fused_input_index = 2;
      }

      graph.AddEdge(input_edge.src_node, fused_node.Index(), input_edge.src_arg_index, fused_input_index);
    }

    for (const auto& output_edge : q_output_edges) {
      graph.AddEdge(fused_node.Index(), output_edge.dst_node, 0, output_edge.dst_arg_index);
    }
    for (const auto& output_edge : k_output_edges) {
      graph.AddEdge(fused_node.Index(), output_edge.dst_node, 1, output_edge.dst_arg_index);
    }
    for (const auto& output_edge : v_output_edges) {
      graph.AddEdge(fused_node.Index(), output_edge.dst_node, 2, output_edge.dst_arg_index);
    }
    if (has_residual_output) {
      for (const auto& output_edge : norm_output_edges) {
        if (output_edge.src_arg_index == 3) {
          graph.AddEdge(fused_node.Index(), output_edge.dst_node, 3, output_edge.dst_arg_index);
        }
      }
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
