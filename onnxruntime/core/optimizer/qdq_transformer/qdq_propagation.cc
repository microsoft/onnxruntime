// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_propagation.h"

#include <optional>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
namespace {
// like graph_utils::GraphEdge, but also allowing the source end to be an input and the destination end to be an output
struct ExtendedGraphEdge {
  enum class End { Source,
                   Destination };

  struct NodeInfo {
    NodeIndex node_idx;
    int arg_idx;
  };

  std::optional<NodeInfo> src;  // if empty, graph input
  std::optional<NodeInfo> dst;  // if empty, graph output
  std::string arg_name;

  bool HasGraphInputOrInitializer() const noexcept { return !src.has_value(); }
  bool HasGraphOutput() const noexcept { return !dst.has_value(); }

  const std::optional<NodeInfo>& GetNodeInfoAtEnd(End end) const {
    return end == End::Source ? src : dst;
  }

  Node* GetNodeAtEnd(Graph& graph, End end) const {
    if (const auto& node_info = GetNodeInfoAtEnd(end); node_info.has_value()) {
      Node* node = graph.GetNode(node_info->node_idx);
      ORT_ENFORCE(node != nullptr, "Invalid node index ", node_info->node_idx);
      return node;
    }
    return nullptr;
  }

  const Node* GetNodeAtEnd(const Graph& graph, End end) const {
    if (const auto& node_info = GetNodeInfoAtEnd(end); node_info.has_value()) {
      const Node* node = graph.GetNode(node_info->node_idx);
      ORT_ENFORCE(node != nullptr, "Invalid node index ", node_info->node_idx);
      return node;
    }
    return nullptr;
  }

  static ExtendedGraphEdge FromGraphEdge(const graph_utils::GraphEdge& graph_edge) {
    return ExtendedGraphEdge{
        NodeInfo{graph_edge.src_node, graph_edge.src_arg_index},
        NodeInfo{graph_edge.dst_node, graph_edge.dst_arg_index},
        graph_edge.arg_name};
  }

  static std::optional<ExtendedGraphEdge> FromInputOrInitializerToNode(
      const Graph& graph, const Node& node, int node_input_def_idx) {
    const auto node_inputs = node.InputDefs();
    ORT_ENFORCE(node_input_def_idx <= node_inputs.size());

    const auto* node_input = node_inputs[node_input_def_idx];
    if (!graph.IsInputsIncludingInitializers(node_input)) {
      return std::nullopt;
    }

    return ExtendedGraphEdge{
        std::nullopt,
        NodeInfo{node.Index(), node_input_def_idx},
        node_input->Name()};
  }

  static std::optional<ExtendedGraphEdge> FromNodeToOutput(
      const Graph& graph, const Node& node, int node_output_def_idx) {
    const auto node_outputs = node.OutputDefs();
    ORT_ENFORCE(node_output_def_idx <= node_outputs.size());

    const auto* node_output = node_outputs[node_output_def_idx];
    if (!graph.IsOutput(node_output)) {
      return std::nullopt;
    }

    return ExtendedGraphEdge{
        NodeInfo{node.Index(), node_output_def_idx},
        std::nullopt,
        node_output->Name()};
  }

  // there is also the case where the graph input is an output, but we don't care about that for this transformer
};

bool CanNodePropagate(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {12}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Reshape", {5, 13, 14}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Transpose", {1, 13});
}

// convert this: src_node -> dst_node
// to this:      src_node -> Q -> DQ -> dst_node
// assumptions:
// 1. insertion_edge is valid - node indexes refer to valid nodes, arg name refers to a valid NodeArg, and it
//    corresponds to an actual graph relationship
// 2. scale_initializer_nodearg and zp_initializer_nodearg are constant initializers
Status InsertQDQPair(Graph& graph, const ExtendedGraphEdge& insertion_edge,
                     NodeArg& scale_initializer_nodearg,
                     NodeArg& zp_initializer_nodearg, const logging::Logger& logger) {
  auto* src_node = insertion_edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Source);
  auto* dst_node = insertion_edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);

  ORT_ENFORCE(src_node || dst_node, "At least one graph node must be specified in the propagation edge.");

  const auto& base_name = insertion_edge.arg_name;
  auto& base_node_arg = *graph.GetNodeArg(base_name);

  LOGS(logger, VERBOSE) << "Inserting Q/DQ pair between "
                        << (src_node ? MakeString("node (\"", src_node->Name(), "\", index: ", src_node->Index(), ")")
                                     : "input")
                        << " and "
                        << (dst_node ? MakeString("node (\"", dst_node->Name(), "\", index: ", dst_node->Index(), ")")
                                     : "output")
                        << " at NodeArg \"" << base_name << "\".";

  // set up new NodeArgs
  auto& pre_q_nodearg = insertion_edge.HasGraphInputOrInitializer()
                            ? base_node_arg
                            : graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(base_name + "_pre_q"),
                                                       nullptr);

  auto& q_to_dq_nodearg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(base_name + "_q_to_dq"),
                                                   nullptr);

  auto& post_dq_nodearg = insertion_edge.HasGraphOutput()
                              ? base_node_arg
                              : graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(base_name + "_post_dq"),
                                                         nullptr);

  // set up new Nodes
  auto& q_node = graph.AddNode(graph.GenerateNodeName(base_name + "_q"),
                               QDQ::QOpName,
                               "Inserted by QDQPropagationTransformer",
                               // inputs
                               {&pre_q_nodearg,
                                &scale_initializer_nodearg,
                                &zp_initializer_nodearg},
                               // outputs
                               {&q_to_dq_nodearg});

  ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(q_node), "Failed to set op schema for added Q node.");

  auto& dq_node = graph.AddNode(graph.GenerateNodeName(base_name + "_dq"),
                                QDQ::DQOpName,
                                "Inserted by QDQPropagationTransformer",
                                // inputs
                                {&q_to_dq_nodearg, &scale_initializer_nodearg, &zp_initializer_nodearg},
                                // outputs
                                {&post_dq_nodearg});

  ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(dq_node), "Failed to set op schema for added DQ node.");

  // set up edges
  if (src_node && dst_node) {
    graph.RemoveEdge(src_node->Index(), dst_node->Index(),
                     insertion_edge.src->arg_idx, insertion_edge.dst->arg_idx);
  }

  if (src_node) {
    src_node->MutableOutputDefs()[insertion_edge.src->arg_idx] = &pre_q_nodearg;
    graph.AddEdge(src_node->Index(), q_node.Index(), insertion_edge.src->arg_idx, 0);
  }

  graph.AddEdge(q_node.Index(), dq_node.Index(), 0, 0);

  if (dst_node) {
    dst_node->MutableInputDefs()[insertion_edge.dst->arg_idx] = &post_dq_nodearg;
    graph.AddEdge(dq_node.Index(), dst_node->Index(), 0, insertion_edge.dst->arg_idx);
  }

  return Status::OK();
}

bool IsMatchingQDQPair(const Graph& graph, const Node& q_node, const Node& dq_node) {
  return QDQ::MatchDQNode(dq_node) && QDQ::MatchQNode(q_node) &&
         QDQ::IsQDQPairSupported(
             q_node, dq_node,
             [&graph](const std::string& name) { return graph.GetConstantInitializer(name, true); },
             graph.ModelPath());
}

std::optional<ExtendedGraphEdge> GetPreviousEdge(const Graph& graph, const Node& node) {
  // for now we can just consider the first input (index 0)

  const auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(node);
  const auto input_edge_it = std::find_if(
      input_edges.begin(), input_edges.end(),
      [](const graph_utils::GraphEdge& edge) { return edge.dst_arg_index == 0; });

  if (input_edge_it == input_edges.end()) {
    // maybe edge from input
    return ExtendedGraphEdge::FromInputOrInitializerToNode(graph, node, 0);
  }

  const auto& src_node = *graph.GetNode(input_edge_it->src_node);
  const auto src_node_output_edges =
      graph_utils::GraphEdge::GetNodeOutputEdges(src_node, input_edge_it->src_arg_index);
  if (!graph.IsOutput(src_node.OutputDefs()[input_edge_it->src_arg_index]) &&
      src_node_output_edges.size() == 1) {
    // single edge from previous node
    return ExtendedGraphEdge::FromGraphEdge(*input_edge_it);
  }

  return std::nullopt;
}

std::optional<ExtendedGraphEdge> GetPreviousPropagationEdge(const Graph& graph,
                                                            const ExtendedGraphEdge& edge) {
  if (edge.HasGraphInputOrInitializer()) {
    return std::nullopt;
  }

  const auto* src_node = edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Source);
  ORT_ENFORCE(src_node != nullptr);

  if (!CanNodePropagate(*src_node)) {
    return std::nullopt;
  }

  return GetPreviousEdge(graph, *src_node);
}

std::optional<ExtendedGraphEdge> GetNextEdge(const Graph& graph, const Node& node) {
  // for now we can just consider the first output (index 0)

  const auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(node, 0);
  if (output_edges.empty()) {
    // maybe edge to output
    return ExtendedGraphEdge::FromNodeToOutput(graph, node, 0);
  }

  if (!graph.IsOutput(node.OutputDefs()[0]) && output_edges.size() == 1) {
    // single edge to next node
    return ExtendedGraphEdge::FromGraphEdge(output_edges.front());
  }

  return std::nullopt;
}

std::optional<ExtendedGraphEdge> GetNextPropagationEdge(const Graph& graph,
                                                        const ExtendedGraphEdge& edge) {
  if (edge.HasGraphOutput()) {
    return std::nullopt;
  }

  const auto* dst_node = edge.GetNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);
  ORT_ENFORCE(dst_node != nullptr);

  if (!CanNodePropagate(*dst_node)) {
    return std::nullopt;
  }

  return GetNextEdge(graph, *dst_node);
}

Status PropagateDQForward(Graph& graph, gsl::span<const NodeIndex> node_indices,
                          const std::unordered_set<std::string>& compatible_eps,
                          const logging::Logger& logger,
                          bool& modified) {
  for (auto node_index : node_indices) {
    auto* dq_node_ptr = graph.GetNode(node_index);
    if (dq_node_ptr == nullptr) {
      continue;  // node removed as part of an earlier fusion
    }

    Node& dq_node = *dq_node_ptr;

    if (!QDQ::MatchDQNode(dq_node) ||
        !graph_utils::IsSupportedProvider(dq_node, compatible_eps) ||
        !optimizer_utils::CheckOutputEdges(graph, dq_node, 1)) {
      continue;
    }

    std::vector<NodeArg*>& dq_input_defs = dq_node.MutableInputDefs();
    if (dq_input_defs.size() != QDQ::InputIndex::TOTAL_COUNT) {
      continue;
    }

    if (!optimizer_utils::IsScalar(*dq_input_defs[QDQ::InputIndex::ZERO_POINT_ID]) ||
        !optimizer_utils::IsScalar(*dq_input_defs[QDQ::InputIndex::SCALE_ID])) {
      continue;
    }

    const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto =
        graph_utils::GetConstantInitializer(graph, dq_input_defs[QDQ::InputIndex::ZERO_POINT_ID]->Name());
    const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto =
        graph_utils::GetConstantInitializer(graph, dq_input_defs[QDQ::InputIndex::SCALE_ID]->Name());

    if (nullptr == dq_zp_tensor_proto || nullptr == dq_scale_tensor_proto) {
      continue;
    }

    const auto edge_after_dq = GetNextEdge(graph, dq_node);
    if (!edge_after_dq) {
      continue;
    }

    for (auto curr_edge = GetNextPropagationEdge(graph, *edge_after_dq);
         curr_edge.has_value();
         curr_edge = GetNextPropagationEdge(graph, *curr_edge)) {
      if (const auto* dst_node = curr_edge->GetNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);
          dst_node && IsMatchingQDQPair(graph, *dst_node, dq_node)) {
        break;
      }

      ORT_RETURN_IF_ERROR(InsertQDQPair(graph, *curr_edge,
                                        *dq_input_defs[QDQ::InputIndex::SCALE_ID],
                                        *dq_input_defs[QDQ::InputIndex::ZERO_POINT_ID],
                                        logger));
      modified = true;
    }
  }

  return Status::OK();
}

Status PropagateQBackward(Graph& graph, gsl::span<const NodeIndex> node_indices,
                          const std::unordered_set<std::string>& compatible_eps,
                          const logging::Logger& logger,
                          bool& modified) {
  for (auto node_index : node_indices) {
    auto* q_node_ptr = graph.GetNode(node_index);
    if (q_node_ptr == nullptr) {
      continue;  // node removed as part of an earlier fusion
    }

    Node& q_node = *q_node_ptr;

    if (!QDQ::MatchQNode(q_node) ||
        !graph_utils::IsSupportedProvider(q_node, compatible_eps)) {
      continue;
    }

    std::vector<NodeArg*>& q_input_defs = q_node.MutableInputDefs();
    if (q_input_defs.size() != QDQ::InputIndex::TOTAL_COUNT ||
        !optimizer_utils::IsScalar(*q_input_defs[QDQ::InputIndex::ZERO_POINT_ID]) ||
        !optimizer_utils::IsScalar(*q_input_defs[QDQ::InputIndex::SCALE_ID])) {
      continue;
    }

    const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto =
        graph_utils::GetConstantInitializer(graph, q_input_defs[QDQ::InputIndex::ZERO_POINT_ID]->Name());
    const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto =
        graph_utils::GetConstantInitializer(graph, q_input_defs[QDQ::InputIndex::SCALE_ID]->Name());

    if (nullptr == q_zp_tensor_proto || nullptr == q_scale_tensor_proto) {
      continue;
    }

    const auto edge_before_q = GetPreviousEdge(graph, q_node);
    if (!edge_before_q) {
      continue;
    }

    for (auto curr_edge = GetPreviousPropagationEdge(graph, *edge_before_q);
         curr_edge.has_value();
         curr_edge = GetPreviousPropagationEdge(graph, *curr_edge)) {
      if (auto* src_node = curr_edge->GetNodeAtEnd(graph, ExtendedGraphEdge::End::Source);
          src_node && IsMatchingQDQPair(graph, q_node, *src_node)) {
        break;
      }

      ORT_RETURN_IF_ERROR(InsertQDQPair(graph, *curr_edge,
                                        *q_input_defs[QDQ::InputIndex::SCALE_ID],
                                        *q_input_defs[QDQ::InputIndex::ZERO_POINT_ID],
                                        logger));
      modified = true;
    }
  }

  return Status::OK();
}
}  // namespace

Status QDQPropagationTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                            const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto node_indices = gsl::make_span(graph_viewer.GetNodesInTopologicalOrder());

  for (auto node_index : node_indices) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // node removed as part of an earlier fusion

    ORT_RETURN_IF_ERROR(Recurse(*node_ptr, modified, graph_level, logger));
  }

  const auto& compatible_eps = GetCompatibleExecutionProviders();

  ORT_RETURN_IF_ERROR(PropagateQBackward(graph, node_indices, compatible_eps, logger, modified));
  ORT_RETURN_IF_ERROR(PropagateDQForward(graph, node_indices, compatible_eps, logger, modified));

  ORT_RETURN_IF_ERROR(QDQ::CancelOutRedundantDQQPairs(graph, node_indices, compatible_eps, logger, modified));

  return Status::OK();
}
}  // namespace onnxruntime
