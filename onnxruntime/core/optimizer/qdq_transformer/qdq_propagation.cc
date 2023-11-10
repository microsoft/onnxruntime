// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_propagation.h"

#include <optional>

#include "core/graph/extended_graph_edge.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/utils.h"

using onnxruntime::graph_utils::ExtendedGraphEdge;

namespace onnxruntime {
namespace {
bool CanNodePropagate(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "MaxPool", {12}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Reshape", {5, 13, 14, 19}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Transpose", {1, 13}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Squeeze", {1, 11, 13}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, "Unsqueeze", {1, 11, 13});
}

// convert this: src_node -> dst_node
// to this:      src_node -> Q -> DQ -> dst_node
// assumptions:
// 1. insertion_edge is valid - node indexes refer to valid nodes, arg name refers to a valid NodeArg, and it
//    corresponds to an actual graph relationship
// 2. scale_initializer_nodearg and zp_initializer_nodearg_ptr (if not null) are constant initializers
Status InsertQDQPair(Graph& graph, const ExtendedGraphEdge& insertion_edge,
                     NodeArg& scale_initializer_nodearg, NodeArg* zp_initializer_nodearg_ptr,
                     const logging::Logger& logger) {
  auto* src_node = insertion_edge.GetMutableNodeAtEnd(graph, ExtendedGraphEdge::End::Source);
  auto* dst_node = insertion_edge.GetMutableNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);

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
  auto make_q_or_dq_inputs = [](NodeArg& data, NodeArg& scale, NodeArg* zero_point) {
    return zero_point ? std::vector<NodeArg*>{&data, &scale, zero_point}
                      : std::vector<NodeArg*>{&data, &scale};
  };

  auto& q_node = graph.AddNode(graph.GenerateNodeName(base_name + "_q"),
                               QDQ::QOpName,
                               "Inserted by QDQPropagationTransformer",
                               // inputs
                               make_q_or_dq_inputs(pre_q_nodearg, scale_initializer_nodearg,
                                                   zp_initializer_nodearg_ptr),
                               // outputs
                               {&q_to_dq_nodearg});

  ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(q_node), "Failed to set op schema for added Q node.");

  auto& dq_node = graph.AddNode(graph.GenerateNodeName(base_name + "_dq"),
                                QDQ::DQOpName,
                                "Inserted by QDQPropagationTransformer",
                                // inputs
                                make_q_or_dq_inputs(q_to_dq_nodearg, scale_initializer_nodearg,
                                                    zp_initializer_nodearg_ptr),
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

std::optional<ExtendedGraphEdge> GetPreviousEdge(const Graph& graph, const Node& node) {
  // for now we can just consider the first input (index 0)

  const auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(node);
  const auto input_edge_it = std::find_if(
      input_edges.begin(), input_edges.end(),
      [](const graph_utils::GraphEdge& edge) { return edge.dst_arg_index == 0; });

  if (input_edge_it == input_edges.end()) {
    // maybe edge from input
    return ExtendedGraphEdge::TryCreateFromInputOrInitializerToNode(graph, node, 0);
  }

  const auto& src_node = *graph.GetNode(input_edge_it->src_node);
  const auto src_node_output_edges =
      graph_utils::GraphEdge::GetNodeOutputEdges(src_node, input_edge_it->src_arg_index);
  if (!graph.IsOutput(src_node.OutputDefs()[input_edge_it->src_arg_index]) &&
      src_node_output_edges.size() == 1) {
    // single edge from previous node
    return ExtendedGraphEdge::CreateFromValidGraphEdge(*input_edge_it);
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
    return ExtendedGraphEdge::TryCreateFromNodeToOutput(graph, node, 0);
  }

  if (!graph.IsOutput(node.OutputDefs()[0]) && output_edges.size() == 1) {
    // single edge to next node
    return ExtendedGraphEdge::CreateFromValidGraphEdge(output_edges.front());
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

class GraphConstantInitializerGetter {
  const Graph& graph_;

 public:
  GraphConstantInitializerGetter(const Graph& graph) : graph_{graph} {}
  const ONNX_NAMESPACE::TensorProto* operator()(const std::string& initializer_name) const {
    return graph_utils::GetConstantInitializer(graph_, initializer_name);
  }
};

Status PropagateDQForward(Graph& graph, gsl::span<const NodeIndex> node_indices,
                          const InlinedHashSet<std::string_view>& compatible_eps,
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

    bool dq_zero_point_exists = false;
    if (!QDQ::QOrDQNodeHasConstantScalarScaleAndZeroPoint(dq_node, GraphConstantInitializerGetter{graph},
                                                          dq_zero_point_exists)) {
      continue;
    }

    auto& dq_scale = *dq_node.MutableInputDefs()[QDQ::InputIndex::SCALE_ID];
    auto* dq_zero_point = dq_zero_point_exists
                              ? dq_node.MutableInputDefs()[QDQ::InputIndex::ZERO_POINT_ID]
                              : nullptr;

    const auto edge_after_dq = GetNextEdge(graph, dq_node);
    if (!edge_after_dq) {
      continue;
    }

    for (auto curr_edge = GetNextPropagationEdge(graph, *edge_after_dq);
         curr_edge.has_value();
         curr_edge = GetNextPropagationEdge(graph, *curr_edge)) {
      if (const auto* dst_node = curr_edge->GetNodeAtEnd(graph, ExtendedGraphEdge::End::Destination);
          dst_node && QDQ::MatchQNode(*dst_node)) {
        break;
      }

      ORT_RETURN_IF_ERROR(InsertQDQPair(graph, *curr_edge, dq_scale, dq_zero_point, logger));
      modified = true;
    }
  }

  return Status::OK();
}

Status PropagateQBackward(Graph& graph, gsl::span<const NodeIndex> node_indices,
                          const InlinedHashSet<std::string_view>& compatible_eps,
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

    bool q_zero_point_exists = false;
    if (!QDQ::QOrDQNodeHasConstantScalarScaleAndZeroPoint(q_node, GraphConstantInitializerGetter{graph},
                                                          q_zero_point_exists)) {
      continue;
    }

    auto& q_scale = *q_node.MutableInputDefs()[QDQ::InputIndex::SCALE_ID];
    auto* q_zero_point = q_zero_point_exists
                             ? q_node.MutableInputDefs()[QDQ::InputIndex::ZERO_POINT_ID]
                             : nullptr;

    const auto edge_before_q = GetPreviousEdge(graph, q_node);
    if (!edge_before_q) {
      continue;
    }

    for (auto curr_edge = GetPreviousPropagationEdge(graph, *edge_before_q);
         curr_edge.has_value();
         curr_edge = GetPreviousPropagationEdge(graph, *curr_edge)) {
      if (auto* src_node = curr_edge->GetNodeAtEnd(graph, ExtendedGraphEdge::End::Source);
          src_node && QDQ::MatchDQNode(*src_node)) {
        break;
      }

      ORT_RETURN_IF_ERROR(InsertQDQPair(graph, *curr_edge, q_scale, q_zero_point, logger));
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

  return Status::OK();
}
}  // namespace onnxruntime
