// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <tuple>

#include "core/optimizer/selectors_actions/actions.h"

#include "core/framework/op_kernel.h"
#include "core/optimizer/selectors_actions/helpers.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {

// Check if a node involved in an optimization can be safely removed due to it only having outputs consumed by nodes
// in the removal_set. If it has an output edge to a node outside of that set it must remain.
// As we can't easily remove a NodeArg from the Node::OutputDefs for the node being removed, we do not check if the
// node provides graph outputs here. The optimizer must correctly handle nodes producing graph outputs
// and not attempt to delete one of those nodes unless it has created a new source for the graph output.
bool CanSafelyRemoveNode(const Node& node_to_remove, const InlinedHashSet<const Node*>& removal_set) {
  bool safe = true;
  for (auto iter = node_to_remove.OutputEdgesBegin(), end = node_to_remove.OutputEdgesEnd(); iter != end; ++iter) {
    if (removal_set.find(&iter->GetNode()) == removal_set.cend()) {
      safe = false;
      break;
    }
  }

  return safe;
}

bool IsReplacementConvex(gsl::span<Node* const> nodes_to_replace) {
  InlinedHashSet<const Node*> replacement_set;
  replacement_set.reserve(nodes_to_replace.size());
  for (const Node* node : nodes_to_replace) {
    if (node != nullptr) {
      replacement_set.insert(node);
    }
  }

  InlinedHashSet<const Node*> visited_external_nodes;
  InlinedVector<const Node*> pending_external_nodes;
  for (const Node* node : replacement_set) {
    for (auto edge = node->OutputEdgesBegin(); edge != node->OutputEdgesEnd(); ++edge) {
      const Node* destination = &edge->GetNode();
      if (replacement_set.find(destination) == replacement_set.end() &&
          visited_external_nodes.insert(destination).second) {
        pending_external_nodes.push_back(destination);
      }
    }
  }

  while (!pending_external_nodes.empty()) {
    const Node* node = pending_external_nodes.back();
    pending_external_nodes.pop_back();
    for (auto edge = node->OutputEdgesBegin(); edge != node->OutputEdgesEnd(); ++edge) {
      const Node* destination = &edge->GetNode();
      if (replacement_set.find(destination) != replacement_set.end()) {
        return false;
      }
      if (visited_external_nodes.insert(destination).second) {
        pending_external_nodes.push_back(destination);
      }
    }
  }

  return true;
}

void RemapBoundaryControlEdges(
    Graph& graph,
    gsl::span<Node* const> nodes_to_replace,
    Node& replacement) {
  InlinedHashSet<const Node*> replacement_set;
  replacement_set.reserve(nodes_to_replace.size());
  replacement_set.insert(nodes_to_replace.begin(), nodes_to_replace.end());

  InlinedVector<std::pair<NodeIndex, NodeIndex>> edges_to_remove;
  InlinedVector<std::pair<NodeIndex, NodeIndex>> edges_to_add;
  const auto append_unique = [](auto& edges, std::pair<NodeIndex, NodeIndex> edge) {
    if (std::find(edges.begin(), edges.end(), edge) == edges.end()) {
      edges.push_back(edge);
    }
  };

  for (const Node* node : nodes_to_replace) {
    if (node == nullptr || node == &replacement) {
      continue;
    }

    for (auto edge = node->InputEdgesBegin(); edge != node->InputEdgesEnd(); ++edge) {
      if (edge->IsControlEdge() && replacement_set.find(&edge->GetNode()) == replacement_set.end()) {
        append_unique(edges_to_remove, {edge->GetNode().Index(), node->Index()});
        append_unique(edges_to_add, {edge->GetNode().Index(), replacement.Index()});
      }
    }

    for (auto edge = node->OutputEdgesBegin(); edge != node->OutputEdgesEnd(); ++edge) {
      if (edge->IsControlEdge()) {
        append_unique(edges_to_remove, {node->Index(), edge->GetNode().Index()});
        if (replacement_set.find(&edge->GetNode()) == replacement_set.end()) {
          append_unique(edges_to_add, {replacement.Index(), edge->GetNode().Index()});
        }
      }
    }
  }

  for (const auto& [src_node_index, dst_node_index] : edges_to_remove) {
    graph.RemoveEdge(src_node_index, dst_node_index, INT_MAX, INT_MAX);
  }
  for (const auto& [src_node_index, dst_node_index] : edges_to_add) {
    ORT_ENFORCE(graph.AddControlEdge(src_node_index, dst_node_index),
                "Failed to remap control edge during node replacement.");
  }
}

// remove nodes if it is 'safe' to do so according to the checks in CanSafelyRemoveNode.
InlinedVector<NodeIndex> SafelyRemoveNodes(
    Graph& graph, gsl::span<Node* const> nodes_to_remove, const Node* ignore_target) {
  InlinedHashSet<const Node*> removal_set;
  removal_set.reserve(nodes_to_remove.size());
  removal_set.insert(nodes_to_remove.begin(), nodes_to_remove.end());

  InlinedVector<NodeIndex> removed_node_indices;
  for (Node* node : nodes_to_remove) {
    if (node && node != ignore_target && CanSafelyRemoveNode(*node, removal_set)) {
      // TODO: It's slightly insane we don't support optionally removing the output edges as part of Graph::RemoveNode
      // but to make that change we need to validate a lot of existing code
      const NodeIndex node_index = node->Index();
      InlinedVector<std::tuple<NodeIndex, int, int>> control_edges;
      for (auto edge = node->OutputEdgesBegin(); edge != node->OutputEdgesEnd(); ++edge) {
        if (edge->IsControlEdge()) {
          control_edges.emplace_back(edge->GetNode().Index(), edge->GetSrcArgIndex(), edge->GetDstArgIndex());
        }
      }
      for (const auto& [dst_node_index, src_arg_index, dst_arg_index] : control_edges) {
        graph.RemoveEdge(node_index, dst_node_index, src_arg_index, dst_arg_index);
      }
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      if (graph.RemoveNode(node_index)) {
        removed_node_indices.push_back(node_index);
      }
    }
  }
  return removed_node_indices;
}
}  // namespace

Status RemoveNodes::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  Node* ignore_target = preserve_target_node_ ? &selected_nodes.Target() : nullptr;
  const auto removed_node_indices =
      SafelyRemoveNodes(graph, selected_nodes.AllNodes(), ignore_target);
  graph.NotifyNodesRemoved(removed_node_indices);

  return Status::OK();
}

Status MergeIntoTarget::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  if (!IsReplacementConvex(selected_nodes.AllNodes())) {
    return Status::OK();
  }

  const RuntimeState runtime_state{graph, selected_nodes};
  ORT_RETURN_IF_ERROR(MoveInputOutput(graph, selected_nodes, selected_nodes.Target(), ValueMoves(runtime_state),
                                      /* only_update_dest_definitions */ false));
  RemapBoundaryControlEdges(graph, selected_nodes.AllNodes(), selected_nodes.Target());

  const auto removed_node_indices =
      SafelyRemoveNodes(graph, selected_nodes.AllNodes(), &selected_nodes.Target());
  graph.NotifyNodeReplacement(removed_node_indices, selected_nodes.Target().Index());
  return Status::OK();
}

// adds a replacement node to the graph
// if provided, `replacement_ptr` is set to the replacement node if successful
static Status CreateReplacementNode(Graph& graph,
                                    const NodesToOptimize& selected_nodes,
                                    std::string op_type,
                                    std::string domain,
                                    NodeAttributes extra_attributes,
                                    std::vector<NodeAndMoveInfo> value_moves,
                                    bool only_update_dest_definitions,
                                    Node** replacement_ptr) {
  const auto& target = selected_nodes.Target();

  auto replacement_attributes = target.GetAttributes();
  for (auto& [name, value] : extra_attributes) {
    replacement_attributes.insert_or_assign(name, std::move(value));
  }

  // create node. we'll populate the input and output defs via moves
  auto& replacement = graph.AddNode(target.Name(),
                                    op_type,
                                    target.Description(),
                                    {},  // input defs
                                    {},  // output defs
                                    &replacement_attributes,
                                    domain);

  // If the target hasn't been partitioned yet (empty EP), leave the replacement's EP empty too
  // so a later partitioning pass can place it freely.
  const auto& target_provider = target.GetExecutionProviderType();
  if (!target_provider.empty()) {
    replacement.SetExecutionProviderType(target_provider);
  }

  ORT_RETURN_IF_ERROR(MoveInputOutput(graph, selected_nodes, replacement, value_moves, only_update_dest_definitions));

  if (replacement_ptr) {
    *replacement_ptr = &replacement;
  }

  return Status::OK();
}

Status ReplaceWithNew::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  if (!IsReplacementConvex(selected_nodes.AllNodes())) {
    return Status::OK();
  }

  const RuntimeState runtime_state{graph, selected_nodes};
  Node* replacement{};
  ORT_RETURN_IF_ERROR(CreateReplacementNode(graph, selected_nodes,
                                            OpType(runtime_state),
                                            Domain(runtime_state),
                                            ExtraAttributes(runtime_state),
                                            ValueMoves(runtime_state),
                                            /* only_update_dest_definitions */ false, &replacement));
  ORT_RETURN_IF_ERROR(ProcessNewNode(graph, selected_nodes, *replacement));
  RemapBoundaryControlEdges(graph, selected_nodes.AllNodes(), *replacement);
  const auto removed_node_indices =
      SafelyRemoveNodes(graph, selected_nodes.AllNodes(), nullptr);
  graph.NotifyNodeReplacement(removed_node_indices, replacement->Index());
  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD)
Status ReplaceWithNew::RunForSave(Graph& graph, const NodesToOptimize& selected_nodes,
                                  const SatRuntimeOptimizationSaveContext& /*save_context*/,
                                  SavedState& saved_state, bool& graph_modified) const {
  // make temporary node, save its op schema, remove temporary node
  const RuntimeState runtime_state{graph, selected_nodes};
  Node* replacement{};
  ORT_RETURN_IF_ERROR(CreateReplacementNode(graph, selected_nodes,
                                            OpType(runtime_state),
                                            Domain(runtime_state),
                                            ExtraAttributes(runtime_state),
                                            ValueMoves(runtime_state),
                                            /* only_update_dest_definitions */ true, &replacement));

  ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(*replacement), "Failed to set node op schema.");
  saved_state.produced_node_op_schemas.push_back(replacement->Op());

  ORT_RETURN_IF_NOT(graph.RemoveNode(replacement->Index()), "Failed to remove node.");

  graph_modified = true;
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace onnxruntime
