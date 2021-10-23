// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
bool CanSafelyRemoveNode(const Node& node_to_remove, const std::unordered_set<const Node*>& removal_set) {
  bool safe = true;
  for (auto iter = node_to_remove.OutputEdgesBegin(), end = node_to_remove.OutputEdgesEnd(); iter != end; ++iter) {
    if (removal_set.find(&iter->GetNode()) == removal_set.cend()) {
      safe = false;
      break;
    }
  }

  return safe;
}

// remove nodes if it is 'safe' to do so according to the checks in CanSafelyRemoveNode.
void SafelyRemoveNodes(Graph& graph, const std::vector<Node*>& nodes_to_remove, const Node* ignore_target) {
  std::unordered_set<const Node*> removal_set(nodes_to_remove.cbegin(), nodes_to_remove.cend());

  for (Node* node : nodes_to_remove) {
    if (node && node != ignore_target && CanSafelyRemoveNode(*node, removal_set)) {
      // TODO: It's slightly insane we don't support optionally removing the output edges as part of Graph::RemoveNode
      // but to make that change we need to validate a lot of existing code
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
    }
  }
}
}  // namespace

Status RemoveNodes::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  Node* ignore_target = preserve_target_node_ ? &selected_nodes.Target() : nullptr;
  SafelyRemoveNodes(graph, selected_nodes.AllNodes(), ignore_target);

  return Status::OK();
}

Status MergeIntoTarget::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  ORT_RETURN_IF_ERROR(MoveInputOutput(graph, selected_nodes, selected_nodes.Target(), value_moves_, false));

  return node_remover_.Run(graph, selected_nodes);
}

ReplaceWithNew::ReplaceWithNew(const std::string& domain,
                               const std::string& op_name,
                               std::vector<NodeAndMoveInfo>&& value_moves)
    : domain_{domain}, op_{op_name}, value_moves_{std ::move(value_moves)} {
}

Status ReplaceWithNew::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  const auto& target = selected_nodes.Target();

  const std::string op_type = OpType(selected_nodes);

  // create node. we'll populate the input and output defs via moves
  auto& replacement = graph.AddNode(target.Name(),
                                    op_type,
                                    target.Description(),
                                    {},  // input defs
                                    {},  // output defs
                                    &target.GetAttributes(),
                                    domain_);

  replacement.SetExecutionProviderType(kCpuExecutionProvider);

  ORT_RETURN_IF_ERROR(MoveInputOutput(graph, selected_nodes, replacement, value_moves_, false));
  return node_remover_.Run(graph, selected_nodes);
}

Status ReplaceWithNew::RunForSave(Graph& graph, const NodesToOptimize& selected_nodes,
                                  const RuntimeOptimizationSaveContext& save_context,
                                  SavedState& saved_state, bool& graph_modified) const {
  // make temporary node, use it to look up kernel def hash, remove temporary node
  const auto& target = selected_nodes.Target();
  const std::string op_type = OpType(selected_nodes);
  Node& new_node = graph.AddNode(target.Name(),
                                 op_type,
                                 target.Description(),
                                 {},
                                 {},
                                 &target.GetAttributes(),
                                 domain_);

  ORT_RETURN_IF_ERROR(MoveInputOutput(graph, selected_nodes, new_node, value_moves_, true));
  new_node.SetExecutionProviderType(kCpuExecutionProvider);
  ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(new_node), "Failed to set node op schema.");

  const KernelCreateInfo* kernel_create_info{};
  ORT_RETURN_IF_ERROR(save_context.kernel_registry_manager.get().SearchKernelRegistry(new_node, &kernel_create_info));
  const auto new_node_kernel_def_hash = kernel_create_info->kernel_def->GetHash();
  saved_state.produced_node_kernel_def_hashes.push_back(new_node_kernel_def_hash);

  graph.RemoveNode(new_node.Index());

  graph_modified = true;
  return Status::OK();
}

}  // namespace onnxruntime
