// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/cast_elimination.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status CastElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  const auto* input_type = node.InputDefs()[0]->TypeAsProto();
  if (input_type == nullptr || !input_type->tensor_type().has_elem_type()) {
    return Status::OK();
  }

  // Check if we can immediateately remove a very common case (casting to the same type as input).
  if (optimizer_utils::IsAttributeWithExpectedValue(node, "to", static_cast<int64_t>(input_type->tensor_type().elem_type()))) {
    graph_utils::RemoveNode(graph, node);
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

    return Status::OK();
  }

  // If not, find the longest chain that repeats the pattern.
  Node* current = &node;
  Node* final_non_cast_node = &node;
  int matching_elem_type = input_type->tensor_type().elem_type();

  while (current->OpType() == "Cast") {
    const auto& to_attr = current->GetAttributes().at("to");

    auto it = current->OutputNodesBegin();
    if (it == current->OutputNodesEnd()) {
      break;
    }
    current = const_cast<Node*>(&(*it));

    // We found we are repeating the conversion.
    if (to_attr.i() == matching_elem_type) {
      final_non_cast_node = current;
    }
  }

  std::vector<Node*> to_remove;
  current = &node;
  while (current != final_non_cast_node && current->OpType() == "Cast") {
    to_remove.push_back(current);
    auto it = current->OutputNodesBegin();
    if (it == current->OutputNodesEnd())
      break;
    current = const_cast<Node*>(&*it);
  }

  // No repeating pattern was found.
  if (to_remove.empty()) {
    return Status::OK();
  }

  std::cout << "to remove size " << to_remove.size() << std::endl;

  for (Node* n : to_remove) {
    std::cout << "el is" << std::endl;
    std::cout << n->Name() << " " << n->Index() << std::endl;
  }

  std::cout << "Explicit args count" << to_remove[0]->MutableInputArgsCount() << std::endl;
  std::cout << "Explicit args count2 " << to_remove[0]->GetInputEdgesCount() << std::endl;

  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

  // First remove all outbound edges.
  for (Node* n : to_remove) {
    std::cout << n->Name() << std::endl;
    graph_utils::RemoveNodeOutputEdges(graph, *n);
  }

  NodeArg* last_node_output_def = to_remove.back()->MutableOutputDefs()[0];
  const std::string& last_node_output_tensor_name = last_node_output_def->Name();

  // Find the matching def slot, so we can wire the final node to the input of the first removeable node.
  int slot = -1;
  auto& inputs = final_non_cast_node->MutableInputDefs();
  for (int i = 0, n = static_cast<int>(inputs.size()); i < n; ++i) {
    if (inputs[i]->Name() == last_node_output_tensor_name) {
      slot = i;
      break;
    }
  }

  final_non_cast_node->MutableInputDefs()[slot] = to_remove[0]->MutableInputDefs()[0];

  graph_utils::MoveAllNodeInputEdges(graph, *to_remove[0], *final_non_cast_node);

  // Finally, remove the nodes itself.
  for (Node* n : to_remove) {
    graph.RemoveNode(n->Index());
  }

  return Status::OK();
}

bool CastElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (!graph_utils::CanRemoveNode(graph, node, logger)) {
    return false;
  }
  return true;
}

}  // namespace onnxruntime
