// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/not_where_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

/**
Transform that fuses two Not -> Where nodes to a single Where node
with the where inputs 1 and 2 flipped.
Condition ->  Not -> Where ->
              value0-|  |
              value1----|

Condition -> Where ->
      value1-|  |
      value0----|
 */
bool NotWhereFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Where", {9})) {
    return false;
  }

  const Node* p_not_node = graph_utils::GetInputNode(node, 0);
  if (p_not_node == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*p_not_node, "Not", {1}) ||
      p_not_node->GetOutputEdgesCount() != 1 ||
      // Make sure the two nodes do not span execution providers.
      p_not_node->GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  if (!graph_utils::CanRemoveNode(graph, *p_not_node, logger)) {
    return false;
  }

  return true;
}

Status NotWhereFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  const Node* p_not_node = graph_utils::GetInputNode(node, 0);
  auto& where_node = node;
  auto& not_node = *graph.GetNode(p_not_node->Index());  // get mutable next node
  NodeArg* not_input = not_node.MutableInputDefs()[0];
  std::vector<NodeArg*> where_inputs = where_node.MutableInputDefs();

  // remove output egdes of not_node
  graph_utils::RemoveNodeOutputEdges(graph, not_node);

  graph_utils::ReplaceNodeInput(where_node, 0, *not_input);
  graph_utils::ReplaceNodeInput(where_node, 1, *where_inputs[2]);
  graph_utils::ReplaceNodeInput(where_node, 2, *where_inputs[1]);

  // Move input egdes from not_node to where_node, remove not_node
  graph_utils::FinalizeNodeFusion(graph, where_node, not_node, false);

  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}
}  // namespace onnxruntime
