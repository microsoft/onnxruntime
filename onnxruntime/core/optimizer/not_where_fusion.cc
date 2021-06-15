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

It also fuses when not node has multiple where consumer nodes:

Condition ->  Not -> Where ->
              |    v0-|  |
              |    v1----|
              |----> Where ->
                  v0-|  |
                  v1----|

Condition -> Where ->
      |   v1-|  |
      |   v0----|
      |----> Where ->
          v1-|  |
          v0----|
 */
bool NotWhereFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Where", {9})) {
    return false;
  }

  const Node* p_not_node = graph_utils::GetInputNode(node, 0);
  if (p_not_node == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*p_not_node, "Not", {1}) ||
      // Make sure the two nodes do not span execution providers.
      p_not_node->GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  if (p_not_node->GetOutputEdgesCount() > 1) {
    // all consumers of not must be where
    for (auto it = p_not_node->OutputNodesBegin(); it != p_not_node->OutputNodesEnd(); ++it) {
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(*it, "Where", {9})) {
        return false;
      }
    }
  }

  if (!graph_utils::CanRemoveNode(graph, *p_not_node, logger)) {
    return false;
  }

  return true;
}

Status NotWhereFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  const Node* p_not_node = graph_utils::GetInputNode(node, 0);

  auto& not_node = *graph.GetNode(p_not_node->Index());  // get mutable next node
  NodeArg* not_input = not_node.MutableInputDefs()[0];

  // get all node ids of consumer where nodes
  std::vector<NodeIndex> where_node_ids;
  for (auto it = p_not_node->OutputNodesBegin(); it != p_not_node->OutputNodesEnd(); ++it) {
    where_node_ids.push_back(it->Index());
  }

  // Move input egdes from not_node to all where_node
  const Node* not_input_node = graph_utils::GetInputNode(not_node, 0);
  if (not_input_node) {
    Node& replacement = *graph.GetNode(not_input_node->Index());
    int replacement_output_idx = graph_utils::GetNodeOutputIndexFromOutputName(replacement, not_input->Name());
    // Replace inputs of all downstream where nodes with input of not_node by
    // removing not's output edges, updating input names of not's consumers, 
    // and adding the edges from not's input to not's consumers.
    graph_utils::ReplaceDownstreamNodeInput(graph, not_node, 0, replacement, replacement_output_idx);
  } else { // not's input is graph input/initializer. Remove the output egdes for not_node
    graph_utils::RemoveNodeOutputEdges(graph, not_node);
  }

  for (auto it = where_node_ids.begin(); it != where_node_ids.end(); ++it) {
    auto& where_node = *graph.GetNode(*it);

    std::vector<NodeArg*> where_inputs = where_node.MutableInputDefs();

    if (!not_input_node) { // not's input is graph input/initializer.
      graph_utils::ReplaceNodeInput(where_node, 0, *not_input);
    }

    const Node* where_input1_node = graph_utils::GetInputNode(where_node, 1);
    const Node* where_input2_node = graph_utils::GetInputNode(where_node, 2);

    int output1_idx = -1, output2_idx = -1;
    if (where_input1_node) {
      output1_idx = graph_utils::GetNodeOutputIndexFromOutputName(*where_input1_node, where_inputs[1]->Name());
      graph.RemoveEdge(where_input1_node->Index(), where_node.Index(), output1_idx, 1);
    }

    if (where_input2_node) {
      output2_idx = graph_utils::GetNodeOutputIndexFromOutputName(*where_input2_node, where_inputs[2]->Name());
      graph.RemoveEdge(where_input2_node->Index(), where_node.Index(), output2_idx, 2);
    }

    graph_utils::ReplaceNodeInput(where_node, 1, *where_inputs[2]);
    graph_utils::ReplaceNodeInput(where_node, 2, *where_inputs[1]);

    if (where_input1_node) {
      graph.AddEdge(where_input1_node->Index(), where_node.Index(), output1_idx, 2);
    }

    if (where_input2_node) {
      graph.AddEdge(where_input2_node->Index(), where_node.Index(), output2_idx, 1);
    } 
  }

  // remove not_node
  graph.RemoveNode(not_node.Index());

  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}
}  // namespace onnxruntime
