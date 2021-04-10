// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/op.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/identity_elimination.h"

namespace onnxruntime {

/**
  Special case to eliminate Identity node when its output is graph output 
  note that there is no output edge for Identity

  X ---> Identity ---> graph output

  becomes 

  X ---> graph output
 */
Status EliminateIdentity::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {

  // regular case - when Identity's output has no graph output 
  if (graph.GetNodeOutputsInGraphOutputs(node).empty()) {
    if (graph_utils::RemoveNode(graph, node)) {
      rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
    }
  } else {
    // handling special case

    // keep a reference of output def to the graph output
    NodeArg* output = node.MutableOutputDefs()[0];

    const Node* p_input_node = graph_utils::GetInputNode(node, 0);
    // get mutable input node
    Node& input_node = *graph.GetNode(p_input_node->Index());

    // remove Identity node and its input edge
    graph.RemoveNode(node.Index());
    // update input node's output def to the graph output
    input_node.MutableOutputDefs()[0] = output;

    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool EliminateIdentity::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (graph_utils::CanRemoveNode(graph, node, logger)) {
    return true;
  }
  
  // relax the condition if Identity is connecting to graph output
  if (node.GetOutputEdgesCount() == 0 && node.OutputDefs().size() == 1 &&
    !graph.GetNodeOutputsInGraphOutputs(node).empty()) {
    return true;
  }
  return false;
}

}  // namespace onnxruntime
