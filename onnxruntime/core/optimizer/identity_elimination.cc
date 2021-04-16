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
  Case to eliminate Identity node when 
  - the input nodearg has only one consumer, which is the Identity itself
  - the input def is not a graph output
  
  For examples: 

  OK to eliminate:
  
    Identity output is another node, and the Identity is the only consumer of X
      X ---> Identity ---> Y where Y could be graph output

    Identity input arg is not shared with other output arg of X
      + (arg0) ---> Identity0 ---> Z 
      |
      X (arg1) ---> Identity1 ---> Y

  Not OK to eliminate:

    Identity input arg, i.e., arg0, is also an input arg of other Identity
      + (arg0) ---> Identity0 ---> Z 
      |
      X (arg0) ---> Identity1 ---> Y

    Identity input def, i.e., def0, is also a graph output
      + (def0) ---> Z where Z is graph output
      |
      X (def0/arg0) ---> Identity ---> Y
 */
Status EliminateIdentity::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  if (graph.GetNodeOutputsInGraphOutputs(node).empty()) {
    if (graph_utils::RemoveNode(graph, node)) {
      rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
    }
  } else {
    // keep a reference of output def to the graph output
    NodeArg* output = node.MutableOutputDefs()[0];
    const Node* p_input_node = graph_utils::GetInputNode(node, 0);
    // get mutable input node
    Node& input_node = *graph.GetNode(p_input_node->Index());
    int output_idx = graph_utils::GetNodeOutputIndexFromOutputName(input_node, node.MutableInputDefs()[0]->Name());
    // remove Identity node and its input edge
    graph.RemoveNode(node.Index());
    // update input node's output def to the graph output
    input_node.MutableOutputDefs()[output_idx] = output;
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode; 
  }
  return Status::OK();
}

bool EliminateIdentity::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (graph_utils::CanRemoveNode(graph, node, logger)) {
    return true;
  }

  // relax the condition if Identity is connecting to graph output
  if (node.GetOutputEdgesCount() != 0 || node.OutputDefs().size() != 1 ||
    graph.GetNodeOutputsInGraphOutputs(node).empty())
    return false;

  const Node* p_input_node = graph_utils::GetInputNode(node, 0);
  if (p_input_node == nullptr)
    return false;
    
  // find the edge between input node and this Identity node, and then get its src arg from input node
  int src_arg_index = -1;
  for (auto it = p_input_node->OutputEdgesBegin(), end = p_input_node->OutputEdgesEnd(); it != end; ++it) {
    if (it->GetNode().Index() == node.Index()) {
      src_arg_index = it->GetSrcArgIndex();
      break;
    }
  }

  // skip if the src arg is also a graph output
  if (graph.IsOutput(p_input_node->OutputDefs()[src_arg_index]))
    return false;

  // count how many consumers are sharing the same src arg
  int count = 0;
  for (auto it = p_input_node->OutputEdgesBegin(), end = p_input_node->OutputEdgesEnd(); it != end; ++it) {
    if (it->GetSrcArgIndex() == src_arg_index) {
      count++;
    }
  }
  // condition not met if there are more than 1 consumer for the same src arg
  if (count > 1) 
    return false;

  return true;
}

}  // namespace onnxruntime
