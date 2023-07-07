// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/split_replacement.h"

#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status SplitReplacement::Apply(Graph& graph, Node& split_node, RewriteRuleEffect& rule_effect,
                               const logging::Logger&) const {
  Node& split_view_node =
      graph.AddNode(graph.GenerateNodeName("SplitView"), "SplitView", "Split view.", split_node.MutableInputDefs(),
                    split_node.MutableOutputDefs(), &split_node.GetAttributes(), kMSDomain);
  // Assign provider to this new node. Provider should be same as the provider for old node.
  split_view_node.SetExecutionProviderType(split_node.GetExecutionProviderType());
  graph_utils::FinalizeNodeFusion(graph, split_view_node, split_node);
  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  return Status::OK();
}

bool SplitReplacement::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  int64_t axis = 0;
  const auto* attr_proto = graph_utils::GetNodeAttribute(node, "axis");
  if ((nullptr != attr_proto) && attr_proto->has_i()) {
    axis = attr_proto->i();
  }
  if (axis != 0) {
    return false;
  }

  for (const NodeArg* output : node.OutputDefs()) {
    if (graph.IsOutput(output)) {
      return false;
    }
  }

  return true;
}

}  // namespace onnxruntime
