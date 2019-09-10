// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/relu_clip_fusion.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/op.h"

namespace onnxruntime {

Status FuseReluClip::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) const {
  // get the following Clip node before we delete the Relu node
  const auto& next_node = *node.OutputNodesBegin();

  if (graph_utils::RemoveNode(graph, node)) {
    // update the following Clip node if the 'min' is < 0.f to set it to 0.f
    // this essentially fuses the Relu and Clip
    // if the Clip 'min' is >= 0.f no change is required as Relu would have set the min to 0.f
    if (graph_utils::GetNodeAttribute(next_node, "min")->f() < 0.f) {
      auto* mutable_next_node = graph.GetNode(next_node.Index());
      mutable_next_node->ClearAttribute("min");
      mutable_next_node->AddAttribute("min", 0.f);
    }

    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool FuseReluClip::SatisfyCondition(const Graph& graph, const Node& node) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6})) {
    return false;
  }

  if (!graph_utils::IsSingleInSingleOutNode(node) ||
      graph.IsNodeOutputsInGraphOutputs(node)) {
    return false;
  }

  // If the Relu is followed by a Clip node the Relu is redundant and can be removed
  // as Clip will apply the minimum. If the Clip 'min' value is < 0 we need
  // to update it to 0 to apply what the Relu would have done. We do that in Apply.
  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Clip", {6}) ||
      next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
