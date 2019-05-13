// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/op.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/dropout_elimination.h"

namespace onnxruntime {

Status EliminateDropout::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) {
  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool EliminateDropout::SatisfyCondition(const Graph& graph, const Node& node) {
  if (graph.IsNodeOutputsInGraphOutputs(node)) {
    return false;
  }

  // Check that the mask output is not an input to downstream nodes.
  if (graph_utils::IsSingleInSingleOutNode(node)) {
    return true;
  } else {
    const std::string& maskName = node.OutputDefs()[1]->Name();
    return !graph_utils::IsOutputUsed(node, maskName);
  }
}

}  // namespace onnxruntime
