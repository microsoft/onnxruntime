// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/op.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/dropout_elimination.h"

namespace onnxruntime {

Status EliminateDropout::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) const {
  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool EliminateDropout::SatisfyCondition(const Graph& graph, const Node& node) const {
  // We currently support elimination for Dropout operator v1, v6, v7, and v10.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Dropout", {1, 6, 7, 10})) {
    return false;
  }

  if (graph.IsNodeOutputsInGraphOutputs(node)) {
    return false;
  }

  // A Dropout Node has one required output and an optional second output (i.e. at index = 1), `mask`.
  // It can be safely removed if a) it has only one output
  // or b) if the `mask` output is present but is not an input to any downstream Nodes.
  // The `is_test` attribute in v1 and v6 is captured by the check for the `mask` output.
  if (graph_utils::IsSingleInSingleOutNode(node)) {
    return true;
  } else {
    // The `mask` output, which must not be used downstream for node removal, is at outputs[1].
    return !graph_utils::IsOutputUsed(node, 1);
  }
}

}  // namespace onnxruntime
