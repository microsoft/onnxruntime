// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/op.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/dropout_elimination.h"

namespace onnxruntime {

Status EliminateDropout::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool EliminateDropout::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  // We currently support elimination for Dropout operator v1, v6, v7, v10 and v12.
  // REVIEW(mzs): v10 implementation does not exist.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Dropout", {1, 6, 7, 10, 12})) {
    return false;
  }

  // A Dropout Node has one required output and an optional second output 'mask' at index == 1.
  // It can be safely removed if it has only one output that is used (checked by CanRemoveNode)
  // and that output is not the 'mask' output.
  // The 'is_test' attribute in v1 and v6 is captured by the check for the 'mask' output.
  return graph_utils::CanRemoveNode(graph, node, logger) && !graph_utils::IsOutputUsed(node, 1);
}

}  // namespace onnxruntime
