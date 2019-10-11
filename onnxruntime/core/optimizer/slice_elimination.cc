// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/slice_elimination.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/op.h"

namespace onnxruntime {

Status EliminateSlice::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) const {
  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool EliminateSlice::SatisfyCondition(const Graph& graph, const Node& node) const {
  // We currently support elimination for Slice operator v1.
  // TODO Extend to support Slice operator v10, which includes "steps" and all attributes are now given as inputs.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Slice", {1, 11})) {
    return false;
  }

  if (!graph_utils::IsSingleInSingleOutNode(node) ||
      graph.IsNodeOutputsInGraphOutputs(node)) {
    return false;
  }

  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  if (!graph_utils::GetRepeatedNodeAttributeValues(node, "starts", starts) ||
      !graph_utils::GetRepeatedNodeAttributeValues(node, "ends", ends) ||
      starts.size() != ends.size()) {
    return false;
  }
  std::vector<int64_t> axes;
  if (!graph_utils::GetRepeatedNodeAttributeValues(node, "axes", axes)) {
    for (int i = 0; static_cast<size_t>(i) < starts.size(); ++i) {
      axes.push_back(i);
    }
  } else if (axes.size() != starts.size()) {
    return false;
  }

  // For now eliminate slice operators if starts=0 and ends=MAX_INT.
  // TODO: Take into account the input's shape to get a tighter bound for the ends.
  for (size_t i = 0; i < axes.size(); ++i) {
    if (starts[i] != 0 || ends[i] < INT64_MAX) {
      return false;
    }
  }

  return true;
}

}  // namespace onnxruntime
