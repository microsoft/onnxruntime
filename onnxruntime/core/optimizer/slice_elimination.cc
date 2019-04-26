// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/slice_elimination.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/op.h"

namespace onnxruntime {

Status EliminateSlice::Apply(Graph& graph, Node& node, bool& modified, bool& removed) {
  if (graph_utils::RemoveSingleInputNode(graph, node)) {
    removed = modified = true;
  }

  return Status::OK();
}

bool EliminateSlice::SatisfyCondition(const Graph& graph, const Node& node) {
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
    for (int i = 0; (size_t)i < starts.size(); ++i) {
      axes.push_back(i);
    }
  } else if (axes.size() != starts.size()) {
    return false;
  }

  auto in_size = node.InputDefs()[0]->Shape()->dim_size();
  auto out_size = node.OutputDefs()[0]->Shape()->dim_size();
  if (in_size != out_size) {
    return false;
  }
  auto dimin = node.InputDefs()[0]->Shape()->dim(0).dim_value();
  auto dimout = node.OutputDefs()[0]->Shape()->dim(0).dim_value();
  if (dimin != dimout) {
    return false;
  }

  // For now eliminate slice operators if starts=0 and ends=MAX_INT.
  // TODO: Take into account the input's shape to get a tighter bound for the ends.
  for (size_t i = 0; i < axes.size(); ++i) {
    if (starts[i] != 0 || ends[i] < INT64_MAX) {
      return false;
    }
  }

  // "steps" attribute is added since version 10. If it exists and is not 1s, slice is not redundant.
  if (graph_utils::MatchesOpSinceVersion(node, 10)) {
    std::vector<int64_t> steps;
    if (graph_utils::GetRepeatedNodeAttributeValues(node, "steps", starts)) {
      if (steps.size() != starts.size() ||
          std::any_of(steps.cbegin(), steps.cend(), [](int64_t step) { return step > 1; })) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace onnxruntime
