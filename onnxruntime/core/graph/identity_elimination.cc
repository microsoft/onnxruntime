// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/identity_elimination.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status EliminateIdentity::Apply(Graph& graph, Node& node, bool& modified) {
  if (utils::RemoveNodeFromPath(graph, node)) {
    modified = true;
  }

  return Status::OK();
}

bool EliminateIdentity::SatisfyCondition(const Node& node) {
  return utils::IsSingleInSingleOutNode(node);
}

}  // namespace onnxruntime
