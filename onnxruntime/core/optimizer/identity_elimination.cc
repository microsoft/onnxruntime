// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/op.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/identity_elimination.h"

namespace onnxruntime {

Status EliminateIdentity::Apply(Graph& graph, Node& node, bool& modified, bool& deleted) {
  if (graph_utils::RemoveSingleInSingleOutNode(graph, node)) {
    modified = deleted = true;
  }

  return Status::OK();
}

bool EliminateIdentity::SatisfyCondition(const Graph& /*graph*/, const Node& node) {
  return OpTypeCondition(node) && graph_utils::IsSingleInSingleOutNode(node);
}

bool EliminateIdentity::OpTypeCondition(const Node& node) {
  return node.OpType() == included_op_type_;
}

}  // namespace onnxruntime
