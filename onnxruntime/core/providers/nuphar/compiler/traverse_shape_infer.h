// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nuphar/common/analysis/shape_expr.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace nuphar {

// A collection of ShapeExpr
struct ShapeExprContext {
  std::map<std::string, ShapeExpr> inputs;
  std::map<const Node*, std::vector<ShapeExpr>> ops;
  std::map<std::string, std::pair<const Node*, size_t>> input_from;

  const ShapeExpr* Lookup(const NodeArg* def) const {
    const std::string& def_name = def->Name();
    auto iter = inputs.find(def_name);
    if (iter != inputs.end())
      return &(iter->second);

    auto iter_out_index = input_from.find(def_name);

    // OK if shape inference is incomplete
    // This is for some per-node unit test where NodeArg does not even have shape ranks
    // We ignore the shape inference in ToCapacity computation in per-node unit tests
    if (iter_out_index == input_from.end())
      return nullptr;

    const Node* from_node = iter_out_index->second.first;
    size_t index = iter_out_index->second.second;
    auto iter_op = ops.find(from_node);
    ORT_ENFORCE(iter_op != ops.end());
    return &(iter_op->second[index]);
  }
};

// Traverse function traverses a GraphViewer,
// performs shape infernce,
// and builds ShapeExpr in ShapeExprContext
Status ShapeInference(const GraphViewer& graph,
                      ShapeExprContext& context);

}  // namespace nuphar
}  // namespace onnxruntime
