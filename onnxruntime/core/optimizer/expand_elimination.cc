// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/expand_elimination.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

Status ExpandElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool ExpandElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  // 1. Get input shape.
  const auto* input_shape = node.InputDefs()[0]->Shape();
  if (input_shape == nullptr) {
    return false;
  }

  std::vector<int64_t> dim_values;
  for (int i = 0; i < input_shape->dim_size(); i++) {
    auto dim = input_shape->dim(i);
    if (!utils::HasDimValue(dim)) {
      return false;
    }
    dim_values.push_back(dim.dim_value());
  }

  // 2. Get target shape if it's constant.
  const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
  if (tensor_proto == nullptr || tensor_proto->dims_size() != 1 || tensor_proto->dims(0) <= 0) {
    return false;
  }

  auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
  if (initializer->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    return false;
  }

  const int64_t* target_shapes = initializer->data<int64_t>();

  // Check the dimensions starting at the trailing dimension.
  int i = int(dim_values.size() - 1);
  int j = int(tensor_proto->dims(0) - 1);
  while (i >= 0 && j >= 0) {
    if (dim_values[i] != target_shapes[j] && target_shapes[j] > 1) {
      return false;
    }

    --i;
    --j;
  }

  // The Expand is useless only when target dimension size is not greater than input's.
  return j < 0;
}

}  // namespace onnxruntime
