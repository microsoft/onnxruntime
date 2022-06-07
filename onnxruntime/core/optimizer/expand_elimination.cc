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

bool ExpandElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (!graph_utils::CanRemoveNode(graph, node, logger)) {
    return false;
  }

  // 1. Check if has input shape.
  const auto* input_shape = node.InputDefs()[0]->Shape();
  if (input_shape == nullptr) {
    return false;
  }

  // 2. Get target shape if it's constant.
  const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
  if (tensor_proto == nullptr || tensor_proto->dims_size() != 1 || tensor_proto->dims(0) <= 0) {
    return false;
  }

  auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
  if (initializer->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    return false;
  }

  const int64_t* target_shapes = initializer->data<int64_t>();

  // Check the dimensions starting at the trailing dimension.
  int i = input_shape->dim_size() - 1;
  int j = static_cast<int>(tensor_proto->dims(0) - 1);

  // The Expand produces same input tensor only when target dimension size is not greater than input's.
  if (i < j) {
    return false;
  }

  while (i >= 0 && j >= 0) {
    auto dim = input_shape->dim(i);
    if (utils::HasDimValue(dim)) {
      auto dim_value = dim.dim_value();
      if (dim_value != target_shapes[j] && target_shapes[j] > 1) {
        return false;
      }
    } else if (target_shapes[j] > 1) {
      return false;
    }

    --i;
    --j;
  }

  
  return true;
}

}  // namespace onnxruntime
