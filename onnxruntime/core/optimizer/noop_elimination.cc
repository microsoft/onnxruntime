// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/noop_elimination.h"

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/op.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
  Eliminate no op node - supporting x+0, 0+x, x-0, x*1, 1*x and x/1 for now.
 */
Status NoopElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool NoopElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  bool input0_is_initializer = graph_utils::IsConstantInitializer(graph, node.InputDefs()[0]->Name());
  bool input1_is_initializer = graph_utils::IsConstantInitializer(graph, node.InputDefs()[1]->Name());

  // reject if both or neither inputs are initializers for now
  if (input0_is_initializer == input1_is_initializer) {
    return false;
  }

  const auto& op_type = node.OpType();
  if ((op_type == "Sub" || op_type == "Div") && !input1_is_initializer) {
    return false;
  }

  const auto* initializer =
      graph_utils::GetConstantInitializer(graph, node.InputDefs()[input0_is_initializer ? 0 : 1]->Name());

  // if initializer_rank is bigger, the output is expected to be initializer_rank per broadcasting rule,
  // but it won't happen if the case is accepted, thus reject it
  const auto& dims = initializer->dims();
  auto initializer_rank = dims.size();
  const auto* other_input_shape = node.InputDefs()[input0_is_initializer ? 1 : 0]->Shape();
  if (other_input_shape == nullptr || initializer_rank > other_input_shape->dim_size()) {
    return false;
  }

  int64_t tensor_size = 1;
  for (auto i : dims) {
    tensor_size *= i;
  }

  if (tensor_size > 1) {
    return false;
  }

  // handle edge case where the total size of the initializer is 0
  if (tensor_size == 0) {
    return true;
  }

  if (op_type == "Add" ||
      op_type == "Sub" ||
      op_type == "Mul" ||
      op_type == "Div") {
    int32_t data_type = initializer->data_type();
    Initializer add_init(*initializer, graph.ModelPath());

    float value = 0.0f;
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        value = *add_init.data<float>();
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        value = math::halfToFloat(add_init.data<MLFloat16>()->val);
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
        value = static_cast<float>(*add_init.data<double>());
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        value = static_cast<float>(*add_init.data<int32_t>());
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        value = static_cast<float>(*add_init.data<int64_t>());
        break;
      default:
        return false;
    }

    if (value != 0.0f && (op_type == "Add" || op_type == "Sub")) {
      return false;
    }

    if (value != 1.0f && (op_type == "Mul" || op_type == "Div")) {
      return false;
    }
  }

  // reject node output is graph output for now
  return graph_utils::CanRemoveNode(graph, node, logger);
}

}  // namespace onnxruntime
