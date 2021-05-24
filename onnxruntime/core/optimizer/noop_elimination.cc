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
  Eliminate no op node - handling Add op for now
  Add example: 

      X     0
       \   /   
        Add
         |
         Y    
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

  const auto* initializer = graph_utils::GetConstantInitializer(graph, node.InputDefs()[input0_is_initializer ? 0 : 1]->Name());
  auto initializer_rank = initializer->dims().size();
  auto other_input_rank = node.InputDefs()[input0_is_initializer ? 1 : 0]->Shape()->dim_size();
  // if initializer_rank is bigger, the output is expected to be initializer_rank per broadcasting rule,
  // but it won't happen if the case is accepted, thus reject it
  if (initializer_rank > other_input_rank) {
    return false;
  }

  int32_t data_type = initializer->data_type();
  Initializer add_init(*initializer, graph.ModelPath());
  if (add_init.size() > 1) {
    return false;
  }
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      if (*add_init.data<float>() != 0.f) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      if (math::halfToFloat(add_init.data<MLFloat16>()->val) != 0.f) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      if (*add_init.data<double>() != static_cast<double>(0.f)) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      if (*add_init.data<int32_t>() != static_cast<int32_t>(0)) {
        return false;
      }
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      if (*add_init.data<int64_t>() != static_cast<int64_t>(0)) {
        return false;
      }
      break;
    default:
      return false;
  }

  // reject node output is graph output for now
  if (!graph_utils::CanRemoveNode(graph, node, logger)) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
