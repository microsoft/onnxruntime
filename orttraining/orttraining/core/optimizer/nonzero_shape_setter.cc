// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "orttraining/core/optimizer/nonzero_shape_setter.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

Status NonZeroShapeSetter::Apply(Graph& /*graph*/,
                                 Node& node,
                                 RewriteRuleEffect& rule_effect,
                                 const logging::Logger& /*logger*/) const {
  // The output shape of the NonZero is [num_of_input_dims, dynamic_nonzero_element_counts].
  ONNX_NAMESPACE::TensorShapeProto result_shape;
  result_shape.add_dim()->set_dim_value(node.InputDefs()[0]->Shape()->dim_size());
  result_shape.add_dim()->set_dim_param(node.OutputDefs()[0]->Name() + "_nonzero_count");
  node.MutableOutputDefs()[0]->SetShape(result_shape);
  rule_effect = RewriteRuleEffect::kUpdatedCurrentNode;
  return Status::OK();
}

bool NonZeroShapeSetter::SatisfyCondition(const Graph& /*graph*/,
                                          const Node& node,
                                          const logging::Logger& /*logger*/) const {
  return node.InputDefs()[0]->Shape() != nullptr
         && node.InputDefs()[0]->Shape()->dim_size() > 0
         && node.OutputDefs()[0]->Shape() == nullptr;
}

}  // namespace onnxruntime
