// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/loss_func/loss_func_common.h"

namespace onnxruntime {
namespace training {

TypeProto* GetSparseTypeProto(const NodeArg* input_arg,
                              ONNX_NAMESPACE::TensorProto_DataType data_type,
                              GraphAugmenter::GraphDefs& graph_defs) {
  ORT_ENFORCE(input_arg != nullptr, "GetSparseTypeProto's input_arg is nullptr");
  const auto* logits_type_proto = input_arg->TypeAsProto();
  const auto& dims = logits_type_proto->tensor_type().shape().dim();

  TypeProto* type_proto = graph_defs.CreateTypeProto();
  type_proto->mutable_tensor_type()->set_elem_type(data_type);

  auto* target_shape = type_proto->mutable_tensor_type()->mutable_shape();
  for (int i = 0; i < dims.size() - 1; i++) {
    target_shape->add_dim()->CopyFrom(dims[i]);
  }

  return type_proto;
}

}  // namespace training
}  // namespace onnxruntime
