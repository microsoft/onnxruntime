// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qdq_op_transformer.h"

#include <string>
#include <vector>

#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

bool QDQOperatorTransformer::Transform(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) {
  if (!Check(dq_nodes, q_nodes)) {
    return false;
  }

  FillQDQOptionalZeroPoint(dq_nodes);
  FillQDQOptionalZeroPoint(q_nodes);

  return TransformImpl(dq_nodes, q_nodes);
}

bool QDQOperatorTransformer::Check(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) const {
  if (node_.MutableInputDefs().size() != dq_nodes.size() ||
      node_.MutableOutputDefs().size() != q_nodes.size() ||
      graph_.GetNodeOutputsInGraphOutputs(node_).size() > 0) {
    return false;
  }

  return true;
}

void QDQOperatorTransformer::FillQDQOptionalZeroPoint(const std::vector<const Node*>& qdq_nodes) {
  for (const Node* p_node_const : qdq_nodes) {
    Node& node = *graph_.GetNode(p_node_const->Index());
    std::vector<NodeArg*>& input_defs = node.MutableInputDefs();
    constexpr size_t max_input_count = 3;
    if (input_defs.size() == max_input_count) {
      continue;  // zero point is not optional. No need to fill.
    }

    bool is_default_zp_signed = false;
    if (node.OpType() == DQOPTypeName) {
      auto input_type = input_defs[0]->TypeAsProto()->tensor_type().elem_type();
      is_default_zp_signed = ONNX_NAMESPACE::TensorProto_DataType_INT8 == input_type;
    }

    const ONNX_NAMESPACE::TensorProto& zp_tensor_proto = is_default_zp_signed ? optional_zero_point_int8_ : optional_zero_point_uint8_;

    const ONNX_NAMESPACE::TensorProto* dummy_zp_tensor_proto;
    if (!graph_.GetInitializedTensor(zp_tensor_proto.name(), dummy_zp_tensor_proto)) {
      graph_.AddInitializedTensor(zp_tensor_proto);
    }

    input_defs.push_back(&graph_.GetOrCreateNodeArg(zp_tensor_proto.name(), nullptr));
  }
}

const ONNX_NAMESPACE::TensorProto QDQOperatorTransformer::optional_zero_point_int8_ = []() {
  const char* const name = "855dd0fa-cd7b-4b10-ae5a-df64cabfe1f8";
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
  tensor_proto.set_raw_data(std::vector<int8_t>{0}.data(), sizeof(int8_t));

  return tensor_proto;
}();

const ONNX_NAMESPACE::TensorProto QDQOperatorTransformer::optional_zero_point_uint8_ = []() {
  const char* const name = "35b188f7-c464-43e3-8692-97ac832bb14a";
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  tensor_proto.set_raw_data(std::vector<int8_t>{0}.data(), sizeof(uint8_t));

  return tensor_proto;
}();

}  // namespace onnxruntime
