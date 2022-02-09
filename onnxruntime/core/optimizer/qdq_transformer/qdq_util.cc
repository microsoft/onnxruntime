// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_util.h"

#include <vector>

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime::QDQ {

bool IsQDQPairSupported(
    const Node& q_node, const Node& dq_node,
    const std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)>& get_const_initializer,
    const Path& model_path) {
  ConstPointerContainer<std::vector<NodeArg*>> dq_input_defs = dq_node.InputDefs();
  ConstPointerContainer<std::vector<NodeArg*>> q_input_defs = q_node.InputDefs();

  // Q/DQ contains optional input is not supported
  // non-scalar Q/DQ scale and zero point needs are not supported
  if (dq_input_defs.size() != InputIndex::TOTAL_COUNT ||
      q_input_defs.size() != InputIndex::TOTAL_COUNT ||
      !optimizer_utils::IsScalar(*q_input_defs[InputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*q_input_defs[InputIndex::ZERO_POINT_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[InputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[InputIndex::ZERO_POINT_ID])) {
    return false;
  }

  // if Q/DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto =
      get_const_initializer(dq_input_defs[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto =
      get_const_initializer(q_input_defs[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto =
      get_const_initializer(dq_input_defs[InputIndex::ZERO_POINT_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto =
      get_const_initializer(q_input_defs[InputIndex::ZERO_POINT_ID]->Name());
  if (nullptr == q_zp_tensor_proto ||
      nullptr == dq_zp_tensor_proto ||
      nullptr == q_scale_tensor_proto ||
      nullptr == dq_scale_tensor_proto) {
    return false;
  }

  // check Q/DQ have same scale and zero point
  Initializer q_zp(*q_zp_tensor_proto, model_path);
  Initializer q_scale(*q_scale_tensor_proto, model_path);
  Initializer dq_zp(*dq_zp_tensor_proto, model_path);
  Initializer dq_scale(*dq_scale_tensor_proto, model_path);

  return q_zp.data_type() == dq_zp.data_type() &&
         *q_zp.data<int8_t>() == *dq_zp.data<int8_t>() &&
         *q_scale.data<float>() == *dq_scale.data<float>();
}

bool DoesQOrDQNodeHaveConstantScalarScaleAndZeroPoint(
    const Node& q_or_dq_node,
    const std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)>& get_const_initializer) {
  ConstPointerContainer<std::vector<NodeArg*>> q_or_dq_input_defs = q_or_dq_node.InputDefs();

  // if Q or DQ contains optional input, return false
  // if Q or DQ scale and zero point are non-scalar, return false
  if (q_or_dq_input_defs.size() != InputIndex::TOTAL_COUNT ||
      !optimizer_utils::IsScalar(*q_or_dq_input_defs[InputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*q_or_dq_input_defs[InputIndex::ZERO_POINT_ID])) {
    return false;
  }

  // if Q or DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* q_or_dq_scale_tensor_proto =
      get_const_initializer(q_or_dq_input_defs[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_or_dq_zp_tensor_proto =
      get_const_initializer(q_or_dq_input_defs[InputIndex::ZERO_POINT_ID]->Name());
  if (nullptr == q_or_dq_zp_tensor_proto ||
      nullptr == q_or_dq_scale_tensor_proto) {
    return false;
  }

  return true;
}

#if !defined(ORT_MINIMAL_BUILD)

bool MatchQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, QOpName, {10, 13});
}

bool MatchDQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, DQOpName, {10, 13});
}

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace onnxruntime::QDQ
