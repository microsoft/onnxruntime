// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_util.h"

#include <vector>

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime::QDQ {

bool IsQDQPairSupported(
    const Node& q_node, const Node& dq_node,
    const GetConstantInitializerFn& get_const_initializer,
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
         q_zp.DataAsByteSpan() == dq_zp.DataAsByteSpan() &&
         *q_scale.data<float>() == *dq_scale.data<float>();
}

bool IsDQSupported(const Node& dq_node, const GetConstantInitializerFn& get_const_initializer) {
  bool zero_point_exists = false;
  if (!QOrDQNodeHasConstantScalarScaleAndZeroPoint(dq_node, get_const_initializer, zero_point_exists)) {
    return false;
  }

  if (!zero_point_exists) {
    return false;
  }

  return true;
}

bool QOrDQNodeHasConstantScalarScaleAndZeroPoint(
    const Node& q_or_dq_node,
    const GetConstantInitializerFn& get_const_initializer,
    bool& zero_point_exists) {
  auto q_or_dq_input_defs = q_or_dq_node.InputDefs();

  ORT_ENFORCE(q_or_dq_input_defs.size() >= 2);

  zero_point_exists = q_or_dq_input_defs.size() > 2 &&
                      q_or_dq_input_defs[InputIndex::ZERO_POINT_ID]->Exists();

  auto is_constant_scalar = [&](const NodeArg& node_arg) {
    return optimizer_utils::IsScalar(node_arg) && get_const_initializer(node_arg.Name()) != nullptr;
  };

  if (!is_constant_scalar(*q_or_dq_input_defs[InputIndex::SCALE_ID])) {
    return false;
  }

  if (zero_point_exists &&
      !is_constant_scalar(*q_or_dq_input_defs[InputIndex::ZERO_POINT_ID])) {
    return false;
  }

  return true;
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

bool MatchQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, QOpName, {10, 13});
}

bool MatchDQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, DQOpName, {10, 13});
}

#endif // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

}  // namespace onnxruntime::QDQ
