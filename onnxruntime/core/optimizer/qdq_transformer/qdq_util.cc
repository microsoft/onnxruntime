// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qdq_util.h"

#include <vector>

#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
namespace QDQ {

bool IsQDQPairSupported(const Graph& graph, const Node& q_node, const Node& dq_node) {
  ConstPointerContainer<std::vector<NodeArg*>> dq_input_defs = dq_node.InputDefs();
  ConstPointerContainer<std::vector<NodeArg*>> q_input_defs = q_node.InputDefs();

  // Q/DQ contains optional input is not supported
  // non-scalar Q/DQ scale and zero point needs are not supported
  if (dq_input_defs.size() != QDQInputIndex::TOTAL_COUNT ||
      q_input_defs.size() != QDQInputIndex::TOTAL_COUNT ||
      !optimizer_utils::IsScalar(*q_input_defs[QDQInputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*q_input_defs[QDQInputIndex::ZERO_POINT_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[QDQInputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[QDQInputIndex::ZERO_POINT_ID])) {
    return false;
  }

  // if Q/DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto =
      graph_utils::GetConstantInitializer(graph, dq_input_defs[QDQInputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto =
      graph_utils::GetConstantInitializer(graph, q_input_defs[QDQInputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto =
      graph_utils::GetConstantInitializer(graph, dq_input_defs[QDQInputIndex::ZERO_POINT_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto =
      graph_utils::GetConstantInitializer(graph, q_input_defs[QDQInputIndex::ZERO_POINT_ID]->Name());
  if (nullptr == q_zp_tensor_proto ||
      nullptr == dq_zp_tensor_proto ||
      nullptr == q_scale_tensor_proto ||
      nullptr == dq_scale_tensor_proto) {
    return false;
  }

  // check Q/DQ have same scale and zero point
  Initializer q_zp(*q_zp_tensor_proto, graph.ModelPath());
  Initializer q_scale(*q_scale_tensor_proto, graph.ModelPath());
  Initializer dq_zp(*dq_zp_tensor_proto, graph.ModelPath());
  Initializer dq_scale(*dq_scale_tensor_proto, graph.ModelPath());

  return *q_zp.data<int8_t>() == *dq_zp.data<int8_t>() &&
         *q_scale.data<float>() == *dq_scale.data<float>();
}

}  // namespace QDQ
}  // namespace onnxruntime
