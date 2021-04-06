// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/relu_quantizelinear.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

bool ReluQuantFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& /*logger*/) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6, 13}) ||
      !optimizer_utils::CheckOutputEdges(graph, node, 1)) {
    return false;
  }

  // if Relu is followed by QuantizeLinear, it can be fused into QuantizeLinear potentially
  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "QuantizeLinear", {10, 13})) {
    return false;
  }

  return true;
}

Status ReluQuantFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  Node& q_node = *graph.GetNode(node.OutputNodesBegin()->Index());

  std::vector<NodeArg*>& q_input_defs = q_node.MutableInputDefs();

  constexpr size_t q_input_cnt_required = 3;
  if (q_input_defs.size() != q_input_cnt_required) {
    return Status::OK();
  }

  constexpr size_t zp_idx = 2;
  const ONNX_NAMESPACE::TensorProto* zp_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph, *q_input_defs[zp_idx]) ||
      !graph.GetInitializedTensor(q_input_defs[zp_idx]->Name(), zp_tensor_proto)) {
    return Status::OK();
  }

  using ONNX_TENSOR_ELEM_TYPE = ONNX_NAMESPACE::TensorProto::DataType;
  Initializer zero_point(*zp_tensor_proto, graph.ModelPath());
  if (zero_point.size() != 1 ||
      zero_point.data_type() == ONNX_TENSOR_ELEM_TYPE::TensorProto_DataType_INT8 && zero_point.data<int8_t>()[0] != -128 ||
      zero_point.data_type() == ONNX_TENSOR_ELEM_TYPE::TensorProto_DataType_UINT8 && zero_point.data<uint8_t>()[0] != 0) {
    return Status::OK();
  }

  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}
}  // namespace onnxruntime
