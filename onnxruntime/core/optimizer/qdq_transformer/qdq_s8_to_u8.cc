// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_s8_to_u8.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/s8_to_u8.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

/**
 * @brief Given a QuantizeLinear and DequantizeLinear pair with type int8_t,
 *        Convert them to uint8_t
 * @param graph
 * @param q_node
 * @param dq_node
 * @return whether conversion happened
 */
static bool QDQ_S8_to_U8(Graph& graph, Node& q_node, Node& dq_node) {
  auto& q_input_defs = q_node.MutableInputDefs();
  auto& dq_input_defs = dq_node.MutableInputDefs();

  constexpr size_t input_cnt_required = 3;
  if (q_input_defs.size() != input_cnt_required ||
      dq_input_defs.size() != input_cnt_required) {
    return false;
  }

  constexpr size_t zp_idx = 2;
  const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto = nullptr;
  const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph, *q_input_defs[zp_idx]) ||
      !graph_utils::NodeArgIsConstant(graph, *dq_input_defs[zp_idx]) ||
      !graph.GetInitializedTensor(q_input_defs[zp_idx]->Name(), q_zp_tensor_proto) ||
      !graph.GetInitializedTensor(dq_input_defs[zp_idx]->Name(), dq_zp_tensor_proto)) {
    return false;
  }

  // TODO(fuchen): need to augment this when we support per row quantization
  using ONNX_TENSOR_ELEM_TYPE = ONNX_NAMESPACE::TensorProto::DataType;
  Initializer q_zero_point(*q_zp_tensor_proto, graph.ModelPath());
  Initializer dq_zero_point(*dq_zp_tensor_proto, graph.ModelPath());
  if (q_zero_point.size() != 1 ||
      dq_zero_point.size() != 1 ||
      q_zero_point.data_type() != ONNX_TENSOR_ELEM_TYPE::TensorProto_DataType_INT8 ||
      dq_zero_point.data_type() != ONNX_TENSOR_ELEM_TYPE::TensorProto_DataType_INT8) {
    return false;
  }

  uint8_t q_zp_value = *q_zero_point.data<int8_t>() + 128;
  uint8_t dq_zp_value = *dq_zero_point.data<int8_t>() + 128;

  if (q_zp_value != dq_zp_value) {
    return false;  // zero points for Q and DQ are expected to be same
  }

  ONNX_NAMESPACE::TensorProto zp_tensor_proto_u8;
  zp_tensor_proto_u8.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  zp_tensor_proto_u8.set_name(graph.GenerateNodeArgName("qdq_s8_to_u8_zp_conversion"));
  zp_tensor_proto_u8.set_raw_data(&q_zp_value, sizeof(uint8_t));
  NodeArg* zp_u8_arg = &graph_utils::AddInitializer(graph, zp_tensor_proto_u8);

  auto q_output_node_arg_name = graph.GenerateNodeArgName("qdq_s8_to_u8_quant");
  NodeArg* q_output_arg = &graph.GetOrCreateNodeArg(q_output_node_arg_name, nullptr);

  q_node.MutableOutputDefs()[0] = q_output_arg;
  dq_input_defs[0] = q_output_arg;
  q_input_defs[zp_idx] = zp_u8_arg;
  dq_input_defs[zp_idx] = zp_u8_arg;
  return true;
}

// Convert QuantizeLinear and DequantizeLinear pair with type int8_t to type uint8_t
Status QDQS8ToU8Transformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                       const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* q_node_ptr = graph.GetNode(node_index);
    if (q_node_ptr == nullptr)
      continue;  // node removed as part of an earlier fusion

    Node& node = *q_node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // recognize Q + DQ pair
    if (QDQ::MatchQNode(node) &&
        optimizer_utils::CheckOutputEdges(graph, node, 1)) {
      Node& dq_node = *graph.GetNode(node.OutputNodesBegin()->Index());
      if (QDQ::MatchDQNode(dq_node)) {
        modified |= QDQ_S8_to_U8(graph, node, dq_node);
      }
      continue;
    }

    // recognize lone DQ node
    if (weights_to_u8_ && QDQ::MatchDQNode(node)) {
      modified |= QDQ::ConvertS8WeightToU8(graph, node, 0, 2);
      continue;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
