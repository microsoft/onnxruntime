// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/s8_to_u8.h"

namespace onnxruntime::QDQ {

bool ConvertS8WeightToU8(Graph& graph, Node& op_node,
                         size_t weights_idx, size_t weight_zp_idx) {
  auto& input_defs = op_node.MutableInputDefs();
  if (input_defs.size() < weights_idx + 1) {
    return false;
  }

  // Weight tensor must be const int8_t
  const ONNX_NAMESPACE::TensorProto* weight_tensor_proto = nullptr;
  const auto* w_def = input_defs[weights_idx];
  if (!graph_utils::NodeArgIsConstant(graph, *w_def) ||
      !graph.GetInitializedTensor(w_def->Name(), weight_tensor_proto) ||
      weight_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    return false;
  }
  ORT_ENFORCE(nullptr != weight_tensor_proto,
              "Internal Error: weight tensor must be const int8 for Avx2WeightS8ToU8Transformer.");

  // Weight zero point must be either const int8_t or null tensor
  const ONNX_NAMESPACE::TensorProto* weight_zp_tensor_proto = nullptr;
  const auto* zp_def = input_defs.size() <= weight_zp_idx ? nullptr : input_defs[weight_zp_idx];
  if (nullptr != zp_def) {
    if (!graph_utils::NodeArgIsConstant(graph, *zp_def) ||
        !graph.GetInitializedTensor(zp_def->Name(), weight_zp_tensor_proto) ||
        weight_zp_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
      return false;
    }
    ORT_ENFORCE(nullptr != weight_zp_tensor_proto,
                "Internal Error: weight zero point must be const int8 for Avx2WeightS8ToU8Transformer.");
  }

  // Convert weight tensor to uint8
  ONNX_NAMESPACE::TensorProto weights_proto_u8;
  bool converted = Int8TensorProto2Uint8(weight_tensor_proto, weights_proto_u8, graph);
  if (!converted) {
    // The weights fits into S7, overflow is not a problem, no need to convert to U8
    return false;
  }
  input_defs[weights_idx] = &graph_utils::AddInitializer(graph, weights_proto_u8);

  // Convert weight zero point to uint8
  ONNX_NAMESPACE::TensorProto weight_zp_proto_u8;
  Int8TensorProto2Uint8(weight_zp_tensor_proto, weight_zp_proto_u8, graph, true);
  input_defs[weight_zp_idx] = &graph_utils::AddInitializer(graph, weight_zp_proto_u8);

  return true;
}

}