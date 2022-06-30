// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if (defined(_M_AMD64) && !defined(_M_ARM64EC)) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)

#include "core/optimizer/avx2_weight_s8_to_u8.h"

#include <algorithm>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

/**
 * @brief Convert the source int8_t TensorProto to a uint8_t one if the tensor contains
 *        value outside of [-64, 63]
 * @param src           The source tensor, must be type int8_t
 * @param dst           An empty tensor, will contain the converted tensor data
 * @param extern_path   In case the source tensor data is external, need to provide
 *                      data path
 * @param force         Perform conversion even when tensor values within [-64, 63]
 * @return              Whether the conversion happened.
*/
static inline
bool Int8TensorProto2Uint8(const ONNX_NAMESPACE::TensorProto& src, ONNX_NAMESPACE::TensorProto& dst,
    const Path& extern_path, bool force = false) {

    dst.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    dst.set_name(src.name() + "_s8_2_u8");
    dst.mutable_dims()->CopyFrom(src.dims());

    // TODO(fuchen): too many copies!
    //
    // Here we do two memory copies: Proto -> Initializer -> Proto.
    // Ideally we only do 1 copy, just iterate the source data, and write directly to the
    // dst raw buffer.
    // Unfortunately iterating the source data is complicated, the data maybe in external
    // file, a raw buffer, or a repeated field depending on the data type.  UnpackTensor()
    // already contains some of these logic and is closest to what we need. But it does
    // not handle external data. Write our own code here means copy the logic of
    // TensorProtoToTensor(), a violation of DRY principle.

    Initializer temp(src, extern_path);
    int8_t* p = temp.data<int8_t>();
    bool should_convert = false;
    for (int i = 0; i < temp.size(); i++) {
      if (*p < -64 || *p > 63) {
        should_convert = true;
      }
      *p ^= 0x80;
      p++;
    }
    if (force || should_convert) {
      dst.set_raw_data(temp.data<int8_t>(), temp.size());
      return true;
    }
    return false;
}

/**
 * @brief If the op_node has an uint8_t const weight tensor, convert it to int8_t
 * @param graph 
 * @param op_node 
 * @param weights_idx
 * @param weight_zp_idx
 * @return true when conversion happened.
*/
static bool ConvertS8WeightToU8(Graph& graph, Node& op_node, size_t weights_idx, size_t weight_zp_idx) {
  auto& input_defs = op_node.MutableInputDefs();
  if (input_defs.size() < std::max(weights_idx, weight_zp_idx) + 1) {
    // TODO(fuchen): under what condition weight zp could be null? or not constant?
    // TODO(fuchen): should we stick in a {0} tensor sometimes?
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* weight_tensor_proto = nullptr;
  const ONNX_NAMESPACE::TensorProto* weight_zp_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph, *input_defs[weights_idx]) ||
      !graph_utils::NodeArgIsConstant(graph, *input_defs[weight_zp_idx]) ||
      !graph.GetInitializedTensor(input_defs[weights_idx]->Name(), weight_tensor_proto) ||
      !graph.GetInitializedTensor(input_defs[weight_zp_idx]->Name(), weight_zp_tensor_proto)) {
    return false;
  }

  if (weight_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8 ||
      weight_zp_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    return false;
  }

  ONNX_NAMESPACE::TensorProto weights_proto_u8;
  bool converted = Int8TensorProto2Uint8(*weight_tensor_proto, weights_proto_u8, graph.ModelPath());
  if (!converted) {
    // The weights fits into S7, overflow is not a problem, no need to convert to U8
    return false;
  }

  ONNX_NAMESPACE::TensorProto weight_zp_proto_u8;
  Int8TensorProto2Uint8(*weight_zp_tensor_proto, weight_zp_proto_u8, graph.ModelPath(), true);

  NodeArg* weights_u8_arg = &graph_utils::AddInitializer(graph, weights_proto_u8);
  NodeArg* weight_zp_u8_arg = &graph_utils::AddInitializer(graph, weight_zp_proto_u8);

  input_defs[weights_idx] = weights_u8_arg;
  input_defs[weight_zp_idx] = weight_zp_u8_arg;
  return true;
}

// For QAttention operator, if the weight is const int8, convert it to const uint8
Status Avx2WeightS8ToU8Transformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                       const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // node removed as part of an earlier fusion

    Node& op_node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(op_node, modified, graph_level, logger));

    // QAttention
    if (graph_utils::IsSupportedOptypeVersionAndDomain(op_node, "QAttention", {1}, kMSDomain) &&
        graph_utils::IsSupportedProvider(op_node, GetCompatibleExecutionProviders()) ) {
      constexpr size_t weights_idx = 1;
      constexpr size_t weight_zp_idx = 7;
      modified |= ConvertS8WeightToU8(graph, op_node, weights_idx, weight_zp_idx);
    }

    if (graph_utils::IsSupportedOptypeVersionAndDomain(op_node, "MatMulIntegerToFloat", {1}, kMSDomain) &&
        graph_utils::IsSupportedProvider(op_node, GetCompatibleExecutionProviders())) {
      constexpr size_t weights_idx = 1;
      constexpr size_t weight_zp_idx = 5;
      modified |= ConvertS8WeightToU8(graph, op_node, weights_idx, weight_zp_idx);
    }

    if (graph_utils::IsSupportedOptypeVersionAndDomain(op_node, "DynamicQuantizeMatMul", {1}, kMSDomain) &&
        graph_utils::IsSupportedProvider(op_node, GetCompatibleExecutionProviders())) {
      constexpr size_t weights_idx = 1;
      constexpr size_t weight_zp_idx = 3;
      modified |= ConvertS8WeightToU8(graph, op_node, weights_idx, weight_zp_idx);
    }

    if (graph_utils::IsSupportedOptypeVersionAndDomain(op_node, "QGemm", {1}, kMSDomain) &&
        graph_utils::IsSupportedProvider(op_node, GetCompatibleExecutionProviders())) {
      constexpr size_t weights_idx = 3;
      constexpr size_t weight_zp_idx = 5;
      modified |= ConvertS8WeightToU8(graph, op_node, weights_idx, weight_zp_idx);
    }
    if (graph_utils::IsSupportedOptypeVersionAndDomain(op_node, "MatMulInteger", {10}, kOnnxDomain) &&
        graph_utils::IsSupportedProvider(op_node, GetCompatibleExecutionProviders())) {
      constexpr size_t weights_idx = 1;
      constexpr size_t weight_zp_idx = 3;
      modified |= ConvertS8WeightToU8(graph, op_node, weights_idx, weight_zp_idx);
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // x86 or x64
