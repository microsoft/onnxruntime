// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "helper.h"
#include <core/graph/graph_viewer.h>

#include "op_builder_factory.h"

namespace onnxruntime {
namespace webnn {

WebnnDeviceType DeviceTypeFromString(const std::string_view& device_type) {
  if (device_type == "gpu") {
    return WebnnDeviceType::GPU;
  }
  if (device_type == "cpu") {
    return WebnnDeviceType::CPU;
  }
  if (device_type == "npu") {
    return WebnnDeviceType::NPU;
  }
  ORT_THROW("Unknown WebNN deviceType.");
}

InitializedTensorSet CollectAllInitializedTensors(const GraphViewer& graph_viewer) {
  InitializedTensorSet all_initializers;
  if (graph_viewer.IsSubgraph()) {
    const Graph* cur_graph = &graph_viewer.GetGraph();
    // Traverse up to the top-level graph, collecting all initializers.
    while (cur_graph->IsSubgraph()) {
      const auto& current_initializers = cur_graph->GetAllInitializedTensors();
      all_initializers.insert(current_initializers.begin(), current_initializers.end());
      cur_graph = cur_graph->ParentGraph();
    }
    // Collect initializers in top-level graph.
    const auto& current_initializers = cur_graph->GetAllInitializedTensors();
    all_initializers.insert(current_initializers.begin(), current_initializers.end());
  }

  return all_initializers;
}

bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger) {
  const auto* shape_proto = node_arg.Shape();
  if (!shape_proto) {
    LOGS(logger, WARNING) << "NodeArg [" << node_arg.Name() << "] has no shape info";
    return false;
  }

  // We already checked the shape has no dynamic dimension.
  for (const auto& dim : shape_proto->dim()) {
    shape.push_back(dim.dim_value());
  }

  return true;
}

bool FindCastFusableNodeIndex(const Node& node, const bool is_prec_node, NodeIndex& index) {
  if (is_prec_node) {
    // Preceding node should have only one input edge.
    if (node.GetInputEdgesCount() == 1) {
      const auto& prec_node = node.InputEdgesBegin()->GetNode();
      // Preceding node should have only one output edge.
      if (prec_node.GetOutputEdgesCount() == 1) {
        index = prec_node.Index();
        return true;
      }
    }
  } else {
    // Successive node should have only one output edge.
    if (node.GetOutputEdgesCount() == 1) {
      const auto& next_node = node.OutputEdgesBegin()->GetNode();
      index = next_node.Index();
      return true;
    }
  }

  return false;
}

bool IsCastFusable(const Node& node, const bool is_prec_node, const logging::Logger& logger) {
  if (node.OpType() != "Cast") {
    return false;
  }

  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type, logger))
    return false;

  NodeAttrHelper cast_helper(node);
  const auto to_type = cast_helper.Get("to", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  if (is_prec_node) {
    // If it is preceding node, only allows casting from int64 to int32.
    if (input_type != ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
        to_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      return false;
    }
  } else {
    // If it is successive node, only allows casting from int32 to int64.
    if (input_type != ONNX_NAMESPACE::TensorProto_DataType_INT32 &&
        to_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      return false;
    }
  }

  NodeIndex index;
  return FindCastFusableNodeIndex(node, is_prec_node, index);
}

bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer, const WebnnDeviceType device_type,
                     const emscripten::val& wnn_limits, bool& is_fusable, const logging::Logger& logger) {
  const auto& op_builders = GetOpBuilders();
  if (Contains(op_builders, node.OpType())) {
    const auto* op_builder = op_builders.at(node.OpType());
    return op_builder->IsOpSupported(
        graph_viewer.GetAllInitializedTensors(), node, device_type, wnn_limits, is_fusable, logger);
  } else {
    return false;
  }
}

bool IsTensorShapeSupported(const NodeArg& node_arg, const std::string& parent_name,
                            const logging::Logger& logger, bool allow_empty_input) {
  const auto& node_arg_name = node_arg.Name();
  const auto* shape_proto = node_arg.Shape();
  // Optional tensors can be indicated by an empty name, just ignore it.
  if (node_arg_name.empty()) {
    return true;
  }
  // We do not support input/output with no shape.
  if (!shape_proto) {
    LOGS(logger, VERBOSE) << "Node arg [" << node_arg_name << "] of [" << parent_name << "] has not shape";
    return false;
  }

  for (const auto& dim : shape_proto->dim()) {
    // WebNN doesn't support dynamic shape - use sessionOptions.freeDimensionOverrides to fix the shape.
    if (!dim.has_dim_value()) {
      LOGS(logger, VERBOSE) << "Dynamic shape is not supported, "
                            << "use sessionOptions.FreeDimensionOverrides to set a fixed shape: " << node_arg_name;
      return false;
    }
    if (dim.dim_value() == 0 && !allow_empty_input) {
      LOGS(logger, VERBOSE) << "The shape of [" << node_arg_name << "] has 0 dimension which is not supported by WebNN";
      return false;
    }
  }

  return true;
}

std::vector<std::vector<NodeIndex>> GetSupportedNodes(
    const GraphViewer& graph_viewer, const emscripten::val& wnn_builder,
    const WebnnDeviceType device_type, const emscripten::val& wnn_limits,
    InlinedHashMap<NodeIndex, NodeIndex>& fused_node_map, const logging::Logger& logger) {
  std::vector<std::vector<size_t>> supported_node_groups;
  std::vector<size_t> supported_node_group;
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();

  // Some WebNN backends do not support the int64 data type, but this limitation can be addressed by converting the
  // model's int64 inputs, outputs, and initializers to int32. However, certain ONNX nodes, such as ArgMax, ArgMin,
  // and ScatterND, require the int64 data type for specific inputs or outputs.
  // To handle such case, we can add Cast nodes before or after these nodes in the model and fuse them during WebNN EP
  // optimization. The fusion strategy is as follows:
  // 1. Verify if the Cast node can be fused with either the preceding node or the successive node.
  // 2. Check if the node requiring the int64 data type can be supported solely by addressing the int64 data type
  //    limitation. Ensure that the node is unsupported only due to the int64 restriction.
  // 3. Use an is_fusable flag to record paired nodes as <Cast node index, fusable node index> that can be
  //    fused together.
  // 4. Mark the fusable nodes as supported after identifying them.
  // 5. During WebNN graph compilation, skip the Cast node and fuse it in its paired fusable node.

  InlinedHashMap<NodeIndex, WebnnNodeInfo> node_info_map;
  std::vector<NodeIndex> fusable_cast_nodes;
  for (size_t i = 0; i < node_indices.size(); i++) {
    auto node_idx = node_indices[i];
    const auto* node(graph_viewer.GetNode(node_idx));
    bool supported = false;
    bool is_fusable = false;
    // Firstly check if platform supports the WebNN op.
    if (CheckSingleOp(node->OpType(), wnn_builder, device_type)) {
      supported = IsNodeSupported(*node, graph_viewer, device_type, wnn_limits, is_fusable, logger);
      node_info_map[node_idx] = {supported, is_fusable};
    }

    if (node->OpType() == "Cast" && is_fusable) {
      fusable_cast_nodes.push_back(node_idx);
    }
  }

  // Try to find the fusable nodes for the Cast nodes.
  // Note: graph partition will make sure fusable nodes are in the same partition.
  for (size_t i = 0; i < fusable_cast_nodes.size(); i++) {
    NodeIndex fusable_node_idx;
    NodeIndex fusable_cast_node_idx = fusable_cast_nodes[i];
    const auto& cast_node = *graph_viewer.GetNode(fusable_cast_node_idx);
    // Cast can only be fused by either preceding node or successive node.
    bool fusable_found = FindCastFusableNodeIndex(cast_node, true, fusable_node_idx) ||
                         FindCastFusableNodeIndex(cast_node, false, fusable_node_idx);

    if (fusable_found && node_info_map[fusable_node_idx].fusable) {
      // Set the fusable nodes to supported.
      node_info_map[fusable_cast_node_idx].supported = true;
      node_info_map[fusable_node_idx].supported = true;
      fused_node_map[fusable_cast_node_idx] = fusable_node_idx;
    }
  }

  for (size_t i = 0; i < node_indices.size(); i++) {
    auto node_idx = node_indices[i];
    const auto* node(graph_viewer.GetNode(node_idx));
    WebnnNodeInfo node_info = node_info_map[node_idx];

    LOGS(logger, VERBOSE) << "Operator type: [" << node->OpType()
                          << "] index: [" << node_idx
                          << "] name: [" << node->Name()
                          << "] supported: [" << node_info.supported
                          << "]";
    if (node_info.supported) {
      supported_node_group.push_back(node_idx);
    } else {
      if (!supported_node_group.empty()) {
        supported_node_groups.push_back(supported_node_group);
        supported_node_group.clear();
      }
    }
  }

  if (!supported_node_group.empty()) {
    supported_node_groups.push_back(supported_node_group);
  }

  return supported_node_groups;
}

bool AreInputDataTypesSame(const std::string& op_type,
                           gsl::span<const int32_t> input_types,
                           const logging::Logger& logger) {
  for (size_t i = 1; i < input_types.size(); i++) {
    if (input_types[0] != input_types[i]) {
      LOGS(logger, VERBOSE) << "[" << op_type
                            << "] Input data types should be the same, but ["
                            << input_types[0] << "] does not match "
                            << input_types[i] << "].";
      return false;
    }
  }
  return true;
}

bool IsSupportedDataType(const int32_t onnx_data_type, const emscripten::val& webnn_supported_data_types) {
  auto it = onnx_to_webnn_data_type_map.find(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(onnx_data_type));
  if (it == onnx_to_webnn_data_type_map.end())
    return false;

  std::string webnn_data_type = it->second;

  // Check if WebNN supports the data type.
  emscripten::val is_supported = webnn_supported_data_types.call<emscripten::val>("includes",
                                                                                  emscripten::val(webnn_data_type));
  return is_supported.as<bool>();
}

// Check if the input or output data type of ONNX node is supported by the WebNN operator.
bool IsDataTypeSupportedByOp(const std::string& onnx_op_type,
                             const int32_t onnx_data_type,
                             const emscripten::val& wnn_limits,
                             const std::string& webnn_input_output_name,
                             const std::string& onnx_input_output_name,
                             const logging::Logger& logger) {
  std::string webnn_op_type;
  if (!GetWebNNOpType(onnx_op_type, webnn_op_type))
    return false;

  return IsDataTypeSupportedByWebNNOp(onnx_op_type, webnn_op_type, onnx_data_type, wnn_limits,
                                      webnn_input_output_name, onnx_input_output_name, logger);
}

bool IsDataTypeSupportedByWebNNOp(const std::string& onnx_op_type,
                                  const std::string& webnn_op_type,
                                  const int32_t onnx_data_type,
                                  const emscripten::val& wnn_limits,
                                  const std::string& webnn_input_output_name,
                                  const std::string& onnx_input_output_name,
                                  const logging::Logger& logger) {
  if (wnn_limits[webnn_op_type].isUndefined()) {
    LOGS(logger, VERBOSE) << "[" << onnx_op_type << "] WebNN op [" << webnn_op_type << "] is not supported for now";
    return false;
  }
  if (wnn_limits[webnn_op_type][webnn_input_output_name].isUndefined()) {
    LOGS(logger, VERBOSE) << "[" << onnx_op_type << "] WebNN op [" << webnn_op_type << "] doesn't have parameter ["
                          << webnn_input_output_name << "]";
    return false;
  }
  if (!IsSupportedDataType(onnx_data_type, wnn_limits[webnn_op_type][webnn_input_output_name]["dataTypes"])) {
    LOGS(logger, VERBOSE) << "[" << onnx_op_type << "] " << onnx_input_output_name << "'s data type: ["
                          << onnx_data_type << "] is not supported by WebNN op [" << webnn_op_type << "] for now";
    return false;
  }
  return true;
}

bool GetBidirectionalBroadcastShape(std::vector<int64_t>& shape_a,
                                    std::vector<int64_t>& shape_b,
                                    std::vector<int64_t>& output_shape) {
  size_t size_a = shape_a.size();
  size_t size_b = shape_b.size();
  size_t smaller_size = std::min(size_a, size_b);
  size_t larger_size = std::max(size_a, size_b);

  output_shape.resize(larger_size);

  for (size_t i = 0; i < larger_size; i++) {
    // right alignment
    size_t axis_a = size_a - i - 1;
    size_t axis_b = size_b - i - 1;

    if (i < smaller_size) {
      // Broadcastable tensors must either have each dimension the same size or equal to one.
      if (shape_a[axis_a] != shape_b[axis_b] && shape_a[axis_a] != 1 && shape_b[axis_b] != 1) {
        return false;
      }
      output_shape[larger_size - i - 1] = std::max(shape_a[axis_a], shape_b[axis_b]);
    } else {
      // For the remaining dimensions in the larger tensor, copy the dimension size directly to the output shape.
      output_shape[larger_size - i - 1] = (size_a > size_b) ? shape_a[axis_a] : shape_b[axis_b];
    }
  }

  return true;
}

bool SetWebnnDataType(emscripten::val& desc, const int32_t data_type) {
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT4:
      desc.set("dataType", emscripten::val("int4"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
      desc.set("dataType", emscripten::val("uint4"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      desc.set("dataType", emscripten::val("uint8"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      desc.set("dataType", emscripten::val("int8"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      desc.set("dataType", emscripten::val("float16"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      desc.set("dataType", emscripten::val("float32"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      desc.set("dataType", emscripten::val("int32"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      desc.set("dataType", emscripten::val("int64"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      desc.set("dataType", emscripten::val("uint32"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      desc.set("dataType", emscripten::val("uint64"));
      return true;
    default:
      return false;
  }
}

bool IsMLTensorSupported() {
  static bool is_supported = !emscripten::val::global("MLTensor").isUndefined();
  return is_supported;
}

// Convert int8 to uint4/int4 (stored as uint8)
uint8_t PackInt8ToUint8AsNibble(int8_t value, const int32_t& data_type) {
  uint8_t result = 0;
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_UINT4) {
    if (value < 0 || value > 15) {
      ORT_THROW("Value cannot be safely converted to uint4.");
    }
    result |= (static_cast<uint8_t>(value) << 4);
  } else {
    if (value < -8 || value > 7) {
      ORT_THROW("Value cannot be safely converted to int4.");
    }
    result |= (value << 4);
  }

  return result;
}

// Convert float32 to float16 (stored as uint16)
uint16_t PackFloat32ToUint16AsFloat16(float value) {
  uint32_t float32_bits;

  // Safely copy the float bits into an integer
  std::memcpy(&float32_bits, &value, sizeof(float));

  // Extract the sign, exponent, and mantissa from the float32 bits
  uint32_t sign = (float32_bits >> 31) & 0x1;
  uint32_t exponent = (float32_bits >> 23) & 0xFF;
  uint32_t mantissa = float32_bits & 0x7FFFFF;

  // Shift the sign for float16
  uint16_t sign_float16 = sign << 15;

  // Handle special cases: Infinity and NaN
  if (exponent == 255) {
    return sign_float16 | (0x1F << 10) | (mantissa ? 0x200 : 0);
  }
  // Handle zero and subnormal numbers in float32
  if (exponent == 0) {
    return sign_float16 | (mantissa >> 13);
  }

  // Adjust the exponent for float16 (subtract bias difference: 127 - 15 = 112)
  int exponent_float16 = exponent - 112;

  // Handle exponent overflow (larger than float16 can represent)
  if (exponent_float16 >= 0x1F) {
    return sign_float16 | (0x1F << 10);
  }
  // Handle exponent underflow (smaller than float16 can represent)
  if (exponent_float16 <= 0) {
    mantissa = (mantissa | 0x800000) >> (1 - exponent_float16);
    return sign_float16 | (mantissa >> 13);
  }

  // Adjust the mantissa by shifting it to fit float16 format (round to nearest even)
  uint16_t mantissa_float16 = (mantissa + 0x1000) >> 13;

  // Combine sign, exponent, and mantissa into the final float16 representation
  return sign_float16 | (exponent_float16 << 10) | mantissa_float16;
}

}  // namespace webnn
}  // namespace onnxruntime
