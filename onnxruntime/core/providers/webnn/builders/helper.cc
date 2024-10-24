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

bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer, const WebnnDeviceType device_type,
                     const emscripten::val& wnn_limits, const logging::Logger& logger) {
  const auto& op_builders = GetOpBuilders();
  if (Contains(op_builders, node.OpType())) {
    const auto* op_builder = op_builders.at(node.OpType());
    return op_builder->IsOpSupported(graph_viewer.GetAllInitializedTensors(), node, device_type, wnn_limits, logger);
  } else {
    return false;
  }
}

bool IsInputSupported(const NodeArg& input, const std::string& parent_name, const logging::Logger& logger) {
  const auto& input_name = input.Name();
  const auto* shape_proto = input.Shape();
  // Optional tensors can be indicated by an empty name, just ignore it.
  if (input_name.empty()) {
    return true;
  }
  // We do not support input with no shape.
  if (!shape_proto) {
    LOGS(logger, VERBOSE) << "Input [" << input_name << "] of [" << parent_name
                          << "] has not shape";
    return false;
  }

  for (const auto& dim : shape_proto->dim()) {
    // WebNN doesn't support dynamic shape - use sessionOptions.freeDimensionOverrides to fix the shape.
    if (!dim.has_dim_value()) {
      LOGS(logger, VERBOSE) << "Dynamic shape is not supported, "
                            << "use sessionOptions.FreeDimensionOverrides to set a fixed shape for input: "
                            << input_name;
      return false;
    }
    if (dim.dim_value() == 0) {
      LOGS(logger, VERBOSE) << "The shape of [" << node_arg_name << "] has 0 dimension which is not supported by WebNN";
      return false;
    }
  }

  return true;
}

std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                      const emscripten::val& wnn_builder,
                                                      const WebnnDeviceType device_type,
                                                      const emscripten::val& wnn_limits,
                                                      const logging::Logger& logger) {
  std::vector<std::vector<size_t>> supported_node_groups;

  for (const auto* input : graph_viewer.GetInputs()) {
    if (!IsInputSupported(*input, "graph", logger)) {
      return supported_node_groups;
    }
  }

  std::vector<size_t> supported_node_group;
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();

  for (size_t i = 0; i < node_indices.size(); i++) {
    auto node_idx = node_indices[i];
    const auto* node(graph_viewer.GetNode(node_idx));
    bool supported = false;
    // Firstly check if platform supports the WebNN op.
    if (CheckSingleOp(node->OpType(), wnn_builder, device_type)) {
      LOGS(logger, VERBOSE) << "Operator type: [" << node->OpType() << "] is supported by browser";
      supported = IsNodeSupported(*node, graph_viewer, device_type, wnn_limits, logger);
    }

    LOGS(logger, VERBOSE) << "Operator type: [" << node->OpType()
                          << "] index: [" << node_idx
                          << "] name: [" << node->Name()
                          << "] supported: [" << supported
                          << "]";
    if (supported) {
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

  if (!IsSupportedDataType(onnx_data_type, wnn_limits[webnn_op_type][webnn_input_output_name]["dataTypes"])) {
    LOGS(logger, VERBOSE) << "[" << onnx_op_type
                          << "] " << onnx_input_output_name
                          << " type: [" << onnx_data_type
                          << "] is not supported for now";
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

}  // namespace webnn
}  // namespace onnxruntime
