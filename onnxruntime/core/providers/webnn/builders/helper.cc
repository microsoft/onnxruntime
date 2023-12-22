// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "helper.h"
#include <core/graph/graph_viewer.h>

#include "op_builder_factory.h"

namespace onnxruntime {
namespace webnn {

InitializedTensorSet CollectAllInitializedTensors(const onnxruntime::GraphViewer& graph_viewer) {
  InitializedTensorSet all_initializers;
  if (graph_viewer.IsSubgraph()) {
    const Graph* cur_graph = &graph_viewer.GetGraph();
    // Traverse up to the top-level graph, collecting all initializers.
    while (cur_graph->IsSubgraph()) {
      const auto& current_initializers = cur_graph->GetAllInitializedTensors();
      all_initializers.insert(current_initializers.begin(), current_initializers.end());
      cur_graph = cur_graph->ParentGraph();
    };
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

bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer,
                     const WebnnDeviceType device_type, const logging::Logger& logger) {
  const auto& op_builders = GetOpBuilders();
  if (Contains(op_builders, node.OpType())) {
    const auto* op_builder = op_builders.at(node.OpType());
    return op_builder->IsOpSupported(graph_viewer.GetAllInitializedTensors(), node, device_type, logger);
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
  }

  return true;
}

std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                      const emscripten::val& wnn_builder_,
                                                      const WebnnDeviceType device_type,
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
    if (CheckSingleOp(node->OpType(), wnn_builder_, device_type)) {
      LOGS(logger, VERBOSE) << "Operator type: [" << node->OpType() << "] is supported by browser";
      supported = IsNodeSupported(*node, graph_viewer, device_type, logger);
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

bool IsSupportedDataType(const int32_t data_type, const WebnnDeviceType device_type) {
  // Current data type implementation status of WebNN is inconsistent along with different backends,
  // The XNNPack backend supports only FP32, while the DML backend POC supports more.
  if (device_type == WebnnDeviceType::CPU) {
    return std::find(supported_cpu_data_types.begin(), supported_cpu_data_types.end(), data_type) !=
           supported_cpu_data_types.end();
  } else {
    return std::find(supported_gpu_data_types.begin(), supported_gpu_data_types.end(), data_type) !=
           supported_gpu_data_types.end();
  }
}

bool IsValidMultidirectionalBroadcast(std::vector<int64_t>& shape_a,
                                      std::vector<int64_t>& shape_b,
                                      const logging::Logger& logger) {
  size_t size_a = shape_a.size();
  size_t size_b = shape_b.size();
  size_t smaller_size = std::min(size_a, size_b);
  for (size_t i = 0; i < smaller_size; i++) {
    // right alignment
    size_t axis_a = size_a - i - 1;
    size_t axis_b = size_b - i - 1;
    // Broadcastable tensors must either have each dimension the same size or equal to one.
    if (shape_a[axis_a] != shape_b[axis_b] && shape_a[axis_a] != 1 && shape_b[axis_b] != 1) {
      return false;
    }
  }
  return true;
}

bool SetWebnnDataType(emscripten::val& desc, const int32_t data_type) {
  // WebNN changed the name of the MLOperandDescriptor's data type from "type" to "dataType",
  // use a duplicate entry temporarily to workaround this API breaking issue.
  // TODO: Remove legacy "type" once all browsers implement the new "dataType".
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      desc.set("type", emscripten::val("uint8"));
      desc.set("dataType", emscripten::val("uint8"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      desc.set("type", emscripten::val("float16"));
      desc.set("dataType", emscripten::val("float16"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      desc.set("type", emscripten::val("float32"));
      desc.set("dataType", emscripten::val("float32"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      desc.set("type", emscripten::val("int32"));
      desc.set("dataType", emscripten::val("int32"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      desc.set("type", emscripten::val("int64"));
      desc.set("dataType", emscripten::val("int64"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      desc.set("type", emscripten::val("uint32"));
      desc.set("dataType", emscripten::val("uint32"));
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      desc.set("type", emscripten::val("uint64"));
      desc.set("dataType", emscripten::val("uint64"));
      return true;
    default:
      return false;
  }
}

}  // namespace webnn
}  // namespace onnxruntime
