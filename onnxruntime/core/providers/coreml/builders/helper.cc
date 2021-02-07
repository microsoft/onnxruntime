// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "helper.h"
#include <core/graph/graph_viewer.h>

#include "core/providers/common.h"
#include "core/providers/coreml/model/host_utils.h"
#include "op_builder_factory.h"

namespace onnxruntime {
namespace coreml {

Status GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape) {
  const auto& input_name = node_arg.Name();
  const auto* shape_proto = node_arg.Shape();
  ORT_RETURN_IF_NOT(shape_proto, "shape_proto cannot be null for input: ", input_name);

  // We already checked the shape has no dynamic dimension
  for (const auto& dim : shape_proto->dim()) {
    shape.push_back(dim.dim_value());
  }

  return Status::OK();
}

// TODO, move this to shared_library
bool GetType(const NodeArg& node_arg, int32_t& type, const logging::Logger& logger) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    LOGS(logger, WARNING) << "NodeArg [" << node_arg.Name() << "] has no input type";
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer, const logging::Logger& logger) {
  const auto& op_builders = GetOpBuilders();
  if (Contains(op_builders, node.OpType())) {
    const auto* op_builder = op_builders.at(node.OpType());
    return op_builder->IsOpSupported(graph_viewer.GetAllInitializedTensors(), node, logger);
  } else {
    return false;
  }
}

bool IsInputSupported(const NodeArg& input, const std::string& parent_name, const logging::Logger& logger) {
  const auto& input_name = input.Name();
  const auto* shape_proto = input.Shape();
  // We do not support input with no shape
  if (!shape_proto) {
    LOGS(logger, VERBOSE) << "Input [" << input_name << "] of [" << parent_name
                          << "] has not shape";
    return false;
  }

  for (const auto& dim : shape_proto->dim()) {
    // For now we do not support dynamic shape
    if (!dim.has_dim_value()) {
      LOGS(logger, WARNING) << "Dynamic shape is not supported yet, for input:" << input_name;
      return false;
    }

    // For some undocuemented reason, apple CoreML lib will fail loading the model if the model has
    // dimension > 16384
    // See this issue, https://github.com/apple/coremltools/issues/1003
    if (dim.dim_value() > 16384) {
      LOGS(logger, WARNING) << "CoreML does not support input dim > 16384, input:" << input_name
                            << ", actual dim: " << dim.dim_value();
      return false;
    }
  }

  return true;
}

std::vector<std::vector<size_t>> GetSupportedNodes(const GraphViewer& graph_viewer, const logging::Logger& logger) {
  std::vector<std::vector<size_t>> supported_node_vecs;
  if (!util::HasRequiredBaseOS()) {
    LOGS(logger, WARNING) << "All ops will fallback to CPU EP, because we do not have supported OS";
    return supported_node_vecs;
  }

  for (const auto* input : graph_viewer.GetInputs()) {
    if (!IsInputSupported(*input, "graph", logger)) {
      return supported_node_vecs;
    }
  }

  std::vector<size_t> supported_node_vec;
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer.GetNode(node_indices[i]));
    bool supported = IsNodeSupported(*node, graph_viewer, logger);
    LOGS(logger, VERBOSE) << "Operator type: [" << node->OpType()
                          << "] index: [" << i
                          << "] name: [" << node->Name()
                          << "] supported: [" << supported
                          << "]";
    if (supported) {
      supported_node_vec.push_back(i);
    } else {
      if (!supported_node_vec.empty()) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }

  if (!supported_node_vec.empty()) {
    supported_node_vecs.push_back(supported_node_vec);
  }

  return supported_node_vecs;
}

}  // namespace coreml
}  // namespace onnxruntime