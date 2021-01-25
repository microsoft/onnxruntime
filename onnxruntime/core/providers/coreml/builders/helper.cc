// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "helper.h"
#include <core/graph/graph_viewer.h>

#include "core/providers/coreml/model/host_utils.h"
#include "op_builder_factory.h"

namespace onnxruntime {
namespace coreml {

// TODO, move this to shared_library
bool GetType(const NodeArg& node_arg, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    LOGS_DEFAULT(WARNING) << "NodeArg [" << node_arg.Name() << "] has no input type";
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer) {
  const auto& op_builders = GetOpBuilders();
  if (Contains(op_builders, node.OpType())) {
    const auto* op_builder = op_builders.at(node.OpType());
    return op_builder->IsOpSupported(graph_viewer.GetAllInitializedTensors(), node);
  } else {
    return false;
  }
}

bool GraphHasSupportedInputs(const GraphViewer& graph_viewer) {
  for (const auto* node_arg : graph_viewer.GetInputs()) {
    const auto& input_name = node_arg->Name();
    const auto* shape_proto = node_arg->Shape();
    if (!shape_proto) {
      LOGS_DEFAULT(WARNING) << "Input [" << input_name << "] has no shape";
      return false;
    }

    for (const auto& dim : shape_proto->dim()) {
      if (!dim.has_dim_value()) {
        LOGS_DEFAULT(WARNING) << "Dynamic shape is not supported yet, for input:" << input_name;
        return false;
      }
    }
  }

  return true;
}

std::vector<std::vector<size_t>> GetSupportedNodes(const GraphViewer& graph_viewer) {
  std::vector<std::vector<size_t>> supported_node_vecs;
  if (!util::HasRequiredBaseOS()) {
    LOGS_DEFAULT(WARNING) << "All ops will fallback to CPU EP, because we do not have supported OS";
    return supported_node_vecs;
  }

  if (!GraphHasSupportedInputs(graph_viewer))
    return supported_node_vecs;

  std::vector<size_t> supported_node_vec;
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer.GetNode(node_indices[i]));
    bool supported = IsNodeSupported(*node, graph_viewer);
    LOGS_DEFAULT(VERBOSE) << "Operator type: [" << node->OpType()
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