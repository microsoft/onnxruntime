// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {

std::unique_ptr<ONNX_NAMESPACE::TypeProto> TypeProto__Create() {
#if BUILD_QNN_EP_STATIC
  return std::make_unique<ONNX_NAMESPACE::TypeProto>();
#else
  return ONNX_NAMESPACE::TypeProto::Create();
#endif
}

std::unique_ptr<Node_EdgeEnd> Node_EdgeEnd__Create(const Node& node, int src_arg_index, int dst_arg_index) {
#if BUILD_QNN_EP_STATIC
  return std::make_unique<Node_EdgeEnd>(node, src_arg_index, dst_arg_index);
#else
  return Node_EdgeEnd::Create(node, src_arg_index, dst_arg_index);
#endif
}

std::unique_ptr<NodeUnit> NodeUnit__Create(gsl::span<const Node* const> dq_nodes,
                                           const Node& target_node,
                                           gsl::span<const Node* const> q_nodes,
                                           NodeUnit::Type unit_type,
                                           gsl::span<const NodeUnitIODef> inputs,
                                           gsl::span<const NodeUnitIODef> outputs,
                                           size_t input_edge_count,
                                           gsl::span<const Node_EdgeEnd* const> output_edges) {
#if BUILD_QNN_EP_STATIC
  Node::EdgeSet output_edge_set;
  for (const Node_EdgeEnd* edge_end : output_edges) {
    output_edge_set.insert(*edge_end);
  }

  return std::make_unique<NodeUnit>(dq_nodes, target_node, q_nodes, unit_type,
                                    inputs, outputs, input_edge_count, output_edge_set);
#else
  return NodeUnit::Create(dq_nodes, target_node, q_nodes, unit_type, inputs, outputs, input_edge_count, output_edges);
#endif
}

namespace logging {
std::unique_ptr<Capture> Capture__Create(const Logger& logger, logging::Severity severity, const char* category,
                                         logging::DataType data_type, const CodeLocation& location) {
#if BUILD_QNN_EP_STATIC
  return std::make_unique<Capture>(logger, severity, category, data_type, location);
#else
  return Capture::Create(logger, severity, category, data_type, location);
#endif
}
}  // namespace logging
}  // namespace onnxruntime
