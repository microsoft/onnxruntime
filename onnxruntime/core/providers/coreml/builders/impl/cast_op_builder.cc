// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class CastOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const Node& node, OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

// Add operator related

Status CastOpBuilder::AddToModelBuilderImpl(ModelBuilder& /* model_builder */,
                                            const Node& /* node */,
                                            const logging::Logger& /* logger */) const {
  return Status::OK();
}

// Operator support related

bool CastOpBuilder::IsOpSupportedImpl(const Node& node, OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  // Check if the preceding node is [ArgMax] and the argmax op is supported
  std::vector<size_t> prec_node_indices;
  for (auto it = node.InputEdgesBegin(), end = node.InputEdgesEnd(); it != end; ++it) {
    prec_node_indices.push_back(it->GetNode().Index());
  }

  if (prec_node_indices.size() > 1) {
    LOGS(logger, VERBOSE) << "Case - [Cast] has multiple preceding nodes: Not supported.";
    return false;
  }

  if (prec_node_indices.empty()) {
    LOGS(logger, VERBOSE) << "Failed to get [Cast]'s preceding nodes.";
    return false;
  }

  const auto* prec_node(input_params.graph_viewer.GetNode(prec_node_indices[0]));
  if (prec_node->OpType() != "ArgMax") {
    LOGS(logger, VERBOSE) << "Case - [Cast]'s preceding node is not [ArgMax]: Not supported. "
                          << "Current previous node: [" << prec_node->OpType()
                          << "]";
    return false;
  }
  if (!IsNodeSupported(*prec_node, input_params.graph_viewer, logger)) {
    LOGS(logger, VERBOSE) << "Case - [Cast]'s preceding node ["
                          << prec_node->OpType()
                          << "] is not a supported op.";
    return false;
  }

  // Check if the output type of cast node is int32
  const auto& node_output = *node.OutputDefs()[0];
  int32_t node_output_type;
  if (!GetType(node_output, node_output_type, logger)) {
    return false;
  }
  if (node_output_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Output type: [" << node_output_type
                          << "] is not supported.";
    return false;
  }

  return true;
}

void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<CastOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
