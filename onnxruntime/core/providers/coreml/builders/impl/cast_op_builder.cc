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
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

// Add operator related

Status CastOpBuilder::AddToModelBuilderImpl(ModelBuilder& /* model_builder */,
                                            const Node& /* node */,
                                            const logging::Logger& /* logger */) const {
  /* CoreML does not have cast op, so [cast] is not actually supported here, only aimed for supporting argmax.
     We use the case [ArgMax(int64)-Cast(int32)] to skip argmax's output int64 type which is not supported in CoreML.
  */
  return Status::OK();
}

// Operator support related

bool CastOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  // Check if the preceding node is [ArgMax] and the argmax op is supported
  if (node.GetInputEdgesCount() == 0) {
    LOGS(logger, VERBOSE) << "Failed to get [Cast]'s preceding nodes.";
    return false;
  }

  if (node.GetInputEdgesCount() > 1) {
    LOGS(logger, VERBOSE) << "Case - [Cast] has multiple preceding nodes: Not supported.";
    return false;
  }

  const auto* prec_node(input_params.graph_viewer.GetNode(node.InputEdgesBegin()->GetNode().Index()));

  if (prec_node->OpType() != "ArgMax") {
    LOGS(logger, VERBOSE) << "Case - [Cast]'s preceding node is not [ArgMax]: Not supported. "
                          << "Current previous node: [" << prec_node->OpType()
                          << "]";
    return false;
  }
  /* Cast node op is only aimed for supporting argmax. 
    If the preceding argmax op is not supported, then cast op is not needed */
  if (!IsNodeSupported(*prec_node, input_params.graph_viewer, logger)) {
    LOGS(logger, VERBOSE) << "Case - [Cast]'s preceding node ["
                          << prec_node->OpType()
                          << "] is not a supported op.";
    return false;
  }

  // Check if the output type of cast node is int32
  NodeAttrHelper helper(node);
  const auto cast_to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto::UNDEFINED);
  if (cast_to_type != ONNX_NAMESPACE::TensorProto::INT32) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Output type: [" << cast_to_type
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
