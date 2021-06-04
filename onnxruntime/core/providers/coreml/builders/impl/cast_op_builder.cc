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
  // Right now we're only handling an ArgMax op followed by a Cast to int32 type.
  // This can fuse the ArgMax's int64 output type which is not supported in CoreML model.
  // And that ArgMax fused with the cast node produces an int32 output, so we're skipping adding the Cast node here.
  return Status::OK();
}

// Operator support related

bool CastOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  if (node.GetInputEdgesCount() == 0) {
    LOGS(logger, VERBOSE) << "Failed to get Cast's preceding nodes.";
    return false;
  }

  if (node.GetInputEdgesCount() > 1) {
    LOGS(logger, VERBOSE) << "Multiple nodes consuming Cast's output.";
    return false;
  }

  const auto* prec_node(input_params.graph_viewer.GetNode(node.InputEdgesBegin()->GetNode().Index()));

  /*Cast node is only aimed for supporting argmax and we are only handling the case where an argmax 
    followed by a cast node. We need to check if the preceding node is an argmax and also if it's a
    supported argmax op type.*/
  if (prec_node->OpType() != "ArgMax") {
    LOGS(logger, VERBOSE) << "Cast's producing node is not ArgMax is not supported."
                          << "Current producing node: [" << prec_node->OpType()
                          << "]";
    return false;
  }
  if (!IsNodeSupported(*prec_node, input_params.graph_viewer, logger)) {
    LOGS(logger, VERBOSE) << "Cast's producing node ["
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
