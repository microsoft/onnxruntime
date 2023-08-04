// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/helper.h"
#ifdef __APPLE__
#include "core/providers/coreml/builders/model_builder.h"
#endif
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class CastOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
#ifdef __APPLE__
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
#endif
  // Operator support related
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const Node& node, const logging::Logger& logger) const override;
};

// Add operator related

#ifdef __APPLE__
Status CastOpBuilder::AddToModelBuilderImpl(ModelBuilder& /* model_builder */,
                                            const Node& /* node */,
                                            const logging::Logger& /* logger */) const {
  // This is a special handling case for ArgMax Op, where argmax is followed by a cast to int32 type.
  // The ArgMax is fused with the Cast node and produces an int32 output.
  // Cast node is not provided in CoreML model, so we're skipping adding the Cast node here.
  return Status::OK();
}
#endif

// Operator support related

bool CastOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  if (node.GetInputEdgesCount() == 0) {
    LOGS(logger, VERBOSE) << "Cast has no preceding nodes.";
    return false;
  }

  if (node.GetInputEdgesCount() > 1) {
    LOGS(logger, VERBOSE) << "Multiple nodes producing Cast's input.";
    return false;
  }

  const auto& prec_node = node.InputEdgesBegin()->GetNode();

  /*Cast node is only aimed for supporting argmax and we are only handling the case where an argmax
    followed by a cast node. We need to check if the preceding node is an argmax and also if it's a
    supported argmax op type.*/
  if (prec_node.OpType() != "ArgMax") {
    LOGS(logger, VERBOSE) << "Cast's producing node is not ArgMax is not supported."
                          << "Current producing node: [" << prec_node.OpType()
                          << "]";
    return false;
  }
  if (!IsNodeSupported(prec_node, input_params, logger)) {
    LOGS(logger, VERBOSE) << "Cast's producing node ["
                          << prec_node.OpType()
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

bool CastOpBuilder::HasSupportedInputsImpl(const Node& node, const logging::Logger& logger) const {
  // We only check the type of input 0
  const auto& input = *node.InputDefs()[0];

  int32_t input_type;
  if (!GetType(input, input_type, logger))
    return false;

  // only support int64 coming from ArgMax (check for ArgMax is done in IsOpSupportedImpl())
  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
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
