// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

// Elementwise binary logical ops. ONNX And -> CoreML ML Program 'logical_and'.
class LogicalOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status LogicalOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                               const logging::Logger& /*logger*/) const {
  using namespace CoreML::Specification::MILSpec;
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();

  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary.logical_and
  std::string_view coreml_op_type;
  if (op_type == "And") {
    coreml_op_type = "logical_and";
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "LogicalOpBuilder::AddToModelBuilderImpl, unexpected op: ", op_type);
  }

  std::unique_ptr<Operation> op = model_builder.CreateOperation(node, coreml_op_type);
  AddOperationInput(*op, "x", input_defs[0]->Name());
  AddOperationInput(*op, "y", input_defs[1]->Name());
  AddOperationOutput(*op, *node.OutputDefs()[0]);
  model_builder.AddOperation(std::move(op));
  return Status::OK();
}

bool LogicalOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                         const logging::Logger& logger) const {
  // logical_and is only emitted on the ML Program path.
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << node.OpType() << " is only supported for the ML Program format.";
    return false;
  }
  return true;
}

bool LogicalOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t x_type = 0, y_type = 0;
  if (!GetType(*input_defs[0], x_type, logger) || !GetType(*input_defs[1], y_type, logger)) {
    return false;
  }

  // ONNX And takes bool inputs; CoreML logical_and likewise operates on bool tensors.
  if (x_type != ONNX_NAMESPACE::TensorProto_DataType_BOOL ||
      y_type != ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
    LOGS(logger, VERBOSE) << node.OpType() << ": both inputs must be bool. Got types: "
                          << x_type << ", " << y_type;
    return false;
  }
  return true;
}

void CreateLogicalOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LogicalOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
