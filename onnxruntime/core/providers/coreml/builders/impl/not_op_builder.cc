// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

// Elementwise unary logical op. ONNX Not -> CoreML ML Program 'logical_not'.
class NotOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status NotOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                           const logging::Logger& /*logger*/) const {
  using namespace CoreML::Specification::MILSpec;
  const auto& input_defs = node.InputDefs();

  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary.logical_not
  std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "logical_not");
  AddOperationInput(*op, "x", input_defs[0]->Name());
  AddOperationOutput(*op, *node.OutputDefs()[0]);
  model_builder.AddOperation(std::move(op));
  return Status::OK();
}

bool NotOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                     const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << node.OpType() << " is only supported for the ML Program format.";
    return false;
  }
  return true;
}

bool NotOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                          const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t x_type = 0;
  if (!GetType(*input_defs[0], x_type, logger)) {
    return false;
  }

  // ONNX Not takes a bool input; CoreML logical_not likewise operates on bool tensors.
  if (x_type != ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
    LOGS(logger, VERBOSE) << node.OpType() << ": input must be bool. Got type: " << x_type;
    return false;
  }
  return true;
}

void CreateNotOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<NotOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
