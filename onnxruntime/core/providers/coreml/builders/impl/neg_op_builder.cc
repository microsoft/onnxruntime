// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

// ONNX Neg. The iOS15 MIL op set has no native 'neg', so we lower it as
// mul(x, -1) with a per-dtype constant.
class NegOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status NegOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                           const logging::Logger& /*logger*/) const {
  using namespace CoreML::Specification::MILSpec;
  const auto& input_defs = node.InputDefs();
  const auto input_dtype = input_defs[0]->TypeAsProto()->tensor_type().elem_type();

  std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "mul");
  AddOperationInput(*op, "x", input_defs[0]->Name());
  if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    AddOperationInput(*op, "y", model_builder.AddScalarConstant(op->type(), "neg_one", -1.0f));
  } else if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    AddOperationInput(*op, "y", model_builder.AddScalarConstant(op->type(), "neg_one", MLFloat16(-1.0f)));
  } else if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    AddOperationInput(*op, "y", model_builder.AddScalarConstant(op->type(), "neg_one", static_cast<int32_t>(-1)));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "NegOpBuilder::AddToModelBuilderImpl, unsupported dtype: ", input_dtype);
  }
  AddOperationOutput(*op, *node.OutputDefs()[0]);
  model_builder.AddOperation(std::move(op));
  return Status::OK();
}

bool NegOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                     const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << node.OpType() << " is only supported for the ML Program format.";
    return false;
  }
  return true;
}

bool NegOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                          const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t x_type = 0;
  if (!GetType(*input_defs[0], x_type, logger)) {
    return false;
  }
  if (x_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      x_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
      x_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    LOGS(logger, VERBOSE) << node.OpType() << ": input dtype " << x_type << " not supported.";
    return false;
  }
  return true;
}

void CreateNegOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<NegOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
