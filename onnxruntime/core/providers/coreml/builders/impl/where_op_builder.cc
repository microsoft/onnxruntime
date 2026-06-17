// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

// ONNX Where(condition, X, Y) maps to the CoreML ML Program 'select' op.
class WhereOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status WhereOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& /*logger*/) const {
  using namespace CoreML::Specification::MILSpec;
  const auto& input_defs = node.InputDefs();

  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.tensor_operation.select
  // select(cond, a, b): elements of 'a' where 'cond' is true, else elements of 'b'.
  // This matches ONNX Where(condition, X, Y), including numpy-style broadcasting.
  std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "select");
  AddOperationInput(*op, "cond", input_defs[0]->Name());  // condition
  AddOperationInput(*op, "a", input_defs[1]->Name());     // X
  AddOperationInput(*op, "b", input_defs[2]->Name());     // Y
  AddOperationOutput(*op, *node.OutputDefs()[0]);
  model_builder.AddOperation(std::move(op));
  return Status::OK();
}

bool WhereOpBuilder::IsOpSupportedImpl(const Node& /*node*/, const OpBuilderInputParams& input_params,
                                       const logging::Logger& logger) const {
  // 'select' is only emitted on the ML Program path; the NeuralNetwork builder
  // has no lowering, so reject it there and let the node fall back to CPU.
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "Where is only supported for the ML Program format.";
    return false;
  }
  return true;
}

bool WhereOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t cond_type = 0, x_type = 0, y_type = 0;
  if (!GetType(*input_defs[0], cond_type, logger) ||
      !GetType(*input_defs[1], x_type, logger) ||
      !GetType(*input_defs[2], y_type, logger)) {
    return false;
  }

  if (cond_type != ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
    LOGS(logger, VERBOSE) << "Where: 'condition' input must be bool. Got type: " << cond_type;
    return false;
  }

  // ONNX requires X and Y to share a type. CoreML 'select' handles float types.
  auto is_supported_data_type = [](int32_t type) {
    return type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
           type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  };
  if (!is_supported_data_type(x_type) || x_type != y_type) {
    LOGS(logger, VERBOSE) << "Where: 'X' and 'Y' inputs must both be float or float16. Got types: "
                          << x_type << ", " << y_type;
    return false;
  }
  return true;
}

void CreateWhereOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<WhereOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
