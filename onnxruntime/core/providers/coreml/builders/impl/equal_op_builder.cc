// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

// ONNX Equal(x, y) -> CoreML ML Program 'equal'. Produces a bool tensor.
class EqualOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status EqualOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& /*logger*/) const {
  using namespace CoreML::Specification::MILSpec;
  const auto& input_defs = node.InputDefs();

  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary.equal
  std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "equal");
  AddOperationInput(*op, "x", input_defs[0]->Name());
  AddOperationInput(*op, "y", input_defs[1]->Name());
  AddOperationOutput(*op, *node.OutputDefs()[0]);
  model_builder.AddOperation(std::move(op));
  return Status::OK();
}

bool EqualOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                       const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << node.OpType() << " is only supported for the ML Program format.";
    return false;
  }
  return true;
}

bool EqualOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t x_type = 0, y_type = 0;
  if (!GetType(*input_defs[0], x_type, logger) || !GetType(*input_defs[1], y_type, logger)) {
    return false;
  }

  // ONNX Equal requires both inputs share a type. CoreML 'equal' supports
  // float/fp16/int32/bool. (int64 is unrepresentable in the CoreML partition,
  // since CoreML tensors are int32 at the boundary.)
  auto is_supported_data_type = [](int32_t type) {
    return type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
           type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
           type == ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
           type == ONNX_NAMESPACE::TensorProto_DataType_BOOL;
  };
  if (!is_supported_data_type(x_type) || x_type != y_type) {
    LOGS(logger, VERBOSE) << node.OpType() << ": both inputs must share a supported type. Got: "
                          << x_type << ", " << y_type;
    return false;
  }
  return true;
}

void CreateEqualOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<EqualOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
