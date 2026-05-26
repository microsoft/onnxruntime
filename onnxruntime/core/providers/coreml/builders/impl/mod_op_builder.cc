// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

// ONNX Mod(x, y) with attribute fmod=0 -> CoreML ML Program 'mod'. Both
// implement floor-mod semantics (sign follows the divisor / Python % style).
// We reject fmod=1 (C-style truncated remainder, sign follows the dividend)
// because the iOS15 MIL ops set has no native fmod and decomposing it would
// require trunc + mul + sub, which is not worth the complexity for now.
class ModOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status ModOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                           const logging::Logger& /*logger*/) const {
  using namespace CoreML::Specification::MILSpec;
  const auto& input_defs = node.InputDefs();

  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary.mod
  std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "mod");
  AddOperationInput(*op, "x", input_defs[0]->Name());
  AddOperationInput(*op, "y", input_defs[1]->Name());
  AddOperationOutput(*op, *node.OutputDefs()[0]);
  model_builder.AddOperation(std::move(op));
  return Status::OK();
}

bool ModOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                     const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << node.OpType() << " is only supported for the ML Program format.";
    return false;
  }

  NodeAttrHelper helper(node);
  const auto fmod = helper.Get("fmod", static_cast<int64_t>(0));
  if (fmod != 0) {
    // ONNX fmod=1 is C-style truncated remainder; MIL 'mod' is floor mod, so
    // the semantics don't match. Fall back to CPU.
    LOGS(logger, VERBOSE) << node.OpType() << " with fmod=1 (C-style remainder) is not supported.";
    return false;
  }
  return true;
}

bool ModOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                          const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t x_type = 0, y_type = 0;
  if (!GetType(*input_defs[0], x_type, logger) || !GetType(*input_defs[1], y_type, logger)) {
    return false;
  }
  // ONNX Mod requires both inputs share a type. MIL 'mod' supports int32/fp16/fp32.
  auto is_supported_data_type = [](int32_t type) {
    return type == ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
           type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
           type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  };
  if (!is_supported_data_type(x_type) || x_type != y_type) {
    LOGS(logger, VERBOSE) << node.OpType() << ": both inputs must share a supported type. Got: "
                          << x_type << ", " << y_type;
    return false;
  }
  return true;
}

void CreateModOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ModOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
