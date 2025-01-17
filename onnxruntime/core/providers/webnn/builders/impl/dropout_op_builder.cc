// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class DropoutOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.

void DropoutOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Skip ratio and training_mode if present.
  for (size_t i = 1; i < node.InputDefs().size(); i++) {
    const auto input_name = node.InputDefs()[i]->Name();
    model_builder.AddInitializerToSkip(input_name);
    model_builder.AddInputToSkip(input_name);
  }
}

Status DropoutOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  // WebNN EP only supports test mode. So we don't need to care about other inputs or
  // attributes about training mode. Simply use WebNN's identity op to copy the input.
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("identity", input, options);

  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));

  // If mask output is requested as output it will contain all ones (bool tensor).
  if (output_defs.size() > 1) {
    std::vector<int64_t> mask_shape;
    ORT_RETURN_IF_NOT(GetShape(*output_defs[1], mask_shape, logger), "Cannot get mask output's shape");
    std::vector<uint32_t> dims = GetVecUint32FromVecInt64(mask_shape);
    emscripten::val one_constant = model_builder.CreateOrGetConstant<uint8_t>(
        ONNX_NAMESPACE::TensorProto_DataType_BOOL, 1, dims);

    emscripten::val options = emscripten::val::object();
    options.set("label", output_defs[1]->Name() + "_identity");
    // Add additional identity op in case the mask is the output of a WebNN graph,
    // beacuse WebNN does not support a constant operand as output.
    emscripten::val mask_output = model_builder.GetBuilder().call<emscripten::val>("identity", one_constant, options);
    model_builder.AddOperand(output_defs[1]->Name(), std::move(mask_output));
  }
  return Status::OK();
}

// Operator support related.
bool DropoutOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                         const Node& node,
                                         const WebnnDeviceType /* device_type */,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  return true;
}

void CreateDropoutOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DropoutOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
