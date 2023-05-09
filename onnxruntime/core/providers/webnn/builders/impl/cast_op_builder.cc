// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class CastOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const logging::Logger& logger) const override;

  int GetMinSupportedOpSet(const Node& node) const override;
};

// Add operator related.

Status CastOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                            const Node& node,
                                            const logging::Logger& logger) const {
  const auto& input_name = node.InputDefs()[0]->Name();
  emscripten::val input = model_builder.GetOperand(input_name);

  NodeAttrHelper helper(node);
  // We already checked the "to" type in IsOpSupportedImpl.
  const auto to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  std::string operand_type =
      to_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ? "float32" : "float16";

  emscripten::val output =
      model_builder.GetBuilder().call<emscripten::val>("cast", input, emscripten::val(operand_type));

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

int CastOpBuilder::GetMinSupportedOpSet(const Node& /* node */) const {
  // Since opset 6, Cast uses attribute "to" as int type.
  return 6;
}

bool CastOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                      const logging::Logger& logger) const {
  NodeAttrHelper helper(node);
  // Check cast output type.
  const auto to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED);
  if (!IsSupportedDataType(to_type)) {
    LOGS(logger, VERBOSE) << "Invalid cast to type " << to_type
                          << " . Current WebNN only support cast to float32 or float16.";
    return false;
  }

  return true;
}

void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<CastOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
