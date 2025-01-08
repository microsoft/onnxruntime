// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ConcatOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
  bool HasSupportedInputsImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
};

// Add operator related.

Status ConcatOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ConcatOpBuilder::AddToModelBuilderImpl, cannot get input shape");
  }
  auto rank = input_shape.size();
  NodeAttrHelper helper(node);
  uint32_t axis = static_cast<uint32_t>(HandleNegativeAxis(helper.Get("axis", 1), rank));

  std::vector<emscripten::val> inputs;
  for (const auto* input : input_defs) {
    inputs.push_back(model_builder.GetOperand(input->Name()));
  }

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  emscripten::val output =
      model_builder.GetBuilder().call<emscripten::val>("concat", emscripten::val::array(inputs), axis, options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

bool ConcatOpBuilder::HasSupportedInputsImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                             const emscripten::val& wnn_limits, const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  int32_t input0_type;

  if (!GetType(*input_defs[0], input0_type, logger))
    return false;

  for (size_t i = 1; i < input_defs.size(); i++) {
    int32_t input_type;
    if (!GetType(*input_defs[i], input_type, logger)) {
      return false;
    }

    std::array<int32_t, 2> input_types{input0_type, input_type};
    if (!AreInputDataTypesSame(op_type, input_types, logger)) {
      return false;
    }
  }

  return IsDataTypeSupportedByOp(op_type, input0_type, wnn_limits, "inputs", "inputs", logger);
}

void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConcatOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
