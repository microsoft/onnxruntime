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
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const logging::Logger& logger) const override;
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

  emscripten::val inputs = emscripten::val::array();
  for (const auto* input : node.InputDefs()) {
    LOGS(logger, VERBOSE) << "input name " << input->Name();
    inputs.call<void>("push", model_builder.GetOperand(input->Name()));
  }

  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("concat", inputs, axis);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool ConcatOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                        const logging::Logger& logger) const {
  std::vector<int64_t> input_shape;
  const auto& input_defs(node.InputDefs());
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4 || input_size == 0) {
    LOGS_DEFAULT(VERBOSE) << "Concat only supports up to 1-4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConcatOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
