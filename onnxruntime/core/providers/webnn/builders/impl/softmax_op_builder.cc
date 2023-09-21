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

class SoftmaxOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

Status SoftmaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
  emscripten::val output = emscripten::val::object();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const auto input_size = input_shape.size();
  // WebNN Softmax only support 2d input shape, reshape input to 2d.
  if (input_size != 2) {
    int32_t new_shape_0 = SafeInt<int32_t>(input_shape.data()[0]);
    for (size_t i = 1; i < input_size - 1; i++) {
      new_shape_0 *= input_shape.data()[i];
    }
    emscripten::val new_shape = emscripten::val::array();
    new_shape.call<void>("push", new_shape_0);
    new_shape.call<void>("push", static_cast<int32_t>(input_shape.back()));
    input = model_builder.GetBuilder().call<emscripten::val>("reshape", input, new_shape);
  }
  output = model_builder.GetBuilder().call<emscripten::val>("softmax", input);
  // Reshape output to the same shape of input.
  if (input_size != 2) {
    emscripten::val new_shape = emscripten::val::array();
    for (size_t i = 0; i < input_size; i++) {
      new_shape.call<void>("push", static_cast<int32_t>(input_shape[i]));
    }
    output = model_builder.GetBuilder().call<emscripten::val>("reshape", output, new_shape);
  }
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool SoftmaxOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */,
                                         const Node& node,
                                         const WebnnDeviceType /* device_type */,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;
  const auto input_size = input_shape.size();
  if (input_size < 2) {
    LOGS(logger, VERBOSE) << "SoftMax only support input size >= 2d shape, input is "
                          << input_size << "d shape";
    return false;
  }
  NodeAttrHelper helper(node);
  const int32_t axis = helper.Get("axis", 1);
  // WebNN softmax only support input axis 1
  if (axis != 1 && axis != -1) {
    LOGS(logger, VERBOSE) << "SoftMax only support axis 1 or -1, input axis: " << axis;
    return false;
  }

  return true;
}

void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SoftmaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
