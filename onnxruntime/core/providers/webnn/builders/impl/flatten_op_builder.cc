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

class FlattenOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

// Add operator related.

Status FlattenOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF(input_defs.size() < 1, "Flatten has no input tensor");
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "FlattenOpBuilder::AddToModelBuilderImpl, cannot get input shape");
  }
  int64_t rank = input_shape.size();
  NodeAttrHelper helper(node);
  int64_t axis = helper.Get("axis", 1);
  ORT_ENFORCE(axis >= -rank && axis <= rank, "axis ", axis,
              " is not in valid range [-", rank, ",", rank, "]");
  if (axis < 0) {
    axis += rank;
  }

  // Use WebNN's reshape to implement Flatten.
  int64_t num_pre_axis_elements = std::accumulate(
      input_shape.begin(), input_shape.begin() + static_cast<int32_t>(axis), 1, std::multiplies<int64_t>());
  int64_t num_post_axis_elements = std::accumulate(
      input_shape.begin() + static_cast<int32_t>(axis), input_shape.end(), 1, std::multiplies<int64_t>());

  std::vector<uint32_t> new_shape = {SafeInt<uint32_t>(num_pre_axis_elements),
                                     SafeInt<uint32_t>(num_post_axis_elements)};

  emscripten::val inputs = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", inputs, emscripten::val::array(new_shape));

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<FlattenOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
