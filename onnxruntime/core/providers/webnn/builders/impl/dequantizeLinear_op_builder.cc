// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class DequantizeLinearOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

Status DequantizeLinearOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                        const Node& node,
                                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val scale = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val zero_point = emscripten::val::null();
  if (input_defs.size() == 3) {
    zero_point = model_builder.GetOperand(node.InputDefs()[2]->Name());
  } else {
    zero_point = model_builder.GetZeroConstant("uint8");
  }
  emscripten::val output;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> scale_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], scale_shape, logger), "Cannot get scale shape");
  NodeAttrHelper helper(node);
  int32_t axis = helper.Get("axis", 1);
  // axis is valid for input shape greater than 1D.
  if (input_shape.size() > 1) {
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_shape.size()));
  }
  // Insert ones before and after the axis dimension for broadcasting of 1D scale tensor.
  if (1 == scale_shape.size() && input_shape.size() > 1) {
    std::vector<int32_t> target_shape{static_cast<int>(input_shape[axis])};
    target_shape.insert(target_shape.begin(), axis, 1);
    target_shape.insert(target_shape.end(), input_shape.size() - axis - 1, 1);
    scale = model_builder.GetBuilder().call<emscripten::val>("reshape", scale, emscripten::val::array(target_shape));
    zero_point = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                                  zero_point, emscripten::val::array(target_shape));
  }
  output = model_builder.GetBuilder().call<emscripten::val>("dequantizeLinear", input, scale, zero_point);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));

  return Status::OK();
}

void CreateDequantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DequantizeLinearOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
