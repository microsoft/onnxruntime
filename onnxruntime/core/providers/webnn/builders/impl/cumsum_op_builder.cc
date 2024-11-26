// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class CumSumOpBuilder : public BaseOpBuilder {
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

void CumSumOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Skip axis.
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

Status CumSumOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const auto input_rank = input_shape.size();

  const auto& initializers = model_builder.GetInitializerTensors();
  const std::string axis_name = GetTensorName(input_defs, 1);
  const auto axis_tensor = *initializers.at(axis_name);
  emscripten::val axis = emscripten::val::undefined();
  ORT_RETURN_IF_NOT(ReadScalarTensorData(axis_tensor, axis, logger), "Cannot get axis value");
  int64_t webnn_axis = HandleNegativeAxis(axis.as<int64_t>(), input_rank);

  NodeAttrHelper helper(node);
  const auto exclusive = helper.Get("exclusive", 0);
  const auto reverse = helper.Get("reverse", 0);

  emscripten::val options = emscripten::val::object();
  options.set("exclusive", exclusive == 1);
  options.set("reversed", reverse == 1);
  options.set("label", node.Name());

  emscripten::val output = emscripten::val::object();
  output = model_builder.GetBuilder().call<emscripten::val>("cumulativeSum", input, gsl::narrow<uint32_t>(webnn_axis),
                                                            options);
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool CumSumOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                        const Node& node,
                                        WebnnDeviceType /* device_type */,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const std::string axis_name = GetTensorName(input_defs, 1);
  // Inputs contain optional 'axis' input.
  if (!Contains(initializers, axis_name)) {
    LOGS(logger, VERBOSE) << "The axis must be a constant initializer.";
    return false;
  }

  return true;
}

void CreateCumSumOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<CumSumOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
