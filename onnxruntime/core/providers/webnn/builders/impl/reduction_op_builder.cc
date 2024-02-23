// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "core/optimizer/initializer.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class ReductionOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.
void ReductionOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() > 1) {
    model_builder.AddInitializerToSkip(input_defs[1]->Name());  // axes
  }
}

// Add operator related.

Status ReductionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const Node& node,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const auto input_rank = input_shape.size();

  NodeAttrHelper helper(node);
  const auto keep_dims = helper.Get("keepdims", 1);
  emscripten::val options = emscripten::val::object();
  options.set("keepDimensions", keep_dims == 1);
  std::vector<int32_t> axes_data;

  emscripten::val output = emscripten::val::object();

  const auto opset = node.SinceVersion();
  const auto& op_type = node.OpType();
  if (opset >= 18 || (op_type == "ReduceSum" && opset >= 13)) {
    // 'axes' is an optional input.
    const auto noop_with_empty_axes = helper.Get("noop_with_empty_axes", 0);
    if (input_defs.size() > 1) {
      // Optional input axes is provided, use axes initializer data.
      const auto& initializers(model_builder.GetInitializerTensors());
      const auto& axes_tensor = *initializers.at(input_defs[1]->Name());
      Initializer axes_initializer(axes_tensor);
      const auto axes_data_span = axes_initializer.DataAsSpan<int64_t>();
      std::transform(
          axes_data_span.begin(), axes_data_span.end(), std::back_inserter(axes_data),
          [input_rank](int64_t axis) -> int32_t { return SafeInt<int32_t>(HandleNegativeAxis(axis, input_rank)); });
    } else {
      if (noop_with_empty_axes) {
        // When axes is empty and this attribute is set to true, input tensor will not be reduced.
        output = input;
        model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
        return Status::OK();
      }
    }
  } else {
    if (helper.HasAttr("axes")) {
      auto axes = helper.Get("axes", std::vector<int64_t>{});
      std::transform(
          axes.begin(), axes.end(), std::back_inserter(axes_data),
          [input_rank](int64_t axis) -> int32_t { return SafeInt<int32_t>(HandleNegativeAxis(axis, input_rank)); });
    }
  }
  if (axes_data.size() > 0) {
    options.set("axes", emscripten::val::array(axes_data));
  }

  if (op_type == "ReduceL1") {
    output = model_builder.GetBuilder().call<emscripten::val>("reduceL1", input, options);
  } else if (op_type == "ReduceL2") {
    output = model_builder.GetBuilder().call<emscripten::val>("reduceL2", input, options);
  } else if (op_type == "ReduceLogSum") {
    output = model_builder.GetBuilder().call<emscripten::val>("reduceLogSum", input, options);
  } else if (op_type == "ReduceLogSumExp") {
    output = model_builder.GetBuilder().call<emscripten::val>("reduceLogSumExp", input, options);
  } else if (op_type == "ReduceMax") {
    output = model_builder.GetBuilder().call<emscripten::val>("reduceMax", input, options);
  } else if (op_type == "ReduceMean") {
    output = model_builder.GetBuilder().call<emscripten::val>("reduceMean", input, options);
  } else if (op_type == "ReduceMin") {
    output = model_builder.GetBuilder().call<emscripten::val>("reduceMin", input, options);
  } else if (op_type == "ReduceProd") {
    output = model_builder.GetBuilder().call<emscripten::val>("reduceProduct", input, options);
  } else if (op_type == "ReduceSum") {
    output = model_builder.GetBuilder().call<emscripten::val>("reduceSum", input, options);
  } else if (op_type == "ReduceSumSquare") {
    output = model_builder.GetBuilder().call<emscripten::val>("reduceSumSquare", input, options);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ReductionOpBuilder, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool ReductionOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                           const Node& node,
                                           const WebnnDeviceType /* device_type */,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const auto& op_type = node.OpType();
  const std::string axes_name = GetTensorName(input_defs, 1);
  // If the optional input 'axes' is provided, it must be an initializer.
  if (!axes_name.empty() && !Contains(initializers, axes_name)) {
    LOGS(logger, VERBOSE) << "Input axes of " << op_type << " must be a constant";
    return false;
  }

  return true;
}

void CreateReductionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "ReduceL1",
          "ReduceL2",
          "ReduceLogSum",
          "ReduceLogSumExp",
          "ReduceMax",
          "ReduceMean",
          "ReduceMin",
          "ReduceProd",
          "ReduceSum",
          "ReduceSumSquare",
      };

  op_registrations.builders.push_back(std::make_unique<ReductionOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
