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

class SqueezeUnsqueezeOpBuilder : public BaseOpBuilder {
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
void SqueezeUnsqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Squeeze/Unsqueeze opset 13 uses input 1 as axes, add it to initializer skip list.
  const auto& input_defs = node.InputDefs();
  if (node.SinceVersion() >= 13 && input_defs.size() > 1) {
    model_builder.AddInitializerToSkip(input_defs[1]->Name());  // "axes"
    model_builder.AddInputToSkip(input_defs[1]->Name());
  }
}

// Add operator related.

Status SqueezeUnsqueezeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                        const Node& node,
                                                        const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const auto input_rank = input_shape.size();

  emscripten::val options = emscripten::val::object();
  std::vector<int32_t> axes_data;

  if (node.SinceVersion() >= 13 && input_defs.size() > 1) {
    // Input axes is provided, use axes initializer data.
    const auto& initializers = model_builder.GetInitializerTensors();
    const auto& axes_tensor = *initializers.at(input_defs[1]->Name());
    Initializer axes_initializer(axes_tensor);
    const auto axes_data_span = axes_initializer.DataAsSpan<int64_t>();
    const auto output_rank = input_rank + axes_data_span.size();
    std::transform(
        axes_data_span.begin(), axes_data_span.end(), std::back_inserter(axes_data),
        [output_rank](int64_t axis) -> int32_t { return SafeInt<int32_t>(HandleNegativeAxis(axis, output_rank)); });
  } else {
    NodeAttrHelper helper(node);
    if (helper.HasAttr("axes")) {
      auto axes = helper.Get("axes", std::vector<int64_t>{});
      const auto output_rank = input_rank + axes.size();
      std::transform(
          axes.begin(), axes.end(), std::back_inserter(axes_data),
          [output_rank](int64_t axis) -> int32_t { return SafeInt<int32_t>(HandleNegativeAxis(axis, output_rank)); });
    }
  }

  if (axes_data.size() > 0) {
    options.set("axes", emscripten::val::array(axes_data));
  }

  emscripten::val output = emscripten::val::undefined();
  if (op_type == "Squeeze") {
    output = model_builder.GetBuilder().call<emscripten::val>("squeeze", input, options);
  } else if (op_type == "Unsqueeze") {
    output = model_builder.GetBuilder().call<emscripten::val>("unsqueeze", input, options);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "SqueezeUnsqueezeOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool SqueezeUnsqueezeOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                                  const Node& node,
                                                  const WebnnDeviceType /* device_type */,
                                                  const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  if (input_defs.size() < 1) {
    LOGS(logger, ERROR) << op_type << " has no input tensor";
    return false;
  }

  // Squeeze/Unsqueeze opset 13 uses input 1 as axes, it needs to be an initializer.
  if (node.SinceVersion() >= 13) {
    if (input_defs.size() > 1) {
      const auto& axes_name = input_defs[1]->Name();
      if (!Contains(initializers, axes_name)) {
        LOGS(logger, ERROR) << "Input axes of " << op_type << " is not present and constant";
        return false;
      }
    } else if (op_type == "Unsqueeze") {
      // The axes are optional for Squeeze, but not Unsqueeze.
      LOGS(logger, ERROR) << "Input axes of Unsqueeze must be provided";
      return false;
    }
  }

  return true;
}

void CreateSqueezeUnsqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Squeeze",
          "Unsqueeze",
      };

  op_registrations.builders.push_back(std::make_unique<SqueezeUnsqueezeOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
