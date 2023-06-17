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

class NormalizationOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

Status NormalizationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                     const Node& node,
                                                     const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  ORT_RETURN_IF_NOT(input_defs.size() >= 2, "LayerNormalization requires at least two inputs.");

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const auto rank = input_shape.size();

  emscripten::val options = emscripten::val::object();

  std::vector<int64_t> scale_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], scale_shape, logger), "Cannot get scale shape");
  const auto scale_size = scale_shape.size();
  ORT_RETURN_IF_NOT(scale_size >= 1 && scale_size <= rank, "The scale size should be less than or equal to input size.");

  if (scale_size < rank) {
    // Enlarge new shape to input.rank, right aligned with leading ones
    scale_shape.insert(scale_shape.begin(), rank - scale_size, 1);
    std::vector<int32_t> new_scale_shape;
    std::transform(scale_shape.cbegin(), scale_shape.cend(),
                   std::back_inserter(new_scale_shape),
                   [](int64_t dim) -> int32_t { return SafeInt<int32_t>(dim); });
    emscripten::val reshape_scale = model_builder.GetOperand(input_defs[1]->Name());
    emscripten::val reshape_output_scale =
        model_builder.GetBuilder().call<emscripten::val>("reshape", reshape_scale, emscripten::val::array(new_scale_shape));
    options.set("scale", reshape_output_scale);
  } else {
    options.set("scale", model_builder.GetOperand(input_defs[1]->Name()));
  }

  if (input_defs.size() == 3) {
    // Inputs contain optional bias
    std::vector<int64_t> bias_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[2], bias_shape, logger), "Cannot get bias shape");
    const auto bias_size = bias_shape.size();
    ORT_RETURN_IF_NOT(bias_size >= 1 && bias_size <= rank, "The bias size should be less than or equal to input size.");

    if (bias_size < rank) {
      // Enlarge new shape to input.rank, right aligned with leading ones
      bias_shape.insert(bias_shape.begin(), rank - bias_size, 1);
      std::vector<int32_t> new_bias_shape;
      std::transform(bias_shape.cbegin(), bias_shape.cend(),
                     std::back_inserter(new_bias_shape),
                     [](int64_t dim) -> int32_t { return SafeInt<int32_t>(dim); });
      emscripten::val reshape_bias = model_builder.GetOperand(input_defs[2]->Name());
      emscripten::val reshape_output_bias =
          model_builder.GetBuilder().call<emscripten::val>("reshape", reshape_bias, emscripten::val::array(new_bias_shape));
      options.set("bias", reshape_output_bias);
    } else {
      options.set("bias", model_builder.GetOperand(input_defs[2]->Name()));
    }
  }

  NodeAttrHelper helper(node);
  options.set("epsilon", helper.Get("epsilon", 1e-05f));

  int64_t axis = helper.Get("axis", -1);
  axis = HandleNegativeAxis(axis, rank);
  std::vector<int32_t> axes{static_cast<int32_t>(axis)};
  options.set("axes", emscripten::val::array(axes));

  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("meanVarianceNormalization", input, options);
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));

  return Status::OK();
}

// Operator support related.

bool NormalizationOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                               const Node& node,
                                               const WebnnDeviceType /* device_type */,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 2) {
    LOGS(logger, VERBOSE) << "LayerNormalization requires at least two inputs.";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get input shape.";
    return false;
  }
  const auto rank = input_shape.size();

  NodeAttrHelper helper(node);
  int64_t axis = helper.Get("axis", -1);
  axis = HandleNegativeAxis(axis, rank);

  const auto& scale_name = input_defs[1]->Name();
  if (!Contains(initializers, scale_name)) {
    LOGS(logger, VERBOSE) << "The scale must be a constant initializer.";
    return false;
  }

  if (input_defs.size() == 3) {
    // Inputs contain optional bias
    const auto& bias_name = input_defs[2]->Name();
    if (!Contains(initializers, bias_name)) {
      LOGS(logger, VERBOSE) << "The bias must be a constant initializer.";
      return false;
    }
  }

  const auto& output_defs = node.OutputDefs();
  if (output_defs.size() != 1) {
    LOGS(logger, VERBOSE) << "MeanVarianceNormalization output count must be one.";
    return false;
  }

  return true;
}

void CreateNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<NormalizationOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
