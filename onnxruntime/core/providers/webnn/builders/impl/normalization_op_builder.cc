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
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  ORT_RETURN_IF_NOT(input_defs.size() >= 2, op_type, " requires at least two inputs.");

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const auto rank = input_shape.size();

  emscripten::val options = emscripten::val::object();

  std::vector<int64_t> scale_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], scale_shape, logger), "Cannot get scale shape");
  const auto scale_size = scale_shape.size();
  // Except LayerNormalization, other normalization ops' scale input should be 1-D.
  if (op_type == "LayerNormalization") {
    ORT_RETURN_IF_NOT(scale_size >= 1 && scale_size <= rank,
                      "The scale size should be less than or equal to input size.");
  } else {
    ORT_RETURN_IF_NOT(scale_size == 1, "The scale size should be one.");
  }

  if (input_defs.size() >= 3 && !input_defs[2]->Name().empty()) {
    // Bias input exists, and bias's shape should be the same as scale's shape.
    std::vector<int64_t> bias_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[2], bias_shape, logger), "Cannot get bias shape");
    ORT_RETURN_IF_NOT(bias_shape == scale_shape, "The bias' shape should be equal to scale's shape.");
  }

  emscripten::val scale = model_builder.GetOperand(input_defs[1]->Name());
  options.set("scale", scale);

  if (input_defs.size() >= 3 && !input_defs[2]->Name().empty()) {
    // Bias input exists, and bias's shape is the same as scale's shape.
    emscripten::val bias = model_builder.GetOperand(input_defs[2]->Name());
    options.set("bias", bias);
  }

  NodeAttrHelper helper(node);
  options.set("epsilon", helper.Get("epsilon", 1e-05f));

  emscripten::val output = emscripten::val::undefined();
  if (op_type == "BatchNormalization") {
    ORT_RETURN_IF_NOT(input_defs.size() == 5, "BatchNormalization requires five inputs.");
    emscripten::val mean = model_builder.GetOperand(input_defs[3]->Name());
    emscripten::val variance = model_builder.GetOperand(input_defs[4]->Name());
    if (model_builder.GetPreferredLayout() == DataLayout::NHWC) {
      options.set("axis", rank - 1);
    }
    output = model_builder.GetBuilder().call<emscripten::val>("batchNormalization", input, mean, variance, options);
  } else if (op_type == "LayerNormalization") {
    int64_t axis = helper.Get("axis", -1);
    axis = HandleNegativeAxis(axis, rank);
    std::vector<uint32_t> axes(rank - SafeInt<uint32_t>(axis));
    if (model_builder.GetPreferredLayout() == DataLayout::NHWC && axis > 1) {
      std::iota(axes.begin(), axes.end(), axis - 1);
    } else {
      std::iota(axes.begin(), axes.end(), axis);
    }
    options.set("axes", emscripten::val::array(axes));
    output = model_builder.GetBuilder().call<emscripten::val>("layerNormalization", input, options);
  } else if (op_type == "InstanceNormalization") {
    if (model_builder.GetPreferredLayout() == DataLayout::NHWC) {
      options.set("layout", emscripten::val("nhwc"));
    }
    output = model_builder.GetBuilder().call<emscripten::val>("instanceNormalization", input, options);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported normalization op: ", op_type);
  }
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));

  return Status::OK();
}

// Operator support related.

bool NormalizationOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                               const Node& node,
                                               const WebnnDeviceType /* device_type */,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  NodeAttrHelper helper(node);

  if (input_defs.size() < 2) {
    LOGS(logger, VERBOSE) << op_type << " requires at least two inputs.";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get input shape.";
    return false;
  }

  const auto& output_defs = node.OutputDefs();
  if (output_defs.size() != 1) {
    LOGS(logger, VERBOSE) << op_type << " output count must be one.";
    return false;
  }

  if (op_type == "BatchNormalization" && helper.Get("training_mode", 0)) {
    LOGS(logger, VERBOSE) << "BatchNormalization with training_mode set to true is not supported.";
    return false;
  }

  if (op_type == "InstanceNormalization") {
    std::vector<int64_t> input_shape;
    if (!GetShape(*input_defs[0], input_shape, logger)) {
      LOGS(logger, VERBOSE) << "Cannot get input shape";
      return false;
    }
    const auto rank = input_shape.size();
    if (rank != 4) {
      LOGS(logger, VERBOSE) << "InstanceNormalization only supports 4D input.";
      return false;
    }
  }
  return true;
}

void CreateNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  constexpr static std::string_view op_types[] =
      {
          "BatchNormalization",
          "InstanceNormalization",
          "LayerNormalization",
      };

  op_registrations.builders.push_back(std::make_unique<NormalizationOpBuilder>());
  for (const auto& op_type : op_types) {
    op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
