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

// All normalization are based on layout NCHW.
// TODO: add support for NHWC.
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

  std::vector<uint32_t> new_scale_shape;
  if (scale_size < rank) {
    if (op_type == "BatchNormalization") {
      scale_shape.insert(scale_shape.begin(), 1);
      scale_shape.insert(scale_shape.end(), rank - 2, 1);
    } else if (op_type == "LayerNormalization") {
      // Align right with leading ones.
      scale_shape.insert(scale_shape.begin(), rank - scale_size, 1);
    } else if (op_type == "InstanceNormalization") {
      // Insert ones before and after the channel dimension.
      scale_shape.insert(scale_shape.begin(), 1);
      ORT_RETURN_IF(scale_size != 1 || rank < 2,
                    "The scale size should be 1 and rank should be at least 2 for InstanceNorm.");
      scale_shape.insert(scale_shape.end(), rank - scale_size - 1, 1);
    } else if (op_type == "GroupNormalization") {
      // The input will be reshaped to 3D later. So just insert ones before the channel and after.
      scale_shape.insert(scale_shape.begin(), 1);
      scale_shape.insert(scale_shape.end(), 1);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported normalization op: ", op_type);
    }

    std::transform(scale_shape.cbegin(), scale_shape.cend(),
                   std::back_inserter(new_scale_shape),
                   [](int64_t dim) -> uint32_t { return SafeInt<uint32_t>(dim); });
    emscripten::val reshape_scale = model_builder.GetOperand(input_defs[1]->Name());
    emscripten::val reshape_output_scale =
        model_builder.GetBuilder().call<emscripten::val>("reshape", reshape_scale, emscripten::val::array(new_scale_shape));
    options.set("scale", reshape_output_scale);

    if (input_defs.size() >= 3 && !input_defs[2]->Name().empty()) {
      // Bias input exists, and bias's shape is the same as scale's shape.
      emscripten::val reshape_bias = model_builder.GetOperand(input_defs[2]->Name());
      emscripten::val reshape_output_bias =
          model_builder.GetBuilder().call<emscripten::val>("reshape", reshape_bias, emscripten::val::array(new_scale_shape));
      options.set("bias", reshape_output_bias);
    }
  } else {
    options.set("scale", model_builder.GetOperand(input_defs[1]->Name()));
    if (input_defs.size() >= 3 && !input_defs[2]->Name().empty()) {
      // Bias input exists, and bias's shape is the same as scale's shape.
      options.set("bias", model_builder.GetOperand(input_defs[2]->Name()));
    }
  }

  NodeAttrHelper helper(node);
  options.set("epsilon", helper.Get("epsilon", 1e-05f));

  emscripten::val output = emscripten::val::undefined();
  if (op_type == "BatchNormalization") {
    ORT_RETURN_IF_NOT(input_defs.size() == 5, "BatchNormalization requires five inputs.");
    emscripten::val mean = model_builder.GetOperand(input_defs[3]->Name());
    emscripten::val variance = model_builder.GetOperand(input_defs[4]->Name());
    // Enlarge 1-D mean and variance to new scale shape.
    emscripten::val reshape_mean =
        model_builder.GetBuilder().call<emscripten::val>("reshape", mean, emscripten::val::array(new_scale_shape));
    emscripten::val reshape_variance =
        model_builder.GetBuilder().call<emscripten::val>("reshape", variance, emscripten::val::array(new_scale_shape));

    std::vector<uint32_t> axes = {0};
    for (uint32_t i = 2; i < rank; i++) {
      axes.push_back(i);
    }

    options.set("axes", emscripten::val::array(axes));
    options.set("mean", reshape_mean);
    options.set("variance", reshape_variance);
    output = model_builder.GetBuilder().call<emscripten::val>("meanVarianceNormalization", input, options);
  } else if (op_type == "LayerNormalization") {
    int64_t axis = helper.Get("axis", -1);
    axis = HandleNegativeAxis(axis, rank);
    std::vector<uint32_t> axes(rank - SafeInt<uint32_t>(axis));
    std::iota(axes.begin(), axes.end(), axis);
    options.set("axes", emscripten::val::array(axes));
    output = model_builder.GetBuilder().call<emscripten::val>("meanVarianceNormalization", input, options);
  } else if (op_type == "InstanceNormalization") {
    std::vector<uint32_t> axes;
    for (uint32_t i = 2; i < rank; i++) {
      axes.emplace_back(i);
    }
    options.set("axes", emscripten::val::array(axes));
    output = model_builder.GetBuilder().call<emscripten::val>("meanVarianceNormalization", input, options);
  } else if (op_type == "GroupNormalization") {
    ORT_RETURN_IF_NOT(helper.HasAttr("num_groups"), "GroupNormalization num_group must be provided.");
    int32_t group_count = helper.Get("num_groups", -1);
    std::vector<uint32_t> orig_shape, new_shape;
    std::transform(input_shape.cbegin(), input_shape.cend(),
                   std::back_inserter(orig_shape),
                   [](int64_t dim) -> uint32_t { return SafeInt<uint32_t>(dim); });
    // Add N and Group.
    ORT_RETURN_IF_NOT(rank >= 2, "Input for GroupNormalization cannot be a scalar or 1D");
    new_shape.emplace_back(SafeInt<uint32_t>(input_shape[0]));
    new_shape.emplace_back(SafeInt<uint32_t>(group_count));

    ORT_RETURN_IF_NOT(group_count > 0 && input_shape[1] % group_count == 0,
                      "GroupNormalization num_group must be divisible by group.");
    new_shape.emplace_back(SafeInt<uint32_t>(std::reduce(input_shape.begin() + 2, input_shape.end(),
                                                         input_shape[1] / group_count, std::multiplies<int64_t>())));
    // Input will be reshaped to (N, group count, channels per group x D1 x D2 ... Dn) and recovered after normalization.
    options.set("axes", emscripten::val::array(std::vector<uint32_t>{2}));
    output = model_builder.GetBuilder().call<emscripten::val>("reshape", input, emscripten::val::array(new_shape));
    output = model_builder.GetBuilder().call<emscripten::val>("meanVarianceNormalization", output, options);
    output = model_builder.GetBuilder().call<emscripten::val>("reshape", output, emscripten::val::array(orig_shape));
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

  return true;
}

void CreateNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  constexpr static std::string_view op_types[] =
      {
          "BatchNormalization",
          "GroupNormalization",
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
