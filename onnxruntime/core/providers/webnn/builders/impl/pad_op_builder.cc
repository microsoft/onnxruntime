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
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class PadOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.

// ONNX mode to WebNN mode mapping.
const InlinedHashMap<std::string, std::string> supported_mode = {
    {"constant", "constant"},
    {"reflect", "reflection"},
    {"edge", "edge"},
};

// Skip for pads, constant value, and axes.
void PadOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  for (size_t i = 1; i < node.InputDefs().size(); i++) {
    model_builder.AddInitializerToSkip(node.InputDefs()[i]->Name());
    model_builder.AddInputToSkip(node.InputDefs()[i]->Name());
  }
}

bool clampNegativeValues(const std::vector<int64_t>& padding,
                         /*out*/ std::vector<uint32_t>& clamped_padding) {
  if (std::any_of(padding.begin(), padding.end(), [](auto pad) { return pad < 0; })) {
    std::transform(padding.begin(), padding.end(), std::back_inserter(clamped_padding),
                   [](int64_t x) -> uint32_t { return SafeInt<uint32_t>(std::max(x, 0LL)); });
    return true;  // Values were coerced.
  } else {
    std::transform(padding.begin(), padding.end(), std::back_inserter(clamped_padding),
                   [](int64_t x) -> uint32_t { return SafeInt<uint32_t>(x); });
  }
  return false;
}

Status PadOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                           const Node& node,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers = model_builder.GetInitializerTensors();
  ORT_RETURN_IF(input_defs.size() < 1, "Pad has no inputs");
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");

  emscripten::val options = emscripten::val::object();

  NodeAttrHelper helper(node);
  const auto pad_mode = helper.Get("mode", std::string("constant"));
  std::vector<int64_t> start_padding;
  std::vector<int64_t> end_padding;
  ORT_RETURN_IF(supported_mode.find(pad_mode) == supported_mode.end(), "WebNN does not support mode", pad_mode);
  const auto webnn_mode = supported_mode.find(pad_mode)->second;
  options.set("mode", emscripten::val(webnn_mode));

  const auto opset = node.SinceVersion();
  // From opset 11, pads, constant value and axes are inputs.
  if (opset >= 11) {
    ORT_RETURN_IF(input_defs.size() < 2, "Pads is required at opset ", opset);
    std::vector<int64_t> pads;
    const auto& pads_tensor = *initializers.at(input_defs[1]->Name());
    ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(pads_tensor, pads, logger), "Error while read pads tensor");

    // Constant value and axes are optional. Make sure they are not empty.
    if (!GetTensorName(input_defs, 2).empty()) {
      const auto value_tensor = *initializers.at(input_defs[2]->Name());
      emscripten::val value = emscripten::val::object();
      ORT_RETURN_IF_NOT(ReadScalarTensorData(value_tensor, value, logger), "Cannot read constant value");
      options.set("value", value);
    }

    if (!GetTensorName(input_defs, 3).empty()) {
      const auto input_rank = input_shape.size();
      std::vector<int64_t> axes;
      const auto& axes_tensor = *initializers.at(input_defs[3]->Name());
      ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(axes_tensor, axes, logger), "Error while read axes tensor");
      std::vector<size_t> axes_index;
      std::transform(
          axes.begin(), axes.end(), std::back_inserter(axes_index),
          [input_rank](int64_t axis) -> int32_t { return SafeInt<int32_t>(HandleNegativeAxis(axis, input_rank)); });
      start_padding.resize(input_rank, 0);
      end_padding.resize(input_rank, 0);
      for (size_t i = 0; i < axes_index.size(); i++) {
        size_t index = axes_index[i];
        start_padding[index] = pads[i];
        end_padding[index] = pads[i + pads.size() / 2];
      }
    } else {
      start_padding.assign(pads.begin(), pads.begin() + pads.size() / 2);
      end_padding.assign(pads.begin() + pads.size() / 2, pads.end());
    }
  } else {
    // Before opset 11, pads, constant value are attributes.
    ORT_RETURN_IF_NOT(helper.HasAttr("pads"), "Pads is required as attribute in opset ", opset);
    const auto pads = helper.Get("pads", std::vector<int>());
    const auto value = helper.Get("value", 0.0f);
    start_padding.assign(pads.begin(), pads.begin() + pads.size() / 2);
    end_padding.assign(pads.begin() + pads.size() / 2, pads.end());
    options.set("value", value);
  }

  // Padding of WebNN cannot be negative.
  std::vector<uint32_t> webnn_start_padding;
  std::vector<uint32_t> webnn_end_padding;
  bool negative_padding = clampNegativeValues(start_padding, webnn_start_padding);
  negative_padding |= clampNegativeValues(end_padding, webnn_end_padding);
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("pad", input,
                                                                            emscripten::val::array(webnn_start_padding),
                                                                            emscripten::val::array(webnn_end_padding),
                                                                            options);
  // Handle the negative padding with slice.
  if (negative_padding) {
    std::vector<uint32_t> starts;
    std::vector<uint32_t> sizes;
    for (size_t i = 0; i < start_padding.size(); i++) {
      starts.push_back(start_padding[i] >= 0 ? SafeInt<uint32_t>(0) : SafeInt<uint32_t>(-start_padding[i]));
      sizes.push_back(SafeInt<uint32_t>(input_shape[i] + start_padding[i] + end_padding[i]));
    }
    output = model_builder.GetBuilder().call<emscripten::val>("slice", output,
                                                              emscripten::val::array(starts),
                                                              emscripten::val::array(sizes));
  }
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool PadOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                     const Node& node,
                                     const WebnnDeviceType /* device_type */,
                                     const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto opset = node.SinceVersion();

  NodeAttrHelper helper(node);
  const auto pad_mode = helper.Get("mode", "constant");
  if (supported_mode.find(pad_mode) == supported_mode.end()) {
    LOGS(logger, VERBOSE) << op_type << " WebNN does not support mode " << pad_mode;
    return false;
  }

  if (input_defs.size() < 1) {
    LOGS(logger, VERBOSE) << op_type << " requires at least one input (data)";
    return false;
  }

  if (opset >= 11) {
    if (input_defs.size() < 2) {
      LOGS(logger, VERBOSE) << op_type << " at opset " << opset << " requires at least two inputs (data and pads)";
      return false;
    }
    for (size_t i = 1; i < input_defs.size(); i++) {
      // Optional tensors (constant_value, axes) can be indicated by an empty name, just ignore it.
      const std::string input_name = GetTensorName(input_defs, i);
      if (!input_name.empty() && !Contains(initializers, input_name)) {
        LOGS(logger, VERBOSE) << "Input [" << input_name << "] must be known as initializer";
        return false;
      }
    }
  }

  return true;
}  // namespace webnn

void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<PadOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
