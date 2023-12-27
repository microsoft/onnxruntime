// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <math.h>

#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ResizeOpBuilder : public BaseOpBuilder {
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

  // Resize opset 10- is very different than Resize opset 11+, with many key attributes missing.
  // We only support Resize opset 11+ here.
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 11; }
};

// Helper functions
bool GetResizeScales(const InitializedTensorSet& initializers,
                     const Node& node, std::vector<float>& scales,
                     const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 3)
    return false;

  const auto& scales_tensor = *initializers.at(input_defs[2]->Name());
  if (scales_tensor.dims_size() != 1 || scales_tensor.dims()[0] != 4)
    return false;

  std::vector<uint8_t> unpacked_tensor;
  auto status = onnxruntime::utils::UnpackInitializerData(scales_tensor, unpacked_tensor);
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Error while unpacking scales_tensor: " << status.ErrorMessage();
    return false;
  }
  const float* scales_data = reinterpret_cast<const float*>(unpacked_tensor.data());
  scales = std::vector<float>{scales_data, scales_data + 4};
  return true;
}

bool GetResizeOutputSizes(const InitializedTensorSet& initializers,
                          const Node& node, std::vector<int64_t>& sizes,
                          const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 4)
    return false;

  const auto& sizes_tensor = *initializers.at(input_defs[3]->Name());
  if (sizes_tensor.dims_size() != 1 || sizes_tensor.dims()[0] != 4)
    return false;

  std::vector<uint8_t> unpacked_tensor;
  auto status = onnxruntime::utils::UnpackInitializerData(sizes_tensor, unpacked_tensor);
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Error while unpacking sizes_tensor: " << status.ErrorMessage();
    return false;
  }
  const int64_t* sizes_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  sizes = std::vector<int64_t>{sizes_data, sizes_data + 4};
  return true;
}

// Add operator related.

void ResizeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // We don't really use ROI here, so add it to skipped list if it's an initializer tensor.
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());  // ROI
  model_builder.AddInputToSkip(node.InputDefs()[1]->Name());        // ROI

  // We will still add scales to the skipped list even sizes are present,
  // since there is no use of it, we will not process it later.
  model_builder.AddInitializerToSkip(node.InputDefs()[2]->Name());  // scales
  model_builder.AddInputToSkip(node.InputDefs()[2]->Name());        // scales

  if (node.InputDefs().size() > 3) {
    model_builder.AddInitializerToSkip(node.InputDefs()[3]->Name());  // sizes
    model_builder.AddInputToSkip(node.InputDefs()[3]->Name());        // sizes
  }
}

Status ResizeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  emscripten::val options = emscripten::val::object();
  NodeAttrHelper helper(node);
  const auto mode = helper.Get("mode", "nearest");
  if (mode == "linear") {
    options.set("mode", emscripten::val("linear"));
  } else {  // we already checked the mode must be NN or Bilinear in IsOpSupportedImpl.
    options.set("mode", emscripten::val("nearest-neighbor"));
  }

  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());

  std::vector<float> scales;
  std::vector<int32_t> sizes;
  std::vector<float> scales_hw;
  std::vector<int32_t> sizes_hw;
  std::vector<int32_t> axes;
  const bool is_nhwc = model_builder.GetPreferredLayout() == DataLayout::NHWC;
  if (input_defs.size() == 3) {  // Use scales.
    ORT_RETURN_IF_NOT(GetResizeScales(initializers, node, scales, logger), "Error getting resize scales");
    if (is_nhwc) {
      scales_hw = {scales[1], scales[2]};
    } else {
      scales_hw = {scales[2], scales[3]};
    }
    options.set("scales", emscripten::val::array(scales_hw));
  } else {  // We already checked number of inputs in IsOpSupportedImpl.
    std::vector<int64_t> output_sizes;
    ORT_RETURN_IF_NOT(GetResizeOutputSizes(initializers, node, output_sizes, logger),
                      "Error getting resize output_sizes");
    std::transform(output_sizes.cbegin(), output_sizes.cend(),
                   std::back_inserter(sizes),
                   [](int64_t dim) -> int32_t { return SafeInt<int32_t>(dim); });
    if (is_nhwc) {
      sizes_hw = {sizes[1], sizes[2]};
    } else {
      sizes_hw = {sizes[2], sizes[3]};
    }
    options.set("sizes", emscripten::val::array(sizes_hw));
  }

  if (is_nhwc) {
    axes = {1, 2};
  } else {
    axes = {2, 3};
  }
  options.set("axes", emscripten::val::array(axes));

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("resample2d", input, options);
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool ResizeOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                        const Node& node,
                                        const WebnnDeviceType /* device_type */,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS(logger, VERBOSE) << "Resize only support 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  {  // Check attributes.
    NodeAttrHelper helper(node);
    const auto mode = helper.Get("mode", "nearest");
    bool is_linear_resize = mode == "linear";
    bool is_nearest_resize = mode == "nearest";
    if (!is_linear_resize && !is_nearest_resize) {
      LOGS(logger, VERBOSE) << "Resize unsupported input mode, " << mode;
      return false;
    }

    const auto exclude_outside = helper.Get("exclude_outside", 0);
    if (exclude_outside != 0) {
      LOGS(logger, VERBOSE) << "Resize does not support exclude_outside for now";
      return false;
    }
  }

  {  // scales and sizes (if present) must be initializers.
    if (input_defs.size() < 3) {
      LOGS(logger, VERBOSE) << "Input scales or sizes of Resize must be known";
      return false;
    }

    // scales
    if (input_defs.size() == 3 && !Contains(initializers, input_defs[2]->Name())) {
      LOGS(logger, VERBOSE) << "Input scales of Resize must be known";
      return false;
    }

    // sizes
    if (input_defs.size() > 3 && !Contains(initializers, input_defs[3]->Name())) {
      LOGS(logger, VERBOSE) << "Input sizes of Resize must be known";
      return false;
    }

    const bool is_nhwc = node.Domain() == kMSInternalNHWCDomain;
    // We want to check if the scales or sizes are not trying to resize on N/C channels here.
    if (input_defs.size() == 3) {  // We are using scales.
      std::vector<float> scales;
      if (!GetResizeScales(initializers, node, scales, logger))
        return false;

      float scale_n = scales[0];
      float scale_c = is_nhwc ? scales[3] : scales[1];
      if (scale_n != 1.0f || scale_c != 1.0f) {
        LOGS(logger, VERBOSE) << "Scales of N/C channel should be 1"
                              << "Resize of N/C channels are not supported"
                              << ", scale_n, " << scale_n << ", scale_c, " << scale_c;
        return false;
      }

      // For now we only support upscale, so the scale_h and scale_w should be an integer >= 1.
      // TODO support ResizeBilinear.
      float scale_h = is_nhwc ? scales[1] : scales[2];
      float scale_w = is_nhwc ? scales[2] : scales[3];

      // Onnx spec requires scale to be a positive float, so we are not checking that here.
      if (roundf(scale_h) != scale_h) {
        LOGS(logger, VERBOSE) << "Resize: scale_h: " << scale_h << " is not a whole number";
        return false;
      }

      if (roundf(scale_w) != scale_w) {
        LOGS(logger, VERBOSE) << "Resize: scale_w: " << scale_w << " is not a whole number";
        return false;
      }
    } else {
      // We are using sizes.
      std::vector<int64_t> output_sizes;
      if (!GetResizeOutputSizes(initializers, node, output_sizes, logger))
        return false;

      auto output_size_n = output_sizes[0];
      const int c_idx = is_nhwc ? 3 : 1;
      if (output_size_n != input_shape[0] || output_sizes[c_idx] != input_shape[c_idx]) {
        LOGS(logger, VERBOSE) << "Output sizes of N/C chanel should match the input sizes, "
                              << "Resize of N/C channels are not supported"
                              << ", input_size_n, " << input_shape[0] << ", output_size_n, " << output_size_n
                              << ". input_size_c, " << input_shape[c_idx] << ", output_size_c, " << output_sizes[c_idx];
        return false;
      }
    }
  }

  return true;
}

void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ResizeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
