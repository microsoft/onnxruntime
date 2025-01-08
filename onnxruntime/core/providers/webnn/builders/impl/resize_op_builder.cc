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
  // Allow roi and scales potentially being empty inputs that are ignored during processing.
  ResizeOpBuilder() : BaseOpBuilder(/*allow empty inputs*/ true) {}
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
bool GetResizeScalesAndAxes(const InitializedTensorSet& initializers,
                            const Node& node, std::vector<float>& scales,
                            std::vector<int64_t>& axes, const bool is_nhwc,
                            const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 3)
    return false;

  const bool has_axes = !axes.empty();
  const auto& scales_tensor = *initializers.at(input_defs[2]->Name());
  if (scales_tensor.dims_size() != 1) {
    LOGS(logger, ERROR) << "'scales' should be a 1D tensor.";
    return false;
  }

  // Number of elements of 'scales' tensor.
  const auto num_of_scales = scales_tensor.dims()[0];

  if (has_axes && num_of_scales != 2) {
    LOGS(logger, ERROR) << "When 'axes' is provided, 'scales' should have 2 elements.";
    return false;
  }

  if (!has_axes && num_of_scales != 4) {
    LOGS(logger, ERROR) << "When 'axes' is not provided, 'scales' should have 4 elements.";
    return false;
  }

  std::vector<uint8_t> unpacked_tensor;
  auto status = onnxruntime::utils::UnpackInitializerData(scales_tensor, unpacked_tensor);
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Error while unpacking scales_tensor: " << status.ErrorMessage();
    return false;
  }
  const float* scales_data = reinterpret_cast<const float*>(unpacked_tensor.data());

  if (has_axes) {
    // 'axes' is specified since opset 18+, 'scales' should have 2 elements.
    scales = std::vector<float>{scales_data, scales_data + 2};
  } else {
    // Before opset 18, 'scales' should have 4 elements.
    // Make sure 'scales' is not trying to scale on N/C channels here.
    std::vector<float> onnx_scales{scales_data, scales_data + 4};
    // 'scales' input has been transposed to NHWC layout if it is NHWC preferred layout.
    const float scale_n = onnx_scales[0];
    const float scale_c = is_nhwc ? onnx_scales[3] : onnx_scales[1];
    const float scale_h = is_nhwc ? onnx_scales[1] : onnx_scales[2];
    const float scale_w = is_nhwc ? onnx_scales[2] : onnx_scales[3];
    if (scale_n != 1.0f || scale_c != 1.0f) {
      LOGS(logger, VERBOSE) << "Scales of N/C channel should be 1"
                            << "Scales of N/C channels are not supported"
                            << ", scale_n, " << scale_n << ", scale_c, " << scale_c;
      return false;
    }

    scales = {scale_h, scale_w};
    axes = {2, 3};
  }

  if (is_nhwc) {
    // For NHWC preferred layout, we need to convert axes from NCHW to NHWC.
    axes = convertAxesFromNCHWtoNHWC(axes);
  }

  return true;
}

bool GetResizeSizesAndAxes(const InitializedTensorSet& initializers,
                           const Node& node, std::vector<int64_t>& sizes,
                           std::vector<int64_t>& axes, const bool is_nhwc,
                           const gsl::span<int64_t>& input_shape,
                           const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 4)
    return false;

  const bool has_axes = !axes.empty();
  const auto& sizes_tensor = *initializers.at(input_defs[3]->Name());
  if (sizes_tensor.dims_size() != 1) {
    LOGS(logger, ERROR) << "'sizes' should be a 1D tensor.";
    return false;
  }

  // Number of elements of sizes tensor.
  const auto num_of_sizes = sizes_tensor.dims()[0];
  if (has_axes && num_of_sizes != 2) {
    LOGS(logger, ERROR) << "When 'axes' is provided, 'sizes' should have 2 elements.";
    return false;
  }

  if (!has_axes && num_of_sizes != 4) {
    LOGS(logger, ERROR) << "When 'axes' is not provided, 'sizes' should have 4 elements.";
    return false;
  }

  std::vector<uint8_t> unpacked_tensor;
  auto status = onnxruntime::utils::UnpackInitializerData(sizes_tensor, unpacked_tensor);
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Error while unpacking sizes_tensor: " << status.ErrorMessage();
    return false;
  }
  const int64_t* sizes_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());

  if (has_axes) {
    // 'axes' is specified since opset 18+, 'sizes' should have 2 elements.
    sizes = std::vector<int64_t>{sizes_data, sizes_data + 2};
  } else {
    // Before opset 18, 'sizes' should have 4 elements.
    // Make sure 'sizes' is not trying to resize on N/C channels here.
    std::vector<int64_t> onnx_sizes{sizes_data, sizes_data + 4};
    auto size_n = onnx_sizes[0];
    const int c_idx = is_nhwc ? 3 : 1;
    if (size_n != input_shape[0] || onnx_sizes[c_idx] != input_shape[c_idx]) {
      LOGS(logger, VERBOSE) << "Output sizes of N/C chanel should match the input sizes, "
                            << "Resize of N/C channels are not supported"
                            << ", input_size_n, " << input_shape[0] << ", output_size_n, " << size_n
                            << ". input_size_c, " << input_shape[c_idx] << ", output_size_c, " << onnx_sizes[c_idx];
      return false;
    }
    // 'sizes' input has been transposed to NHWC layout if it is NHWC preferred layout.
    const int64_t sizes_h = is_nhwc ? onnx_sizes[1] : onnx_sizes[2];
    const int64_t sizes_w = is_nhwc ? onnx_sizes[2] : onnx_sizes[3];
    sizes = {sizes_h, sizes_w};
    axes = {2, 3};
  }

  if (is_nhwc) {
    // For NHWC preferred layout, we need to convert 'axes' from NCHW to NHWC.
    axes = convertAxesFromNCHWtoNHWC(axes);
  }

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
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");

  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node);

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  const auto mode = helper.Get("mode", "nearest");
  if (mode == "linear") {
    options.set("mode", emscripten::val("linear"));
  } else {  // we already checked the mode must be NN or Bilinear in IsOpSupportedImpl.
    options.set("mode", emscripten::val("nearest-neighbor"));
  }

  std::vector<float> scales;
  std::vector<int64_t> sizes;
  std::vector<uint32_t> webnn_sizes;
  std::vector<int64_t> axes = GetResolvedAxes(helper, 4);  // We already checked input shape is 4D in IsOpSupportedImpl.
  std::string sizes_name = GetTensorName(input_defs, 3);
  const bool is_nhwc = model_builder.GetPreferredLayout() == DataLayout::NHWC;

  // We know we have either a 'scales' or 'sizes' input so this is safe.
  // Check for 'sizes' first.
  // This handles Resize-11 where 'scales' was a required input but 'sizes' were used if provided.
  bool using_sizes = !sizes_name.empty() && Contains(initializers, sizes_name);
  if (using_sizes) {
    ORT_RETURN_IF_NOT(GetResizeSizesAndAxes(initializers, node, sizes, axes, is_nhwc, input_shape, logger),
                      "Error getting Resize sizes");
    webnn_sizes = GetVecUint32FromVecInt64(sizes);
    options.set("sizes", emscripten::val::array(webnn_sizes));
  } else {
    ORT_RETURN_IF_NOT(GetResizeScalesAndAxes(initializers, node, scales, axes, is_nhwc, logger),
                      "Error getting Resize scales");
    options.set("scales", emscripten::val::array(scales));
  }

  std::vector<uint32_t> webnn_axes = GetVecUint32FromVecInt64(axes);
  options.set("axes", emscripten::val::array(webnn_axes));

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
  NodeAttrHelper helper(node);

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
    // antialias
    if (helper.Get("antialias", 0) != 0) {
      LOGS(logger, VERBOSE) << "Resize does not support antialias";
      return false;
    }

    // Ignore coordinate_transformation_mode because WebNN only supports half_pixel mode.
    // TODO: Validate coordinate_transformation_mode. Related spec issue for supporting attribute coordinate
    // transformation modes: https://github.com/webmachinelearning/webnn/issues/270

    // exclude_outside
    const auto exclude_outside = helper.Get("exclude_outside", 0);
    if (exclude_outside != 0) {
      LOGS(logger, VERBOSE) << "Resize does not support exclude_outside for now";
      return false;
    }

    // keep_aspect_ratio_policy
    const auto keep_aspect_ratio_policy = helper.Get("keep_aspect_ratio_policy", "stretch");
    if (keep_aspect_ratio_policy != "stretch") {
      LOGS(logger, VERBOSE) << "Resize does not support keep_aspect_ratio_policy: " << keep_aspect_ratio_policy;
      return false;
    }

    // mode
    const auto mode = helper.Get("mode", "nearest");
    bool is_linear_resize = mode == "linear";
    bool is_nearest_resize = mode == "nearest";
    // WebNN only supports "linear" and "nearest" modes.
    if (!is_linear_resize && !is_nearest_resize) {
      LOGS(logger, VERBOSE) << "Resize does not support input mode: " << mode;
      return false;
    }
  }

  {  // 'scales' and 'sizes' (if present) must be non-empty initializers.
    const std::string scales_name = GetTensorName(input_defs, 2);
    const std::string sizes_name = GetTensorName(input_defs, 3);

    // Check for 'sizes' first.
    // This handles Resize-11 where 'scales' was a required input but 'sizes' were used if provided.
    // 'scales' or 'sizes' may be empty tensor.
    bool using_sizes = !IsEmptyTensor(initializers, sizes_name);
    bool using_scales = !using_sizes && !IsEmptyTensor(initializers, scales_name);

    if (!using_scales && !using_sizes) {
      LOGS(logger, VERBOSE) << "Resize: only one of 'scales' and 'sizes' can be specified";
      return false;
    }

    // 'axes' is from opset 18 on and allows 'scales' or 'sizes' to have entries for the subset of 'axes'.
    // We fill with default values if necessary so that the processing is consistent across all supported opsets.
    std::vector<int64_t> axes = GetResolvedAxes(helper, input_size);
    if (!axes.empty()) {  // We have 'axes' attribute.
      if (axes.size() != 2 || axes[0] >= input_size || axes[1] >= input_size) {
        LOGS(logger, VERBOSE) << "Resize: invalid axes attribute";
        return false;
      }
    }

    const bool is_nhwc = node.Domain() == kMSInternalNHWCDomain;
    if (using_sizes) {  // We are using 'sizes'.
      std::vector<int64_t> sizes;
      if (!GetResizeSizesAndAxes(initializers, node, sizes, axes, is_nhwc, input_shape, logger)) {
        return false;
      }
    } else {  // We are using 'scales'.
      std::vector<float> scales;
      if (!GetResizeScalesAndAxes(initializers, node, scales, axes, is_nhwc, logger)) {
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
