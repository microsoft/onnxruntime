// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/tensor/resize_impl.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/tensor/upsample.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace webgpu {

Status Upsample::BaseCompute(ComputeContext& context,
                             gsl::span<const float> roi,
                             gsl::span<const float> scales,
                             gsl::span<const int64_t> output_dims) const {
  const auto* X = context.Input(0);
  auto dims = X->Shape().GetDims();
  ORT_ENFORCE(output_dims.size() == dims.size(), "Rank of input and output tensor should be same.");

  if (dims.size() == 0) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  is_resize_ ? "Resize: input tensor cannot be scalar."
                             : "Upsample: input tensor cannot be scalar.");
  }
  if (dims.size() != scales.size()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  is_resize_ ? "Resize: input tensor's dimension does not match the scales."
                             : "Upsample: input tensor's dimension does not match the scales.");
  }
  if (roi.size() != 2 * dims.size()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Resize: size of roi array should be 2 * N where N is the rank of input tensor X.");
  }

  Tensor* Y = context.Output(0, output_dims);
  // Return early if the output tensor is going to be of size 0
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  if (is_resize_) {
    if (!antialias_) {
      return ResizeImpl(context, X, mode_, output_dims, roi, scales, use_extrapolation_, extrapolation_value_,
                        cubic_coeff_a_, exclude_outside_, coordinate_transform_mode_, nearest_mode_);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "The antialias attribute of Resize operator is NOT implemented.");
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Upsample operator is NOT implemented.");
  }
}

Status Upsample::ComputeInternal(ComputeContext& context) const {
  const auto* X = context.Input(0);
  auto input_dims = X->Shape().GetDims();
  TensorShapeVector output_dims(input_dims.size());

  // Get roi data
  // Initialize the roi array to all zeros as this will be the most common case
  // Roi data is needed only when coordinate transformation mode is set to tf_crop_and_resize
  // for all other cases we need a 0 initialized roi array
  InlinedVector<float> roi_array(roi_);

  if (!roi_cached_) {
    bool use_default_roi = true;
    if (need_roi_input_) {
      ORT_ENFORCE(roi_input_idx_ > 0, "Invalid roi input index.");
      const auto* roi = context.Input(roi_input_idx_);
      if (roi != nullptr) {
        ParseRoiData(roi, roi_array);
        use_default_roi = false;
      }
    }
    if (use_default_roi) {
      // default roi includes ensures all the values in that axis are included in the roi
      // normalized roi is thus : [start, end] = [0, 1]
      size_t input_rank = input_dims.size();
      roi_array.resize(input_rank * 2);
      for (size_t i = 0; i < input_rank; ++i) {
        roi_array[i] = 0;
        roi_array[i + input_rank] = 1;
      }
    }
  }

  ComputeROIWithAxes(roi_array, input_dims.size());

  InlinedVector<float> scales_array(input_dims.size());
  // opset < 10
  if (OpKernel::Node().InputDefs().size() == 1) {
    scales_array = scales_;
    // Compute output shape from scales attributes and input dims
    ComputeOutputShape(scales_array, input_dims, output_dims);
    return BaseCompute(context, roi_array, scales_array, output_dims);
  }

  const auto* scales = context.Input(scales_input_idx_);
  const auto* sizes = context.Input(sizes_input_idx_);

  // This is when scales are obtained and cached from a constant initializer
  if (scales_cached_) {
    ORT_RETURN_IF_NOT(sizes == nullptr, "Only one of scales or sizes must be provided as input.");
    scales_array = scales_;
    // Compute output shape from scales and input dims
    ComputeOutputShape(scales_array, input_dims, output_dims);
    return BaseCompute(context, roi_array, scales_array, output_dims);
  }

  // Scales and sizes are input to the node
  if (scales != nullptr && scales->Shape().Size() != 0) {
    // use scales input data
    ORT_ENFORCE(sizes == nullptr, "Only one of scales or sizes must be provided as input.");
    ORT_RETURN_IF_ERROR(ParseScalesData(scales, scales_array, input_dims.size()));

    // Compute output shape from scales and input dims
    ComputeOutputShape(scales_array, input_dims, output_dims);
  } else {
    // When sizes input is available directly populate it into the output_dims array.
    ORT_ENFORCE(sizes != nullptr && sizes->Shape().Size() != 0,
                "Either scales or sizes MUST be provided as input.");
    ORT_RETURN_IF_ERROR(ParseSizesData(sizes, output_dims, input_dims));
    ORT_RETURN_IF_ERROR(ParseScalesDataAndAdjustOutputSize(output_dims, input_dims, scales_array));
  }

  return BaseCompute(context, roi_array, scales_array, output_dims);
}

}  // namespace webgpu
}  // namespace onnxruntime
