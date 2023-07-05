// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/tensor/upsamplebase.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Resize : public JsKernel, public UpsampleBase {
 public:
  using UpsampleBase::KeepAspectRatioPolicyToString;
  using UpsampleBase::NearestModeToString;
  using UpsampleBase::ResizeCoordinateTransformationModeToString;
  using UpsampleBase::UpsampleModeToString;
  Resize(const OpKernelInfo& info) : JsKernel(info), UpsampleBase(info) {
    auto resize_coordinate_transformation_mode = ResizeCoordinateTransformationModeToString(coordinate_transform_mode_);
    auto keep_aspect_ratio_policy = KeepAspectRatioPolicyToString(keep_aspect_ratio_policy_);
    auto nearest_mode = NearestModeToString(nearest_mode_);
    auto mode = UpsampleModeToString(mode_);
    std::vector<int32_t> axes;
    std::transform(axes_.begin(), axes_.end(),std::back_inserter(axes),[](auto& axis) {
      return gsl::narrow_cast<int32_t>(axis); });
    JSEP_INIT_KERNEL_ATTRIBUTE(Resize, ({
                                 "antialias" : $1,
                                 "axes" : $2 ? Array.from(HEAP32.subarray($3, $3 + $2)) : [],
                                 "coordinateTransformMode" : Array.from(HEAP8.subarray($4, $4 + $5)).toString(),
                                 "cubicCoeffA" : $5,
                                 "excludeOutside" : !!$6,
                                 "extrapolationValue" : $7,
                                 "keepAspectRatioPolicy" : Array.from(HEAP8.subarray($9, $9 + $10)).toString(),
                                 "mode" : Array.from(HEAP8.subarray($11, $11 + $12)).toString(),
                                 "nearestMode" : Array.from(HEAP8.subarray($13, $13 + $14)).toString(),
                               }),
                               static_cast<int32_t>(antialias_),
                               gsl::narrow_cast<int32_t>(axes.size()),
                               reinterpret_cast<int32_t>((axes.size() > 0) ? axes.data() : nullptr) >> 2,
                               resize_coordinate_transformation_mode.c_str(),
                               resize_coordinate_transformation_mode.length(),
                               static_cast<double>(cubic_coeff_a_),
                               static_cast<int32_t>(exclude_outside_),
                               static_cast<double>(extrapolation_value_),
                               keep_aspect_ratio_policy.c_str(),
                               keep_aspect_ratio_policy.length(),
                               mode.c_str(),
                               mode.length(),
                               nearest_mode.c_str(),
                               nearest_mode.length());
  }
};

}  // namespace js
}  // namespace onnxruntime
