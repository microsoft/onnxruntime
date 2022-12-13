// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "upsamplebase.h"
namespace onnxruntime {

void UpsampleBase::ParseScalesDataFromOutputSize(gsl::span<const int64_t> output_dims,
                                                 gsl::span<const int64_t> input_dims,
                                                 std::vector<float>& scales) const {
  auto* mutable_out_dim = const_cast<int64_t*>(output_dims.data());

  if (axes_.size()) {
    std::vector<int64_t> output_dim_tmp(output_dims.begin(), output_dims.end());
    for (size_t i = 0; i < axes_.size(); i++) {
      output_dim_tmp[i] = output_dims[static_cast<size_t>(axes_[i])];
    }
    memcpy(mutable_out_dim, output_dim_tmp.data(), output_dim_tmp.size() * sizeof(int64_t));
  }

  for (size_t i = 0, end = input_dims.size(); i < end; ++i) {
    // Handle corner case to avoid dividing by zero in the next step
    if (input_dims[i] == 0) {
      // Enforce that output_dim is 0, given that we cannot scale 0 by any factor to
      // result in any non-zero value
      ORT_ENFORCE(output_dims[i] == 0,
                  "Input dim is zero but required output dim is non-zero. ",
                  "Cannot scale 0 by any factor to generate a non-zero value. ",
                  "Dimension: ", i, " Input dim value: ", input_dims[i], " Output dim value: ", output_dims[i]);
      // Scale can be any arbitrary value as technically scaling 0 by any factor
      // results in 0. Keeping scale as 1 is more intuitive given that input_dim == output_dim.
      scales[i] = 1.f;
    } else {
      scales[i] = static_cast<float>(output_dims[i]) / static_cast<float>(input_dims[i]);
    }
  }

  InlinedHashSet<int64_t> axes_set(axes_.begin(), axes_.end());
  if (keep_aspect_ratio_policy_ != STRETCH) {
    float scale_in_policy = 0.0f;
    if (keep_aspect_ratio_policy_ == NOT_LARGER) {
      scale_in_policy = std::numeric_limits<float>::max();
      for (size_t i = 0; i < scales.size(); i++) {
        if (axes_set.empty() || axes_set.count(i) > 0) {
          scale_in_policy = std::min(scale_in_policy, scales[i]);
        }
      }
    } else if (keep_aspect_ratio_policy_ == NOT_SMALLER) {
      scale_in_policy = std::numeric_limits<float>::min();
      for (size_t i = 0; i < scales.size(); i++) {
        if (axes_set.empty() || axes_set.count(i) > 0) {
          scale_in_policy = std::max(scale_in_policy, scales[i]);
        }
      }
    }
    for (size_t i = 0; i < scales.size(); i++) {
      if (axes_set.empty() || axes_set.count(i) > 0) {
        scales[i] = scale_in_policy;
        mutable_out_dim[i] = static_cast<int64_t>((scales[i] * input_dims[i] + 0.5f));
      } else {
        scales[i] = 1.0f;
        mutable_out_dim[i] = input_dims[i];
      }
    }
  }

  ScalesValidation(scales, mode_);
}

void UpsampleBase::ComputeOutputShape(const std::vector<float>& scales,
                                      gsl::span<const int64_t> input_dims,
                                      TensorShapeVector& output_dims) const {
  auto* mutable_scale = const_cast<float*>(scales.data());
  if (axes_.size()) {
    std::vector<float> scales_tmp(scales);
    for (size_t i = 0; i < axes_.size(); i++) {
      scales_tmp[i] = scales[static_cast<size_t>(axes_[i])];
    }
    memcpy(mutable_scale, scales_tmp.data(), scales.size() * sizeof(float));
  }

  for (std::size_t i = 0; i < input_dims.size(); i++) {
    output_dims[i] = static_cast<int64_t>(scales[i] * input_dims[i]);
  }
}

void UpsampleBase::ComputeROIWithAxes(std::vector<float>& roi_array) const {
  if (axes_.size()) {
    std::vector<float> roi_tmp(roi_array.size(), 0);
    for (size_t i = roi_array.size() / 2; i < roi_array.size(); ++i) {
      roi_tmp[i] = 1;
    }
    for (size_t i = 0; i < axes_.size(); i++) {
      auto v_in_axes = static_cast<size_t>(axes_[i]);
      roi_tmp[v_in_axes] = (roi_array[i]);
      roi_tmp[i + v_in_axes] = (roi_array[axes_.size() + i]);
    }
    roi_array = roi_tmp;
  }
}

}  // namespace onnxruntime
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
