// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {

Status CheckROIAlignValidInput(const Tensor* X_ptr, const Tensor* rois_ptr, const Tensor* batch_indices_ptr);

enum struct RoiAlignMode {
  avg = 0,
  max
};

class RoiAlignBase {
 public:
  explicit RoiAlignBase(const OpKernelInfo& info) {
    // mode
    std::string mode;
    if (info.GetAttr<std::string>("mode", &mode).IsOK()) {
      std::transform(mode.begin(), mode.end(), mode.begin(), [](char i) { return static_cast<char>(::tolower(i)); });
      if (mode == "avg") {
        mode_ = RoiAlignMode::avg;
      } else if (mode == "max") {
        mode_ = RoiAlignMode::max;
      } else {
        ORT_THROW("Invalid mode of value ", mode, " specified. It should be either avg or max");
      }
    }

    // output_height
    int64_t output_height_tmp;
    if (info.GetAttr<int64_t>("output_height", &output_height_tmp).IsOK()) {
      output_height_ = output_height_tmp;
    }

    // output_width
    int64_t output_width_tmp;
    if (info.GetAttr<int64_t>("output_width", &output_width_tmp).IsOK()) {
      output_width_ = output_width_tmp;
    }

    // sampling_ratio
    int64_t sampling_ratio_tmp;
    if (info.GetAttr<int64_t>("sampling_ratio", &sampling_ratio_tmp).IsOK()) {
      sampling_ratio_ = sampling_ratio_tmp;
      ORT_ENFORCE(sampling_ratio_ >= 0, "Sampling ratio should be >=0, but it was ", sampling_ratio_);
    }

    // spatial_scale
    float spatial_scale_tmp;
    if (info.GetAttr<float>("spatial_scale", &spatial_scale_tmp).IsOK()) {
      spatial_scale_ = spatial_scale_tmp;
    }

    std::string coordinate_transformation_mode;
    if (info.GetAttr<std::string>("coordinate_transformation_mode", &coordinate_transformation_mode).IsOK()) {
      if (coordinate_transformation_mode == "half_pixel")
        half_pixel_ = true;
      else
        half_pixel_ = false;
    }

    if (mode_ == RoiAlignMode::max && sampling_ratio_ != 1) {
      // TODO(fdwr): Issue #6146. ORT 1.13 will correct the incorrect summation of max mode with PR #7354.
      LOGS_DEFAULT(WARNING) << "The existing summation for max mode and sampling ratios besides 1 is incorrect "
                            << "and will be fixed in the next ORT 1.13 release. Thus the results of RoiAlign "
                            << "will be different.";
    }
  }

 protected:
  RoiAlignMode mode_{RoiAlignMode::avg};
  int64_t output_height_{1};
  int64_t output_width_{1};
  int64_t sampling_ratio_{0};
  float spatial_scale_{1.0f};
  bool half_pixel_{true};

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoiAlignBase);
};

template <typename T>
class RoiAlign final : public OpKernel, public RoiAlignBase {
 public:
  explicit RoiAlign(const OpKernelInfo& info) : OpKernel(info), RoiAlignBase(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoiAlign);
};
}  // namespace onnxruntime
