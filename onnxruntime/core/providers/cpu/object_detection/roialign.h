// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {
template <typename T>
class RoiAlign final : public OpKernel {
 public:
  explicit RoiAlign(const OpKernelInfo& info) : OpKernel(info) {
    // mode
    std::string mode_tmp;
    if (info.GetAttr<std::string>("mode", &mode_tmp).IsOK()) {
      mode_ = mode_tmp;
      std::transform(mode_.begin(), mode_.end(), mode_.begin(), [](auto& i) { return static_cast<char>(::tolower(i)); });
      if (mode_ != "avg" && mode_ != "max") {
        ORT_THROW("Invalid mode of value ", mode_, " specified. It should be either avg or max");
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
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::string mode_{"avg"};
  int64_t output_height_{1};
  int64_t output_width_{1};
  int64_t sampling_ratio_{0};
  float spatial_scale_{1.0f};

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RoiAlign);
};
}  // namespace onnxruntime
