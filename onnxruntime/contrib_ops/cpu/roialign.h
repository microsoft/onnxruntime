// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {
namespace contrib {
template <typename T>
class ROIAlign final : public OpKernel {
 public:
  explicit ROIAlign(const OpKernelInfo& info) : OpKernel(info) {
    // mode
    if (info.GetAttr<std::string>("mode", &mode_).IsOK()) {
      std::transform(mode_.begin(), mode_.end(), mode_.begin(), [](auto& i) { return static_cast<char>(::tolower(i)); });
      if (mode_ != "avg" && mode_ != "max") {
        ORT_THROW("Invalid mode of value ", mode_, " specified. It should be either avg or max");
      }
    }

    // pooled_h
    info.GetAttr<int64_t>("pooled_h", &pooled_h_);

    // pooled_w
    info.GetAttr<int64_t>("pooled_w", &pooled_w_);

    // sampling_ratio
    if (info.GetAttr<int64_t>("sampling_ratio", &sampling_ratio_).IsOK()) {
      ORT_ENFORCE(sampling_ratio_ >= 0, "Sampling ratio should be >=0, but it was ", sampling_ratio_);
    }

    // spatial_scale
    info.GetAttr<float>("spatial_scale", &spatial_scale_);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::string mode_{"avg"};
  int64_t pooled_h_{1};
  int64_t pooled_w_{1};
  int64_t sampling_ratio_{0};
  float spatial_scale_{1.0f};

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ROIAlign);
};
}  // namespace contrib
}  // namespace onnxruntime
