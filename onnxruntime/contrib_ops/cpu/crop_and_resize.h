// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {
namespace contrib {

template <typename T>
class CropAndResize final : public OpKernel {
 public:
  explicit CropAndResize(const OpKernelInfo& info) : OpKernel(info) {
    // mode
    std::string mode_tmp;
    if (info.GetAttr<std::string>("mode", &mode_tmp).IsOK()) {
      mode_ = mode_tmp;
      std::transform(mode_.begin(), mode_.end(), mode_.begin(), [](auto& i) { return static_cast<char>(::tolower(i)); });
      if (mode_ != "bilinear" && mode_ != "nearest") {
        ORT_THROW("Invalid mode of value ", mode_, " specified. It should be either bilinear or nearest");
      }
    }

    // crop_height
    int64_t crop_height_tmp;
    if (info.GetAttr<int64_t>("crop_height", &crop_height_tmp).IsOK()) {
      crop_height_ = crop_height_tmp;
    }

    // crop_width
    int64_t crop_width_tmp;
    if (info.GetAttr<int64_t>("crop_width", &crop_width_tmp).IsOK()) {
      crop_width_ = crop_width_tmp;
    }

    // extrapolation_value
    float extrapolation_value_tmp;
    if (info.GetAttr<float>("extrapolation_value", &extrapolation_value_tmp).IsOK()) {
      extrapolation_value_ = extrapolation_value_tmp;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  std::string mode_{"bilinear"};
  int64_t crop_height_{1};
  int64_t crop_width_{1};
  float extrapolation_value_{0.0f};
};

}  // namespace contrib
}  // namespace onnxruntime
