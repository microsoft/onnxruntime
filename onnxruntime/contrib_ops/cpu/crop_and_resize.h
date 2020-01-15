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
      std::transform(mode_.begin(), mode_.end(), mode_.begin(), [](char i) { return static_cast<char>(::tolower(i)); });
      if (mode_ != "bilinear" && mode_ != "nearest") {
        ORT_THROW("Invalid mode of value ", mode_, " specified. It should be either bilinear or nearest");
      }
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
  float extrapolation_value_{0.0f};

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CropAndResize);
};

}  // namespace contrib
}  // namespace onnxruntime
