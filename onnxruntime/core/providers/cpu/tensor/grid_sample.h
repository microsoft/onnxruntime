// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/tensor.h"
#include "concatbase.h"

namespace onnxruntime {

template <typename T>
class GridSample final : public OpKernel {
 public:
  explicit GridSample(const OpKernelInfo& info) : OpKernel(info) {
    std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "bilinear");
    std::string padding_mode_str = info.GetAttrOrDefault<std::string>("padding_mode", "zeros");
    align_corners_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("align_corners", 0));
    ORT_ENFORCE(mode_str == "bilinear" || mode_str == "nearest" || mode_str == "bicubic",
                "mode \"", mode_str, "\" not supported, expect bilinear, nearest or bicubic");
    ORT_ENFORCE(padding_mode_str == "zeros" || padding_mode_str == "border" || padding_mode_str == "reflection",
                "padding_mode \"", padding_mode_str, "\" not supported, expect zeros, border or reflection");
    if (mode_str == "bicubic") {
      mode_ = Bicubic;
    } else if (mode_str == "nearest") {
      mode_ = Nearest;
    } else {
      mode_ = Bilinear;
    }
    if (padding_mode_str == "reflection") {
      padding_mode_ = Reflection;
    } else if (padding_mode_str == "border") {
      padding_mode_ = Border;
    } else {
      padding_mode_ = Zeros;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  enum GridSampleInterpolationMode {
    Bilinear,
    Nearest,
    Bicubic
  };

  enum GridSamplePaddingMode {
    Zeros,
    Border,
    Reflection
  };

  T PixelAtGrid(const T* image, int64_t r, int64_t c, int64_t H, int64_t W, float border[/* 4 */]) const;

  GridSampleInterpolationMode mode_{Bilinear};
  GridSamplePaddingMode padding_mode_{Zeros};
  bool align_corners_{0};
};

}  // namespace onnxruntime
