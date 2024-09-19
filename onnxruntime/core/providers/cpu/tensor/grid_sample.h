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
    int start_version = info.node().SinceVersion();
    if (start_version >= 20) {
      std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "linear");
      if (mode_str == "cubic") {
        mode_ = Cubic;
      } else if (mode_str == "nearest") {
        mode_ = Nearest;
      } else if (mode_str == "linear") {
        mode_ = Linear;
      } else {
        ORT_THROW("mode \"", mode_str, "\" not supported, expect linear, nearest or cubic");
      }
    } else {
      std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "bilinear");
      if (mode_str == "bicubic") {
        mode_ = Cubic;
      } else if (mode_str == "nearest") {
        mode_ = Nearest;
      } else if (mode_str == "bilinear") {
        mode_ = Linear;
      } else {
        ORT_THROW("mode \"", mode_str, "\" not supported, expect bilinear, nearest or bicubic");
      }
    }

    std::string padding_mode_str = info.GetAttrOrDefault<std::string>("padding_mode", "zeros");
    align_corners_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("align_corners", 0));
    if (padding_mode_str == "reflection") {
      padding_mode_ = Reflection;
    } else if (padding_mode_str == "border") {
      padding_mode_ = Border;
    } else if (padding_mode_str == "zeros") {
      padding_mode_ = Zeros;
    } else {
      ORT_THROW("padding_mode \"", padding_mode_str, "\" not supported, expect zeros, border or reflection");
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  typedef enum {
    Linear,
    Cubic,
    Nearest,
  } GridSampleInterpolationMode;

  enum GridSamplePaddingMode {
    Zeros,
    Border,
    Reflection
  };

  T PixelAtGrid(const T* image, int64_t r, int64_t c, int64_t H, int64_t W, T border[/* 4 */]) const;
  T PixelAtGrid3D(const T* image, int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W, T border[/* 6 */]) const;

  GridSampleInterpolationMode mode_{Linear};
  GridSamplePaddingMode padding_mode_{Zeros};
  bool align_corners_{0};
};

}  // namespace onnxruntime
