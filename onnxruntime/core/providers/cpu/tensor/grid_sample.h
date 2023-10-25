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
      ORT_ENFORCE(mode_str == "linear" || mode_str == "nearest" || mode_str == "cubic",
                  "mode \"", mode_str, "\" not supported, expect linear, nearest or cubic");
      if (mode_str == "cubic") {
        mode_ = Cubic;
      } else if (mode_str == "nearest") {
        mode_ = Nearest;
      } else {
        mode_ = Linear;
      }
    } else {
      std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "bilinear");
      ORT_ENFORCE(mode_str == "bilinear" || mode_str == "nearest" || mode_str == "bicubic",
                  "mode \"", mode_str, "\" not supported, expect bilinear, nearest or bicubic");
      if (mode_str == "bicubic") {
        mode_ = Cubic;
      } else if (mode_str == "nearest") {
        mode_ = Nearest;
      } else {
        mode_ = Linear;
      }
    }

    std::string padding_mode_str = info.GetAttrOrDefault<std::string>("padding_mode", "zeros");
    align_corners_ = static_cast<bool>(info.GetAttrOrDefault<int64_t>("align_corners", 0));
    ORT_ENFORCE(padding_mode_str == "zeros" || padding_mode_str == "border" || padding_mode_str == "reflection",
                "padding_mode \"", padding_mode_str, "\" not supported, expect zeros, border or reflection");
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
