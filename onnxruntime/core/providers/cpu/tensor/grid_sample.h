// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename T>
class GridSample final : public OpKernel {
 public:
  explicit GridSample(const OpKernelInfo& info) : OpKernel(info) {
    std::tie(mode_, padding_mode_, align_corners_) = GridSample::ParseAttributes(info);
  }

  Status Compute(OpKernelContext* context) const override;

  typedef enum {
    Linear = 0,
    Nearest = 1,
    Cubic = 2,
  } GridSampleInterpolationMode;

  enum GridSamplePaddingMode {
    Zeros = 0,
    Border = 1,
    Reflection = 2,
  };

  static std::tuple<GridSampleInterpolationMode, GridSamplePaddingMode, bool> ParseAttributes(const OpKernelInfo& info) {
   GridSampleInterpolationMode mode{Linear};
   GridSamplePaddingMode padding_mode{Zeros};
   bool align_corners{0};

   int start_version = info.node().SinceVersion();
   if (start_version >= 20) {
     std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "linear");
     if (mode_str == "cubic") {
       mode = Cubic;
     } else if (mode_str == "nearest") {
       mode = Nearest;
     } else if (mode_str == "linear") {
       mode = Linear;
     } else {
       ORT_THROW("mode \"", mode_str, "\" not supported, expect linear, nearest or cubic");
     }
   } else {
     std::string mode_str = info.GetAttrOrDefault<std::string>("mode", "bilinear");
     if (mode_str == "bicubic") {
       mode = Cubic;
     } else if (mode_str == "nearest") {
       mode = Nearest;
     } else if (mode_str == "bilinear") {
       mode = Linear;
     } else {
       ORT_THROW("mode \"", mode_str, "\" not supported, expect bilinear, nearest or bicubic");
     }
   }

   std::string padding_mode_str = info.GetAttrOrDefault<std::string>("padding_mode", "zeros");
   align_corners = static_cast<bool>(info.GetAttrOrDefault<int64_t>("align_corners", 0));
   if (padding_mode_str == "reflection") {
     padding_mode = Reflection;
   } else if (padding_mode_str == "border") {
     padding_mode = Border;
   } else if (padding_mode_str == "zeros") {
     padding_mode = Zeros;
   } else {
     ORT_THROW("padding_mode \"", padding_mode_str, "\" not supported, expect zeros, border or reflection");
   }

   return std::make_tuple(mode, padding_mode, align_corners);
  }

 private:
  T PixelAtGrid(const T* image, int64_t r, int64_t c, int64_t H, int64_t W, T border[/* 4 */]) const;
  T PixelAtGrid3D(const T* image, int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W, T border[/* 6 */]) const;

  GridSampleInterpolationMode mode_{Linear};
  GridSamplePaddingMode padding_mode_{Zeros};
  bool align_corners_{0};
};

}  // namespace onnxruntime
