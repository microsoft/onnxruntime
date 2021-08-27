// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib{

template <typename T>
class GridSample final : public OpKernel {
 public:
  explicit GridSample(const OpKernelInfo& info);
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

}  //namespace contrib
}  //namespace onnxruntime
