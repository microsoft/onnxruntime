// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/math/clip.h"
#include <limits>

namespace onnxruntime {
namespace cuda {

template <typename T>
class Clip_6 final : public CudaKernel {
 public:
  explicit Clip_6(const OpKernelInfo& info) : CudaKernel(info) {
    constexpr auto min_val = std::numeric_limits<T>::lowest();
    constexpr auto max_val = std::numeric_limits<T>::max();
    info.GetAttrOrDefault("min", &min_, min_val);
    info.GetAttrOrDefault("max", &max_, max_val);
    ORT_ENFORCE(min_ <= max_);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  T max_;
  T min_;
};

// Since version 11. Min and Max are inputs
// version 12 adds type support
class Clip final : public CudaKernel {
 public:
  explicit Clip(const OpKernelInfo& info) : CudaKernel{info} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;
};

}  // namespace cuda
}  // namespace onnxruntime
