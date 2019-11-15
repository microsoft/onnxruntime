// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Clip_6 final : public CudaKernel {
 public:
  Clip_6(const OpKernelInfo& info) : CudaKernel{info} {
    auto min_val = -std::numeric_limits<T>::infinity();
    auto max_val = std::numeric_limits<T>::infinity();

    info.GetAttrOrDefault("min", &min_, min_val);
    info.GetAttrOrDefault("max", &max_, max_val);

    // Make sure the range of interval is sensible
    ORT_ENFORCE(min_val <= max_val);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  T min_, max_;
};

template <typename T>
class Clip final : public CudaKernel {
 public:
  Clip(const OpKernelInfo& info) : CudaKernel{info} {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
