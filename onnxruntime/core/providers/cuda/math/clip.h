// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/math/clip.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Clip_6 final : public onnxruntime::clip_internal::Clip_6Base<T>, public CudaKernel {
 public:
  explicit Clip_6(const OpKernelInfo& info) : onnxruntime::clip_internal::Clip_6Base<T>(info), CudaKernel{info} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
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
