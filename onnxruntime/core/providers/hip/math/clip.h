// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/hip/hip_common.h"
#include "core/providers/cpu/math/clip.h"

namespace onnxruntime {
namespace hip {

template <typename T>
class Clip_6 final : public onnxruntime::clip_internal::Clip_6Base<T>,  public HipKernel {
 public:
  explicit Clip_6(const OpKernelInfo& info) : onnxruntime::clip_internal::Clip_6Base<T>(info), HipKernel{info} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

// Since version 11. Min and Max are inputs
// version 12 adds type support
class Clip final : public HipKernel {
 public:
  explicit Clip(const OpKernelInfo& info) : HipKernel{info} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template<typename T>
  struct ComputeImpl;
};

}  // namespace hip
}  // namespace onnxruntime
