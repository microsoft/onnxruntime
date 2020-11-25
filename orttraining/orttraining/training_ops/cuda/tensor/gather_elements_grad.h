// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

class GatherElementsGrad final : public CudaKernel {
 public:
  GatherElementsGrad(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(0));
  }
  ~GatherElementsGrad() = default;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;

  int64_t axis_;
};

}  // namespace cuda
}  // namespace onnxruntime
