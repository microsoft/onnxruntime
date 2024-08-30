// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

class PadAndUnflatten final : public CudaKernel {
 public:
  PadAndUnflatten(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
