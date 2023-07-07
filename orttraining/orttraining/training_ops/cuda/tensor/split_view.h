// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class SplitView final : public CudaKernel {
 public:
  SplitView(const OpKernelInfo& info) : CudaKernel(info) {
    num_outputs_ = info.GetAttrOrDefault<int64_t>("num_outputs", -1);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t num_outputs_ = -1;
};

}  // namespace cuda
}  // namespace onnxruntime
