// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

class TopK final : public CudaKernel {
 public:
  TopK(const OpKernelInfo&);
  Status ComputeInternal(OpKernelContext*) const override;

 private:
  int64_t axis_     = -1;
  int64_t largest_  = -1;
  int64_t sorted_   = -1;
};
}  // namespace cuda
}  // namespace onnxruntime
