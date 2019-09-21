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
  int64_t axis_;
  int64_t largest_;
  int64_t sorted_;
};
}  // namespace cuda
}  // namespace onnxruntime
