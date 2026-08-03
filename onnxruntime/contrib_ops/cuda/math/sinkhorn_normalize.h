// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class SinkhornNormalize final : public CudaKernel {
 public:
  SinkhornNormalize(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int iterations_;
  float epsilon_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
