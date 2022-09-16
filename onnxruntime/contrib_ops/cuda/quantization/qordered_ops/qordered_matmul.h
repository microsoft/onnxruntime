// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class QOrderedMatMul final : public CudaKernel {
 public:
  explicit QOrderedMatMul(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t order_A_;
  int64_t order_B_;
  int64_t order_Y_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
