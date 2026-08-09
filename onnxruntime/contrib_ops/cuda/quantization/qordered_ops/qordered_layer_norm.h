// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using onnxruntime::cuda::CudaKernel;

class QOrderedLayerNormalization final : public CudaKernel {
 public:
  explicit QOrderedLayerNormalization(const OpKernelInfo& op_kernel_info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  double epsilon_;
  int64_t axis_;
  int64_t order_X_;
  int64_t order_Y_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
