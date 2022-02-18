// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class QOrderedLayerNormalization final : public CudaKernel {
 public:
  QOrderedLayerNormalization(const OpKernelInfo& op_kernel_info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
  double epsilon_;
  int order_X_;
  int order_Y_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
