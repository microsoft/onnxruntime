// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;//??????

class FasterTransformerVit final : public CudaKernel {
 public:
  ///explicit FasterTransformerVit(const OpKernelInfo& info);
  FasterTransformerVit(const OpKernelInfo& info);
  ///Status ComputeInternal(OpKernelContext* context) const override;
  Status Compute(OpKernelContext* context) const override;

 private:
  Status ComputeInternal(OpKernelContext* context) const;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
