// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class SequenceConstructionWithTensorAndRepeat final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit SequenceConstructionWithTensorAndRepeat(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
