// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class SequenceConstructUsingTensorAndRepeat final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit SequenceConstructUsingTensorAndRepeat(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
