// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
struct MaskFillCuda {
  void operator()(
        cudaStream_t stream,
        Tensor* output_tensor,
        const Tensor* mask_tensor,
        int axis);
};

class MaskFill final : public onnxruntime::cuda::CudaKernel
{
public:
    explicit MaskFill(const OpKernelInfo& info) : CudaKernel{info}
    {
        info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(0));
    }

    Status ComputeInternal(OpKernelContext* context) const override;

private:
    int64_t axis_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
