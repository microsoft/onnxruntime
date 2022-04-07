// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
struct PackImageToSeqsCuda {
  void operator()(
        cudaStream_t stream,
        const Tensor* input_tensor,
        Tensor* output_tensor,
        int* seq_offset_gpu,
        int* seq_output_indexs_gpu);
};

class PackImageToSeqs final : public onnxruntime::cuda::CudaKernel
{
public:
    explicit PackImageToSeqs(const OpKernelInfo& info) : CudaKernel{info}
    {
        int64_t margin_tmp;
        info.GetAttrOrDefault("margin", &margin_tmp, static_cast<int64_t>(1));
        margin_ = static_cast<int>(margin_tmp);
    }

    Status ComputeInternal(OpKernelContext* context) const override;

private:
    int margin_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
