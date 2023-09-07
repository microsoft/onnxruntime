// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class RmsNormalization final : public CudaKernel {
 public:
  RmsNormalization(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
};

template <typename T>
void LaunchRMSNormKernel(
    const cudaStream_t stream,
    void* out,           // [num_tokens, hidden_size]
    const void* input,   // [num_tokens, hidden_size]
    const void* weight,  // [hidden_size]
    float epsilon,
    const int64_t* input_shape);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
