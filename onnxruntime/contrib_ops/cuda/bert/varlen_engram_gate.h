// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class VarlenEngramGate final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit VarlenEngramGate(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
};

template <typename T>
Status LaunchVarlenEngramGateKernel(
    cudaStream_t stream,
    const T* key,
    const T* query,
    const T* value,
    const T* key_norm_scale,
    const T* query_norm_scale,
    const int32_t* cumulative_sequence_length,
    T* output,
    int batch_size,
    int total_tokens,
    int hc_mult,
    int hidden_size,
    float epsilon);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
