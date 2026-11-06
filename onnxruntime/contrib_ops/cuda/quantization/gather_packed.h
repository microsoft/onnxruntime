// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

Status LaunchGatherPackedCuda(
    cudaStream_t stream,
    const uint8_t* data,
    const void* indices,
    bool indices_are_int64,
    void* output,
    int output_dtype,
    int64_t num_rows,
    int64_t vocab_size,
    int64_t k,
    int64_t row_bytes,
    int ggml_type);

template <typename T2, typename Tind>
class GatherPacked final : public CudaKernel {
 public:
  explicit GatherPacked(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t k_{0};
  int64_t ggml_type_{-1};
  int64_t row_bytes_{0};
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
