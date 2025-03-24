// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename QK>
Status CopyQK(cudaStream_t stream,
              const int qk_size,
              const T* input,
              QK* output);

template <>
Status CopyQK(cudaStream_t stream,
              const int qk_size,
              const float* input,
              float* output) {
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output, input, qk_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  return Status::OK();
}

template <>
Status CopyQK(cudaStream_t stream,
              const int qk_size,
              const half* input,
              half* output) {
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output, input, qk_size * sizeof(half), cudaMemcpyDeviceToDevice, stream));
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
