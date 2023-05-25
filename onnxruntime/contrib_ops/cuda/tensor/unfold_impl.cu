// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/tensor/unfold_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/common/common.h"
#include <core/common/safeint.h>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void UnfoldTensorKenel(
    const T* input, T* output,
    int64_t N,
    int64_t tailing_dims_size,
    int64_t stride_fold_dim,
    int64_t stride_leading,
    int64_t stride_leading_orig,
    int64_t step_size
) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  const int64_t idx_leading = idx / stride_leading;
  int64_t n = idx % stride_leading;
  const int64_t idx_fold_new = n / stride_fold_dim;
  n %= stride_fold_dim;
  const int64_t idx_append_new = n / tailing_dims_size;
  n %= tailing_dims_size;

  const int64_t idx_fold_orig = idx_fold_new * step_size + idx_append_new;
  int64_t idx_orig = idx_leading * stride_leading_orig + idx_fold_orig * tailing_dims_size + n;
  output[idx] = input[idx_orig];
}


Status LaunchUnfoldTensor(
    cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    size_t element_size,
    const void* input,
    void* output,
    int64_t leading_dims_size,
    int64_t tailing_dims_size,
    int64_t orig_dim_size,
    int64_t unfold_size,
    int64_t step_size
) {
  int64_t TPB = device_prop.maxThreadsPerBlock;
  int64_t result_dim_size = (orig_dim_size - unfold_size) / step_size + 1;
  int64_t N = result_dim_size * unfold_size * tailing_dims_size * leading_dims_size;
  int64_t num_blocks = (N + TPB - 1) / TPB;
  int64_t new_dim_size = (orig_dim_size - unfold_size) / step_size + 1;

  int64_t stride_fold_dim = tailing_dims_size * unfold_size;
  int64_t stride_leading = stride_fold_dim * new_dim_size;
  int64_t stride_leading_orig = tailing_dims_size * orig_dim_size;

  dim3 block((unsigned)SafeInt<unsigned>(TPB));
  dim3 grid((unsigned)SafeInt<unsigned>(num_blocks));
  switch (element_size) {
    case 1:
        UnfoldTensorKenel<int8_t><<<grid, block, 0, stream>>>(
            (const int8_t*)input, (int8_t*)output, N, tailing_dims_size,
            stride_fold_dim, stride_leading, stride_leading_orig, step_size);
        break;
    case 2:
        UnfoldTensorKenel<int16_t><<<grid, block, 0, stream>>>(
            (const int16_t*)input, (int16_t*)output, N, tailing_dims_size,
            stride_fold_dim, stride_leading, stride_leading_orig, step_size);
        break;
    case 4:
        UnfoldTensorKenel<int32_t><<<grid, block, 0, stream>>>(
            (const int32_t*)input, (int32_t*)output, N, tailing_dims_size,
            stride_fold_dim, stride_leading, stride_leading_orig, step_size);
        break;
    case 8:
        UnfoldTensorKenel<int64_t><<<grid, block, 0, stream>>>(
            (const int64_t*)input, (int64_t*)output, N, tailing_dims_size,
            stride_fold_dim, stride_leading, stride_leading_orig, step_size);
        break;
    case 16:
        UnfoldTensorKenel<float4><<<grid, block, 0, stream>>>(
            (const float4*)input, (float4*)output, N, tailing_dims_size,
            stride_fold_dim, stride_leading, stride_leading_orig, step_size);
        break;
    default:
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported element_size");
  }

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
