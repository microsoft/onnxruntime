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
__global__ void UnfoldTensorKernel(
    const T* input,
    T* output,
    int64_t N,
    int64_t unfold_size, // stride_tailing_dim_dst
    int64_t tailing_dims_size, // stride_fold_dim_dst = tailing_dims_size * unfold_size, stride_append_dim_src = tailing_dims_size
    int64_t stride_leading_dst,
    int64_t stride_fold_dim_src,
    int64_t stride_leading_src
) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  const int64_t idx_leading = idx / stride_leading_dst;
  int64_t n = idx % stride_leading_dst;
  const int64_t stride_fold_dim_dst = tailing_dims_size * unfold_size;
  const int64_t idx_fold = n / stride_fold_dim_dst;
  n %= stride_fold_dim_dst;
  const int64_t idx_tailing = n / unfold_size;
  const int64_t idx_append = n % unfold_size;

  int64_t idx_src = idx_leading * stride_leading_src + idx_fold * stride_fold_dim_src + idx_tailing + idx_append * tailing_dims_size;
  output[idx] = input[idx_src];
}


Status LaunchUnfoldTensor(
    cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    size_t element_size,
    const void* input,
    void* output,
    int64_t leading_dims_size,
    int64_t unfold_dim_size,
    int64_t tailing_dims_size,
    int64_t unfold_size,
    int64_t step_size
) {
  int64_t TPB = device_prop.maxThreadsPerBlock;
  int64_t unfold_dim_size_dst = (unfold_dim_size - unfold_size) / step_size + 1;
  int64_t N = leading_dims_size * unfold_dim_size_dst * tailing_dims_size * unfold_size;
  int64_t num_blocks = (N + TPB - 1) / TPB;

  int64_t stride_leading_dst = unfold_size * tailing_dims_size * unfold_dim_size_dst;

  int64_t stride_fold_dim_src = tailing_dims_size * step_size;
  int64_t stride_leading_src = tailing_dims_size * unfold_dim_size;

  dim3 block((unsigned)SafeInt<unsigned>(TPB));
  dim3 grid((unsigned)SafeInt<unsigned>(num_blocks));
  switch (element_size) {
    case 1:
        UnfoldTensorKernel<int8_t><<<grid, block, 0, stream>>>(
            (const int8_t*)input, (int8_t*)output, N, unfold_size,
            tailing_dims_size, stride_leading_dst, stride_fold_dim_src, stride_leading_src);
        break;
    case 2:
        UnfoldTensorKernel<int16_t><<<grid, block, 0, stream>>>(
            (const int16_t*)input, (int16_t*)output, N, unfold_size,
            tailing_dims_size, stride_leading_dst, stride_fold_dim_src, stride_leading_src);
        break;
    case 4:
        UnfoldTensorKernel<int32_t><<<grid, block, 0, stream>>>(
            (const int32_t*)input, (int32_t*)output, N, unfold_size,
            tailing_dims_size, stride_leading_dst, stride_fold_dim_src, stride_leading_src);
        break;
    case 8:
        UnfoldTensorKernel<int64_t><<<grid, block, 0, stream>>>(
            (const int64_t*)input, (int64_t*)output, N, unfold_size,
            tailing_dims_size, stride_leading_dst, stride_fold_dim_src, stride_leading_src);
        break;
    case 16:
        UnfoldTensorKernel<float4><<<grid, block, 0, stream>>>(
            (const float4*)input, (float4*)output, N, unfold_size,
            tailing_dims_size, stride_leading_dst, stride_fold_dim_src, stride_leading_src);
        break;
    default:
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported element_size");
  }

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
