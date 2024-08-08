// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_scale.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace cuda {


template <typename CudaT>
__global__ void ComputeStdDevCoefficientsForScaleKernel(const CudaT* tensor_data, CudaT* d_scale_coef)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  static const float scale_coef_power = 1.0f / 3.0f;

  float val = tensor_data[i];
  float abs_val = fabsf(val);
  d_scale_coef[i] = powf(abs_val, scale_coef_power) * val / abs_val;
}

// h_scale_coef is an array of size num_coef allocated by the caller on the host.
// It will also be subsequently freed by the caller.
Status ComputeStdDevCoefficientsForScale(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, MLFloat16* h_scale_coef)
{
  typedef typename ToCudaType<MLFloat16>::MappedType CudaT;
  const CudaT* tensor_data = reinterpret_cast<const CudaT*>(tensor->Data<MLFloat16>());
  int blocks_per_grid = static_cast<int>((num_coef + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);

  CudaT* d_scale_coef; // Device memory
  cudaMalloc(&d_scale_coef, num_coef * sizeof(CudaT)); // Allocate device memory for the kernel output

  int threads_per_block = min(num_coef, GridDim::maxThreadsPerBlock);
  ComputeStdDevCoefficientsForScaleKernel<CudaT><<<blocks_per_grid, threads_per_block, 0, stream>>>(tensor_data, d_scale_coef);

  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  cudaMemcpyAsync(h_scale_coef, d_scale_coef, num_coef * sizeof(CudaT), cudaMemcpyDeviceToHost, stream);  // Copy results back to host
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));

  cudaFree(d_scale_coef);  // Free device memory
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
