// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_scale.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {


template <typename CudaT>
__global__ void ComputeStdDevCoefficientsForScaleKernel(const CudaT* tensor_data, CudaT* d_scale_coef)
{
  static const CudaT scale_coef_power = 1.0f / 3.0f;

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  CudaT abs_val = std::abs(tensor_data[i]);
  d_scale_coef[i] = std::pow(abs_val, scale_coef_power) * tensor_data[i] / abs_val;
}

// h_scale_coef is an array of size num_coef allocated by the caller on the host.
// It will also be subsequently freed by the caller.
template <typename T>
void ComputeStdDevCoefficientsForScale(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, T* h_scale_coef)
{
  typedef typename ToCudaType<T>::MappedType CudaT;
  const CudaT* tensor_data = reinterpret_cast<const CudaT*>(tensor->Data<T>());
  int blocksPerGrid = static_cast<int>((num_coef + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);

  CudaT* d_scale_coef; // Device memory
  cudaMalloc(&d_scale_coef, num_coef * sizeof(T)); // Allocate device memory for the kernel output
  ComputeStdDevCoefficientsForScaleKernel<CudaT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(tensor_data, d_scale_coef);

  cudaMemcpy(&h_scale_coef, d_scale_coef, num_coef * sizeof(T), cudaMemcpyDeviceToHost);  // Copy results back to host
  cudaFree(d_scale_coef);  // Free device memory
}

}  // namespace cuda
}  // namespace onnxruntime
