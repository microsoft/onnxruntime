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
  static const float scale_coef_power = 1.0f / 3.0f;

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float val = tensor_data[i];
  float abs_val = std::abs(val);
  d_scale_coef[i] = std::pow(abs_val, scale_coef_power) * val / abs_val;
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

// The instantiations are based on the MatMul type constraints.
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul
// TODO where is float16 defined?
// template void ComputeStdDevCoefficientsForScale<float16> (cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, float16*  h_scale_coef);
template void ComputeStdDevCoefficientsForScale<float>   (cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, float*    h_scale_coef);
template void ComputeStdDevCoefficientsForScale<double>  (cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, double*   h_scale_coef);
template void ComputeStdDevCoefficientsForScale<uint32_t>(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, uint32_t* h_scale_coef);
template void ComputeStdDevCoefficientsForScale<uint64_t>(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, uint64_t* h_scale_coef);
template void ComputeStdDevCoefficientsForScale<int32_t> (cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, int32_t*  h_scale_coef);
template void ComputeStdDevCoefficientsForScale<int64_t> (cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, int64_t*  h_scale_coef);
template void ComputeStdDevCoefficientsForScale<BFloat16>(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, BFloat16* h_scale_coef);
// TODO these are not listed in the MatMul operator type constraints, but we still get runtime errors without the MLFloat16 instantiation
template void ComputeStdDevCoefficientsForScale<half>(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, half* h_scale_coef);
template void ComputeStdDevCoefficientsForScale<MLFloat16>(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, MLFloat16* h_scale_coef);

}  // namespace cuda
}  // namespace onnxruntime
