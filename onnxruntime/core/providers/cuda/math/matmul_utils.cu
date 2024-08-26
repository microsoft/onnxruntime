// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_utils.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace cuda {

template <typename CudaFp16T, typename CudaFp8T>
__global__ void MLFloat16ToFloat8E4M3FNKernel(const CudaFp16T* src_data, CudaFp8T* dest_data)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  dest_data[i] = Float8E4M3FN(src_data[i]);
}

Status MLFloat16ToFloat8E4M3FN(cudaStream_t stream, const Tensor* src, Tensor* dest)
{
  typedef typename ToCudaType<MLFloat16>::MappedType CudaFp16T;
  const CudaFp16T* src_data = reinterpret_cast<const CudaFp16T*>(src->Data<MLFloat16>());

  typedef typename ToCudaType<Float8E4M3FN>::MappedType CudaFp8T;
  CudaFp8T* dest_data = reinterpret_cast<CudaFp8T*>(dest->MutableData<Float8E4M3FN>());

  int num_elems = src->SizeInBytes() / sizeof(CudaFp16T);
  int blocks_per_grid = static_cast<int>((num_elems + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);
  int threads_per_block = min(num_elems, GridDim::maxThreadsPerBlock);
  MLFloat16ToFloat8E4M3FNKernel<CudaFp16T, CudaFp8T><<<blocks_per_grid, threads_per_block, 0, stream>>>(src_data, dest_data);

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));

  return Status::OK();
}



template <typename CudaT>
__global__ void ComputeStdDevCoefficientsForScaleKernel(const CudaT* tensor_data, CudaT* d_scale_coef)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  static const float scale_coef_power = 1.0f / 3.0f;

  float val = tensor_data[i];
  float abs_val = fabsf(val);
  d_scale_coef[i] = CudaT(powf(abs_val, scale_coef_power) * val / abs_val);
}

// h_scale_coef is an array of size num_coef allocated by the caller on the host.
// It will also be subsequently freed by the caller.
Status ComputeStdDevCoefficientsForScale(cudaStream_t stream, const Tensor* tensor, const int32_t num_coef, Float8E4M3FN* h_scale_coef)
{
  typedef typename ToCudaType<Float8E4M3FN>::MappedType CudaT;
  const CudaT* tensor_data = reinterpret_cast<const CudaT*>(tensor->Data<Float8E4M3FN>());
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



// Debugging utility that prints values for all indexes < last_index.
// If last_index is -1, then it prints values for all indexes.
template <typename CudaT>
__global__ void PrintTensorDataKernel(const CudaT* tensor_data, int last_index)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (last_index == -1 || i < last_index) {
    printf("tensor_data[%d] = %f\n", i, float(tensor_data[i]));
  }
}

template <typename T>
void PrintTensorData(cudaStream_t stream,  const Tensor* tensor, int last_index)
{
  typedef typename ToCudaType<T>::MappedType CudaT;
  const CudaT* tensor_data = reinterpret_cast<const CudaT*>(tensor->Data<T>());

  PrintTensorDataKernel<CudaT><<<1, GridDim::maxThreadsPerBlock, 0, stream>>>(tensor_data, last_index);
  cudaStreamSynchronize(stream);
}

template void PrintTensorData<MLFloat16>(cudaStream_t stream,  const Tensor* tensor, int last_index);
template void PrintTensorData<Float8E4M3FN>(cudaStream_t stream,  const Tensor* tensor, int last_index);

}  // namespace cuda
}  // namespace onnxruntime
