// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_utils.cuh"
#include "core/framework/float8.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/mlas/inc/mlas.h"

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#endif

namespace onnxruntime {
namespace cuda {

template <typename CudaFp16T, typename CudaFp8T>
__global__ void MLFloat16ToFloat8E4M3FNKernel(const CudaFp16T* src_data, CudaFp8T* dest_data, int num_elems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_elems) {
    dest_data[i] = CudaFp8T(static_cast<unsigned char>(__nv_cvt_halfraw_to_fp8(src_data[i], __NV_SATFINITE, __NV_E4M3)),
                            CudaFp8T::FromBits());
  }
}

Status MLFloat16ToFloat8E4M3FN(cudaStream_t stream, const Tensor* src, void* dest)
{
  typedef typename ToCudaType<MLFloat16>::MappedType CudaFp16T;
  const CudaFp16T* src_data = reinterpret_cast<const CudaFp16T*>(src->Data<MLFloat16>());

  // CudaFp16T* dest_data = reinterpret_cast<CudaFp16T*>(dest);
  typedef typename ToCudaType<Float8E4M3FN>::MappedType CudaFp8T;
  CudaFp8T* dest_data = reinterpret_cast<CudaFp8T*>(dest);

  int num_elems = src->SizeInBytes() / sizeof(MLFloat16);
  int blocks_per_grid = static_cast<int>((num_elems + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);
  int threads_per_block = GridDim::maxThreadsPerBlock;
  MLFloat16ToFloat8E4M3FNKernel<CudaFp16T, CudaFp8T><<<blocks_per_grid, threads_per_block, 0, stream>>>(
    src_data, dest_data, num_elems);

  CUDA_RETURN_IF_ERROR(cudaGetLastError());

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
void PrintTensorData(cudaStream_t stream,  const void* tensor_data, int num_elems, int last_index)
{
  typedef typename ToCudaType<T>::MappedType CudaT;
  const CudaT* data = reinterpret_cast<const CudaT*>(tensor_data);

  int blocks_per_grid = static_cast<int>((num_elems + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);
  int threads_per_block = GridDim::maxThreadsPerBlock;
  PrintTensorDataKernel<CudaT><<<blocks_per_grid, threads_per_block, 0, stream>>>(data, last_index);

  cudaStreamSynchronize(stream);
}

template void PrintTensorData<MLFloat16>(cudaStream_t stream,  const void* tensor_data, int num_elems, int last_index);
template void PrintTensorData<Float8E4M3FN>(cudaStream_t stream,  const void* tensor_data, int num_elems, int last_index);

}  // namespace cuda
}  // namespace onnxruntime
