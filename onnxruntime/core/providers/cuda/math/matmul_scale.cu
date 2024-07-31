// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_scale.cuh"

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename CudaT>
__global__ void ComputeScaleKernel(const CudaT* tensor_data, float *d_scale)
{
  // TODO implement
  *d_scale = 1.0f;
}

template <typename T>
float ComputeScale(cudaStream_t stream, const Tensor* tensor)
{
  typedef typename ToCudaType<T>::MappedType CudaT;
  const CudaT* tensor_data = reinterpret_cast<const CudaT*>(tensor->Data<T>());

  // TODO how many blocksPerGrid do we want?
  const TensorShape& shape = tensor->Shape();
  const auto size = shape.Size();
  int blocksPerGrid = static_cast<int>((size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);

  float *d_scale; // Device memory
  cudaMalloc(&d_scale, sizeof(float)); // Allocate device memory for scale output
  ComputeScaleKernel<CudaT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(tensor_data, d_scale);

  float h_scale; // Host memory
  cudaMemcpy(&h_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost);  // Copy results back to host
  cudaFree(d_scale);  // Free device memory

  return h_scale;
}

}  // namespace cuda
}  // namespace onnxruntime
