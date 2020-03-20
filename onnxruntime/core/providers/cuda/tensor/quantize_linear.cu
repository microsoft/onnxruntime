// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear.cuh"

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void QuantizeLinearKernel(const float* input, int8_t* output, const float* scale, const int8_t* zero_point, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      int value = __float2int_rn(input[id] / (*scale)) + *zero_point;
      output[id] = static_cast<int8_t>(max(-127, min(127, value)));
      id += NumThreadsPerBlock;
    }
  }
}

template <int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void QuantizeLinearKernel(const float* input, uint8_t* output, const float* scale, const uint8_t* zero_point, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      int value = __float2int_rn(input[id] / (*scale)) + *zero_point;
      output[id] = static_cast<uint8_t>(max(0, min(255, value)));
      id += NumThreadsPerBlock;
    }
  }
}

template <class T>
Status CudaQuantizeLinear(const float* input, T* output, const float* scale, const T* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernel<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          input,
          output,
          scale,
          zero_point,
          num_of_element);
  return Status::OK();
}

template <class T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernel(const T* input, float* output, const float* scale, const T* zero_point, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = (input[id] - *zero_point) * (*scale);
      id += NumThreadsPerBlock;
    }
  }
}

template <class T>
Status CudaDequantizeLinear(const T* input, float* output, const float* scale, const T* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          input,
          output,
          scale,
          zero_point,
          num_of_element);
  return Status::OK();
}

template Status CudaQuantizeLinear<int8_t>(const float* input, int8_t* output, const float* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinear<uint8_t>(const float* input, uint8_t* output, const float* scale, const uint8_t* zero_point, size_t num_of_element);

template Status CudaDequantizeLinear<int8_t>(const int8_t* input, float* output, const float* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinear<uint8_t>(const uint8_t* input, float* output, const float* scale, const uint8_t* zero_point, size_t num_of_element);

}  // namespace cuda
}  // namespace onnxruntime
