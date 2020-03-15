// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "attention_impl.h"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {
template <int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void QuantizeLinearKernel(const float* input, int8_t* output, float scale, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      int value = __float2int_rn(input[id] / scale);
      output[id] = static_cast<int8_t>(max(-127, min(127, value)));
      id += NumThreadsPerBlock;
    }
  }
}

template <class Tin>
Status CudaQuantizeLinear(const Tin* input, int8_t* output, Tin scale, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(num_of_element);
  QuantizeLinearKernel<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          input,
          output,
          scale,
          N);
  return Status::OK();
}

template <class T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernel(const int32_t* quantize, const T* bias, T* output, T scale, int bias_len, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = (quantize[id] * scale) + bias[id % bias_len];
      id += NumThreadsPerBlock;
    }
  }
}

template <class T>
Status CudaDequantizeWithBias(const int32_t* quantize, const T* bias, T* output, T scale, int m, int n) {
  int blocksPerGrid = static_cast<int>(CeilDiv(m * n, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(m * n);
  DequantizeLinearKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          quantize,
          bias,
          output,
          scale,
          n,
          N);
  return Status::OK();
}

template Status CudaQuantizeLinear(const float* input, int8_t* output, float scale, size_t num_of_element);
template Status CudaDequantizeWithBias(const int32_t* quantize, const float* bias, float* output, float scale, int m, int n);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
