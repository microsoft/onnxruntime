// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "attention_quantization_impl.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernel(const int32_t* quantize, const T* bias, T* output, T scale, int bias_len, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = (static_cast<T>(quantize[id]) * scale) + bias[id % bias_len];
      id += NumThreadsPerBlock;
    }
  }
}

template <class T>
Status CudaDequantizeWithBias(cudaStream_t stream, const int32_t* quantize, const T* bias, T* output, T scale, int m, int n) {
  int blocksPerGrid = static_cast<int>(CeilDiv(m * n, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(m * n);
  DequantizeLinearKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      quantize,
      bias,
      output,
      scale,
      n,
      N);
  return Status::OK();
}

template Status CudaDequantizeWithBias<float>(cudaStream_t stream, const int32_t* quantize, const float* bias, float* output, float scale, int m, int n);
template Status CudaDequantizeWithBias<half>(cudaStream_t stream, const int32_t* quantize, const half* bias, half* output, half scale, int m, int n);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
