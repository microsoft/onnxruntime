// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/qordered_ops/qordered_unary_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

static constexpr int kNumLinePerThread = 4;
static constexpr int kNumThreadsPerBlock = 256;
static constexpr int kNumElementsPerBlockLine = sizeof(char4) * kNumThreadsPerBlock;
static constexpr int kNumElementsPerBlock = sizeof(char4) * kNumLinePerThread * kNumThreadsPerBlock;

template <typename FuncT>
__global__ void QOrderedUnaryElementWiseSharedMemoryKernel(
    const int8_t* input_data, float input_scale, int8_t* output_data,
    float inverse_output_scale, const FuncT& functor, CUDA_LONG N) {
  __shared__ char table[256];

  const int calc_id = static_cast<int>(threadIdx.x) - 128;
  float gelu_value = inverse_output_scale * functor(input_scale * calc_id);
  gelu_value = fmaxf(-128.0f, fmin(127.0f, gelu_value));
  table[threadIdx.x] = static_cast<char>(__float2int_rn(gelu_value));
  __syncthreads();

  CUDA_LONG id = kNumElementsPerBlock * blockIdx.x + threadIdx.x * (CUDA_LONG)sizeof(char4);

#pragma unroll
  for (int line = 0; line < kNumLinePerThread; line++) {
    if (id < N) {
      char4 i4 = *(const char4*)(input_data + id);
      i4.x = table[128 + i4.x];
      i4.y = table[128 + i4.y];
      i4.z = table[128 + i4.z];
      i4.w = table[128 + i4.w];
      *reinterpret_cast<char4*>(output_data + id) = i4;
      id += kNumElementsPerBlockLine;
    }
  }
}

template <typename FuncT>
Status QOrderedUnaryElementWiseSharedMemoryImpl(
    cudaStream_t stream,
    const int8_t* input_data,
    const float* input_scale,
    int8_t* output_data,
    const float* output_scale,
    const FuncT& func,
    size_t count) {
  if (count > 0) {
    float inverse_output_scale = 1.0f / *output_scale;
    int blocksPerGrid = static_cast<int>(CeilDiv(count, kNumElementsPerBlock));

    QOrderedUnaryElementWiseSharedMemoryKernel<FuncT><<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(
        input_data, *input_scale, output_data, inverse_output_scale, func, static_cast<CUDA_LONG>(count));
  }
  return CUDA_CALL(cudaGetLastError());
}

struct QOrderedUnaryOpGelu {
  static constexpr float sqrt2 = 1.4142135623730950488016887242097f;

  __device__ __inline__ float operator()(const float& x) const {
    return x * 0.5f * (1.0f + erff(x / sqrt2));
  }
};

QORDERED_UNARY_OP_SHARED_MEMORY_DECLARATION(Gelu) {
  return QOrderedUnaryElementWiseSharedMemoryImpl<QOrderedUnaryOpGelu>(
      stream, input_data, input_scale, output_data, output_scale, QOrderedUnaryOpGelu(), count);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
