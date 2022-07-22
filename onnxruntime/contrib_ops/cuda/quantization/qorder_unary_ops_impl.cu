// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/quantization/qorder_unary_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

constexpr int kNumLinePerThread = 4;
constexpr int kNumThreadsPerBlock = 256;
constexpr int kNumElementsPerBlockLine = sizeof(char4) * kNumThreadsPerBlock;
constexpr int kNumElementsPerBlock = sizeof(char4) * kNumLinePerThread * kNumThreadsPerBlock;

// Half2 kernel
template <typename FuncT>
__global__ void QOrderUnaryElementWiseKernel(
    const int8_t* input_data, half2 input_scale, int8_t* output_data, half2 inverse_output_scale, const FuncT functor, CUDA_LONG N) {
  CUDA_LONG id = kNumElementsPerBlock * blockIdx.x + threadIdx.x * (CUDA_LONG)sizeof(char4);
  union U1S2 { unsigned u1; short2 s2; char4 c4; } u1s2;
  char4 i4;

  #pragma unroll
  for (int line = 0; line < kNumLinePerThread; line++) {
    if (id < N) {
      i4 = *(const char4*)(input_data + id);
      half2 low2 = __halves2half2(__short2half_rn((short)i4.x), __short2half_rn((short)i4.y));
      low2 = functor(low2 * input_scale) * inverse_output_scale;
      half2 high2 = __halves2half2(__short2half_rn((short)i4.z), __short2half_rn((short)i4.w));
      high2 = functor(high2 * input_scale) * inverse_output_scale;

      u1s2.s2.x = __half2short_rn(low2.x);
      u1s2.s2.y = __half2short_rn(low2.y);
      u1s2.u1 = __vmaxs2(__vmins2(u1s2.u1, 0x007F007F), 0xFF80FF80);
      i4.x = u1s2.c4.x;
      i4.y = u1s2.c4.z;

      u1s2.s2.x = __half2short_rn(high2.x);
      u1s2.s2.y = __half2short_rn(high2.y);
      u1s2.u1 = __vmaxs2(__vmins2(u1s2.u1, 0x007F007F), 0xFF80FF80);
      i4.z = u1s2.c4.x;
      i4.w = u1s2.c4.z;

      *(char4*)(output_data + id) = i4;
      id += kNumElementsPerBlockLine;
    }
  }
}

template <typename FuncT>
void QOrderUnaryElementWiseImpl(
    cudaStream_t stream,
    const int8_t* input_data,
    const float* input_scale,
    int8_t* output_data,
    const float* output_scale,
    const FuncT& func,
    size_t count) {
  if (count & 0x3) {
    throw std::runtime_error("Count must group in 4");
  }

  if (count > 0) {
    half2 half_input_scale = __float2half2_rn(*input_scale);
    half2 half_inverse_output_scale = __float2half2_rn(1.0f / *output_scale);
    int blocksPerGrid = static_cast<int>(CeilDiv(count, kNumElementsPerBlock));
    QOrderUnaryElementWiseKernel<FuncT><<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(
        input_data, half_input_scale, output_data, half_inverse_output_scale, func, static_cast<CUDA_LONG>(count));
  }
}

struct QOrderUnaryOpComputeFastGelu {
  static constexpr float A = 0.5f;
  static constexpr float B = 0.7978845608028654f;  // sqrt(2.0/M_PI)
  static constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0/M_PI)

  const half2 A2 = __float2half2_rn(A);
  const half2 B2 = __float2half2_rn(B);
  const half2 C2 = __float2half2_rn(C);

  __device__ __inline__ half2 operator()(const half2& x) const {
    return x * (A2 + A2 * _Tanh(x * (C2 * x * x + B2)));
  }

  __device__ __inline__ float operator()(const float& x) const {
    return x * (A + A * _Tanh(x * (C * x * x + B)));
  }
};


QORDER_UNARY_OP_DECLARATION(Gelu) {
  QOrderUnaryElementWiseImpl<QOrderUnaryOpComputeFastGelu>(
    stream, input_data, input_scale, output_data, output_scale, QOrderUnaryOpComputeFastGelu(), count);
}



template <typename FuncT>
__global__ void QOrderUnaryElementWiseShareMemoryKernel(
    const int8_t* input_data, float input_scale, int8_t* output_data, float inverse_output_scale, const FuncT functor, CUDA_LONG N) {

  __shared__ char table[256];

  const int calc_id = (int)threadIdx.x - 128;
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
      *(char4*)(output_data + id) = i4;
      id += kNumElementsPerBlockLine;
    }
  }
}


template <typename FuncT>
void QOrderUnaryElementWiseShareMemoryImpl(
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
    QOrderUnaryElementWiseShareMemoryKernel<FuncT><<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(
        input_data, *input_scale, output_data, inverse_output_scale, func, static_cast<CUDA_LONG>(count));
  }
}


struct QOrderUnaryOpComputeGelu {
  static constexpr float sqrt2 = 1.4142135623730950488016887242097f;

  __device__ __inline__ float operator()(const float& x) const {
    return x * 0.5f * (1.0f + erff(x / sqrt2));
  }
};


QORDER_UNARY_OP_SHAREMEMORY_DECLARATION(Gelu) {
  QOrderUnaryElementWiseShareMemoryImpl<QOrderUnaryOpComputeGelu>(
    stream, input_data, input_scale, output_data, output_scale, QOrderUnaryOpComputeGelu(), count);
}


/*
#define LIST_OF_QORDER_UNARY_OPS()          \
  QORDER_UNARY_OP_NAME_EXPR(Gelu, _Gelu(a))


#define DEFINE_QORDER_OP(name, expr)                                 \
  struct QOrderUnaryOp##name {                                       \
    __device__ __inline__ float operator()(const float& a) const {   \
      return expr;                                                   \
    }                                                                \
  };

#define QORDER_UNARY_OP_IMPL(name)                                                         \
  QORDER_UNARY_OP_DECLARATION(name) {                                                      \
    QOrderUnaryElementWiseImpl<QOrderUnaryOp##name>(stream, input_data, input_scale, output_data, output_scale, \
                               QOrderUnaryOp##name(), count);                              \
  }


#define QORDER_UNARY_OP_NAME_EXPR(name, expr) \
  DEFINE_QORDER_OP(name, expr)                \
  QORDER_UNARY_OP_IMPL(name)

LIST_OF_QORDER_UNARY_OPS()
#undef QORDER_UNARY_OP_NAME_EXPR

*/

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
