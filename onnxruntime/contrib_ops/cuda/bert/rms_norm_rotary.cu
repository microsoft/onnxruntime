// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/rms_norm_rotary.h"
#include <cuda_bf16.h>
#include "core/providers/cuda/cuda_common.h"
namespace onnxruntime::contrib::cuda {
namespace {
__global__ void RMSNormRotaryKernel(
    const __nv_bfloat16* input, const float* scale, const int64_t* positions,
    const __nv_bfloat16* cosine, const __nv_bfloat16* sine, __nv_bfloat16* output,
    float* normalized, int heads, int sequence, int input_stride, int width, int rotary_dim,
    int cache_rows, bool scalar_position, float epsilon, bool dense_prefill, bool fp32_rotary) {
  constexpr int threads = 64;
  __shared__ float partial[2];
  __shared__ __nv_bfloat16 values[1024];
  const int group = blockIdx.x;
  const int tid = threadIdx.x;
  const int token = group / heads;
  const auto* x = input + static_cast<int64_t>(group) * input_stride;
  float sum = 0.f;
  for (int column = tid; column < width; column += threads) {
    const float v = __bfloat162float(x[column]);
    sum += __fmul_rn(v, v);
  }
#pragma unroll
  for (int step = 16; step > 0; step >>= 1) sum += __shfl_down_sync(0xffffffff, sum, step);
  if ((tid & 31) == 0) partial[tid / 32] = sum;
  __syncthreads();
  const float total = partial[0] + partial[1];
  const float rms = sqrtf(__fadd_rn(total / static_cast<float>(width), epsilon));
  for (int column = tid; column < width; column += threads) {
    const float v = __bfloat162float(x[column]);
    const float norm = dense_prefill ? __fmul_rn(v, __fdiv_rn(1.f, rms)) : v / rms;
    const auto rounded = __float2bfloat16_rn(__fmul_rn(norm, scale[column]));
    values[column] = rounded;
    if (normalized) normalized[static_cast<int64_t>(group) * width + column] = __bfloat162float(rounded);
  }
  __syncthreads();
  const int64_t base = positions[scalar_position ? 0 : token];
  const bool valid = scalar_position ? base >= 0 && base <= static_cast<int64_t>(cache_rows) - sequence
                                     : base >= 0 && base < cache_rows;
  const int64_t position = valid && scalar_position ? base + token % sequence : base;
  const int half = rotary_dim / 2;
  for (int column = tid; column < width; column += threads) {
    auto value = values[column];
    if (column < rotary_dim && valid) {
      const int partner = (column + half) % rotary_dim;
      const int64_t cache = position * half + column % half;
      if (!dense_prefill && fp32_rotary) {
        const float a = __bfloat162float(value), b = __bfloat162float(values[partner]);
        value = __float2bfloat16_rn(__fadd_rn(
            __fmul_rn(a, __bfloat162float(cosine[cache])),
            __fmul_rn(column < half ? -b : b, __bfloat162float(sine[cache]))));
      } else {
        const __nv_bfloat16 sign = __float2bfloat16_rn(column < half ? -1.f : 1.f);
        value = value * cosine[cache] + sign * values[partner] * sine[cache];
      }
    }
    output[static_cast<int64_t>(group) * width + column] = value;
  }
}

}  // namespace
Status LaunchRMSNormRotary(cudaStream_t stream, const uint16_t* input, const float* scale,
                           const int64_t* positions, const uint16_t* cosine, const uint16_t* sine,
                           uint16_t* output, float* normalized, int groups, int heads, int sequence,
                           int input_stride, int width, int rotary_dim, int cache_rows, bool scalar_position,
                           float epsilon, bool dense_prefill, bool fp32_rotary) {
  ORT_RETURN_IF_NOT(groups >= 1 && heads > 0 && sequence > 0 && width > 0 && width <= 1024 &&
                        input_stride >= width && rotary_dim > 0 && rotary_dim <= width &&
                        rotary_dim % 2 == 0 && cache_rows > 0,
                    "RMSNormRotary requires valid dimensions and width <= 1024");
  RMSNormRotaryKernel<<<groups, 64, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input), scale, positions,
      reinterpret_cast<const __nv_bfloat16*>(cosine), reinterpret_cast<const __nv_bfloat16*>(sine),
      reinterpret_cast<__nv_bfloat16*>(output), normalized, heads, sequence, input_stride,
      width, rotary_dim, cache_rows, scalar_position, epsilon, dense_prefill, fp32_rotary);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
