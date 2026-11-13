// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/rms_norm_rotary.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

template <bool kInputFp16, bool kFp32Cos, bool kOutputFp16>
__global__ void RMSNormRotaryExactKernel(
    const void* input_raw, const float* scale, const int64_t* positions,
    const void* cosine_raw, const void* sine_raw, void* output_raw,
    float* normalized, int heads, int sequence, int input_stride, int width, int rotary_dim,
    int cache_rows, bool scalar_position, float epsilon) {
  constexpr int threads = 64;
  __shared__ float partial[2];
  __shared__ float values_f32[1024];
  const int group = blockIdx.x;
  const int tid = threadIdx.x;
  const int token = group / heads;
  const auto* x_bf16 = reinterpret_cast<const __nv_bfloat16*>(input_raw) + static_cast<int64_t>(group) * input_stride;
  const auto* x_fp16 = reinterpret_cast<const __half*>(input_raw) + static_cast<int64_t>(group) * input_stride;
  float sum = 0.f;
  for (int column = tid; column < width; column += threads) {
    const float v = kInputFp16 ? __half2float(x_fp16[column]) : __bfloat162float(x_bf16[column]);
    sum += __fmul_rn(v, v);
  }
#pragma unroll
  for (int step = 16; step > 0; step >>= 1) sum += __shfl_down_sync(0xffffffff, sum, step);
  if ((tid & 31) == 0) partial[tid / 32] = sum;
  __syncthreads();
  const float total = partial[0] + partial[1];
  const float inv_rms = rsqrtf(__fadd_rn(total / static_cast<float>(width), epsilon));
  for (int column = tid; column < width; column += threads) {
    const float v = kInputFp16 ? __half2float(x_fp16[column]) : __bfloat162float(x_bf16[column]);
    const float scaled = __fmul_rn(__fmul_rn(v, inv_rms), scale[column]);
    values_f32[column] = scaled;
    if (normalized) normalized[static_cast<int64_t>(group) * width + column] = scaled;
  }
  __syncthreads();
  const int64_t base = positions[scalar_position ? 0 : token];
  const bool valid = scalar_position ? base >= 0 && base <= static_cast<int64_t>(cache_rows) - sequence
                                     : base >= 0 && base < cache_rows;
  const int64_t position = valid && scalar_position ? base + token % sequence : base;
  const int half = rotary_dim / 2;
  const float* cos_f32 = reinterpret_cast<const float*>(cosine_raw);
  const float* sin_f32 = reinterpret_cast<const float*>(sine_raw);
  const __nv_bfloat16* cos_bf16 = reinterpret_cast<const __nv_bfloat16*>(cosine_raw);
  const __nv_bfloat16* sin_bf16 = reinterpret_cast<const __nv_bfloat16*>(sine_raw);
  for (int column = tid; column < width; column += threads) {
    float val = values_f32[column];
    if (column < rotary_dim && valid) {
      const int partner = (column + half) % rotary_dim;
      const int64_t cache = position * half + column % half;
      const float a = val;
      const float b = values_f32[partner];
      const float c = kFp32Cos ? cos_f32[cache] : __bfloat162float(cos_bf16[cache]);
      const float s = kFp32Cos ? sin_f32[cache] : __bfloat162float(sin_bf16[cache]);
      val = __fadd_rn(__fmul_rn(a, c), __fmul_rn(column < half ? -b : b, s));
    }
    const int64_t out_idx = static_cast<int64_t>(group) * width + column;
    if constexpr (kOutputFp16) {
      reinterpret_cast<__half*>(output_raw)[out_idx] = __float2half_rn(val);
    } else {
      reinterpret_cast<__nv_bfloat16*>(output_raw)[out_idx] = __float2bfloat16_rn(val);
    }
  }
}

}  // namespace
Status LaunchRMSNormRotary(cudaStream_t stream, const uint16_t* input, const float* scale,
                           const int64_t* positions, const void* cosine, const void* sine,
                           void* output, float* normalized, int groups, int heads, int sequence,
                           int input_stride, int width, int rotary_dim, int cache_rows, bool scalar_position,
                           float epsilon, bool dense_prefill, bool fp32_rotary,
                           bool fp32_cos, bool output_fp16, bool input_fp16) {
  ORT_RETURN_IF_NOT(groups >= 1 && heads > 0 && sequence > 0 && width > 0 && width <= 1024 &&
                        input_stride >= width && rotary_dim > 0 && rotary_dim <= width &&
                        rotary_dim % 2 == 0 && cache_rows > 0,
                    "RMSNormRotary requires valid dimensions and width <= 1024");
  if (input_fp16 || fp32_cos || output_fp16) {
    const auto dispatch = [&](auto in_fp16_tag, auto cos_fp32_tag, auto out_fp16_tag) {
      RMSNormRotaryExactKernel<decltype(in_fp16_tag)::value,
                               decltype(cos_fp32_tag)::value,
                               decltype(out_fp16_tag)::value><<<groups, 64, 0, stream>>>(
          input, scale, positions, cosine, sine, output, normalized,
          heads, sequence, input_stride, width, rotary_dim, cache_rows, scalar_position, epsilon);
    };
    if (input_fp16) {
      if (fp32_cos && output_fp16) {
        dispatch(std::true_type{}, std::true_type{}, std::true_type{});
      } else if (fp32_cos && !output_fp16) {
        dispatch(std::true_type{}, std::true_type{}, std::false_type{});
      } else if (!fp32_cos && output_fp16) {
        dispatch(std::true_type{}, std::false_type{}, std::true_type{});
      } else {
        dispatch(std::true_type{}, std::false_type{}, std::false_type{});
      }
    } else {
      if (fp32_cos && output_fp16) {
        dispatch(std::false_type{}, std::true_type{}, std::true_type{});
      } else if (fp32_cos && !output_fp16) {
        dispatch(std::false_type{}, std::true_type{}, std::false_type{});
      } else {
        dispatch(std::false_type{}, std::false_type{}, std::true_type{});
      }
    }
  } else {
    RMSNormRotaryKernel<<<groups, 64, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input), scale, positions,
        reinterpret_cast<const __nv_bfloat16*>(cosine), reinterpret_cast<const __nv_bfloat16*>(sine),
        reinterpret_cast<__nv_bfloat16*>(output), normalized, heads, sequence, input_stride,
        width, rotary_dim, cache_rows, scalar_position, epsilon, dense_prefill, fp32_rotary);
  }
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
