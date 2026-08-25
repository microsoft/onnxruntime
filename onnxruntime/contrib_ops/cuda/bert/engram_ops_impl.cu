// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/engram_ops_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxGridDimX = 65535;

inline int GridSize(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  return static_cast<int>(std::min(blocks, kMaxGridDimX));
}

__device__ __forceinline__ float SigmoidFloat(float x) {
  return x > 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}

__device__ __forceinline__ float SiluFloat(float x) {
  return x * SigmoidFloat(x);
}

template <typename T>
__global__ void ShortConvKernel(
    const T* input,
    const T* weight,
    const T* norm_scale,
    const T* bias,
    T* output,
    int64_t total,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t kernel_size,
    int64_t dilation,
    float epsilon,
    bool apply_silu) {
  const int64_t channels = hc_mult * hidden_size;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t c = linear % hidden_size;
    const int64_t g = (linear / hidden_size) % hc_mult;
    const int64_t t = (linear / channels) % sequence_length;
    const int64_t b = linear / (sequence_length * channels);
    const int64_t flat_channel = g * hidden_size + c;

    float sum = bias == nullptr ? 0.0f : to_float<T>(bias[flat_channel]);
    for (int64_t k = 0; k < kernel_size; ++k) {
      const int64_t source_t = t - (kernel_size - 1 - k) * dilation;
      if (source_t < 0) {
        continue;
      }

      const int64_t row_base = ((b * sequence_length + source_t) * hc_mult + g) * hidden_size;
      float sum_sq = 0.0f;
      for (int64_t i = 0; i < hidden_size; ++i) {
        const float value = to_float<T>(input[row_base + i]);
        sum_sq += value * value;
      }
      const float inv_rms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + epsilon);
      const float normed = to_float<T>(input[row_base + c]) * inv_rms *
                           to_float<T>(norm_scale[g * hidden_size + c]);
      sum += normed * to_float<T>(weight[flat_channel * kernel_size + k]);
    }
    output[linear] = from_float<T>(apply_silu ? SiluFloat(sum) : sum);
  }
}

template <typename T>
__device__ __forceinline__ T PositiveMod(T value, T mod) {
  T result = value % mod;
  return result < 0 ? result + mod : result;
}

template <typename T>
__device__ __forceinline__ T WrappedMultiply(T a, T b);

template <>
__device__ __forceinline__ int32_t WrappedMultiply<int32_t>(int32_t a, int32_t b) {
  return static_cast<int32_t>(static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
}

template <>
__device__ __forceinline__ int64_t WrappedMultiply<int64_t>(int64_t a, int64_t b) {
  return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

template <typename T>
__global__ void NgramHashMappingKernel(
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    T* output,
    int64_t total,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id) {
  const int64_t num_heads = (max_ngram_size - 1) * n_head_per_ngram;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = linear % sequence_length;
    const int64_t b = linear / sequence_length;
    const int64_t input_base = b * sequence_length;
    const int64_t output_base = linear * num_heads;

    for (int64_t n = 2; n <= max_ngram_size; ++n) {
      T mix = 0;
      for (int64_t k = 0; k < n; ++k) {
        const int64_t source_t = t - k;
        const T token = source_t < 0 ? pad_id : input_ids[input_base + source_t];
        const T product = WrappedMultiply<T>(token, multipliers[k]);
        mix = k == 0 ? product : static_cast<T>(mix ^ product);
      }

      const int64_t ngram_offset = (n - 2) * n_head_per_ngram;
      for (int64_t h = 0; h < n_head_per_ngram; ++h) {
        const int64_t out_h = ngram_offset + h;
        const T mod = vocab_sizes[out_h];
        output[output_base + out_h] = mod <= 0 ? T{} : PositiveMod(mix, mod);
      }
    }
  }
}

template <typename T>
__global__ void EngramGateKernel(
    const T* embeddings,
    const T* hidden_states,
    const T* key_weight,
    const T* key_bias,
    const T* value_weight,
    const T* value_bias,
    const T* key_norm_scale,
    const T* query_norm_scale,
    T* output,
    int64_t total,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t embedding_size,
    float epsilon) {
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t c = linear % hidden_size;
    const int64_t g = (linear / hidden_size) % hc_mult;
    const int64_t token = linear / (hc_mult * hidden_size);
    const T* embedding_row = embeddings + token * embedding_size;
    const T* hidden_row = hidden_states + (token * hc_mult + g) * hidden_size;

    float key_sum_sq = 0.0f;
    float query_sum_sq = 0.0f;
    float dot_numerator = 0.0f;
    float value = value_bias == nullptr ? 0.0f : to_float<T>(value_bias[c]);

    for (int64_t i = 0; i < embedding_size; ++i) {
      value += to_float<T>(embedding_row[i]) * to_float<T>(value_weight[i * hidden_size + c]);
    }

    for (int64_t d = 0; d < hidden_size; ++d) {
      float key = key_bias == nullptr ? 0.0f : to_float<T>(key_bias[g * hidden_size + d]);
      for (int64_t e = 0; e < embedding_size; ++e) {
        key += to_float<T>(embedding_row[e]) *
               to_float<T>(key_weight[(g * embedding_size + e) * hidden_size + d]);
      }
      const float query = to_float<T>(hidden_row[d]);
      key_sum_sq += key * key;
      query_sum_sq += query * query;
      dot_numerator += key * to_float<T>(key_norm_scale[g * hidden_size + d]) *
                       query * to_float<T>(query_norm_scale[g * hidden_size + d]);
    }

    const float key_inv_rms = rsqrtf(key_sum_sq / static_cast<float>(hidden_size) + epsilon);
    const float query_inv_rms = rsqrtf(query_sum_sq / static_cast<float>(hidden_size) + epsilon);
    const float dot = dot_numerator * key_inv_rms * query_inv_rms / sqrtf(static_cast<float>(hidden_size));
    const float gate_arg = copysignf(sqrtf(fmaxf(fabsf(dot), 1.0e-6f)), dot);
    output[linear] = from_float<T>(SigmoidFloat(gate_arg) * value);
  }
}

}  // namespace

template <typename T>
Status LaunchShortConvKernel(
    cudaStream_t stream,
    const T* input,
    const T* weight,
    const T* norm_scale,
    const T* bias,
    T* output,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t kernel_size,
    int64_t dilation,
    float epsilon,
    bool apply_silu) {
  const int64_t total = batch_size * sequence_length * hc_mult * hidden_size;
  if (total == 0) {
    return Status::OK();
  }
  ShortConvKernel<T><<<GridSize(total), kThreads, 0, stream>>>(
      input, weight, norm_scale, bias, output, total, sequence_length, hc_mult, hidden_size,
      kernel_size, dilation, epsilon, apply_silu);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchNgramHashMappingKernel(
    cudaStream_t stream,
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    T* output,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id) {
  const int64_t total = batch_size * sequence_length;
  if (total == 0) {
    return Status::OK();
  }
  NgramHashMappingKernel<T><<<GridSize(total), kThreads, 0, stream>>>(
      input_ids, multipliers, vocab_sizes, output, total, sequence_length, max_ngram_size,
      n_head_per_ngram, pad_id);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchEngramGateKernel(
    cudaStream_t stream,
    const T* embeddings,
    const T* hidden_states,
    const T* key_weight,
    const T* key_bias,
    const T* value_weight,
    const T* value_bias,
    const T* key_norm_scale,
    const T* query_norm_scale,
    T* output,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t embedding_size,
    float epsilon) {
  const int64_t total = batch_size * sequence_length * hc_mult * hidden_size;
  if (total == 0) {
    return Status::OK();
  }
  EngramGateKernel<T><<<GridSize(total), kThreads, 0, stream>>>(
      embeddings, hidden_states, key_weight, key_bias, value_weight, value_bias, key_norm_scale,
      query_norm_scale, output, total, hc_mult, hidden_size, embedding_size, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_FLOAT(T)                                                                     \
  template Status LaunchShortConvKernel<T>(cudaStream_t, const T*, const T*, const T*, const T*, \
                                           T*, int64_t, int64_t, int64_t, int64_t, int64_t,      \
                                           int64_t, float, bool);                                \
  template Status LaunchEngramGateKernel<T>(cudaStream_t, const T*, const T*, const T*,          \
                                            const T*, const T*, const T*, const T*, const T*,    \
                                            T*, int64_t, int64_t, int64_t, int64_t, int64_t,     \
                                            float);

INSTANTIATE_FLOAT(float)
INSTANTIATE_FLOAT(half)
INSTANTIATE_FLOAT(__nv_bfloat16)

#undef INSTANTIATE_FLOAT

template Status LaunchNgramHashMappingKernel<int32_t>(cudaStream_t, const int32_t*, const int32_t*,
                                                      const int32_t*, int32_t*, int64_t, int64_t,
                                                      int64_t, int64_t, int32_t);
template Status LaunchNgramHashMappingKernel<int64_t>(cudaStream_t, const int64_t*, const int64_t*,
                                                      const int64_t*, int64_t*, int64_t, int64_t,
                                                      int64_t, int64_t, int64_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
