// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "decoder_masked_multihead_attention_impl_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, int Dh>
struct Qk_vec_m_ {
};

template <>
struct Qk_vec_m_<float, 32> {
  using Type = float;
};

template <>
struct Qk_vec_m_<float, 64> {
  using Type = float2;
};

template <>
struct Qk_vec_m_<float, 128> {
  using Type = float4;
};

template <>
struct Qk_vec_m_<float, 256> {
  using Type = float4;
};

template <>
struct Qk_vec_m_<uint16_t, 32> {
  using Type = uint32_t;
};

template <>
struct Qk_vec_m_<uint16_t, 64> {
  using Type = uint32_t;
};

template <>
struct Qk_vec_m_<uint16_t, 128> {
  using Type = uint2;
};

template <>
struct Qk_vec_m_<uint16_t, 256> {
  using Type = uint4;
};

template <typename T, int Dh>
struct Qk_vec_k_ {
  using Type = typename Qk_vec_m_<T, Dh>::Type;
};

template <typename T, int THREADS_PER_KEY>
struct K_vec_m_ {
};

template <>
struct K_vec_m_<float, 4> {
  using Type = float;
};

template <>
struct K_vec_m_<float, 2> {
  using Type = float2;
};

template <>
struct K_vec_m_<float, 1> {
  using Type = float4;
};

template <>
struct K_vec_m_<uint16_t, 4> {
  using Type = uint32_t;
};

template <>
struct K_vec_m_<uint16_t, 2> {
  using Type = uint2;
};

template <>
struct K_vec_m_<uint16_t, 1> {
  using Type = uint4;
};

template <typename T, int THREADS_PER_KEY>
struct K_vec_k_ {
  using Type = typename K_vec_m_<T, THREADS_PER_KEY>::Type;
};

template <typename T, int V_VEC_SIZE>
struct V_vec_m_ {
};

template <>
struct V_vec_m_<float, 1> {
  using Type = float;
};

template <>
struct V_vec_m_<float, 2> {
  using Type = float2;
};

template <>
struct V_vec_m_<float, 4> {
  using Type = float4;
};

template <>
struct V_vec_m_<uint16_t, 2> {
  using Type = uint32_t;
};

template <>
struct V_vec_m_<uint16_t, 4> {
  using Type = uint2;
};

template <>
struct V_vec_m_<uint16_t, 8> {
  using Type = uint4;
};

template <typename T, int V_VEC_SIZE>
struct V_vec_k_ {
  using Type = typename V_vec_m_<T, V_VEC_SIZE>::Type;
};

template <typename T>
struct V_vec_acum_fp32_ {
};

template <>
struct V_vec_acum_fp32_<float> {
  using Type = float;
};

template <>
struct V_vec_acum_fp32_<float2> {
  using Type = float2;
};

template <>
struct V_vec_acum_fp32_<float4> {
  using Type = float4;
};

template <typename T>
inline __device__ void zero(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;
#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

template <typename T>
inline __device__ void ones(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;
#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 1u;
  }
  dst = tmp.raw;
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x) {
  return x;
}

inline __device__ float add_vec(float a, float b) {
  return a + b;
}

inline __device__ float2 add_vec(float2 a, float2 b) {
  float2 c;
  c.x = add_vec(a.x, b.x);
  c.y = add_vec(a.y, b.y);
  return c;
}

inline __device__ float4 add_vec(float4 a, float4 b) {
  float4 c;
  c.x = add_vec(a.x, b.x);
  c.y = add_vec(a.y, b.y);
  c.z = add_vec(a.z, b.z);
  c.w = add_vec(a.w, b.w);
  return c;
}

template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b) {
  return Acc{};  // for compile
}

template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

template <>
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
  return c;
}

template <>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

inline __device__ float sum(float v) {
  return v;
}

inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

template <typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

template <int WARPS_PER_BLOCK, int WARP_SIZE = 32>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < WARPS_PER_BLOCK) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

inline __device__ constexpr uint32_t shfl_mask(int threads) {
  return threads == 32 ? uint32_t(-1) : (1u << threads) - 1u;
}

inline __device__ float fma(float a, float b, float c) {
  return a * b + c;
}

inline __device__ float2 fma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ float4 fma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ float2 fma(float a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline __device__ float4 fma(float a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

template <int THREADS_PER_KEY, typename K_vec, int N>
inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N]) {
  using K_vec_acum = K_vec;

  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2) {
    qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
  }
  return qk;
}

template <typename T, int THREADS_PER_KEY>
struct Qk_dot {
  template <typename K_vec, int N>
  static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N]) {
    return qk_dot_<THREADS_PER_KEY>(q, k);
  }
};

template <typename T, int head_size>
struct threads_per_value_t {
  static const int value = head_size * sizeof(T) / 16;
};

template <typename T>
inline size_t smem_size_in_bytes(const DecoderMaskedMultiheadAttentionParams& params, int threads_per_value, int threads_per_block) {
  // The amount of shared memory needed to store the Q*K^T values in float.

  const int total_sequence_length = params.total_sequence_length;
  size_t qk_sz = (((total_sequence_length + 3) / 4) * 16);

  // The extra memory needed if we are not using floats for the final logits.
  size_t logits_sz = 0;

  if (sizeof(T) != 4) {
    logits_sz = (((total_sequence_length + 3) / 4) * 4 * sizeof(T));
  }

  // The total size needed during softmax.
  size_t softmax_sz = qk_sz + logits_sz;

  // The number of partial rows to reduce in the final reduction.
  int rows_per_red = threads_per_block / threads_per_value;

  // The amount of storage needed to finalize the outputs.
  size_t red_sz = rows_per_red * params.head_size * sizeof(T) / 2;

  // The max.
  return std::max(softmax_sz, red_sz);
}

inline __device__ void convert_from_float(float& dst, float src) {
  dst = src;
}

inline __device__ void convert_from_float(float2& dst, float2 src) {
  dst = src;
}

inline __device__ void convert_from_float(float4& dst, float4 src) {
  dst = src;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
