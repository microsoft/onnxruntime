/*
 * The implementation of this file is based on code provided by https://github.com/NVIDIA/FasterTransformer
 * 
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modifications Copyright (c) Microsoft.
// Licensed under the MIT License.

// Modifications:
// (1) Minor routine name changes for integration into the ORT code-base

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "decoder_masked_multihead_attention_impl_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace decoder_masked_multihead_attention_details {

//------------------------------------------------------------
// Qk_vec
//------------------------------------------------------------
template <typename T, int head_size>
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

template <typename T, int head_size>
struct Qk_vec_k_ {
  using Type = typename Qk_vec_m_<T, head_size>::Type;
};

//------------------------------------------------------------
// K_vec
//------------------------------------------------------------
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

//------------------------------------------------------------
// V_vec
//------------------------------------------------------------
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

//------------------------------------------------------------
// V_vec_acum_fp32
//------------------------------------------------------------
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

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

struct Float4_ {
  float2 x;
  float2 y;
};

template <>
struct V_vec_acum_fp32_<uint32_t> {
  using Type = float2;
};
template <>
struct V_vec_acum_fp32_<uint2> {
  using Type = Float4_;
};
template <>
struct V_vec_acum_fp32_<uint4> {
  using Type = Float8_;
};

//------------------------------------------------------------
// Zero
//------------------------------------------------------------

// TODO: fp16 ?

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

//------------------------------------------------------------
// vec_conversion
//------------------------------------------------------------

template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x) {
  return x;
}

//------------------------------------------------------------
// add_vec
//------------------------------------------------------------

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

inline __device__ float HalfToFloat(uint16_t h) {
  float f;
  asm volatile("cvt.f32.f16 %0, %1;\n"
               : "=f"(f)
               : "h"(h));
  return f;
}

inline __device__ float2 Half2ToFloat2(uint32_t v) {
  uint16_t lo, hi;
  asm volatile("mov.b32 {%0, %1}, %2;\n"
               : "=h"(lo), "=h"(hi)
               : "r"(v));
  return make_float2(HalfToFloat(lo), HalfToFloat(hi));
}

inline __device__ uint16_t add_vec(uint16_t a, uint16_t b) {
  uint16_t c;
  asm volatile("add.f16 %0, %1, %2;\n"
               : "=h"(c)
               : "h"(a), "h"(b));
  return c;
}

inline __device__ uint32_t add_vec(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("add.f16x2 %0, %1, %2;\n"
               : "=r"(c)
               : "r"(a), "r"(b));
  return c;
}

inline __device__ uint2 add_vec(uint2 a, uint2 b) {
  uint2 c;
  c.x = add_vec(a.x, b.x);
  c.y = add_vec(a.y, b.y);
  return c;
}

inline __device__ uint4 add_vec(uint4 a, uint4 b) {
  uint4 c;
  c.x = add_vec(a.x, b.x);
  c.y = add_vec(a.y, b.y);
  c.z = add_vec(a.z, b.z);
  c.w = add_vec(a.w, b.w);
  return c;
}

inline __device__ float add_vec(float a, uint16_t b) {
  return a + HalfToFloat(b);
}

inline __device__ float2 add_vec(uint32_t a, float2 fb) {
  float2 fa = Half2ToFloat2(a);
  return add_vec(fa, fb);
}

inline __device__ Float4_ add_vec(uint2 a, Float4_ fb) {
  Float4_ fc;
  fc.x = add_vec(a.x, fb.x);
  fc.y = add_vec(a.y, fb.y);
  return fc;
}

inline __device__ Float8_ add_vec(uint4 a, Float8_ fb) {
  Float8_ fc;
  fc.x = add_vec(a.x, fb.x);
  fc.y = add_vec(a.y, fb.y);
  fc.z = add_vec(a.z, fb.z);
  fc.w = add_vec(a.w, fb.w);
  return fc;
}

//------------------------------------------------------------
// Sum
//------------------------------------------------------------

inline __device__ float sum(float v) {
  return v;
}

inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

inline __device__ float sum(uint16_t v) {
  return HalfToFloat(v);
}

inline __device__ float sum(uint32_t v) {
  float2 tmp = Half2ToFloat2(v);
  return tmp.x + tmp.y;
}

inline __device__ float sum(uint2 v) {
  uint32_t c = add_vec(v.x, v.y);
  return sum(c);
}

inline __device__ float sum(uint4 v) {
  uint32_t c = add_vec(v.x, v.y);
  c = add_vec(c, v.z);
  c = add_vec(c, v.w);
  return sum(c);
}

//------------------------------------------------------------
// Mul
//------------------------------------------------------------

template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b) {
  return Acc{};  // for compile
}

template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
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

template <>
inline __device__ uint16_t mul(uint16_t a, uint16_t b) {
  uint16_t c;
  asm volatile("mul.f16 %0, %1, %2;\n"
               : "=h"(c)
               : "h"(a), "h"(b));
  return c;
}

template <>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("mul.f16x2 %0, %1, %2;\n"
               : "=r"(c)
               : "r"(a), "r"(b));
  return c;
}

template <>
inline __device__ uint2 mul(uint2 a, uint2 b) {
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  return c;
}

template <>
inline __device__ uint4 mul(uint4 a, uint4 b) {
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
  return c;
}

//------------------------------------------------------------
// Fma
//------------------------------------------------------------
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

inline __device__ uint32_t h0_h0(uint16_t a) {
  uint32_t b;
  asm volatile("mov.b32 %0, {%1, %1};"
               : "=r"(b)
               : "h"(a));
  return b;
}

inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c));
  return d;
}

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c) {
  uint2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c) {
  uint4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c) {
  return fma(h0_h0(a), b, c);
}

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c) {
  uint32_t s = h0_h0(a);
  uint2 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c) {
  uint32_t s = h0_h0(a);
  uint4 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

inline __device__ float fma(uint16_t a, uint16_t b, float fc) {
  float fa = HalfToFloat(a);
  float fb = HalfToFloat(b);
  return fa * fb + fc;
}

inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc) {
  float2 fa = Half2ToFloat2(a);
  float2 fb = Half2ToFloat2(b);
  return fma(fa, fb, fc);
}

inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc) {
  return fma(h0_h0(a), b, fc);
}

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc) {
  uint32_t s = h0_h0(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc) {
  uint32_t s = h0_h0(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
  return fd;
}

//------------------------------------------------------------
// Block_sum
//------------------------------------------------------------

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

//------------------------------------------------------------
// Shfl_Mask
//------------------------------------------------------------

inline __device__ constexpr uint32_t shfl_mask(int threads) {
  return threads == 32 ? uint32_t(-1) : (1u << threads) - 1u;
}

//------------------------------------------------------------
// Dot
//------------------------------------------------------------

template <typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

//------------------------------------------------------------
// Qk_Dot
//------------------------------------------------------------

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

//------------------------------------------------------------
// ThreadsPerValue
//------------------------------------------------------------

template <typename T, int head_size>
struct ThreadsPerValue {
  static const int value = head_size * sizeof(T) / 16;
};

//------------------------------------------------------------
// ConvertFromFloat
//------------------------------------------------------------

inline __device__ void ConvertFromFloat(float& dst, float src) {
  dst = src;
}

inline __device__ void ConvertFromFloat(float2& dst, float2 src) {
  dst = src;
}

inline __device__ void ConvertFromFloat(float4& dst, float4 src) {
  dst = src;
}

inline __device__ uint16_t FloatToHalf(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  asm volatile("cvt.rn.f16.f32 %0, %1;\n"
               : "=h"(tmp.u16[0])
               : "f"(f));
  return tmp.u16[0];
}

inline __device__ uint32_t Float2ToHalf2(float2 f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n"
               : "=r"(tmp.u32)
               : "f"(f.y), "f"(f.x));
#else
  asm volatile("cvt.rn.f16.f32 %0, %1;\n"
               : "=h"(tmp.u16[0])
               : "f"(f.x));
  asm volatile("cvt.rn.f16.f32 %0, %1;\n"
               : "=h"(tmp.u16[1])
               : "f"(f.y));
#endif
  return tmp.u32;
}

inline __device__ void ConvertFromFloat(uint16_t& dst, float src) {
  dst = FloatToHalf(src);
}

inline __device__ void ConvertFromFloat(uint32_t& dst, float2 src) {
  dst = Float2ToHalf2(src);
}

inline __device__ void ConvertFromFloat(uint2& dst, Float4_ src) {
  dst.x = Float2ToHalf2(src.x);
  dst.y = Float2ToHalf2(src.y);
}

inline __device__ void ConvertFromFloat(uint4& dst, Float8_ src) {
  dst.x = Float2ToHalf2(src.x);
  dst.y = Float2ToHalf2(src.y);
  dst.z = Float2ToHalf2(src.z);
  dst.w = Float2ToHalf2(src.w);
}
//------------------------------------------------------------
// CalcDynamicBlockMemory
//------------------------------------------------------------

template <typename T>
inline size_t CalcDynamicBlockMemory(const DecoderMaskedMultiheadAttentionParams& params,
                                     int threads_per_value, int threads_per_block) {
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

}  // namespace decoder_masked_multihead_attention_details
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
