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
#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

struct __align__(8) Half4 {
  half2 x;
  half2 y;
};

__device__ __forceinline__ Half4 operator+(const Half4& a, const Half4& b) {
  Half4 r;
  r.x = a.x + b.x;
  r.y = a.y + b.y;
  return r;
}

__device__ __forceinline__ float2 operator+(const float2& a, const float2& b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float4 operator+(const float4& a, const float4& b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

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

#ifndef USE_ROCM

template <typename T>
struct num_elems;
template <>
struct num_elems<float> {
  static constexpr int value = 1;
};
template <>
struct num_elems<float2> {
  static constexpr int value = 2;
};
template <>
struct num_elems<float4> {
  static constexpr int value = 4;
};
template <>
struct num_elems<Float4_> {
  static constexpr int value = 4;
};
template <>
struct num_elems<Float8_> {
  static constexpr int value = 8;
};

template <>
struct num_elems<uint32_t> {
  static constexpr int value = 2;
};
template <>
struct num_elems<uint2> {
  static constexpr int value = 4;
};
template <>
struct num_elems<uint4> {
  static constexpr int value = 8;
};

template <typename T>
struct Vec_t {
  static constexpr int size = 0;
};

template <>
struct Vec_t<float> {
  using Type = float2;
  static constexpr int size = 2;
};

template <>
struct Vec_t<float2> {
  using Type = float4;
  static constexpr int size = 4;
};

template <>
struct Vec_t<float4> {
  using Type = Float8_;
  static constexpr int size = 8;
};

template <>
struct Vec_t<half> {
  using Type = uint32_t;
  static constexpr int size = 2;
};

template <>
struct Vec_t<half2> {
  using Type = uint2;
  static constexpr int size = 4;
};

template <>
struct Vec_t<Half4> {
  using Type = uint4;
  static constexpr int size = 8;
};

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

template <>
struct V_vec_m_<half, 2> {
  using Type = half2;
};

template <>
struct V_vec_m_<half, 4> {
  using Type = Half4;
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

inline __device__ void zero(uint16_t& dst) {
  dst = uint16_t(0);
}

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

inline __device__ uint32_t h0_h0(uint16_t a) {
  uint32_t b;
  asm volatile("mov.b32 %0, {%1, %1};"
               : "=r"(b)
               : "h"(a));
  return b;
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

inline __device__ Float8_ add_vec(Float8_ a, Float8_ b) {
  Float8_ c;
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

inline __device__ float sum(Float4_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y;
}

inline __device__ float sum(Float8_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
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
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
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
inline __device__ float4 mul(float a, float4 b) {
  float4 c;
  c.x = a * b.x;
  c.y = a * b.y;
  c.z = a * b.z;
  c.w = a * b.w;
  return c;
}

template <>
inline __device__ Float8_ mul(float a, Float8_ b) {
  Float8_ c;
  c.x = make_float2(a * b.x.x, a * b.x.y);
  c.y = make_float2(a * b.y.x, a * b.y.y);
  c.z = make_float2(a * b.z.x, a * b.z.y);
  c.w = make_float2(a * b.w.x, a * b.w.y);
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
inline __device__ uint32_t mul(uint16_t a, uint32_t b) {
  return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

template <>
inline __device__ uint2 mul(uint2 a, uint2 b) {
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  return c;
}

template <>
inline __device__ uint2 mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
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

template <>
inline __device__ uint4 mul(uint16_t a, uint4 b) {
  uint32_t s = h0_h0(a);
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(s, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(s, b.w);
  return c;
}

template <>
inline __device__ float mul(uint16_t a, float b) {
  return HalfToFloat(a) * b;
}

template <>
inline __device__ float mul(uint16_t a, uint16_t b) {
  float fa = HalfToFloat(a);
  float fb = HalfToFloat(b);
  return fa * fb;
}

template <>
inline __device__ float2 mul(uint32_t a, uint32_t b) {
  float2 fa = Half2ToFloat2(a);
  float2 fb = Half2ToFloat2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template <>
inline __device__ float2 mul(uint16_t a, uint32_t b) {
  return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);
}

template <>
inline __device__ Float4_ mul(uint2 a, uint2 b) {
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  return fc;
}

template <>
inline __device__ Float4_ mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  return fc;
}

template <>
inline __device__ Float8_ mul(uint4 a, uint4 b) {
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);
  return fc;
}

template <>
inline __device__ Float8_ mul(uint16_t a, uint4 b) {
  uint32_t s = h0_h0(a);
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
  return fc;
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

inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc) {
  uint32_t s = h0_h0(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
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
#if 0 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800  // Is it better?
      float zero = 0.f;
      asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(zero), "f"(f));
#else
  asm volatile("cvt.rn.f16.f32 %0, %1;\n"
               : "=h"(tmp.u16[0])
               : "f"(f));
#endif
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

#endif

}  // namespace cuda
}  // namespace onnxruntime
