/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"

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

#ifndef USE_ROCM
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Float4_ {
  float2 x;
  float2 y;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float add(float a, float b) {
  return a + b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 add(float2 a, float2 b) {
  float2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 add(float4 a, float4 b) {
  float4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ add(Float8_ a, Float8_ b) {
  Float8_ c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint16_t add(uint16_t a, uint16_t b) {
  uint16_t c;
  asm volatile("add.f16 %0, %1, %2;\n"
               : "=h"(c)
               : "h"(a), "h"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t add(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("add.f16x2 %0, %1, %2;\n"
               : "=r"(c)
               : "r"(a), "r"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 add(uint2 a, uint2 b) {
  uint2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 add(uint4 a, uint4 b) {
  uint4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint16_t float_to_half(float f) {
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

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t float2_to_half2(float2 f) {
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

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float half_to_float(uint16_t h) {
  float f;
  asm volatile("cvt.f32.f16 %0, %1;\n"
               : "=f"(f)
               : "h"(h));
  return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 half2_to_float2(uint32_t v) {
  uint16_t lo, hi;
  asm volatile("mov.b32 {%0, %1}, %2;\n"
               : "=h"(lo), "=h"(hi)
               : "r"(v));
  return make_float2(half_to_float(lo), half_to_float(hi));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float add(float a, uint16_t b) {
  return a + half_to_float(b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 add(uint32_t a, float2 fb) {
  float2 fa = half2_to_float2(a);
  return add(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ add(uint2 a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ add(uint4 a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t h0_h0(uint16_t a) {
  uint32_t b;
  asm volatile("mov.b32 %0, {%1, %1};"
               : "=r"(b)
               : "h"(a));
  return b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(float a, float b, float c) {
  return a * b + c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c) {
  Float4_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c) {
  Float8_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c));
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c) {
  return fma(h0_h0(a), b, c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c) {
  uint2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c) {
  uint32_t s = h0_h0(a);
  uint2 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c) {
  uint4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c) {
  uint32_t s = h0_h0(a);
  uint4 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(uint16_t a, uint16_t b, float fc) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb + fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return fma(fa, fb, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc) {
  return fma(h0_h0(a), b, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc) {
  uint32_t s = h0_h0(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
  return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc) {
  uint32_t s = h0_h0(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
  return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b) {
  return a * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float4 mul(float a, float4 b) {
  float4 c;
  c.x = a * b.x;
  c.y = a * b.y;
  c.z = a * b.z;
  c.w = a * b.w;
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ Float8_ mul(float a, Float8_ b) {
  Float8_ c;
  c.x = make_float2(a * b.x.x, a * b.x.y);
  c.y = make_float2(a * b.y.x, a * b.y.y);
  c.z = make_float2(a * b.z.x, a * b.z.y);
  c.w = make_float2(a * b.w.x, a * b.w.y);
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ uint16_t mul(uint16_t a, uint16_t b) {
  uint16_t c;
  asm volatile("mul.f16 %0, %1, %2;\n"
               : "=h"(c)
               : "h"(a), "h"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("mul.f16x2 %0, %1, %2;\n"
               : "=r"(c)
               : "r"(a), "r"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ uint32_t mul(uint16_t a, uint32_t b) {
  return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ uint2 mul(uint2 a, uint2 b) {
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ uint2 mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ uint4 mul(uint4 a, uint4 b) {
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float mul(uint16_t a, uint16_t b) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float mul(uint16_t a, float b) {
  return half_to_float(a) * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float2 mul(uint32_t a, uint32_t b) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float2 mul(uint16_t a, uint32_t b) {
  return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ Float4_ mul(uint2 a, uint2 b) {
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ Float4_ mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ Float8_ mul(uint4 a, uint4 b) {
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);
  return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float v) {
  return v;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint16_t v) {
  return half_to_float(v);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint32_t v) {
  float2 tmp = half2_to_float2(v);
  return tmp.x + tmp.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint2 v) {
  uint32_t c = add(v.x, v.y);
  return sum(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint4 v) {
#if 1
  uint32_t c = add(v.x, v.y);
  c = add(c, v.z);
  c = add(c, v.w);
#else
  uint32_t c = add(v.x, v.y);
  uint32_t d = add(v.z, v.w);
  c = add(c, d);
#endif
  return sum(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(Float4_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(Float8_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void zero(uint16_t& dst) {
  dst = uint16_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 rotary_embedding_coefficient(const int zid, const int rot_embed_dim, const float t_step) {
  const float inv_freq = t_step / pow(10000.0f, zid / (float)rot_embed_dim);
  return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef) {
  float2 rot_v;
  rot_v.x = coef.x * v.x - coef.y * v.y;
  rot_v.y = coef.x * v.y + coef.y * v.x;
  return rot_v;
}

inline __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef) {
  float2 fv = half2_to_float2(v);
  float2 rot_fv = rotary_embedding_transform(fv, coef);
  return float2_to_half2(rot_fv);
}

inline __device__ void apply_rotary_embedding(float& q, int zid, int rot_embed_dim, int t_step) {
  return;
}

inline __device__ void apply_rotary_embedding(float& q, float& k, int zid, int rot_embed_dim, int t_step) {
  return;
}

inline __device__ void apply_rotary_embedding(Float8_& q, Float8_& k, int zid, int rot_embed_dim, int t_step) {
  return;
}

inline __device__ void apply_rotary_embedding(float2& q, int tid, int rot_embed_dim, int t_step) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(float2& q, float2& k, int tid, int rot_embed_dim, int t_step) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(float4& q, int tid, int rot_embed_dim, int t_step) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }

  Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
  const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
  q_.x = rotary_embedding_transform(q_.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
  q_.y = rotary_embedding_transform(q_.y, coef1);
}

inline __device__ void apply_rotary_embedding(float4& q, float4& k, int tid, int rot_embed_dim, int t_step) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }

  Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
  Float4_& k_ = *reinterpret_cast<Float4_*>(&k);
  const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
  q_.x = rotary_embedding_transform(q_.x, coef0);
  k_.x = rotary_embedding_transform(k_.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
  q_.y = rotary_embedding_transform(q_.y, coef1);
  k_.y = rotary_embedding_transform(k_.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint32_t& q, int tid, int rot_embed_dim, int t_step) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(uint32_t& q, uint32_t& k, int tid, int rot_embed_dim, int t_step) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(uint2& q, int tid, int rot_embed_dim, int t_step) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint2& q, uint2& k, int tid, int rot_embed_dim, int t_step) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint4& q, int tid, int rot_embed_dim, int t_step) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
  q.z = rotary_embedding_transform(q.z, coef2);
  const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
  q.w = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(uint4& q, uint4& k, int tid, int rot_embed_dim, int t_step) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
  const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
  q.z = rotary_embedding_transform(q.z, coef2);
  k.z = rotary_embedding_transform(k.z, coef2);
  const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
  q.w = rotary_embedding_transform(q.w, coef3);
  k.w = rotary_embedding_transform(k.w, coef3);
}

template <typename Vec_T, typename T>
__device__ __inline__ void vec_from_smem_transpose(Vec_T& vec, T* smem, int transpose_idx, int smem_pitch);

template <>
__device__ __inline__ void vec_from_smem_transpose(float& vec, float* smem, int transpose_idx, int smem_pitch) {
  return;
}

template <>
__device__ __inline__ void vec_from_smem_transpose(float4& vec, float2* smem, int transpose_idx, int smem_pitch) {
  return;
}

template <>
__device__ __inline__ void vec_from_smem_transpose(Float8_& vec, float4* smem, int transpose_idx, int smem_pitch) {
  return;
}

template <>
__device__ __inline__ void vec_from_smem_transpose(uint2& vec, half2* smem, int transpose_idx, int smem_pitch) {
  return;
}

template <>
__device__ __inline__ void vec_from_smem_transpose(uint4& vec, Half4* smem, int transpose_idx, int smem_pitch) {
  return;
}

template <>
__device__ __inline__ void vec_from_smem_transpose(uint32_t& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  tmp.u16[0] = smem[transpose_idx];
  tmp.u16[1] = smem[smem_pitch + transpose_idx];

  vec = tmp.u32;
}

template <>
__device__ __inline__ void vec_from_smem_transpose(uint2& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp_1, tmp_2;
  tmp_1.u32 = *reinterpret_cast<uint32_t*>(&smem[transpose_idx]);
  tmp_2.u32 = *reinterpret_cast<uint32_t*>(&smem[smem_pitch + transpose_idx]);

  union {
    uint2 u32x2;
    uint16_t u16[4];
  } tmp_3;
  tmp_3.u16[0] = tmp_1.u16[0];
  tmp_3.u16[1] = tmp_2.u16[0];
  tmp_3.u16[2] = tmp_1.u16[1];
  tmp_3.u16[3] = tmp_2.u16[1];

  vec = tmp_3.u32x2;
}

template <>
__device__ __inline__ void vec_from_smem_transpose(uint4& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
  union {
    uint64_t u64;
    uint16_t u16[4];
  } tmp_1, tmp_2;
  tmp_1.u64 = *reinterpret_cast<uint64_t*>(&smem[transpose_idx]);
  tmp_2.u64 = *reinterpret_cast<uint64_t*>(&smem[smem_pitch + transpose_idx]);

  union {
    uint4 u32x4;
    uint16_t u16[8];
  } tmp_3;
  tmp_3.u16[0] = tmp_1.u16[0];
  tmp_3.u16[1] = tmp_2.u16[0];
  tmp_3.u16[2] = tmp_1.u16[1];
  tmp_3.u16[3] = tmp_2.u16[1];
  tmp_3.u16[4] = tmp_1.u16[2];
  tmp_3.u16[5] = tmp_2.u16[2];
  tmp_3.u16[6] = tmp_1.u16[3];
  tmp_3.u16[7] = tmp_2.u16[3];

  vec = tmp_3.u32x4;
}

template <>
__device__ __inline__ void vec_from_smem_transpose(float4& vec, float* smem, int transpose_idx, int smem_pitch) {
  vec.x = smem[transpose_idx];
  vec.z = smem[transpose_idx + 1];
  vec.y = smem[smem_pitch + transpose_idx];
  vec.w = smem[smem_pitch + transpose_idx + 1];
}

template <>
__device__ __inline__ void vec_from_smem_transpose(uint32_t& vec, half* smem, int transpose_idx, int smem_pitch) {
  union {
    uint32_t u32;
    half u16[2];
  } tmp;
  tmp.u16[0] = smem[transpose_idx];
  tmp.u16[1] = smem[smem_pitch + transpose_idx];

  vec = tmp.u32;
}

template <>
__device__ __inline__ void vec_from_smem_transpose(float2& vec, float* smem, int transpose_idx, int smem_pitch) {
  vec.x = smem[transpose_idx];
  vec.y = smem[smem_pitch + transpose_idx];
}

template <typename Vec_T, typename T>
__device__ __inline__ void write_smem_transpose(const Vec_T& vec, T* smem, int transpose_idx, int smem_pitch);

template <>
__device__ __inline__ void write_smem_transpose(const float& vec, float* smem, int transpose_idx, int smem_pitch) {
  return;
}

template <>
__device__ __inline__ void write_smem_transpose(const float4& vec, float2* smem, int transpose_idx, int smem_pitch) {
  return;
}

template <>
__device__ __inline__ void write_smem_transpose(const Float8_& vec, float4* smem, int transpose_idx, int smem_pitch) {
  return;
}

template <>
__device__ __inline__ void write_smem_transpose(const uint2& vec, half2* smem, int transpose_idx, int smem_pitch) {
  return;
}

template <>
__device__ __inline__ void write_smem_transpose(const uint4& vec, Half4* smem, int transpose_idx, int smem_pitch) {
  return;
}

template <>
__device__ __inline__ void write_smem_transpose(const uint4& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
  union {
    uint64_t u64;
    uint16_t u16[4];
  } tmp_1, tmp_2;

  union {
    uint4 u32x4;
    uint16_t u16[8];
  } tmp_3;
  tmp_3.u32x4 = vec;
  tmp_1.u16[0] = tmp_3.u16[0];
  tmp_2.u16[0] = tmp_3.u16[1];
  tmp_1.u16[1] = tmp_3.u16[2];
  tmp_2.u16[1] = tmp_3.u16[3];
  tmp_1.u16[2] = tmp_3.u16[4];
  tmp_2.u16[2] = tmp_3.u16[5];
  tmp_1.u16[3] = tmp_3.u16[6];
  tmp_2.u16[3] = tmp_3.u16[7];

  *reinterpret_cast<uint64_t*>(&smem[transpose_idx]) = tmp_1.u64;
  *reinterpret_cast<uint64_t*>(&smem[smem_pitch + transpose_idx]) = tmp_2.u64;
}

template <>
__device__ __inline__ void write_smem_transpose(const uint2& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp_1, tmp_2;

  union {
    uint2 u32x2;
    uint16_t u16[4];
  } tmp_3;
  tmp_3.u32x2 = vec;
  tmp_1.u16[0] = tmp_3.u16[0];
  tmp_2.u16[0] = tmp_3.u16[1];
  tmp_1.u16[1] = tmp_3.u16[2];
  tmp_2.u16[1] = tmp_3.u16[3];

  *reinterpret_cast<uint32_t*>(&smem[transpose_idx]) = tmp_1.u32;
  *reinterpret_cast<uint32_t*>(&smem[smem_pitch + transpose_idx]) = tmp_2.u32;
}

template <>
__device__ __inline__ void write_smem_transpose(const uint32_t& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  tmp.u32 = vec;

  smem[transpose_idx] = tmp.u16[0];
  smem[smem_pitch + transpose_idx] = tmp.u16[1];
}

template <>
__device__ __inline__ void write_smem_transpose(const float4& vec, float* smem, int transpose_idx, int smem_pitch) {
  smem[transpose_idx] = vec.x;
  smem[transpose_idx + 1] = vec.z;
  smem[smem_pitch + transpose_idx] = vec.y;
  smem[smem_pitch + transpose_idx + 1] = vec.w;
}

template <>
__device__ __inline__ void write_smem_transpose(const uint32_t& vec, half* smem, int transpose_idx, int smem_pitch) {
  union {
    uint32_t u32;
    half u16[2];
  } tmp;

  tmp.u32 = vec;
  smem[transpose_idx] = tmp.u16[0];
  smem[smem_pitch + transpose_idx] = tmp.u16[1];
}

template <>
__device__ __inline__ void write_smem_transpose(const float2& vec, float* smem, int transpose_idx, int smem_pitch) {
  smem[transpose_idx] = vec.x;
  smem[smem_pitch + transpose_idx] = vec.y;
}

#endif

}  // namespace cuda
}  // namespace onnxruntime
