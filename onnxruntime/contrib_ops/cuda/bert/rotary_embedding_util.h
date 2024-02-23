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
#include "contrib_ops/cuda/bert/utils.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

#ifndef USE_ROCM

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
  float2 fv = Half2ToFloat2(v);
  float2 rot_fv = rotary_embedding_transform(fv, coef);
  return Float2ToHalf2(rot_fv);
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
