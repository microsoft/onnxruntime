// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Tensor-core prefill kernel for DynamicSparseAttention (attention_mode=selected_only, main KV source,
// bfloat16, head_size 256). One CTA handles one (query row, KV head): the heads of the GQA group form the
// M dimension of an m16n8k16 MMA (padded to 16), and the row's selected key/value vectors are gathered
// into shared memory 64 candidates at a time and consumed with an online softmax (flash-attention style).
// The scalar kernels it replaces use one CTA per (row, head) and re-read the gathered keys once per head.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace dsa_prefill {

constexpr int kHeadDim = 256;
constexpr int kTile = 64;
constexpr int kThreads = 128;
constexpr int kRowStride = kHeadDim + 8;  // bf16 elements; 528 B keeps ldmatrix conflict free
constexpr int kSStride = kTile + 4;       // floats
constexpr int kPStride = kTile + 8;       // bf16 elements
constexpr int kMaxGroup = 16;

constexpr size_t kQBytes = kMaxGroup * kRowStride * sizeof(__nv_bfloat16);
constexpr size_t kKBytes = kTile * kRowStride * sizeof(__nv_bfloat16);
constexpr size_t kSBytes = kMaxGroup * kSStride * sizeof(float);
constexpr size_t kPBytes = kMaxGroup * kPStride * sizeof(__nv_bfloat16);
// sQ is only used to load qf[][] into registers before the tile loop and aliases sK thereafter.
constexpr size_t kSharedBytes = 2 * kKBytes + kSBytes + kPBytes + 2 * kMaxGroup * sizeof(float);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t SmemAddr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void CpAsync16(void* smem_dst, const void* gmem_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(SmemAddr(smem_dst)), "l"(gmem_src));
}
__device__ __forceinline__ void CpAsync8(void* smem_dst, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(SmemAddr(smem_dst)), "l"(gmem_src));
}
__device__ __forceinline__ void CpAsyncCommit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void CpAsyncWait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void LdmatrixX4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(SmemAddr(p)));
}
__device__ __forceinline__ void LdmatrixX2Fp16(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(SmemAddr(p)));
}
__device__ __forceinline__ void LdmatrixX4Trans(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(SmemAddr(p)));
}
__device__ __forceinline__ void CpAsync16Fp16Zfill(uint32_t smem_dst, const void* gmem_src, bool valid) {
  const int src_size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_dst), "l"(gmem_src), "r"(src_size));
}

__device__ __forceinline__ void MmaBf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void MmaFp16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ConvertBf16x8ToFp16(uint4 v) {
  const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&v.x);
  const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
  const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&v.z);
  const __nv_bfloat162 b3 = *reinterpret_cast<const __nv_bfloat162*>(&v.w);
  uint4 out;
  *reinterpret_cast<__half2*>(&out.x) = __float22half2_rn(__bfloat1622float2(b0));
  *reinterpret_cast<__half2*>(&out.y) = __float22half2_rn(__bfloat1622float2(b1));
  *reinterpret_cast<__half2*>(&out.z) = __float22half2_rn(__bfloat1622float2(b2));
  *reinterpret_cast<__half2*>(&out.w) = __float22half2_rn(__bfloat1622float2(b3));
  return out;
}

__device__ __forceinline__ uint4 DequantizeInt8x8ToBf16(uint2 packed, float4 s0, float4 s1) {
  const int32_t w0 = static_cast<int32_t>(packed.x);
  const int32_t w1 = static_cast<int32_t>(packed.y);
  const float b0 = static_cast<float>(static_cast<int32_t>(w0 << 24) >> 24) * s0.x;
  const float b1 = static_cast<float>(static_cast<int32_t>(w0 << 16) >> 24) * s0.y;
  const float b2 = static_cast<float>(static_cast<int32_t>(w0 << 8) >> 24) * s0.z;
  const float b3 = static_cast<float>(static_cast<int32_t>(w0) >> 24) * s0.w;
  const float b4 = static_cast<float>(static_cast<int32_t>(w1 << 24) >> 24) * s1.x;
  const float b5 = static_cast<float>(static_cast<int32_t>(w1 << 16) >> 24) * s1.y;
  const float b6 = static_cast<float>(static_cast<int32_t>(w1 << 8) >> 24) * s1.z;
  const float b7 = static_cast<float>(static_cast<int32_t>(w1) >> 24) * s1.w;
  uint4 out;
  *reinterpret_cast<__nv_bfloat162*>(&out.x) = __floats2bfloat162_rn(b0, b1);
  *reinterpret_cast<__nv_bfloat162*>(&out.y) = __floats2bfloat162_rn(b2, b3);
  *reinterpret_cast<__nv_bfloat162*>(&out.z) = __floats2bfloat162_rn(b4, b5);
  *reinterpret_cast<__nv_bfloat162*>(&out.w) = __floats2bfloat162_rn(b6, b7);
  return out;
}
#endif

// grid = (batch * sequence_length, kv_num_heads), block = kThreads, dynamic smem = kSharedBytes.
// key, value: [batch, kv_heads, capacity, 256]. query: [rows, num_heads, 256], already normalized / rotated; float16 for
// the FP16 cache and bfloat16 otherwise (float16 queries over an INT8 cache use the INT8 tensor-core or generic kernels).
template <bool kQuantizedCache = false, bool kFp16Cache = false>
__global__ void __launch_bounds__(kThreads, 3) DynamicSparseAttentionPrefillKernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const int32_t* __restrict__ selected_indices,
    const int32_t* __restrict__ selected_counts,
    const int32_t* __restrict__ seqlens_k,
    __nv_bfloat16* __restrict__ output,
    int sequence_length,
    int num_heads,
    int kv_num_heads,
    int capacity,
    int max_selected,
    float scale,
    const float* __restrict__ key_scale = nullptr,
    const float* __restrict__ value_scale = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using MmaT = std::conditional_t<kFp16Cache, __half, __nv_bfloat16>;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  MmaT* sK = reinterpret_cast<MmaT*>(smem_raw);
  MmaT* sQ = sK;
  MmaT* sV = sK + kTile * kRowStride;
  float* sS = reinterpret_cast<float*>(sV + kTile * kRowStride);
  MmaT* sP = reinterpret_cast<MmaT*>(sS + kMaxGroup * kSStride);
  float* sAlpha = reinterpret_cast<float*>(sP + kMaxGroup * kPStride);
  float* sL = sAlpha + kMaxGroup;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int batch = row / sequence_length;
  const int s = row - batch * sequence_length;
  const int group = num_heads / kv_num_heads;

  const int64_t total_length = static_cast<int64_t>(seqlens_k[batch]) + 1;
  const bool valid_length = total_length >= sequence_length && total_length <= capacity;
  const int query_position = valid_length ? static_cast<int>(total_length) - sequence_length + s : -1;
  const int total = valid_length ? static_cast<int>(total_length) : 0;
  const int count = valid_length ? max(0, min(selected_counts[row], max_selected)) : 0;
  const int32_t* row_indices = selected_indices + static_cast<int64_t>(row) * max_selected;
  const int64_t kv_head_offset = (static_cast<int64_t>(batch) * kv_num_heads + kv_head) * capacity * kHeadDim;
  const MmaT* key_base = reinterpret_cast<const MmaT*>(key) + kv_head_offset;
  const MmaT* value_base = reinterpret_cast<const MmaT*>(value) + kv_head_offset;
  const int8_t* key_base_i8 = reinterpret_cast<const int8_t*>(key) + kv_head_offset;
  const int8_t* value_base_i8 = reinterpret_cast<const int8_t*>(value) + kv_head_offset;
  float4 ks0 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  float4 ks1 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  float4 vs0 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  float4 vs1 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  if constexpr (kQuantizedCache) {
    const float* ks_ptr = key_scale + kv_head * kHeadDim + lane * 8;
    const float* vs_ptr = value_scale + kv_head * kHeadDim + lane * 8;
    ks0 = *reinterpret_cast<const float4*>(ks_ptr);
    ks1 = *reinterpret_cast<const float4*>(ks_ptr + 4);
    vs0 = *reinterpret_cast<const float4*>(vs_ptr);
    vs1 = *reinterpret_cast<const float4*>(vs_ptr + 4);
  }

  // Stage the group's query heads; the padding rows are zero so their (ignored) scores stay finite.
  const MmaT* query_group =
      reinterpret_cast<const MmaT*>(query) + (static_cast<int64_t>(row) * num_heads + kv_head * group) * kHeadDim;
  for (int i = tid; i < kMaxGroup * (kHeadDim / 8); i += kThreads) {
    const int r = i / (kHeadDim / 8);
    const int c = (i % (kHeadDim / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < group) {
      v = *reinterpret_cast<const uint4*>(query_group + static_cast<int64_t>(r) * kHeadDim + c);
    }
    *reinterpret_cast<uint4*>(sQ + r * kRowStride + c) = v;
  }
  __syncthreads();

  uint32_t qf[16][4];
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    LdmatrixX4(qf[k], sQ + (lane & 15) * kRowStride + k * 16 + (lane >> 4) * 8);
  }
  __syncthreads();

  auto candidate_index = [&](int candidate) -> int {
    if (candidate >= count) return -1;
    const int index = row_indices[candidate];
    return (index >= 0 && index < total && index <= query_position && index < capacity) ? index : -1;
  };

  auto issue_rows = [&](MmaT* dst, const MmaT* base, const int8_t* base_i8, int tile) {
  // 4 candidate rows per pass (one per warp), 32 lanes x 16 B = one 512 B row.
#pragma unroll
    for (int it = 0; it < kTile / 4; ++it) {
      const int cand_in_tile = it * 4 + warp;
      const int index = candidate_index(tile * kTile + cand_in_tile);
      MmaT* row_dst = dst + cand_in_tile * kRowStride + lane * 8;
      if (index >= 0) {
        if constexpr (kQuantizedCache) {
          CpAsync8(row_dst, base_i8 + static_cast<int64_t>(index) * kHeadDim + lane * 8);
        } else {
          CpAsync16(row_dst, base + static_cast<int64_t>(index) * kHeadDim + lane * 8);
        }
      } else {
        if constexpr (kQuantizedCache) {
          *reinterpret_cast<uint2*>(row_dst) = make_uint2(0, 0);
        } else {
          *reinterpret_cast<uint4*>(row_dst) = make_uint4(0, 0, 0, 0);
        }
      }
    }
  };

  auto dequantize_rows = [&](MmaT* dst, float4 sc0, float4 sc1) {
    if constexpr (kQuantizedCache) {
#pragma unroll
      for (int it = 0; it < kTile / 4; ++it) {
        const int cand_in_tile = it * 4 + warp;
        MmaT* row_dst = dst + cand_in_tile * kRowStride + lane * 8;
        const uint2 packed = *reinterpret_cast<const uint2*>(row_dst);
        *reinterpret_cast<uint4*>(row_dst) = DequantizeInt8x8ToBf16(packed, sc0, sc1);
      }
    }
  };

  const int tiles = (count + kTile - 1) / kTile;
  if (tiles > 0) {
    issue_rows(sK, key_base, key_base_i8, 0);
    CpAsyncCommit();
    issue_rows(sV, value_base, value_base_i8, 0);
    CpAsyncCommit();
  }

  float acc_o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    acc_o[n][0] = acc_o[n][1] = acc_o[n][2] = acc_o[n][3] = 0.0f;
  }
  // Softmax state of the row this thread reduces for (replicated across the 8 threads of the row).
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  const int soft_row = tid >> 3;
  const int soft_col = (tid & 7) * 8;

  for (int tile = 0; tile < tiles; ++tile) {
    CpAsyncWait<1>();  // K(tile) has landed, V(tile) may still be in flight
    dequantize_rows(sK, ks0, ks1);
    __syncthreads();

    // S = Q K^T for this warp's 16 candidate columns.
    float acc_s[2][4];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      acc_s[n][0] = acc_s[n][1] = acc_s[n][2] = acc_s[n][3] = 0.0f;
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      uint32_t b[4];
      LdmatrixX4(b, sK + (warp * 16 + (lane & 7) + (lane >> 4) * 8) * kRowStride + k * 16 + ((lane >> 3) & 1) * 8);
      if constexpr (kFp16Cache) {
        MmaFp16(acc_s[0], qf[k], b[0], b[1]);
        MmaFp16(acc_s[1], qf[k], b[2], b[3]);
      } else {
        MmaBf16(acc_s[0], qf[k], b[0], b[1]);
        MmaBf16(acc_s[1], qf[k], b[2], b[3]);
      }
    }
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int col = warp * 16 + n * 8 + (lane & 3) * 2;
      const int r0 = lane >> 2;
      *reinterpret_cast<float2*>(sS + r0 * kSStride + col) = make_float2(acc_s[n][0] * scale, acc_s[n][1] * scale);
      *reinterpret_cast<float2*>(sS + (r0 + 8) * kSStride + col) = make_float2(acc_s[n][2] * scale, acc_s[n][3] * scale);
    }
    __syncthreads();  // scores visible; sK is free

    if (tile + 1 < tiles) {
      issue_rows(sK, key_base, key_base_i8, tile + 1);
    }
    CpAsyncCommit();  // K(tile+1) (possibly empty)

    // Online softmax: 8 threads per row, 8 columns each.
    float s_val[8];
    float tile_max = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int candidate = tile * kTile + soft_col + j;
      float x = sS[soft_row * kSStride + soft_col + j];
      if (candidate_index(candidate) < 0) x = -FLT_MAX;
      s_val[j] = x;
      tile_max = fmaxf(tile_max, x);
    }
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 2));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 4));
    const float new_max = fmaxf(running_max, tile_max);
    const float alpha = running_sum == 0.0f ? 0.0f : __expf(running_max - new_max);
    float tile_sum = 0.0f;
    MmaT p_mma[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float p = s_val[j] <= -FLT_MAX ? 0.0f : __expf(s_val[j] - new_max);
      tile_sum += p;
      if constexpr (kFp16Cache) {
        p_mma[j] = __float2half_rn(p);
      } else {
        p_mma[j] = __float2bfloat16_rn(p);
      }
    }
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 2);
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 4);
    running_sum = running_sum * alpha + tile_sum;
    running_max = new_max;
    *reinterpret_cast<uint4*>(sP + soft_row * kPStride + soft_col) = *reinterpret_cast<const uint4*>(p_mma);
    if ((tid & 7) == 0) sAlpha[soft_row] = alpha;
    __syncthreads();  // P and alpha visible

    {
      const float a0 = sAlpha[lane >> 2];
      const float a1 = sAlpha[(lane >> 2) + 8];
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        acc_o[n][0] *= a0;
        acc_o[n][1] *= a0;
        acc_o[n][2] *= a1;
        acc_o[n][3] *= a1;
      }
    }

    CpAsyncWait<1>();  // V(tile) has landed (K(tile+1) may be in flight)
    dequantize_rows(sV, vs0, vs1);
    __syncthreads();

    // O += P V for this warp's 64 output dimensions.
#pragma unroll
    for (int k = 0; k < kTile / 16; ++k) {
      uint32_t a[4];
      LdmatrixX4(a, sP + (lane & 15) * kPStride + k * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        uint32_t b[4];
        LdmatrixX4Trans(b, sV + (k * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kRowStride + warp * 64 + pair * 16 +
                               (lane >> 4) * 8);
        if constexpr (kFp16Cache) {
          MmaFp16(acc_o[pair * 2], a, b[0], b[1]);
          MmaFp16(acc_o[pair * 2 + 1], a, b[2], b[3]);
        } else {
          MmaBf16(acc_o[pair * 2], a, b[0], b[1]);
          MmaBf16(acc_o[pair * 2 + 1], a, b[2], b[3]);
        }
      }
    }
    __syncthreads();  // sV and sP are free

    if (tile + 1 < tiles) {
      issue_rows(sV, value_base, value_base_i8, tile + 1);
    }
    CpAsyncCommit();  // V(tile+1) (possibly empty)
  }

  if ((tid & 7) == 0) sL[soft_row] = running_sum;
  __syncthreads();
  const int r0 = lane >> 2;
  const float l0 = sL[r0];
  const float l1 = sL[r0 + 8];
  const float inv0 = l0 == 0.0f ? 0.0f : 1.0f / l0;
  const float inv1 = l1 == 0.0f ? 0.0f : 1.0f / l1;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int dim = warp * 64 + n * 8 + (lane & 3) * 2;
    if (r0 < group) {
      __nv_bfloat16* dst = output + (static_cast<int64_t>(row) * num_heads + kv_head * group + r0) * kHeadDim + dim;
      *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc_o[n][0] * inv0, acc_o[n][1] * inv0);
    }
    if (r0 + 8 < group) {
      __nv_bfloat16* dst = output + (static_cast<int64_t>(row) * num_heads + kv_head * group + r0 + 8) * kHeadDim + dim;
      *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc_o[n][2] * inv1, acc_o[n][3] * inv1);
    }
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)selected_indices;
  (void)selected_counts;
  (void)seqlens_k;
  (void)output;
  (void)sequence_length;
  (void)num_heads;
  (void)kv_num_heads;
  (void)capacity;
  (void)max_selected;
  (void)scale;
  (void)key_scale;
  (void)value_scale;
#endif
}

constexpr size_t Fp16TcSharedBytes(int cand_tile, bool split_k) {
  const size_t k_bytes = static_cast<size_t>(cand_tile) * kRowStride * sizeof(__half);
  const size_t p_stride = static_cast<size_t>(cand_tile) + 8;
  const size_t p_bytes = kMaxGroup * p_stride * sizeof(__half);
  const size_t idx_bytes = 3 * static_cast<size_t>(cand_tile) * sizeof(int32_t);
  const size_t m_bytes = kMaxGroup * 4 * sizeof(float);
  return 2 * k_bytes + idx_bytes + m_bytes + (split_k ? 2 * p_bytes : p_bytes);
}

constexpr size_t kFp16TcTile32PrefillSharedBytes = Fp16TcSharedBytes(32, false);
constexpr size_t kFp16TcTile64PrefillSharedBytes = Fp16TcSharedBytes(64, false);
constexpr size_t kFp16TcTile32SplitKSharedBytes = Fp16TcSharedBytes(32, true);
constexpr size_t kFp16TcTile64SplitKSharedBytes = Fp16TcSharedBytes(64, true);

// Occupancy-tuned two-buffer FP16-KV Tensor-Core kernel for both prefill (kSplitK=false) and decode (kSplitK=true).
// With kCandTile=32 (~35.7 KB prefill / ~37.0 KB decode shared memory), two CTAs reside per SM concurrently on
// 100 KB/SM GPUs (SM 8.6 / 8.9 / 12.0 / 12.1), doubling concurrent cp.async KV gathers and eliminating the scalar
// GroupedDecode fallback. With kSplitK=true, probabilities are split into P_hi + P_residual (~22 mantissa bits).
template <bool kSplitK, int kCandTile>
__global__ void __launch_bounds__(kThreads, 2) DynamicSparseAttentionFp16TcKernel(
    const __nv_bfloat16* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    const int32_t* __restrict__ selected_indices,
    const int32_t* __restrict__ selected_counts,
    const int32_t* __restrict__ seqlens_k,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_output,
    int split_count,
    int sequence_length,
    int num_heads,
    int kv_num_heads,
    int capacity,
    int max_selected,
    float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  static_assert(kCandTile == 32 || kCandTile == 64, "kCandTile must be 32 or 64");
  constexpr int kCandPStride = kCandTile + 8;
  constexpr size_t kCandKBytes = static_cast<size_t>(kCandTile) * kRowStride * sizeof(__half);
  constexpr float kProbScale = 32768.0f;
  constexpr float kProbInvScale = 1.0f / kProbScale;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* sK = reinterpret_cast<__half*>(smem_raw);
  __half* sQ = sK;
  __half* sV = reinterpret_cast<__half*>(smem_raw + kCandKBytes);
  int32_t* sIdx = reinterpret_cast<int32_t*>(smem_raw + 2 * kCandKBytes);
  float* sM = reinterpret_cast<float*>(sIdx + 3 * kCandTile);
  __half* sP = reinterpret_cast<__half*>(sM + kMaxGroup * 4);
  [[maybe_unused]] __half* sPResidual = sP + kMaxGroup * kCandPStride;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int r0 = lane >> 2;
  const int t = lane & 3;
  const int row = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int batch = row / sequence_length;
  const int s = row - batch * sequence_length;
  const int group = num_heads / kv_num_heads;

  const int64_t total_length = static_cast<int64_t>(seqlens_k[batch]) + 1;
  const bool valid_length = total_length >= sequence_length && total_length <= capacity;
  const int query_position = valid_length ? static_cast<int>(total_length) - sequence_length + s : -1;
  const int total = valid_length ? static_cast<int>(total_length) : 0;
  const int count = valid_length ? max(0, min(selected_counts[row], max_selected)) : 0;
  const int32_t* row_indices = selected_indices + static_cast<int64_t>(row) * max_selected;
  const int64_t kv_head_offset = (static_cast<int64_t>(batch) * kv_num_heads + kv_head) * capacity * kHeadDim;
  const __half* key_base = key + kv_head_offset;
  const __half* value_base = value + kv_head_offset;

  int tile_begin = 0;
  int tiles = (count + kCandTile - 1) / kCandTile;
  if constexpr (kSplitK) {
    const int split = static_cast<int>(blockIdx.z);
    const int tiles_per_split = (tiles + split_count - 1) / split_count;
    const int first = min(tiles, split * tiles_per_split);
    tiles = min(tiles, first + tiles_per_split) - first;
    tile_begin = first;
    if (tiles == 0) {
      if (tid < group) {
        const int partial_block = (row * num_heads + kv_head * group + tid) * split_count + split;
        partial_max[partial_block] = -FLT_MAX;
        partial_sum[partial_block] = 0.0f;
      }
      return;
    }
  }

  const __half* query_group =
      reinterpret_cast<const __half*>(query) + (static_cast<int64_t>(row) * num_heads + kv_head * group) * kHeadDim;
  for (int i = tid; i < kMaxGroup * (kHeadDim / 8); i += kThreads) {
    const int r = i / (kHeadDim / 8);
    const int c = (i % (kHeadDim / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < group) {
      v = *reinterpret_cast<const uint4*>(query_group + static_cast<int64_t>(r) * kHeadDim + c);
    }
    *reinterpret_cast<uint4*>(sQ + r * kRowStride + c) = v;
  }
  __syncthreads();

  uint32_t qf[16][4];
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    LdmatrixX4(qf[k], sQ + (lane & 15) * kRowStride + k * 16 + (lane >> 4) * 8);
  }
  __syncthreads();

  auto resolve_index = [&](int raw) -> int {
    return (raw >= 0 && raw < total && raw <= query_position && raw < capacity) ? raw : -1;
  };

  const int first_candidate = tile_begin * kCandTile;
  const int init_tile = tid / kCandTile;
  const int init_col = tid % kCandTile;
  if (init_tile < 2 && init_tile < tiles) {
    const int candidate = first_candidate + init_tile * kCandTile + init_col;
    sIdx[init_tile * kCandTile + init_col] = candidate < count ? resolve_index(row_indices[candidate]) : -1;
  }
  __syncthreads();

  const uint32_t k_smem_base = SmemAddr(sK);
  const uint32_t v_smem_base = SmemAddr(sV);

  auto issue_tile_half = [&](uint32_t smem_base, const __half* base, int idx_buf) {
#pragma unroll
    for (int it = 0; it < kCandTile / 4; ++it) {
      const int cand = it * 4 + warp;
      const int index = sIdx[idx_buf * kCandTile + cand];
      const bool valid = index >= 0;
      const int64_t source = static_cast<int64_t>(valid ? index : 0) * kHeadDim + lane * 8;
      const uint32_t offset = static_cast<uint32_t>(cand * kRowStride + lane * 8) * sizeof(__half);
      CpAsync16Fp16Zfill(smem_base + offset, base + source, valid);
    }
  };

  if (tiles > 0) {
    issue_tile_half(k_smem_base, key_base, 0);
  }
  CpAsyncCommit();

  float acc_o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    acc_o[n][0] = acc_o[n][1] = acc_o[n][2] = acc_o[n][3] = 0.0f;
  }
  float run_max0 = -FLT_MAX;
  float run_max1 = -FLT_MAX;
  float run_sum0 = 0.0f;
  float run_sum1 = 0.0f;

  int idx_cur = 0;
  int idx_next = 1;
  int idx_next2 = 2;
  for (int tile = 0; tile < tiles; ++tile) {
    CpAsyncWait<0>();
    __syncthreads();

    issue_tile_half(v_smem_base, value_base, idx_cur);
    CpAsyncCommit();

    int raw_next2 = -1;
    const bool stage_next2 = (tile + 2 < tiles) && tid < kCandTile;
    if (stage_next2) {
      const int candidate = first_candidate + (tile + 2) * kCandTile + tid;
      raw_next2 = candidate < count ? row_indices[candidate] : -1;
    }

    if constexpr (kCandTile == 32) {
      float acc_s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        uint32_t b[2];
        LdmatrixX2Fp16(b, sK + (warp * 8 + (lane & 7)) * kRowStride + k * 16 + ((lane >> 3) & 1) * 8);
        MmaFp16(acc_s, qf[k], b[0], b[1]);
      }

      const int cbase = warp * 8 + 2 * t;
      const int2 ia = *reinterpret_cast<const int2*>(sIdx + idx_cur * kCandTile + cbase);
      const bool va0 = ia.x >= 0;
      const bool va1 = ia.y >= 0;
      const float s00 = va0 ? acc_s[0] * scale : -FLT_MAX;
      const float s01 = va1 ? acc_s[1] * scale : -FLT_MAX;
      const float s10 = va0 ? acc_s[2] * scale : -FLT_MAX;
      const float s11 = va1 ? acc_s[3] * scale : -FLT_MAX;
      float m0 = fmaxf(s00, s01);
      float m1 = fmaxf(s10, s11);
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      if (t == 0) {
        sM[r0 * 4 + warp] = m0;
        sM[(r0 + 8) * 4 + warp] = m1;
      }
      __syncthreads();

      const float4 mx0 = *reinterpret_cast<const float4*>(sM + r0 * 4);
      const float4 mx1 = *reinterpret_cast<const float4*>(sM + (r0 + 8) * 4);
      const float new_max0 = fmaxf(run_max0, fmaxf(fmaxf(mx0.x, mx0.y), fmaxf(mx0.z, mx0.w)));
      const float new_max1 = fmaxf(run_max1, fmaxf(fmaxf(mx1.x, mx1.y), fmaxf(mx1.z, mx1.w)));
      const float alpha0 = run_sum0 == 0.0f ? 0.0f : expf(run_max0 - new_max0);
      const float alpha1 = run_sum1 == 0.0f ? 0.0f : expf(run_max1 - new_max1);
      const float p00 = va0 ? expf(s00 - new_max0) : 0.0f;
      const float p01 = va1 ? expf(s01 - new_max0) : 0.0f;
      const float p10 = va0 ? expf(s10 - new_max1) : 0.0f;
      const float p11 = va1 ? expf(s11 - new_max1) : 0.0f;
      run_sum0 = run_sum0 * alpha0 + (p00 + p01);
      run_sum1 = run_sum1 * alpha1 + (p10 + p11);
      run_max0 = new_max0;
      run_max1 = new_max1;

      const float x00 = p00 * kProbScale;
      const float x01 = p01 * kProbScale;
      const float x10 = p10 * kProbScale;
      const float x11 = p11 * kProbScale;
      const __half2 h00 = __floats2half2_rn(x00, x01);
      const __half2 h10 = __floats2half2_rn(x10, x11);
      *reinterpret_cast<__half2*>(sP + r0 * kCandPStride + cbase) = h00;
      *reinterpret_cast<__half2*>(sP + (r0 + 8) * kCandPStride + cbase) = h10;
      if constexpr (kSplitK) {
        const float2 f00 = __half22float2(h00);
        const float2 f10 = __half22float2(h10);
        *reinterpret_cast<__half2*>(sPResidual + r0 * kCandPStride + cbase) =
            __floats2half2_rn(x00 - f00.x, x01 - f00.y);
        *reinterpret_cast<__half2*>(sPResidual + (r0 + 8) * kCandPStride + cbase) =
            __floats2half2_rn(x10 - f10.x, x11 - f10.y);
      }

      CpAsyncWait<0>();
      __syncthreads();
      if (tile + 1 < tiles) {
        issue_tile_half(k_smem_base, key_base, idx_next);
      }
      CpAsyncCommit();

      if (__any_sync(0xffffffffu, alpha0 != 1.0f || alpha1 != 1.0f)) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          acc_o[n][0] *= alpha0;
          acc_o[n][1] *= alpha0;
          acc_o[n][2] *= alpha1;
          acc_o[n][3] *= alpha1;
        }
      }
    } else {
      float acc_s[2][4];
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        acc_s[n][0] = acc_s[n][1] = acc_s[n][2] = acc_s[n][3] = 0.0f;
      }
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        uint32_t b[4];
        LdmatrixX4(b, sK + (warp * 16 + (lane & 7) + (lane >> 4) * 8) * kRowStride + k * 16 + ((lane >> 3) & 1) * 8);
        MmaFp16(acc_s[0], qf[k], b[0], b[1]);
        MmaFp16(acc_s[1], qf[k], b[2], b[3]);
      }

      const int cbase = warp * 16 + 2 * t;
      const int2 ia = *reinterpret_cast<const int2*>(sIdx + idx_cur * kCandTile + cbase);
      const int2 ib = *reinterpret_cast<const int2*>(sIdx + idx_cur * kCandTile + cbase + 8);
      const bool va0 = ia.x >= 0;
      const bool va1 = ia.y >= 0;
      const bool vb0 = ib.x >= 0;
      const bool vb1 = ib.y >= 0;
      const float s00 = va0 ? acc_s[0][0] * scale : -FLT_MAX;
      const float s01 = va1 ? acc_s[0][1] * scale : -FLT_MAX;
      const float s02 = vb0 ? acc_s[1][0] * scale : -FLT_MAX;
      const float s03 = vb1 ? acc_s[1][1] * scale : -FLT_MAX;
      const float s10 = va0 ? acc_s[0][2] * scale : -FLT_MAX;
      const float s11 = va1 ? acc_s[0][3] * scale : -FLT_MAX;
      const float s12 = vb0 ? acc_s[1][2] * scale : -FLT_MAX;
      const float s13 = vb1 ? acc_s[1][3] * scale : -FLT_MAX;
      float m0 = fmaxf(fmaxf(s00, s01), fmaxf(s02, s03));
      float m1 = fmaxf(fmaxf(s10, s11), fmaxf(s12, s13));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      if (t == 0) {
        sM[r0 * 4 + warp] = m0;
        sM[(r0 + 8) * 4 + warp] = m1;
      }
      __syncthreads();

      const float4 mx0 = *reinterpret_cast<const float4*>(sM + r0 * 4);
      const float4 mx1 = *reinterpret_cast<const float4*>(sM + (r0 + 8) * 4);
      const float new_max0 = fmaxf(run_max0, fmaxf(fmaxf(mx0.x, mx0.y), fmaxf(mx0.z, mx0.w)));
      const float new_max1 = fmaxf(run_max1, fmaxf(fmaxf(mx1.x, mx1.y), fmaxf(mx1.z, mx1.w)));
      const float alpha0 = run_sum0 == 0.0f ? 0.0f : expf(run_max0 - new_max0);
      const float alpha1 = run_sum1 == 0.0f ? 0.0f : expf(run_max1 - new_max1);
      const float p00 = va0 ? expf(s00 - new_max0) : 0.0f;
      const float p01 = va1 ? expf(s01 - new_max0) : 0.0f;
      const float p02 = vb0 ? expf(s02 - new_max0) : 0.0f;
      const float p03 = vb1 ? expf(s03 - new_max0) : 0.0f;
      const float p10 = va0 ? expf(s10 - new_max1) : 0.0f;
      const float p11 = va1 ? expf(s11 - new_max1) : 0.0f;
      const float p12 = vb0 ? expf(s12 - new_max1) : 0.0f;
      const float p13 = vb1 ? expf(s13 - new_max1) : 0.0f;
      run_sum0 = run_sum0 * alpha0 + ((p00 + p01) + (p02 + p03));
      run_sum1 = run_sum1 * alpha1 + ((p10 + p11) + (p12 + p13));
      run_max0 = new_max0;
      run_max1 = new_max1;

      const float x00 = p00 * kProbScale;
      const float x01 = p01 * kProbScale;
      const float x02 = p02 * kProbScale;
      const float x03 = p03 * kProbScale;
      const float x10 = p10 * kProbScale;
      const float x11 = p11 * kProbScale;
      const float x12 = p12 * kProbScale;
      const float x13 = p13 * kProbScale;
      const __half2 h00 = __floats2half2_rn(x00, x01);
      const __half2 h01 = __floats2half2_rn(x02, x03);
      const __half2 h10 = __floats2half2_rn(x10, x11);
      const __half2 h11 = __floats2half2_rn(x12, x13);
      *reinterpret_cast<__half2*>(sP + r0 * kCandPStride + cbase) = h00;
      *reinterpret_cast<__half2*>(sP + r0 * kCandPStride + cbase + 8) = h01;
      *reinterpret_cast<__half2*>(sP + (r0 + 8) * kCandPStride + cbase) = h10;
      *reinterpret_cast<__half2*>(sP + (r0 + 8) * kCandPStride + cbase + 8) = h11;
      if constexpr (kSplitK) {
        const float2 f00 = __half22float2(h00);
        const float2 f01 = __half22float2(h01);
        const float2 f10 = __half22float2(h10);
        const float2 f11 = __half22float2(h11);
        *reinterpret_cast<__half2*>(sPResidual + r0 * kCandPStride + cbase) =
            __floats2half2_rn(x00 - f00.x, x01 - f00.y);
        *reinterpret_cast<__half2*>(sPResidual + r0 * kCandPStride + cbase + 8) =
            __floats2half2_rn(x02 - f01.x, x03 - f01.y);
        *reinterpret_cast<__half2*>(sPResidual + (r0 + 8) * kCandPStride + cbase) =
            __floats2half2_rn(x10 - f10.x, x11 - f10.y);
        *reinterpret_cast<__half2*>(sPResidual + (r0 + 8) * kCandPStride + cbase + 8) =
            __floats2half2_rn(x12 - f11.x, x13 - f11.y);
      }

      CpAsyncWait<0>();
      __syncthreads();
      if (tile + 1 < tiles) {
        issue_tile_half(k_smem_base, key_base, idx_next);
      }
      CpAsyncCommit();

      if (__any_sync(0xffffffffu, alpha0 != 1.0f || alpha1 != 1.0f)) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          acc_o[n][0] *= alpha0;
          acc_o[n][1] *= alpha0;
          acc_o[n][2] *= alpha1;
          acc_o[n][3] *= alpha1;
        }
      }
    }

#pragma unroll
    for (int kk = 0; kk < kCandTile / 16; ++kk) {
      uint32_t a[4];
      LdmatrixX4(a, sP + (lane & 15) * kCandPStride + kk * 16 + (lane >> 4) * 8);
      uint32_t a_residual[4] = {0, 0, 0, 0};
      if constexpr (kSplitK) {
        LdmatrixX4(a_residual, sPResidual + (lane & 15) * kCandPStride + kk * 16 + (lane >> 4) * 8);
      }
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        uint32_t b[4];
        LdmatrixX4Trans(b, sV + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kRowStride + warp * 64 + pair * 16 +
                               (lane >> 4) * 8);
        MmaFp16(acc_o[pair * 2], a, b[0], b[1]);
        MmaFp16(acc_o[pair * 2 + 1], a, b[2], b[3]);
        if constexpr (kSplitK) {
          MmaFp16(acc_o[pair * 2], a_residual, b[0], b[1]);
          MmaFp16(acc_o[pair * 2 + 1], a_residual, b[2], b[3]);
        }
      }
    }

    if (stage_next2) {
      sIdx[idx_next2 * kCandTile + tid] = resolve_index(raw_next2);
    }
    idx_cur = idx_next;
    idx_next = idx_next2;
    idx_next2 = idx_next2 + 1 == 3 ? 0 : idx_next2 + 1;
  }

  float l0 = run_sum0;
  float l1 = run_sum1;
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  __syncthreads();
  if (t == 0) {
    sM[r0 * 4 + warp] = l0;
    sM[(r0 + 8) * 4 + warp] = l1;
  }
  __syncthreads();
  const float4 ls0 = *reinterpret_cast<const float4*>(sM + r0 * 4);
  const float4 ls1 = *reinterpret_cast<const float4*>(sM + (r0 + 8) * 4);
  const float sum0 = (ls0.x + ls0.y) + (ls0.z + ls0.w);
  const float sum1 = (ls1.x + ls1.y) + (ls1.z + ls1.w);

  if constexpr (kSplitK) {
    const int split = static_cast<int>(blockIdx.z);
    const int partial_block0 = (row * num_heads + kv_head * group + r0) * split_count + split;
    const int partial_block1 = partial_block0 + 8 * split_count;
    if (warp == 0 && t == 0) {
      if (r0 < group) {
        partial_max[partial_block0] = run_max0;
        partial_sum[partial_block0] = sum0;
      }
      if (r0 + 8 < group) {
        partial_max[partial_block1] = run_max1;
        partial_sum[partial_block1] = sum1;
      }
    }
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int dim = warp * 64 + n * 8 + t * 2;
      if (r0 < group) {
        *reinterpret_cast<float2*>(partial_output + static_cast<int64_t>(partial_block0) * kHeadDim + dim) =
            make_float2(acc_o[n][0] * kProbInvScale, acc_o[n][1] * kProbInvScale);
      }
      if (r0 + 8 < group) {
        *reinterpret_cast<float2*>(partial_output + static_cast<int64_t>(partial_block1) * kHeadDim + dim) =
            make_float2(acc_o[n][2] * kProbInvScale, acc_o[n][3] * kProbInvScale);
      }
    }
  } else {
    const float inv0 = sum0 == 0.0f ? 0.0f : kProbInvScale / sum0;
    const float inv1 = sum1 == 0.0f ? 0.0f : kProbInvScale / sum1;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int dim = warp * 64 + n * 8 + t * 2;
      if (r0 < group) {
        __nv_bfloat16* dst = output + (static_cast<int64_t>(row) * num_heads + kv_head * group + r0) * kHeadDim + dim;
        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc_o[n][0] * inv0, acc_o[n][1] * inv0);
      }
      if (r0 + 8 < group) {
        __nv_bfloat16* dst = output + (static_cast<int64_t>(row) * num_heads + kv_head * group + r0 + 8) * kHeadDim + dim;
        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc_o[n][2] * inv1, acc_o[n][3] * inv1);
      }
    }
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)selected_indices;
  (void)selected_counts;
  (void)seqlens_k;
  (void)output;
  (void)partial_max;
  (void)partial_sum;
  (void)partial_output;
  (void)split_count;
  (void)sequence_length;
  (void)num_heads;
  (void)kv_num_heads;
  (void)capacity;
  (void)max_selected;
  (void)scale;
#endif
}

}  // namespace dsa_prefill
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
