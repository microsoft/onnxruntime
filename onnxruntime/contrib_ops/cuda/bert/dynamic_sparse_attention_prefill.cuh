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
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

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
__device__ __forceinline__ void LdmatrixX4Trans(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(SmemAddr(p)));
}

__device__ __forceinline__ void MmaBf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
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
// query: [rows, num_heads, 256] (already normalized / rotated); key, value: [batch, kv_heads, capacity, 256].
template <bool kQuantizedCache = false>
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
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sQ = sK;
  __nv_bfloat16* sV = sK + kTile * kRowStride;
  float* sS = reinterpret_cast<float*>(sV + kTile * kRowStride);
  __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(sS + kMaxGroup * kSStride);
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
  const __nv_bfloat16* key_base = key + kv_head_offset;
  const __nv_bfloat16* value_base = value + kv_head_offset;
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
  const __nv_bfloat16* query_group = query + (static_cast<int64_t>(row) * num_heads + kv_head * group) * kHeadDim;
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

  auto issue_rows = [&](__nv_bfloat16* dst, const __nv_bfloat16* base, const int8_t* base_i8, int tile) {
  // 4 candidate rows per pass (one per warp), 32 lanes x 16 B = one 512 B row.
#pragma unroll
    for (int it = 0; it < kTile / 4; ++it) {
      const int cand_in_tile = it * 4 + warp;
      const int index = candidate_index(tile * kTile + cand_in_tile);
      __nv_bfloat16* row_dst = dst + cand_in_tile * kRowStride + lane * 8;
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

  auto dequantize_rows = [&](__nv_bfloat16* dst, float4 sc0, float4 sc1) {
    if constexpr (kQuantizedCache) {
#pragma unroll
      for (int it = 0; it < kTile / 4; ++it) {
        const int cand_in_tile = it * 4 + warp;
        __nv_bfloat16* row_dst = dst + cand_in_tile * kRowStride + lane * 8;
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
      MmaBf16(acc_s[0], qf[k], b[0], b[1]);
      MmaBf16(acc_s[1], qf[k], b[2], b[3]);
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
    __nv_bfloat16 p_bf[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float p = s_val[j] <= -FLT_MAX ? 0.0f : __expf(s_val[j] - new_max);
      tile_sum += p;
      p_bf[j] = __float2bfloat16_rn(p);
    }
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 2);
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 4);
    running_sum = running_sum * alpha + tile_sum;
    running_max = new_max;
    *reinterpret_cast<uint4*>(sP + soft_row * kPStride + soft_col) = *reinterpret_cast<const uint4*>(p_bf);
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
        MmaBf16(acc_o[pair * 2], a, b[0], b[1]);
        MmaBf16(acc_o[pair * 2 + 1], a, b[2], b[3]);
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

}  // namespace dsa_prefill
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
