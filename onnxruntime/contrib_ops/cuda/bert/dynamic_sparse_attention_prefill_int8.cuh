// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// INT8-cache variant of the tensor-core prefill kernel of DynamicSparseAttention (selected_only, main KV source,
// bfloat16 query / output, head_size 256). One CTA handles one (query row, KV head); the heads of the GQA group form
// the M dimension of an m16n8k16 MMA (padded to 16).
//
// The scales of the INT8 cache are per (kv head, channel) constants, so they leave the tile loop:
//   * the key scale is folded into the query once per CTA: Q' = Q * key_scale (rounded once to fp16), S = Q' K_int8;
//   * the value scale is applied to the output once per CTA: O = (P V_int8) * value_scale.
// The gathered int8 rows stay int8 in shared memory (cp.async, double buffered, one tile ahead). A fragment read with
// ldmatrix is converted to fp16 exactly in registers (1024 + (byte ^ 0x80) - 1152, "magic number" conversion) and
// consumed by fp16 MMAs with fp32 accumulation, so the per-tile dequantization pass, its shared memory round trip and
// its barriers of the BF16-dequantizing kernel disappear. The scores stay in registers; the only cross-warp exchanges
// per tile are the row maxima and the fp16 probability tile.
//
// The channel (reduction) dimension of S = Q' K^T is permuted so that one 4-byte ldmatrix element of a K row supplies
// both B registers of an MMA k step: slot 2t+e <-> channel 16j + 4t + e and slot 2t+8+e <-> channel 16j + 4t + 2 + e,
// with the same permutation applied to the A fragments of Q'. Likewise ldmatrix.trans on the V tile delivers channels
// 2n and 2n + 1 of two consecutive candidates, i.e. the B registers of two n tiles whose columns are the even / odd
// channels.

#pragma once

#include "contrib_ops/cuda/bert/dynamic_sparse_attention_prefill.cuh"

#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace dsa_prefill {

constexpr int kInt8RowStride = kHeadDim + 16;  // bytes; 272 = 17 x 16 keeps ldmatrix conflict free
constexpr int kInt8TileBytes = kTile * kInt8RowStride;
constexpr int kInt8QStride = kHeadDim + 8;  // fp16 elements of the staged query rows
constexpr int kInt8PStride = kTile + 8;     // fp16 elements
constexpr int kIdxStages = 3;

// sK8[2] + sV8[2] (sQ aliases sK8[0] before the tile loop) + candidate indices + row maxima / sums + probabilities.
constexpr size_t kInt8SharedBytes = 4 * static_cast<size_t>(kInt8TileBytes) + kIdxStages * kTile * sizeof(int32_t) +
                                    kMaxGroup * 4 * sizeof(float) + kMaxGroup * kInt8PStride * sizeof(__half);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void CpAsync16Zfill(uint32_t smem_dst, const void* gmem_src, bool valid) {
  const int src_size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_dst), "l"(gmem_src), "r"(src_size));
}

__device__ __forceinline__ void LdmatrixX4Addr(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void LdmatrixX4TransAddr(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void MmaF16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float FastExp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// Two bytes b0, b1 (offset binary u = b ^ 0x80 in [0, 255]) placed in the mantissa of fp16 1024: 1024 + u; subtracting
// 1152 gives the signed value exactly.
__device__ __forceinline__ uint32_t Half2MinusBias(uint32_t x) {
  uint32_t r;
  asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(x), "r"(0x64806480u));
  return r;
}

// bytes (0, 1) -> lo, bytes (2, 3) -> hi (two fp16 each).
__device__ __forceinline__ void Int8x4ToHalf2Adjacent(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t x = w ^ 0x80808080u;
  uint32_t a;
  uint32_t b;
  asm("prmt.b32 %0, %1, %2, 0x5140;\n" : "=r"(a) : "r"(x), "r"(0x64646464u));
  asm("prmt.b32 %0, %1, %2, 0x5342;\n" : "=r"(b) : "r"(x), "r"(0x64646464u));
  lo = Half2MinusBias(a);
  hi = Half2MinusBias(b);
}

// bytes (0, 2) -> even, bytes (1, 3) -> odd (two fp16 each).
__device__ __forceinline__ void Int8x4ToHalf2Strided(uint32_t w, uint32_t& even, uint32_t& odd) {
  const uint32_t x = w ^ 0x80808080u;
  uint32_t a;
  uint32_t b;
  asm("prmt.b32 %0, %1, %2, 0x5240;\n" : "=r"(a) : "r"(x), "r"(0x64646464u));
  asm("prmt.b32 %0, %1, %2, 0x5341;\n" : "=r"(b) : "r"(x), "r"(0x64646464u));
  even = Half2MinusBias(a);
  odd = Half2MinusBias(b);
}
#endif

// grid = (batch * sequence_length, kv_num_heads), block = kThreads, dynamic smem = kInt8SharedBytes.
// query: [rows, num_heads, 256] (already normalized / rotated); key, value: int8 [batch, kv_heads, capacity, 256];
// key_scale, value_scale: float [kv_heads, 256].
__global__ void __launch_bounds__(kThreads, 3) DynamicSparseAttentionPrefillInt8Kernel(
    const __nv_bfloat16* __restrict__ query,
    const int8_t* __restrict__ key,
    const int8_t* __restrict__ value,
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
    const float* __restrict__ key_scale,
    const float* __restrict__ value_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(16) unsigned char smem_raw[];
  unsigned char* sK8 = smem_raw;
  unsigned char* sV8 = smem_raw + 2 * kInt8TileBytes;
  int32_t* sIdx = reinterpret_cast<int32_t*>(smem_raw + 4 * kInt8TileBytes);
  float* sM = reinterpret_cast<float*>(sIdx + kIdxStages * kTile);
  __half* sP = reinterpret_cast<__half*>(sM + kMaxGroup * 4);
  __half* sQ = reinterpret_cast<__half*>(smem_raw);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int r0 = lane >> 2;  // MMA row of the C / A fragments (and row + 8)
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
  const int8_t* key_base = key + kv_head_offset;
  const int8_t* value_base = value + kv_head_offset;

  // Q' = Q * key_scale as fp16 (clamped so that it cannot overflow); the padding rows are zero so their (ignored)
  // scores stay finite.
  {
    const int chunk = tid & 31;
    const float4 k0 = *reinterpret_cast<const float4*>(key_scale + kv_head * kHeadDim + chunk * 8);
    const float4 k1 = *reinterpret_cast<const float4*>(key_scale + kv_head * kHeadDim + chunk * 8 + 4);
    const __nv_bfloat16* query_group = query + (static_cast<int64_t>(row) * num_heads + kv_head * group) * kHeadDim;
#pragma unroll
    for (int i = 0; i < kMaxGroup / 4; ++i) {
      const int r = (tid >> 5) + 4 * i;
      uint4 packed = make_uint4(0, 0, 0, 0);
      if (r < group) {
        const uint4 q = *reinterpret_cast<const uint4*>(query_group + static_cast<int64_t>(r) * kHeadDim + chunk * 8);
        const float2 q01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&q.x));
        const float2 q23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&q.y));
        const float2 q45 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&q.z));
        const float2 q67 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&q.w));
        auto clamp = [](float x) { return fminf(fmaxf(x, -65504.0f), 65504.0f); };
        const __half2 h01 = __floats2half2_rn(clamp(q01.x * k0.x), clamp(q01.y * k0.y));
        const __half2 h23 = __floats2half2_rn(clamp(q23.x * k0.z), clamp(q23.y * k0.w));
        const __half2 h45 = __floats2half2_rn(clamp(q45.x * k1.x), clamp(q45.y * k1.y));
        const __half2 h67 = __floats2half2_rn(clamp(q67.x * k1.z), clamp(q67.y * k1.w));
        packed.x = *reinterpret_cast<const uint32_t*>(&h01);
        packed.y = *reinterpret_cast<const uint32_t*>(&h23);
        packed.z = *reinterpret_cast<const uint32_t*>(&h45);
        packed.w = *reinterpret_cast<const uint32_t*>(&h67);
      }
      *reinterpret_cast<uint4*>(sQ + r * kInt8QStride + chunk * 8) = packed;
    }
  }
  __syncthreads();

  // A fragments of Q' with the channel permutation described at the top: a0 / a2 hold channels 16j + 4t + {0, 1} /
  // {2, 3} of row r0, a1 / a3 the same of row r0 + 8.
  uint32_t qf[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint2 lo = *reinterpret_cast<const uint2*>(sQ + r0 * kInt8QStride + 16 * j + 4 * t);
    const uint2 hi = *reinterpret_cast<const uint2*>(sQ + (r0 + 8) * kInt8QStride + 16 * j + 4 * t);
    qf[j][0] = lo.x;
    qf[j][1] = hi.x;
    qf[j][2] = lo.y;
    qf[j][3] = hi.y;
  }
  __syncthreads();  // sQ aliases sK8[0]

  auto resolve_index = [&](int raw) -> int {
    return (raw >= 0 && raw < total && raw <= query_position && raw < capacity) ? raw : -1;
  };

  const int tiles = (count + kTile - 1) / kTile;
  if ((tid >> 6) < tiles) {
    const int tile = tid >> 6;
    const int candidate = tile * kTile + (tid & 63);
    sIdx[tile * kTile + (tid & 63)] = candidate < count ? resolve_index(row_indices[candidate]) : -1;
  }
  __syncthreads();

  const uint32_t k_smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(sK8));
  const uint32_t v_smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(sV8));
  const uint32_t p_smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(sP));
  const int load_chunk = tid & 15;

  // 16 B per lane: 16 lanes per candidate row, 8 rows per pass, 8 passes per tile (K and V).
  auto issue_tile = [&](int tile, int stage, int idx_buf) {
#pragma unroll
    for (int i = 0; i < kTile / 8; ++i) {
      const int cand = i * 8 + (tid >> 4);
      const int index = sIdx[idx_buf * kTile + cand];
      const bool valid = index >= 0;
      const int64_t source = static_cast<int64_t>(valid ? index : 0) * kHeadDim + load_chunk * 16;
      const uint32_t offset = static_cast<uint32_t>(stage * kInt8TileBytes + cand * kInt8RowStride + load_chunk * 16);
      CpAsync16Zfill(k_smem_base + offset, key_base + source, valid);
      CpAsync16Zfill(v_smem_base + offset, value_base + source, valid);
    }
  };

  if (tiles > 0) {
    issue_tile(0, 0, 0);
  }
  CpAsyncCommit();

  float acc_o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    acc_o[n][0] = acc_o[n][1] = acc_o[n][2] = acc_o[n][3] = 0.0f;
  }
  // Softmax state of rows r0 / r0 + 8: the running maximum is identical in all threads of the row, the running sum only
  // covers this thread's columns (merged across threads and warps after the loop).
  float run_max0 = -FLT_MAX;
  float run_max1 = -FLT_MAX;
  float run_sum0 = 0.0f;
  float run_sum1 = 0.0f;
  const float scale_log2 = scale * 1.44269504088896341f;

  // Shared memory byte addresses that do not change from tile to tile.
  const uint32_t k_lane_offset =
      static_cast<uint32_t>((warp * 16 + ((lane >> 3) & 1) * 8 + (lane & 7)) * kInt8RowStride + (lane >> 4) * 16);
  const uint32_t v_lane_offset =
      static_cast<uint32_t>((((lane >> 3) & 1) * 8 + (lane & 7)) * kInt8RowStride + (warp * 4 + (lane >> 4)) * 16);
  const uint32_t p_lane_offset = static_cast<uint32_t>(((lane & 15) * kInt8PStride + (lane >> 4) * 8) * sizeof(__half));

  int idx_cur = 0;                 // tile % 3
  int idx_next = 1 % kIdxStages;   // (tile + 1) % 3
  int idx_next2 = 2 % kIdxStages;  // (tile + 2) % 3
  for (int tile = 0; tile < tiles; ++tile) {
    CpAsyncWait<0>();  // K / V of this tile have landed (this thread's copies)
    __syncthreads();   // (1) everyone's copies are visible; the previous tile's smem consumers are done

    const int stage = tile & 1;
    if (tile + 1 < tiles) {
      issue_tile(tile + 1, stage ^ 1, idx_next);
    }
    CpAsyncCommit();
    // Raw indices of tile + 2: loaded now, resolved and stored at the end of the iteration (hides the load latency).
    int raw_next2 = -1;
    const bool stage_next2 = (tile + 2 < tiles) && tid < kTile;
    if (stage_next2) {
      const int candidate = (tile + 2) * kTile + tid;
      raw_next2 = candidate < count ? row_indices[candidate] : -1;
    }

    // S = Q' K^T for this warp's 16 candidates (two n tiles of 8).
    float acc_s[2][4];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      acc_s[n][0] = acc_s[n][1] = acc_s[n][2] = acc_s[n][3] = 0.0f;
    }
    {
      const uint32_t k_addr = k_smem_base + stage * kInt8TileBytes + k_lane_offset;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        uint32_t r[4];
        LdmatrixX4Addr(r, k_addr + jj * 32);
        uint32_t b0;
        uint32_t b1;
        Int8x4ToHalf2Adjacent(r[0], b0, b1);
        MmaF16(acc_s[0], qf[2 * jj], b0, b1);
        Int8x4ToHalf2Adjacent(r[1], b0, b1);
        MmaF16(acc_s[1], qf[2 * jj], b0, b1);
        Int8x4ToHalf2Adjacent(r[2], b0, b1);
        MmaF16(acc_s[0], qf[2 * jj + 1], b0, b1);
        Int8x4ToHalf2Adjacent(r[3], b0, b1);
        MmaF16(acc_s[1], qf[2 * jj + 1], b0, b1);
      }
    }

    // Scores in the log2 domain; invalid candidates are masked.
    const int cbase = warp * 16 + 2 * t;
    const int2 ia = *reinterpret_cast<const int2*>(sIdx + idx_cur * kTile + cbase);
    const int2 ib = *reinterpret_cast<const int2*>(sIdx + idx_cur * kTile + cbase + 8);
    const bool va0 = ia.x >= 0;
    const bool va1 = ia.y >= 0;
    const bool vb0 = ib.x >= 0;
    const bool vb1 = ib.y >= 0;
    const float s00 = va0 ? acc_s[0][0] * scale_log2 : -FLT_MAX;
    const float s01 = va1 ? acc_s[0][1] * scale_log2 : -FLT_MAX;
    const float s02 = vb0 ? acc_s[1][0] * scale_log2 : -FLT_MAX;
    const float s03 = vb1 ? acc_s[1][1] * scale_log2 : -FLT_MAX;
    const float s10 = va0 ? acc_s[0][2] * scale_log2 : -FLT_MAX;
    const float s11 = va1 ? acc_s[0][3] * scale_log2 : -FLT_MAX;
    const float s12 = vb0 ? acc_s[1][2] * scale_log2 : -FLT_MAX;
    const float s13 = vb1 ? acc_s[1][3] * scale_log2 : -FLT_MAX;
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
    __syncthreads();  // (2) the four warps' row maxima are visible

    const float4 mx0 = *reinterpret_cast<const float4*>(sM + r0 * 4);
    const float4 mx1 = *reinterpret_cast<const float4*>(sM + (r0 + 8) * 4);
    const float new_max0 = fmaxf(run_max0, fmaxf(fmaxf(mx0.x, mx0.y), fmaxf(mx0.z, mx0.w)));
    const float new_max1 = fmaxf(run_max1, fmaxf(fmaxf(mx1.x, mx1.y), fmaxf(mx1.z, mx1.w)));
    const float alpha0 = FastExp2(run_max0 - new_max0);
    const float alpha1 = FastExp2(run_max1 - new_max1);
    const float p00 = va0 ? FastExp2(s00 - new_max0) : 0.0f;
    const float p01 = va1 ? FastExp2(s01 - new_max0) : 0.0f;
    const float p02 = vb0 ? FastExp2(s02 - new_max0) : 0.0f;
    const float p03 = vb1 ? FastExp2(s03 - new_max0) : 0.0f;
    const float p10 = va0 ? FastExp2(s10 - new_max1) : 0.0f;
    const float p11 = va1 ? FastExp2(s11 - new_max1) : 0.0f;
    const float p12 = vb0 ? FastExp2(s12 - new_max1) : 0.0f;
    const float p13 = vb1 ? FastExp2(s13 - new_max1) : 0.0f;
    run_sum0 = run_sum0 * alpha0 + ((p00 + p01) + (p02 + p03));
    run_sum1 = run_sum1 * alpha1 + ((p10 + p11) + (p12 + p13));
    run_max0 = new_max0;
    run_max1 = new_max1;
    {
      const __half2 h00 = __floats2half2_rn(p00, p01);
      const __half2 h01 = __floats2half2_rn(p02, p03);
      const __half2 h10 = __floats2half2_rn(p10, p11);
      const __half2 h11 = __floats2half2_rn(p12, p13);
      *reinterpret_cast<__half2*>(sP + r0 * kInt8PStride + cbase) = h00;
      *reinterpret_cast<__half2*>(sP + r0 * kInt8PStride + cbase + 8) = h01;
      *reinterpret_cast<__half2*>(sP + (r0 + 8) * kInt8PStride + cbase) = h10;
      *reinterpret_cast<__half2*>(sP + (r0 + 8) * kInt8PStride + cbase + 8) = h11;
    }
    __syncthreads();  // (3) the probability tile is visible

    // The maximum rarely changes after the first tiles: skip the accumulator rescale when the whole warp is unchanged.
    if (__any_sync(0xffffffffu, alpha0 != 1.0f || alpha1 != 1.0f)) {
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        acc_o[n][0] *= alpha0;
        acc_o[n][1] *= alpha0;
        acc_o[n][2] *= alpha1;
        acc_o[n][3] *= alpha1;
      }
    }

    // O += P V for this warp's 64 output channels (four 16-channel chunks, each an even and an odd n tile).
    {
      const uint32_t v_addr = v_smem_base + stage * kInt8TileBytes + v_lane_offset;
#pragma unroll
      for (int kk = 0; kk < kTile / 16; ++kk) {
        uint32_t a[4];
        LdmatrixX4Addr(a, p_smem_base + p_lane_offset + kk * 32);
#pragma unroll
        for (int cp = 0; cp < 2; ++cp) {
          uint32_t r[4];
          LdmatrixX4TransAddr(r, v_addr + kk * 16 * kInt8RowStride + cp * 32);
          uint32_t e0;
          uint32_t o0;
          uint32_t e1;
          uint32_t o1;
          // r[0] / r[1]: candidates 0-7 / 8-15 of the k step, chunk 2cp; r[2] / r[3]: the same for chunk 2cp + 1.
          Int8x4ToHalf2Strided(r[0], e0, o0);
          Int8x4ToHalf2Strided(r[1], e1, o1);
          MmaF16(acc_o[4 * cp], a, e0, e1);
          MmaF16(acc_o[4 * cp + 1], a, o0, o1);
          Int8x4ToHalf2Strided(r[2], e0, o0);
          Int8x4ToHalf2Strided(r[3], e1, o1);
          MmaF16(acc_o[4 * cp + 2], a, e0, e1);
          MmaF16(acc_o[4 * cp + 3], a, o0, o1);
        }
      }
    }

    if (stage_next2) {
      sIdx[idx_next2 * kTile + tid] = resolve_index(raw_next2);
    }
    idx_cur = idx_next;
    idx_next = idx_next2;
    idx_next2 = idx_next2 + 1 == kIdxStages ? 0 : idx_next2 + 1;
  }

  // Merge the per-thread running sums: across the four threads of a row, then across the four warps.
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
  const float inv0 = sum0 == 0.0f ? 0.0f : 1.0f / sum0;
  const float inv1 = sum1 == 0.0f ? 0.0f : 1.0f / sum1;

  // Thread t owns channels 64 * warp + 16 * chunk + 4t + {0, 1, 2, 3}
  // = (even tile c0, odd tile c0, even tile c1, odd tile c1).
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    const int dim = warp * 64 + chunk * 16 + t * 4;
    const float4 vs = *reinterpret_cast<const float4*>(value_scale + kv_head * kHeadDim + dim);
    if (r0 < group) {
      __nv_bfloat16* dst = output + (static_cast<int64_t>(row) * num_heads + kv_head * group + r0) * kHeadDim + dim;
      const __nv_bfloat162 lo =
          __floats2bfloat162_rn(acc_o[2 * chunk][0] * inv0 * vs.x, acc_o[2 * chunk + 1][0] * inv0 * vs.y);
      const __nv_bfloat162 hi =
          __floats2bfloat162_rn(acc_o[2 * chunk][1] * inv0 * vs.z, acc_o[2 * chunk + 1][1] * inv0 * vs.w);
      uint2 packed;
      packed.x = *reinterpret_cast<const uint32_t*>(&lo);
      packed.y = *reinterpret_cast<const uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(dst) = packed;
    }
    if (r0 + 8 < group) {
      __nv_bfloat16* dst = output + (static_cast<int64_t>(row) * num_heads + kv_head * group + r0 + 8) * kHeadDim + dim;
      const __nv_bfloat162 lo =
          __floats2bfloat162_rn(acc_o[2 * chunk][2] * inv1 * vs.x, acc_o[2 * chunk + 1][2] * inv1 * vs.y);
      const __nv_bfloat162 hi =
          __floats2bfloat162_rn(acc_o[2 * chunk][3] * inv1 * vs.z, acc_o[2 * chunk + 1][3] * inv1 * vs.w);
      uint2 packed;
      packed.x = *reinterpret_cast<const uint32_t*>(&lo);
      packed.y = *reinterpret_cast<const uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(dst) = packed;
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
