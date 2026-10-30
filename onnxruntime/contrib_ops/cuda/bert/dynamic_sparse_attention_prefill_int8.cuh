// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// INT8-cache variant of the tensor-core prefill kernel of DynamicSparseAttention (selected_only, main KV source,
// bfloat16 query / output, head_size 256). One CTA handles one (query row, KV head); the heads of the GQA group form
// the M dimension of an m16n8k16 MMA (padded to 16).
//
// The same body also serves decode (kSplitK = true): the grid gets a third dimension, each CTA consumes a contiguous
// range of the row's 64-candidate tiles and writes an un-normalized partial result (running maximum in the natural
// logarithm domain, running sum, value-scaled accumulator) in the layout of the grouped decode kernel, which the
// existing merge kernel combines. This replaces the CUDA-core grouped decode kernel, whose per-candidate dot product,
// warp reduction and exponentials made it issue-bound, with the tensor-core pipeline below.
//
// The scales of the INT8 cache are per (kv head, channel) constants, so they leave the tile loop:
//   * the key scale is folded into the query once per CTA: Q' = Q * key_scale (rounded once to fp16), S = Q' K_int8;
//     a CTA whose Q' does not fit fp16 (or is not finite) evaluates its query heads in fp32 instead (AttendRowFp32).
//     Products below 2^-25 flush to zero, which moves a score by at most 2^-10 * scale (256 channels, |K| <= 128), so
//     the dispatcher selects this kernel only for scale <= kInt8TcMaxScale;
//   * the value scale is applied to the output once per CTA: O = (P V_int8) * value_scale; a CTA whose value_scale
//     exceeds kInt8TcMaxValueScale (so that a probability flushed below 2^-40 by the fp16 probability tile could
//     still move the scaled output) is also evaluated in fp32 by AttendRowFp32.
// The gathered int8 rows stay int8 in shared memory (cp.async, double buffered, one tile ahead). A fragment read with
// ldmatrix is converted to fp16 exactly in registers (1024 + (byte ^ 0x80) - 1152, "magic number" conversion) and
// consumed by fp16 MMAs with fp32 accumulation, so the per-tile dequantization pass, its shared memory round trip and
// its barriers of the BF16-dequantizing kernel disappear. The scores stay in registers; the only cross-warp exchanges
// per tile are the row maxima and the fp16 probability tile, which holds kInt8ProbScale * p so that the tail of the
// softmax stays out of the fp16 subnormal range.
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

// The fp16 probability tile holds kInt8ProbScale * p for p in [0, 1]. Without the factor a candidate whose probability
// is below 2^-25 rounds to zero although a large value scale can make its contribution matter; with it the flush point
// drops to 2^-40 and the tile stays below the fp16 maximum. The PV accumulator carries the factor until the epilogue
// removes it; the softmax denominator is accumulated in fp32 without it. When any channel has
// |value_scale| > kInt8TcMaxValueScale (so that |p * V * value_scale| for p <= 2^-40 can exceed 2^-17), the CTA takes
// AttendRowFp32 instead.
constexpr float kInt8ProbScale = 32768.0f;  // 2^15
constexpr float kInt8ProbInverseScale = 1.0f / kInt8ProbScale;
constexpr float kInt8TcMaxValueScale = 65536.0f;  // 2^16

// Largest softmax scale served by the kernel below (the op validates it as finite and non-negative). Rounding
// Q * key_scale to fp16 flushes products below 2^-25 to zero, which moves a dot product by at most
// 256 * 128 * 2^-25 = 2^-10 before the scale multiplies it; a larger scale would amplify that error, so the dispatcher
// uses the BF16-dequantizing prefill / CUDA-core decode kernels for it.
constexpr float kInt8TcMaxScale = 1.0f;

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

// FP32 evaluation of the query heads of one (row, KV head), one warp per head and eight channels per lane, for rows
// whose Q * key_scale leaves the FP16 range of the tensor-core path or whose value_scale exceeds kInt8TcMaxValueScale.
// Clipping Q * key_scale would break the cancellation between large terms, and a larger value_scale can amplify
// probabilities below the 2^-40 FP16 probability-tile flush boundary. `query` and `output` point at the first head of
// the group; `key_scale` and `value_scale` at the KV head. The candidates [begin, end) of `row_indices` are consumed:
// all of them for prefill, the range of the CTA for a decode split. With kSplitK the un-normalized partial (running
// maximum in the natural logarithm domain, running sum, value-scaled accumulator) of block
// partial_block + head * split_count goes to partial_*, as the main path does; otherwise the normalized result goes to
// output.
template <bool kSplitK>
__device__ __noinline__ void AttendRowFp32(const __nv_bfloat16* __restrict__ query, const int8_t* __restrict__ key,
                                           const int8_t* __restrict__ value, const float* __restrict__ key_scale,
                                           const float* __restrict__ value_scale,
                                           const int32_t* __restrict__ row_indices, int begin, int end, int total,
                                           int query_position, float scale, int group,
                                           __nv_bfloat16* __restrict__ output, float* __restrict__ partial_max,
                                           float* __restrict__ partial_sum, float* __restrict__ partial_output,
                                           int partial_block, int split_count, int warp, int lane) {
  float ks[8];
  float vs[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    ks[j] = key_scale[lane + j * 32];
    vs[j] = value_scale[lane + j * 32];
  }
  for (int h = warp; h < group; h += kThreads / 32) {
    float q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      q[j] = __bfloat162float(query[static_cast<int64_t>(h) * kHeadDim + lane + j * 32]);
    }
    float numerator[8] = {};
    float maximum = -FLT_MAX;
    float denominator = 0.0f;
    for (int i = begin; i < end; ++i) {
      const int index = row_indices[i];
      if (index < 0 || index >= total || index > query_position) {
        continue;
      }
      const int64_t offset = static_cast<int64_t>(index) * kHeadDim + lane;
      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        dot += q[j] * (static_cast<float>(key[offset + j * 32]) * ks[j]);
      }
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2) {
        dot += __shfl_xor_sync(0xffffffffu, dot, delta);
      }
      const float score = dot * scale;
      const float next_max = fmaxf(maximum, score);
      const float correction = expf(maximum - next_max);
      const float diff = score - next_max;
      // Shift by 2^80 before expf when diff < -80 so tiny probabilities that value_scale brings back into range
      // do not underflow fp32 before multiplying by vs[j].
      const bool shift_exp = diff < -80.0f;
      const float probability = expf(shift_exp ? diff + 80.0f * 0.693147180559945309f : diff);
      const float prob_unscale = shift_exp ? 0x1p-80f : 1.0f;
      denominator = denominator * correction + (shift_exp ? 0.0f : probability);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float pv = probability * static_cast<float>(value[offset + j * 32]);
        numerator[j] = numerator[j] * correction + pv * (vs[j] * prob_unscale);
      }
      maximum = next_max;
    }
    if constexpr (kSplitK) {
      const int block = partial_block + h * split_count;
      if (lane == 0) {
        partial_max[block] = maximum;
        partial_sum[block] = denominator;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int d = lane + j * 32;
        partial_output[static_cast<int64_t>(block) * kHeadDim + d] = numerator[j];
      }
    } else {
      const float inverse = denominator == 0.0f ? 0.0f : 1.0f / denominator;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int d = lane + j * 32;
        output[static_cast<int64_t>(h) * kHeadDim + d] = __float2bfloat16_rn(numerator[j] * inverse);
      }
    }
  }
}
#endif

// grid = (batch * sequence_length, kv_num_heads) or, with kSplitK, (batch * sequence_length, kv_num_heads,
// split_count); block = kThreads, dynamic smem = kInt8SharedBytes.
// query: [rows, num_heads, 256] (already normalized / rotated); key, value: int8 [batch, kv_heads, capacity, 256];
// key_scale, value_scale: float [kv_heads, 256]. The normalized result goes to output; with kSplitK the partials
// go to partial_max / partial_sum [rows * num_heads * split_count] and partial_output [.., 256] instead, indexed
// (row * num_heads + head) * split_count + split (output is unused; partial_output must be 16-byte aligned).
template <bool kSplitK>
__global__ void __launch_bounds__(kThreads, 3) DynamicSparseAttentionInt8TcKernel(
    const __nv_bfloat16* __restrict__ query,
    const int8_t* __restrict__ key,
    const int8_t* __restrict__ value,
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

  // The 64-candidate tiles this CTA consumes: all of the row's for prefill, a contiguous range for a decode split.
  // A split without tiles only publishes an empty partial (maximum -FLT_MAX, sum 0), which the merge skips.
  int tile_begin = 0;
  int tiles = (count + kTile - 1) / kTile;
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

  // Q' = Q * key_scale as fp16; the padding rows are zero so their (ignored) scores stay finite. A row whose Q' does
  // not fit fp16 (including inf / NaN) or whose value_scale exceeds kInt8TcMaxValueScale is evaluated in fp32 by the
  // whole CTA instead. Underflow to fp16 subnormals changes a product by at most 2^-25, i.e. a score by at most
  // 2^-10 * scale, which the dispatcher keeps at the level of the fp16 rounding by requiring scale <= kInt8TcMaxScale.
  const __nv_bfloat16* query_group = query + (static_cast<int64_t>(row) * num_heads + kv_head * group) * kHeadDim;
  bool needs_fp32 = false;
  {
    const int chunk = tid & 31;
    const float4 k0 = *reinterpret_cast<const float4*>(key_scale + kv_head * kHeadDim + chunk * 8);
    const float4 k1 = *reinterpret_cast<const float4*>(key_scale + kv_head * kHeadDim + chunk * 8 + 4);
    if (tid < kHeadDim / 4) {
      const float4 v = *reinterpret_cast<const float4*>(value_scale + kv_head * kHeadDim + tid * 4);
      needs_fp32 = !(fabsf(v.x) <= kInt8TcMaxValueScale) || !(fabsf(v.y) <= kInt8TcMaxValueScale) ||
                   !(fabsf(v.z) <= kInt8TcMaxValueScale) || !(fabsf(v.w) <= kInt8TcMaxValueScale);
    }
    auto to_half2 = [&needs_fp32](float x, float y) {
      needs_fp32 |= !(fabsf(x) <= 65504.0f) || !(fabsf(y) <= 65504.0f);
      return __floats2half2_rn(x, y);
    };
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
        const __half2 h01 = to_half2(q01.x * k0.x, q01.y * k0.y);
        const __half2 h23 = to_half2(q23.x * k0.z, q23.y * k0.w);
        const __half2 h45 = to_half2(q45.x * k1.x, q45.y * k1.y);
        const __half2 h67 = to_half2(q67.x * k1.z, q67.y * k1.w);
        packed.x = *reinterpret_cast<const uint32_t*>(&h01);
        packed.y = *reinterpret_cast<const uint32_t*>(&h23);
        packed.z = *reinterpret_cast<const uint32_t*>(&h45);
        packed.w = *reinterpret_cast<const uint32_t*>(&h67);
      }
      *reinterpret_cast<uint4*>(sQ + r * kInt8QStride + chunk * 8) = packed;
    }
  }
  if (__syncthreads_or(needs_fp32)) {
    // The condition is uniform across the CTA and no cp.async is in flight yet.
    const float* kv_key_scale = key_scale + kv_head * kHeadDim;
    const float* kv_value_scale = value_scale + kv_head * kHeadDim;
    if constexpr (kSplitK) {
      const int partial_block = (row * num_heads + kv_head * group) * split_count + static_cast<int>(blockIdx.z);
      AttendRowFp32<true>(query_group, key_base, value_base, kv_key_scale, kv_value_scale, row_indices,
                          tile_begin * kTile, min(count, (tile_begin + tiles) * kTile), total, query_position, scale,
                          group, nullptr, partial_max, partial_sum, partial_output, partial_block, split_count, warp,
                          lane);
    } else {
      AttendRowFp32<false>(query_group, key_base, value_base, kv_key_scale, kv_value_scale, row_indices, 0, count,
                           total, query_position, scale, group,
                           output + (static_cast<int64_t>(row) * num_heads + kv_head * group) * kHeadDim, nullptr,
                           nullptr, nullptr, 0, 1, warp, lane);
    }
    return;
  }

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

  const int first_candidate = tile_begin * kTile;
  if ((tid >> 6) < tiles) {
    const int tile = tid >> 6;
    const int candidate = first_candidate + tile * kTile + (tid & 63);
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
      const int candidate = first_candidate + (tile + 2) * kTile + tid;
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
      const __half2 h00 = __floats2half2_rn(p00 * kInt8ProbScale, p01 * kInt8ProbScale);
      const __half2 h01 = __floats2half2_rn(p02 * kInt8ProbScale, p03 * kInt8ProbScale);
      const __half2 h10 = __floats2half2_rn(p10 * kInt8ProbScale, p11 * kInt8ProbScale);
      const __half2 h11 = __floats2half2_rn(p12 * kInt8ProbScale, p13 * kInt8ProbScale);
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
  // acc_o carries kInt8ProbScale, which the normalization removes.
  const float inv0 = sum0 == 0.0f ? 0.0f : kInt8ProbInverseScale / sum0;
  const float inv1 = sum1 == 0.0f ? 0.0f : kInt8ProbInverseScale / sum1;

  if constexpr (kSplitK) {
    // Un-normalized partial of this split in the natural logarithm domain, as the merge kernel expects. Thread t owns
    // channels 64 * warp + 16 * chunk + 4t + {0, 1, 2, 3} (see below), i.e. one float4 per chunk and row.
    constexpr float kLn2 = 0.693147180559945309f;
    const int split = static_cast<int>(blockIdx.z);
    const int partial_block0 = (row * num_heads + kv_head * group + r0) * split_count + split;
    const int partial_block1 = partial_block0 + 8 * split_count;
    if (warp == 0 && t == 0) {
      if (r0 < group) {
        partial_max[partial_block0] = run_max0 * kLn2;
        partial_sum[partial_block0] = sum0;
      }
      if (r0 + 8 < group) {
        partial_max[partial_block1] = run_max1 * kLn2;
        partial_sum[partial_block1] = sum1;
      }
    }
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      const int dim = warp * 64 + chunk * 16 + t * 4;
      // acc_o carries kInt8ProbScale (the partial_sum above does not), which the value scale takes out.
      const float4 vs_raw = *reinterpret_cast<const float4*>(value_scale + kv_head * kHeadDim + dim);
      const float4 vs = make_float4(vs_raw.x * kInt8ProbInverseScale, vs_raw.y * kInt8ProbInverseScale,
                                    vs_raw.z * kInt8ProbInverseScale, vs_raw.w * kInt8ProbInverseScale);
      if (r0 < group) {
        *reinterpret_cast<float4*>(partial_output + static_cast<int64_t>(partial_block0) * kHeadDim + dim) =
            make_float4(acc_o[2 * chunk][0] * vs.x, acc_o[2 * chunk + 1][0] * vs.y, acc_o[2 * chunk][1] * vs.z,
                        acc_o[2 * chunk + 1][1] * vs.w);
      }
      if (r0 + 8 < group) {
        *reinterpret_cast<float4*>(partial_output + static_cast<int64_t>(partial_block1) * kHeadDim + dim) =
            make_float4(acc_o[2 * chunk][2] * vs.x, acc_o[2 * chunk + 1][2] * vs.y, acc_o[2 * chunk][3] * vs.z,
                        acc_o[2 * chunk + 1][3] * vs.w);
      }
    }
  } else {
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
        __nv_bfloat16* dst =
            output + (static_cast<int64_t>(row) * num_heads + kv_head * group + r0 + 8) * kHeadDim + dim;
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
  (void)key_scale;
  (void)value_scale;
#endif
}

}  // namespace dsa_prefill
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
