// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tensor-core (mma.sync.m16n8k16, BF16 inputs, FP32 accumulation) dense GEMV for BF16 activations and INT8
// symmetric group-wise weights in the SM80-interleaved layout of fpA_intB_gemv (ColumnMajorInterleaved, 64-wide K
// tiles, two real columns per interleaved row), for 1..8 activation rows.
//
// The scalar kernel in dispatcher.h is instruction-issue bound on the decode projections (ncu on the Qwen3.8 INT8
// in_proj, H200: ~11 thread instructions and an FMA pipe at 62% for ~30% of the DRAM bandwidth), because every weight
// is converted, scaled, repacked and multiplied once per activation row. Here the weights are the mma A operand and the
// (up to 8) activation rows the B operand, so a weight costs 0.5 FMA-pipe instructions of dequantization plus its share
// of the mma, independent of the row count.
//
// Mapping (per warp and 64-wide K tile; mma m16n8k16, so four mma per tile per 16 columns):
//   * A rows 0..15 are the 16 real columns of a column tile. Lane (g, t) owns columns g (row g) and g + 8 (row g + 8).
//   * Lane (g, t) loads, with one LDG.128 per column, the 16 consecutive K of the weight layout's chunk t of the tile.
//     Word w of that chunk converts to bf16x2 pairs holding logical K {2w, 2w + 1} and {2w + 8, 2w + 9}, which are the
//     k-slot pairs 2t.. and 2t + 8.. of mma number w. The contraction order inside an mma is free as long as A and B
//     agree, so the matching activation pairs are the words w and w + 4 of the 32 bytes the lane loads from row g
//     (B column g) at the same K.
//   * Warps split the K tiles (tile i of warp w is w + i * KWarps) and the partial sums are reduced through shared
//     memory in a fixed order.
//
// Dequantization. The stored byte is q = value + 128. Placed alone in a bf16 lane, the bits 0x00qq are exactly
// q * 2^-133 (bf16 subnormals continue linearly into the first normal binade, so this holds for all 0 <= q <= 255).
// One fma.rn.bf16x2 with the per-column constants S = scale * 2^66 and C = -scale * 2^-60 then gives
// q * scale * 2^-67 - 128 * scale * 2^-67 = (q - 128) * scale * 2^-67 with a single rounding, which is the same bf16
// rounding of (q - 128) * scale that the scalar kernel and the CUTLASS path apply, times the exact power of two 2^-67.
// The factor keeps every intermediate normal, and the FP32 accumulators are multiplied by 2^67 at the end.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace onnxruntime::llm::kernels::fpA_intB_gemv {
namespace dense_tc {

constexpr int kMaxRows = 8;      // mma N
constexpr int kTileK = 64;       // K consumed per warp step
constexpr int kMmaColumns = 16;  // real columns per mma A tile

#if defined(__CUDACC__)

__device__ __forceinline__ uint4 LoadWeights(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint4 LoadActivations(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint32_t LoadScale(const void* ptr) {
  unsigned short v;
  asm volatile("ld.global.nc.u16 %0, [%1];\n" : "=h"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint32_t Bf16x2Fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ void MmaBf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// bf16x2 {2^66, 2^66} and {-2^-60, -2^-60}: the per-column constants S = scale * 2^66 and C = -scale * 2^-60.
constexpr uint32_t kScaleUp = 0x60806080u;
constexpr uint32_t kBiasFold = 0xA180A180u;
constexpr float kUnscale = 0x1p67f;

// bf16x2 {(q_lo - 128) * scale * 2^-67, (q_hi - 128) * scale * 2^-67} for bytes {q_lo, q_hi} of the lanes of `bytes`
// (the two bytes are already placed in the low byte of each 16-bit lane).
__device__ __forceinline__ uint32_t Dequant(uint32_t bytes, uint32_t scale_up, uint32_t bias_fold) {
  return Bf16x2Fma(bytes, scale_up, bias_fold);
}

template <int ColTiles>
struct Stage {
  uint4 w[ColTiles][2];  // weights of column g and g + 8 of every column tile
  uint4 a[2];            // activation K 0..7 and 8..15 of the lane's chunk
  uint32_t s[ColTiles][2];
};

// Block: KWarps warps; ColTiles * 16 columns; every warp walks the K tiles w, w + KWarps, ... with Depth tiles of loads
// in flight.
template <int KWarps, int ColTiles, int GroupSize, int Depth, int MinBlocks>
__global__ void __launch_bounds__(KWarps * 32, MinBlocks)
    Int8TcGemvKernel(const __nv_bfloat16* __restrict__ act, const uint8_t* __restrict__ weight,
                     const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ out, int m, int n, int k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#if (__CUDA_ARCH__ >= 900)
  // Lets a programmatic-dependent-launch successor start and prefetch its constants while this kernel runs; it reads
  // our output only after griddepcontrol.wait. No-op for plain launches.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  static_assert(GroupSize == 32 || GroupSize == 64 || GroupSize == 128, "unsupported group size");

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int col0 = blockIdx.x * (ColTiles * kMmaColumns);
  const int num_tiles = k / kTileK;
  const int iters = (num_tiles - warp + KWarps - 1) / KWarps;

  // An interleaved row holds the two real columns of a pair (k bytes each); inside a 64-wide K tile the two columns are
  // consecutive 64-byte segments, and a segment is four 16-byte chunks.
  const int64_t row_bytes = 2 * static_cast<int64_t>(k);
  const uint8_t* wp[ColTiles][2];
  const __nv_bfloat16* sp[ColTiles];
#pragma unroll
  for (int ct = 0; ct < ColTiles; ++ct) {
    const int64_t pair = (col0 >> 1) + ct * 8 + (g >> 1);
    wp[ct][0] = weight + pair * row_bytes + (g & 1) * 64 + t * 16 + warp * 128;
    wp[ct][1] = wp[ct][0] + 4 * row_bytes;
    sp[ct] = scales + col0 + ct * kMmaColumns + g;
  }
  const bool act_valid = g < m;
  const __nv_bfloat16* ap = act + static_cast<int64_t>(g) * k + warp * kTileK + t * 16;

  Stage<ColTiles> st[Depth];

  auto load = [&](Stage<ColTiles>& s, int i) {
    const int tile = warp + i * KWarps;
#pragma unroll
    for (int ct = 0; ct < ColTiles; ++ct) {
      s.w[ct][0] = LoadWeights(wp[ct][0] + static_cast<int64_t>(i) * (KWarps * 128));
      s.w[ct][1] = LoadWeights(wp[ct][1] + static_cast<int64_t>(i) * (KWarps * 128));
    }
    if (act_valid) {
      s.a[0] = LoadActivations(ap + static_cast<int64_t>(i) * (KWarps * kTileK));
      s.a[1] = LoadActivations(ap + static_cast<int64_t>(i) * (KWarps * kTileK) + 8);
    } else {
      s.a[0] = make_uint4(0, 0, 0, 0);
      s.a[1] = make_uint4(0, 0, 0, 0);
    }
    const int64_t group_row = static_cast<int64_t>((tile * kTileK + t * 16) / GroupSize) * n;
#pragma unroll
    for (int ct = 0; ct < ColTiles; ++ct) {
      s.s[ct][0] = LoadScale(sp[ct] + group_row);
      s.s[ct][1] = LoadScale(sp[ct] + group_row + 8);
    }
  };

  float acc[2][ColTiles][4];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int ct = 0; ct < ColTiles; ++ct) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        acc[j][ct][c] = 0.f;
      }
    }
  }

  auto consume = [&](const Stage<ColTiles>& s) {
    const uint32_t b0[4] = {s.a[0].x, s.a[0].y, s.a[0].z, s.a[0].w};
    const uint32_t b1[4] = {s.a[1].x, s.a[1].y, s.a[1].z, s.a[1].w};
#pragma unroll
    for (int ct = 0; ct < ColTiles; ++ct) {
      uint32_t scale_up[2], bias_fold[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t sc2 = __byte_perm(s.s[ct][h], s.s[ct][h], 0x1010);
        scale_up[h] = Bf16x2Fma(sc2, kScaleUp, 0x80008000u);
        bias_fold[h] = Bf16x2Fma(sc2, kBiasFold, 0x80008000u);
      }
      const uint32_t wl[4] = {s.w[ct][0].x, s.w[ct][0].y, s.w[ct][0].z, s.w[ct][0].w};
      const uint32_t wh[4] = {s.w[ct][1].x, s.w[ct][1].y, s.w[ct][1].z, s.w[ct][1].w};
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        // Bytes {0, 2} of a word are logical K {2w, 2w + 1}; bytes {1, 3} are {2w + 8, 2w + 9}.
        const uint32_t a0 = Dequant(__byte_perm(wl[w], 0u, 0x4240), scale_up[0], bias_fold[0]);
        const uint32_t a2 = Dequant(__byte_perm(wl[w], 0u, 0x4341), scale_up[0], bias_fold[0]);
        const uint32_t a1 = Dequant(__byte_perm(wh[w], 0u, 0x4240), scale_up[1], bias_fold[1]);
        const uint32_t a3 = Dequant(__byte_perm(wh[w], 0u, 0x4341), scale_up[1], bias_fold[1]);
        MmaBf16(acc[w & 1][ct], a0, a1, a2, a3, b0[w], b1[w]);
      }
    }
  };

#pragma unroll
  for (int s = 0; s < Depth; ++s) {
    if (s < iters) {
      load(st[s], s);
    }
  }
  for (int base = 0; base < iters; base += Depth) {
#pragma unroll
    for (int s = 0; s < Depth; ++s) {
      const int i = base + s;
      if (i < iters) {
        consume(st[s]);
        if (i + Depth < iters) {
          load(st[s], i + Depth);
        }
      }
    }
  }

  __shared__ float4 reduce[KWarps * ColTiles * 32];
#pragma unroll
  for (int ct = 0; ct < ColTiles; ++ct) {
    reduce[(warp * ColTiles + ct) * 32 + lane] = make_float4(acc[0][ct][0] + acc[1][ct][0], acc[0][ct][1] + acc[1][ct][1],
                                                             acc[0][ct][2] + acc[1][ct][2], acc[0][ct][3] + acc[1][ct][3]);
  }
  __syncthreads();
  if (threadIdx.x < ColTiles * 32) {
    const int ct = threadIdx.x >> 5;
    const int ln = threadIdx.x & 31;
    float4 sum = reduce[ct * 32 + ln];
#pragma unroll
    for (int w = 1; w < KWarps; ++w) {
      const float4 v = reduce[(w * ColTiles + ct) * 32 + ln];
      sum.x += v.x;
      sum.y += v.y;
      sum.z += v.z;
      sum.w += v.w;
    }
    // D[row][col]: c0, c1 = (row g, rows' tokens 2t, 2t + 1), c2, c3 = (row g + 8, tokens 2t, 2t + 1).
    const int col_lo = col0 + ct * kMmaColumns + (ln >> 2);
    const int tok = 2 * (ln & 3);
    if (tok < m) {
      out[static_cast<int64_t>(tok) * n + col_lo] = __float2bfloat16_rn(sum.x * kUnscale);
      out[static_cast<int64_t>(tok) * n + col_lo + 8] = __float2bfloat16_rn(sum.z * kUnscale);
    }
    if (tok + 1 < m) {
      out[static_cast<int64_t>(tok + 1) * n + col_lo] = __float2bfloat16_rn(sum.y * kUnscale);
      out[static_cast<int64_t>(tok + 1) * n + col_lo + 8] = __float2bfloat16_rn(sum.w * kUnscale);
    }
  }
#endif
}

template <int KWarps, int ColTiles, int GroupSize, int Depth, int MinBlocks = 1>
inline void LaunchInt8TcGemv(const __nv_bfloat16* act, const uint8_t* weight, const __nv_bfloat16* scales,
                             __nv_bfloat16* out, int m, int n, int k, cudaStream_t stream) {
  Int8TcGemvKernel<KWarps, ColTiles, GroupSize, Depth, MinBlocks>
      <<<n / (ColTiles * kMmaColumns), KWarps * 32, 0, stream>>>(act, weight, scales, out, m, n, k);
}

struct LaunchConfig {
  int k_warps;
  int depth;
  int blocks_per_sm;  // blocks per SM that the launch bound guarantees
};

// Picks the launch configuration from the grid size (n / 16 blocks). The kernel is DRAM-latency bound at decode batch
// sizes: a second, partly filled wave costs a full block latency, and too few warps per SM leave the memory system
// idle. So the configuration with the fewest blocks per SM that still holds the whole grid in one wave is used (more
// warps and deeper loads per SM the smaller the grid). Grids beyond one wave of the widest configuration take
// whichever of the two narrow configurations needs fewer waves (the 7-block one on a tie). MinBlocks is the
// occupancy that ptxas reaches for each configuration without spills (checked with -Xptxas -v: 112, 72, 66 and 64
// registers).
// H200 (132 SMs), K = 2560 .. 6144, M = 8: N <= 4224 uses 8 warps with depth 4, N <= 6336 8 warps with depth 2,
// N <= 14784 4 warps with 7 blocks and N <= 16896 4 warps with 8 blocks. Within each window the pick is within about
// 5% of the best of the four configurations; the legacy scalar kernel is 1.3 - 2.0x slower.
template <int GroupSize>
inline LaunchConfig LaunchInt8TcGemvAuto(const __nv_bfloat16* act, const uint8_t* weight, const __nv_bfloat16* scales,
                                         __nv_bfloat16* out, int m, int n, int k, int sm_count, cudaStream_t stream) {
  const int64_t blocks = n / kMmaColumns;
  auto waves = [&](int blocks_per_sm) {
    const int64_t slots = static_cast<int64_t>(sm_count) * blocks_per_sm;
    return (blocks + slots - 1) / slots;
  };
  if (blocks <= 2 * sm_count) {
    LaunchInt8TcGemv<8, 1, GroupSize, 4, 2>(act, weight, scales, out, m, n, k, stream);
    return {8, 4, 2};
  }
  if (blocks <= 3 * sm_count) {
    LaunchInt8TcGemv<8, 1, GroupSize, 2, 3>(act, weight, scales, out, m, n, k, stream);
    return {8, 2, 3};
  }
  if (blocks <= 7 * sm_count || waves(7) <= waves(8)) {
    LaunchInt8TcGemv<4, 1, GroupSize, 2, 7>(act, weight, scales, out, m, n, k, stream);
    return {4, 2, 7};
  }
  LaunchInt8TcGemv<4, 1, GroupSize, 2, 8>(act, weight, scales, out, m, n, k, stream);
  return {4, 2, 8};
}

#endif  // defined(__CUDACC__)

}  // namespace dense_tc
}  // namespace onnxruntime::llm::kernels::fpA_intB_gemv
