// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_packed.h"

#include <gsl/util>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <type_traits>

#include "contrib_ops/cuda/moe/qmoe_fused_route.cuh"
#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kMoEPackedMaxExperts = 512;
constexpr int kMoEPackedMaxK = 16;

// Softmax + top-k over 512 experts for arbitrary num_rows (8 warps per CTA, 1 warp per row).
// Uses the exact same warp reduction and tie-breaking key as FusedSoftmaxTopKMapsKernel.
template <typename CudaT>
__global__ __launch_bounds__(256) void MoEPackedSoftmaxTopK512Kernel(
    const CudaT* __restrict__ logits,
    float* __restrict__ topk_scales,
    int* __restrict__ topk_indices,
    int num_rows,
    int k,
    bool normalize_scales) {
  constexpr int num_experts = 512;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * 8 + warp;
  if (row >= num_rows) return;

  const CudaT* row_logits = logits + static_cast<size_t>(row) * num_experts;
  float scores[16];
  uint32_t keys[16];
  float local_max = onnxruntime::cuda::topk::kNegativeInfinity;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const int base = (lane + 32 * w) * 4;
    float v[4];
    if ((reinterpret_cast<uintptr_t>(row_logits + base) & 7u) == 0) {
      FusedRouteLoad4(row_logits + base, v);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[j] = gguf_cuda::CudaTypeToFloat<CudaT>(row_logits[base + j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      scores[w * 4 + j] = v[j] == 0.0f ? 0.0f : v[j];
      keys[w * 4 + j] = FusedRouteKey(v[j], base + j, true);
      local_max = fmaxf(local_max, v[j]);
    }
  }

#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float other = __shfl_xor_sync(0xFFFFFFFFu, local_max, offset);
    local_max = (local_max < other) ? other : local_max;
  }
  const float max_val = local_max;

  uint32_t winners[kMoEPackedMaxK];
#pragma unroll
  for (int r = 0; r < kMoEPackedMaxK; ++r) {
    if (r < k) {
      uint32_t best = keys[0];
#pragma unroll
      for (int j = 1; j < 16; ++j) {
        best = max(best, keys[j]);
      }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      winners[r] = __reduce_max_sync(0xFFFFFFFFu, best);
#else
#pragma unroll
      for (int offset = 16; offset >= 1; offset >>= 1) {
        best = max(best, __shfl_xor_sync(0xFFFFFFFFu, best, offset));
      }
      winners[r] = best;
#endif
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        keys[j] = keys[j] == winners[r] ? 0u : keys[j];
      }
    } else {
      winners[r] = 0u;
    }
  }

  float part[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sum += expf(scores[w * 4 + j] - max_val);
    }
    part[w] = sum;
  }
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      part[w] += __shfl_xor_sync(0xFFFFFFFFu, part[w], offset);
    }
  }
  const float block_sum = ((__shfl_sync(0xFFFFFFFFu, part[0], 0) + __shfl_sync(0xFFFFFFFFu, part[1], 0)) +
                           __shfl_sync(0xFFFFFFFFu, part[2], 0)) +
                          __shfl_sync(0xFFFFFFFFu, part[3], 0);
  const float inv_sum = onnxruntime::cuda::topk::SafeInvSum(block_sum);

  uint32_t selected = 0;
  float prob = 0.0f;
  float scale_sum = 0.0f;
#pragma unroll
  for (int r = 0; r < kMoEPackedMaxK; ++r) {
    if (r < k) {
      const float p = onnxruntime::cuda::topk::SoftmaxScale(FusedRouteScore(winners[r]), max_val, inv_sum);
      scale_sum += p;
      if (lane == r) {
        selected = winners[r];
        prob = p;
      }
    }
  }
  const float denom = onnxruntime::cuda::topk::TopKNormalizeDenom(normalize_scales, scale_sum);
  if (lane < k) {
    const int expert = 0x1FFF - static_cast<int>(selected & 0x1FFFu);
    topk_scales[static_cast<size_t>(row) * k + lane] = normalize_scales ? (prob / denom) : prob;
    topk_indices[static_cast<size_t>(row) * k + lane] = expert;
  }
}

// Generic Softmax + TopK for num_experts != 512 (e.g. unit tests with smaller E).
template <typename CudaT>
__global__ __launch_bounds__(256) void MoEPackedSoftmaxTopKGenericKernel(
    const CudaT* __restrict__ logits,
    float* __restrict__ topk_scales,
    int* __restrict__ topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * 8 + warp;
  if (row >= num_rows) return;

  const CudaT* row_logits = logits + static_cast<size_t>(row) * num_experts;
  float local_max = onnxruntime::cuda::topk::kNegativeInfinity;
  for (int e = lane; e < num_experts; e += 32) {
    const float v = gguf_cuda::CudaTypeToFloat<CudaT>(row_logits[e]);
    local_max = fmaxf(local_max, v);
  }
#pragma unroll
  for (int offset = 16; offset >= 1; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFFu, local_max, offset));
  }
  const float max_val = local_max;

  float local_sum = 0.0f;
  for (int e = lane; e < num_experts; e += 32) {
    const float v = gguf_cuda::CudaTypeToFloat<CudaT>(row_logits[e]);
    local_sum += expf(v - max_val);
  }
#pragma unroll
  for (int offset = 16; offset >= 1; offset >>= 1) {
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, offset);
  }
  const float inv_sum = onnxruntime::cuda::topk::SafeInvSum(local_sum);

  uint32_t winners[kMoEPackedMaxK];
  for (int r = 0; r < k; ++r) {
    uint32_t best = 0u;
    for (int e = lane; e < num_experts; e += 32) {
      const float v = gguf_cuda::CudaTypeToFloat<CudaT>(row_logits[e]);
      uint32_t key = FusedRouteKey(v, e, true);
      for (int prev = 0; prev < r; ++prev) {
        if (key == winners[prev]) key = 0u;
      }
      best = max(best, key);
    }
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
      best = max(best, __shfl_xor_sync(0xFFFFFFFFu, best, offset));
    }
    winners[r] = best;
  }

  float scale_sum = 0.0f;
  uint32_t selected = 0u;
  float prob = 0.0f;
  for (int r = 0; r < k; ++r) {
    const float p = onnxruntime::cuda::topk::SoftmaxScale(FusedRouteScore(winners[r]), max_val, inv_sum);
    scale_sum += p;
    if (lane == r) {
      selected = winners[r];
      prob = p;
    }
  }
  const float denom = onnxruntime::cuda::topk::TopKNormalizeDenom(normalize_scales, scale_sum);
  if (lane < k) {
    const int expert = 0x1FFF - static_cast<int>(selected & 0x1FFFu);
    topk_scales[static_cast<size_t>(row) * k + lane] = normalize_scales ? (prob / denom) : prob;
    topk_indices[static_cast<size_t>(row) * k + lane] = expert;
  }
}

// Parallel single-CTA bucket & tile builder: groups total_routes (route_id = 0..total_routes-1)
// by expert (0..num_experts-1) into tiles of up to kTileRoutes routes sharing the same expert.
template <int kTileRoutes>
__global__ __launch_bounds__(512) void BuildMoEPackedTilesKernel(
    const int* __restrict__ topk_indices,
    int total_routes,
    int num_experts,
    int* __restrict__ sorted_routes,
    MoEPackedTile* __restrict__ tiles,
    int* __restrict__ d_num_tiles) {
  __shared__ int s_counts[kMoEPackedMaxExperts];
  __shared__ int s_cursor[kMoEPackedMaxExperts];
  __shared__ int s_warp_routes[16];
  __shared__ int s_warp_tiles[16];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;

  s_counts[tid] = 0;
  __syncthreads();

  for (int r = tid; r < total_routes; r += 512) {
    const int e = topk_indices[r];
    if (static_cast<unsigned int>(e) < static_cast<unsigned int>(num_experts)) {
      atomicAdd(&s_counts[e], 1);
    }
  }
  __syncthreads();

  const int c = (tid < num_experts) ? s_counts[tid] : 0;
  const int n_tiles = (c + kTileRoutes - 1) / kTileRoutes;

  int r_scan = c;
  int t_scan = n_tiles;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int r_up = __shfl_up_sync(0xFFFFFFFFu, r_scan, offset);
    const int t_up = __shfl_up_sync(0xFFFFFFFFu, t_scan, offset);
    if (lane >= offset) {
      r_scan += r_up;
      t_scan += t_up;
    }
  }
  if (lane == 31) {
    s_warp_routes[warp] = r_scan;
    s_warp_tiles[warp] = t_scan;
  }
  __syncthreads();

  if (warp == 0) {
    int wr = (lane < 16) ? s_warp_routes[lane] : 0;
    int wt = (lane < 16) ? s_warp_tiles[lane] : 0;
#pragma unroll
    for (int offset = 1; offset < 16; offset <<= 1) {
      const int wr_up = __shfl_up_sync(0xFFFFFFFFu, wr, offset);
      const int wt_up = __shfl_up_sync(0xFFFFFFFFu, wt, offset);
      if (lane >= offset) {
        wr += wr_up;
        wt += wt_up;
      }
    }
    if (lane < 16) {
      s_warp_routes[lane] = wr;
      s_warp_tiles[lane] = wt;
    }
    if (lane == 15) {
      *d_num_tiles = wt;
    }
  }
  __syncthreads();

  const int warp_r_base = (warp > 0) ? s_warp_routes[warp - 1] : 0;
  const int warp_t_base = (warp > 0) ? s_warp_tiles[warp - 1] : 0;
  const int exc_r = warp_r_base + r_scan - c;
  const int exc_t = warp_t_base + t_scan - n_tiles;
  if (tid < num_experts) {
    s_cursor[tid] = exc_r;
  }
  __syncthreads();

  for (int r = tid; r < total_routes; r += 512) {
    const int e = topk_indices[r];
    if (static_cast<unsigned int>(e) < static_cast<unsigned int>(num_experts)) {
      const int pos = atomicAdd(&s_cursor[e], 1);
      sorted_routes[pos] = r;
    }
  }

  if (tid < num_experts && n_tiles > 0) {
    for (int t = 0; t < n_tiles; ++t) {
      MoEPackedTile tile{};
      tile.expert = tid;
      tile.start_route = exc_r + t * kTileRoutes;
      const int rem = c - t * kTileRoutes;
      tile.count = rem < kTileRoutes ? rem : kTileRoutes;
      tile.pad = 0;
      tiles[exc_t + t] = tile;
    }
  }
}

// Decode Warp-K-Split Tensor-Core BF16 FC1 + SwiGLU kernel (inspired by bolt_ops.cu MmaGemvKernel):
// KW=4 warps (128 threads) per CTA split K across the 4 warps (only 20 iterations/warp for K=2560,
// zero __syncthreads() in the K-loop) while computing MT*8 = 16 inter_size channels (32 gate+up rows).
template <int KW, int MT, int kGateUpType>
__global__ __launch_bounds__(KW * 32) void MoEPackedFc1DecodeWarpKSplitKernel(
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ gate_weights,
    const uint8_t* __restrict__ up_weights,
    const int* __restrict__ sorted_routes,
    const MoEPackedTile* __restrict__ tiles,
    const int* __restrict__ d_num_tiles,
    MoEPackedParams p,
    __nv_bfloat16* __restrict__ fc1_out) {
  const int tile_idx = static_cast<int>(blockIdx.y);
  if (tile_idx >= __ldg(d_num_tiles)) return;

  constexpr int kThreads = KW * 32;
  using namespace onnxruntime::contrib::gguf_quant;
  constexpr int kLutU64Count = (kGateUpType == kGgmlTypeIQ2_S)
                                   ? 1024
                                   : ((kGateUpType == kGgmlTypeIQ2_XS)
                                          ? 512
                                          : ((kGateUpType == kGgmlTypeIQ2_XXS)
                                                 ? 256
                                                 : ((kGateUpType == kGgmlTypeIQ3_XXS) ? 128 : 1)));

  __shared__ uint64_t s_lut[kLutU64Count];
  __shared__ float s_red[KW][MT * 16][8];
  __shared__ int s_routes[8];
  __shared__ int s_token_rows[8];

  const MoEPackedTile tile = tiles[tile_idx];
  const int count = tile.count;
  const int expert = tile.expert;

  const int tid = static_cast<int>(threadIdx.x);
  if (tid < 8) {
    const int r = (tid < count) ? __ldg(&sorted_routes[tile.start_route + tid]) : 0;
    s_routes[tid] = r;
    s_token_rows[tid] = r / p.k;
  }
  if constexpr (kGateUpType == kGgmlTypeIQ2_XXS) {
    for (int i = tid; i < 256; i += kThreads) s_lut[i] = __ldg(iq2xxs_grid + i);
  } else if constexpr (kGateUpType == kGgmlTypeIQ2_XS) {
    for (int i = tid; i < 512; i += kThreads) s_lut[i] = __ldg(iq2xs_grid + i);
  } else if constexpr (kGateUpType == kGgmlTypeIQ2_S) {
    for (int i = tid; i < 1024; i += kThreads) s_lut[i] = __ldg(iq2s_grid + i);
  } else if constexpr (kGateUpType == kGgmlTypeIQ3_XXS) {
    uint32_t* s_lut32 = reinterpret_cast<uint32_t*>(s_lut);
    for (int i = tid; i < 256; i += kThreads) s_lut32[i] = __ldg(iq3xxs_grid + i);
  }
  __syncthreads();

  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3
  const int cta_ch_base = static_cast<int>(blockIdx.x) * (MT * 8);

  const uint8_t* gate_ptrs[MT];
  const uint8_t* up_ptrs[MT];
  bool valid_rows[MT];
  float acc[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    const int j_out = cta_ch_base + m * 8 + g;
    valid_rows[m] = (j_out < p.inter_size);
    const int64_t exp_row_idx =
        static_cast<int64_t>(expert) * p.inter_size + (valid_rows[m] ? j_out : 0);
    gate_ptrs[m] = gate_weights + exp_row_idx * p.gate_row_bytes;
    up_ptrs[m] = up_weights + exp_row_idx * p.up_row_bytes;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[m][i] = 0.0f;
    }
  }

  const bool x_valid = (g < count);
  const __nv_bfloat16* x_row =
      input + static_cast<int64_t>(x_valid ? s_token_rows[g] : 0) * p.hidden_size;

  const int num_sb32 = p.hidden_size >> 5;
  for (int sb32_idx = warp; sb32_idx < num_sb32; sb32_idx += KW) {
    const uint4 b_vec = x_valid
                            ? gguf_cuda::LoadU4Dev(x_row + sb32_idx * 32 + t * 8)
                            : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (valid_rows[m]) {
        float g_vals[8];
        float u_vals[8];
        gguf_cuda::DecodeGgufSubBlock8SmemLutTpl<kGateUpType>(
            gate_ptrs[m], p.gate_type, sb32_idx, t, s_lut, g_vals);
        gguf_cuda::DecodeGgufSubBlock8SmemLutTpl<kGateUpType>(
            up_ptrs[m], p.up_type, sb32_idx, t, s_lut, u_vals);
        const uint4 e_vec = gguf_cuda::PackFloat8ToBf16x2(g_vals);
        const uint4 o_vec = gguf_cuda::PackFloat8ToBf16x2(u_vals);
        gguf_cuda::MmaBf16(acc[m], e_vec.x, o_vec.x, e_vec.y, o_vec.y, b_vec.x, b_vec.y);
        gguf_cuda::MmaBf16(acc[m], e_vec.z, o_vec.z, e_vec.w, o_vec.w, b_vec.z, b_vec.w);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MT; ++m) {
    s_red[warp][m * 16 + g][2 * t + 0] = acc[m][0];
    s_red[warp][m * 16 + g][2 * t + 1] = acc[m][1];
    s_red[warp][m * 16 + g + 8][2 * t + 0] = acc[m][2];
    s_red[warp][m * 16 + g + 8][2 * t + 1] = acc[m][3];
  }
  __syncthreads();

  for (int idx = tid; idx < MT * 8 * 8; idx += kThreads) {
    const int col = idx / (MT * 8);
    const int ch = idx - col * (MT * 8);
    const int j_out = cta_ch_base + ch;
    if (col < count && j_out < p.inter_size) {
      const int m = ch >> 3;
      const int g_ch = ch & 7;
      float g_raw = 0.0f;
      float u_raw = 0.0f;
#pragma unroll
      for (int w = 0; w < KW; ++w) {
        g_raw += s_red[w][m * 16 + g_ch][col];
        u_raw += s_red[w][m * 16 + g_ch + 8][col];
      }
      const float g_val = __bfloat162float(__float2bfloat16_rn(g_raw));
      const float u_val = __bfloat162float(__float2bfloat16_rn(u_raw));
      const float swiglu = u_val * (g_val / (1.0f + expf(-g_val)));
      fc1_out[static_cast<int64_t>(s_routes[col]) * p.inter_size + j_out] =
          __float2bfloat16_rn(swiglu);
    }
  }
}

// Tensor-Core BF16 FC1 + SwiGLU kernel (mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32).
// Each warp (32 threads) computes 8 inter_size channels (8 gate rows + 8 up rows = 16 weight rows)
// across up to kMaxSubTiles * 8 routes (8 routes in decode, 32 routes in prefill).
// 8 warps per CTA = 64 inter_size channels per CTA.
template <int kMaxSubTiles, int kGateUpType>
__global__ __launch_bounds__((kMaxSubTiles == 1) ? 128 : 256) void MoEPackedFc1SwiGLUBf16TcKernel(
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ gate_weights,
    const uint8_t* __restrict__ up_weights,
    const int* __restrict__ sorted_routes,
    const MoEPackedTile* __restrict__ tiles,
    const int* __restrict__ d_num_tiles,
    MoEPackedParams p,
    __nv_bfloat16* __restrict__ fc1_out) {
  const int tile_idx = static_cast<int>(blockIdx.y);
  if (tile_idx >= __ldg(d_num_tiles)) return;

  constexpr int kWarps = (kMaxSubTiles == 1) ? 4 : 8;
  constexpr int kThreads = kWarps * 32;
  constexpr int kChunkK = 512;
  constexpr int kChunkU4 = kChunkK / 8;         // 64
  constexpr int kRowStrideU4 = kChunkU4 + 4;    // 68 (bank-conflict-free for LDS.128)
  constexpr int kMaxRoutes = kMaxSubTiles * 8;  // 8 or 32
  using namespace onnxruntime::contrib::gguf_quant;
  constexpr int kLutU64Count = (kGateUpType == kGgmlTypeIQ2_S)
                                   ? 1024
                                   : ((kGateUpType == kGgmlTypeIQ2_XS)
                                          ? 512
                                          : ((kGateUpType == kGgmlTypeIQ2_XXS)
                                                 ? 256
                                                 : ((kGateUpType == kGgmlTypeIQ3_XXS) ? 128 : 1)));

  __shared__ uint4 s_x_u4[kMaxRoutes * kRowStrideU4];
  __shared__ uint64_t s_lut[kLutU64Count];
  __shared__ int s_routes[kMaxRoutes];
  __shared__ int s_token_rows[kMaxRoutes];

  const MoEPackedTile tile = tiles[tile_idx];
  const int count = tile.count;
  const int expert = tile.expert;
  const int num_subtiles = (count + 7) >> 3;
  const int padded_routes = num_subtiles * 8;

  const int tid = static_cast<int>(threadIdx.x);
  if (tid < kMaxRoutes) {
    const int r = (tid < count) ? __ldg(&sorted_routes[tile.start_route + tid]) : 0;
    s_routes[tid] = r;
    s_token_rows[tid] = r / p.k;
  }
  if constexpr (kGateUpType == kGgmlTypeIQ2_XXS) {
    for (int i = tid; i < 256; i += kThreads) {
      s_lut[i] = __ldg(iq2xxs_grid + i);
    }
  } else if constexpr (kGateUpType == kGgmlTypeIQ2_XS) {
    for (int i = tid; i < 512; i += kThreads) {
      s_lut[i] = __ldg(iq2xs_grid + i);
    }
  } else if constexpr (kGateUpType == kGgmlTypeIQ2_S) {
    for (int i = tid; i < 1024; i += kThreads) {
      s_lut[i] = __ldg(iq2s_grid + i);
    }
  } else if constexpr (kGateUpType == kGgmlTypeIQ3_XXS) {
    uint32_t* s_lut32 = reinterpret_cast<uint32_t*>(s_lut);
    for (int i = tid; i < 256; i += kThreads) {
      s_lut32[i] = __ldg(iq3xxs_grid + i);
    }
  }
  __syncthreads();

  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3
  const int j_out = (static_cast<int>(blockIdx.x) * kWarps + warp) * 8 + g;
  const bool valid_row = (j_out < p.inter_size);

  const int64_t exp_row_idx = static_cast<int64_t>(expert) * p.inter_size + (valid_row ? j_out : 0);
  const uint8_t* gate_row_ptr = gate_weights + exp_row_idx * p.gate_row_bytes;
  const uint8_t* up_row_ptr = up_weights + exp_row_idx * p.up_row_bytes;

  float acc[kMaxSubTiles][4];
#pragma unroll
  for (int st = 0; st < kMaxSubTiles; ++st) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[st][i] = 0.0f;
    }
  }

  for (int k_base = 0; k_base < p.hidden_size; k_base += kChunkK) {
    const int rem_k = p.hidden_size - k_base;
    const int actual_chunk_k = rem_k < kChunkK ? rem_k : kChunkK;
    const int actual_chunk_u4 = actual_chunk_k >> 3;
    const int total_u4 = padded_routes * actual_chunk_u4;

    if (actual_chunk_u4 == 64) {
      for (int idx = tid; idx < total_u4; idx += kThreads) {
        const int m = idx >> 6;
        const int u4_col = idx & 63;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (m < count) {
          const int token_row = s_token_rows[m];
          const __nv_bfloat16* src_ptr =
              input + static_cast<int64_t>(token_row) * p.hidden_size + k_base + (u4_col << 3);
          val = gguf_cuda::LoadU4Dev(src_ptr);
        }
        s_x_u4[m * kRowStrideU4 + u4_col] = val;
      }
    } else {
      for (int idx = tid; idx < total_u4; idx += kThreads) {
        const int m = idx / actual_chunk_u4;
        const int u4_col = idx - m * actual_chunk_u4;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (m < count) {
          const int token_row = s_token_rows[m];
          const __nv_bfloat16* src_ptr =
              input + static_cast<int64_t>(token_row) * p.hidden_size + k_base + (u4_col << 3);
          val = gguf_cuda::LoadU4Dev(src_ptr);
        }
        s_x_u4[m * kRowStrideU4 + u4_col] = val;
      }
    }
    __syncthreads();

    if (valid_row) {
      const int sb32_base = k_base >> 5;
      const int num_sb32 = actual_chunk_k >> 5;
      for (int sb_local = 0; sb_local < num_sb32; ++sb_local) {
        const int sb32_idx = sb32_base + sb_local;
        float g_vals[8];
        float u_vals[8];
        gguf_cuda::DecodeGgufSubBlock8SmemLutTpl<kGateUpType>(gate_row_ptr, p.gate_type, sb32_idx, t, s_lut, g_vals);
        gguf_cuda::DecodeGgufSubBlock8SmemLutTpl<kGateUpType>(up_row_ptr, p.up_type, sb32_idx, t, s_lut, u_vals);
        const uint4 e_vec = gguf_cuda::PackFloat8ToBf16x2(g_vals);
        const uint4 o_vec = gguf_cuda::PackFloat8ToBf16x2(u_vals);
#pragma unroll
        for (int st = 0; st < kMaxSubTiles; ++st) {
          if (st < num_subtiles) {
            const uint4 b_vec = s_x_u4[(st * 8 + g) * kRowStrideU4 + sb_local * 4 + t];
            gguf_cuda::MmaBf16(acc[st], e_vec.x, o_vec.x, e_vec.y, o_vec.y, b_vec.x, b_vec.y);
            gguf_cuda::MmaBf16(acc[st], e_vec.z, o_vec.z, e_vec.w, o_vec.w, b_vec.z, b_vec.w);
          }
        }
      }
    }
    if (k_base + kChunkK < p.hidden_size) {
      __syncthreads();
    }
  }

  if (valid_row) {
#pragma unroll
    for (int st = 0; st < kMaxSubTiles; ++st) {
      if (st < num_subtiles) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = st * 8 + 2 * t + h;
          if (m < count) {
            const float g_raw = (h == 0) ? acc[st][0] : acc[st][1];
            const float u_raw = (h == 0) ? acc[st][2] : acc[st][3];
            const float g_val = __bfloat162float(__float2bfloat16_rn(g_raw));
            const float u_val = __bfloat162float(__float2bfloat16_rn(u_raw));
            const float swiglu = u_val * (g_val / (1.0f + expf(-g_val)));
            const int route = s_routes[m];
            fc1_out[static_cast<int64_t>(route) * p.inter_size + j_out] = __float2bfloat16_rn(swiglu);
          }
        }
      }
    }
  }
}

// Tensor-Core BF16 FC2 kernel (mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32).
// Each warp (32 threads) computes 16 hidden_size channels across up to kMaxSubTiles * 8 routes.
// 8 warps per CTA = 128 hidden_size channels per CTA.
template <int kMaxSubTiles, int kDownType>
__global__ __launch_bounds__(256) void MoEPackedFc2Bf16TcKernel(
    const __nv_bfloat16* __restrict__ fc1_out,
    const uint8_t* __restrict__ down_weights,
    const int* __restrict__ sorted_routes,
    const MoEPackedTile* __restrict__ tiles,
    const int* __restrict__ d_num_tiles,
    MoEPackedParams p,
    __nv_bfloat16* __restrict__ fc2_out) {
  const int tile_idx = static_cast<int>(blockIdx.y);
  if (tile_idx >= __ldg(d_num_tiles)) return;

  constexpr int kChunkK = 640;
  constexpr int kChunkU4 = kChunkK / 8;         // 80
  constexpr int kRowStrideU4 = kChunkU4 + 4;    // 84 (bank-conflict-free for LDS.128)
  constexpr int kMaxRoutes = kMaxSubTiles * 8;  // 8 or 32

  __shared__ uint4 s_act_u4[kMaxRoutes * kRowStrideU4];
  __shared__ int s_routes[kMaxRoutes];

  const MoEPackedTile tile = tiles[tile_idx];
  const int count = tile.count;
  const int expert = tile.expert;
  const int num_subtiles = (count + 7) >> 3;
  const int padded_routes = num_subtiles * 8;

  const int tid = static_cast<int>(threadIdx.x);
  if (tid < kMaxRoutes) {
    s_routes[tid] = (tid < count) ? __ldg(&sorted_routes[tile.start_route + tid]) : 0;
  }
  __syncthreads();

  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3
  const int n_out_base = (static_cast<int>(blockIdx.x) * 8 + warp) * 16;
  const int r_lo = n_out_base + 2 * g;
  const int r_hi = r_lo + 1;
  const bool valid_lo = (r_lo < p.hidden_size);
  const bool valid_hi = (r_hi < p.hidden_size);

  const int64_t exp_base = static_cast<int64_t>(expert) * p.hidden_size;
  const uint8_t* down_row_lo = down_weights + (exp_base + (valid_lo ? r_lo : 0)) * p.down_row_bytes;
  const uint8_t* down_row_hi = down_weights + (exp_base + (valid_hi ? r_hi : 0)) * p.down_row_bytes;

  float acc[kMaxSubTiles][4];
#pragma unroll
  for (int st = 0; st < kMaxSubTiles; ++st) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[st][i] = 0.0f;
    }
  }

  for (int k_base = 0; k_base < p.inter_size; k_base += kChunkK) {
    const int rem_k = p.inter_size - k_base;
    const int actual_chunk_k = rem_k < kChunkK ? rem_k : kChunkK;
    const int actual_chunk_u4 = actual_chunk_k >> 3;
    const int total_u4 = padded_routes * actual_chunk_u4;

    for (int idx = tid; idx < total_u4; idx += 256) {
      const int m = idx / actual_chunk_u4;
      const int u4_col = idx - m * actual_chunk_u4;
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (m < count) {
        const int route = s_routes[m];
        const __nv_bfloat16* src_ptr =
            fc1_out + static_cast<int64_t>(route) * p.inter_size + k_base + (u4_col << 3);
        val = __ldg(reinterpret_cast<const uint4*>(src_ptr));
      }
      s_act_u4[m * kRowStrideU4 + u4_col] = val;
    }
    __syncthreads();

    if (valid_lo) {
      const int sb32_base = k_base >> 5;
      const int num_sb32 = actual_chunk_k >> 5;
      for (int sb_local = 0; sb_local < num_sb32; ++sb_local) {
        const int sb32_idx = sb32_base + sb_local;
        float d_lo[8];
        float d_hi[8];
        gguf_cuda::DecodeGgufSubBlock8Tpl<kDownType>(down_row_lo, p.down_type, sb32_idx, t, d_lo);
        if (valid_hi) {
          gguf_cuda::DecodeGgufSubBlock8Tpl<kDownType>(down_row_hi, p.down_type, sb32_idx, t, d_hi);
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) d_hi[j] = 0.0f;
        }
        const uint4 e_vec = gguf_cuda::PackFloat8ToBf16x2(d_lo);
        const uint4 o_vec = gguf_cuda::PackFloat8ToBf16x2(d_hi);
#pragma unroll
        for (int st = 0; st < kMaxSubTiles; ++st) {
          if (st < num_subtiles) {
            const uint4 b_vec = s_act_u4[(st * 8 + g) * kRowStrideU4 + sb_local * 4 + t];
            gguf_cuda::MmaBf16(acc[st], e_vec.x, o_vec.x, e_vec.y, o_vec.y, b_vec.x, b_vec.y);
            gguf_cuda::MmaBf16(acc[st], e_vec.z, o_vec.z, e_vec.w, o_vec.w, b_vec.z, b_vec.w);
          }
        }
      }
    }
    if (k_base + kChunkK < p.inter_size) {
      __syncthreads();
    }
  }

  if (valid_lo) {
#pragma unroll
    for (int st = 0; st < kMaxSubTiles; ++st) {
      if (st < num_subtiles) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = st * 8 + 2 * t + h;
          if (m < count) {
            const int route = s_routes[m];
            const float v_lo = (h == 0) ? acc[st][0] : acc[st][1];
            const float v_hi = (h == 0) ? acc[st][2] : acc[st][3];
            __nv_bfloat16* dst = fc2_out + static_cast<int64_t>(route) * p.hidden_size + r_lo;
            if (valid_hi && ((p.hidden_size & 1) == 0)) {
              *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v_lo, v_hi);
            } else {
              dst[0] = __float2bfloat16_rn(v_lo);
              if (valid_hi) dst[1] = __float2bfloat16_rn(v_hi);
            }
          }
        }
      }
    }
  }
}

// SIMT fallback for FP16 and BF16 builds without Tensor Core bodies.
template <typename CudaT, bool kUseShared>
__global__ __launch_bounds__(256) void MoEPackedFc1SwiGLUSimtKernel(
    const CudaT* __restrict__ input,
    const uint8_t* __restrict__ gate_weights,
    const uint8_t* __restrict__ up_weights,
    const int* __restrict__ sorted_routes,
    const MoEPackedTile* __restrict__ tiles,
    const int* __restrict__ d_num_tiles,
    MoEPackedParams p,
    CudaT* __restrict__ fc1_out) {
  const int tile_idx = static_cast<int>(blockIdx.y);
  if (tile_idx >= *d_num_tiles) return;

  extern __shared__ char smem_raw[];
  CudaT* s_x = reinterpret_cast<CudaT*>(smem_raw);

  const MoEPackedTile tile = tiles[tile_idx];
  const int count = tile.count < 8 ? tile.count : 8;
  const int expert = tile.expert;

  if constexpr (kUseShared) {
    const int total_elems = count * p.hidden_size;
    for (int idx = static_cast<int>(threadIdx.x); idx < total_elems; idx += 256) {
      const int m = idx / p.hidden_size;
      const int col = idx - m * p.hidden_size;
      const int token_row = sorted_routes[tile.start_route + m] / p.k;
      s_x[idx] = input[static_cast<int64_t>(token_row) * p.hidden_size + col];
    }
    __syncthreads();
  }

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int j_out = static_cast<int>(blockIdx.x) * 8 + warp;
  if (j_out >= p.inter_size) return;

  const int64_t exp_row_idx = static_cast<int64_t>(expert) * p.inter_size + j_out;
  const uint8_t* gate_row_ptr = gate_weights + exp_row_idx * p.gate_row_bytes;
  const uint8_t* up_row_ptr = up_weights + exp_row_idx * p.up_row_bytes;

  float gate_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float up_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  const int total_halves = p.hidden_size / 16;
  for (int h_idx = lane; h_idx < total_halves; h_idx += 32) {
    const int sb32_idx = h_idx >> 1;
    const int half16 = h_idx & 1;
    float g_vals[16];
    float u_vals[16];
    gguf_cuda::DecodeGgufSubBlock32(gate_row_ptr, p.gate_type, sb32_idx, half16, g_vals);
    gguf_cuda::DecodeGgufSubBlock32(up_row_ptr, p.up_type, sb32_idx, half16, u_vals);
    if constexpr (std::is_same_v<CudaT, __nv_bfloat16>) {
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        g_vals[c] = __bfloat162float(__float2bfloat16_rn(g_vals[c]));
        u_vals[c] = __bfloat162float(__float2bfloat16_rn(u_vals[c]));
      }
    }

    const int k_offset = h_idx * 16;
    for (int m = 0; m < count; ++m) {
      const CudaT* xm;
      if constexpr (kUseShared) {
        xm = s_x + m * p.hidden_size + k_offset;
      } else {
        const int token_row = sorted_routes[tile.start_route + m] / p.k;
        xm = input + static_cast<int64_t>(token_row) * p.hidden_size + k_offset;
      }
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        const float xv = gguf_cuda::CudaTypeToFloat<CudaT>(xm[c]);
        gate_acc[m] += g_vals[c] * xv;
        up_acc[m] += u_vals[c] * xv;
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      gate_acc[m] += __shfl_down_sync(0xFFFFFFFFu, gate_acc[m], offset);
      up_acc[m] += __shfl_down_sync(0xFFFFFFFFu, up_acc[m], offset);
    }
  }

  if (lane == 0) {
    for (int m = 0; m < count; ++m) {
      const float g = gguf_cuda::CudaTypeToFloat<CudaT>(gguf_cuda::FloatToCudaType<CudaT>(gate_acc[m]));
      const float u = gguf_cuda::CudaTypeToFloat<CudaT>(gguf_cuda::FloatToCudaType<CudaT>(up_acc[m]));
      const float swiglu = u * (g / (1.0f + expf(-g)));
      fc1_out[static_cast<int64_t>(sorted_routes[tile.start_route + m]) * p.inter_size + j_out] =
          gguf_cuda::FloatToCudaType<CudaT>(swiglu);
    }
  }
}

// SIMT fallback for FP16 and BF16 builds without Tensor Core bodies.
template <typename CudaT, bool kUseShared>
__global__ __launch_bounds__(256) void MoEPackedFc2SimtKernel(
    const CudaT* __restrict__ fc1_out,
    const uint8_t* __restrict__ down_weights,
    const int* __restrict__ sorted_routes,
    const MoEPackedTile* __restrict__ tiles,
    const int* __restrict__ d_num_tiles,
    MoEPackedParams p,
    CudaT* __restrict__ fc2_out) {
  const int tile_idx = static_cast<int>(blockIdx.y);
  if (tile_idx >= *d_num_tiles) return;

  extern __shared__ char smem_raw[];
  CudaT* s_act = reinterpret_cast<CudaT*>(smem_raw);

  const MoEPackedTile tile = tiles[tile_idx];
  const int count = tile.count < 8 ? tile.count : 8;
  const int expert = tile.expert;

  if constexpr (kUseShared) {
    const int total_elems = count * p.inter_size;
    for (int idx = static_cast<int>(threadIdx.x); idx < total_elems; idx += 256) {
      const int m = idx / p.inter_size;
      const int col = idx - m * p.inter_size;
      s_act[idx] = fc1_out[static_cast<int64_t>(sorted_routes[tile.start_route + m]) * p.inter_size + col];
    }
    __syncthreads();
  }

  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int n_out = static_cast<int>(blockIdx.x) * 8 + warp;
  if (n_out >= p.hidden_size) return;

  const int64_t exp_row_idx = static_cast<int64_t>(expert) * p.hidden_size + n_out;
  const uint8_t* down_row_ptr = down_weights + exp_row_idx * p.down_row_bytes;

  float down_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  const int total_halves = p.inter_size / 16;
  for (int h_idx = lane; h_idx < total_halves; h_idx += 32) {
    const int sb32_idx = h_idx >> 1;
    const int half16 = h_idx & 1;
    float d_vals[16];
    gguf_cuda::DecodeGgufSubBlock32(down_row_ptr, p.down_type, sb32_idx, half16, d_vals);
    if constexpr (std::is_same_v<CudaT, __nv_bfloat16>) {
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        d_vals[c] = __bfloat162float(__float2bfloat16_rn(d_vals[c]));
      }
    }

    const int k_offset = h_idx * 16;
    for (int m = 0; m < count; ++m) {
      const CudaT* am;
      if constexpr (kUseShared) {
        am = s_act + m * p.inter_size + k_offset;
      } else {
        am = fc1_out + static_cast<int64_t>(sorted_routes[tile.start_route + m]) * p.inter_size + k_offset;
      }
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        down_acc[m] += d_vals[c] * gguf_cuda::CudaTypeToFloat<CudaT>(am[c]);
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      down_acc[m] += __shfl_down_sync(0xFFFFFFFFu, down_acc[m], offset);
    }
  }

  if (lane == 0) {
    for (int m = 0; m < count; ++m) {
      fc2_out[static_cast<int64_t>(sorted_routes[tile.start_route + m]) * p.hidden_size + n_out] =
          gguf_cuda::FloatToCudaType<CudaT>(down_acc[m]);
    }
  }
}

// Vectorized 128-bit Finalize: accumulates the k routed expert outputs per token in rank order 0..k-1.
template <typename CudaT>
__global__ __launch_bounds__(256) void MoEPackedFinalizeVec8Kernel(
    const CudaT* __restrict__ fc2_out,
    const float* __restrict__ topk_scales,
    int num_rows,
    int hidden_size,
    int k,
    CudaT* __restrict__ output) {
  const int row = static_cast<int>(blockIdx.y);
  const int col8 = (static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x)) * 8;
  if (row >= num_rows || col8 >= hidden_size) return;

  const float* scales = topk_scales + static_cast<int64_t>(row) * k;
  const CudaT* row_fc2 = fc2_out + static_cast<int64_t>(row) * k * hidden_size + col8;
  float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int r = 0; r < k; ++r) {
    const float sc = __ldg(scales + r);
    const uint4 v_u4 = __ldg(reinterpret_cast<const uint4*>(row_fc2 + static_cast<int64_t>(r) * hidden_size));
    const CudaT* vals = reinterpret_cast<const CudaT*>(&v_u4);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sum[j] += sc * gguf_cuda::CudaTypeToFloat<CudaT>(vals[j]);
    }
  }
  uint4 out_u4;
  CudaT* out_vals = reinterpret_cast<CudaT*>(&out_u4);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out_vals[j] = gguf_cuda::FloatToCudaType<CudaT>(sum[j]);
  }
  *reinterpret_cast<uint4*>(output + static_cast<int64_t>(row) * hidden_size + col8) = out_u4;
}

template <typename CudaT>
__global__ __launch_bounds__(256) void MoEPackedFinalizeKernel(
    const CudaT* __restrict__ fc2_out,
    const float* __restrict__ topk_scales,
    int num_rows,
    int hidden_size,
    int k,
    CudaT* __restrict__ output) {
  const int row = static_cast<int>(blockIdx.y);
  const int col = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
  if (row >= num_rows || col >= hidden_size) return;

  const float* scales = topk_scales + static_cast<int64_t>(row) * k;
  const CudaT* row_fc2 = fc2_out + static_cast<int64_t>(row) * k * hidden_size + col;
  float sum = 0.0f;
  for (int r = 0; r < k; ++r) {
    sum += scales[r] * gguf_cuda::CudaTypeToFloat<CudaT>(row_fc2[static_cast<int64_t>(r) * hidden_size]);
  }
  output[static_cast<int64_t>(row) * hidden_size + col] = gguf_cuda::FloatToCudaType<CudaT>(sum);
}

size_t AlignUpWorkspace(size_t count, size_t element_bytes = 1) {
  size_t bytes = 0;
  ORT_ENFORCE(IAllocator::CalcMemSizeForArrayWithAlignment(count, element_bytes, 256, &bytes),
              "MoEPacked workspace byte size overflow");
  return bytes;
}

void LaunchMoEPackedFc1DecodeBf16TcDispatch(
    cudaStream_t stream,
    dim3 grid,
    const __nv_bfloat16* input,
    const uint8_t* gate_weights,
    const uint8_t* up_weights,
    const int* sorted_routes,
    const MoEPackedTile* tiles,
    const int* d_num_tiles,
    const MoEPackedParams& params,
    __nv_bfloat16* fc1_out) {
  using namespace onnxruntime::contrib::gguf_quant;
  constexpr int KW = 4;
  constexpr int MT = 2;
  constexpr unsigned int kBlock = KW * 32;
  if (params.gate_type == params.up_type) {
    switch (params.gate_type) {
      case kGgmlTypeIQ2_XXS:
        MoEPackedFc1DecodeWarpKSplitKernel<KW, MT, kGgmlTypeIQ2_XXS><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeIQ2_XS:
        MoEPackedFc1DecodeWarpKSplitKernel<KW, MT, kGgmlTypeIQ2_XS><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeIQ2_S:
        MoEPackedFc1DecodeWarpKSplitKernel<KW, MT, kGgmlTypeIQ2_S><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeIQ3_XXS:
        MoEPackedFc1DecodeWarpKSplitKernel<KW, MT, kGgmlTypeIQ3_XXS><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeIQ4_XS:
        MoEPackedFc1DecodeWarpKSplitKernel<KW, MT, kGgmlTypeIQ4_XS><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeIQ4_NL:
        MoEPackedFc1DecodeWarpKSplitKernel<KW, MT, kGgmlTypeIQ4_NL><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeQ2_0:
        MoEPackedFc1DecodeWarpKSplitKernel<KW, MT, kGgmlTypeQ2_0><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeQ5_0:
        MoEPackedFc1DecodeWarpKSplitKernel<KW, MT, kGgmlTypeQ5_0><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeQ8_0:
        MoEPackedFc1DecodeWarpKSplitKernel<KW, MT, kGgmlTypeQ8_0><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      default:
        break;
    }
  }
  MoEPackedFc1DecodeWarpKSplitKernel<KW, MT, -1><<<grid, kBlock, 0, stream>>>(
      input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
}

template <int kMaxSubTiles>
void LaunchMoEPackedFc1Bf16TcDispatch(
    cudaStream_t stream,
    dim3 grid,
    const __nv_bfloat16* input,
    const uint8_t* gate_weights,
    const uint8_t* up_weights,
    const int* sorted_routes,
    const MoEPackedTile* tiles,
    const int* d_num_tiles,
    const MoEPackedParams& params,
    __nv_bfloat16* fc1_out) {
  using namespace onnxruntime::contrib::gguf_quant;
  constexpr unsigned int kBlock = (kMaxSubTiles == 1) ? 128u : 256u;
  if (params.gate_type == params.up_type) {
    switch (params.gate_type) {
      case kGgmlTypeIQ2_XXS:
        MoEPackedFc1SwiGLUBf16TcKernel<kMaxSubTiles, kGgmlTypeIQ2_XXS><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeIQ2_XS:
        MoEPackedFc1SwiGLUBf16TcKernel<kMaxSubTiles, kGgmlTypeIQ2_XS><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeIQ2_S:
        MoEPackedFc1SwiGLUBf16TcKernel<kMaxSubTiles, kGgmlTypeIQ2_S><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeIQ3_XXS:
        MoEPackedFc1SwiGLUBf16TcKernel<kMaxSubTiles, kGgmlTypeIQ3_XXS><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeIQ4_XS:
        MoEPackedFc1SwiGLUBf16TcKernel<kMaxSubTiles, kGgmlTypeIQ4_XS><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeIQ4_NL:
        MoEPackedFc1SwiGLUBf16TcKernel<kMaxSubTiles, kGgmlTypeIQ4_NL><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeQ2_0:
        MoEPackedFc1SwiGLUBf16TcKernel<kMaxSubTiles, kGgmlTypeQ2_0><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeQ5_0:
        MoEPackedFc1SwiGLUBf16TcKernel<kMaxSubTiles, kGgmlTypeQ5_0><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      case kGgmlTypeQ8_0:
        MoEPackedFc1SwiGLUBf16TcKernel<kMaxSubTiles, kGgmlTypeQ8_0><<<grid, kBlock, 0, stream>>>(
            input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
        return;
      default:
        break;
    }
  }
  MoEPackedFc1SwiGLUBf16TcKernel<kMaxSubTiles, -1><<<grid, kBlock, 0, stream>>>(
      input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
}

template <int kMaxSubTiles>
void LaunchMoEPackedFc2Bf16TcDispatch(
    cudaStream_t stream,
    dim3 grid,
    const __nv_bfloat16* fc1_out,
    const uint8_t* down_weights,
    const int* sorted_routes,
    const MoEPackedTile* tiles,
    const int* d_num_tiles,
    const MoEPackedParams& params,
    __nv_bfloat16* fc2_out) {
  using namespace onnxruntime::contrib::gguf_quant;
  switch (params.down_type) {
    case kGgmlTypeQ2_0:
      MoEPackedFc2Bf16TcKernel<kMaxSubTiles, kGgmlTypeQ2_0><<<grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      return;
    case kGgmlTypeIQ4_NL:
      MoEPackedFc2Bf16TcKernel<kMaxSubTiles, kGgmlTypeIQ4_NL><<<grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      return;
    case kGgmlTypeQ5_0:
      MoEPackedFc2Bf16TcKernel<kMaxSubTiles, kGgmlTypeQ5_0><<<grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      return;
    case kGgmlTypeIQ2_XXS:
      MoEPackedFc2Bf16TcKernel<kMaxSubTiles, kGgmlTypeIQ2_XXS><<<grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      return;
    case kGgmlTypeIQ2_XS:
      MoEPackedFc2Bf16TcKernel<kMaxSubTiles, kGgmlTypeIQ2_XS><<<grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      return;
    case kGgmlTypeIQ2_S:
      MoEPackedFc2Bf16TcKernel<kMaxSubTiles, kGgmlTypeIQ2_S><<<grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      return;
    case kGgmlTypeIQ3_XXS:
      MoEPackedFc2Bf16TcKernel<kMaxSubTiles, kGgmlTypeIQ3_XXS><<<grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      return;
    case kGgmlTypeIQ4_XS:
      MoEPackedFc2Bf16TcKernel<kMaxSubTiles, kGgmlTypeIQ4_XS><<<grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      return;
    case kGgmlTypeQ8_0:
      MoEPackedFc2Bf16TcKernel<kMaxSubTiles, kGgmlTypeQ8_0><<<grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      return;
    default:
      MoEPackedFc2Bf16TcKernel<kMaxSubTiles, -1><<<grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      return;
  }
}

template <typename CudaT>
Status DispatchMoEPackedCuda(
    cudaStream_t stream,
    const CudaT* input,
    const CudaT* router_probs,
    const uint8_t* gate_weights,
    const uint8_t* up_weights,
    const uint8_t* down_weights,
    const MoEPackedParams& params,
    void* workspace,
    CudaT* output) {
  ORT_RETURN_IF_NOT(params.num_rows >= 0 && params.num_rows <= kMoEPackedMaxChunkRows &&
                        params.hidden_size > 0 && params.hidden_size <= kMoEPackedMaxDimension &&
                        params.inter_size > 0 && params.inter_size <= kMoEPackedMaxDimension &&
                        params.hidden_size % 32 == 0 && params.inter_size % 32 == 0 &&
                        params.num_experts > 0 && params.num_experts <= kMoEPackedMaxExperts &&
                        params.k > 0 && params.k <= kMoEPackedMaxK && params.k <= params.num_experts,
                    "MoEPacked parameters exceed the supported CUDA index range");
  if (params.num_rows == 0) return Status::OK();
  const int total_routes = params.num_rows * params.k;
  constexpr bool kIsBf16 = std::is_same<CudaT, __nv_bfloat16>::value;
  const bool use_tensor_cores = kIsBf16 && params.bf16_tc_supported;
  const bool use_prefill_tiles = use_tensor_cores && (total_routes > 128);
  const int max_tiles = use_prefill_tiles
                            ? ((total_routes + 31) / 32 + params.num_experts)
                            : ((total_routes <= 128) ? total_routes : ((total_routes + 7) / 8 + params.num_experts));
  // Compact tile descriptors each contain at least one route. Keep workspace offsets unchanged.
  const int launch_tiles = std::min(max_tiles, total_routes);

  uint8_t* ws_ptr = static_cast<uint8_t*>(workspace);
  float* topk_scales = reinterpret_cast<float*>(ws_ptr);
  ws_ptr += AlignUpWorkspace(total_routes, sizeof(float));

  int* topk_indices = reinterpret_cast<int*>(ws_ptr);
  ws_ptr += AlignUpWorkspace(total_routes, sizeof(int));

  int* sorted_routes = reinterpret_cast<int*>(ws_ptr);
  ws_ptr += AlignUpWorkspace(total_routes, sizeof(int));

  MoEPackedTile* tiles = reinterpret_cast<MoEPackedTile*>(ws_ptr);
  ws_ptr += AlignUpWorkspace(max_tiles, sizeof(MoEPackedTile));

  int* d_num_tiles = reinterpret_cast<int*>(ws_ptr);
  ws_ptr += AlignUpWorkspace(sizeof(int));

  CudaT* fc1_out = reinterpret_cast<CudaT*>(ws_ptr);
  ws_ptr += AlignUpWorkspace(SafeInt<size_t>(total_routes) * params.inter_size, sizeof(CudaT));

  CudaT* fc2_out = reinterpret_cast<CudaT*>(ws_ptr);

  // 1. Softmax + TopK
  const dim3 route_grid(static_cast<unsigned int>((params.num_rows + 7) / 8));
  if (params.num_experts == 512) {
    MoEPackedSoftmaxTopK512Kernel<CudaT><<<route_grid, 256, 0, stream>>>(
        router_probs, topk_scales, topk_indices, params.num_rows, params.k, params.normalize_routing_weights);
  } else {
    MoEPackedSoftmaxTopKGenericKernel<CudaT><<<route_grid, 256, 0, stream>>>(
        router_probs, topk_scales, topk_indices, params.num_rows, params.num_experts, params.k,
        params.normalize_routing_weights);
  }
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  // 2. Build expert-grouped tiles (8 routes/tile in decode, 32 routes/tile in prefill)
  if (use_prefill_tiles) {
    BuildMoEPackedTilesKernel<32><<<1, 512, 0, stream>>>(
        topk_indices, total_routes, params.num_experts, sorted_routes, tiles, d_num_tiles);
  } else {
    BuildMoEPackedTilesKernel<8><<<1, 512, 0, stream>>>(
        topk_indices, total_routes, params.num_experts, sorted_routes, tiles, d_num_tiles);
  }
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  if (use_tensor_cores) {
    if constexpr (kIsBf16) {
      // 3. Tensor-Core FC1 (64 inter_size channels/CTA in prefill, 16 inter_size channels/CTA Warp-K-Split in decode)
      if (use_prefill_tiles) {
        const dim3 fc1_grid(static_cast<unsigned int>((params.inter_size + 63) / 64),
                            static_cast<unsigned int>(launch_tiles));
        LaunchMoEPackedFc1Bf16TcDispatch<4>(
            stream, fc1_grid, input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      } else {
        const dim3 fc1_grid(static_cast<unsigned int>((params.inter_size + 15) / 16),
                            static_cast<unsigned int>(launch_tiles));
        LaunchMoEPackedFc1DecodeBf16TcDispatch(
            stream, fc1_grid, input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
      }
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

      // 4. Tensor-Core FC2 (128 hidden_size channels per CTA)
      const dim3 fc2_grid(static_cast<unsigned int>((params.hidden_size + 127) / 128),
                          static_cast<unsigned int>(launch_tiles));
      if (use_prefill_tiles) {
        LaunchMoEPackedFc2Bf16TcDispatch<4>(
            stream, fc2_grid, fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      } else {
        LaunchMoEPackedFc2Bf16TcDispatch<1>(
            stream, fc2_grid, fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
      }
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
    }
  } else {
    // Keep staging within CUDA's default shared-memory limit; larger shapes read activations directly.
    constexpr size_t kMaxStagingBytes = 48 * 1024;
    const size_t fc1_smem = SafeInt<size_t>(8) * params.hidden_size * sizeof(CudaT);
    const dim3 fc1_grid(static_cast<unsigned int>((params.inter_size + 7) / 8), static_cast<unsigned int>(launch_tiles));
    if (fc1_smem <= kMaxStagingBytes) {
      MoEPackedFc1SwiGLUSimtKernel<CudaT, true><<<fc1_grid, 256, fc1_smem, stream>>>(
          input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
    } else {
      MoEPackedFc1SwiGLUSimtKernel<CudaT, false><<<fc1_grid, 256, 0, stream>>>(
          input, gate_weights, up_weights, sorted_routes, tiles, d_num_tiles, params, fc1_out);
    }
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

    const size_t fc2_smem = SafeInt<size_t>(8) * params.inter_size * sizeof(CudaT);
    const dim3 fc2_grid(static_cast<unsigned int>((params.hidden_size + 7) / 8), static_cast<unsigned int>(launch_tiles));
    if (fc2_smem <= kMaxStagingBytes) {
      MoEPackedFc2SimtKernel<CudaT, true><<<fc2_grid, 256, fc2_smem, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
    } else {
      MoEPackedFc2SimtKernel<CudaT, false><<<fc2_grid, 256, 0, stream>>>(
          fc1_out, down_weights, sorted_routes, tiles, d_num_tiles, params, fc2_out);
    }
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  }

  // IO-bound output may only be element-aligned.
  if ((params.hidden_size & 7) == 0 && (reinterpret_cast<uintptr_t>(output) & 15u) == 0) {
    const int vec_cols = params.hidden_size >> 3;
    const dim3 fin_grid(static_cast<unsigned int>((vec_cols + 255) / 256),
                        static_cast<unsigned int>(params.num_rows));
    MoEPackedFinalizeVec8Kernel<CudaT><<<fin_grid, 256, 0, stream>>>(
        fc2_out, topk_scales, params.num_rows, params.hidden_size, params.k, output);
  } else {
    const dim3 fin_grid(static_cast<unsigned int>((params.hidden_size + 255) / 256),
                        static_cast<unsigned int>(params.num_rows));
    MoEPackedFinalizeKernel<CudaT><<<fin_grid, 256, 0, stream>>>(
        fc2_out, topk_scales, params.num_rows, params.hidden_size, params.k, output);
  }
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace

Status GetMoEPackedWorkspaceBytes(const MoEPackedParams& params, size_t elem_bytes, size_t& workspace_bytes) {
  ORT_RETURN_IF_NOT(params.num_rows >= 0 && params.num_rows <= kMoEPackedMaxChunkRows &&
                        params.k > 0 && params.k <= kMoEPackedMaxK &&
                        params.num_experts > 0 && params.num_experts <= kMoEPackedMaxExperts &&
                        params.hidden_size > 0 && params.hidden_size <= kMoEPackedMaxDimension &&
                        params.inter_size > 0 && params.inter_size <= kMoEPackedMaxDimension &&
                        elem_bytes == 2,
                    "MoEPacked workspace parameters exceed the supported range");
  if (params.num_rows == 0) {
    workspace_bytes = 0;
    return Status::OK();
  }
  const size_t total_routes = SafeInt<size_t>(params.num_rows) * params.k;
  const size_t max_tiles =
      (total_routes <= 128) ? total_routes : ((total_routes + 7) / 8 + static_cast<size_t>(params.num_experts));
  size_t bytes = 0;
  const auto add_buffer = [&bytes](size_t count, size_t element_bytes) {
    size_t buffer_bytes = 0;
    if (!IAllocator::CalcMemSizeForArrayWithAlignment(count, element_bytes, 256, &buffer_bytes) ||
        buffer_bytes > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) - bytes) {
      return false;
    }
    bytes += buffer_bytes;
    return true;
  };
  ORT_RETURN_IF_NOT(add_buffer(total_routes, sizeof(float)) &&
                        add_buffer(total_routes, sizeof(int)) &&
                        add_buffer(total_routes, sizeof(int)) &&
                        add_buffer(max_tiles, sizeof(MoEPackedTile)) &&
                        add_buffer(1, sizeof(int)) &&
                        add_buffer(SafeInt<size_t>(total_routes) * params.inter_size, elem_bytes) &&
                        add_buffer(SafeInt<size_t>(total_routes) * params.hidden_size, elem_bytes),
                    "MoEPacked workspace byte size exceeds the supported address range");
  workspace_bytes = bytes;
  return Status::OK();
}

Status GetMoEPackedBf16KernelSupport(int device_id, bool& supported) {
  int previous_device = -1;
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&previous_device));
  auto restore_device = gsl::finally([previous_device, device_id]() {
    if (previous_device != device_id) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaSetDevice(previous_device)));
    }
  });
  if (previous_device != device_id) {
    CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
  }
  cudaFuncAttributes attributes{};
  // All BF16 MMA instantiations in this translation unit use the same virtual architecture.
  CUDA_RETURN_IF_ERROR(cudaFuncGetAttributes(&attributes, MoEPackedFc1DecodeWarpKSplitKernel<4, 2, -1>));
  // binaryVersion can be >= 80 after JIT of compute_75 PTX, whose MMA body is still unsupported.
  supported = attributes.ptxVersion >= 80;
  return Status::OK();
}

Status LaunchMoEPackedCuda(
    cudaStream_t stream,
    bool is_bf16,
    const void* input,
    const void* router_probs,
    const uint8_t* gate_weights,
    const uint8_t* up_weights,
    const uint8_t* down_weights,
    const MoEPackedParams& params,
    void* workspace,
    void* output) {
  if (is_bf16) {
    return DispatchMoEPackedCuda<__nv_bfloat16>(
        stream, static_cast<const __nv_bfloat16*>(input), static_cast<const __nv_bfloat16*>(router_probs),
        gate_weights, up_weights, down_weights, params, workspace, static_cast<__nv_bfloat16*>(output));
  }
  return DispatchMoEPackedCuda<half>(
      stream, static_cast<const half*>(input), static_cast<const half*>(router_probs),
      gate_weights, up_weights, down_weights, params, workspace, static_cast<half*>(output));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
