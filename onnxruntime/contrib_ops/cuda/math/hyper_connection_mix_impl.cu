// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/hyper_connection_mix_impl.h"

#include <cfloat>
#include <cuda_fp16.h>

#include "contrib_ops/cuda/math/sinkhorn_normalize_impl.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kWarps = kThreadsPerBlock / 32;

// The unfused graph rounds to the activation type between the post mix and the pre mix, and
// again between the pre mix and the layer norm. `Round` reproduces those two trips so the
// operator stays comparable against the subgraph it replaces.
template <typename CudaT>
struct Conv;

template <>
struct Conv<float> {
  static __device__ __forceinline__ float ToFloat(float v) { return v; }
  static __device__ __forceinline__ float Round(float v) { return v; }
  static __device__ __forceinline__ float FromFloat(float v) { return v; }
};

template <>
struct Conv<half> {
  static __device__ __forceinline__ float ToFloat(half v) { return __half2float(v); }
  static __device__ __forceinline__ float Round(float v) { return __half2float(__float2half_rn(v)); }
  static __device__ __forceinline__ half FromFloat(float v) { return __float2half_rn(v); }
};

template <>
struct Conv<BFloat16> {
  static __device__ __forceinline__ float ToFloat(BFloat16 v) { return static_cast<float>(v); }
  static __device__ __forceinline__ float Round(float v) { return static_cast<float>(BFloat16(v)); }
  static __device__ __forceinline__ BFloat16 FromFloat(float v) { return BFloat16(v); }
};

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

// Post mix plus the mixing GEMV, over a slice of the hidden dimension.
//
// grid is (split, num_tokens, G). HC is a template parameter so every mixing accumulator is
// indexed at compile time and stays in registers; a runtime bound would spill the 25-wide
// array to local memory and the fused operator would lose to the subgraph it replaces.
//
// G splits the `hc` streams across blocks, so a block owns HC / G of them; see
// HyperConnectionPartialGroups(). It is a template parameter for the same reason HC is --
// the owned count has to be a compile-time trip count to keep `acc` in registers.
template <typename CudaT, int HC, int PT, int G>
__global__ __launch_bounds__(PT) void HyperConnectionPartialKernel(
    const CudaT* __restrict__ x,
    const CudaT* __restrict__ residual,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    const float* __restrict__ fn,
    CudaT* __restrict__ residual_out,
    float* __restrict__ partial,
    int dim, int split) {
  constexpr int kMixDim = (2 + HC) * HC;
  constexpr int kPtWarps = PT / 32;
  constexpr int kOwned = HC / G;
  using C = Conv<CudaT>;

  __shared__ float s_pin[HC];
  __shared__ float s_cin[HC * HC];
  __shared__ float s_red[kPtWarps * (kMixDim + 1)];

  const int part = blockIdx.x;
  const int token = blockIdx.y;
  const int group = (G > 1) ? static_cast<int>(blockIdx.z) : 0;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const size_t off_x = static_cast<size_t>(token) * dim;
  const size_t off_r = static_cast<size_t>(token) * HC * dim;

  if (tid < HC) s_pin[tid] = post_mix[token * HC + tid];
  if (tid < HC * HC) s_cin[tid] = comb_mix[token * HC * HC + tid];
  __syncthreads();

  float acc[kMixDim + 1];
#pragma unroll
  for (int i = 0; i <= kMixDim; ++i) acc[i] = 0.f;

  for (int d = part * PT + tid; d < dim; d += split * PT) {
    const float xv = C::ToFloat(x[off_x + d]);
    float rv[HC];
#pragma unroll
    for (int g = 0; g < HC; ++g) rv[g] = C::ToFloat(residual[off_r + static_cast<size_t>(g) * dim + d]);

    // One pass over the owned streams: forming v[h] and consuming it are independent across h,
    // and at G == 1 this still visits h in order, so `acc` is summed exactly as before.
#pragma unroll
    for (int i = 0; i < kOwned; ++i) {
      const int h = group + i * G;
      float t2 = 0.f;
#pragma unroll
      for (int g = 0; g < HC; ++g) t2 += s_cin[g * HC + h] * rv[g];
      const float vh = C::Round(s_pin[h] * xv + t2);

      residual_out[off_r + static_cast<size_t>(h) * dim + d] = C::FromFloat(vh);
      acc[kMixDim] += vh * vh;

      // A row of `fn` is kMixDim consecutive floats and rows are kMixDim apart, so whenever
      // kMixDim is a multiple of 4 every row is 16-byte aligned and the same bytes can be
      // fetched with a quarter of the instructions. Lanes are kMixDim * 4 bytes apart either
      // way, so this is purely an instruction-count change; each acc[j] keeps its order.
      const float* frow = fn + static_cast<size_t>(h * dim + d) * kMixDim;
      if constexpr (kMixDim % 4 == 0) {
        const float4* frow4 = reinterpret_cast<const float4*>(frow);
#pragma unroll
        for (int j = 0; j < kMixDim / 4; ++j) {
          const float4 f = frow4[j];
          acc[4 * j + 0] += vh * f.x;
          acc[4 * j + 1] += vh * f.y;
          acc[4 * j + 2] += vh * f.z;
          acc[4 * j + 3] += vh * f.w;
        }
      } else {
#pragma unroll
        for (int j = 0; j < kMixDim; ++j) acc[j] += vh * frow[j];
      }
    }
  }

#pragma unroll
  for (int i = 0; i <= kMixDim; ++i) {
    const float v = WarpReduceSum(acc[i]);
    if (lane == 0) s_red[warp * (kMixDim + 1) + i] = v;
  }
  __syncthreads();
  if (tid <= kMixDim) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < kPtWarps; ++w) v += s_red[w * (kMixDim + 1) + tid];
    // The finish pass walks this buffer as one run of G * split entries per token, so the
    // group index simply selects which run of `split` this block lands in.
    const size_t slot = (static_cast<size_t>(token) * G + group) * split + part;
    partial[slot * (kMixDim + 1) + tid] = v;
  }
}

// One block per token: finish the reduction, derive the gates, then the weighted sum of the
// streams and the layer's RMS norm.
template <typename CudaT, int HC>
__global__ void HyperConnectionFinishKernel(const CudaT* __restrict__ residual_out,
                                            const float* __restrict__ partial,
                                            const float* __restrict__ scale,
                                            const float* __restrict__ base,
                                            const float* __restrict__ norm_weight,
                                            float* __restrict__ post_mix_out,
                                            float* __restrict__ comb_mix_out,
                                            CudaT* __restrict__ layer_input,
                                            int dim, int split, int iterations, float epsilon,
                                            float hc_epsilon, float sinkhorn_epsilon,
                                            float post_alpha) {
  constexpr int kMixDim = (2 + HC) * HC;
  using C = Conv<CudaT>;

  extern __shared__ float smem[];
  float* s_y = smem;                     // [dim]      pre-mix output, kept for the norm
  float* s_red = s_y + dim;              // [kWarps]
  float* s_tot = s_red + kWarps;         // [kMixDim + 1]
  float* s_mix = s_tot + (kMixDim + 1);  // [kMixDim]
  float* s_pre = s_mix + kMixDim;        // [HC]
  float* s_comb = s_pre + HC;            // [HC * HC]
  float* s_sums = s_comb + HC * HC;      // [HC]       Sinkhorn scratch
  float* s_scalar = s_sums + HC;         // [1]

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const size_t off_x = static_cast<size_t>(token) * dim;
  const size_t off_r = static_cast<size_t>(token) * HC * dim;

  if (tid <= kMixDim) {
    const float* p = partial + static_cast<size_t>(token) * split * (kMixDim + 1) + tid;
    float v = 0.f;
    for (int s = 0; s < split; ++s) v += p[static_cast<size_t>(s) * (kMixDim + 1)];
    s_tot[tid] = v;
  }
  __syncthreads();

  // ---- pre mix: gates and the doubly stochastic combination matrix ----
  if (warp == 0) {
    const float rs = 1.0f / sqrtf(s_tot[kMixDim] / static_cast<float>(HC * dim) + epsilon);
    if (lane < kMixDim) s_mix[lane] = s_tot[lane] * rs;
    __syncwarp();

    if (lane < HC) {
      const float p = s_mix[lane] * scale[0] + base[lane];
      s_pre[lane] = 1.0f / (1.0f + expf(-p)) + hc_epsilon;

      const float q = s_mix[HC + lane] * scale[1] + base[HC + lane];
      const float gate = (1.0f / (1.0f + expf(-q))) * post_alpha;
      post_mix_out[token * HC + lane] = gate;

      // Row `lane` of the combination matrix, softmaxed over its last axis.
      float row[HC];
      float m = -FLT_MAX;
#pragma unroll
      for (int h = 0; h < HC; ++h) {
        const int idx = 2 * HC + lane * HC + h;
        row[h] = s_mix[idx] * scale[2] + base[idx];
        m = fmaxf(m, row[h]);
      }
      float s = 0.f;
#pragma unroll
      for (int h = 0; h < HC; ++h) {
        row[h] = expf(row[h] - m);
        s += row[h];
      }
#pragma unroll
      for (int h = 0; h < HC; ++h) s_comb[lane * HC + h] = row[h] / s + hc_epsilon;
    }
    __syncwarp();

    SinkhornNormalizeWarp(s_comb, s_sums, HC, lane, iterations, sinkhorn_epsilon);
    if (lane < HC * HC) comb_mix_out[token * HC * HC + lane] = s_comb[lane];
  }
  __syncthreads();

  // ---- weighted sum of the streams, then the layer's RMS norm ----
  float sy2 = 0.f;
  for (int d = tid; d < dim; d += kThreadsPerBlock) {
    float a = 0.f;
#pragma unroll
    for (int h = 0; h < HC; ++h) {
      a += s_pre[h] * C::ToFloat(residual_out[off_r + static_cast<size_t>(h) * dim + d]);
    }
    const float yv = C::Round(a);
    s_y[d] = yv;
    sy2 += yv * yv;
  }

  {
    const float v = WarpReduceSum(sy2);
    if (lane == 0) s_red[warp] = v;
    __syncthreads();
    if (tid == 0) {
      float t = 0.f;
      for (int w = 0; w < kWarps; ++w) t += s_red[w];
      s_scalar[0] = 1.0f / sqrtf(t / static_cast<float>(dim) + epsilon);
    }
    __syncthreads();
  }

  const float inv = s_scalar[0];
  for (int d = tid; d < dim; d += kThreadsPerBlock) {
    layer_input[off_x + d] = C::FromFloat(s_y[d] * inv * norm_weight[d]);
  }
}

// Same operator as HyperConnectionFinishKernel, restructured for the single-block decode case
// where the grid is one block and the whole kernel is a latency chain. Every step is
// value-preserving, so the two kernels agree bit for bit:
//
//   * `scale` and `base` are loaded before the reduction over `partial`, so their global
//     latency overlaps that reduction instead of sitting behind it in the gate expression.
//   * `s_tot` and `s_mix` were only ever read by warp 0, so they move into registers and the
//     first __syncthreads disappears; the other warps start streaming `residual_out` right
//     away instead of waiting out the reduction. The weighted sum still multiplies by
//     `s_pre` in h order after the barrier, so only the loads moved, not the arithmetic.
//   * Each lane of the combination matrix recomputes its own row's softmax, over the same
//     h = 0..HC-1 max and sum, rather than one lane per row publishing through shared memory.
//   * Sinkhorn runs on registers through SinkhornNormalizeWarpReg, which walks each axis in
//     the same order and performs the same single division per element.
//   * Sinkhorn feeds `comb_mix_out` alone -- `s_pre` comes off the sigmoid gate ahead of it --
//     so the barrier is placed as soon as `s_pre` is published and warp 0 runs its Sinkhorn
//     chain while the other warps stream the weighted sum. Warp 0 is therefore held out of the
//     dim loops and the remaining BLOCK - 32 threads cover `dim`. That only relabels which
//     thread handles which `d`; each `d` is still one thread accumulating over h in order.
//   * The dim loops use the whole worker set, but `sy2` is deliberately replayed afterwards by
//     threads 0..255 over d = t, t + 256, ... out of s_y, feeding the same eight-warp tree and
//     the same serial tail. The RMS scale therefore keeps the 256-thread grouping at any block
//     width, which is what makes the wider block bit-exact.
template <typename CudaT, int HC, int BLOCK, int PER, bool VEC>
__global__ __launch_bounds__(BLOCK) void HyperConnectionFinishFastKernel(
    const CudaT* __restrict__ residual_out, const float* __restrict__ partial,
    const float* __restrict__ scale, const float* __restrict__ base,
    const float* __restrict__ norm_weight, float* __restrict__ post_mix_out,
    float* __restrict__ comb_mix_out, CudaT* __restrict__ layer_input, int dim, int split,
    int iterations, float epsilon, float hc_epsilon, float sinkhorn_epsilon, float post_alpha) {
  constexpr int kMixDim = (2 + HC) * HC;
  constexpr int kWorkers = BLOCK - 32;  // warp 0 is reserved for the pre-mix and Sinkhorn
  using C = Conv<CudaT>;

  extern __shared__ float smem[];
  float* s_y = smem;              // [dim]
  float* s_red = s_y + dim;       // [kWarps]  the 256-thread tree, whatever BLOCK is
  float* s_pre = s_red + kWarps;  // [HC]
  float* s_scalar = s_pre + HC;   // [1]

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wtid = tid - 32;  // worker index, valid when warp != 0

  const size_t off_x = static_cast<size_t>(token) * dim;
  const size_t off_r = static_cast<size_t>(token) * HC * dim;

  // Nothing warp 0 is about to compute feeds these addresses, so issue them first.
  //
  // At decode this kernel is one block on one SM, so the only thing between it and its
  // floor is how many loads it can keep in flight.  Element-wise prefetching covered
  // PER * kWorkers = 1920 of dim = 4096 and walked the remaining 2176 through a
  // dependent tail loop -- four more serialised HBM round trips.  A 16-byte slot covers
  // kVec elements per thread, so two slots clear any dim up to 2 * kVec * kWorkers in one
  // shot, with every warp issuing full cache-line transactions instead of 2-byte ones.
  //
  // VEC is a template parameter, not a flag: as a runtime bool the compiler has to keep
  // registers live for both prefetch buffers (kSlots * HC uint4 = 32 plus PER * HC floats
  // = 16), which costs more than the vectorisation saves.
  constexpr int kVec = 16 / sizeof(CudaT);
  constexpr int kSlots = 2;
  const int vdim = dim / kVec;

  uint4 rvec[VEC ? kSlots : 1][HC];
  float rv[VEC ? 1 : PER * HC];
  if (warp != 0) {
    if constexpr (VEC) {
#pragma unroll
      for (int i = 0; i < kSlots; ++i) {
        const int v = wtid + i * kWorkers;
        if (v < vdim) {
#pragma unroll
          for (int h = 0; h < HC; ++h) {
            rvec[i][h] = reinterpret_cast<const uint4*>(
                residual_out + off_r + static_cast<size_t>(h) * dim)[v];
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const int d = wtid + i * kWorkers;
        if (d < dim) {
#pragma unroll
          for (int h = 0; h < HC; ++h) {
            rv[i * HC + h] = C::ToFloat(residual_out[off_r + static_cast<size_t>(h) * dim + d]);
          }
        }
      }
    }
  }

  float cv = 0.f;
  if (warp == 0) {
    const float bv = (lane < kMixDim) ? base[lane] : 0.f;
    const float sc = (lane < 3) ? scale[lane] : 0.f;

    float tot = 0.f;
    if (lane <= kMixDim) {
      const float* p = partial + static_cast<size_t>(token) * split * (kMixDim + 1) + lane;
      for (int s = 0; s < split; ++s) tot += p[static_cast<size_t>(s) * (kMixDim + 1)];
    }
    const float sq = __shfl_sync(0xffffffffu, tot, kMixDim);
    const float rs = 1.0f / sqrtf(sq / static_cast<float>(HC * dim) + epsilon);
    const float mix = tot * rs;

    const float sc0 = __shfl_sync(0xffffffffu, sc, 0);
    const float sc1 = __shfl_sync(0xffffffffu, sc, 1);
    const float sc2 = __shfl_sync(0xffffffffu, sc, 2);

    // The shuffles must run on the whole warp, so gather first and narrow afterwards.
    const int g = (lane < HC) ? lane : 0;
    const float mp = __shfl_sync(0xffffffffu, mix, g);
    const float bp = __shfl_sync(0xffffffffu, bv, g);
    const float mq = __shfl_sync(0xffffffffu, mix, HC + g);
    const float bq = __shfl_sync(0xffffffffu, bv, HC + g);
    if (lane < HC) {
      const float p = mp * sc0 + bp;
      s_pre[lane] = 1.0f / (1.0f + expf(-p)) + hc_epsilon;

      const float q = mq * sc1 + bq;
      post_mix_out[token * HC + lane] = (1.0f / (1.0f + expf(-q))) * post_alpha;
    }

    // Lane `lane` owns element (r, c) and softmaxes row r locally.
    const int e = (lane < HC * HC) ? lane : 0;
    const int r = e / HC;
    const int c = e - r * HC;
    float row[HC];
    float m = -FLT_MAX;
#pragma unroll
    for (int h = 0; h < HC; ++h) {
      const int idx = 2 * HC + r * HC + h;
      row[h] = __shfl_sync(0xffffffffu, mix, idx) * sc2 + __shfl_sync(0xffffffffu, bv, idx);
      m = fmaxf(m, row[h]);
    }
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < HC; ++h) {
      row[h] = expf(row[h] - m);
      s += row[h];
    }
    cv = row[c] / s + hc_epsilon;
  }
  __syncthreads();  // publishes s_pre; Sinkhorn has deliberately not run yet

  if (warp == 0) {
    // Overlaps the weighted sum below, which does not depend on it.
    cv = SinkhornNormalizeWarpReg<HC>(cv, lane, iterations, sinkhorn_epsilon);
    if (lane < HC * HC) comb_mix_out[token * HC * HC + lane] = cv;
  } else {
    float pre[HC];
#pragma unroll
    for (int h = 0; h < HC; ++h) pre[h] = s_pre[h];

    if constexpr (VEC) {
#pragma unroll
      for (int i = 0; i < kSlots; ++i) {
        const int v = wtid + i * kWorkers;
        if (v < vdim) {
          CudaT e[HC][kVec];
#pragma unroll
          for (int h = 0; h < HC; ++h) *reinterpret_cast<uint4*>(e[h]) = rvec[i][h];
#pragma unroll
          for (int j = 0; j < kVec; ++j) {
            float a = 0.f;
#pragma unroll
            for (int h = 0; h < HC; ++h) a += pre[h] * C::ToFloat(e[h][j]);
            s_y[v * kVec + j] = C::Round(a);
          }
        }
      }
      // Only runs for dim > kSlots * kVec * kWorkers, which no supported hidden size hits.
      for (int v = wtid + kSlots * kWorkers; v < vdim; v += kWorkers) {
        CudaT e[HC][kVec];
#pragma unroll
        for (int h = 0; h < HC; ++h) {
          *reinterpret_cast<uint4*>(e[h]) = reinterpret_cast<const uint4*>(
              residual_out + off_r + static_cast<size_t>(h) * dim)[v];
        }
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
          float a = 0.f;
#pragma unroll
          for (int h = 0; h < HC; ++h) a += pre[h] * C::ToFloat(e[h][j]);
          s_y[v * kVec + j] = C::Round(a);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const int d = wtid + i * kWorkers;
        if (d < dim) {
          float a = 0.f;
#pragma unroll
          for (int h = 0; h < HC; ++h) a += pre[h] * rv[i * HC + h];
          s_y[d] = C::Round(a);
        }
      }
      for (int d = wtid + PER * kWorkers; d < dim; d += kWorkers) {
        float a = 0.f;
#pragma unroll
        for (int h = 0; h < HC; ++h) {
          a += pre[h] * C::ToFloat(residual_out[off_r + static_cast<size_t>(h) * dim + d]);
        }
        s_y[d] = C::Round(a);
      }
    }
  }
  __syncthreads();

  // Replay of the 256-thread accumulation so the RMS scale does not depend on BLOCK.
  if (tid < kThreadsPerBlock) {
    float sy2 = 0.f;
    for (int d = tid; d < dim; d += kThreadsPerBlock) {
      const float yv = s_y[d];
      sy2 += yv * yv;
    }
    const float v = WarpReduceSum(sy2);
    if (lane == 0) s_red[warp] = v;
  }
  __syncthreads();
  if (tid == 0) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) t += s_red[w];
    s_scalar[0] = 1.0f / sqrtf(t / static_cast<float>(dim) + epsilon);
  }
  __syncthreads();

  const float inv = s_scalar[0];
  if constexpr (VEC) {
    for (int v = tid; v < vdim; v += BLOCK) {
      CudaT out[kVec];
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const int d = v * kVec + j;
        out[j] = C::FromFloat(s_y[d] * inv * norm_weight[d]);
      }
      reinterpret_cast<uint4*>(layer_input + off_x)[v] = *reinterpret_cast<const uint4*>(out);
    }
  } else {
    for (int d = tid; d < dim; d += BLOCK) {
      layer_input[off_x + d] = C::FromFloat(s_y[d] * inv * norm_weight[d]);
    }
  }
}

// Kill switch for the restructured finish kernel. It is bit-identical to the original, so this
// exists only to bisect a regression; set ORT_DISABLE_HC_FINISH_FAST=1 to take the old path.
// The environment helper drags in logging macros that clash with provider_api.h under nvcc,
// so the query itself lives in hyper_connection_mix.cc.

template <typename CudaT, int HC>
Status Launch(cudaStream_t stream, const HyperConnectionMixParams& params, const CudaT* x,
              const CudaT* residual, const float* post_mix, const float* comb_mix,
              const float* fn, const float* scale, const float* base, const float* norm_weight,
              float* workspace, CudaT* residual_out, float* post_mix_out, float* comb_mix_out,
              CudaT* layer_input) {
  constexpr int kMixDim = (2 + HC) * HC;
  const size_t floats = static_cast<size_t>(params.dim) + kWarps + (kMixDim + 1) + kMixDim +
                        2 * HC + HC * HC + 1;
  const size_t shared_bytes = floats * sizeof(float);
  if (shared_bytes > 48u * 1024u) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "HyperConnectionMix needs ", shared_bytes,
                           " bytes of shared memory per block, which exceeds the 48 KiB limit. "
                           "Reduce the hidden size.");
  }

  const int split = HyperConnectionMixSplit(params.num_tokens, params.dim);
  const int groups = HyperConnectionPartialGroups(params.num_tokens, HC, params.dim);

  // Block width picks how many SMs the pass can occupy; see HyperConnectionPartialThreads().
  const int partial_threads = HyperConnectionPartialThreads(params.num_tokens);
  const dim3 partial_grid(split, params.num_tokens, groups);
#define ORT_HC_LAUNCH_PARTIAL(pt, g)                                               \
  HyperConnectionPartialKernel<CudaT, HC, pt, g><<<partial_grid, pt, 0, stream>>>( \
      x, residual, post_mix, comb_mix, fn, residual_out, workspace, params.dim, split)
  if (groups == 1) {
    if (partial_threads == 32) {
      ORT_HC_LAUNCH_PARTIAL(32, 1);
    } else if (partial_threads == 128) {
      ORT_HC_LAUNCH_PARTIAL(128, 1);
    } else {
      ORT_HC_LAUNCH_PARTIAL(64, 1);
    }
  } else {
    if (partial_threads == 32) {
      ORT_HC_LAUNCH_PARTIAL(32, HC);
    } else if (partial_threads == 128) {
      ORT_HC_LAUNCH_PARTIAL(128, HC);
    } else {
      ORT_HC_LAUNCH_PARTIAL(64, HC);
    }
  }
#undef ORT_HC_LAUNCH_PARTIAL

  // Each group contributed its own run of `split` partials, and they reduce the same way.
  const int finish_split = split * groups;

  if (HyperConnectionFinishFastDisabled()) {
    HyperConnectionFinishKernel<CudaT, HC>
        <<<params.num_tokens, kThreadsPerBlock, shared_bytes, stream>>>(
            residual_out, workspace, scale, base, norm_weight, post_mix_out, comb_mix_out,
            layer_input, params.dim, finish_split, params.sinkhorn_iterations, params.epsilon,
            params.hc_epsilon, params.sinkhorn_epsilon, params.post_alpha);
    return CUDA_CALL(cudaGetLastError());
  }

  // One block per token, so at decode the whole kernel is a single block on a single SM and
  // wants a wide block; at prefill the blocks hide each other and a wide block only costs
  // occupancy. 512 measured best or tied at every token count from 1 to 4096, so there is no
  // reason to carry a second instantiation.
  constexpr int kFastBlock = 512;
  constexpr int kFastPrefetch = 4;
  constexpr int kFastVec = 16 / sizeof(CudaT);
  if (params.dim % kFastVec == 0 && !HyperConnectionFinishVecDisabled()) {
    HyperConnectionFinishFastKernel<CudaT, HC, kFastBlock, kFastPrefetch, true>
        <<<params.num_tokens, kFastBlock, shared_bytes, stream>>>(
            residual_out, workspace, scale, base, norm_weight, post_mix_out, comb_mix_out,
            layer_input, params.dim, finish_split, params.sinkhorn_iterations, params.epsilon,
            params.hc_epsilon, params.sinkhorn_epsilon, params.post_alpha);
  } else {
    HyperConnectionFinishFastKernel<CudaT, HC, kFastBlock, kFastPrefetch, false>
        <<<params.num_tokens, kFastBlock, shared_bytes, stream>>>(
            residual_out, workspace, scale, base, norm_weight, post_mix_out, comb_mix_out,
            layer_input, params.dim, finish_split, params.sinkhorn_iterations, params.epsilon,
            params.hc_epsilon, params.sinkhorn_epsilon, params.post_alpha);
  }

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace

template <typename T>
Status LaunchHyperConnectionMix(cudaStream_t stream, const HyperConnectionMixParams& params,
                                const T* x, const T* residual, const float* post_mix,
                                const float* comb_mix, const float* fn, const float* scale,
                                const float* base, const float* norm_weight, float* workspace,
                                T* residual_out, float* post_mix_out, float* comb_mix_out,
                                T* layer_input) {
  if (params.num_tokens == 0) {
    return Status::OK();
  }

  using CudaT = typename onnxruntime::cuda::ToCudaType<T>::MappedType;
  auto* xc = reinterpret_cast<const CudaT*>(x);
  auto* rc = reinterpret_cast<const CudaT*>(residual);
  auto* ro = reinterpret_cast<CudaT*>(residual_out);
  auto* li = reinterpret_cast<CudaT*>(layer_input);

#define LAUNCH_HC(hc)                                                                     \
  case hc:                                                                                \
    return Launch<CudaT, hc>(stream, params, xc, rc, post_mix, comb_mix, fn, scale, base, \
                             norm_weight, workspace, ro, post_mix_out, comb_mix_out, li)

  switch (params.hc) {
    LAUNCH_HC(1);
    LAUNCH_HC(2);
    LAUNCH_HC(3);
    LAUNCH_HC(4);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "HyperConnectionMix supports a multiplicity of at most ",
                             kHyperConnectionMaxMult, ", got ", params.hc);
  }
#undef LAUNCH_HC
}

template Status LaunchHyperConnectionMix<float>(cudaStream_t, const HyperConnectionMixParams&,
                                                const float*, const float*, const float*,
                                                const float*, const float*, const float*,
                                                const float*, const float*, float*, float*,
                                                float*, float*, float*);

template Status LaunchHyperConnectionMix<MLFloat16>(cudaStream_t, const HyperConnectionMixParams&,
                                                    const MLFloat16*, const MLFloat16*,
                                                    const float*, const float*, const float*,
                                                    const float*, const float*, const float*,
                                                    float*, MLFloat16*, float*, float*,
                                                    MLFloat16*);

template Status LaunchHyperConnectionMix<BFloat16>(cudaStream_t, const HyperConnectionMixParams&,
                                                   const BFloat16*, const BFloat16*, const float*,
                                                   const float*, const float*, const float*,
                                                   const float*, const float*, float*, BFloat16*,
                                                   float*, float*, BFloat16*);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
