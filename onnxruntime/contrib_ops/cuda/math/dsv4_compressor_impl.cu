// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/dsv4_compressor_impl.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>

#include "contrib_ops/cuda/math/dsv4_common.cuh"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

constexpr int kThreads = kDSV4CompressorThreads;
constexpr int kWarps = kThreads / 32;
constexpr float kNegInf = -1e30f;

// The pooling kernel gives every channel its own thread and a private team of loaders, so its
// block is shaped (kPoolChannels, loaders) instead of the flat kThreads the other two use.
constexpr int kPoolChannels = 32;    // channels per block; one warp wide, so smem is conflict-free
constexpr int kPoolMaxLoaders = 32;  // cap: kPoolChannels * kPoolMaxLoaders == 1024 threads
constexpr int kPoolMinLoaders = 8;
constexpr int kPoolReduceUnroll = 8;  // in-flight shared loads per sequential accumulate step
constexpr size_t kPoolMaxSharedBytes = 48u * 1024u;

// Kill switch for the fast pooling kernel. Set ORT_DISABLE_DSV4_POOL_FAST=1 to fall back to the
// one-thread-per-channel reference kernel below, which is what the fast path is validated against.
bool DSV4PoolFastDisabledByEnv() {
  // Parsed once via ORT's environment helper (consistent parsing/thread-safety across platforms).
  static const bool disabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_DSV4_POOL_FAST", 0) == 1;
  return disabled;
}

int DSV4PoolLoaders(int span) {
  const int want = span < kPoolMinLoaders ? kPoolMinLoaders : span;
  return want > kPoolMaxLoaders ? kPoolMaxLoaders : want;
}

size_t DSV4PoolSharedBytes(int span) {
  return (2u * static_cast<size_t>(span) + 2u) * kPoolChannels * sizeof(float);
}

// One candidate row's worth of pooling, one thread per channel.
//
// grid is (channel blocks, J, B).  The softmax runs over the pooling window independently for
// every channel, so the channels never have to talk to each other and the row can be spread
// over as many blocks as it takes to fill the device -- which matters, because a decode step
// only produces two rows per sequence.
template <typename CudaT, typename CudaP>
__global__ void DSV4CompressorPoolKernel(const DSV4CompressorParams p,
                                         const CudaP* __restrict__ kv,
                                         const CudaP* __restrict__ score,
                                         const float* __restrict__ past_kv,
                                         const float* __restrict__ past_score,
                                         const float* __restrict__ ape,
                                         const int64_t* __restrict__ past_lens,
                                         CudaT* __restrict__ rows) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= p.head_dim) return;
  const int j = blockIdx.y;
  const int b = blockIdx.z;

  const int64_t past = past_lens[b];
  const int64_t total = past + p.seq_len;
  const int64_t n_full = p.span + p.seq_len;
  const int64_t base = past - p.span;
  const int64_t pos0 = (past / p.ratio + j) * static_cast<int64_t>(p.ratio);

  const int64_t past_row = static_cast<int64_t>(b) * p.span;
  const int64_t cur_row = static_cast<int64_t>(b) * p.seq_len;

  // The graph normalizes every weight and then sums v * w, so the pooling has to do the same
  // in the same order: a fused sum-then-divide rounds differently, and the FP8/FP4 grids
  // downstream turn that into whole-step flips.
  auto gather = [&](int s, int64_t* off, int* k, bool* valid, bool* use_past) {
    // The overlapping form pools the previous window into the low half of the projection and
    // the current one into the high half.
    int fo = ch;
    int64_t pos = pos0 + s;
    bool ok = true;
    *k = s;
    if (p.coff == 2) {
      if (s < p.ratio) {
        pos = pos0 + s - p.ratio;
        ok = pos >= 0;
      } else {
        *k = s - p.ratio;
        pos = pos0 + *k;
        fo = p.head_dim + ch;
      }
    }

    const int64_t idx = pos - base;
    *valid = ok && idx >= 0 && idx < n_full && pos < total;
    int64_t cl = idx < 0 ? 0 : idx;
    if (cl > n_full - 1) cl = n_full - 1;

    *use_past = cl < p.span;
    *off = *use_past ? (past_row + cl) * p.feat + fo
                     : (cur_row + cl - p.span) * p.proj_feat + fo;
    return fo;
  };

  float m = -FLT_MAX;
  for (int s = 0; s < p.span; ++s) {
    int64_t off;
    int k;
    bool valid, use_past;
    const int fo = gather(s, &off, &k, &valid, &use_past);
    const float w = valid ? (use_past ? past_score[off] : DSV4Conv<CudaP>::ToFloat(score[off])) +
                                ape[k * p.feat + fo]
                          : kNegInf;
    m = fmaxf(m, w);
  }

  float denom = 0.0f;
  for (int s = 0; s < p.span; ++s) {
    int64_t off;
    int k;
    bool valid, use_past;
    const int fo = gather(s, &off, &k, &valid, &use_past);
    const float w = valid ? (use_past ? past_score[off] : DSV4Conv<CudaP>::ToFloat(score[off])) +
                                ape[k * p.feat + fo]
                          : kNegInf;
    denom += expf(w - m);
  }

  float acc = 0.0f;
  for (int s = 0; s < p.span; ++s) {
    int64_t off;
    int k;
    bool valid, use_past;
    const int fo = gather(s, &off, &k, &valid, &use_past);
    const float w = valid ? (use_past ? past_score[off] : DSV4Conv<CudaP>::ToFloat(score[off])) +
                                ape[k * p.feat + fo]
                          : kNegInf;
    acc += (use_past ? past_kv[off] : DSV4Conv<CudaP>::ToFloat(kv[off])) * (expf(w - m) / denom);
  }

  rows[(static_cast<int64_t>(b) * p.num_rows + j) * p.head_dim + ch] =
      DSV4Conv<CudaT>::FromFloat(acc);
}

// The same pooling, but with the window staged through shared memory by a team of loaders.
//
// The reference kernel above exposes only `head_dim * J * B` threads, so a decode step runs on a
// handful of blocks and every one of the three passes pays the full global-load latency `span`
// times over. Here the block is (kPoolChannels, loaders): the loader team fetches the whole
// window once into shared memory, and the per-element `expf` and the divide -- which are what the
// three passes actually spend their time on, the IEEE divide most of all -- are spread over the
// same team.
//
// What stays strictly serial is the arithmetic that is allowed to be: `m`, `denom` and `acc` are
// still accumulated by one thread in increasing `s`. The graph normalizes every weight and then
// sums v * w, so the pooling has to do the same in the same order: a fused sum-then-divide rounds
// differently, and the FP8/FP4 grids downstream turn that into whole-step flips. Hoisting `expf`
// and the divide out is safe because each result only depends on its own `s`; reassociating the
// sums would not be.
template <typename CudaT, typename CudaP, int COFF>
__global__ void DSV4CompressorPoolFastKernel(const DSV4CompressorParams p,
                                             const CudaP* __restrict__ kv,
                                             const CudaP* __restrict__ score,
                                             const float* __restrict__ past_kv,
                                             const float* __restrict__ past_score,
                                             const float* __restrict__ ape,
                                             const int64_t* __restrict__ past_lens,
                                             CudaT* __restrict__ rows) {
  extern __shared__ float smem[];
  const int c = threadIdx.x;
  const int loader = threadIdx.y;
  const int loaders = blockDim.y;
  const int ch = blockIdx.x * kPoolChannels + c;
  const int j = blockIdx.y;
  const int b = blockIdx.z;

  const int span = p.span;
  const int ratio = p.ratio;
  const int feat = p.feat;
  const int proj_feat = p.proj_feat;
  const int64_t past = past_lens[b];
  const int64_t total = past + p.seq_len;
  const int64_t n_full = span + p.seq_len;
  const int64_t base = past - span;
  const int64_t pos0 = (past / ratio + j) * static_cast<int64_t>(ratio);
  const int64_t past_base = static_cast<int64_t>(b) * span * feat;
  const int64_t cur_base = (static_cast<int64_t>(b) * p.seq_len - span) *
                           static_cast<int64_t>(proj_feat);
  const bool live = ch < p.head_dim;

  // [span][kPoolChannels] weights (rewritten in place as exp, then as the normalized weight),
  // [span][kPoolChannels] values, then one `m` and one `denom` per channel.
  float* s_w = smem;
  float* s_v = s_w + static_cast<size_t>(span) * kPoolChannels;
  float* s_max = s_v + static_cast<size_t>(span) * kPoolChannels;
  float* s_denom = s_max + kPoolChannels;

  auto stage = [&](int s) {
    // The overlapping form pools the previous window into the low half of the projection and
    // the current one into the high half.
    int fo = ch;
    int k = s;
    int64_t pos = pos0 + s;
    bool ok = true;
    if (COFF == 2) {
      if (s < ratio) {
        pos = pos0 + s - ratio;
        ok = pos >= 0;
      } else {
        k = s - ratio;
        pos = pos0 + k;
        fo = p.head_dim + ch;
      }
    }
    const int64_t idx = pos - base;
    const bool valid = ok && idx >= 0 && idx < n_full && pos < total;
    int64_t cl = idx < 0 ? 0 : idx;
    if (cl > n_full - 1) cl = n_full - 1;
    const bool use_past = cl < span;
    const int64_t off = use_past ? past_base + cl * feat + fo : cur_base + cl * proj_feat + fo;
    const size_t slot = static_cast<size_t>(s) * kPoolChannels + c;
    s_w[slot] = (valid && live)
                    ? (use_past ? past_score[off] : DSV4Conv<CudaP>::ToFloat(score[off])) +
                          ape[static_cast<int64_t>(k) * feat + fo]
                    : kNegInf;
    s_v[slot] = live ? (use_past ? past_kv[off] : DSV4Conv<CudaP>::ToFloat(kv[off])) : 0.0f;
  };

#pragma unroll 4
  for (int s = loader; s < span; s += loaders) stage(s);
  __syncthreads();

  if (loader == 0) {
    float m = -FLT_MAX;
    int s = 0;
    for (; s + kPoolReduceUnroll <= span; s += kPoolReduceUnroll) {
      float w[kPoolReduceUnroll];
#pragma unroll
      for (int u = 0; u < kPoolReduceUnroll; ++u)
        w[u] = s_w[static_cast<size_t>(s + u) * kPoolChannels + c];
#pragma unroll
      for (int u = 0; u < kPoolReduceUnroll; ++u) m = fmaxf(m, w[u]);
    }
    for (; s < span; ++s) m = fmaxf(m, s_w[static_cast<size_t>(s) * kPoolChannels + c]);
    s_max[c] = m;
  }
  __syncthreads();

  {
    const float m = s_max[c];
#pragma unroll 4
    for (int s = loader; s < span; s += loaders) {
      const size_t slot = static_cast<size_t>(s) * kPoolChannels + c;
      s_w[slot] = expf(s_w[slot] - m);
    }
  }
  __syncthreads();

  if (loader == 0) {
    float denom = 0.0f;
    int s = 0;
    for (; s + kPoolReduceUnroll <= span; s += kPoolReduceUnroll) {
      float e[kPoolReduceUnroll];
#pragma unroll
      for (int u = 0; u < kPoolReduceUnroll; ++u)
        e[u] = s_w[static_cast<size_t>(s + u) * kPoolChannels + c];
#pragma unroll
      for (int u = 0; u < kPoolReduceUnroll; ++u) denom += e[u];
    }
    for (; s < span; ++s) denom += s_w[static_cast<size_t>(s) * kPoolChannels + c];
    s_denom[c] = denom;
  }
  __syncthreads();

  {
    const float denom = s_denom[c];
#pragma unroll 4
    for (int s = loader; s < span; s += loaders) {
      const size_t slot = static_cast<size_t>(s) * kPoolChannels + c;
      s_w[slot] = s_w[slot] / denom;
    }
  }
  __syncthreads();

  if (loader != 0 || !live) return;

  float acc = 0.0f;
  int s = 0;
  for (; s + kPoolReduceUnroll <= span; s += kPoolReduceUnroll) {
    float w[kPoolReduceUnroll];
    float v[kPoolReduceUnroll];
#pragma unroll
    for (int u = 0; u < kPoolReduceUnroll; ++u) {
      const size_t slot = static_cast<size_t>(s + u) * kPoolChannels + c;
      w[u] = s_w[slot];
      v[u] = s_v[slot];
    }
#pragma unroll
    for (int u = 0; u < kPoolReduceUnroll; ++u) acc += v[u] * w[u];
  }
  for (; s < span; ++s) {
    const size_t slot = static_cast<size_t>(s) * kPoolChannels + c;
    acc += s_v[slot] * s_w[slot];
  }

  rows[(static_cast<int64_t>(b) * p.num_rows + j) * p.head_dim + ch] =
      DSV4Conv<CudaT>::FromFloat(acc);
}

// Norm, rotate, and the simulated low-precision round trips, one block per candidate row.
//
// Reads back what the pooling kernel wrote, which is also where the graph rounds to the
// activation type, so the round trip is reproduced rather than skipped.
template <typename CudaT>
__global__ void DSV4CompressorFinishKernel(const DSV4CompressorParams p,
                                           const float* __restrict__ norm_weight,
                                           const float* __restrict__ cos_table,
                                           const float* __restrict__ sin_table,
                                           const int64_t* __restrict__ past_lens,
                                           CudaT* __restrict__ rows) {
  extern __shared__ float smem[];
  float* s_row = smem;
  float* s_red = s_row + p.head_dim;
  float* s_rope = s_red + kWarps;
  float* s_scale = s_rope + p.rope_head_dim;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int d = p.head_dim;
  const int j = blockIdx.x;
  const int b = blockIdx.y;
  CudaT* row = rows + (static_cast<int64_t>(b) * p.num_rows + j) * d;

  float ss = 0.0f;
  for (int c = tid; c < d; c += kThreads) {
    const float v = DSV4Conv<CudaT>::ToFloat(row[c]);
    s_row[c] = v;
    ss += v * v;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, offset);
  if (lane == 0) s_red[warp] = ss;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kWarps; ++w) total += s_red[w];
    s_red[0] = 1.0f / sqrtf(total / d + p.epsilon);
  }
  __syncthreads();

  const float rs = s_red[0];
  for (int c = tid; c < d; c += kThreads) s_row[c] = s_row[c] * rs * norm_weight[c];
  __syncthreads();

  const int rd = p.rope_head_dim;
  if (rd > 0) {
    int64_t slot_pos = (past_lens[b] / p.ratio + j) * static_cast<int64_t>(p.ratio);
    if (slot_pos < 0) slot_pos = 0;
    if (slot_pos > p.max_seq_len - 1) slot_pos = p.max_seq_len - 1;
    const float* cos_row = cos_table + slot_pos * rd;
    const float* sin_row = sin_table + slot_pos * rd;
    for (int t = tid; t < rd; t += kThreads) s_rope[t] = s_row[p.nope_dim + t];
    __syncthreads();
    for (int t = tid; t < rd; t += kThreads) {
      // The tables are already interleaved, so the rotation is the signed swap of each pair.
      const float rot = (t & 1) ? s_rope[t - 1] : -s_rope[t + 1];
      s_row[p.nope_dim + t] = s_rope[t] * cos_row[t] + rot * sin_row[t];
    }
    __syncthreads();
  }

  if (p.act_quant) {
    const int blocks = p.nope_dim / 64;
    if (tid < blocks) {
      float amax = 0.0f;
      for (int i = 0; i < 64; ++i) amax = fmaxf(amax, fabsf(s_row[tid * 64 + i]));
      s_scale[tid] = DSV4BlockScale(amax, kDSV4Fp8Max, 1e-30f);
    }
    __syncthreads();
    for (int c = tid; c < p.nope_dim; c += kThreads) {
      const float scale = s_scale[c >> 6];
      const float q = fminf(fmaxf(s_row[c] / scale, -kDSV4Fp8Max), kDSV4Fp8Max);
      s_row[c] = DSV4RoundE4M3(q) * scale;
    }
    __syncthreads();
  }

  if (p.rotate_fp4) DSV4RotateFp4<CudaT>(s_row, s_scale, d, tid, kThreads);

  for (int c = tid; c < d; c += kThreads) row[c] = DSV4Conv<CudaT>::FromFloat(s_row[c]);
}

// Roll the raw projections forward by one step and publish the slot bookkeeping.
//
// The state is kept in float whatever the projections arrive as: it is a few hundred KiB that
// every later step re-reads, and widening it here keeps the pooling above reading one type.
template <typename CudaP>
__global__ void DSV4CompressorStateKernel(const DSV4CompressorParams p,
                                          const CudaP* __restrict__ kv,
                                          const CudaP* __restrict__ score,
                                          const float* __restrict__ past_kv,
                                          const float* __restrict__ past_score,
                                          const int64_t* __restrict__ past_lens,
                                          float* __restrict__ out_kv,
                                          float* __restrict__ out_score,
                                          int64_t* __restrict__ first_slot,
                                          int64_t* __restrict__ last_slot,
                                          int64_t* __restrict__ row_count) {
  const int b = blockIdx.y;
  const int64_t n = static_cast<int64_t>(p.span) * p.feat;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const int64_t r = i / p.feat;
    const int64_t f = i - r * p.feat;
    const int64_t src = p.seq_len + r;  // the tail `span` rows of past ++ current
    const int64_t dst = (static_cast<int64_t>(b) * p.span + r) * p.feat + f;
    if (src < p.span) {
      const int64_t off = (static_cast<int64_t>(b) * p.span + src) * p.feat + f;
      out_kv[dst] = past_kv[off];
      out_score[dst] = past_score[off];
    } else {
      const int64_t off = (static_cast<int64_t>(b) * p.seq_len + src - p.span) * p.proj_feat + f;
      out_kv[dst] = DSV4Conv<CudaP>::ToFloat(kv[off]);
      out_score[dst] = DSV4Conv<CudaP>::ToFloat(score[off]);
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int64_t past = past_lens[b];
    first_slot[b] = past / p.ratio;
    last_slot[b] = (past + p.seq_len - 1) / p.ratio;
    if (b == 0) *row_count = (p.seq_len - 1) / p.ratio + 2;
  }
}

}  // namespace

template <typename T, typename P>
Status LaunchDSV4Compressor(cudaStream_t stream, const DSV4CompressorParams& params,
                            const P* kv, const P* score,
                            const float* past_state_kv, const float* past_state_score,
                            const float* ape, const float* norm_weight,
                            const float* cos_table, const float* sin_table,
                            const int64_t* past_lens,
                            T* rows, int64_t* first_slot, int64_t* last_slot, int64_t* row_count,
                            float* present_state_kv, float* present_state_score) {
  using CudaT = typename ::onnxruntime::cuda::ToCudaType<T>::MappedType;
  using CudaP = typename ::onnxruntime::cuda::ToCudaType<P>::MappedType;
  auto* out = reinterpret_cast<CudaT*>(rows);
  const auto* proj_kv = reinterpret_cast<const CudaP*>(kv);
  const auto* proj_score = reinterpret_cast<const CudaP*>(score);

  const size_t pool_shared = DSV4PoolSharedBytes(params.span);
  if (!DSV4PoolFastDisabledByEnv() && pool_shared <= kPoolMaxSharedBytes) {
    const dim3 grid((params.head_dim + kPoolChannels - 1) / kPoolChannels, params.num_rows,
                    params.batch);
    const dim3 block(kPoolChannels, DSV4PoolLoaders(params.span));
    if (params.coff == 2) {
      DSV4CompressorPoolFastKernel<CudaT, CudaP, 2><<<grid, block, pool_shared, stream>>>(
          params, proj_kv, proj_score, past_state_kv, past_state_score, ape, past_lens, out);
    } else {
      DSV4CompressorPoolFastKernel<CudaT, CudaP, 1><<<grid, block, pool_shared, stream>>>(
          params, proj_kv, proj_score, past_state_kv, past_state_score, ape, past_lens, out);
    }
  } else {
    const int channel_blocks = (params.head_dim + kThreads - 1) / kThreads;
    DSV4CompressorPoolKernel<CudaT, CudaP><<<dim3(channel_blocks, params.num_rows, params.batch),
                                             kThreads, 0, stream>>>(
        params, proj_kv, proj_score, past_state_kv, past_state_score, ape, past_lens, out);
  }

  const size_t shared = DSV4CompressorFinishSharedFloats(params) * sizeof(float);
  if (shared > 48u * 1024u) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "DSV4Compressor needs ", shared,
                           " bytes of shared memory per row, "
                           "which is over the 48 KiB limit.");
  }
  DSV4CompressorFinishKernel<CudaT><<<dim3(params.num_rows, params.batch), kThreads, shared,
                                      stream>>>(
      params, norm_weight, cos_table, sin_table, past_lens, out);

  const int64_t state_elems = static_cast<int64_t>(params.span) * params.feat;
  const int state_blocks = static_cast<int>(std::min<int64_t>(
      1024, (state_elems + kThreads - 1) / kThreads));
  DSV4CompressorStateKernel<CudaP><<<dim3(state_blocks, params.batch), kThreads, 0, stream>>>(
      params, proj_kv, proj_score, past_state_kv, past_state_score, past_lens,
      present_state_kv, present_state_score, first_slot, last_slot, row_count);

  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T, P)                                                                   \
  template Status LaunchDSV4Compressor<T, P>(                                               \
      cudaStream_t, const DSV4CompressorParams&, const P*, const P*, const float*,          \
      const float*, const float*, const float*, const float*, const float*, const int64_t*, \
      T*, int64_t*, int64_t*, int64_t*, float*, float*);

#define INSTANTIATE_ALL_P(T) \
  INSTANTIATE(T, float)      \
  INSTANTIATE(T, MLFloat16)  \
  INSTANTIATE(T, BFloat16)

INSTANTIATE_ALL_P(float)
INSTANTIATE_ALL_P(MLFloat16)
INSTANTIATE_ALL_P(BFloat16)

#undef INSTANTIATE_ALL_P
#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
