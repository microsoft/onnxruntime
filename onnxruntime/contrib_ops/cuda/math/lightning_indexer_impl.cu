// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/lightning_indexer_impl.h"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>

#include "contrib_ops/cuda/math/dsv4_common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

constexpr int kThreads = kLightningIndexerThreads;
constexpr int kWarps = kThreads / 32;

// Order-preserving map from float to uint32 so the selection can be done with integer radix
// passes. Scores are finite here, so the only subtlety is that -0 sorts below +0.
__device__ __forceinline__ uint32_t FloatToKey(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive rank of this thread among the flagged threads of the block, plus the block total.
// Every thread must call it: it synchronises internally, and `s_warp` is reusable afterwards.
__device__ __forceinline__ void BlockRank(bool flag, int lane, int warp, int* s_warp,
                                          int* rank, int* total) {
  const unsigned mask = __ballot_sync(0xffffffffu, flag);
  const int within = __popc(mask & ((1u << lane) - 1u));
  if (lane == 0) s_warp[warp] = __popc(mask);
  __syncthreads();
  int before = 0;
  int sum = 0;
  for (int w = 0; w < kWarps; ++w) {
    if (w < warp) before += s_warp[w];
    sum += s_warp[w];
  }
  *rank = before + within;
  *total = sum;
  __syncthreads();
}

// Fold this step's compressed rows into the dense indexer cache.
//
// The cache is not paged: the scoring GEMM reads all of it every step, so there is nothing to
// gain from indirection. A float copy goes out alongside the T one because the GEMM wants both
// operands in the same precision and the query side is built in float.
template <typename CudaT>
__global__ void LightningIndexerCacheKernel(const LightningIndexerParams p,
                                            const CudaT* __restrict__ rows,
                                            const int64_t* __restrict__ first_slot,
                                            const int64_t* __restrict__ last_slot,
                                            const CudaT* __restrict__ past_cache,
                                            CudaT* __restrict__ present_cache,
                                            float* __restrict__ cache_f) {
  const int b = blockIdx.y;
  const int64_t first = first_slot[b];
  const int64_t last = last_slot[b];
  const int64_t n = static_cast<int64_t>(p.capacity) * p.head_dim;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const int64_t c = i / p.head_dim;
    const int64_t d = i - c * p.head_dim;
    const int64_t sel = c - first;
    int64_t cl = sel < 0 ? 0 : sel;
    if (cl > p.num_rows - 1) cl = p.num_rows - 1;
    const bool take = sel >= 0 && c <= last;
    const int64_t dst = (static_cast<int64_t>(b) * p.capacity + c) * p.head_dim + d;
    const CudaT v = take ? rows[(static_cast<int64_t>(b) * p.num_rows + cl) * p.head_dim + d]
                         : past_cache[dst];
    present_cache[dst] = v;
    cache_f[dst] = DSV4Conv<CudaT>::ToFloat(v);
  }
}

// Rotate the query rows and put them through the same simulated FP4 grid as the cache.
//
// One block per (batch, token, head): a decode step has only one token per sequence, so
// splitting by head is what keeps the launch from collapsing onto a single SM.
template <typename CudaT>
__global__ void LightningIndexerQueryKernel(const LightningIndexerParams p,
                                            const CudaT* __restrict__ query,
                                            const float* __restrict__ cos_table,
                                            const float* __restrict__ sin_table,
                                            float* __restrict__ q_out) {
  extern __shared__ float smem[];
  float* s_row = smem;
  float* s_rope = s_row + p.head_dim;
  float* s_scale = s_rope + p.rope_head_dim;

  const int h = blockIdx.x;
  const int s = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int d = p.head_dim;
  const int64_t token = static_cast<int64_t>(b) * p.seq_len + s;
  const int64_t row = token * p.num_heads + h;

  const CudaT* src = query + row * d;
  for (int t = tid; t < d; t += kThreads) s_row[t] = DSV4Conv<CudaT>::ToFloat(src[t]);
  __syncthreads();

  const int rd = p.rope_head_dim;
  if (rd > 0) {
    const float* cos_row = cos_table + token * rd;
    const float* sin_row = sin_table + token * rd;
    for (int t = tid; t < rd; t += kThreads) s_rope[t] = s_row[p.nope_dim + t];
    __syncthreads();
    for (int t = tid; t < rd; t += kThreads) {
      // The tables are already interleaved, so the rotation is the signed swap of each pair.
      const float rot = (t & 1) ? s_rope[t - 1] : -s_rope[t + 1];
      s_row[p.nope_dim + t] = s_rope[t] * cos_row[t] + rot * sin_row[t];
    }
    __syncthreads();
  }

  if (p.rotate_fp4) DSV4RotateFp4<CudaT>(s_row, s_scale, d, tid, kThreads);

  float* dst = q_out + row * d;
  for (int t = tid; t < d; t += kThreads) dst[t] = s_row[t];
}

// Fold the per-head scores into one number per cache row and key it for the selection.
template <typename CudaT>
__global__ void LightningIndexerReduceKernel(const LightningIndexerParams p,
                                             const float* __restrict__ score,
                                             const CudaT* __restrict__ weights,
                                             uint32_t* __restrict__ keys) {
  extern __shared__ float s_w[];
  const int s = blockIdx.y;
  const int b = blockIdx.z;
  const int64_t token = static_cast<int64_t>(b) * p.seq_len + s;

  for (int h = threadIdx.x; h < p.num_heads; h += kThreads) {
    s_w[h] = DSV4Conv<CudaT>::ToFloat(weights[token * p.num_heads + h]) * p.scale;
  }
  __syncthreads();

  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= p.capacity) return;
  const float* e = score + token * p.num_heads * p.capacity + c;
  float acc = 0.0f;
  for (int h = 0; h < p.num_heads; ++h) {
    acc += fmaxf(e[static_cast<int64_t>(h) * p.capacity], 0.0f) * s_w[h];
  }
  keys[token * p.capacity + c] = FloatToKey(acc);
}

// Keep the `topk` best-scoring rows a query is allowed to see.
//
// Attention over the selected rows does not depend on their order, so this only has to produce
// the right *set*: a radix select finds the k-th largest key, and a second pass emits the rows
// above it in ascending order, which is both deterministic and friendlier to the gather that
// follows. When a query can see no more rows than `topk`, the whole selection is its visible
// prefix and the search is skipped.
__global__ void LightningIndexerSelectKernel(const LightningIndexerParams p,
                                             const uint32_t* __restrict__ keys,
                                             const int64_t* __restrict__ past_lens,
                                             int64_t* __restrict__ selection) {
  __shared__ int s_hist[256];
  __shared__ int s_warp[kWarps];
  __shared__ int s_digit;
  __shared__ int s_above;

  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t token = static_cast<int64_t>(b) * p.seq_len + s;
  int64_t* out = selection + token * p.topk;

  int64_t visible = (past_lens[b] + s + 1) / p.ratio;
  if (visible > p.capacity) visible = p.capacity;
  const int n_vis = static_cast<int>(visible);

  if (n_vis <= p.topk) {
    for (int i = tid; i < p.topk; i += kThreads) {
      out[i] = i < n_vis ? static_cast<int64_t>(i) + p.max_seq_len : -1;
    }
    return;
  }

  const uint32_t* key = keys + token * p.capacity;

  // Narrow one byte at a time onto the k-th largest key.
  uint32_t prefix = 0;
  uint32_t mask = 0;
  int remaining = p.topk;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = tid; i < 256; i += kThreads) s_hist[i] = 0;
    __syncthreads();
    for (int i = tid; i < n_vis; i += kThreads) {
      const uint32_t u = key[i];
      if ((u & mask) == prefix) atomicAdd(&s_hist[(u >> shift) & 0xffu], 1);
    }
    __syncthreads();
    if (tid == 0) {
      int above = 0;
      int digit = 0;
      for (int dg = 255; dg >= 0; --dg) {
        if (above + s_hist[dg] >= remaining) {
          digit = dg;
          break;
        }
        above += s_hist[dg];
      }
      s_digit = digit;
      s_above = above;
    }
    __syncthreads();
    prefix |= static_cast<uint32_t>(s_digit) << shift;
    mask |= 0xffu << shift;
    remaining -= s_above;
  }

  // `remaining` is how many of the keys equal to the threshold still have to be taken.
  int written = 0;
  int seen_eq = 0;
  for (int start = 0; start < n_vis; start += kThreads) {
    const int i = start + tid;
    const bool live = i < n_vis;
    const uint32_t u = live ? key[i] : 0u;
    const bool eq = live && u == prefix;

    int eq_rank, eq_count;
    BlockRank(eq, lane, warp, s_warp, &eq_rank, &eq_count);
    const bool take = (live && u > prefix) || (eq && seen_eq + eq_rank < remaining);
    int take_rank, take_count;
    BlockRank(take, lane, warp, s_warp, &take_rank, &take_count);

    if (take) out[written + take_rank] = static_cast<int64_t>(i) + p.max_seq_len;
    written += take_count;
    seen_eq += eq_count;
  }
  for (int i = written + tid; i < p.topk; i += kThreads) out[i] = -1;
}

}  // namespace

template <typename T>
Status LaunchLightningIndexer(cudaStream_t stream, cublasHandle_t cublas,
                              const cudaDeviceProp& prop, bool use_tf32,
                              const LightningIndexerParams& p,
                              const T* query, const float* cos_table, const float* sin_table,
                              const T* rows, const int64_t* first_slot, const int64_t* last_slot,
                              const T* past_cache, const T* weights, const int64_t* past_lens,
                              int64_t* selection, T* present_cache,
                              float* query_scratch, float* cache_scratch, float* score_scratch,
                              uint32_t* key_scratch) {
  using CudaT = typename ::onnxruntime::cuda::ToCudaType<T>::MappedType;

  const int64_t cache_elems = LightningIndexerCacheElems(p) / p.batch;
  const int cache_blocks = static_cast<int>(
      std::min<int64_t>(1024, (cache_elems + kThreads - 1) / kThreads));
  LightningIndexerCacheKernel<CudaT><<<dim3(cache_blocks, p.batch), kThreads, 0, stream>>>(
      p, reinterpret_cast<const CudaT*>(rows), first_slot, last_slot,
      reinterpret_cast<const CudaT*>(past_cache), reinterpret_cast<CudaT*>(present_cache),
      cache_scratch);

  const size_t q_shared = LightningIndexerQuerySharedFloats(p) * sizeof(float);
  if (q_shared > 48u * 1024u) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "LightningIndexer needs ", q_shared,
                           " bytes of shared memory per query row, which is over the 48 KiB "
                           "limit.");
  }
  LightningIndexerQueryKernel<CudaT>
      <<<dim3(p.num_heads, p.seq_len, p.batch), kThreads, q_shared, stream>>>(
          p, reinterpret_cast<const CudaT*>(query), cos_table, sin_table, query_scratch);

  // score[b, s, h, c] = sum_d q[b, s, h, d] * cache[b, c, d]. Both operands sit on the FP4
  // grid with a power-of-two scale, so every product is exact even under TF32.
  const float one = 1.0f;
  const float zero = 0.0f;
  const int rows_per_batch = p.seq_len * p.num_heads;
  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N, p.capacity, rows_per_batch, p.head_dim, &one,
      cache_scratch, p.head_dim, static_cast<int64_t>(p.capacity) * p.head_dim,
      query_scratch, p.head_dim, static_cast<int64_t>(rows_per_batch) * p.head_dim, &zero,
      score_scratch, p.capacity, static_cast<int64_t>(rows_per_batch) * p.capacity,
      p.batch, prop, use_tf32));

  const int score_blocks = (p.capacity + kThreads - 1) / kThreads;
  LightningIndexerReduceKernel<CudaT>
      <<<dim3(score_blocks, p.seq_len, p.batch), kThreads, p.num_heads * sizeof(float),
         stream>>>(p, score_scratch, reinterpret_cast<const CudaT*>(weights), key_scratch);

  LightningIndexerSelectKernel<<<dim3(p.seq_len, p.batch), kThreads, 0, stream>>>(
      p, key_scratch, past_lens, selection);

  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                          \
  template Status LaunchLightningIndexer<T>(                                                    \
      cudaStream_t, cublasHandle_t, const cudaDeviceProp&, bool, const LightningIndexerParams&, \
      const T*, const float*, const float*, const T*, const int64_t*, const int64_t*,           \
      const T*, const T*, const int64_t*, int64_t*, T*, float*, float*, float*, uint32_t*);

INSTANTIATE(float)
INSTANTIATE(MLFloat16)
INSTANTIATE(BFloat16)

#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
