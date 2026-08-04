// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/dsv4_compressor_impl.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

constexpr int kThreads = kDSV4CompressorThreads;
constexpr int kWarps = kThreads / 32;
constexpr float kNegInf = -1e30f;
constexpr float kLog2 = 0.69314718f;  // the graph divides by float(math.log(2.0))
constexpr float kFp8Max = 448.0f;
constexpr float kFp4Max = 6.0f;

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

// A power-of-two scale whose exponent is `ceil(log2(amax / limit))`, matching the graph's
// Log/Div/Ceil/Pow chain including its clamp on the way into the logarithm.
__device__ __forceinline__ float BlockScale(float amax, float limit, float floor_value) {
  float r = amax / limit;
  if (r < floor_value) r = floor_value;
  return amax > 0.0f ? exp2f(ceilf(logf(r) / kLog2)) : 1.0f;
}

// Round onto the FP8-E4M3FN grid. `v` is already clipped to the finite range, so this only has
// to snap to the 3-bit mantissa of the containing binade, with 2^-6 as the subnormal floor.
__device__ __forceinline__ float RoundE4M3(float v) {
  const float a = fabsf(v);
  if (!(a > 0.0f)) return v;
  int e;
  frexpf(a, &e);
  e -= 1;
  if (e < -6) e = -6;
  const float step = ldexpf(1.0f, e - 3);
  return copysignf(rintf(a / step) * step, v);
}

// One candidate row's worth of pooling, one thread per channel.
//
// grid is (channel blocks, J, B).  The softmax runs over the pooling window independently for
// every channel, so the channels never have to talk to each other and the row can be spread
// over as many blocks as it takes to fill the device -- which matters, because a decode step
// only produces two rows per sequence.
template <typename CudaT>
__global__ void DSV4CompressorPoolKernel(const DSV4CompressorParams p,
                                         const float* __restrict__ kv,
                                         const float* __restrict__ score,
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
                     : (cur_row + cl - p.span) * p.feat + fo;
    return fo;
  };

  float m = -FLT_MAX;
  for (int s = 0; s < p.span; ++s) {
    int64_t off;
    int k;
    bool valid, use_past;
    const int fo = gather(s, &off, &k, &valid, &use_past);
    const float w = valid ? (use_past ? past_score[off] : score[off]) + ape[k * p.feat + fo]
                          : kNegInf;
    m = fmaxf(m, w);
  }

  float denom = 0.0f;
  for (int s = 0; s < p.span; ++s) {
    int64_t off;
    int k;
    bool valid, use_past;
    const int fo = gather(s, &off, &k, &valid, &use_past);
    const float w = valid ? (use_past ? past_score[off] : score[off]) + ape[k * p.feat + fo]
                          : kNegInf;
    denom += expf(w - m);
  }

  float acc = 0.0f;
  for (int s = 0; s < p.span; ++s) {
    int64_t off;
    int k;
    bool valid, use_past;
    const int fo = gather(s, &off, &k, &valid, &use_past);
    const float w = valid ? (use_past ? past_score[off] : score[off]) + ape[k * p.feat + fo]
                          : kNegInf;
    acc += (use_past ? past_kv[off] : kv[off]) * (expf(w - m) / denom);
  }

  rows[(static_cast<int64_t>(b) * p.num_rows + j) * p.head_dim + ch] =
      Conv<CudaT>::FromFloat(acc);
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
    const float v = Conv<CudaT>::ToFloat(row[c]);
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
      s_scale[tid] = BlockScale(amax, kFp8Max, 1e-30f);
    }
    __syncthreads();
    for (int c = tid; c < p.nope_dim; c += kThreads) {
      const float scale = s_scale[c >> 6];
      const float q = fminf(fmaxf(s_row[c] / scale, -kFp8Max), kFp8Max);
      s_row[c] = RoundE4M3(q) * scale;
    }
    __syncthreads();
  }

  if (p.rotate_fp4) {
    // Walsh-Hadamard butterfly: the same orthogonal rotation the graph writes as a dense
    // MatMul against a Sylvester matrix, minus the matrix.
    for (int len = 1; len < d; len <<= 1) {
      for (int i = tid; i < d / 2; i += kThreads) {
        const int lo = (i / len) * (len << 1) + (i % len);
        const float a = s_row[lo];
        const float c = s_row[lo + len];
        s_row[lo] = a + c;
        s_row[lo + len] = a - c;
      }
      __syncthreads();
    }
    const float hscale = rsqrtf(static_cast<float>(d));
    for (int c = tid; c < d; c += kThreads) {
      s_row[c] = Conv<CudaT>::Round(s_row[c] * hscale);
    }
    __syncthreads();

    const int blocks = d / 32;
    if (tid < blocks) {
      float amax = 0.0f;
      for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(s_row[tid * 32 + i]));
      s_scale[tid] = BlockScale(amax, kFp4Max, 1e-38f);
    }
    __syncthreads();
    for (int c = tid; c < d; c += kThreads) {
      const float scale = s_scale[c >> 5];
      const float v = fminf(fmaxf(s_row[c] / scale, -kFp4Max), kFp4Max);
      const float u = fabsf(v);
      // E2M1 grid {0,.5,1,1.5,2,3,4,6}: the step doubles at 2 and 4, and ties go toward zero.
      const float step = u < 2.0f ? 0.5f : (u < 4.0f ? 1.0f : 2.0f);
      const float sign = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
      s_row[c] = sign * step * ceilf(u / step - 0.5f) * scale;
    }
    __syncthreads();
  }

  for (int c = tid; c < d; c += kThreads) row[c] = Conv<CudaT>::FromFloat(s_row[c]);
}

// Roll the raw projections forward by one step and publish the slot bookkeeping.
__global__ void DSV4CompressorStateKernel(const DSV4CompressorParams p,
                                          const float* __restrict__ kv,
                                          const float* __restrict__ score,
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
    int64_t off;
    const float* from_kv;
    const float* from_sc;
    if (src < p.span) {
      off = (static_cast<int64_t>(b) * p.span + src) * p.feat + f;
      from_kv = past_kv;
      from_sc = past_score;
    } else {
      off = (static_cast<int64_t>(b) * p.seq_len + src - p.span) * p.feat + f;
      from_kv = kv;
      from_sc = score;
    }
    const int64_t dst = (static_cast<int64_t>(b) * p.span + r) * p.feat + f;
    out_kv[dst] = from_kv[off];
    out_score[dst] = from_sc[off];
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int64_t past = past_lens[b];
    first_slot[b] = past / p.ratio;
    last_slot[b] = (past + p.seq_len - 1) / p.ratio;
    if (b == 0) *row_count = (p.seq_len - 1) / p.ratio + 2;
  }
}

}  // namespace

template <typename T>
Status LaunchDSV4Compressor(cudaStream_t stream, const DSV4CompressorParams& params,
                            const float* kv, const float* score,
                            const float* past_state_kv, const float* past_state_score,
                            const float* ape, const float* norm_weight,
                            const float* cos_table, const float* sin_table,
                            const int64_t* past_lens,
                            T* rows, int64_t* first_slot, int64_t* last_slot, int64_t* row_count,
                            float* present_state_kv, float* present_state_score) {
  using CudaT = typename ::onnxruntime::cuda::ToCudaType<T>::MappedType;
  auto* out = reinterpret_cast<CudaT*>(rows);

  const int channel_blocks = (params.head_dim + kThreads - 1) / kThreads;
  DSV4CompressorPoolKernel<CudaT><<<dim3(channel_blocks, params.num_rows, params.batch), kThreads,
                                    0, stream>>>(
      params, kv, score, past_state_kv, past_state_score, ape, past_lens, out);

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
  DSV4CompressorStateKernel<<<dim3(state_blocks, params.batch), kThreads, 0, stream>>>(
      params, kv, score, past_state_kv, past_state_score, past_lens,
      present_state_kv, present_state_score, first_slot, last_slot, row_count);

  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                          \
  template Status LaunchDSV4Compressor<T>(                                                      \
      cudaStream_t, const DSV4CompressorParams&, const float*, const float*, const float*,      \
      const float*, const float*, const float*, const float*, const float*, const int64_t*, T*, \
      int64_t*, int64_t*, int64_t*, float*, float*);

INSTANTIATE(float)
INSTANTIATE(MLFloat16)
INSTANTIATE(BFloat16)

#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
