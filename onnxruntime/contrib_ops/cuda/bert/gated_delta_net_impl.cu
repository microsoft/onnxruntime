// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GatedDeltaNet CUDA kernels.
//
// Two engines, chosen by gated_delta_net_plan.h:
//
//  * chunked  -- tensor-core chunked scan for prefill. One CTA owns one
//                (sequence, v-head, v-block) and walks the sequence in BT=64 token chunks,
//                keeping the recurrent state in shared memory for the whole walk. One
//                launch, no per-chunk state materialization.
//  * recurrent -- scalar per-token recurrence for decode / MTP verify, and the only engine
//                that can emit compact per-token transition factors.
//
// Recurrence (gated_delta), per v-head, with S the [K x V] state:
//     S_t = exp(g_t) S_{t-1} + k_t (beta_t (v_t - exp(g_t) S_{t-1}^T k_t))^T
//     o_t = scale * S_t^T q_t
//
// Chunked form over a chunk of BT tokens with gc the within-chunk cumulative log-decay:
//     M[t,s]  = beta_t (k_t . k_s) exp(gc_t - gc_s)             strictly lower
//     U       = (I + M)^-1 (Ubar - Wbar S0),  Ubar = beta v,  Wbar[t] = beta_t exp(gc_t) k_t
//     P[t,s]  = (q_t . k_s) exp(gc_t - gc_s)                    inclusive lower
//     o       = scale (P U + Qg S0),          Qg[t] = q_t exp(gc_t)
//     S1      = exp(gc_BT) S0 + Kd^T U,       Kd[t] = k_t exp(gc_BT - gc_t)
//
// Two properties of that form matter for speed and are relied on below:
//
//  1. W is only ever used as `W S0`, and W = (I+M)^-1 Wbar, so
//     U = (I+M)^-1 Ubar - W S0 = (I+M)^-1 (Ubar - Wbar S0). The 64x128 triangular solve
//     disappears, and since Wbar is only a row scaling of k, Wbar is never materialized
//     either -- the scaling folds into the epilogue of (k S0).
//  2. (I+M)^-1 has an exact closed form. With Dinv the block diagonal of the four 16x16
//     inverses and N = Dinv M (its own 16x16 diagonal blocks are exactly zero), N is
//     strictly block lower over four block levels so N^4 = 0 and
//     (I+M)^-1 = (I - N + N^2 - N^3) Dinv exactly. Four full 64x64x64 GEMMs replace eight
//     tiny serial ones.
//
// k * exp(-gc) is never formed; the decay ratio is applied to the [BT x BT] gram matrices,
// which keeps the exponent bounded within a chunk.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <mutex>
#include <type_traits>

#include "contrib_ops/cuda/bert/gated_delta_net_impl.h"
#include "contrib_ops/cuda/bert/gated_delta_net_mma.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

// Include env_var_utils.h after cuda_common.h: the latter transitively pulls in provider_api.h,
// which defines SHARED_PROVIDER and suppresses env_var_utils.h's own logging.h include.
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace gated_delta_net {

namespace {

constexpr int kDVB = 64;  // v columns owned by one CTA in the chunked engine

template <typename T>
__device__ __forceinline__ int CaptureCount(const VariantPack<T>& pack, const KernelParams& p,
                                            int b, int seq_len) {
  if (pack.capture_count == nullptr || p.state_update_capacity <= 0) return 0;
  return max(0, min(min(pack.capture_count[b], p.state_update_capacity), seq_len));
}

template <typename T>
__device__ __forceinline__ float* StateUpdateDecay(
    const VariantPack<T>& pack, const KernelParams& p, int b, int t, int hv, int d_k, int d_v) {
  if (pack.state_update == nullptr) return nullptr;
  const int64_t row_width = static_cast<int64_t>(p.state_update_capacity) *
                            (p.num_heads_v + p.num_heads_k * d_k + p.num_heads_v * d_v);
  return pack.state_update + static_cast<int64_t>(b) * row_width +
         static_cast<int64_t>(t) * p.num_heads_v + hv;
}

template <typename T>
__device__ __forceinline__ float* StateUpdateKey(
    const VariantPack<T>& pack, const KernelParams& p, int b, int t, int hk, int d_k, int d_v) {
  if (pack.state_update == nullptr) return nullptr;
  const int64_t row_width = static_cast<int64_t>(p.state_update_capacity) *
                            (p.num_heads_v + p.num_heads_k * d_k + p.num_heads_v * d_v);
  const int64_t key_offset = static_cast<int64_t>(p.state_update_capacity) * p.num_heads_v;
  return pack.state_update + static_cast<int64_t>(b) * row_width + key_offset +
         (static_cast<int64_t>(t) * p.num_heads_k + hk) * d_k;
}

template <typename T>
__device__ __forceinline__ float* StateUpdateDelta(
    const VariantPack<T>& pack, const KernelParams& p, int b, int t, int hv, int d_k, int d_v) {
  if (pack.state_update == nullptr) return nullptr;
  const int64_t row_width = static_cast<int64_t>(p.state_update_capacity) *
                            (p.num_heads_v + p.num_heads_k * d_k + p.num_heads_v * d_v);
  const int64_t delta_offset = static_cast<int64_t>(p.state_update_capacity) *
                               (p.num_heads_v + p.num_heads_k * d_k);
  return pack.state_update + static_cast<int64_t>(b) * row_width + delta_offset +
         (static_cast<int64_t>(t) * p.num_heads_v + hv) * d_v;
}

template <typename T>
__device__ __forceinline__ bool IsRecurrentTailRow(const VariantPack<T>& pack,
                                                   const KernelParams& p, int b, int seq_len) {
  return (p.state_update_tail_pass && CaptureCount(pack, p, b, seq_len) > 0) ||
         (p.short_row_tail_pass && seq_len < kChunkedMinTokens);
}

// Padded leading dimensions. A row stride of DK halves (256 B) makes every mma fragment
// load hit the same shared-memory bank across the 8 rows a warp reads, costing an 8-way
// conflict per load; +8 halves (+4 floats) staggers them.
template <int DK, int BT>
struct Ld {
  static constexpr int kKh = DK + 8;
  static constexpr int kVh = kDVB + 8;
  static constexpr int kMh = BT + 8;
  static constexpr int kVf = (kDVB > BT ? kDVB : BT) + 4;
};

template <typename T, int DK, int BT>
struct ChunkedSmem {
  T* k_h;       // [BT][LdKh]  k, scaled in place to Kd late in the chunk
  T* q_h;       // [BT][LdKh]  q, scaled in place to Qg late in the chunk
  T* v_h;       // [BT][LdVh]
  T* u_h;       // [BT][LdVh]  R, then U
  T* m_h;       // [BT][LdMh]  M, then P
  T* s_h;       // [DK][LdVh]  16-bit operand copy of the state
  T* db_h;      // [BT][LdMh]  block-diagonal inverse of (I+M)
  T* nb_h;      // [BT][LdMh]  N = Dinv M with the diagonal blocks zeroed
  T* ti_h;      // [BT][LdMh]  Neumann iterate, finally (I+M)^-1
  float* s_f;   // [DK][LdVf]  fp32 state
  float* t_f;   // [BT][LdVf]  gemm scratch, also used as [BT][BT]
  float* gc;    // [BT]
  float* beta;  // [BT]
};

template <typename T, int DK, int BT>
__device__ __forceinline__ ChunkedSmem<T, DK, BT> CarveChunked(char* raw) {
  using L = Ld<DK, BT>;
  ChunkedSmem<T, DK, BT> s;
  float* f = reinterpret_cast<float*>(raw);
  s.s_f = f;
  f += DK * L::kVf;
  s.t_f = f;
  f += BT * L::kVf;
  s.gc = f;
  f += BT;
  s.beta = f;
  f += BT;
  T* h = reinterpret_cast<T*>(f);
  s.k_h = h;
  h += BT * L::kKh;
  s.q_h = h;
  h += BT * L::kKh;
  s.u_h = h;
  h += BT * L::kVh;
  s.m_h = h;
  h += BT * L::kMh;
  s.s_h = h;
  h += DK * L::kVh;
  s.ti_h = h;
  h += BT * L::kMh;
  // v_h, db_h, and nb_h are contiguous so that once BuildTriInverse and R complete,
  // their combined region (BT * (kVh + 2 * kMh) * 2 B >= BT * kVf * 4 B) can hold u_f.
  s.v_h = h;
  h += BT * L::kVh;
  s.db_h = h;
  h += BT * L::kMh;
  s.nb_h = h;
  h += BT * L::kMh;
  return s;
}

template <typename T, typename MT, bool kExactFp32StateAndU, int DK, int DV, int BT>
__device__ __forceinline__ void RunChunkedStep(
    char* smem_raw, const VariantPack<T>& pack, const KernelParams& p, int64_t tok_base,
    int chunk0, int len, int hv, int v0, int64_t v_stride, bool needs_retrieval, int tid,
    int warp_id, int lane) {
  using L = Ld<DK, BT>;
  const ChunkedSmem<MT, DK, BT> s = CarveChunked<MT, DK, BT>(smem_raw);
  T* k_in = reinterpret_cast<T*>(s.k_h);
  T* q_in = reinterpret_cast<T*>(s.q_h);
  T* v_in = reinterpret_cast<T*>(s.v_h);

  if constexpr (!kExactFp32StateAndU) {
    for (int idx = tid; idx < DK * kDVB; idx += kChunkedThreads) {
      const int r = idx / kDVB, c = idx % kDVB;
      s.s_h[r * L::kVh + c] = FloatToElem<MT>(s.s_f[r * L::kVf + c]);
    }
  }
  if constexpr (!std::is_same_v<T, MT>) {
    for (int idx = tid; idx < BT * kDVB; idx += kChunkedThreads) {
      const int t = idx / kDVB, d = idx % kDVB;
      s.v_h[t * L::kVh + d] = FloatToElem<MT>(ElemToFloat(v_in[t * L::kVh + d]));
    }
    if (!p.qk_l2_norm) {
      for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
        const int t = idx / DK, d = idx % DK;
        s.k_h[t * L::kKh + d] = FloatToElem<MT>(ElemToFloat(k_in[t * L::kKh + d]));
        s.q_h[t * L::kKh + d] = FloatToElem<MT>(ElemToFloat(q_in[t * L::kKh + d]));
      }
    }
  }

  if (p.qk_l2_norm) {
    // One warp per token row; DK=128 so each lane holds four elements.
    for (int t = warp_id; t < BT; t += kChunkedWarps) {
      float sq = 0.0f, sk = 0.0f;
      for (int d = lane; d < DK; d += 32) {
        const float qv = ElemToFloat(q_in[t * L::kKh + d]);
        const float kv = ElemToFloat(k_in[t * L::kKh + d]);
        sq += qv * qv;
        sk += kv * kv;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        sq += __shfl_xor_sync(0xffffffff, sq, off);
        sk += __shfl_xor_sync(0xffffffff, sk, off);
      }
      const float rq = rsqrtf(sq + 1e-12f), rk = rsqrtf(sk + 1e-12f);
      for (int d = lane; d < DK; d += 32) {
        const float qv = ElemToFloat(q_in[t * L::kKh + d]);
        const float kv = ElemToFloat(k_in[t * L::kKh + d]);
        s.q_h[t * L::kKh + d] = FloatToElem<MT>(qv * rq);
        s.k_h[t * L::kKh + d] = FloatToElem<MT>(kv * rk);
      }
    }
  }
  __syncthreads();

  InclusiveScanBT<BT>(s.gc, tid);
  __syncthreads();
  const float g_total = s.gc[BT - 1];

  SmemGemm<BT, BT, DK, false, true, false>(s.t_f, L::kVf, s.k_h, L::kKh, s.k_h, L::kKh, warp_id,
                                           lane);
  __syncthreads();
  for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
    const int t = idx / BT, sc = idx % BT;
    const float m = (sc < t && needs_retrieval)
                        ? s.beta[t] * s.t_f[t * L::kVf + sc] * __expf(s.gc[t] - s.gc[sc])
                        : 0.0f;
    s.m_h[t * L::kMh + sc] = FloatToElem<MT>(m);
  }
  __syncthreads();

  BuildTriInverse<BT>(s.m_h, L::kMh, s.db_h, s.nb_h, s.ti_h, L::kMh, s.t_f, L::kVf, warp_id,
                      lane, tid);

  if constexpr (kExactFp32StateAndU) {
    SmemGemmFp32State<BT, kDVB, DK, false>(s.t_f, L::kVf, s.k_h, L::kKh, s.s_f, L::kVf, s.s_h,
                                           L::kVh, warp_id, lane, tid);
    __syncthreads();
    for (int idx = tid; idx < BT * kDVB; idx += kChunkedThreads) {
      const int t = idx / kDVB, c = idx % kDVB;
      const float vv = ElemToFloat(s.v_h[t * L::kVh + c]);
      const float r = needs_retrieval ? vv - __expf(s.gc[t]) * s.t_f[t * L::kVf + c] : vv;
      s.t_f[t * L::kVf + c] = s.beta[t] * r;
    }
    __syncthreads();
    float* u_f = reinterpret_cast<float*>(s.v_h);
    SmemGemmFp32State<BT, kDVB, BT, false>(u_f, L::kVf, s.ti_h, L::kMh, s.t_f, L::kVf, s.u_h,
                                           L::kVh, warp_id, lane, tid);
    __syncthreads();

    SmemGemm<BT, BT, DK, false, true, false>(s.t_f, L::kVf, s.q_h, L::kKh, s.k_h, L::kKh, warp_id,
                                             lane);
    __syncthreads();
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      const int t = idx / BT, sc = idx % BT;
      s.m_h[t * L::kMh + sc] = FloatToElem<MT>(
          (sc <= t && t < len) ? s.t_f[t * L::kVf + sc] * __expf(s.gc[t] - s.gc[sc]) : 0.0f);
    }
    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      const int t = idx / DK, d = idx % DK;
      s.k_h[t * L::kKh + d] = ScaleElem(s.k_h[t * L::kKh + d], __expf(g_total - s.gc[t]));
      s.q_h[t * L::kKh + d] = ScaleElem(s.q_h[t * L::kKh + d], __expf(s.gc[t]));
    }
    __syncthreads();

    SmemGemmFp32State<BT, kDVB, BT, false>(s.t_f, L::kVf, s.m_h, L::kMh, u_f, L::kVf, s.u_h,
                                           L::kVh, warp_id, lane, tid);
    __syncthreads();
    SmemGemmFp32State<BT, kDVB, DK, true>(s.t_f, L::kVf, s.q_h, L::kKh, s.s_f, L::kVf, s.s_h,
                                          L::kVh, warp_id, lane, tid);
    __syncthreads();
    for (int idx = tid; idx < BT * kDVB; idx += kChunkedThreads) {
      const int t = idx / kDVB, c = idx % kDVB;
      if (t < len) {
        const int64_t tok = tok_base + chunk0 + t;
        pack.output[tok * v_stride + hv * DV + v0 + c] =
            FloatToElem<T>(p.scale * s.t_f[t * L::kVf + c]);
      }
    }

    const float decay_all = __expf(g_total);
    for (int idx = tid; idx < DK * kDVB; idx += kChunkedThreads) {
      s.s_f[(idx / kDVB) * L::kVf + idx % kDVB] *= decay_all;
    }
    __syncthreads();
    SmemGemmFp32State<DK, kDVB, BT, true, kChunkedWarps, MT, /*TA=*/true>(
        s.s_f, L::kVf, s.k_h, L::kKh, u_f, L::kVf, s.u_h, L::kVh, warp_id, lane, tid);
    __syncthreads();
  } else {
    SmemGemm<BT, kDVB, DK, false, false, false>(s.t_f, L::kVf, s.k_h, L::kKh, s.s_h, L::kVh,
                                                warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * kDVB; idx += kChunkedThreads) {
      const int t = idx / kDVB, c = idx % kDVB;
      const float vv = ElemToFloat(s.v_h[t * L::kVh + c]);
      const float r = needs_retrieval ? vv - __expf(s.gc[t]) * s.t_f[t * L::kVf + c] : vv;
      s.u_h[t * L::kVh + c] = FloatToElem<MT>(s.beta[t] * r);
    }
    __syncthreads();
    SmemGemm<BT, kDVB, BT, false, false, false>(s.t_f, L::kVf, s.ti_h, L::kMh, s.u_h, L::kVh,
                                                warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * kDVB; idx += kChunkedThreads) {
      const int t = idx / kDVB, c = idx % kDVB;
      s.u_h[t * L::kVh + c] = FloatToElem<MT>(s.t_f[t * L::kVf + c]);
    }
    __syncthreads();

    SmemGemm<BT, BT, DK, false, true, false>(s.t_f, L::kVf, s.q_h, L::kKh, s.k_h, L::kKh, warp_id,
                                             lane);
    __syncthreads();
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      const int t = idx / BT, sc = idx % BT;
      s.m_h[t * L::kMh + sc] = FloatToElem<MT>(
          (sc <= t && t < len) ? s.t_f[t * L::kVf + sc] * __expf(s.gc[t] - s.gc[sc]) : 0.0f);
    }
    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      const int t = idx / DK, d = idx % DK;
      s.k_h[t * L::kKh + d] = ScaleElem(s.k_h[t * L::kKh + d], __expf(g_total - s.gc[t]));
      s.q_h[t * L::kKh + d] = ScaleElem(s.q_h[t * L::kKh + d], __expf(s.gc[t]));
    }
    __syncthreads();

    SmemGemm<BT, kDVB, BT, false, false, false>(s.t_f, L::kVf, s.m_h, L::kMh, s.u_h, L::kVh,
                                                warp_id, lane);
    __syncthreads();
    SmemGemm<BT, kDVB, DK, false, false, true>(s.t_f, L::kVf, s.q_h, L::kKh, s.s_h, L::kVh,
                                               warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * kDVB; idx += kChunkedThreads) {
      const int t = idx / kDVB, c = idx % kDVB;
      if (t < len) {
        const int64_t tok = tok_base + chunk0 + t;
        pack.output[tok * v_stride + hv * DV + v0 + c] =
            FloatToElem<T>(p.scale * s.t_f[t * L::kVf + c]);
      }
    }

    const float decay_all = __expf(g_total);
    for (int idx = tid; idx < DK * kDVB; idx += kChunkedThreads) {
      s.s_f[(idx / kDVB) * L::kVf + idx % kDVB] *= decay_all;
    }
    __syncthreads();
    SmemGemm<DK, kDVB, BT, true, false, true>(s.s_f, L::kVf, s.k_h, L::kKh, s.u_h, L::kVh,
                                              warp_id, lane);
    __syncthreads();
  }
}

template <typename T, int DK, int DV, int BT>
__device__ __noinline__ void RunChunkedStepExactBf16Fallback(
    char* smem_raw, const VariantPack<T>& pack, const KernelParams& p, int64_t tok_base,
    int chunk0, int len, int hv, int v0, int64_t v_stride, bool needs_retrieval, int tid,
    int warp_id, int lane) {
  RunChunkedStep<T, T, /*kExactFp32StateAndU=*/true, DK, DV, BT>(
      smem_raw, pack, p, tok_base, chunk0, len, hv, v0, v_stride, needs_retrieval, tid, warp_id,
      lane);
}

// ---------------------------------------------------------------------------
// Chunked engine
// ---------------------------------------------------------------------------
template <typename T, int DK, int DV, int BT, bool InputTypeGates,
          Bf16Precision kBf16Prec = Bf16Precision::kFp32>
__global__ __launch_bounds__(kChunkedThreads) void GatedDeltaNetChunkedKernel(
    VariantPack<T> pack, KernelParams p) {
  using L = Ld<DK, BT>;
  extern __shared__ char smem_raw[];
  const ChunkedSmem<T, DK, BT> s = CarveChunked<T, DK, BT>(smem_raw);

  const int b = blockIdx.x, hv = blockIdx.y, v0 = blockIdx.z * kDVB;
  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  const int hq = hv * p.num_heads_q / p.num_heads_v;
  const int hk = hv * p.num_heads_k / p.num_heads_v;

  int64_t tok_base, seq_len64;
  SequenceRange(pack.cu_seqlens, b, p.total_tokens, p.uniform_len, &tok_base, &seq_len64);
  const int seq_len = static_cast<int>(seq_len64);
  // Uniform across the CTA: seq_len depends only on blockIdx.x.
  if (IsRecurrentTailRow(pack, p, b, seq_len)) return;

  const int64_t q_stride = static_cast<int64_t>(p.num_heads_q) * DK;
  const int64_t k_stride = static_cast<int64_t>(p.num_heads_k) * DK;
  const int64_t v_stride = static_cast<int64_t>(p.num_heads_v) * DV;
  const int64_t st_off = (static_cast<int64_t>(b) * p.num_heads_v + hv) * DV * DK;

  const bool needs_decay = pack.decay != nullptr &&
                           (p.update_rule == UpdateRule::kGated ||
                            p.update_rule == UpdateRule::kGatedDelta);
  const bool needs_retrieval =
      p.update_rule == UpdateRule::kDelta || p.update_rule == UpdateRule::kGatedDelta;

  for (int idx = tid; idx < DK * kDVB; idx += kChunkedThreads) {
    const int r = idx / kDVB, c = idx % kDVB;
    s.s_f[r * L::kVf + c] =
        pack.initial_state != nullptr
            ? pack.initial_state[st_off + static_cast<int64_t>(v0 + c) * DK + r]
            : 0.0f;
  }
  __syncthreads();

  for (int chunk0 = 0; chunk0 < seq_len; chunk0 += BT) {
    const int len = min(BT, seq_len - chunk0);
    // Safe FP16 range check in registers during the single global->shared load pass:
    // fall back when conservative FP16 operand and intermediate-product bounds are exceeded.
    constexpr bool kCheckFp16 =
        std::is_same_v<T, __nv_bfloat16> && kBf16Prec == Bf16Precision::kFp16;
    bool thread_out_of_fp16 = kCheckFp16 && (fabsf(p.scale) > 16.0f);
    if constexpr (kCheckFp16) {
      int thread_s_max = 0, s_flags = 0;
      for (int idx = tid; idx < DK * kDVB; idx += kChunkedThreads) {
        const int r = idx / kDVB, c = idx % kDVB;
        const int abits = __float_as_int(fabsf(s.s_f[r * L::kVf + c]));
        thread_s_max = max(thread_s_max, abits);
        s_flags |= (abits == 0) ? 1 : ((abits < 0x33800000) ? 2 : 0);
      }
      if (thread_s_max > 0x43800000 || (thread_s_max != 0 && thread_s_max < 0x38800000) ||
          (s_flags & 2) != 0) {
        thread_out_of_fp16 = true;
      }
    }
    // Bound ||k||_2 <= 1, not just individual components, so retrieval cannot amplify the inverse.
    const int max_k_bits = p.qk_l2_norm      ? 0x46000000
                           : needs_retrieval ? __float_as_int(1.0f / sqrtf(static_cast<float>(DK)))
                                             : 0x41000000;
    const int max_q_bits = p.qk_l2_norm ? 0x46000000 : 0x41000000;
    int thread_q_max = 0, thread_k_max = 0;
    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      const int t = idx / DK, d = idx % DK;
      const int64_t tok = tok_base + chunk0 + t;
      const T kv = (t < len) ? pack.key[tok * k_stride + hk * DK + d] : FloatToElem<T>(0.f);
      const T qv = (t < len) ? pack.query[tok * q_stride + hq * DK + d] : FloatToElem<T>(0.f);
      s.k_h[t * L::kKh + d] = kv;
      s.q_h[t * L::kKh + d] = qv;
      if constexpr (kCheckFp16) {
        if (t < len) {
          thread_k_max = max(thread_k_max, __float_as_int(fabsf(ElemToFloat(kv))));
          thread_q_max = max(thread_q_max, __float_as_int(fabsf(ElemToFloat(qv))));
        }
      }
    }
    if constexpr (kCheckFp16) {
      if (thread_k_max > max_k_bits || thread_q_max > max_q_bits ||
          (!p.qk_l2_norm && ((thread_k_max != 0 && thread_k_max < 0x38800000) ||
                             (thread_q_max != 0 && thread_q_max < 0x38800000)))) {
        thread_out_of_fp16 = true;
      }
    }
    int thread_v_max = 0;
    for (int idx = tid; idx < BT * kDVB; idx += kChunkedThreads) {
      const int t = idx / kDVB, d = idx % kDVB;
      const int64_t tok = tok_base + chunk0 + t;
      const T vv = (t < len) ? pack.value[tok * v_stride + hv * DV + v0 + d] : FloatToElem<T>(0.f);
      s.v_h[t * L::kVh + d] = vv;
      if constexpr (kCheckFp16) {
        if (t < len) {
          thread_v_max = max(thread_v_max, __float_as_int(fabsf(ElemToFloat(vv))));
        }
      }
    }
    if constexpr (kCheckFp16) {
      if (thread_v_max > 0x43800000 || (thread_v_max != 0 && thread_v_max < 0x38800000)) {
        thread_out_of_fp16 = true;
      }
    }
    if (tid < BT) {
      const int64_t tok = tok_base + chunk0 + tid;
      const int64_t gi = tok * p.num_heads_v + hv;
      float g = 0.0f, be = 0.0f;
      if (tid < len) {
        g = needs_decay ? EffectiveDecay(LoadGate<T>(pack.decay, gi, InputTypeGates), pack.a_log, pack.dt_bias, hv,
                                         p.gate_activation)
                        : 0.0f;
        be = pack.beta != nullptr ? EffectiveBeta(LoadGate<T>(pack.beta, gi, InputTypeGates), p.beta_activation) : 1.0f;
      }
      s.gc[tid] = g;
      s.beta[tid] = be;
      if constexpr (kCheckFp16) {
        if (tid < len && (!(g <= 0.0f) || !(fabsf(be) <= 1.0f) || (needs_retrieval && be < 0.0f))) {
          thread_out_of_fp16 = true;
        }
      }
    }

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      if constexpr (kBf16Prec == Bf16Precision::kBf16) {
        __syncthreads();
        RunChunkedStep<T, T, /*kExactFp32StateAndU=*/false, DK, DV, BT>(
            smem_raw, pack, p, tok_base, chunk0, len, hv, v0, v_stride, needs_retrieval, tid,
            warp_id, lane);
      } else if constexpr (kBf16Prec == Bf16Precision::kFp16) {
        const bool out_of_fp16 = __syncthreads_or(thread_out_of_fp16) != 0;
        if (!out_of_fp16) {
          RunChunkedStep<T, __half, /*kExactFp32StateAndU=*/false, DK, DV, BT>(
              smem_raw, pack, p, tok_base, chunk0, len, hv, v0, v_stride, needs_retrieval, tid,
              warp_id, lane);
        } else {
          RunChunkedStepExactBf16Fallback<T, DK, DV, BT>(
              smem_raw, pack, p, tok_base, chunk0, len, hv, v0, v_stride, needs_retrieval, tid,
              warp_id, lane);
        }
      } else {
        __syncthreads();
        RunChunkedStep<T, T, /*kExactFp32StateAndU=*/true, DK, DV, BT>(
            smem_raw, pack, p, tok_base, chunk0, len, hv, v0, v_stride, needs_retrieval, tid,
            warp_id, lane);
      }
    } else {
      __syncthreads();
      RunChunkedStep<T, T, /*kExactFp32StateAndU=*/false, DK, DV, BT>(
          smem_raw, pack, p, tok_base, chunk0, len, hv, v0, v_stride, needs_retrieval, tid,
          warp_id, lane);
    }
  }

  if (pack.final_state != nullptr) {
    for (int idx = tid; idx < DK * kDVB; idx += kChunkedThreads) {
      const int r = idx / kDVB, c = idx % kDVB;
      pack.final_state[st_off + static_cast<int64_t>(v0 + c) * DK + r] = s.s_f[r * L::kVf + c];
    }
  }
}

// ---------------------------------------------------------------------------
// Warp-specialised decode engine.
//
// One warp owns one (sequence, v-head, v-column); its lanes span the K axis, so each lane
// keeps DK/32 state elements in registers. Three things fall out of that mapping:
//
//  * The state is V-major [.., V, K], so lanes reading consecutive k are fully coalesced.
//    The generic kernel below walks the same buffer with consecutive threads on v, which
//    strides every access by DK.
//  * Both reductions (S^T k and S^T q) are over K, so they are warp shuffles. The token
//    loop needs no __syncthreads() and no shared memory at all.
//  * The grid is (sequences, v-heads, V/warps) instead of (sequences, v-heads), which at
//    the Qwen3.8 geometry is 768 CTAs against 48.
// ---------------------------------------------------------------------------
template <int DK>
__device__ __forceinline__ float WarpReduceK(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
  return v;
}

template <typename T, int DK, int kWarps, bool InputTypeGates, bool PackedQkv, bool QwenS2 = false>
__global__ __launch_bounds__(32 * kWarps) void GatedDeltaNetDecodeWarpKernel(
    VariantPack<T> pack, KernelParams p, int d_v) {
  if constexpr (QwenS2) {
    // Host dispatch checks every field before replacing it with a compile-time constant.
    p.num_heads_q = p.num_heads_k = 16;
    p.num_heads_v = 48;
    p.gate_activation = GateActivation::kSoftplusDecay;
    p.beta_activation = BetaActivation::kSigmoid;
    p.update_rule = UpdateRule::kGatedDelta;
    p.qk_l2_norm = true;
    p.decay_per_key_dim_flag = 0;
    p.state_update_tail_pass = p.short_row_tail_pass = false;
    d_v = 128;
  }
  constexpr int kRegs = DK / 32;  // state elements per lane
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int b = blockIdx.x, hv = blockIdx.y;
  const int col = blockIdx.z * kWarps + warp;
  if (col >= d_v) return;

  const int hq = hv * p.num_heads_q / p.num_heads_v;
  const int hk = hv * p.num_heads_k / p.num_heads_v;

  int64_t tok_base, seq_len64;
  SequenceRange(pack.cu_seqlens, b, p.total_tokens, p.uniform_len, &tok_base, &seq_len64);
  const int seq_len = static_cast<int>(seq_len64);
  if ((p.state_update_tail_pass || p.short_row_tail_pass) &&
      !IsRecurrentTailRow(pack, p, b, seq_len)) {
    return;
  }
  const int capture_count = CaptureCount(pack, p, b, seq_len);

  const int64_t packed_stride = (static_cast<int64_t>(p.num_heads_q) + p.num_heads_k) * DK +
                                static_cast<int64_t>(p.num_heads_v) * d_v;
  const int64_t q_stride = PackedQkv ? packed_stride : static_cast<int64_t>(p.num_heads_q) * DK;
  const int64_t k_stride = PackedQkv ? packed_stride : static_cast<int64_t>(p.num_heads_k) * DK;
  const int64_t v_stride = static_cast<int64_t>(p.num_heads_v) * d_v;
  const int64_t v_input_stride = PackedQkv ? packed_stride : v_stride;
  const int64_t st_off =
      (static_cast<int64_t>(b) * p.num_heads_v + hv) * static_cast<int64_t>(d_v) * DK +
      static_cast<int64_t>(col) * DK;

  const bool needs_decay = pack.decay != nullptr && (p.update_rule == UpdateRule::kGated ||
                                                     p.update_rule == UpdateRule::kGatedDelta);
  const bool needs_retrieval =
      p.update_rule == UpdateRule::kDelta || p.update_rule == UpdateRule::kGatedDelta;
  const bool decay_per_key_dim = p.decay_per_key_dim_flag != 0;

  // Read the whole incoming row before writing anything: initial_state and final_state are
  // permitted to be the same allocation. Lane l takes k = l, l+32, ... rather than a
  // contiguous run: each of the kRegs accesses is then its own fully coalesced warp
  // transaction, and the extra requests in flight beat the single wide float4 a contiguous
  // run would allow (measured 5.86 us against 6.37 us at the Qwen3.8 decode shape).
  float s[kRegs];
#pragma unroll
  for (int i = 0; i < kRegs; ++i) {
    const int k = lane + i * 32;
    s[i] = pack.initial_state != nullptr ? pack.initial_state[st_off + k] : 0.0f;
  }

  for (int t = 0; t < seq_len; ++t) {
    const int64_t tok = tok_base + t;
    const int64_t gi = tok * p.num_heads_v + hv;

    float kv[kRegs], qv[kRegs];
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      const int k = lane + i * 32;
      kv[i] = static_cast<float>(pack.key[tok * k_stride + hk * DK + k]);
      qv[i] = static_cast<float>(pack.query[tok * q_stride + hq * DK + k]);
    }

    if (p.qk_l2_norm) {
      float nq = 0.0f, nk = 0.0f;
#pragma unroll
      for (int i = 0; i < kRegs; ++i) {
        nq += qv[i] * qv[i];
        nk += kv[i] * kv[i];
      }
      const float rq = rsqrtf(WarpReduceK<DK>(nq) + 1e-12f);
      const float rk = rsqrtf(WarpReduceK<DK>(nk) + 1e-12f);
#pragma unroll
      for (int i = 0; i < kRegs; ++i) {
        qv[i] *= rq;
        kv[i] *= rk;
      }
    }

    float decay_scalar = 1.0f;
    if (needs_decay && !decay_per_key_dim) {
      decay_scalar = __expf(EffectiveDecay(LoadGate<T>(pack.decay, gi, InputTypeGates), pack.a_log, pack.dt_bias, hv,
                                           p.gate_activation));
    }
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      if (needs_decay && decay_per_key_dim) {
        const int k = lane + i * 32;
        const float raw = LoadGate<T>(pack.decay, (tok * p.num_heads_v + hv) * DK + k, InputTypeGates);
        s[i] *= __expf(EffectiveDecay(raw, pack.a_log, pack.dt_bias, hv, p.gate_activation));
      } else {
        s[i] *= decay_scalar;
      }
    }

    float r = 0.0f;
    if (needs_retrieval) {
#pragma unroll
      for (int i = 0; i < kRegs; ++i) r += s[i] * kv[i];
      r = WarpReduceK<DK>(r);
    }

    const float beta_t =
        pack.beta != nullptr ? EffectiveBeta(LoadGate<T>(pack.beta, gi, InputTypeGates), p.beta_activation) : 1.0f;
    const float vv = static_cast<float>(pack.value[tok * v_input_stride + hv * d_v + col]);
    const float delta = beta_t * (vv - r);

    if (t < capture_count) {
      float* update_decay = StateUpdateDecay(pack, p, b, t, hv, DK, d_v);
      if (blockIdx.z == 0 && warp == 0 && lane == 0 && update_decay != nullptr) {
        *update_decay = decay_scalar;
      }
      float* update_key = StateUpdateKey(pack, p, b, t, hk, DK, d_v);
      if (blockIdx.z == 0 && warp == 0 && hv == hk * p.num_heads_v / p.num_heads_k &&
          update_key != nullptr) {
#pragma unroll
        for (int i = 0; i < kRegs; ++i) {
          update_key[lane + i * 32] = kv[i];
        }
      }
      float* update_delta = StateUpdateDelta(pack, p, b, t, hv, DK, d_v);
      if (lane == 0 && update_delta != nullptr) {
        update_delta[col] = delta;
      }
    }

    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      s[i] += kv[i] * delta;
      o += s[i] * qv[i];
    }
    o = WarpReduceK<DK>(o);
    if (lane == 0) {
      pack.output[tok * v_stride + hv * d_v + col] =
          static_cast<T>(p.scale * o);
    }
  }

  if (pack.final_state != nullptr) {
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      pack.final_state[st_off + lane + i * 32] = s[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Generic recurrent engine: any head geometry the warp kernel cannot take.
// One CTA per (sequence, v-head); the [DK x DV] state lives in shared memory.
// ---------------------------------------------------------------------------
template <typename T, int kThreads, bool InputTypeGates>
__global__ __launch_bounds__(kThreads) void GatedDeltaNetRecurrentKernel(VariantPack<T> pack,
                                                                         KernelParams p, int d_k,
                                                                         int d_v) {
  extern __shared__ float rsmem[];
  float* S = rsmem;                          // [d_k][d_v]
  float* kbuf = S + d_k * d_v;               // [d_k]
  float* qbuf = kbuf + d_k;                  // [d_k]
  float* rbuf = qbuf + d_k;                  // [max(d_v, 2)], see RecurrentSmemBytes
  float* gbuf = rbuf + (d_v > 2 ? d_v : 2);  // [d_k] per-key-dim decay

  const int b = blockIdx.x, hv = blockIdx.y;
  const int tid = threadIdx.x;
  const int hq = hv * p.num_heads_q / p.num_heads_v;
  const int hk = hv * p.num_heads_k / p.num_heads_v;

  int64_t tok_base, seq_len64;
  SequenceRange(pack.cu_seqlens, b, p.total_tokens, p.uniform_len, &tok_base, &seq_len64);
  const int seq_len = static_cast<int>(seq_len64);
  // Uniform across the CTA: seq_len depends only on blockIdx.x.
  if ((p.state_update_tail_pass || p.short_row_tail_pass) &&
      !IsRecurrentTailRow(pack, p, b, seq_len)) {
    return;
  }
  const int capture_count = CaptureCount(pack, p, b, seq_len);

  const int64_t q_stride = static_cast<int64_t>(p.num_heads_q) * d_k;
  const int64_t k_stride = static_cast<int64_t>(p.num_heads_k) * d_k;
  const int64_t v_stride = static_cast<int64_t>(p.num_heads_v) * d_v;
  const int64_t st_off = (static_cast<int64_t>(b) * p.num_heads_v + hv) *
                         static_cast<int64_t>(d_v) * d_k;

  const bool needs_decay = pack.decay != nullptr &&
                           (p.update_rule == UpdateRule::kGated ||
                            p.update_rule == UpdateRule::kGatedDelta);
  const bool needs_retrieval =
      p.update_rule == UpdateRule::kDelta || p.update_rule == UpdateRule::kGatedDelta;
  const bool decay_per_key_dim = p.decay_per_key_dim_flag != 0;

  // Read the whole incoming state before anything is written back: initial_state and
  // final_state are permitted to be the same allocation.
  for (int idx = tid; idx < d_k * d_v; idx += kThreads) {
    const int r = idx / d_v, c = idx % d_v;
    S[idx] = pack.initial_state != nullptr
                 ? pack.initial_state[st_off + static_cast<int64_t>(c) * d_k + r]
                 : 0.0f;
  }
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    const int64_t tok = tok_base + t;
    const int64_t gi = tok * p.num_heads_v + hv;

    for (int i = tid; i < d_k; i += kThreads) {
      kbuf[i] = (float)pack.key[tok * k_stride + hk * d_k + i];
      qbuf[i] = (float)pack.query[tok * q_stride + hq * d_k + i];
    }
    if (needs_decay && decay_per_key_dim) {
      for (int i = tid; i < d_k; i += kThreads) {
        const float raw = LoadGate<T>(pack.decay, (tok * p.num_heads_v + hv) * d_k + i, InputTypeGates);
        gbuf[i] = __expf(EffectiveDecay(raw, pack.a_log, pack.dt_bias, hv, p.gate_activation));
      }
    }
    __syncthreads();

    if (p.qk_l2_norm) {
      if (tid == 0) {
        float sq = 0.0f, sk = 0.0f;
        for (int i = 0; i < d_k; ++i) {
          sq += qbuf[i] * qbuf[i];
          sk += kbuf[i] * kbuf[i];
        }
        rbuf[0] = rsqrtf(sq + 1e-12f);
        rbuf[1] = rsqrtf(sk + 1e-12f);
      }
      __syncthreads();
      const float rq = rbuf[0], rk = rbuf[1];
      for (int i = tid; i < d_k; i += kThreads) {
        qbuf[i] *= rq;
        kbuf[i] *= rk;
      }
      __syncthreads();
    }

    const float g_scalar =
        (needs_decay && !decay_per_key_dim)
            ? __expf(EffectiveDecay(LoadGate<T>(pack.decay, gi, InputTypeGates), pack.a_log, pack.dt_bias, hv,
                                    p.gate_activation))
            : 1.0f;
    const float beta_t =
        pack.beta != nullptr ? EffectiveBeta(LoadGate<T>(pack.beta, gi, InputTypeGates), p.beta_activation) : 1.0f;

    // Decay, then retrieval against the decayed state.
    for (int idx = tid; idx < d_k * d_v; idx += kThreads) {
      const int r = idx / d_v;
      S[idx] *= decay_per_key_dim ? gbuf[r] : g_scalar;
    }
    __syncthreads();

    for (int c = tid; c < d_v; c += kThreads) {
      float acc = 0.0f;
      if (needs_retrieval) {
        for (int r = 0; r < d_k; ++r) acc += S[r * d_v + c] * kbuf[r];
      }
      const float vv = (float)pack.value[tok * v_stride + hv * d_v + c];
      rbuf[c] = beta_t * (vv - acc);
    }
    __syncthreads();

    if (t < capture_count) {
      float* update_decay = StateUpdateDecay(pack, p, b, t, hv, d_k, d_v);
      if (tid == 0 && update_decay != nullptr) {
        *update_decay = g_scalar;
      }
      float* update_key = StateUpdateKey(pack, p, b, t, hk, d_k, d_v);
      if (hv == hk * p.num_heads_v / p.num_heads_k && update_key != nullptr) {
        for (int r = tid; r < d_k; r += kThreads) {
          update_key[r] = kbuf[r];
        }
      }
      float* update_delta = StateUpdateDelta(pack, p, b, t, hv, d_k, d_v);
      if (update_delta != nullptr) {
        for (int c = tid; c < d_v; c += kThreads) {
          update_delta[c] = rbuf[c];
        }
      }
    }

    for (int idx = tid; idx < d_k * d_v; idx += kThreads) {
      const int r = idx / d_v, c = idx % d_v;
      S[idx] += kbuf[r] * rbuf[c];
    }
    __syncthreads();

    for (int c = tid; c < d_v; c += kThreads) {
      float acc = 0.0f;
      for (int r = 0; r < d_k; ++r) acc += S[r * d_v + c] * qbuf[r];
      pack.output[tok * v_stride + hv * d_v + c] = (T)(p.scale * acc);
    }

    __syncthreads();
  }

  if (pack.final_state != nullptr) {
    for (int idx = tid; idx < d_k * d_v; idx += kThreads) {
      const int r = idx / d_v, c = idx % d_v;
      pack.final_state[st_off + static_cast<int64_t>(c) * d_k + r] = S[idx];
    }
  }
}

constexpr int kRecurrentThreads = 256;
constexpr int kDecodeWarps = kRecurrentThreads / 32;

template <typename T, bool InputTypeGates, bool PackedQkv, int Warps>
void LaunchDecodeWarpSized(const Descriptor& desc, const VariantPack<T>& pack,
                           const KernelParams& p, cudaStream_t stream) {
  const dim3 grid(desc.batch, desc.num_heads_v,
                  (desc.head_size_v + Warps - 1) / Warps);
  const dim3 block(32, Warps, 1);
  switch (desc.head_size_qk) {
    case 64:
      GatedDeltaNetDecodeWarpKernel<T, 64, Warps, InputTypeGates, PackedQkv>
          <<<grid, block, 0, stream>>>(pack, p, desc.head_size_v);
      break;
    case 128:
      if constexpr (PackedQkv) {
        static const int specialize = ParseEnvironmentVariableWithDefault<int>("ORT_GDN_S2_SPECIALIZE", 1);
        ORT_ENFORCE(specialize == 0 || specialize == 1, "ORT_GDN_S2_SPECIALIZE must be 0 or 1.");
        if (specialize && desc.batch == 1 && p.total_tokens == p.uniform_len &&
            p.total_tokens >= 1 && p.total_tokens <= 8 &&
            p.num_heads_q == 16 && p.num_heads_k == 16 && p.num_heads_v == 48 &&
            desc.head_size_v == 128 && pack.cu_seqlens == nullptr &&
            p.gate_activation == GateActivation::kSoftplusDecay &&
            p.beta_activation == BetaActivation::kSigmoid && p.update_rule == UpdateRule::kGatedDelta &&
            p.qk_l2_norm && p.decay_per_key_dim_flag == 0 &&
            !p.state_update_tail_pass && !p.short_row_tail_pass) {
          GatedDeltaNetDecodeWarpKernel<T, 128, Warps, InputTypeGates, PackedQkv, true>
              <<<grid, block, 0, stream>>>(pack, p, desc.head_size_v);
          break;
        }
      }
      GatedDeltaNetDecodeWarpKernel<T, 128, Warps, InputTypeGates, PackedQkv>
          <<<grid, block, 0, stream>>>(pack, p, desc.head_size_v);
      break;
    default:
      GatedDeltaNetDecodeWarpKernel<T, 256, Warps, InputTypeGates, PackedQkv>
          <<<grid, block, 0, stream>>>(pack, p, desc.head_size_v);
      break;
  }
}

template <typename T, bool InputTypeGates>
Status LaunchRecurrent(const Descriptor& desc, bool warp_specialized, const VariantPack<T>& pack,
                       const KernelParams& p, size_t max_shared_memory_per_block,
                       cudaStream_t stream) {
  if (warp_specialized) {
    if (pack.packed_qkv) {
      LaunchDecodeWarpSized<T, InputTypeGates, true, kDecodeWarps>(desc, pack, p, stream);
    } else {
      LaunchDecodeWarpSized<T, InputTypeGates, false, kDecodeWarps>(desc, pack, p, stream);
    }
    return CUDA_CALL(cudaGetLastError());
  }

  const size_t smem = RecurrentSmemBytes(desc.head_size_qk, desc.head_size_v);
  static DynamicSmemConfig recurrent_smem_config;
  ORT_RETURN_IF_ERROR(SetMaxDynamicSmem(
      GatedDeltaNetRecurrentKernel<T, kRecurrentThreads, InputTypeGates>,
      max_shared_memory_per_block, recurrent_smem_config));
  const dim3 grid(desc.batch, desc.num_heads_v, 1);
  GatedDeltaNetRecurrentKernel<T, kRecurrentThreads, InputTypeGates>
      <<<grid, kRecurrentThreads, smem, stream>>>(pack, p, desc.head_size_qk, desc.head_size_v);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace

template <typename T, bool InputTypeGates>
static Status LaunchGatedDeltaNetTypedGates(const Descriptor& desc, const Plan& plan, const VariantPack<T>& pack,
                                            float scale, int max_threads_per_block,
                                            size_t max_shared_memory_per_block, cudaStream_t stream) {
  KernelParams p{};
  p.total_tokens = desc.total_tokens;
  p.uniform_len = desc.batch > 0 ? desc.total_tokens / desc.batch : 0;
  p.num_heads_q = desc.num_heads_q;
  p.num_heads_k = desc.num_heads_k;
  p.num_heads_v = desc.num_heads_v;
  p.scale = scale;
  p.gate_activation = desc.gate_activation;
  p.beta_activation = desc.beta_activation;
  p.update_rule = desc.update_rule;
  p.qk_l2_norm = desc.qk_l2_norm;
  p.state_update_capacity = desc.state_update_capacity;
  p.decay_per_key_dim_flag = desc.decay_per_key_dim ? 1 : 0;
  p.bf16_precision = desc.bf16_precision;

  if (plan.engine == Engine::kChunkedSplit) {
    return LaunchGatedDeltaNetSplit<T>(desc, plan, pack, scale, stream);
  }

  if (plan.engine == Engine::kChunked) {
    // SelectPlan only routes float16 and bfloat16 here, so instantiating the mma kernels for
    // float would emit cubin that can never launch.
    if constexpr (!std::is_same_v<T, half> && !std::is_same_v<T, __nv_bfloat16>) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "GatedDeltaNet: the chunked engine requires float16 or bfloat16 input");
    } else {
      p.state_update_tail_pass = plan.state_update_tail_pass;
      p.short_row_tail_pass = plan.short_row_tail_pass;
      const dim3 grid(desc.batch, desc.num_heads_v, desc.head_size_v / kDVB);
      auto launch_chunked = [&]<int BT, Bf16Precision kPrec>() -> Status {
        static DynamicSmemConfig smem_config;
        ORT_RETURN_IF_ERROR(SetMaxDynamicSmem(
            GatedDeltaNetChunkedKernel<T, 128, 128, BT, InputTypeGates, kPrec>, plan.smem_bytes, smem_config));
        GatedDeltaNetChunkedKernel<T, 128, 128, BT, InputTypeGates, kPrec>
            <<<grid, kChunkedThreads, plan.smem_bytes, stream>>>(pack, p);
        return CUDA_CALL(cudaGetLastError());
      };
      auto dispatch_prec = [&]<int BT>() -> Status {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
          if (desc.bf16_precision == Bf16Precision::kFp16) {
            return launch_chunked.template operator()<BT, Bf16Precision::kFp16>();
          }
          if (desc.bf16_precision == Bf16Precision::kBf16) {
            return launch_chunked.template operator()<BT, Bf16Precision::kBf16>();
          }
        }
        return launch_chunked.template operator()<BT, Bf16Precision::kFp32>();
      };
      // BT=32 halves the [BT x BT] and per-token tiles, which is what lets the chunked
      // engine fit devices with a 99 KB opt-in limit (SM120) instead of SM90's 227 KB.
      if (plan.chunk_size == 32) {
        ORT_RETURN_IF_ERROR(dispatch_prec.template operator()<32>());
      } else {
        ORT_RETURN_IF_ERROR(dispatch_prec.template operator()<64>());
      }
      if (!plan.state_update_tail_pass && !plan.short_row_tail_pass) {
        return Status::OK();
      }

      // The rows the pass above skipped are emitted by the recurrent engine. Same stream, and
      // the two row sets are disjoint, so the merged result is deterministic.
      // The chunked engine only accepts head_size_qk == head_size_v == 128, so the warp
      // kernel is always the right shape here.
      return LaunchRecurrent<T, InputTypeGates>(desc, /*warp_specialized=*/true, pack, p,
                                                max_shared_memory_per_block, stream);
    }
  }

  (void)max_threads_per_block;
  return LaunchRecurrent<T, InputTypeGates>(desc, plan.warp_specialized, pack, p, max_shared_memory_per_block,
                                            stream);
}

template <typename T>
Status LaunchGatedDeltaNet(const Descriptor& desc, const Plan& plan, const VariantPack<T>& pack,
                           float scale, int max_threads_per_block,
                           size_t max_shared_memory_per_block, cudaStream_t stream) {
  if (pack.gates_in_input_type) {
    return LaunchGatedDeltaNetTypedGates<T, true>(
        desc, plan, pack, scale, max_threads_per_block, max_shared_memory_per_block, stream);
  }
  return LaunchGatedDeltaNetTypedGates<T, false>(
      desc, plan, pack, scale, max_threads_per_block, max_shared_memory_per_block, stream);
}

template Status LaunchGatedDeltaNet<float>(const Descriptor&, const Plan&, const VariantPack<float>&,
                                           float, int, size_t, cudaStream_t);
template Status LaunchGatedDeltaNet<half>(const Descriptor&, const Plan&, const VariantPack<half>&,
                                          float, int, size_t, cudaStream_t);
template Status LaunchGatedDeltaNet<__nv_bfloat16>(
    const Descriptor&, const Plan&, const VariantPack<__nv_bfloat16>&, float, int, size_t,
    cudaStream_t);

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
