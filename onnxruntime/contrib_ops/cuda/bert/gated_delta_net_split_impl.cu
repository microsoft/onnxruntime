// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GatedDeltaNet split chunked engine (K1 "prepare" + K2 "scan").
//
// The fused engine in gated_delta_net_impl.cu does everything for a chunk inside one CTA
// that also carries the recurrent state, so every chunk pays for work that has nothing to
// do with the state. Profiling that kernel on H200 at the Qwen3.8 geometry showed the cost:
// 157.5 KB of shared memory pins it to one CTA per SM, the grid is 96 CTAs against 132 SMs
// (0.73 waves), tensor pipes are 7.1% busy and DRAM 1.8%. It is bound by the serial
// dependency chain inside a CTA, not by the machine. Splitting the v axis to raise the CTA
// count made it 1.6x slower, because the state-independent half of the chunk is
// v-invariant and simply got duplicated.
//
// This engine moves that half into its own launch. Starting from the chunked form
// (derived in gated_delta_net_impl.cu) and pushing (I+M)^-1 through the subtraction:
//
//     U = (I+M)^-1 (Ubar - Wbar S0) = Uv - W S0,   Uv = (I+M)^-1 Ubar,  W = (I+M)^-1 Wbar
//
// so U's state-independent part factorizes out. K1 owns one (sequence, v-head, chunk) --
// 6144 CTAs at T=8192 instead of 96 -- and emits per chunk
//
//     W  = (I+M)^-1 diag(beta exp(gc)) k      [BT x DK]
//     Uv = (I+M)^-1 (beta v)                  [BT x DV]
//     P[t,s] = (q_t . k_s) exp(gc_t - gc_s)   [BT x BT], inclusive lower
//     Qg = q exp(gc), Kd = k exp(gc_BT - gc)  [BT x DK]
//     decay = exp(gc_BT)
//
// K2 keeps the sequence walk and the state, and is left with four GEMMs per chunk:
//
//     U = Uv - W S,   o = scale (P U + Qg S),   S = decay S + Kd^T U
//
// Two consequences, both of which the fused kernel could not have:
//
//  1. The serial chain per chunk drops from ~8.4 MFLOP and ~15 barriers to ~3.7 MFLOP and
//     6 barriers per CTA.
//  2. Every one of those four GEMMs scales with the v-block width, so narrowing the block
//     to raise the CTA count now genuinely divides the work instead of duplicating it.
//
// The state is kept in fp32 across chunks with a 16-bit mma staging tile, and K2 reuses
// the upper half of the stacked [W; Qg] buffer for Kd once the state GEMM completes, keeping
// K2 at 112.6 KB so two CTAs can still share an SM on Hopper.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <mutex>

#include "contrib_ops/cuda/bert/gated_delta_net_impl.h"
#include "contrib_ops/cuda/bert/gated_delta_net_mma.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace gated_delta_net {

namespace {

// Padded leading dimensions; see the note in gated_delta_net_impl.cu on why the +8.
template <int DK, int DV, int BT, int DVB>
struct SplitLd {
  static constexpr int kKh = DK + 8;                     // rows of length DK
  static constexpr int kVh = DV + 8;                     // rows of length DV
  static constexpr int kBh = DVB + 8;                    // rows of length DVB
  static constexpr int kMh = BT + 8;                     // rows of length BT
  static constexpr int kSh = DVB + 8;                    // state rows, DVB wide
  static constexpr int kSf = DVB + 4;                    // fp32 state rows, DVB wide
  static constexpr int kF1 = (DV > BT ? DV : BT) + 4;    // K1 fp32 scratch
  static constexpr int kF2 = (DVB > BT ? DVB : BT) + 4;  // K2 fp32 scratch
};

// ---------------------------------------------------------------------------
// K1 shared memory
// ---------------------------------------------------------------------------
template <typename T, int DK, int DV, int BT>
struct PrepSmem {
  T* k_h;       // [BT][kKh]
  T* q_h;       // [BT][kKh]
  T* v_h;       // [BT][kVh]  beta v, then Uv
  T* m_h;       // [BT][kMh]  M, then N (BuildTriInverse is done with M by then)
  T* db_h;      // [BT][kMh]
  T* ti_h;      // [BT][kMh]  (I+M)^-1
  float* t_f;   // [BT][kF1]
  float* gc;    // [BT]
  float* beta;  // [BT]
};

template <typename T, int DK, int DV, int BT>
__device__ __forceinline__ PrepSmem<T, DK, DV, BT> CarvePrep(char* raw) {
  using L = SplitLd<DK, DV, BT, 64>;
  PrepSmem<T, DK, DV, BT> s;
  float* f = reinterpret_cast<float*>(raw);
  s.t_f = f;
  f += BT * L::kF1;
  s.gc = f;
  f += BT;
  s.beta = f;
  f += BT;
  T* h = reinterpret_cast<T*>(f);
  s.k_h = h;
  h += BT * L::kKh;
  s.q_h = h;
  h += BT * L::kKh;
  s.v_h = h;
  h += BT * L::kVh;
  s.m_h = h;
  h += BT * L::kMh;
  s.db_h = h;
  h += BT * L::kMh;
  s.ti_h = h;
  return s;
}

// Layout of one chunk's K1 output. Chunks are indexed ((b * Hv + hv) * chunks_per_pass + c),
// so K2's walk over c is a unit stride. W and Qg are adjacent so K2 can multiply both by the
// state in a single [2 BT x DK] GEMM. Uv is stored in a parallel float array (`uv_tiles`) so
// FP32 delta-update cancellation residuals survive between K1 and K2.
template <int DK, int DV, int BT>
struct PrepTile {
  static constexpr int64_t kW = static_cast<int64_t>(BT) * DK;
  static constexpr int64_t kQg = static_cast<int64_t>(BT) * DK;
  static constexpr int64_t kUv = static_cast<int64_t>(BT) * DV;
  static constexpr int64_t kP = static_cast<int64_t>(BT) * BT;
  static constexpr int64_t kKd = static_cast<int64_t>(BT) * DK;
  static constexpr int64_t kHalves = kW + kQg + kP + kKd;
};

// ---------------------------------------------------------------------------
// K1: everything in a chunk that does not touch the recurrent state.
// One CTA per (sequence, v-head, chunk).
// ---------------------------------------------------------------------------
template <typename T, typename MT, bool kExactFp32Uv, int DK, int DV, int BT>
__device__ __forceinline__ void RunPrepareStep(
    char* smem_raw, const KernelParams& p, int len, bool needs_retrieval, MT* out, float* uv_out,
    float* decay_out, bool negate_decay, int tid, int warp_id, int lane) {
  using L = SplitLd<DK, DV, BT, 64>;
  using Tile = PrepTile<DK, DV, BT>;
  const PrepSmem<MT, DK, DV, BT> s = CarvePrep<MT, DK, DV, BT>(smem_raw);
  T* k_in = reinterpret_cast<T*>(s.k_h);
  T* q_in = reinterpret_cast<T*>(s.q_h);
  T* v_in = reinterpret_cast<T*>(s.v_h);

  if constexpr (!std::is_same_v<T, MT>) {
    for (int idx = tid; idx < BT * DV; idx += kChunkedThreads) {
      const int t = idx / DV, d = idx % DV;
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

  // M and its exact inverse.
  SmemGemm<BT, BT, DK, false, true, false>(s.t_f, L::kF1, s.k_h, L::kKh, s.k_h, L::kKh, warp_id,
                                           lane);
  __syncthreads();
  for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
    const int t = idx / BT, sc = idx % BT;
    const float m = (sc < t && needs_retrieval)
                        ? s.beta[t] * s.t_f[t * L::kF1 + sc] * __expf(s.gc[t] - s.gc[sc])
                        : 0.0f;
    s.m_h[t * L::kMh + sc] = FloatToElem<MT>(m);
  }
  __syncthreads();

  // The Neumann iterate lands on top of M: BuildTriInverse has consumed M by the time it
  // writes N, and the prepare kernel only fits two CTAs on an SM without the extra tile.
  BuildTriInverse<BT>(s.m_h, L::kMh, s.db_h, s.m_h, s.ti_h, L::kMh, s.t_f, L::kF1, warp_id, lane,
                      tid);

  MT* w_out = out;
  MT* qg_out = w_out + Tile::kW;
  MT* p_out = qg_out + Tile::kQg;
  MT* kd_out = p_out + Tile::kP;

  // Uv = (I+M)^-1 (beta v).
  if constexpr (kExactFp32Uv) {
    for (int idx = tid; idx < BT * DV; idx += kChunkedThreads) {
      const int t = idx / DV, d = idx % DV;
      s.t_f[t * L::kF1 + d] = ElemToFloat(s.v_h[t * L::kVh + d]) * s.beta[t];
    }
    __syncthreads();
    SmemGemmFp32State<BT, DV, BT, false>(uv_out, DV, s.ti_h, L::kMh, s.t_f, L::kF1, s.v_h, L::kVh,
                                         warp_id, lane, tid);
    __syncthreads();
  } else {
    for (int idx = tid; idx < BT * DV; idx += kChunkedThreads) {
      const int t = idx / DV, d = idx % DV;
      s.v_h[t * L::kVh + d] = ScaleElem(s.v_h[t * L::kVh + d], s.beta[t]);
    }
    __syncthreads();
    SmemGemm<BT, DV, BT, false, false, false>(s.t_f, L::kF1, s.ti_h, L::kMh, s.v_h, L::kVh, warp_id,
                                              lane);
    __syncthreads();
    for (int idx = tid; idx < BT * DV; idx += kChunkedThreads) {
      uv_out[idx] = s.t_f[(idx / DV) * L::kF1 + idx % DV];
    }
  }

  // Kd and Qg are just row scalings of k and q, so emit them before k is overwritten by the
  // row scaling W needs.
  for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
    const int t = idx / DK, d = idx % DK;
    kd_out[idx] = ScaleElem(s.k_h[t * L::kKh + d], __expf(g_total - s.gc[t]));
    qg_out[idx] = ScaleElem(s.q_h[t * L::kKh + d], __expf(s.gc[t]));
  }
  __syncthreads();

  // P = (q k^T) exp(gc_t - gc_s), inclusive lower. q_h is still unscaled here.
  SmemGemm<BT, BT, DK, false, true, false>(s.t_f, L::kF1, s.q_h, L::kKh, s.k_h, L::kKh, warp_id,
                                           lane);
  __syncthreads();
  for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
    const int t = idx / BT, sc = idx % BT;
    p_out[idx] = FloatToElem<MT>(
        (sc <= t && t < len) ? s.t_f[t * L::kF1 + sc] * __expf(s.gc[t] - s.gc[sc]) : 0.0f);
  }
  for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
    const int t = idx / DK, d = idx % DK;
    const float bw = needs_retrieval ? s.beta[t] * __expf(s.gc[t]) : 0.0f;
    s.k_h[t * L::kKh + d] = FloatToElem<MT>(ElemToFloat(s.k_h[t * L::kKh + d]) * bw);
  }
  __syncthreads();
  // W = (I+M)^-1 diag(beta exp(gc)) k, with the row scaling folded into k in place.
  SmemGemm<BT, DK, BT, false, false, false>(s.t_f, L::kF1, s.ti_h, L::kMh, s.k_h, L::kKh, warp_id,
                                            lane);
  __syncthreads();
  for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
    w_out[idx] = FloatToElem<MT>(s.t_f[(idx / DK) * L::kF1 + idx % DK]);
  }

  if (tid == 0) {
    const float d = __expf(g_total);
    *decay_out = negate_decay ? -d : d;
  }
}

template <typename T, int DK, int DV, int BT>
__device__ __noinline__ void RunPrepareStepExactBf16Fallback(
    char* smem_raw, const KernelParams& p, int len, bool needs_retrieval, T* out, float* uv_out,
    float* decay_out, bool negate_decay, int tid, int warp_id, int lane) {
  RunPrepareStep<T, T, /*kExactFp32Uv=*/true, DK, DV, BT>(
      smem_raw, p, len, needs_retrieval, out, uv_out, decay_out, negate_decay, tid, warp_id, lane);
}

template <typename T, int DK, int DV, int BT,
          Bf16Precision kBf16Prec = Bf16Precision::kFp32>
__global__ __launch_bounds__(kChunkedThreads) void GatedDeltaNetPrepareKernel(
    VariantPack<T> pack, KernelParams p, T* tiles, float* uv_tiles, float* decays, int chunk_base,
    int chunks_per_pass) {
  using L = SplitLd<DK, DV, BT, 64>;
  using Tile = PrepTile<DK, DV, BT>;
  extern __shared__ char smem_raw[];
  const PrepSmem<T, DK, DV, BT> s = CarvePrep<T, DK, DV, BT>(smem_raw);

  const int b = blockIdx.x, hv = blockIdx.y, c_local = blockIdx.z;
  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  const int hq = hv * p.num_heads_q / p.num_heads_v;
  const int hk = hv * p.num_heads_k / p.num_heads_v;

  int64_t tok_base, seq_len64;
  SequenceRange(pack.cu_seqlens, b, p.total_tokens, p.uniform_len, &tok_base, &seq_len64);
  const int seq_len = static_cast<int>(seq_len64);
  const int chunk0 = (chunk_base + c_local) * BT;
  if (chunk0 >= seq_len) return;
  const int len = min(BT, seq_len - chunk0);

  const int64_t q_stride = static_cast<int64_t>(p.num_heads_q) * DK;
  const int64_t k_stride = static_cast<int64_t>(p.num_heads_k) * DK;
  const int64_t v_stride = static_cast<int64_t>(p.num_heads_v) * DV;

  const bool needs_decay = pack.decay != nullptr &&
                           (p.update_rule == UpdateRule::kGated ||
                            p.update_rule == UpdateRule::kGatedDelta);
  const bool needs_retrieval =
      p.update_rule == UpdateRule::kDelta || p.update_rule == UpdateRule::kGatedDelta;

  constexpr bool kCheckFp16 =
      std::is_same_v<T, __nv_bfloat16> && kBf16Prec == Bf16Precision::kFp16;
  bool thread_out_of_fp16 = kCheckFp16 && (fabsf(p.scale) > 16.0f);
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
  for (int idx = tid; idx < BT * DV; idx += kChunkedThreads) {
    const int t = idx / DV, d = idx % DV;
    const int64_t tok = tok_base + chunk0 + t;
    const T vv = (t < len) ? pack.value[tok * v_stride + hv * DV + d] : FloatToElem<T>(0.f);
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
      g = needs_decay ? EffectiveDecay(pack.decay[gi], pack.a_log, pack.dt_bias, hv,
                                       p.gate_activation)
                      : 0.0f;
      be = pack.beta != nullptr ? EffectiveBeta(pack.beta[gi], p.beta_activation) : 1.0f;
    }
    s.gc[tid] = g;
    s.beta[tid] = be;
    if constexpr (kCheckFp16) {
      if (tid < len && (!(g <= 0.0f) || !(fabsf(be) <= 1.0f) || (needs_retrieval && be < 0.0f))) {
        thread_out_of_fp16 = true;
      }
    }
  }

  const int64_t tile_base =
      (static_cast<int64_t>(b) * p.num_heads_v + hv) * chunks_per_pass + c_local;
  T* out = tiles + tile_base * Tile::kHalves;
  float* uv_out = uv_tiles + tile_base * Tile::kUv;
  float* decay_out = decays + tile_base;

  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if constexpr (kBf16Prec == Bf16Precision::kBf16) {
      __syncthreads();
      RunPrepareStep<T, T, /*kExactFp32Uv=*/false, DK, DV, BT>(
          smem_raw, p, len, needs_retrieval, out, uv_out, decay_out, /*negate_decay=*/false, tid,
          warp_id, lane);
    } else if constexpr (kBf16Prec == Bf16Precision::kFp16) {
      const bool out_of_fp16 = __syncthreads_or(thread_out_of_fp16) != 0;
      if (!out_of_fp16) {
        RunPrepareStep<T, __half, /*kExactFp32Uv=*/false, DK, DV, BT>(
            smem_raw, p, len, needs_retrieval, reinterpret_cast<__half*>(out), uv_out, decay_out,
            /*negate_decay=*/false, tid, warp_id, lane);
      } else {
        RunPrepareStepExactBf16Fallback<T, DK, DV, BT>(
            smem_raw, p, len, needs_retrieval, out, uv_out, decay_out, /*negate_decay=*/true, tid,
            warp_id, lane);
      }
    } else {
      __syncthreads();
      RunPrepareStep<T, T, /*kExactFp32Uv=*/true, DK, DV, BT>(
          smem_raw, p, len, needs_retrieval, out, uv_out, decay_out, /*negate_decay=*/false, tid,
          warp_id, lane);
    }
  } else {
    __syncthreads();
    RunPrepareStep<T, T, /*kExactFp32Uv=*/false, DK, DV, BT>(
        smem_raw, p, len, needs_retrieval, out, uv_out, decay_out, /*negate_decay=*/false, tid,
        warp_id, lane);
  }
}

// ---------------------------------------------------------------------------
// K2 shared memory. The recurrent state lives in fp32 (`s_f`) for the whole walk
// and is staged into `s_h` for 16-bit mma; `kd_h` reuses the upper half of `wq_h`.
// ---------------------------------------------------------------------------
template <typename T, int DK, int DV, int BT, int DVB>
struct ScanSmem {
  float* s_f;  // [DK][kSf]    recurrent state in fp32
  float* t_f;  // [DK][kF2], and DK == 2 BT holds both halves of the stacked GEMM
  T* s_h;      // [DK][kSh]    16-bit mma operand staging tile for the state
  T* wq_h;     // [2 BT][kKh]  W stacked on Qg
  T* u_h;      // [BT][kBh]    Uv, then U in place
  T* p_h;      // [BT][kMh]
  T* kd_h;     // [BT][kKh]    aliases wq_h + BT * kKh after W/Qg GEMM
};

template <typename T, int DK, int DV, int BT, int DVB>
__device__ __forceinline__ ScanSmem<T, DK, DV, BT, DVB> CarveScan(char* raw) {
  using L = SplitLd<DK, DV, BT, DVB>;
  static_assert(DK == 2 * BT, "t_f must cover both halves of the stacked [2 BT x DVB] GEMM");
  ScanSmem<T, DK, DV, BT, DVB> s;
  float* f = reinterpret_cast<float*>(raw);
  s.s_f = f;
  f += DK * L::kSf;
  s.t_f = f;
  f += DK * L::kF2;
  T* h = reinterpret_cast<T*>(f);
  s.s_h = h;
  h += DK * L::kSh;
  s.wq_h = h;
  h += 2 * BT * L::kKh;
  s.u_h = h;
  h += BT * L::kBh;
  s.p_h = h;
  h += BT * L::kMh;
  s.kd_h = s.wq_h + BT * L::kKh;
  return s;
}

template <typename T, int DK, int DV, int BT, int DVB, bool kRecomputeFromInputs>
__device__ __noinline__ void RunScanStepExactBf16Fallback(
    char* smem_raw, const VariantPack<T>& pack, const KernelParams& p, const T* in,
    const float* uv_in, float decay_all, int64_t tok_base, int chunk0, int len, int hv, int v0,
    int64_t v_stride, int tid, int warp_id, int lane) {
  using L = SplitLd<DK, DV, BT, DVB>;
  using Tile = PrepTile<DK, DV, BT>;
  const ScanSmem<T, DK, DV, BT, DVB> s = CarveScan<T, DK, DV, BT, DVB>(smem_raw);
  float* u_f = reinterpret_cast<float*>(s.s_h);

  if constexpr (kRecomputeFromInputs && std::is_same_v<T, __nv_bfloat16>) {
    // K1 prepared FP16 tiles, which may already have rounded tiny BF16 components (such as
    // normalized query/key entries or exp(gc) factors below 2^-24) to zero. Recompute the
    // chunk's BF16 preparation tiles from the original inputs before multiplying by the state.
    const int hq = hv * p.num_heads_q / p.num_heads_v;
    const int hk = hv * p.num_heads_k / p.num_heads_v;
    const int64_t q_stride = static_cast<int64_t>(p.num_heads_q) * DK;
    const int64_t k_stride = static_cast<int64_t>(p.num_heads_k) * DK;
    const bool needs_decay = pack.decay != nullptr &&
                             (p.update_rule == UpdateRule::kGated ||
                              p.update_rule == UpdateRule::kGatedDelta);
    const bool needs_retrieval =
        p.update_rule == UpdateRule::kDelta || p.update_rule == UpdateRule::kGatedDelta;

    T* k_h = s.wq_h;
    T* q_h = s.wq_h + BT * L::kKh;
    auto load_and_normalize_qk = [&]() {
      for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
        const int t = idx / DK, d = idx % DK;
        const int64_t tok = tok_base + chunk0 + t;
        k_h[t * L::kKh + d] =
            (t < len) ? pack.key[tok * k_stride + hk * DK + d] : FloatToElem<T>(0.f);
        q_h[t * L::kKh + d] =
            (t < len) ? pack.query[tok * q_stride + hq * DK + d] : FloatToElem<T>(0.f);
      }
      if (p.qk_l2_norm) {
        __syncthreads();
        for (int t = warp_id; t < BT; t += kChunkedWarps) {
          float sq = 0.0f, sk = 0.0f;
          for (int d = lane; d < DK; d += 32) {
            const float qv = ElemToFloat(q_h[t * L::kKh + d]);
            const float kv = ElemToFloat(k_h[t * L::kKh + d]);
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
            const float qv = ElemToFloat(q_h[t * L::kKh + d]);
            const float kv = ElemToFloat(k_h[t * L::kKh + d]);
            q_h[t * L::kKh + d] = FloatToElem<T>(qv * rq);
            k_h[t * L::kKh + d] = FloatToElem<T>(kv * rk);
          }
        }
      }
    };

    float* gc = reinterpret_cast<float*>(s.u_h);
    float* beta = gc + BT;
    load_and_normalize_qk();
    if (tid < BT) {
      const int64_t tok = tok_base + chunk0 + tid;
      const int64_t gi = tok * p.num_heads_v + hv;
      float g = 0.0f, be = 0.0f;
      if (tid < len) {
        g = needs_decay ? EffectiveDecay(pack.decay[gi], pack.a_log, pack.dt_bias, hv,
                                         p.gate_activation)
                        : 0.0f;
        be = pack.beta != nullptr ? EffectiveBeta(pack.beta[gi], p.beta_activation) : 1.0f;
      }
      gc[tid] = g;
      beta[tid] = be;
    }
    __syncthreads();

    InclusiveScanBT<BT>(gc, tid);
    __syncthreads();
    const float g_total = gc[BT - 1];
    decay_all = __expf(g_total);

    float* tf0 = s.t_f;
    T* m_h = reinterpret_cast<T*>(s.t_f + BT * L::kF2);
    T* db_h = s.s_h;
    T* ti_h = s.p_h;
    SmemGemm<BT, BT, DK, false, true, false>(tf0, L::kF2, k_h, L::kKh, k_h, L::kKh, warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      const int t = idx / BT, sc = idx % BT;
      const float m = (sc < t && needs_retrieval)
                          ? beta[t] * tf0[t * L::kF2 + sc] * __expf(gc[t] - gc[sc])
                          : 0.0f;
      m_h[t * L::kMh + sc] = FloatToElem<T>(m);
    }
    __syncthreads();
    BuildTriInverse<BT>(m_h, L::kMh, db_h, m_h, ti_h, L::kMh, tf0, L::kF2, warp_id, lane, tid);

    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      const int t = idx / DK, d = idx % DK;
      q_h[t * L::kKh + d] = ScaleElem(q_h[t * L::kKh + d], __expf(gc[t]));
      const float bw = needs_retrieval ? beta[t] * __expf(gc[t]) : 0.0f;
      k_h[t * L::kKh + d] = FloatToElem<T>(ElemToFloat(k_h[t * L::kKh + d]) * bw);
    }
    __syncthreads();
    T* w_tmp = reinterpret_cast<T*>(tf0);
    SmemGemm<BT, DK, BT, false, false, false, T, kChunkedWarps, T>(
        w_tmp, L::kKh, ti_h, L::kMh, k_h, L::kKh, warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      const int t = idx / DK, d = idx % DK;
      k_h[t * L::kKh + d] = w_tmp[t * L::kKh + d];
    }
    __syncthreads();

    SmemGemmFp32State<2 * BT, DVB, DK, false>(s.t_f, L::kF2, s.wq_h, L::kKh, s.s_f, L::kSf, s.s_h,
                                              L::kSh, warp_id, lane, tid);
    __syncthreads();

    float* v_f = reinterpret_cast<float*>(s.wq_h);
    for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
      const int t = idx / DVB, c = idx % DVB;
      const int64_t tok = tok_base + chunk0 + t;
      const T vv = (t < len) ? pack.value[tok * v_stride + hv * DV + v0 + c] : FloatToElem<T>(0.f);
      v_f[t * L::kF2 + c] = ElemToFloat(vv) * beta[t];
    }
    const float saved_gc = (tid < BT) ? gc[tid] : 0.0f;
    __syncthreads();
    SmemGemmFp32State<BT, DVB, BT, false>(u_f, L::kSf, ti_h, L::kMh, v_f, L::kF2, s.u_h, L::kBh,
                                          warp_id, lane, tid);
    __syncthreads();
    for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
      const int t = idx / DVB, c = idx % DVB;
      u_f[t * L::kSf + c] -= tf0[t * L::kF2 + c];
    }
    if (tid < BT) {
      gc[tid] = saved_gc;
    }

    load_and_normalize_qk();
    __syncthreads();
    SmemGemm<BT, BT, DK, false, true, false>(tf0, L::kF2, q_h, L::kKh, k_h, L::kKh, warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      const int t = idx / BT, sc = idx % BT;
      s.p_h[t * L::kMh + sc] = FloatToElem<T>(
          (sc <= t && t < len) ? tf0[t * L::kF2 + sc] * __expf(gc[t] - gc[sc]) : 0.0f);
    }
    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      const int t = idx / DK, d = idx % DK;
      s.kd_h[t * L::kKh + d] = ScaleElem(k_h[t * L::kKh + d], __expf(g_total - gc[t]));
    }
    __syncthreads();
  } else {
    const T* wq_in = in;
    const T* p_in = wq_in + Tile::kW + Tile::kQg;
    const T* kd_in = p_in + Tile::kP;
    for (int idx = tid; idx < 2 * BT * DK; idx += kChunkedThreads) {
      s.wq_h[(idx / DK) * L::kKh + idx % DK] = wq_in[idx];
    }
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      s.p_h[(idx / BT) * L::kMh + idx % BT] = p_in[idx];
    }
    __syncthreads();
    SmemGemmFp32State<2 * BT, DVB, DK, false>(s.t_f, L::kF2, s.wq_h, L::kKh, s.s_f, L::kSf, s.s_h,
                                              L::kSh, warp_id, lane, tid);
    __syncthreads();
    for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
      const int t = idx / DVB, c = idx % DVB;
      u_f[t * L::kSf + c] = uv_in[t * DV + v0 + c] - s.t_f[t * L::kF2 + c];
    }
    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      s.kd_h[(idx / DK) * L::kKh + idx % DK] = kd_in[idx];
    }
    __syncthreads();
  }

  SmemGemmFp32State<BT, DVB, BT, false>(s.t_f, L::kF2, s.p_h, L::kMh, u_f, L::kSf, s.u_h, L::kBh,
                                        warp_id, lane, tid);
  __syncthreads();
  for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
    const int t = idx / DVB, c = idx % DVB;
    if (t < len) {
      const int64_t tok = tok_base + chunk0 + t;
      pack.output[tok * v_stride + hv * DV + v0 + c] =
          FloatToElem<T>(p.scale * (s.t_f[t * L::kF2 + c] + s.t_f[(BT + t) * L::kF2 + c]));
    }
  }
  for (int idx = tid; idx < DK * DVB; idx += kChunkedThreads) {
    s.s_f[(idx / DVB) * L::kSf + idx % DVB] *= decay_all;
  }
  __syncthreads();
  SmemGemmFp32State<DK, DVB, BT, true, kChunkedWarps, T, /*TA=*/true>(
      s.s_f, L::kSf, s.kd_h, L::kKh, u_f, L::kSf, s.u_h, L::kBh, warp_id, lane, tid);
  __syncthreads();
}

// ---------------------------------------------------------------------------
// K2: the recurrence. One CTA per (sequence, v-head, v-block).
// ---------------------------------------------------------------------------
template <typename T, int DK, int DV, int BT, int DVB,
          Bf16Precision kBf16Prec = Bf16Precision::kFp32>
__global__ __launch_bounds__(kChunkedThreads) void GatedDeltaNetScanKernel(
    VariantPack<T> pack, KernelParams p, const T* tiles, const float* uv_tiles,
    const float* decays, float* carry, int chunk_base, int chunks_per_pass, int n_chunks,
    bool first_pass, bool last_pass) {
  using L = SplitLd<DK, DV, BT, DVB>;
  using Tile = PrepTile<DK, DV, BT>;
  extern __shared__ char smem_raw[];
  const ScanSmem<T, DK, DV, BT, DVB> s = CarveScan<T, DK, DV, BT, DVB>(smem_raw);

  const int b = blockIdx.x, hv = blockIdx.y, v0 = blockIdx.z * DVB;
  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

  int64_t tok_base, seq_len64;
  SequenceRange(pack.cu_seqlens, b, p.total_tokens, p.uniform_len, &tok_base, &seq_len64);
  const int seq_len = static_cast<int>(seq_len64);
  const int64_t v_stride = static_cast<int64_t>(p.num_heads_v) * DV;
  const int64_t st_off = (static_cast<int64_t>(b) * p.num_heads_v + hv) * DV * DK;

  // The state crosses a pass boundary through `carry`, which is workspace the launcher only
  // reserves when there is more than one pass. final_state is not usable for that: the
  // caller may not have asked for it.
  const float* seed = first_pass ? pack.initial_state : carry;
  for (int idx = tid; idx < DK * DVB; idx += kChunkedThreads) {
    const int r = idx / DVB, c = idx % DVB;
    s.s_f[r * L::kSf + c] =
        seed != nullptr ? seed[st_off + static_cast<int64_t>(v0 + c) * DK + r] : 0.0f;
  }
  __syncthreads();

  const int64_t head_base =
      (static_cast<int64_t>(b) * p.num_heads_v + hv) * chunks_per_pass;

  for (int c_local = 0; c_local < n_chunks; ++c_local) {
    const int chunk0 = (chunk_base + c_local) * BT;
    if (chunk0 >= seq_len) break;
    const int len = min(BT, seq_len - chunk0);

    const T* in = tiles + (head_base + c_local) * Tile::kHalves;
    const float* uv_in = uv_tiles + (head_base + c_local) * Tile::kUv;
    const float raw_decay = decays[head_base + c_local];
    const float decay_all = fabsf(raw_decay);

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      if constexpr (kBf16Prec == Bf16Precision::kFp16) {
        if (signbit(raw_decay)) {
          RunScanStepExactBf16Fallback<T, DK, DV, BT, DVB, /*kRecomputeFromInputs=*/false>(
              smem_raw, pack, p, in, uv_in, decay_all, tok_base, chunk0, len, hv, v0, v_stride,
              tid, warp_id, lane);
          continue;
        }
        int thread_s_max = 0, s_flags = 0;
        for (int idx = tid; idx < DK * DVB; idx += kChunkedThreads) {
          const int r = idx / DVB, c = idx % DVB;
          const int abits = __float_as_int(fabsf(s.s_f[r * L::kSf + c]));
          thread_s_max = max(thread_s_max, abits);
          s_flags |= (abits == 0) ? 1 : ((abits < 0x33800000) ? 2 : 0);
        }
        bool thread_out_of_fp16 =
            thread_s_max > 0x43800000 || (thread_s_max != 0 && thread_s_max < 0x38800000) ||
            (s_flags & 2) != 0;
        for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
          const int t = idx / DVB, c = idx % DVB;
          if (t < len && __float_as_int(fabsf(uv_in[t * DV + v0 + c])) > 0x43800000) {
            thread_out_of_fp16 = true;
          }
        }
        const bool k2_out_of_fp16 = __syncthreads_or(thread_out_of_fp16) != 0;
        if (!k2_out_of_fp16) {
          const __half* in_fp16 = reinterpret_cast<const __half*>(in);
          const __half* wq_fp16 = in_fp16;
          const __half* p_fp16 = wq_fp16 + Tile::kW + Tile::kQg;
          const __half* kd_fp16 = p_fp16 + Tile::kP;
          const ScanSmem<__half, DK, DV, BT, DVB> s_fp16 =
              CarveScan<__half, DK, DV, BT, DVB>(smem_raw);
          for (int idx = tid; idx < 2 * BT * DK; idx += kChunkedThreads) {
            s_fp16.wq_h[(idx / DK) * L::kKh + idx % DK] = wq_fp16[idx];
          }
          for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
            s_fp16.p_h[(idx / BT) * L::kMh + idx % BT] = p_fp16[idx];
          }
          for (int idx = tid; idx < DK * DVB; idx += kChunkedThreads) {
            const int r = idx / DVB, c = idx % DVB;
            s_fp16.s_h[r * L::kSh + c] = __float2half(s_fp16.s_f[r * L::kSf + c]);
          }
          __syncthreads();
          SmemGemm<2 * BT, DVB, DK, false, false, false>(
              s_fp16.t_f, L::kF2, s_fp16.wq_h, L::kKh, s_fp16.s_h, L::kSh, warp_id, lane);
          __syncthreads();
          for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
            const int t = idx / DVB, c = idx % DVB;
            s_fp16.u_h[t * L::kBh + c] =
                __float2half(uv_in[t * DV + v0 + c] - s_fp16.t_f[t * L::kF2 + c]);
          }
          for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
            s_fp16.kd_h[(idx / DK) * L::kKh + idx % DK] = kd_fp16[idx];
          }
          __syncthreads();

          SmemGemm<BT, DVB, BT, false, false, false>(
              s_fp16.t_f, L::kF2, s_fp16.p_h, L::kMh, s_fp16.u_h, L::kBh, warp_id, lane);
          __syncthreads();
          for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
            const int t = idx / DVB, c = idx % DVB;
            if (t < len) {
              const int64_t tok = tok_base + chunk0 + t;
              pack.output[tok * v_stride + hv * DV + v0 + c] = FloatToElem<T>(
                  p.scale * (s_fp16.t_f[t * L::kF2 + c] + s_fp16.t_f[(BT + t) * L::kF2 + c]));
            }
          }

          SmemGemm<DK, DVB, BT, true, false, true, float>(
              s_fp16.s_f, L::kSf, s_fp16.kd_h, L::kKh, s_fp16.u_h, L::kBh, warp_id, lane,
              decay_all);
          __syncthreads();
          continue;
        }
        RunScanStepExactBf16Fallback<T, DK, DV, BT, DVB, /*kRecomputeFromInputs=*/true>(
            smem_raw, pack, p, in, uv_in, decay_all, tok_base, chunk0, len, hv, v0, v_stride, tid,
            warp_id, lane);
        continue;
      } else if constexpr (kBf16Prec == Bf16Precision::kFp32) {
        const T* wq_in = in;
        const T* p_in = wq_in + Tile::kW + Tile::kQg;
        const T* kd_in = p_in + Tile::kP;

        for (int idx = tid; idx < 2 * BT * DK; idx += kChunkedThreads) {
          s.wq_h[(idx / DK) * L::kKh + idx % DK] = wq_in[idx];
        }
        for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
          s.p_h[(idx / BT) * L::kMh + idx % BT] = p_in[idx];
        }
        __syncthreads();
        SmemGemmFp32State<2 * BT, DVB, DK, false>(s.t_f, L::kF2, s.wq_h, L::kKh, s.s_f, L::kSf,
                                                  s.s_h, L::kSh, warp_id, lane, tid);
        __syncthreads();
        float* u_f = reinterpret_cast<float*>(s.s_h);
        for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
          const int t = idx / DVB, c = idx % DVB;
          u_f[t * L::kSf + c] = uv_in[t * DV + v0 + c] - s.t_f[t * L::kF2 + c];
        }
        for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
          s.kd_h[(idx / DK) * L::kKh + idx % DK] = kd_in[idx];
        }
        __syncthreads();

        SmemGemmFp32State<BT, DVB, BT, false>(s.t_f, L::kF2, s.p_h, L::kMh, u_f, L::kSf, s.u_h,
                                              L::kBh, warp_id, lane, tid);
        __syncthreads();
        for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
          const int t = idx / DVB, c = idx % DVB;
          if (t < len) {
            const int64_t tok = tok_base + chunk0 + t;
            pack.output[tok * v_stride + hv * DV + v0 + c] =
                FloatToElem<T>(p.scale * (s.t_f[t * L::kF2 + c] + s.t_f[(BT + t) * L::kF2 + c]));
          }
        }

        for (int idx = tid; idx < DK * DVB; idx += kChunkedThreads) {
          s.s_f[(idx / DVB) * L::kSf + idx % DVB] *= decay_all;
        }
        __syncthreads();
        SmemGemmFp32State<DK, DVB, BT, true, kChunkedWarps, T, /*TA=*/true>(
            s.s_f, L::kSf, s.kd_h, L::kKh, u_f, L::kSf, s.u_h, L::kBh, warp_id, lane, tid);
        __syncthreads();
        continue;
      }
    }

    const T* wq_in = in;
    const T* p_in = wq_in + Tile::kW + Tile::kQg;
    const T* kd_in = p_in + Tile::kP;

    for (int idx = tid; idx < 2 * BT * DK; idx += kChunkedThreads) {
      s.wq_h[(idx / DK) * L::kKh + idx % DK] = wq_in[idx];
    }
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      s.p_h[(idx / BT) * L::kMh + idx % BT] = p_in[idx];
    }
    __syncthreads();

    for (int idx = tid; idx < DK * DVB; idx += kChunkedThreads) {
      const int r = idx / DVB, c = idx % DVB;
      s.s_h[r * L::kSh + c] = FloatToElem<T>(s.s_f[r * L::kSf + c]);
    }
    __syncthreads();
    SmemGemm<2 * BT, DVB, DK, false, false, false>(s.t_f, L::kF2, s.wq_h, L::kKh, s.s_h, L::kSh,
                                                   warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
      const int t = idx / DVB, c = idx % DVB;
      s.u_h[t * L::kBh + c] = FloatToElem<T>(uv_in[t * DV + v0 + c] - s.t_f[t * L::kF2 + c]);
    }
    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      s.kd_h[(idx / DK) * L::kKh + idx % DK] = kd_in[idx];
    }
    __syncthreads();

    // P U overwrites the W S half; Qg S is still live above it.
    SmemGemm<BT, DVB, BT, false, false, false>(s.t_f, L::kF2, s.p_h, L::kMh, s.u_h, L::kBh,
                                               warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
      const int t = idx / DVB, c = idx % DVB;
      if (t < len) {
        const int64_t tok = tok_base + chunk0 + t;
        pack.output[tok * v_stride + hv * DV + v0 + c] =
            FloatToElem<T>(p.scale * (s.t_f[t * L::kF2 + c] + s.t_f[(BT + t) * L::kF2 + c]));
      }
    }

    // S = decay S + Kd^T U in FP32 so small per-chunk decays do not round away between chunks.
    SmemGemm<DK, DVB, BT, true, false, true, float>(s.s_f, L::kSf, s.kd_h, L::kKh, s.u_h,
                                                    L::kBh, warp_id, lane, decay_all);
    __syncthreads();
  }

  // Written after every pass, not just the last: the next pass reads it back as its seed.
  for (int idx = tid; idx < DK * DVB; idx += kChunkedThreads) {
    const int r = idx / DVB, c = idx % DVB;
    const int64_t off = st_off + static_cast<int64_t>(v0 + c) * DK + r;
    const float v = s.s_f[r * L::kSf + c];
    if (carry != nullptr) carry[off] = v;
    if (last_pass && pack.final_state != nullptr) pack.final_state[off] = v;
  }
}

}  // namespace

template <typename T>
Status LaunchGatedDeltaNetSplit(const Descriptor& desc, const Plan& plan,
                                const VariantPack<T>& pack, float scale, cudaStream_t stream) {
  constexpr int DK = 128, DV = 128, BT = 64;
  static_assert(DK == 128 && DV == 128, "the split engine is specialised for 128-wide heads");

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
  p.decay_per_key_dim_flag = desc.decay_per_key_dim ? 1 : 0;
  p.bf16_precision = desc.bf16_precision;

  const int64_t longest = desc.batch > 0 ? (desc.total_tokens + desc.batch - 1) / desc.batch : 0;
  const int total_chunks = static_cast<int>((longest + BT - 1) / BT);
  if (total_chunks <= 0) return Status::OK();
  const int chunks_per_pass =
      std::min(plan.chunks_per_pass > 0 ? plan.chunks_per_pass : total_chunks, total_chunks);

  const int64_t tiles_per_pass =
      static_cast<int64_t>(desc.batch) * desc.num_heads_v * chunks_per_pass;
  float* uv_tiles = reinterpret_cast<float*>(pack.workspace);
  float* decays = uv_tiles + tiles_per_pass * PrepTile<DK, DV, BT>::kUv;
  // Reserved by SelectPlan only when the sequence takes more than one pass.
  float* carry = total_chunks > chunks_per_pass ? decays + tiles_per_pass : nullptr;
  float* after_floats = carry != nullptr ? carry + static_cast<int64_t>(desc.batch) *
                                                       desc.num_heads_v * DV * DK
                                         : decays + tiles_per_pass;
  T* tiles = reinterpret_cast<T*>(after_floats);

  const int dvb = plan.v_block;
  const size_t prep_smem = SplitPrepareSmemBytes(BT, DK, DV);
  const size_t scan_smem = SplitScanSmemBytes(BT, DK, dvb);

  auto launch_pass = [&]<Bf16Precision kPrec>(int base, int n, bool first, bool last) -> Status {
    static DynamicSmemConfig prep_smem_config;
    ORT_RETURN_IF_ERROR(SetMaxDynamicSmem(
        GatedDeltaNetPrepareKernel<T, DK, DV, BT, kPrec>, prep_smem, prep_smem_config));
    const dim3 prep_grid(desc.batch, desc.num_heads_v, n);
    GatedDeltaNetPrepareKernel<T, DK, DV, BT, kPrec>
        <<<prep_grid, kChunkedThreads, prep_smem, stream>>>(pack, p, tiles, uv_tiles, decays, base,
                                                            chunks_per_pass);

    const dim3 scan_grid(desc.batch, desc.num_heads_v, DV / dvb);
    if (dvb == 32) {
      static DynamicSmemConfig scan32_smem_config;
      ORT_RETURN_IF_ERROR(SetMaxDynamicSmem(
          GatedDeltaNetScanKernel<T, DK, DV, BT, 32, kPrec>, scan_smem, scan32_smem_config));
      GatedDeltaNetScanKernel<T, DK, DV, BT, 32, kPrec>
          <<<scan_grid, kChunkedThreads, scan_smem, stream>>>(pack, p, tiles, uv_tiles, decays,
                                                              carry, base, chunks_per_pass, n,
                                                              first, last);
    } else {
      static DynamicSmemConfig scan64_smem_config;
      ORT_RETURN_IF_ERROR(SetMaxDynamicSmem(
          GatedDeltaNetScanKernel<T, DK, DV, BT, 64, kPrec>, scan_smem, scan64_smem_config));
      GatedDeltaNetScanKernel<T, DK, DV, BT, 64, kPrec>
          <<<scan_grid, kChunkedThreads, scan_smem, stream>>>(pack, p, tiles, uv_tiles, decays,
                                                              carry, base, chunks_per_pass, n,
                                                              first, last);
    }
    return CUDA_CALL(cudaGetLastError());
  };

  for (int base = 0; base < total_chunks; base += chunks_per_pass) {
    const int n = std::min(chunks_per_pass, total_chunks - base);
    const bool first = base == 0;
    const bool last = base + n >= total_chunks;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      if (desc.bf16_precision == Bf16Precision::kFp16) {
        ORT_RETURN_IF_ERROR(launch_pass.template operator()<Bf16Precision::kFp16>(base, n, first, last));
        continue;
      }
      if (desc.bf16_precision == Bf16Precision::kBf16) {
        ORT_RETURN_IF_ERROR(launch_pass.template operator()<Bf16Precision::kBf16>(base, n, first, last));
        continue;
      }
    }
    ORT_RETURN_IF_ERROR(launch_pass.template operator()<Bf16Precision::kFp32>(base, n, first, last));
  }
  return Status::OK();
}

template Status LaunchGatedDeltaNetSplit<half>(const Descriptor&, const Plan&,
                                               const VariantPack<half>&, float, cudaStream_t);
template Status LaunchGatedDeltaNetSplit<__nv_bfloat16>(
    const Descriptor&, const Plan&, const VariantPack<__nv_bfloat16>&, float, cudaStream_t);

// The GEMM operands are 16-bit mma fragments, so SelectPlan never routes float here; the
// definition exists only to close the float instantiation of LaunchGatedDeltaNet.
template <>
Status LaunchGatedDeltaNetSplit<float>(const Descriptor&, const Plan&, const VariantPack<float>&,
                                       float, cudaStream_t) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "GatedDeltaNet: the split engine requires float16 or bfloat16 input");
}

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
