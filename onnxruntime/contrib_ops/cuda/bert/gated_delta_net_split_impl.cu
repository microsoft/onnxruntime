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
// The state is kept in fp16 rather than the fused engine's fp32, which is what brings K2
// under the 113.5 KB that lets two CTAs share an SM. fp16 (not bf16, which is what the
// reference implementations use) because the operator's q/k/v/output are fp16 already,
// qk_l2_norm bounds |k| <= 1 and beta = sigmoid(.) < 1, so the state stays around O(10)
// against fp16's 65504 ceiling, and fp16 carries three more mantissa bits than bf16 on
// what is an accumulator running the length of the sequence.

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
  static constexpr int kF1 = (DV > BT ? DV : BT) + 4;    // K1 fp32 scratch
  static constexpr int kF2 = (DVB > BT ? DVB : BT) + 4;  // K2 fp32 scratch
};

// ---------------------------------------------------------------------------
// K1 shared memory
// ---------------------------------------------------------------------------
template <int DK, int DV, int BT>
struct PrepSmem {
  __half* k_h;   // [BT][kKh]
  __half* q_h;   // [BT][kKh]
  __half* v_h;   // [BT][kVh]  beta v, then Uv
  __half* m_h;   // [BT][kMh]  M, then N (BuildTriInverse is done with M by then)
  __half* db_h;  // [BT][kMh]
  __half* ti_h;  // [BT][kMh]  (I+M)^-1
  float* t_f;    // [BT][kF1]
  float* gc;     // [BT]
  float* beta;   // [BT]
};

template <int DK, int DV, int BT>
__device__ __forceinline__ PrepSmem<DK, DV, BT> CarvePrep(char* raw) {
  using L = SplitLd<DK, DV, BT, 64>;
  PrepSmem<DK, DV, BT> s;
  float* f = reinterpret_cast<float*>(raw);
  s.t_f = f;
  f += BT * L::kF1;
  s.gc = f;
  f += BT;
  s.beta = f;
  f += BT;
  __half* h = reinterpret_cast<__half*>(f);
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

// Layout of one chunk's K1 output, in units of __half. Chunks are indexed
// ((b * Hv + hv) * chunks_per_pass + c), so K2's walk over c is a unit stride. W and Qg are
// adjacent so K2 can multiply both by the state in a single [2 BT x DK] GEMM.
template <int DK, int DV, int BT>
struct PrepTile {
  static constexpr int64_t kW = static_cast<int64_t>(BT) * DK;
  static constexpr int64_t kQg = static_cast<int64_t>(BT) * DK;
  static constexpr int64_t kUv = static_cast<int64_t>(BT) * DV;
  static constexpr int64_t kP = static_cast<int64_t>(BT) * BT;
  static constexpr int64_t kKd = static_cast<int64_t>(BT) * DK;
  static constexpr int64_t kHalves = kW + kQg + kUv + kP + kKd;
};

// ---------------------------------------------------------------------------
// K1: everything in a chunk that does not touch the recurrent state.
// One CTA per (sequence, v-head, chunk).
// ---------------------------------------------------------------------------
template <typename T, int DK, int DV, int BT>
__global__ __launch_bounds__(kChunkedThreads) void GatedDeltaNetPrepareKernel(
    VariantPack<T> pack, KernelParams p, __half* tiles, float* decays, int chunk_base,
    int chunks_per_pass) {
  using L = SplitLd<DK, DV, BT, 64>;
  using Tile = PrepTile<DK, DV, BT>;
  extern __shared__ char smem_raw[];
  const PrepSmem<DK, DV, BT> s = CarvePrep<DK, DV, BT>(smem_raw);

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

  for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
    const int t = idx / DK, d = idx % DK;
    const int64_t tok = tok_base + chunk0 + t;
    s.k_h[t * L::kKh + d] =
        (t < len) ? __float2half((float)pack.key[tok * k_stride + hk * DK + d]) : __float2half(0.f);
    s.q_h[t * L::kKh + d] =
        (t < len) ? __float2half((float)pack.query[tok * q_stride + hq * DK + d])
                  : __float2half(0.f);
  }
  for (int idx = tid; idx < BT * DV; idx += kChunkedThreads) {
    const int t = idx / DV, d = idx % DV;
    const int64_t tok = tok_base + chunk0 + t;
    s.v_h[t * L::kVh + d] =
        (t < len) ? __float2half((float)pack.value[tok * v_stride + hv * DV + d])
                  : __float2half(0.f);
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
  }
  __syncthreads();

  if (p.qk_l2_norm) {
    for (int t = warp_id; t < BT; t += kChunkedWarps) {
      float sq = 0.0f, sk = 0.0f;
      for (int d = lane; d < DK; d += 32) {
        const float qv = __half2float(s.q_h[t * L::kKh + d]);
        const float kv = __half2float(s.k_h[t * L::kKh + d]);
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
        s.q_h[t * L::kKh + d] = __float2half(__half2float(s.q_h[t * L::kKh + d]) * rq);
        s.k_h[t * L::kKh + d] = __float2half(__half2float(s.k_h[t * L::kKh + d]) * rk);
      }
    }
    __syncthreads();
  }

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
    s.m_h[t * L::kMh + sc] = __float2half(m);
  }
  __syncthreads();

  // The Neumann iterate lands on top of M: BuildTriInverse has consumed M by the time it
  // writes N, and the prepare kernel only fits two CTAs on an SM without the extra tile.
  BuildTriInverse<BT>(s.m_h, L::kMh, s.db_h, s.m_h, s.ti_h, L::kMh, s.t_f, L::kF1, warp_id, lane,
                      tid);

  const int64_t tile_base =
      (static_cast<int64_t>(b) * p.num_heads_v + hv) * chunks_per_pass + c_local;
  __half* out = tiles + tile_base * Tile::kHalves;
  __half* w_out = out;
  __half* qg_out = w_out + Tile::kW;
  __half* uv_out = qg_out + Tile::kQg;
  __half* p_out = uv_out + Tile::kUv;
  __half* kd_out = p_out + Tile::kP;

  // Uv = (I+M)^-1 (beta v). v_h holds v; scale it in place, then overwrite with Uv.
  for (int idx = tid; idx < BT * DV; idx += kChunkedThreads) {
    const int t = idx / DV, d = idx % DV;
    s.v_h[t * L::kVh + d] = __hmul(s.v_h[t * L::kVh + d], __float2half(s.beta[t]));
  }
  __syncthreads();
  SmemGemm<BT, DV, BT, false, false, false>(s.t_f, L::kF1, s.ti_h, L::kMh, s.v_h, L::kVh, warp_id,
                                            lane);
  __syncthreads();
  for (int idx = tid; idx < BT * DV; idx += kChunkedThreads) {
    uv_out[idx] = __float2half(s.t_f[(idx / DV) * L::kF1 + idx % DV]);
  }

  // Kd and Qg are just row scalings of k and q, so emit them before k is overwritten by the
  // row scaling W needs.
  for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
    const int t = idx / DK, d = idx % DK;
    kd_out[idx] = __hmul(s.k_h[t * L::kKh + d], __float2half(__expf(g_total - s.gc[t])));
    qg_out[idx] = __hmul(s.q_h[t * L::kKh + d], __float2half(__expf(s.gc[t])));
  }
  __syncthreads();

  // P = (q k^T) exp(gc_t - gc_s), inclusive lower. q_h is still unscaled here.
  SmemGemm<BT, BT, DK, false, true, false>(s.t_f, L::kF1, s.q_h, L::kKh, s.k_h, L::kKh, warp_id,
                                           lane);
  __syncthreads();
  for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
    const int t = idx / BT, sc = idx % BT;
    p_out[idx] = __float2half(
        (sc <= t && t < len) ? s.t_f[t * L::kF1 + sc] * __expf(s.gc[t] - s.gc[sc]) : 0.0f);
  }
  for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
    const int t = idx / DK, d = idx % DK;
    const float bw = needs_retrieval ? s.beta[t] * __expf(s.gc[t]) : 0.0f;
    s.k_h[t * L::kKh + d] = __float2half(__half2float(s.k_h[t * L::kKh + d]) * bw);
  }
  __syncthreads();
  // W = (I+M)^-1 diag(beta exp(gc)) k, with the row scaling folded into k in place.
  SmemGemm<BT, DK, BT, false, false, false>(s.t_f, L::kF1, s.ti_h, L::kMh, s.k_h, L::kKh, warp_id,
                                            lane);
  __syncthreads();
  for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
    w_out[idx] = __float2half(s.t_f[(idx / DK) * L::kF1 + idx % DK]);
  }

  if (tid == 0) decays[tile_base] = __expf(g_total);
}

// ---------------------------------------------------------------------------
// K2 shared memory. The state lives here in fp16 for the whole walk.
// ---------------------------------------------------------------------------
template <int DK, int DV, int BT, int DVB>
struct ScanSmem {
  __half* s_h;   // [DK][kSh]    recurrent state
  __half* wq_h;  // [2 BT][kKh]  W stacked on Qg
  __half* u_h;   // [BT][kBh]    Uv, then U in place
  __half* p_h;   // [BT][kMh]
  __half* kd_h;  // [BT][kKh]
  float* t_f;    // [DK][kF2], and DK == 2 BT holds both halves of the stacked GEMM
};

template <int DK, int DV, int BT, int DVB>
__device__ __forceinline__ ScanSmem<DK, DV, BT, DVB> CarveScan(char* raw) {
  using L = SplitLd<DK, DV, BT, DVB>;
  static_assert(DK == 2 * BT, "t_f must cover both halves of the stacked [2 BT x DVB] GEMM");
  ScanSmem<DK, DV, BT, DVB> s;
  float* f = reinterpret_cast<float*>(raw);
  s.t_f = f;
  f += DK * L::kF2;
  __half* h = reinterpret_cast<__half*>(f);
  s.s_h = h;
  h += DK * L::kSh;
  s.wq_h = h;
  h += 2 * BT * L::kKh;
  s.u_h = h;
  h += BT * L::kBh;
  s.p_h = h;
  h += BT * L::kMh;
  s.kd_h = h;
  return s;
}

// ---------------------------------------------------------------------------
// K2: the recurrence. One CTA per (sequence, v-head, v-block).
// ---------------------------------------------------------------------------
template <typename T, int DK, int DV, int BT, int DVB>
__global__ __launch_bounds__(kChunkedThreads) void GatedDeltaNetScanKernel(
    VariantPack<T> pack, KernelParams p, const __half* tiles, const float* decays, float* carry,
    int chunk_base, int chunks_per_pass, int n_chunks, bool first_pass, bool last_pass) {
  using L = SplitLd<DK, DV, BT, DVB>;
  using Tile = PrepTile<DK, DV, BT>;
  extern __shared__ char smem_raw[];
  const ScanSmem<DK, DV, BT, DVB> s = CarveScan<DK, DV, BT, DVB>(smem_raw);

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
    s.s_h[r * L::kSh + c] =
        seed != nullptr
            ? __float2half(seed[st_off + static_cast<int64_t>(v0 + c) * DK + r])
            : __float2half(0.f);
  }
  __syncthreads();

  const int64_t head_base =
      (static_cast<int64_t>(b) * p.num_heads_v + hv) * chunks_per_pass;

  for (int c_local = 0; c_local < n_chunks; ++c_local) {
    const int chunk0 = (chunk_base + c_local) * BT;
    if (chunk0 >= seq_len) break;
    const int len = min(BT, seq_len - chunk0);

    const __half* in = tiles + (head_base + c_local) * Tile::kHalves;
    const __half* wq_in = in;
    const __half* uv_in = wq_in + Tile::kW + Tile::kQg;
    const __half* p_in = uv_in + Tile::kUv;
    const __half* kd_in = p_in + Tile::kP;

    for (int idx = tid; idx < 2 * BT * DK; idx += kChunkedThreads) {
      s.wq_h[(idx / DK) * L::kKh + idx % DK] = wq_in[idx];
    }
    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      s.kd_h[(idx / DK) * L::kKh + idx % DK] = kd_in[idx];
    }
    for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
      const int t = idx / DVB, d = idx % DVB;
      s.u_h[t * L::kBh + d] = uv_in[t * DV + v0 + d];
    }
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      s.p_h[(idx / BT) * L::kMh + idx % BT] = p_in[idx];
    }
    __syncthreads();

    // One GEMM for both products of the state: rows [0, BT) are W S, rows [BT, 2 BT) are Qg S.
    SmemGemm<2 * BT, DVB, DK, false, false, false>(s.t_f, L::kF2, s.wq_h, L::kKh, s.s_h, L::kSh,
                                                   warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * DVB; idx += kChunkedThreads) {
      const int t = idx / DVB, c = idx % DVB;
      s.u_h[t * L::kBh + c] =
          __float2half(__half2float(s.u_h[t * L::kBh + c]) - s.t_f[t * L::kF2 + c]);
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
            (T)(p.scale *
                (s.t_f[t * L::kF2 + c] + s.t_f[(BT + t) * L::kF2 + c]));
      }
    }

    // S = decay S + Kd^T U, straight into the fp16 state: the mma accumulator is float
    // either way, so this is the same single rounding as staging it through t_f, and it
    // needs no barrier against the output epilogue above, which only reads t_f.
    SmemGemm<DK, DVB, BT, true, false, true, __half>(s.s_h, L::kSh, s.kd_h, L::kKh, s.u_h,
                                                     L::kBh, warp_id, lane,
                                                     decays[head_base + c_local]);
    __syncthreads();
  }

  // Written after every pass, not just the last: the next pass reads it back as its seed.
  for (int idx = tid; idx < DK * DVB; idx += kChunkedThreads) {
    const int r = idx / DVB, c = idx % DVB;
    const int64_t off = st_off + static_cast<int64_t>(v0 + c) * DK + r;
    const float v = __half2float(s.s_h[r * L::kSh + c]);
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

  const int64_t longest = desc.batch > 0 ? (desc.total_tokens + desc.batch - 1) / desc.batch : 0;
  const int total_chunks = static_cast<int>((longest + BT - 1) / BT);
  if (total_chunks <= 0) return Status::OK();
  const int chunks_per_pass =
      std::min(plan.chunks_per_pass > 0 ? plan.chunks_per_pass : total_chunks, total_chunks);

  const int64_t tiles_per_pass =
      static_cast<int64_t>(desc.batch) * desc.num_heads_v * chunks_per_pass;
  __half* tiles = reinterpret_cast<__half*>(pack.workspace);
  float* decays = reinterpret_cast<float*>(
      tiles + tiles_per_pass * PrepTile<DK, DV, BT>::kHalves);
  // Reserved by SelectPlan only when the sequence takes more than one pass.
  float* carry = total_chunks > chunks_per_pass ? decays + tiles_per_pass : nullptr;

  const int dvb = plan.v_block;
  const size_t prep_smem = SplitPrepareSmemBytes(BT, DK, DV);
  const size_t scan_smem = SplitScanSmemBytes(BT, DK, dvb);

  static DynamicSmemConfig prep_smem_config;
  ORT_RETURN_IF_ERROR(SetMaxDynamicSmem(GatedDeltaNetPrepareKernel<T, DK, DV, BT>, prep_smem,
                                        prep_smem_config));

  for (int base = 0; base < total_chunks; base += chunks_per_pass) {
    const int n = std::min(chunks_per_pass, total_chunks - base);
    const bool first = base == 0;
    const bool last = base + n >= total_chunks;

    const dim3 prep_grid(desc.batch, desc.num_heads_v, n);
    GatedDeltaNetPrepareKernel<T, DK, DV, BT>
        <<<prep_grid, kChunkedThreads, prep_smem, stream>>>(pack, p, tiles, decays, base,
                                                            chunks_per_pass);

    const dim3 scan_grid(desc.batch, desc.num_heads_v, DV / dvb);
    if (dvb == 32) {
      static DynamicSmemConfig scan32_smem_config;
      ORT_RETURN_IF_ERROR(SetMaxDynamicSmem(GatedDeltaNetScanKernel<T, DK, DV, BT, 32>,
                                            scan_smem, scan32_smem_config));
      GatedDeltaNetScanKernel<T, DK, DV, BT, 32>
          <<<scan_grid, kChunkedThreads, scan_smem, stream>>>(pack, p, tiles, decays, carry, base,
                                                              chunks_per_pass, n, first, last);
    } else {
      static DynamicSmemConfig scan64_smem_config;
      ORT_RETURN_IF_ERROR(SetMaxDynamicSmem(GatedDeltaNetScanKernel<T, DK, DV, BT, 64>,
                                            scan_smem, scan64_smem_config));
      GatedDeltaNetScanKernel<T, DK, DV, BT, 64>
          <<<scan_grid, kChunkedThreads, scan_smem, stream>>>(pack, p, tiles, decays, carry, base,
                                                              chunks_per_pass, n, first, last);
    }
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  }
  return Status::OK();
}

template Status LaunchGatedDeltaNetSplit<half>(const Descriptor&, const Plan&,
                                               const VariantPack<half>&, float, cudaStream_t);

// The GEMM operands are fp16 mma fragments, so SelectPlan never routes float here; the
// definition exists only to close the float instantiation of LaunchGatedDeltaNet.
template <>
Status LaunchGatedDeltaNetSplit<float>(const Descriptor&, const Plan&, const VariantPack<float>&,
                                       float, cudaStream_t) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "GatedDeltaNet: the split engine requires float16 input");
}

template <>
Status LaunchGatedDeltaNetSplit<__nv_bfloat16>(
    const Descriptor&, const Plan&, const VariantPack<__nv_bfloat16>&, float, cudaStream_t) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "GatedDeltaNet: the split engine requires float16 input");
}

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
