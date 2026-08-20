// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GatedDeltaNet CUDA kernels.
//
// Two engines, chosen by gated_delta_net_plan.cc:
//
//  * chunked  -- tensor-core chunked scan for prefill. One CTA owns one
//                (sequence, v-head, v-block) and walks the sequence in BT=64 token chunks,
//                keeping the recurrent state in shared memory for the whole walk. One
//                launch, no per-chunk state materialisation.
//  * recurrent -- scalar per-token recurrence for decode / MTP verify, and the only engine
//                that can emit the per-token state checkpoint series.
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
//     disappears, and since Wbar is only a row scaling of k, Wbar is never materialised
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

#include <algorithm>
#include <mutex>

#include "contrib_ops/cuda/bert/gated_delta_net_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace gated_delta_net {

namespace {

constexpr int kDVB = 64;  // v columns owned by one CTA in the chunked engine
constexpr int kChunkedThreads = 512;
constexpr int kChunkedWarps = kChunkedThreads / 32;

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

__device__ __forceinline__ float SoftPlus(float x) {
  // log1p(exp(x)) without overflow.
  return x > 20.0f ? x : __logf(1.0f + __expf(x));
}

__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

// Resolve the [start, start+len) token range of sequence `b`, guarding malformed device
// offsets so a bad producer cannot steer an out-of-bounds access.
__device__ __forceinline__ void SequenceRange(const int32_t* cu_seqlens, int b, int64_t total,
                                              int64_t uniform_len, int64_t* start, int64_t* len) {
  if (cu_seqlens == nullptr) {
    *start = static_cast<int64_t>(b) * uniform_len;
    *len = uniform_len;
  } else {
    int64_t s = static_cast<int64_t>(cu_seqlens[b]);
    int64_t e = static_cast<int64_t>(cu_seqlens[b + 1]);
    s = s < 0 ? 0 : (s > total ? total : s);
    e = e < 0 ? 0 : (e > total ? total : e);
    *start = s;
    *len = e > s ? e - s : 0;
  }
}

// ---------------------------------------------------------------------------
// mma.sync.m16n8k16 helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t PackHalf2(__half lo, __half hi) {
  return static_cast<uint32_t>(__half_as_ushort(lo)) |
         (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
}

__device__ __forceinline__ void MmaM16N8K16(float (&d)[4], const uint32_t (&a)[4],
                                            const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A-fragment for m16n8k16: lane l holds rows (l>>2, l>>2 + 8) and k-pairs at (l&3)*2.
template <bool Transposed>
__device__ __forceinline__ void LoadFragA(uint32_t (&a)[4], const __half* A, int lda, int m0,
                                          int k0, int lane) {
  const int gid = lane >> 2, tig = lane & 3;
  const int r0 = m0 + gid, r1 = m0 + gid + 8;
  const int c0 = k0 + tig * 2, c1 = k0 + tig * 2 + 8;
  if (Transposed) {
    a[0] = PackHalf2(A[c0 * lda + r0], A[(c0 + 1) * lda + r0]);
    a[1] = PackHalf2(A[c0 * lda + r1], A[(c0 + 1) * lda + r1]);
    a[2] = PackHalf2(A[c1 * lda + r0], A[(c1 + 1) * lda + r0]);
    a[3] = PackHalf2(A[c1 * lda + r1], A[(c1 + 1) * lda + r1]);
  } else {
    a[0] = *reinterpret_cast<const uint32_t*>(A + r0 * lda + c0);
    a[1] = *reinterpret_cast<const uint32_t*>(A + r1 * lda + c0);
    a[2] = *reinterpret_cast<const uint32_t*>(A + r0 * lda + c1);
    a[3] = *reinterpret_cast<const uint32_t*>(A + r1 * lda + c1);
  }
}

// B-fragment for m16n8k16 (col layout): lane l holds column (l>>2) and k-pairs at (l&3)*2.
template <bool Transposed>
__device__ __forceinline__ void LoadFragB(uint32_t (&b)[2], const __half* B, int ldb, int k0,
                                          int n0, int lane) {
  const int gid = lane >> 2, tig = lane & 3;
  const int col = n0 + gid;
  const int r0 = k0 + tig * 2, r1 = k0 + tig * 2 + 8;
  if (Transposed) {
    b[0] = *reinterpret_cast<const uint32_t*>(B + col * ldb + r0);
    b[1] = *reinterpret_cast<const uint32_t*>(B + col * ldb + r1);
  } else {
    b[0] = PackHalf2(B[r0 * ldb + col], B[(r0 + 1) * ldb + col]);
    b[1] = PackHalf2(B[r1 * ldb + col], B[(r1 + 1) * ldb + col]);
  }
}

// C[M][N] (float, ldc) = A[M][K] @ B[K][N] (+ C if Accumulate).
//
// Each warp owns one m-tile (16 rows) and a strided subset of the n-tiles, so the A
// fragment is loaded once per k-step and reused across every n-tile the warp owns.
template <int M, int N, int K, bool TA, bool TB, bool Accumulate>
__device__ __forceinline__ void SmemGemm(float* C, int ldc, const __half* A, int lda,
                                         const __half* B, int ldb, int warp_id, int lane) {
  constexpr int kMTiles = M / 16;
  constexpr int kNTiles = N / 8;
  constexpr int kWarpsPerM = kChunkedWarps > kMTiles ? kChunkedWarps / kMTiles : 1;
  constexpr int kMGroups = kChunkedWarps / kWarpsPerM;
  constexpr int kNPerWarp = (kNTiles + kWarpsPerM - 1) / kWarpsPerM;

  const int m_group = warp_id / kWarpsPerM;
  const int n_group = warp_id % kWarpsPerM;
  const int gid = lane >> 2, tig = lane & 3;

#pragma unroll 1
  for (int mt = m_group; mt < kMTiles; mt += kMGroups) {
    const int m0 = mt * 16;
    const int r0 = m0 + gid, r1 = m0 + gid + 8;
    float acc[kNPerWarp][4];

#pragma unroll
    for (int i = 0; i < kNPerWarp; ++i) {
      const int nt = n_group + i * kWarpsPerM;
      if (nt < kNTiles) {
        const int c0 = nt * 8 + tig * 2;
        acc[i][0] = Accumulate ? C[r0 * ldc + c0] : 0.0f;
        acc[i][1] = Accumulate ? C[r0 * ldc + c0 + 1] : 0.0f;
        acc[i][2] = Accumulate ? C[r1 * ldc + c0] : 0.0f;
        acc[i][3] = Accumulate ? C[r1 * ldc + c0 + 1] : 0.0f;
      }
    }

    for (int k0 = 0; k0 < K; k0 += 16) {
      uint32_t a[4];
      LoadFragA<TA>(a, A, lda, m0, k0, lane);
#pragma unroll
      for (int i = 0; i < kNPerWarp; ++i) {
        const int nt = n_group + i * kWarpsPerM;
        if (nt < kNTiles) {
          uint32_t b[2];
          LoadFragB<TB>(b, B, ldb, k0, nt * 8, lane);
          MmaM16N8K16(acc[i], a, b);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < kNPerWarp; ++i) {
      const int nt = n_group + i * kWarpsPerM;
      if (nt < kNTiles) {
        const int c0 = nt * 8 + tig * 2;
        C[r0 * ldc + c0] = acc[i][0];
        C[r0 * ldc + c0 + 1] = acc[i][1];
        C[r1 * ldc + c0] = acc[i][2];
        C[r1 * ldc + c0 + 1] = acc[i][3];
      }
    }
  }
}

template <int BT>
__device__ __forceinline__ void InclusiveScanBT(float* x, int tid) {
  static_assert(BT == 32 || BT == 64, "chunk length must be 32 or 64");
  if (tid >= 32) return;
  if (BT == 32) {
    float a = x[tid];
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      float ta = __shfl_up_sync(0xffffffff, a, off);
      if (tid >= off) a += ta;
    }
    x[tid] = a;
    return;
  }
  float a = x[tid], b = x[tid + 32];
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    float ta = __shfl_up_sync(0xffffffff, a, off);
    float tb = __shfl_up_sync(0xffffffff, b, off);
    if (tid >= off) {
      a += ta;
      b += tb;
    }
  }
  const float total = __shfl_sync(0xffffffff, a, 31);
  x[tid] = a;
  x[tid + 32] = b + total;
}

template <int DK, int BT>
struct ChunkedSmem {
  __half* k_h;   // [BT][LdKh]  k, scaled in place to Kd late in the chunk
  __half* q_h;   // [BT][LdKh]  q, scaled in place to Qg late in the chunk
  __half* v_h;   // [BT][LdVh]
  __half* u_h;   // [BT][LdVh]  R, then U
  __half* m_h;   // [BT][LdMh]  M, then P
  __half* s_h;   // [DK][LdVh]  fp16 operand copy of the state
  __half* db_h;  // [BT][LdMh]  block-diagonal inverse of (I+M)
  __half* nb_h;  // [BT][LdMh]  N = Dinv M with the diagonal blocks zeroed
  __half* ti_h;  // [BT][LdMh]  Neumann iterate, finally (I+M)^-1
  float* s_f;    // [DK][LdVf]  fp32 state
  float* t_f;    // [BT][LdVf]  gemm scratch, also used as [BT][BT]
  float* gc;     // [BT]
  float* beta;   // [BT]
};

template <int DK, int BT>
__device__ __forceinline__ ChunkedSmem<DK, BT> CarveChunked(char* raw) {
  using L = Ld<DK, BT>;
  ChunkedSmem<DK, BT> s;
  float* f = reinterpret_cast<float*>(raw);
  s.s_f = f;
  f += DK * L::kVf;
  s.t_f = f;
  f += BT * L::kVf;
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
  s.u_h = h;
  h += BT * L::kVh;
  s.m_h = h;
  h += BT * L::kMh;
  s.s_h = h;
  h += DK * L::kVh;
  s.db_h = h;
  h += BT * L::kMh;
  s.nb_h = h;
  h += BT * L::kMh;
  s.ti_h = h;
  h += BT * L::kMh;
  return s;
}

// Tinv = (I + M)^-1 for the strictly lower BT x BT M in s.m_h, left in s.ti_h. Exact.
template <int DK, int BT>
__device__ __forceinline__ void BuildTriInverse(const ChunkedSmem<DK, BT>& s, int warp_id,
                                                int lane, int tid) {
  using L = Ld<DK, BT>;
  for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
    s.db_h[(idx / BT) * L::kMh + idx % BT] = __float2half(0.0f);
  }
  __syncthreads();

  // Column `lane` of the inverse of diagonal block `warp_id`, by forward substitution.
  // Lane-local, so no synchronisation inside.
  if (warp_id < BT / 16 && lane < 16) {
    const int base = warp_id * 16;
    float col[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) col[i] = 0.0f;
    col[lane] = 1.0f;
    for (int i = lane + 1; i < 16; ++i) {
      float acc = 0.0f;
      for (int j = lane; j < i; ++j) {
        acc += __half2float(s.m_h[(base + i) * L::kMh + base + j]) * col[j];
      }
      col[i] = -acc;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      s.db_h[(base + i) * L::kMh + base + lane] = __float2half(col[i]);
    }
  }
  __syncthreads();

  SmemGemm<BT, BT, BT, false, false, false>(s.t_f, L::kVf, s.db_h, L::kMh, s.m_h, L::kMh,
                                            warp_id, lane);
  __syncthreads();
  for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
    const int r = idx / BT, c = idx % BT;
    const bool same_block = (r >> 4) == (c >> 4);
    const float n = same_block ? 0.0f : s.t_f[r * L::kVf + c];
    s.nb_h[r * L::kMh + c] = __float2half(n);
    s.ti_h[r * L::kMh + c] = __float2half((r == c ? 1.0f : 0.0f) - n);  // Z1 = I - N
  }
  __syncthreads();

  // N is strictly block lower over BT/16 levels, so N^(BT/16) = 0 and the alternating
  // series terminates exactly; Horner needs BT/16 - 2 steps after Z1 = I - N.
#pragma unroll
  for (int it = 0; it < BT / 16 - 2; ++it) {
    SmemGemm<BT, BT, BT, false, false, false>(s.t_f, L::kVf, s.nb_h, L::kMh, s.ti_h, L::kMh,
                                              warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      const int r = idx / BT, c = idx % BT;
      s.ti_h[r * L::kMh + c] = __float2half((r == c ? 1.0f : 0.0f) - s.t_f[r * L::kVf + c]);
    }
    __syncthreads();
  }

  SmemGemm<BT, BT, BT, false, false, false>(s.t_f, L::kVf, s.ti_h, L::kMh, s.db_h, L::kMh,
                                            warp_id, lane);
  __syncthreads();
  for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
    const int r = idx / BT, c = idx % BT;
    s.ti_h[r * L::kMh + c] = __float2half(s.t_f[r * L::kVf + c]);
  }
  __syncthreads();
}

// Effective per-token decay and beta, after the optional fused activations.
__device__ __forceinline__ float EffectiveDecay(float raw, const float* a_log, const float* dt_bias,
                                                int h, GateActivation act) {
  if (act == GateActivation::kQwen) {
    const float bias = dt_bias != nullptr ? dt_bias[h] : 0.0f;
    const float scale = a_log != nullptr ? -__expf(a_log[h]) : -1.0f;
    return scale * SoftPlus(raw + bias);
  }
  return raw;
}

__device__ __forceinline__ float EffectiveBeta(float raw, BetaActivation act) {
  return act == BetaActivation::kSigmoid ? Sigmoid(raw) : raw;
}

struct KernelParams {
  int64_t total_tokens;
  int64_t uniform_len;
  int num_heads_q;
  int num_heads_k;
  int num_heads_v;
  float scale;
  GateActivation gate_activation;
  BetaActivation beta_activation;
  UpdateRule update_rule;
  bool qk_l2_norm;
  int state_checkpoints;
  int batch;
  int decay_per_key_dim_flag;
};

// ---------------------------------------------------------------------------
// Chunked engine
// ---------------------------------------------------------------------------
template <typename T, int DK, int DV, int BT>
__global__ __launch_bounds__(kChunkedThreads) void GatedDeltaNetChunkedKernel(
    VariantPack<T> pack, KernelParams p) {
  using L = Ld<DK, BT>;
  extern __shared__ char smem_raw[];
  const ChunkedSmem<DK, BT> s = CarveChunked<DK, BT>(smem_raw);

  const int b = blockIdx.x, hv = blockIdx.y, v0 = blockIdx.z * kDVB;
  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  const int hq = hv * p.num_heads_q / p.num_heads_v;
  const int hk = hv * p.num_heads_k / p.num_heads_v;

  int64_t tok_base, seq_len64;
  SequenceRange(pack.cu_seqlens, b, p.total_tokens, p.uniform_len, &tok_base, &seq_len64);
  const int seq_len = static_cast<int>(seq_len64);

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

    for (int idx = tid; idx < DK * kDVB; idx += kChunkedThreads) {
      const int r = idx / kDVB, c = idx % kDVB;
      s.s_h[r * L::kVh + c] = __float2half(s.s_f[r * L::kVf + c]);
    }
    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      const int t = idx / DK, d = idx % DK;
      const int64_t tok = tok_base + chunk0 + t;
      s.k_h[t * L::kKh + d] =
          (t < len) ? __float2half((float)pack.key[tok * k_stride + hk * DK + d]) : __float2half(0.f);
      s.q_h[t * L::kKh + d] =
          (t < len) ? __float2half((float)pack.query[tok * q_stride + hq * DK + d]) : __float2half(0.f);
    }
    for (int idx = tid; idx < BT * kDVB; idx += kChunkedThreads) {
      const int t = idx / kDVB, d = idx % kDVB;
      const int64_t tok = tok_base + chunk0 + t;
      s.v_h[t * L::kVh + d] =
          (t < len) ? __float2half((float)pack.value[tok * v_stride + hv * DV + v0 + d])
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
        if (!needs_retrieval) be = pack.beta != nullptr ? be : 1.0f;
      }
      s.gc[tid] = g;
      s.beta[tid] = be;
    }
    __syncthreads();

    if (p.qk_l2_norm) {
      // One warp per token row; DK=128 so each lane holds four elements.
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

    SmemGemm<BT, BT, DK, false, true, false>(s.t_f, L::kVf, s.k_h, L::kKh, s.k_h, L::kKh, warp_id,
                                             lane);
    __syncthreads();
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      const int t = idx / BT, sc = idx % BT;
      const float m = (sc < t && needs_retrieval)
                          ? s.beta[t] * s.t_f[t * L::kVf + sc] * __expf(s.gc[t] - s.gc[sc])
                          : 0.0f;
      s.m_h[t * L::kMh + sc] = __float2half(m);
    }
    __syncthreads();

    BuildTriInverse<DK, BT>(s, warp_id, lane, tid);

    SmemGemm<BT, kDVB, DK, false, false, false>(s.t_f, L::kVf, s.k_h, L::kKh, s.s_h, L::kVh,
                                                warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * kDVB; idx += kChunkedThreads) {
      const int t = idx / kDVB, c = idx % kDVB;
      const float vv = __half2float(s.v_h[t * L::kVh + c]);
      const float r = needs_retrieval ? vv - __expf(s.gc[t]) * s.t_f[t * L::kVf + c] : vv;
      s.u_h[t * L::kVh + c] = __float2half(s.beta[t] * r);
    }
    __syncthreads();
    SmemGemm<BT, kDVB, BT, false, false, false>(s.t_f, L::kVf, s.ti_h, L::kMh, s.u_h, L::kVh,
                                                warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * kDVB; idx += kChunkedThreads) {
      const int t = idx / kDVB, c = idx % kDVB;
      s.u_h[t * L::kVh + c] = __float2half(s.t_f[t * L::kVf + c]);
    }
    __syncthreads();

    SmemGemm<BT, BT, DK, false, true, false>(s.t_f, L::kVf, s.q_h, L::kKh, s.k_h, L::kKh, warp_id,
                                             lane);
    __syncthreads();
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      const int t = idx / BT, sc = idx % BT;
      s.m_h[t * L::kMh + sc] = __float2half(
          (sc <= t && t < len) ? s.t_f[t * L::kVf + sc] * __expf(s.gc[t] - s.gc[sc]) : 0.0f);
    }
    for (int idx = tid; idx < BT * DK; idx += kChunkedThreads) {
      const int t = idx / DK, d = idx % DK;
      s.k_h[t * L::kKh + d] =
          __hmul(s.k_h[t * L::kKh + d], __float2half(__expf(g_total - s.gc[t])));
      s.q_h[t * L::kKh + d] = __hmul(s.q_h[t * L::kKh + d], __float2half(__expf(s.gc[t])));
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
            (T)(p.scale * s.t_f[t * L::kVf + c]);
      }
    }

    const float decay_all = __expf(g_total);
    for (int idx = tid; idx < DK * kDVB; idx += kChunkedThreads) {
      s.s_f[(idx / kDVB) * L::kVf + idx % kDVB] *= decay_all;
    }
    __syncthreads();
    SmemGemm<DK, kDVB, BT, true, false, true>(s.s_f, L::kVf, s.k_h, L::kKh, s.u_h, L::kVh, warp_id,
                                              lane);
    __syncthreads();
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

template <typename T, int DK, int kWarps>
__global__ __launch_bounds__(32 * kWarps) void GatedDeltaNetDecodeWarpKernel(
    VariantPack<T> pack, KernelParams p, int d_v) {
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

  const int64_t q_stride = static_cast<int64_t>(p.num_heads_q) * DK;
  const int64_t k_stride = static_cast<int64_t>(p.num_heads_k) * DK;
  const int64_t v_stride = static_cast<int64_t>(p.num_heads_v) * d_v;
  const int64_t st_off =
      (static_cast<int64_t>(b) * p.num_heads_v + hv) * static_cast<int64_t>(d_v) * DK +
      static_cast<int64_t>(col) * DK;
  const int64_t slot_stride =
      static_cast<int64_t>(p.batch) * p.num_heads_v * static_cast<int64_t>(d_v) * DK;

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
      decay_scalar = __expf(
          EffectiveDecay(pack.decay[gi], pack.a_log, pack.dt_bias, hv, p.gate_activation));
    }
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      if (needs_decay && decay_per_key_dim) {
        const int k = lane + i * 32;
        const float raw = pack.decay[(tok * p.num_heads_v + hv) * DK + k];
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
        pack.beta != nullptr ? EffectiveBeta(pack.beta[gi], p.beta_activation) : 1.0f;
    const float vv = static_cast<float>(pack.value[tok * v_stride + hv * d_v + col]);
    const float delta = beta_t * (vv - r);

    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      s[i] += kv[i] * delta;
      o += s[i] * qv[i];
    }
    o = WarpReduceK<DK>(o);
    if (lane == 0) {
      pack.output[tok * v_stride + hv * d_v + col] = static_cast<T>(p.scale * o);
    }

    // Checkpoints are right-aligned: the last slot holds the state after the final token, so
    // slot W-1-k is the state after the k-th token from the end. Slots this call does not
    // write are left unspecified because the buffer may alias live state.
    const int slot = t + p.state_checkpoints - seq_len;
    if (pack.checkpoints != nullptr && slot >= 0) {
#pragma unroll
      for (int i = 0; i < kRegs; ++i) {
        pack.checkpoints[static_cast<int64_t>(slot) * slot_stride + st_off + lane + i * 32] = s[i];
      }
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
template <typename T, int kThreads>
__global__ __launch_bounds__(kThreads) void GatedDeltaNetRecurrentKernel(VariantPack<T> pack,
                                                                         KernelParams p, int d_k,
                                                                         int d_v) {
  extern __shared__ float rsmem[];
  float* S = rsmem;             // [d_k][d_v]
  float* kbuf = S + d_k * d_v;  // [d_k]
  float* qbuf = kbuf + d_k;     // [d_k]
  float* rbuf = qbuf + d_k;     // [d_v]
  float* gbuf = rbuf + d_v;     // [d_k] per-key-dim decay

  const int b = blockIdx.x, hv = blockIdx.y;
  const int tid = threadIdx.x;
  const int hq = hv * p.num_heads_q / p.num_heads_v;
  const int hk = hv * p.num_heads_k / p.num_heads_v;

  int64_t tok_base, seq_len64;
  SequenceRange(pack.cu_seqlens, b, p.total_tokens, p.uniform_len, &tok_base, &seq_len64);
  const int seq_len = static_cast<int>(seq_len64);

  const int64_t q_stride = static_cast<int64_t>(p.num_heads_q) * d_k;
  const int64_t k_stride = static_cast<int64_t>(p.num_heads_k) * d_k;
  const int64_t v_stride = static_cast<int64_t>(p.num_heads_v) * d_v;
  const int64_t st_off = (static_cast<int64_t>(b) * p.num_heads_v + hv) *
                         static_cast<int64_t>(d_v) * d_k;
  const int64_t slot_stride = static_cast<int64_t>(p.batch) * p.num_heads_v * d_v * d_k;

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
        const float raw = pack.decay[(tok * p.num_heads_v + hv) * d_k + i];
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
            ? __expf(EffectiveDecay(pack.decay[gi], pack.a_log, pack.dt_bias, hv,
                                    p.gate_activation))
            : 1.0f;
    const float beta_t =
        pack.beta != nullptr ? EffectiveBeta(pack.beta[gi], p.beta_activation) : 1.0f;

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

    // Checkpoints are right-aligned: the last slot holds the state after the final token.
    // Slots this call does not write are left unspecified on purpose: the buffer may alias
    // live state.
    const int slot = t + p.state_checkpoints - seq_len;
    if (pack.checkpoints != nullptr && slot >= 0) {
      for (int idx = tid; idx < d_k * d_v; idx += kThreads) {
        const int r = idx / d_v, c = idx % d_v;
        pack.checkpoints[static_cast<int64_t>(slot) * slot_stride + st_off +
                         static_cast<int64_t>(c) * d_k + r] = S[idx];
      }
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

// Raise the opt-in shared-memory maximum once per (device, kernel) to the device limit.
// Setting it per call to the current request lets a concurrent smaller launch lower the
// bound underneath a larger one.
template <typename KernelT>
Status SetMaxDynamicSmemOnce(KernelT kernel, size_t device_max, std::once_flag& flag) {
  Status status = Status::OK();
  std::call_once(flag, [&]() {
    cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(device_max));
    if (err != cudaSuccess) {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GatedDeltaNet: cudaFuncSetAttribute failed: ",
                               cudaGetErrorString(err));
    }
  });
  return status;
}

}  // namespace

template <typename T>
Status LaunchGatedDeltaNet(const Descriptor& desc, const Plan& plan, const VariantPack<T>& pack,
                           float scale, int max_threads_per_block, cudaStream_t stream) {
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
  p.state_checkpoints = desc.state_checkpoints;
  p.batch = desc.batch;
  p.decay_per_key_dim_flag = desc.decay_per_key_dim ? 1 : 0;

  if (plan.engine == Engine::kChunked) {
    const dim3 grid(desc.batch, desc.num_heads_v, desc.head_size_v / kDVB);
    // BT=32 halves the [BT x BT] and per-token tiles, which is what lets the chunked
    // engine fit devices with a 99 KB opt-in limit (SM120) instead of SM90's 227 KB.
    if (plan.chunk_size == 32) {
      static std::once_flag once32;
      ORT_RETURN_IF_ERROR(SetMaxDynamicSmemOnce(GatedDeltaNetChunkedKernel<T, 128, 128, 32>,
                                                plan.smem_bytes, once32));
      GatedDeltaNetChunkedKernel<T, 128, 128, 32>
          <<<grid, kChunkedThreads, plan.smem_bytes, stream>>>(pack, p);
    } else {
      static std::once_flag once64;
      ORT_RETURN_IF_ERROR(SetMaxDynamicSmemOnce(GatedDeltaNetChunkedKernel<T, 128, 128, 64>,
                                                plan.smem_bytes, once64));
      GatedDeltaNetChunkedKernel<T, 128, 128, 64>
          <<<grid, kChunkedThreads, plan.smem_bytes, stream>>>(pack, p);
    }
    return CUDA_CALL(cudaGetLastError());
  }

  constexpr int kRecurrentThreads = 256;
  constexpr int kDecodeWarps = kRecurrentThreads / 32;

  if (plan.warp_specialized) {
    const dim3 grid(desc.batch, desc.num_heads_v,
                    (desc.head_size_v + kDecodeWarps - 1) / kDecodeWarps);
    const dim3 block(32, kDecodeWarps, 1);
    switch (desc.head_size_qk) {
      case 64:
        GatedDeltaNetDecodeWarpKernel<T, 64, kDecodeWarps>
            <<<grid, block, 0, stream>>>(pack, p, desc.head_size_v);
        break;
      case 128:
        GatedDeltaNetDecodeWarpKernel<T, 128, kDecodeWarps>
            <<<grid, block, 0, stream>>>(pack, p, desc.head_size_v);
        break;
      default:
        GatedDeltaNetDecodeWarpKernel<T, 256, kDecodeWarps>
            <<<grid, block, 0, stream>>>(pack, p, desc.head_size_v);
        break;
    }
    return CUDA_CALL(cudaGetLastError());
  }

  const size_t smem =
      sizeof(float) * (static_cast<size_t>(desc.head_size_qk) * desc.head_size_v +
                       2 * desc.head_size_qk + desc.head_size_v + desc.head_size_qk);
  static std::once_flag once_rec;
  ORT_RETURN_IF_ERROR(
      SetMaxDynamicSmemOnce(GatedDeltaNetRecurrentKernel<T, kRecurrentThreads>, smem, once_rec));
  const dim3 grid(desc.batch, desc.num_heads_v, 1);
  GatedDeltaNetRecurrentKernel<T, kRecurrentThreads>
      <<<grid, kRecurrentThreads, smem, stream>>>(pack, p, desc.head_size_qk, desc.head_size_v);
  (void)max_threads_per_block;
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchGatedDeltaNet<float>(const Descriptor&, const Plan&, const VariantPack<float>&,
                                           float, int, cudaStream_t);
template Status LaunchGatedDeltaNet<half>(const Descriptor&, const Plan&, const VariantPack<half>&,
                                          float, int, cudaStream_t);

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
