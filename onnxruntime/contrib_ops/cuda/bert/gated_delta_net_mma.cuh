// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Device-side building blocks shared by the fused chunked engine
// (gated_delta_net_impl.cu) and the split K1/K2 engine (gated_delta_net_split_impl.cu):
// the m16n8k16 fragment loaders, the shared-memory GEMM they drive, the within-chunk
// decay scan, and the exact (I + M)^-1 construction.

#pragma once

#include <cuda_fp16.h>

#include <mutex>
#include <unordered_set>

#include "contrib_ops/cuda/bert/gated_delta_net_plan.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace gated_delta_net {

constexpr int kChunkedThreads = 512;
constexpr int kChunkedWarps = kChunkedThreads / 32;

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

// Descriptor fields the kernels read at run time; everything else is a template parameter.
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
  int state_update_capacity;
  int decay_per_key_dim_flag;
  // Hybrid dispatch. The chunked and recurrent launches use complementary predicates, so
  // every row is computed exactly once.
  bool state_update_tail_pass;
  bool short_row_tail_pass;
};

// ---------------------------------------------------------------------------
// mma.sync.m16n8k16 helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t PackHalf2(__half lo, __half hi) {
  return static_cast<uint32_t>(__half_as_ushort(lo)) |
         (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
}

// mma.sync.m16n8k16 is Ampere and later, so ptxas rejects it for a pre-Ampere target. The only
// caller is the chunked engine, and SelectPlan requires sm_major >= 8 before choosing it, so the
// pre-Ampere body below exists purely to let those targets compile and is never executed.
__device__ __forceinline__ void MmaM16N8K16(float (&d)[4], const uint32_t (&a)[4],
                                            const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void)d;
  (void)a;
  (void)b;
#endif
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

__device__ __forceinline__ float LoadC(const float* p) { return *p; }
__device__ __forceinline__ float LoadC(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void StoreC(float* p, float v) { *p = v; }
__device__ __forceinline__ void StoreC(__half* p, float v) { *p = __float2half(v); }

// C[M][N] (ldc) = A[M][K] @ B[K][N] (+ c_scale * C if Accumulate).
//
// Each warp owns one m-tile (16 rows) and a strided subset of the n-tiles, so the A
// fragment is loaded once per k-step and reused across every n-tile the warp owns. C may be
// float or __half; a __half C with Accumulate lets the recurrent state be updated in place,
// which is one rounding either way because the mma accumulator is float regardless.
template <int M, int N, int K, bool TA, bool TB, bool Accumulate, typename CT = float,
          int kWarps = kChunkedWarps>
__device__ __forceinline__ void SmemGemm(CT* C, int ldc, const __half* A, int lda,
                                         const __half* B, int ldb, int warp_id, int lane,
                                         float c_scale = 1.0f) {
  constexpr int kMTiles = M / 16;
  constexpr int kNTiles = N / 8;
  constexpr int kWarpsPerM = kWarps > kMTiles ? kWarps / kMTiles : 1;
  constexpr int kMGroups = kWarps / kWarpsPerM;
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
        acc[i][0] = Accumulate ? c_scale * LoadC(C + r0 * ldc + c0) : 0.0f;
        acc[i][1] = Accumulate ? c_scale * LoadC(C + r0 * ldc + c0 + 1) : 0.0f;
        acc[i][2] = Accumulate ? c_scale * LoadC(C + r1 * ldc + c0) : 0.0f;
        acc[i][3] = Accumulate ? c_scale * LoadC(C + r1 * ldc + c0 + 1) : 0.0f;
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
        StoreC(C + r0 * ldc + c0, acc[i][0]);
        StoreC(C + r0 * ldc + c0 + 1, acc[i][1]);
        StoreC(C + r1 * ldc + c0, acc[i][2]);
        StoreC(C + r1 * ldc + c0 + 1, acc[i][3]);
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

// ti = (I + m)^-1 for the strictly lower BT x BT matrix in `m`. Exact.
//
// (I+M)^-1 has a closed form: with Dinv the block diagonal of the four 16x16 inverses and
// N = Dinv M (its own 16x16 diagonal blocks are exactly zero), N is strictly block lower
// over BT/16 levels, so N^(BT/16) = 0 and (I+M)^-1 = (I - N + N^2 - ...) Dinv exactly.
// Full BT^3 GEMMs replace the serial tiny ones. `db`, `nb` and `tf` are scratch.
template <int BT>
__device__ __forceinline__ void BuildTriInverse(const __half* m, int ldm, __half* db, __half* nb,
                                                __half* ti, int ldh, float* tf, int ldf,
                                                int warp_id, int lane, int tid) {
  for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
    db[(idx / BT) * ldh + idx % BT] = __float2half(0.0f);
  }
  __syncthreads();

  // Column `lane` of the inverse of diagonal block `warp_id`, by forward substitution.
  // Lane-local, so no synchronization inside.
  if (warp_id < BT / 16 && lane < 16) {
    const int base = warp_id * 16;
    float col[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) col[i] = 0.0f;
    col[lane] = 1.0f;
    for (int i = lane + 1; i < 16; ++i) {
      float acc = 0.0f;
      for (int j = lane; j < i; ++j) {
        acc += __half2float(m[(base + i) * ldm + base + j]) * col[j];
      }
      col[i] = -acc;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      db[(base + i) * ldh + base + lane] = __float2half(col[i]);
    }
  }
  __syncthreads();

  SmemGemm<BT, BT, BT, false, false, false>(tf, ldf, db, ldh, m, ldm, warp_id, lane);
  __syncthreads();
  for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
    const int r = idx / BT, c = idx % BT;
    const bool same_block = (r >> 4) == (c >> 4);
    const float n = same_block ? 0.0f : tf[r * ldf + c];
    nb[r * ldh + c] = __float2half(n);
    ti[r * ldh + c] = __float2half((r == c ? 1.0f : 0.0f) - n);  // Z1 = I - N
  }
  __syncthreads();

  // Horner needs BT/16 - 2 steps after Z1 = I - N.
#pragma unroll
  for (int it = 0; it < BT / 16 - 2; ++it) {
    SmemGemm<BT, BT, BT, false, false, false>(tf, ldf, nb, ldh, ti, ldh, warp_id, lane);
    __syncthreads();
    for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
      const int r = idx / BT, c = idx % BT;
      ti[r * ldh + c] = __float2half((r == c ? 1.0f : 0.0f) - tf[r * ldf + c]);
    }
    __syncthreads();
  }

  SmemGemm<BT, BT, BT, false, false, false>(tf, ldf, ti, ldh, db, ldh, warp_id, lane);
  __syncthreads();
  for (int idx = tid; idx < BT * BT; idx += kChunkedThreads) {
    const int r = idx / BT, c = idx % BT;
    ti[r * ldh + c] = __float2half(tf[r * ldf + c]);
  }
  __syncthreads();
}

struct DynamicSmemConfig {
  std::mutex mutex;
  std::unordered_set<int> configured_devices;
};

// Raise the opt-in shared-memory maximum once per (device, kernel) to the device limit.
template <typename KernelT>
Status SetMaxDynamicSmem(KernelT kernel, size_t device_max, DynamicSmemConfig& config) {
  int device_id = 0;
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetDevice(&device_id)));

  std::lock_guard<std::mutex> lock(config.mutex);
  if (config.configured_devices.count(device_id) != 0) {
    return Status::OK();
  }

  const cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(device_max));
  if (err != cudaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GatedDeltaNet: cudaFuncSetAttribute failed: ",
                           cudaGetErrorString(err));
  }
  config.configured_devices.insert(device_id);
  return Status::OK();
}

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
