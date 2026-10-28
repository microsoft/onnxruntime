// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Fused low-rank hyper-connection block (see the HyperConnectionLowRankMix schema in bert_defs.cc).
//
// Fused path (rows <= kHyperConnectionLowRankMixMaxFusedRows, 16-bit T, 16-byte aligned shapes): two kernels.
//   HcDownKernel  grid (column tiles + 1, splits, row groups)
//                 [post-mix of the previous block] + branchwise RMS norm + down GEMV + inject GEMV.
//                 The K = C * H reduction is split in slices that each stay inside one branch; every CTA
//                 writes fp32 partial sums for its slice. No atomics, no clusters, no grid barrier.
//   HcUpPreKernel grid (ceil(H / 16), 1, row groups)
//                 reduces the partials in a fixed order, then ScaledSiLU, up GEMV, sigmoid, pre-mix and the
//                 inject gate.
// When the weight inputs (norm_scale, down_weight, up_weight, inject_weight) are constant initializers, both
// kernels are launched with programmatic dependent launch on sm_90+ (plain launches otherwise). Each one
// triggers its dependents immediately and prefetches its constant weights into shared memory before
// griddepcontrol.wait; everything that consumes another kernel's output comes after the wait, so correctness
// never depends on the trigger position.
// Every intermediate is rounded to T at the points where the unfused ONNX subgraph rounds; the GEMVs
// accumulate in fp32 in a fixed (deterministic) order.
//
// Unfused path (any shape, any T): norm kernel + cuBLAS GEMMs (fp32 accumulation) + elementwise kernels.

#include "contrib_ops/cuda/bert/hyper_connection_low_rank_mix_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <limits>
#include <type_traits>

#include "core/providers/cuda/cu_inc/cub.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/providers/cuda/cuda_common.h"

// Include env_var_utils.h after cuda_common.h: the latter transitively pulls in provider_api.h,
// which defines SHARED_PROVIDER and suppresses env_var_utils.h's own logging.h include.
#include "core/platform/env_var_utils.h"

namespace onnxruntime::contrib::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kColTile = 16;          // down columns per CTA of the down kernel (two 16-byte weight vectors)
constexpr int kUpTile = 16;           // features per CTA of the up kernel
constexpr int kMaxSliceRows = 1280;   // K rows handled by one CTA of the down kernel
constexpr int kWeightSlots = 10;      // 16-byte weight vectors per thread in the down pass
constexpr int kAsyncWeightSlots = 9;  // slots staged in shared memory so 5 CTAs/SM fit in 228 KB on sm_90+
constexpr int kMaxVecIters = 2;       // 16-byte vectors per thread in the norm pass
constexpr int kMaxFusedHidden = kMaxVecIters * kThreads * 8;
constexpr int kMaxFusedRank = 512;
constexpr int kMaxFusedBranches = 8;

__device__ __forceinline__ float SigmoidF32(float v) {
  if (v >= 0.0f) return 1.0f / (1.0f + expf(-v));
  const float e = expf(v);
  return e / (1.0f + e);
}

template <typename T>
__device__ __forceinline__ float Rnd(float v) {
  return to_float<T>(from_float<T>(v));
}

template <typename T>
__device__ __forceinline__ float BitsToFloat(uint16_t bits) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
  } else {
    return __half2float(__ushort_as_half(bits));
  }
}

template <typename T>
__device__ __forceinline__ uint16_t FloatToBits(float v) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat16_as_ushort(__float2bfloat16(v));
  } else {
    return __half_as_ushort(__float2half(v));
  }
}

template <typename T>
__device__ __forceinline__ void Unpack2(uint32_t u, float& a, float& b) {
  a = BitsToFloat<T>(static_cast<uint16_t>(u & 0xffffu));
  b = BitsToFloat<T>(static_cast<uint16_t>(u >> 16));
}

template <typename T>
__device__ __forceinline__ uint32_t Pack2(float a, float b) {
  return static_cast<uint32_t>(FloatToBits<T>(a)) | (static_cast<uint32_t>(FloatToBits<T>(b)) << 16);
}

template <typename T>
__device__ __forceinline__ void Unpack8(const uint4& u, float (&f)[8]) {
  Unpack2<T>(u.x, f[0], f[1]);
  Unpack2<T>(u.y, f[2], f[3]);
  Unpack2<T>(u.z, f[4], f[5]);
  Unpack2<T>(u.w, f[6], f[7]);
}

template <typename T>
__device__ __forceinline__ uint4 Pack8(const float (&f)[8]) {
  uint4 u;
  u.x = Pack2<T>(f[0], f[1]);
  u.y = Pack2<T>(f[2], f[3]);
  u.z = Pack2<T>(f[4], f[5]);
  u.w = Pack2<T>(f[6], f[7]);
  return u;
}

__device__ __forceinline__ void GridWait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void GridTrigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ void CpAsync16(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))),
               "l"(gmem)
               : "memory");
#else
  *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<const uint4*>(gmem);
#endif
}

__device__ __forceinline__ void CpCommit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void CpWaitAll() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

struct DownParams {
  int64_t rows;
  int branches;
  int hidden;
  int rank;
  int pieces;          // slices per branch
  int slice;           // K rows per slice (multiple of 8)
  int splits;          // branches * pieces
  int partial_stride;  // rank + branches, rounded up to a multiple of 4
  float epsilon;
};

struct UpParams {
  int64_t rows;
  int branches;
  int hidden;
  int rank;
  int splits;
  int partial_stride;
  float silu_alpha;
  float reduction_scale;
  float gate_div;
  float gate_mul;
};

__device__ __forceinline__ bool RowOk(int64_t row0, int r, int64_t total_rows) {
  return row0 + r < total_rows;
}

__device__ __forceinline__ int64_t RowIdx(int64_t row0, int r, int64_t total_rows) {
  return (row0 + r < total_rows) ? (row0 + r) : (total_rows - 1);
}

// CTA (j, s, z): K slice s (rows [k0, k0 + len) of the flattened stream, inside branch c) of the ROWS rows
// z * ROWS ... z * ROWS + ROWS - 1, which share every weight read.
// j < ceil(R / 16) owns down columns [16 j, 16 j + 16); the last j owns the C inject columns.
template <typename T, bool kPost, int ROWS>
__global__ void __launch_bounds__(kThreads, ROWS >= 4 ? 4 : 5)
    HcDownKernel(const T* __restrict__ streams, const T* __restrict__ prev_out, const T* __restrict__ prev_gate,
                 const T* __restrict__ norm_w, const T* __restrict__ w_down, const T* __restrict__ w_inject,
                 T* __restrict__ streams_out, float* __restrict__ partial, float* __restrict__ inv_rms_out,
                 DownParams p) {
  constexpr int kLocalAsyncWeightSlots = ROWS >= 4 ? 8 : kAsyncWeightSlots;
  const int j = blockIdx.x;
  const int s = blockIdx.y;
  const int64_t row0 = static_cast<int64_t>(blockIdx.z) * ROWS;
  const int tid = threadIdx.x;
  const int branches = p.branches;
  const int hidden = p.hidden;
  const int64_t K = static_cast<int64_t>(branches) * hidden;
  const int c = s / p.pieces;
  const int piece = s - c * p.pieces;
  const int h0 = piece * p.slice;
  const int len = min(p.slice, hidden - h0);
  const int col_tiles = (p.rank + kColTile - 1) / kColTile;
  const bool is_inject = (j == col_tiles);
  const int64_t k0 = static_cast<int64_t>(c) * hidden + h0;

  __shared__ __align__(16) uint4 wsm[kLocalAsyncWeightSlots * kThreads];
  __shared__ __align__(16) uint16_t xs[ROWS][kMaxSliceRows];
  __shared__ float sq_red[8][ROWS];

  const int cg = tid & 1;
  const int rp = tid >> 1;
  const int col = j * kColTile + cg * 8;
  // Let the up kernel start right away and prefetch its weights; its results are only read after GridWait().
  GridTrigger();

  // Weights do not depend on any previous kernel: start streaming them into shared memory first.
  if (!is_inject) {
    if (col < p.rank) {
      const T* base = w_down + k0 * p.rank + col;
#pragma unroll
      for (int i = 0; i < kLocalAsyncWeightSlots; ++i) {
        const int row = rp + 128 * i;
        if (row < len) CpAsync16(&wsm[i * kThreads + tid], base + static_cast<int64_t>(row) * p.rank);
      }
    }
  } else {
    const int chunks = len * branches / 8;
    for (int q = tid; q < chunks; q += kThreads) CpAsync16(&wsm[q], w_inject + k0 * branches + q * 8);
  }
  CpCommit();

  // Norm weights for the 8 consecutive features of the slice this thread normalizes below.
  const bool norm_active = tid * 8 < len;
  uint4 wv = make_uint4(0u, 0u, 0u, 0u);
  if (norm_active) wv = *reinterpret_cast<const uint4*>(norm_w + k0 + tid * 8);

  // Everything below consumes results of the previous kernel.
  GridWait();

  const int nvec = hidden / 8;
  float sum_sq[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int64_t r_idx = RowIdx(row0, r, p.rows);
    const bool r_ok = RowOk(row0, r, p.rows);
    uint4 sv[kMaxVecIters];
    uint4 bv[kMaxVecIters];
    float gate_c = 0.0f;
    const uint4* sp = reinterpret_cast<const uint4*>(streams + r_idx * K + static_cast<int64_t>(c) * hidden);
#pragma unroll
    for (int i = 0; i < kMaxVecIters; ++i) {
      const int qi = tid + kThreads * i;
      sv[i] = qi < nvec ? __ldcg(sp + qi) : make_uint4(0u, 0u, 0u, 0u);
    }
    if constexpr (kPost) {
      gate_c = BitsToFloat<T>(__ldcg(reinterpret_cast<const uint16_t*>(prev_gate + r_idx * branches + c)));
      const uint4* bp = reinterpret_cast<const uint4*>(prev_out + r_idx * hidden);
#pragma unroll
      for (int i = 0; i < kMaxVecIters; ++i) {
        const int qi = tid + kThreads * i;
        bv[i] = qi < nvec ? __ldcg(bp + qi) : make_uint4(0u, 0u, 0u, 0u);
      }
    }
    sum_sq[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxVecIters; ++i) {
      const int qi = tid + kThreads * i;
      if (qi < nvec) {
        const int h = qi * 8;
        float v[8];
        Unpack8<T>(sv[i], v);
        if constexpr (kPost) {
          float b[8];
          Unpack8<T>(bv[i], b);
#pragma unroll
          for (int m = 0; m < 8; ++m) v[m] = Rnd<T>(fmaf(gate_c, b[m], v[m]));
        }
#pragma unroll
        for (int m = 0; m < 8; ++m) sum_sq[r] += v[m] * v[m];
        if (h >= h0 && h < h0 + len) {
          const uint4 packed = Pack8<T>(v);
          *reinterpret_cast<uint4*>(&xs[r][h - h0]) = packed;
          if constexpr (kPost) {
            if (j == 0 && r_ok) {
              *reinterpret_cast<uint4*>(streams_out + r_idx * K + static_cast<int64_t>(c) * hidden + h) =
                  packed;
            }
          }
        }
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum_sq[r] += __shfl_xor_sync(0xffffffffu, sum_sq[r], off);
    if ((tid & 31) == 0) sq_red[tid >> 5][r] = sum_sq[r];
  }
  __syncthreads();
  float inv_rms[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += sq_red[w][r];
    inv_rms[r] = rsqrtf(total / static_cast<float>(hidden) + p.epsilon);
    if (j == 0 && piece == 0 && tid == 0 && RowOk(row0, r, p.rows)) {
      inv_rms_out[RowIdx(row0, r, p.rows) * branches + c] = inv_rms[r];
    }
    if (norm_active) {
      float x[8], w[8];
      Unpack8<T>(*reinterpret_cast<const uint4*>(&xs[r][tid * 8]), x);
      Unpack8<T>(wv, w);
#pragma unroll
      for (int m = 0; m < 8; ++m) x[m] = x[m] * inv_rms[r] * w[m];
      *reinterpret_cast<uint4*>(&xs[r][tid * 8]) = Pack8<T>(x);
    }
  }
  CpWaitAll();
  __syncthreads();

  if (!is_inject) {
    float acc[ROWS][8];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int m = 0; m < 8; ++m) acc[r][m] = 0.0f;
    }
    if (col < p.rank) {
      const T* base = w_down + k0 * p.rank + col;
#pragma unroll
      for (int i = 0; i < kWeightSlots; ++i) {
        const int row = rp + 128 * i;
        if (row < len) {
          float wf[8];
          const uint4 w_vec = (i < kLocalAsyncWeightSlots)
                                  ? wsm[i * kThreads + tid]
                                  : __ldcg(reinterpret_cast<const uint4*>(base + static_cast<int64_t>(row) * p.rank));
          Unpack8<T>(w_vec, wf);
#pragma unroll
          for (int r = 0; r < ROWS; ++r) {
            const float xv = BitsToFloat<T>(xs[r][row]);
#pragma unroll
            for (int m = 0; m < 8; ++m) acc[r][m] = fmaf(xv, wf[m], acc[r][m]);
          }
        }
      }
    }
    // Lanes of a warp share cg = lane & 1 and differ in rp: reduce over lane bits 1..4.
    __syncthreads();
    float (*red)[ROWS][kColTile] = reinterpret_cast<float (*)[ROWS][kColTile]>(wsm);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int m = 0; m < 8; ++m) {
#pragma unroll
        for (int off = 2; off < 32; off <<= 1) acc[r][m] += __shfl_xor_sync(0xffffffffu, acc[r][m], off);
      }
      if ((tid & 31) < 2) {
#pragma unroll
        for (int m = 0; m < 8; ++m) red[tid >> 5][r][cg * 8 + m] = acc[r][m];
      }
    }
    __syncthreads();
    if (tid < kColTile * ROWS) {
      const int r = tid / kColTile;
      const int t = tid % kColTile;
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < 8; ++q) v += red[q][r][t];
      const int out_col = j * kColTile + t;
      if (out_col < p.rank && RowOk(row0, r, p.rows)) {
        partial[(RowIdx(row0, r, p.rows) * p.splits + s) * p.partial_stride + out_col] = v;
      }
    }
  } else {
    const T* wi = reinterpret_cast<const T*>(wsm);
    float a[ROWS][kMaxFusedBranches];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int m = 0; m < kMaxFusedBranches; ++m) a[r][m] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const int k = tid + kThreads * i;
      if (k < len) {
        float wf[kMaxFusedBranches];
#pragma unroll
        for (int m = 0; m < kMaxFusedBranches; ++m) wf[m] = m < branches ? to_float<T>(wi[k * branches + m]) : 0.0f;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          const float xv = BitsToFloat<T>(xs[r][k]);
#pragma unroll
          for (int m = 0; m < kMaxFusedBranches; ++m) a[r][m] = fmaf(xv, wf[m], a[r][m]);
        }
      }
    }
    __syncthreads();
    float (*red)[ROWS][kColTile] = reinterpret_cast<float (*)[ROWS][kColTile]>(wsm);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int m = 0; m < kMaxFusedBranches; ++m) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a[r][m] += __shfl_xor_sync(0xffffffffu, a[r][m], off);
        if ((tid & 31) == 0) red[tid >> 5][r][m] = a[r][m];
      }
    }
    __syncthreads();
    if (tid < branches * ROWS) {
      const int r = tid / branches;
      const int t = tid % branches;
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < 8; ++q) v += red[q][r][t];
      if (RowOk(row0, r, p.rows)) {
        partial[(RowIdx(row0, r, p.rows) * p.splits + s) * p.partial_stride + p.rank + t] = v;
      }
    }
  }
}

// CTA (hb, 0, z) owns features [16 hb, 16 hb + 16) of every branch for ROWS rows.
template <typename T, int ROWS, int BRANCHES = 0, int RANK = 0>
__global__ void __launch_bounds__(kThreads, 5)
    HcUpPreKernel(const T* __restrict__ streams, const T* __restrict__ norm_w, const T* __restrict__ w_up,
                  const float* __restrict__ partial, const float* __restrict__ inv_rms_in,
                  T* __restrict__ pre_out, T* __restrict__ gate_out, UpParams p) {
  static_assert(ROWS <= 2, "the cross-warp gate reduction below assumes ROWS * 16 * branches <= kThreads");
  constexpr int kLocalWeightSlots = kWeightSlots;
  const int hb = blockIdx.x;
  const int64_t row0 = static_cast<int64_t>(blockIdx.z) * ROWS;
  const int tid = threadIdx.x;
  const int branches = BRANCHES == 0 ? p.branches : BRANCHES;
  const int rank = RANK == 0 ? p.rank : RANK;
  constexpr int kBranchStorage = BRANCHES == 0 ? kMaxFusedBranches : BRANCHES;
  const int hidden = p.hidden;
  const int64_t K = static_cast<int64_t>(branches) * hidden;
  const int h0 = hb * kUpTile;
  const int groups = 2 * branches;       // (branch, 8-feature half) pairs; divides 32
  const int phases = kThreads / groups;  // row phases
  const int g = tid % groups;
  const int c = g >> 1;
  const int half_id = g & 1;
  const int rp = tid / groups;
  const int col = h0 + half_id * 8;
  const bool valid = col < hidden;
  const int chunk_rows = phases * kLocalWeightSlots;

  __shared__ __align__(16) uint4 wsm[kLocalWeightSlots * kThreads];
  __shared__ __align__(16) float sf[ROWS][RANK == 0 ? kMaxFusedRank : RANK];
  __shared__ float sg[ROWS][kBranchStorage * kUpTile];

  // Prefetch the first chunk of up weights before waiting for the down kernel.
  const T* wbase = w_up + static_cast<int64_t>(c) * hidden + col;
  if (valid) {
#pragma unroll
    for (int i = 0; i < kLocalWeightSlots; ++i) {
      const int row = rp + phases * i;
      if (row < rank) CpAsync16(&wsm[i * kThreads + tid], wbase + static_cast<int64_t>(row) * K);
    }
  }
  CpCommit();

  // The next block's down kernel may start (and prefetch its weights) while this kernel waits.
  GridTrigger();
  GridWait();

  // Fixed-order reduction of the K-split partial sums, then ScaledSiLU; four columns per thread.
  const int nq = rank / 4;
  for (int idx = tid; idx < ROWS * nq; idx += kThreads) {
    const int r = idx / nq;
    const int cq = (idx % nq) * 4;
    const int64_t r_idx = RowIdx(row0, r, p.rows);
    float4 va = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int sp0 = 0; sp0 < p.splits; sp0 += 8) {
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        if (sp0 + u < p.splits) {
          const float4 q = __ldcg(reinterpret_cast<const float4*>(
              partial + (r_idx * p.splits + sp0 + u) * p.partial_stride + cq));
          va.x += q.x;
          va.y += q.y;
          va.z += q.z;
          va.w += q.w;
        }
      }
    }
    const float v[4] = {va.x, va.y, va.z, va.w};
    float out[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float z = Rnd<T>(Rnd<T>(v[m]) * p.silu_alpha);
      out[m] = Rnd<T>(z * Rnd<T>(SigmoidF32(z)));
    }
    *reinterpret_cast<float4*>(&sf[r][cq]) = make_float4(out[0], out[1], out[2], out[3]);
  }
  if (hb == 0 && tid < branches * ROWS) {
    const int r = tid / branches;
    const int t = tid % branches;
    const int64_t r_idx = RowIdx(row0, r, p.rows);
    float v = 0.0f;
    for (int sp = 0; sp < p.splits; ++sp) {
      v += __ldcg(partial + (r_idx * p.splits + sp) * p.partial_stride + rank + t);
    }
    const float inject = Rnd<T>(v);
    const float u = Rnd<T>(SigmoidF32(Rnd<T>(inject / p.gate_div)));
    if (RowOk(row0, r, p.rows)) gate_out[r_idx * branches + t] = from_float<T>(u * p.gate_mul);
  }
  __syncthreads();

  float acc[ROWS][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
#pragma unroll
    for (int m = 0; m < 8; ++m) acc[r][m] = 0.0f;
  }
  for (int chunk0 = 0; chunk0 < rank; chunk0 += chunk_rows) {
    if (chunk0 > 0 && valid) {
#pragma unroll
      for (int i = 0; i < kLocalWeightSlots; ++i) {
        const int row = chunk0 + rp + phases * i;
        if (row < rank) CpAsync16(&wsm[i * kThreads + tid], wbase + static_cast<int64_t>(row) * K);
      }
      CpCommit();
    }
    CpWaitAll();
    if (valid) {
#pragma unroll
      for (int i = 0; i < kLocalWeightSlots; ++i) {
        const int row = chunk0 + rp + phases * i;
        if (row < rank) {
          float wf[8];
          Unpack8<T>(wsm[i * kThreads + tid], wf);
#pragma unroll
          for (int r = 0; r < ROWS; ++r) {
            const float sv = sf[r][row];
#pragma unroll
            for (int m = 0; m < 8; ++m) acc[r][m] = fmaf(sv, wf[m], acc[r][m]);
          }
        }
      }
    }
  }
  // Lanes of a warp share g = lane % groups and differ in rp.
  for (int off = groups; off < 32; off <<= 1) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int m = 0; m < 8; ++m) acc[r][m] += __shfl_xor_sync(0xffffffffu, acc[r][m], off);
    }
  }
  __syncthreads();                             // every thread is done with its weights: reuse that shared memory for the cross-warp sums
  float* red = reinterpret_cast<float*>(wsm);  // [8 warps][ROWS][groups * 8]
  if ((tid & 31) < groups) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int m = 0; m < 8; ++m) red[((tid >> 5) * ROWS + r) * (groups * 8) + g * 8 + m] = acc[r][m];
    }
  }
  __syncthreads();
  // The up kernel runs with at most 2 rows, so ROWS * groups * 8 = 16 * ROWS * branches <= kThreads (branches <= 8).
  const int g8 = groups * 8;
  if (tid < ROWS * g8) {
    const int r = tid / g8;
    const int idx = tid % g8;
    float v = 0.0f;
#pragma unroll
    for (int q = 0; q < 8; ++q) v += red[(q * ROWS + r) * g8 + idx];
    const int gg = idx >> 3;
    const int local = (gg & 1) * 8 + (idx & 7);
    sg[r][(gg >> 1) * kUpTile + local] = Rnd<T>(SigmoidF32(Rnd<T>(v)));
  }
  __syncthreads();
  if (tid < ROWS * kUpTile) {
    const int r = tid / kUpTile;
    const int hl = tid % kUpTile;
    const int64_t r_idx = RowIdx(row0, r, p.rows);
    if ((h0 + hl) < hidden && RowOk(row0, r, p.rows)) {
      float sum = 0.0f;
#pragma unroll
      for (int b = 0; b < kBranchStorage; ++b) {
        if (b < branches) {
          const float nw = to_float<T>(norm_w[static_cast<int64_t>(b) * hidden + h0 + hl]);
          const float xv = BitsToFloat<T>(__ldcg(reinterpret_cast<const uint16_t*>(
              streams + r_idx * K + static_cast<int64_t>(b) * hidden + h0 + hl)));
          const float inv = __ldcg(inv_rms_in + r_idx * branches + b);
          const float n = Rnd<T>(xv * inv * nw);
          sum = fmaf(n, sg[r][b * kUpTile + hl], sum);
        }
      }
      pre_out[r_idx * hidden + h0 + hl] = from_float<T>(sum * p.reduction_scale);
    }
  }
}

// ----------------------------------------------------------------------------- unfused path kernels

template <typename T, bool kPost>
__global__ void NormVec8Kernel(const T* __restrict__ streams, const T* __restrict__ prev_out,
                               const T* __restrict__ prev_gate, const T* __restrict__ norm_w, T* __restrict__ hn,
                               T* __restrict__ streams_out, int branches, int hidden, float epsilon) {
  __shared__ float sq_red[8];
  const int tid = threadIdx.x;
  const int64_t group = blockIdx.x;
  const int64_t row = group / branches;
  const int c = static_cast<int>(group - row * branches);
  const int64_t base = group * hidden;
  const int nvec = hidden >> 3;
  float gate_c = 0.0f;
  if constexpr (kPost) gate_c = to_float<T>(__ldg(prev_gate + group));

  const uint4* sp = reinterpret_cast<const uint4*>(streams + base);
  const uint4* bp = kPost ? reinterpret_cast<const uint4*>(prev_out + row * hidden) : nullptr;
  uint4* sout_p = (kPost && streams_out != nullptr) ? reinterpret_cast<uint4*>(streams_out + base) : nullptr;

  float v_reg[kMaxVecIters][8];
  float sum_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kMaxVecIters; ++i) {
    const int qi = tid + kThreads * i;
    if (qi < nvec) {
      Unpack8<T>(__ldg(sp + qi), v_reg[i]);
      if constexpr (kPost) {
        float b[8];
        Unpack8<T>(__ldg(bp + qi), b);
#pragma unroll
        for (int m = 0; m < 8; ++m) v_reg[i][m] = Rnd<T>(fmaf(gate_c, b[m], v_reg[i][m]));
        if (sout_p != nullptr) {
          sout_p[qi] = Pack8<T>(v_reg[i]);
        }
      }
#pragma unroll
      for (int m = 0; m < 8; ++m) sum_sq += v_reg[i][m] * v_reg[i][m];
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, off);
  if ((tid & 31) == 0) sq_red[tid >> 5] = sum_sq;
  __syncthreads();

  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) total += sq_red[w];
  const float inv_rms = rsqrtf(total / static_cast<float>(hidden) + epsilon);

  const uint4* nwp = reinterpret_cast<const uint4*>(norm_w + static_cast<int64_t>(c) * hidden);
  uint4* hnp = reinterpret_cast<uint4*>(hn + base);
#pragma unroll
  for (int i = 0; i < kMaxVecIters; ++i) {
    const int qi = tid + kThreads * i;
    if (qi < nvec) {
      float nw[8];
      Unpack8<T>(__ldg(nwp + qi), nw);
      float out[8];
#pragma unroll
      for (int m = 0; m < 8; ++m) out[m] = v_reg[i][m] * inv_rms * nw[m];
      hnp[qi] = Pack8<T>(out);
    }
  }
}

template <typename T, bool kPost>
__global__ void NormKernel(const T* __restrict__ streams, const T* __restrict__ prev_out,
                           const T* __restrict__ prev_gate, const T* __restrict__ norm_w, T* __restrict__ hn,
                           T* __restrict__ streams_out, int branches, int hidden, float epsilon) {
  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage storage;
  __shared__ float inv_rms;
  const int64_t group = blockIdx.x;
  const int64_t row = group / branches;
  const int c = static_cast<int>(group - row * branches);
  const int64_t base = group * hidden;
  float gate_c = 0.0f;
  if constexpr (kPost) gate_c = to_float<T>(prev_gate[group]);

  float sum_sq = 0.0f;
  for (int64_t h = threadIdx.x; h < hidden; h += kThreads) {
    float v = to_float<T>(streams[base + h]);
    if constexpr (kPost) {
      v = Rnd<T>(fmaf(gate_c, to_float<T>(prev_out[row * hidden + h]), v));
      if (streams_out != nullptr) {
        streams_out[base + h] = from_float<T>(v);
      }
    }
    sum_sq += v * v;
  }
  const float total = BlockReduce(storage).Sum(sum_sq);
  if (threadIdx.x == 0) inv_rms = rsqrtf(total / static_cast<float>(hidden) + epsilon);
  __syncthreads();
  for (int64_t h = threadIdx.x; h < hidden; h += kThreads) {
    float v = to_float<T>(streams[base + h]);
    if constexpr (kPost) v = Rnd<T>(fmaf(gate_c, to_float<T>(prev_out[row * hidden + h]), v));
    hn[base + h] = from_float<T>(v * inv_rms * to_float<T>(norm_w[static_cast<int64_t>(c) * hidden + h]));
  }
}

template <typename T>
__global__ void ScaledSiluKernel(T* __restrict__ x, int64_t count, float alpha) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float z = Rnd<T>(to_float<T>(x[i]) * alpha);
  const float sig = Rnd<T>(SigmoidF32(z));
  x[i] = from_float<T>(z * sig);
}

template <typename T>
__global__ void PreMixVec8Kernel(const T* __restrict__ hn, const T* __restrict__ up, T* __restrict__ pre,
                                 uint32_t vec_count, uint32_t vec_cols, int branches, int hidden,
                                 float reduction_scale) {
  const uint32_t vec_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_i >= vec_count) return;
  const uint32_t row = vec_i / vec_cols;
  const int h = static_cast<int>((vec_i - row * vec_cols) * 8u);
  const int64_t K = static_cast<int64_t>(branches) * hidden;
  float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int b = 0; b < branches; ++b) {
    const int64_t idx = static_cast<int64_t>(row) * K + static_cast<int64_t>(b) * hidden + h;
    float hn_v[8], up_v[8];
    Unpack8<T>(__ldg(reinterpret_cast<const uint4*>(hn + idx)), hn_v);
    Unpack8<T>(__ldg(reinterpret_cast<const uint4*>(up + idx)), up_v);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      const float gate = Rnd<T>(SigmoidF32(up_v[m]));
      sum[m] = fmaf(hn_v[m], gate, sum[m]);
    }
  }
  float out[8];
#pragma unroll
  for (int m = 0; m < 8; ++m) {
    out[m] = sum[m] * reduction_scale;
  }
  *reinterpret_cast<uint4*>(pre + static_cast<int64_t>(vec_i) * 8) = Pack8<T>(out);
}

template <typename T>
__global__ void PreMixKernel(const T* __restrict__ hn, const T* __restrict__ up, T* __restrict__ pre,
                             int64_t count, int branches, int hidden, float reduction_scale) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int64_t row = i / hidden;
  const int h = static_cast<int>(i - row * hidden);
  const int64_t K = static_cast<int64_t>(branches) * hidden;
  float sum = 0.0f;
  for (int b = 0; b < branches; ++b) {
    const int64_t idx = row * K + static_cast<int64_t>(b) * hidden + h;
    const float gate = Rnd<T>(SigmoidF32(to_float<T>(up[idx])));
    sum = fmaf(to_float<T>(hn[idx]), gate, sum);
  }
  pre[i] = from_float<T>(sum * reduction_scale);
}

template <typename T, int ROWS_PER_CTA = 4>
__global__ void FusedInjectPostGateB4Kernel(const T* __restrict__ hn, const T* __restrict__ w_inject,
                                            T* __restrict__ gate_out, int64_t rows, int64_t K,
                                            float gate_div, float gate_mul) {
  const int64_t row0 = static_cast<int64_t>(blockIdx.x) * ROWS_PER_CTA;
  if (row0 >= rows) return;
  const int tid = threadIdx.x;
  __shared__ float smem[8][ROWS_PER_CTA][4];
  float acc[ROWS_PER_CTA][4];
#pragma unroll
  for (int r = 0; r < ROWS_PER_CTA; ++r) {
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[r][b] = 0.0f;
  }
  const int valid_rows = static_cast<int>(min(static_cast<int64_t>(ROWS_PER_CTA), rows - row0));
  for (int64_t k = tid; k < K; k += kThreads) {
    const uint2 wk_u2 = __ldg(reinterpret_cast<const uint2*>(w_inject + k * 4));
    float wf[4];
    Unpack2<T>(wk_u2.x, wf[0], wf[1]);
    Unpack2<T>(wk_u2.y, wf[2], wf[3]);
#pragma unroll
    for (int r = 0; r < ROWS_PER_CTA; ++r) {
      if (r < valid_rows) {
        const float xv = to_float<T>(hn[(row0 + r) * K + k]);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          acc[r][b] = fmaf(xv, wf[b], acc[r][b]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS_PER_CTA; ++r) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc[r][b] += __shfl_down_sync(0xffffffffu, acc[r][b], off);
      if ((tid & 31) == 0) smem[tid >> 5][r][b] = acc[r][b];
    }
  }
  __syncthreads();
  if (tid < valid_rows * 4) {
    const int r = tid >> 2;
    const int b = tid & 3;
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += smem[w][r][b];
    const float t = Rnd<T>(Rnd<T>(v) / gate_div);
    const float u = Rnd<T>(SigmoidF32(t));
    gate_out[(row0 + r) * 4 + b] = from_float<T>(u * gate_mul);
  }
}

// Grid (rows, ceil(branches / 8)): blockIdx.y selects a tile of up to 8 branch columns, so any num_branches works.
template <typename T>
__global__ void FusedInjectPostGateKernel(const T* __restrict__ hn, const T* __restrict__ w_inject,
                                          T* __restrict__ gate_out, int64_t rows, int branches, int64_t K,
                                          float gate_div, float gate_mul) {
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const int b0 = blockIdx.y * 8;
  const int tile = min(8, branches - b0);
  const T* x_row = hn + row * K;
  __shared__ float smem[8][8];
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int64_t k = tid; k < K; k += kThreads) {
    const float xv = to_float<T>(x_row[k]);
    const T* wk = w_inject + k * branches + b0;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      if (b < tile) acc[b] = fmaf(xv, to_float<T>(wk[b]), acc[b]);
    }
  }
#pragma unroll
  for (int b = 0; b < 8; ++b) {
    if (b < tile) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
      if ((tid & 31) == 0) smem[tid >> 5][b] = acc[b];
    }
  }
  __syncthreads();
  if (tid < tile) {
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += smem[w][tid];
    const float t = Rnd<T>(Rnd<T>(v) / gate_div);
    const float u = Rnd<T>(SigmoidF32(t));
    gate_out[row * branches + b0 + tid] = from_float<T>(u * gate_mul);
  }
}

template <typename T>
constexpr cudaDataType_t CudaDataType() {
  if constexpr (std::is_same_v<T, float>) {
    return CUDA_R_32F;
  } else if constexpr (std::is_same_v<T, half>) {
    return CUDA_R_16F;
  } else {
    return CUDA_R_16BF;
  }
}

// Disallows reduced-precision split-K reductions (and TF32 tensor op math) around the unfused GEMMs,
// restoring the caller's cuBLAS math mode on scope exit including on failure.
class ScopedCublasMathMode {
 public:
  ScopedCublasMathMode() = default;
  ScopedCublasMathMode(const ScopedCublasMathMode&) = delete;
  ScopedCublasMathMode& operator=(const ScopedCublasMathMode&) = delete;

  Status Set(cublasHandle_t cublas, cublasMath_t mode) {
    CUBLAS_RETURN_IF_ERROR(cublasGetMathMode(cublas, &prev_mode_));
    if (prev_mode_ != mode) {
      CUBLAS_RETURN_IF_ERROR(cublasSetMathMode(cublas, mode));
      handle_ = cublas;
      restore_ = true;
    }
    return Status::OK();
  }

  ~ScopedCublasMathMode() {
    if (restore_) {
      (void)cublasSetMathMode(handle_, prev_mode_);
    }
  }

 private:
  cublasHandle_t handle_ = nullptr;
  cublasMath_t prev_mode_ = CUBLAS_DEFAULT_MATH;
  bool restore_ = false;
};

// Row-major C[m, n] = A[m, k] * B[k, n] with fp32 accumulation and T output.
template <typename T>
Status Gemm(cublasHandle_t cublas, int64_t m, int64_t n, int64_t k, const T* a, const T* b, T* c) {
  ORT_RETURN_IF_NOT(m <= std::numeric_limits<int>::max() && n <= std::numeric_limits<int>::max() &&
                        k <= std::numeric_limits<int>::max(),
                    "HyperConnectionLowRankMix GEMM dimension is too large");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  constexpr cudaDataType_t type = CudaDataType<T>();
  CUBLAS_RETURN_IF_ERROR(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                                      static_cast<int>(k), &alpha, b, type, static_cast<int>(n), a, type,
                                      static_cast<int>(k), &beta, c, type, static_cast<int>(n),
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  return Status::OK();
}

int64_t BlocksFor(int64_t count) { return (count + kThreads - 1) / kThreads; }

struct FusedGeometry {
  int pieces;
  int slice;
  int splits;
  int col_tiles;
  int partial_stride;
};

FusedGeometry GetFusedGeometry(const HyperConnectionLowRankMixConfig& config) {
  FusedGeometry g;
  g.pieces = (config.hidden + kMaxSliceRows - 1) / kMaxSliceRows;
  g.slice = ((config.hidden + g.pieces - 1) / g.pieces + 7) / 8 * 8;
  g.pieces = (config.hidden + g.slice - 1) / g.slice;
  g.splits = config.branches * g.pieces;
  g.col_tiles = (config.rank + kColTile - 1) / kColTile;
  g.partial_stride = (config.rank + config.branches + 3) / 4 * 4;  // keeps every row of partials 16-byte aligned
  return g;
}

template <typename Kernel, typename... Args>
Status LaunchWithProgrammaticSerialization(bool use_pdl, Kernel kernel, dim3 grid, dim3 block, cudaStream_t stream,
                                           Args... args) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
  if (use_pdl) {
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = 0;
    launch_config.stream = stream;
    launch_config.attrs = &attribute;
    launch_config.numAttrs = 1;
    const cudaError_t status = cudaLaunchKernelEx(&launch_config, kernel, args...);
    if (status == cudaSuccess) {
      return Status::OK();
    }
    // An older driver can reject the programmatic serialization attribute; plain launches are always correct.
    if (status != cudaErrorInvalidValue && status != cudaErrorNotSupported &&
        status != cudaErrorCallRequiresNewerDriver) {
      return CUDA_CALL(status);
    }
    (void)cudaGetLastError();
  }
#else
  ORT_UNUSED_PARAMETER(use_pdl);
#endif
  kernel<<<grid, block, 0, stream>>>(args...);
  return CUDA_CALL(cudaGetLastError());
}

// griddepcontrol is compiled only for sm_90+; a kernel image JIT-compiled from older PTX would not contain it,
// so programmatic dependent launch must not be requested for such an image (or for a driver older than 12.3).
template <typename T, int DOWN_ROWS, int UP_ROWS = DOWN_ROWS>
bool IsProgrammaticLaunchAvailable() {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
  static thread_local int cached_device = -1;
  static thread_local bool cached_available = false;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  if (device != cached_device) {
    int driver_version = 0;
    int sm_major = 0;
    bool available = cudaDriverGetVersion(&driver_version) == cudaSuccess && driver_version >= 12030 &&
                     cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
                     sm_major >= 9;
    auto image_has_griddepcontrol = [](auto kernel) {
      cudaFuncAttributes attributes{};
      return cudaFuncGetAttributes(&attributes, kernel) == cudaSuccess && attributes.ptxVersion >= 90 &&
             attributes.binaryVersion >= 90;
    };
    available = available && image_has_griddepcontrol(HcDownKernel<T, true, DOWN_ROWS>) &&
                image_has_griddepcontrol(HcDownKernel<T, false, DOWN_ROWS>) &&
                image_has_griddepcontrol(HcUpPreKernel<T, UP_ROWS>) &&
                image_has_griddepcontrol(HcUpPreKernel<T, UP_ROWS, 4, 320>);
    (void)cudaGetLastError();
    cached_device = device;
    cached_available = available;
  }
  return cached_available;
#else
  return false;
#endif
}

template <typename T, int DOWN_ROWS, int UP_ROWS = DOWN_ROWS>
Status LaunchFusedRows(bool use_pdl, cudaStream_t stream, const HyperConnectionLowRankMixConfig& config,
                       const HyperConnectionLowRankMixTensors<T>& tensors, const FusedGeometry& g, float* partial,
                       float* inv_rms, const DownParams& down, const UpParams& up) {
  static const bool carveout_set = []() {
    (void)cudaFuncSetAttribute(HcDownKernel<T, true, DOWN_ROWS>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               cudaSharedmemCarveoutMaxShared);
    (void)cudaFuncSetAttribute(HcDownKernel<T, false, DOWN_ROWS>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               cudaSharedmemCarveoutMaxShared);
    (void)cudaFuncSetAttribute(HcUpPreKernel<T, UP_ROWS, 4, 320>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               cudaSharedmemCarveoutMaxShared);
    (void)cudaFuncSetAttribute(HcUpPreKernel<T, UP_ROWS>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               cudaSharedmemCarveoutMaxShared);
    (void)cudaGetLastError();
    return true;
  }();
  (void)carveout_set;
  use_pdl = use_pdl && IsProgrammaticLaunchAvailable<T, DOWN_ROWS, UP_ROWS>();
  const unsigned down_row_groups = static_cast<unsigned>((config.rows + DOWN_ROWS - 1) / DOWN_ROWS);
  const unsigned up_row_groups = static_cast<unsigned>((config.rows + UP_ROWS - 1) / UP_ROWS);
  const dim3 down_grid(g.col_tiles + 1, g.splits, down_row_groups);
  const dim3 up_grid((config.hidden + kUpTile - 1) / kUpTile, 1, up_row_groups);
  if (tensors.prev_block_output != nullptr) {
    ORT_RETURN_IF_ERROR(LaunchWithProgrammaticSerialization(
        use_pdl, HcDownKernel<T, true, DOWN_ROWS>, down_grid, dim3(kThreads), stream, tensors.streams,
        tensors.prev_block_output, tensors.prev_post_gate, tensors.norm_scale, tensors.down_weight,
        tensors.inject_weight, tensors.streams_out, partial, inv_rms, down));
  } else {
    ORT_RETURN_IF_ERROR(LaunchWithProgrammaticSerialization(
        use_pdl, HcDownKernel<T, false, DOWN_ROWS>, down_grid, dim3(kThreads), stream, tensors.streams,
        static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), tensors.norm_scale,
        tensors.down_weight, tensors.inject_weight, static_cast<T*>(nullptr), partial, inv_rms, down));
  }

  const T* up_streams = tensors.prev_block_output != nullptr ? tensors.streams_out : tensors.streams;
  if (config.branches == 4 && config.rank == 320) {
    return LaunchWithProgrammaticSerialization(use_pdl, HcUpPreKernel<T, UP_ROWS, 4, 320>, up_grid, dim3(kThreads),
                                               stream, up_streams, tensors.norm_scale, tensors.up_weight,
                                               static_cast<const float*>(partial),
                                               static_cast<const float*>(inv_rms), tensors.pre_output,
                                               tensors.post_gate, up);
  }
  return LaunchWithProgrammaticSerialization(use_pdl, HcUpPreKernel<T, UP_ROWS>, up_grid, dim3(kThreads), stream,
                                             up_streams, tensors.norm_scale, tensors.up_weight,
                                             static_cast<const float*>(partial),
                                             static_cast<const float*>(inv_rms), tensors.pre_output,
                                             tensors.post_gate, up);
}

}  // namespace

template <typename T>
bool CanUseFusedHyperConnectionLowRankMix(const cudaDeviceProp& device_prop,
                                          const HyperConnectionLowRankMixConfig& config) {
  if constexpr (sizeof(T) != 2) {
    return false;
  } else {
    const bool branches_ok = config.branches == 1 || config.branches == 2 || config.branches == 4 ||
                             config.branches == 8;
    return device_prop.major >= 8 && config.rows >= 1 &&
           config.rows <= kHyperConnectionLowRankMixMaxFusedRows && branches_ok &&
           config.hidden % 8 == 0 && config.hidden <= kMaxFusedHidden &&
           config.rank % 8 == 0 && config.rank >= 8 && config.rank <= kMaxFusedRank;
  }
}

size_t HyperConnectionLowRankMixFusedScratchElements(const HyperConnectionLowRankMixConfig& config) {
  const FusedGeometry g = GetFusedGeometry(config);
  return static_cast<size_t>(config.rows) *
         (static_cast<size_t>(g.splits) * g.partial_stride + static_cast<size_t>(config.branches));
}

template <typename T>
Status LaunchHyperConnectionLowRankMixFused(const cudaDeviceProp& device_prop, cudaStream_t stream,
                                            const HyperConnectionLowRankMixConfig& config,
                                            const HyperConnectionLowRankMixTensors<T>& tensors,
                                            float* scratch, bool weights_are_constant) {
  if constexpr (sizeof(T) != 2) {
    ORT_UNUSED_PARAMETER(device_prop);
    ORT_UNUSED_PARAMETER(stream);
    ORT_UNUSED_PARAMETER(config);
    ORT_UNUSED_PARAMETER(tensors);
    ORT_UNUSED_PARAMETER(scratch);
    ORT_UNUSED_PARAMETER(weights_are_constant);
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "fused HyperConnectionLowRankMix requires a 16-bit type");
  } else {
    ORT_RETURN_IF_NOT(CanUseFusedHyperConnectionLowRankMix<T>(device_prop, config),
                      "unsupported problem for the fused HyperConnectionLowRankMix path");
    ORT_RETURN_IF_NOT(tensors.prev_block_output == nullptr || tensors.streams_out != nullptr,
                      "fused HyperConnectionLowRankMix requires streams_out buffer when prologue is present");
    const FusedGeometry g = GetFusedGeometry(config);
    float* partial = scratch;
    float* inv_rms = scratch + static_cast<size_t>(config.rows) * g.splits * g.partial_stride;

    DownParams down{config.rows, config.branches, config.hidden, config.rank, g.pieces, g.slice,
                    g.splits, g.partial_stride, config.epsilon};
    UpParams up{config.rows, config.branches, config.hidden, config.rank, g.splits, g.partial_stride,
                config.silu_alpha, config.reduction_scale, config.gate_div, config.gate_mul};
    static const bool enable_pdl =
        onnxruntime::ParseEnvironmentVariableWithDefault<bool>("ORT_HYPER_CONNECTION_PDL", true);
    const bool use_pdl = enable_pdl && device_prop.major >= 9 && weights_are_constant;
    if (config.rows == 1) {
      return LaunchFusedRows<T, 1>(use_pdl, stream, config, tensors, g, partial, inv_rms, down, up);
    }
    // Fused row-group dispatch: rows == 1 -> 1x1, rows 2..7 -> 2x2, rows == 8 -> 4x2 (down x up rows per CTA).
    if (config.rows >= 8) {
      return LaunchFusedRows<T, 4, 2>(use_pdl, stream, config, tensors, g, partial, inv_rms, down, up);
    }
    return LaunchFusedRows<T, 2, 2>(use_pdl, stream, config, tensors, g, partial, inv_rms, down, up);
  }
}

template <typename T>
Status LaunchHyperConnectionLowRankMixUnfused(cublasHandle_t cublas, cudaStream_t stream,
                                              const HyperConnectionLowRankMixConfig& config,
                                              const HyperConnectionLowRankMixTensors<T>& tensors,
                                              T* hn, T* down, T* up) {
  const int64_t rows = config.rows;
  const int64_t K = static_cast<int64_t>(config.branches) * config.hidden;
  const int64_t groups = rows * config.branches;
  ORT_RETURN_IF_NOT(groups <= std::numeric_limits<int>::max(), "CUDA launch requires too many blocks");

  const bool use_vec8_norm =
      sizeof(T) == 2 && config.hidden % 8 == 0 && config.hidden <= kMaxFusedHidden &&
      (reinterpret_cast<uintptr_t>(tensors.streams) & 15u) == 0 &&
      (reinterpret_cast<uintptr_t>(tensors.norm_scale) & 15u) == 0 &&
      (reinterpret_cast<uintptr_t>(hn) & 15u) == 0 &&
      (tensors.prev_block_output == nullptr ||
       ((reinterpret_cast<uintptr_t>(tensors.prev_block_output) & 15u) == 0 &&
        (tensors.streams_out == nullptr || (reinterpret_cast<uintptr_t>(tensors.streams_out) & 15u) == 0)));
  if (tensors.prev_block_output != nullptr) {
    if constexpr (sizeof(T) == 2) {
      if (use_vec8_norm) {
        NormVec8Kernel<T, true><<<static_cast<int>(groups), kThreads, 0, stream>>>(
            tensors.streams, tensors.prev_block_output, tensors.prev_post_gate, tensors.norm_scale, hn,
            tensors.streams_out, config.branches, config.hidden, config.epsilon);
      } else {
        NormKernel<T, true><<<static_cast<int>(groups), kThreads, 0, stream>>>(
            tensors.streams, tensors.prev_block_output, tensors.prev_post_gate, tensors.norm_scale, hn,
            tensors.streams_out, config.branches, config.hidden, config.epsilon);
      }
    } else {
      NormKernel<T, true><<<static_cast<int>(groups), kThreads, 0, stream>>>(
          tensors.streams, tensors.prev_block_output, tensors.prev_post_gate, tensors.norm_scale, hn,
          tensors.streams_out, config.branches, config.hidden, config.epsilon);
    }
  } else {
    if constexpr (sizeof(T) == 2) {
      if (use_vec8_norm) {
        NormVec8Kernel<T, false><<<static_cast<int>(groups), kThreads, 0, stream>>>(
            tensors.streams, nullptr, nullptr, tensors.norm_scale, hn, nullptr, config.branches, config.hidden,
            config.epsilon);
      } else {
        NormKernel<T, false><<<static_cast<int>(groups), kThreads, 0, stream>>>(
            tensors.streams, nullptr, nullptr, tensors.norm_scale, hn, nullptr, config.branches, config.hidden,
            config.epsilon);
      }
    } else {
      NormKernel<T, false><<<static_cast<int>(groups), kThreads, 0, stream>>>(
          tensors.streams, nullptr, nullptr, tensors.norm_scale, hn, nullptr, config.branches, config.hidden,
          config.epsilon);
    }
  }
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  ScopedCublasMathMode math_mode;
  ORT_RETURN_IF_ERROR(math_mode.Set(cublas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));

  ORT_RETURN_IF_ERROR(Gemm<T>(cublas, rows, config.rank, K, hn, tensors.down_weight, down));

  const int64_t down_count = rows * config.rank;
  ORT_RETURN_IF_NOT(BlocksFor(down_count) <= std::numeric_limits<int>::max(),
                    "CUDA launch requires too many blocks");
  ScaledSiluKernel<T><<<static_cast<int>(BlocksFor(down_count)), kThreads, 0, stream>>>(
      down, down_count, config.silu_alpha);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  ORT_RETURN_IF_ERROR(Gemm<T>(cublas, rows, K, config.rank, down, tensors.up_weight, up));

  const int64_t pre_count = rows * config.hidden;
  if constexpr (sizeof(T) == 2) {
    const int64_t vec_count = pre_count / 8;
    if (config.hidden % 8 == 0 && vec_count <= std::numeric_limits<uint32_t>::max() &&
        (reinterpret_cast<uintptr_t>(hn) & 15u) == 0 &&
        (reinterpret_cast<uintptr_t>(up) & 15u) == 0 &&
        (reinterpret_cast<uintptr_t>(tensors.pre_output) & 15u) == 0) {
      PreMixVec8Kernel<T><<<static_cast<int>(BlocksFor(vec_count)), kThreads, 0, stream>>>(
          hn, up, tensors.pre_output, static_cast<uint32_t>(vec_count),
          static_cast<uint32_t>(config.hidden / 8), config.branches, config.hidden, config.reduction_scale);
    } else {
      ORT_RETURN_IF_NOT(BlocksFor(pre_count) <= std::numeric_limits<int>::max(),
                        "CUDA launch requires too many blocks");
      PreMixKernel<T><<<static_cast<int>(BlocksFor(pre_count)), kThreads, 0, stream>>>(
          hn, up, tensors.pre_output, pre_count, config.branches, config.hidden, config.reduction_scale);
    }
  } else {
    ORT_RETURN_IF_NOT(BlocksFor(pre_count) <= std::numeric_limits<int>::max(),
                      "CUDA launch requires too many blocks");
    PreMixKernel<T><<<static_cast<int>(BlocksFor(pre_count)), kThreads, 0, stream>>>(
        hn, up, tensors.pre_output, pre_count, config.branches, config.hidden, config.reduction_scale);
  }
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  if constexpr (sizeof(T) == 2) {
    if (config.branches == 4 && (reinterpret_cast<uintptr_t>(tensors.inject_weight) & 7u) == 0) {
      const int64_t row_blocks = (rows + 3) / 4;
      FusedInjectPostGateB4Kernel<T, 4><<<static_cast<int>(row_blocks), kThreads, 0, stream>>>(
          hn, tensors.inject_weight, tensors.post_gate, rows, K, config.gate_div, config.gate_mul);
      return CUDA_CALL(cudaGetLastError());
    }
  }
  const dim3 inject_grid(static_cast<unsigned>(rows), static_cast<unsigned>((config.branches + 7) / 8));
  FusedInjectPostGateKernel<T><<<inject_grid, kThreads, 0, stream>>>(
      hn, tensors.inject_weight, tensors.post_gate, rows, config.branches, K, config.gate_div, config.gate_mul);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                           \
  template bool CanUseFusedHyperConnectionLowRankMix<T>(const cudaDeviceProp&,                   \
                                                        const HyperConnectionLowRankMixConfig&); \
  template Status LaunchHyperConnectionLowRankMixFused<T>(                                       \
      const cudaDeviceProp&, cudaStream_t, const HyperConnectionLowRankMixConfig&,               \
      const HyperConnectionLowRankMixTensors<T>&, float*, bool);                                 \
  template Status LaunchHyperConnectionLowRankMixUnfused<T>(                                     \
      cublasHandle_t, cudaStream_t, const HyperConnectionLowRankMixConfig&,                      \
      const HyperConnectionLowRankMixTensors<T>&, T*, T*, T*);

INSTANTIATE(float)
INSTANTIATE(half)
INSTANTIATE(__nv_bfloat16)

#undef INSTANTIATE

}  // namespace onnxruntime::contrib::cuda
