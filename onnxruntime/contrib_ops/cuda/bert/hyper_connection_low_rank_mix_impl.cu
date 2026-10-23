// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Fused low-rank hyper-connection block (see the HyperConnectionLowRankMix schema in bert_defs.cc).
//
// Fused path (rows <= kHyperConnectionLowRankMixMaxFusedRows, 16-bit T, 16-byte aligned shapes):
// For rows 1/2, a cooperative kernel runs the same down/up bodies with a grid.sync() between them
// when the entire grid can reside concurrently. Otherwise the existing two-kernel path is used.
//   HcDownKernel  grid (column tiles + 1, splits, row groups)
//                 [post-mix of the previous block] + branchwise RMS norm + down GEMV + inject GEMV.
//                 The K = C * H reduction is split in slices that each stay inside one branch; every CTA
//                 writes fp32 partial sums for its slice. No atomics, no clusters, no grid barrier.
//   HcUpPreKernel grid (ceil(H / 16), 1, row groups)
//                 reduces the partials in a fixed order, then ScaledSiLU, up GEMV, sigmoid, pre-mix and the
//                 inject gate.
// Both kernels are launched with programmatic dependent launch (sm_90+, plain launches elsewhere). Each one
// triggers its dependents immediately and prefetches its weights (which never depend on a previous kernel)
// into shared memory before griddepcontrol.wait; everything that consumes another kernel's output comes after
// the wait, so correctness never depends on the trigger position.
// Every intermediate is rounded to T at the points where the unfused ONNX subgraph rounds; the GEMVs
// accumulate in fp32 in a fixed (deterministic) order.
//
// Unfused path (any shape, any T): norm kernel + cuBLAS GEMMs (fp32 accumulation) + elementwise kernels.

#include "contrib_ops/cuda/bert/hyper_connection_low_rank_mix_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include "core/platform/env_var_utils.h"
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <limits>
#include <type_traits>

#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kColTile = 16;         // down columns per CTA of the down kernel (two 16-byte weight vectors)
constexpr int kUpTile = 16;          // features per CTA of the up kernel
constexpr int kMaxSliceRows = 1280;  // K rows handled by one CTA of the down kernel
constexpr int kWeightSlots = 10;     // 16-byte weight vectors per thread held in shared memory
constexpr int kMaxVecIters = 2;      // 16-byte vectors per thread in the norm pass
constexpr int kMaxFusedHidden = kMaxVecIters * kThreads * 8;
constexpr int kMaxFusedRank = 512;
constexpr int kMaxFusedBranches = 8;

__global__ void PackHcWeightKernel(uint4* packed, const uint16_t* source, bool up) {
  const int tile = blockIdx.x;
  for (int q = threadIdx.x; q < kWeightSlots * kThreads; q += blockDim.x) {
    const int i = q / kThreads;
    const int tid = q % kThreads;
    int64_t offset;
    if (up) {
      const int row = tid / 8 + 32 * i;
      const int c = (tid % 8) / 2;
      offset = static_cast<int64_t>(row) * 10240 + c * 2560 + tile * 16 + (tid % 2) * 8;
    } else {
      const int row = (tile / 20) * 1280 + tid / 2 + 128 * i;
      offset = static_cast<int64_t>(row) * 320 + (tile % 20) * 16 + (tid % 2) * 8;
    }
    packed[tile * kWeightSlots * kThreads + q] = *reinterpret_cast<const uint4*>(source + offset);
  }
}

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

__device__ __forceinline__ uint4 LoadActivation(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

template <typename T>
__device__ __forceinline__ float LoadActivation(const T* p) {
  uint16_t bits;
  asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(bits) : "l"(p) : "memory");
  return BitsToFloat<T>(bits);
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

template <int ROWS>
struct DownShared {
  __align__(16) uint4 wsm[kWeightSlots * kThreads];
  __align__(16) uint16_t xs[ROWS][kMaxSliceRows];
  float red[8][ROWS][kColTile];
  float sq_red[8][ROWS];
};

template <int ROWS, int BRANCHES, int RANK>
struct UpShared {
  __align__(16) uint4 wsm[kWeightSlots * kThreads];
  __align__(16) float sf[ROWS][RANK == 0 ? kMaxFusedRank : RANK];
  float sg[ROWS][(BRANCHES == 0 ? kMaxFusedBranches : BRANCHES) * kUpTile];
};

template <int ROWS, int BRANCHES, int RANK>
union CooperativeShared {
  DownShared<ROWS> down;
  UpShared<ROWS, BRANCHES, RANK> up;
};

template <int ROWS>
struct CooperativePrefetchShared {
  DownShared<ROWS> down;
  UpShared<ROWS, 4, 320> up;
};

// CTA (j, s, z): K slice s (rows [k0, k0 + len) of the flattened stream, inside branch c) of the ROWS rows
// z * ROWS ... z * ROWS + ROWS - 1, which share every weight read.
// j < ceil(R / 16) owns down columns [16 j, 16 j + 16); the last j owns the C inject columns.
template <typename T, bool kPost, int ROWS, bool COOPERATIVE = false, bool PACKED = false>
__device__ __forceinline__ void
    HcDownBody(const T* __restrict__ streams, const T* __restrict__ prev_out, const T* __restrict__ prev_gate,
                 const T* __restrict__ norm_w, const T* __restrict__ w_down, const T* __restrict__ w_inject,
                 T* __restrict__ streams_out, float* __restrict__ partial, float* __restrict__ inv_rms_out,
                 DownParams p, DownShared<ROWS>& shared, int j, int s, int64_t row0) {
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
  int64_t row_idx[ROWS];
  bool row_ok[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    row_ok[r] = row0 + r < p.rows;
    row_idx[r] = row_ok[r] ? row0 + r : p.rows - 1;  // keep loads of unused rows in bounds
  }

  auto& wsm = shared.wsm;
  auto& xs = shared.xs;
  auto& red = shared.red;
  auto& sq_red = shared.sq_red;

  const int cg = tid & 1;
  const int rp = tid >> 1;
  const int col = j * kColTile + cg * 8;
  // Let the up kernel start right away and prefetch its weights; its results are only read after GridWait().
  if constexpr (!COOPERATIVE) GridTrigger();

  // Weights do not depend on any previous kernel: start streaming them into shared memory first.
  if (!COOPERATIVE && !is_inject) {
    if constexpr (PACKED) {
      const uint4* tile = reinterpret_cast<const uint4*>(w_down) +
                          (s * 20 + j) * kWeightSlots * kThreads;
#pragma unroll
      for (int i = 0; i < kWeightSlots; ++i) {
        CpAsync16(&wsm[i * kThreads + tid], tile + i * kThreads + tid);
      }
    } else if (col < p.rank) {
      const T* base = w_down + k0 * p.rank + col;
#pragma unroll
      for (int i = 0; i < kWeightSlots; ++i) {
        const int row = rp + 128 * i;
        if (row < len) CpAsync16(&wsm[i * kThreads + tid], base + static_cast<int64_t>(row) * p.rank);
      }
    }
  } else if (!COOPERATIVE) {
    const int chunks = len * branches / 8;
    for (int q = tid; q < chunks; q += kThreads) CpAsync16(&wsm[q], w_inject + k0 * branches + q * 8);
  }
  if constexpr (!COOPERATIVE) CpCommit();

  // Norm weights for the 8 consecutive features of the slice this thread normalizes below.
  const bool norm_active = tid * 8 < len;
  uint4 wv = make_uint4(0u, 0u, 0u, 0u);
  if (norm_active) wv = *reinterpret_cast<const uint4*>(norm_w + k0 + tid * 8);

  // Everything below consumes results of the previous kernel.
  if constexpr (!COOPERATIVE) GridWait();

  const int nvec = hidden / 8;
  uint4 sv[ROWS][kMaxVecIters];
  uint4 bv[ROWS][kMaxVecIters];
  float gate_c[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const uint4* sp = reinterpret_cast<const uint4*>(streams + row_idx[r] * K + static_cast<int64_t>(c) * hidden);
#pragma unroll
    for (int i = 0; i < kMaxVecIters; ++i) {
      const int qi = tid + kThreads * i;
      sv[r][i] = qi < nvec ? LoadActivation(sp + qi) : make_uint4(0u, 0u, 0u, 0u);
    }
    if constexpr (kPost) {
      gate_c[r] = LoadActivation(prev_gate + row_idx[r] * branches + c);
      const uint4* bp = reinterpret_cast<const uint4*>(prev_out + row_idx[r] * hidden);
#pragma unroll
      for (int i = 0; i < kMaxVecIters; ++i) {
        const int qi = tid + kThreads * i;
        bv[r][i] = qi < nvec ? LoadActivation(bp + qi) : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }

  float sum_sq[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    sum_sq[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxVecIters; ++i) {
      const int qi = tid + kThreads * i;
      if (qi < nvec) {
        const int h = qi * 8;
        float v[8];
        Unpack8<T>(sv[r][i], v);
        if constexpr (kPost) {
          float b[8];
          Unpack8<T>(bv[r][i], b);
#pragma unroll
          for (int m = 0; m < 8; ++m) v[m] = Rnd<T>(fmaf(gate_c[r], b[m], v[m]));
        }
#pragma unroll
        for (int m = 0; m < 8; ++m) sum_sq[r] += v[m] * v[m];
        if (h >= h0 && h < h0 + len) {
          const uint4 packed = Pack8<T>(v);
          *reinterpret_cast<uint4*>(&xs[r][h - h0]) = packed;
          if (j == 0 && row_ok[r] && streams_out != nullptr) {
            // A no-prologue output is a bitwise pass-through, including NaN payloads.
            *reinterpret_cast<uint4*>(streams_out + row_idx[r] * K + static_cast<int64_t>(c) * hidden + h) =
                kPost ? packed : sv[r][i];
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
    if (j == 0 && piece == 0 && tid == 0 && row_ok[r]) inv_rms_out[row_idx[r] * branches + c] = inv_rms[r];
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
#pragma unroll
      for (int i = 0; i < kWeightSlots; ++i) {
        const int row = rp + 128 * i;
        if (row < len) {
          float wf[8];
          Unpack8<T>(wsm[i * kThreads + tid], wf);
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
      if (out_col < p.rank && row_ok[r]) partial[(row_idx[r] * p.splits + s) * p.partial_stride + out_col] = v;
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
      if (row_ok[r]) partial[(row_idx[r] * p.splits + s) * p.partial_stride + p.rank + t] = v;
    }
  }
}

// CTA (hb, 0, z) owns features [16 hb, 16 hb + 16) of every branch for ROWS rows.
template <typename T, int ROWS, int BRANCHES = 0, int RANK = 0, bool COOPERATIVE = false,
          bool PREFETCHED = false, bool PACKED = false>
__device__ __forceinline__ void
    HcUpPreBody(const T* __restrict__ streams, const T* __restrict__ norm_w, const T* __restrict__ w_up,
                  const float* __restrict__ partial, const float* __restrict__ inv_rms_in,
                  T* __restrict__ pre_out, T* __restrict__ gate_out, UpParams p,
                  UpShared<ROWS, BRANCHES, RANK>& shared, int hb, int64_t row0) {
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
  const int chunk_rows = phases * kWeightSlots;
  int64_t row_idx[ROWS];
  bool row_ok[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    row_ok[r] = row0 + r < p.rows;
    row_idx[r] = row_ok[r] ? row0 + r : p.rows - 1;
  }

  auto& wsm = shared.wsm;
  auto& sf = shared.sf;
  auto& sg = shared.sg;

  // Prefetch the first chunk of up weights before waiting for the down kernel.
  const T* wbase = w_up + static_cast<int64_t>(c) * hidden + col;
  if (!PREFETCHED && valid) {
#pragma unroll
    for (int i = 0; i < kWeightSlots; ++i) {
      if constexpr (PACKED) {
        const uint4* tile = reinterpret_cast<const uint4*>(w_up) + hb * kWeightSlots * kThreads;
        CpAsync16(&wsm[i * kThreads + tid], tile + i * kThreads + tid);
      } else {
        const int row = rp + phases * i;
        if (row < rank) CpAsync16(&wsm[i * kThreads + tid], wbase + static_cast<int64_t>(row) * K);
      }
    }
  }
  if constexpr (!PREFETCHED) CpCommit();

  // The next block's down kernel may start (and prefetch its weights) while this kernel waits.
  if constexpr (!COOPERATIVE) {
    GridTrigger();
    GridWait();
  }

  float xv[ROWS][kBranchStorage], inv[ROWS][kBranchStorage], nw[kBranchStorage];
  const int hl = tid;  // feature owned by threads < 16 in the final pre-mix
  const bool owner = tid < kUpTile && (h0 + hl) < hidden;
  if (owner) {
#pragma unroll
    for (int b = 0; b < kBranchStorage; ++b) {
      if (b < branches) {
        nw[b] = to_float<T>(norm_w[static_cast<int64_t>(b) * hidden + h0 + hl]);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          xv[r][b] = LoadActivation(streams + row_idx[r] * K + static_cast<int64_t>(b) * hidden + h0 + hl);
          inv[r][b] = inv_rms_in[row_idx[r] * branches + b];
        }
      }
    }
  }

  // Fixed-order reduction of the K-split partial sums, then ScaledSiLU; four columns per thread.
  const int nq = rank / 4;
  const int cq = tid * 4;
  float4 va[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) va[r] = make_float4(0.f, 0.f, 0.f, 0.f);
  if (tid < nq) {
    for (int sp0 = 0; sp0 < p.splits; sp0 += 8) {
      float4 pq[8][ROWS];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          pq[u][r] = sp0 + u < p.splits
                         ? __ldca(reinterpret_cast<const float4*>(
                               partial + (row_idx[r] * p.splits + sp0 + u) * p.partial_stride + cq))
                         : make_float4(0.f, 0.f, 0.f, 0.f);
        }
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          va[r].x += pq[u][r].x;
          va[r].y += pq[u][r].y;
          va[r].z += pq[u][r].z;
          va[r].w += pq[u][r].w;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const float v[4] = {va[r].x, va[r].y, va[r].z, va[r].w};
      float out[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float z = Rnd<T>(Rnd<T>(v[m]) * p.silu_alpha);
        out[m] = Rnd<T>(z * Rnd<T>(SigmoidF32(z)));
      }
      *reinterpret_cast<float4*>(&sf[r][cq]) = make_float4(out[0], out[1], out[2], out[3]);
    }
  }
  if (hb == 0 && tid < branches * ROWS) {
    const int r = tid / branches;
    const int t = tid % branches;
    float v = 0.0f;
    for (int sp = 0; sp < p.splits; ++sp) v += partial[(row_idx[r] * p.splits + sp) * p.partial_stride + rank + t];
    const float inject = Rnd<T>(v);
    const float u = Rnd<T>(SigmoidF32(Rnd<T>(inject / p.gate_div)));
    if (row_ok[r]) gate_out[row_idx[r] * branches + t] = from_float<T>(u * p.gate_mul);
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
      for (int i = 0; i < kWeightSlots; ++i) {
        const int row = chunk0 + rp + phases * i;
        if (row < rank) CpAsync16(&wsm[i * kThreads + tid], wbase + static_cast<int64_t>(row) * K);
      }
      CpCommit();
    }
    CpWaitAll();
    if (valid) {
#pragma unroll
      for (int i = 0; i < kWeightSlots; ++i) {
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
  if (tid < groups * 8) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < 8; ++q) v += red[(q * ROWS + r) * (groups * 8) + tid];
      const int gg = tid >> 3;
      const int local = (gg & 1) * 8 + (tid & 7);
      sg[r][(gg >> 1) * kUpTile + local] = Rnd<T>(SigmoidF32(Rnd<T>(v)));
    }
  }
  __syncthreads();
  if (owner) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float sum = 0.0f;
#pragma unroll
      for (int b = 0; b < kBranchStorage; ++b) {
        if (b < branches) {
          const float n = Rnd<T>(xv[r][b] * inv[r][b] * nw[b]);
          sum = fmaf(n, sg[r][b * kUpTile + hl], sum);
        }
      }
      if (row_ok[r]) pre_out[row_idx[r] * hidden + h0 + hl] = from_float<T>(sum * p.reduction_scale);
    }
  }
}

template <typename T, bool kPost, int ROWS, bool PACKED = false>
__global__ void __launch_bounds__(kThreads)
    HcDownKernel(const T* streams, const T* prev_out, const T* prev_gate,
                 const T* norm_w, const T* w_down, const T* w_inject,
                 T* streams_out, float* partial, float* inv_rms_out, DownParams p) {
  __shared__ DownShared<ROWS> shared;
  HcDownBody<T, kPost, ROWS, false, PACKED>(streams, prev_out, prev_gate, norm_w, w_down, w_inject,
                             streams_out, partial, inv_rms_out, p, shared, blockIdx.x, blockIdx.y,
                             static_cast<int64_t>(blockIdx.z) * ROWS);
}

template <typename T, int ROWS, int BRANCHES = 0, int RANK = 0, bool PACKED = false>
__global__ void __launch_bounds__(kThreads)
    HcUpPreKernel(const T* streams, const T* norm_w, const T* w_up,
                  const float* partial, const float* inv_rms_in,
                  T* pre_out, T* gate_out, UpParams p) {
  __shared__ UpShared<ROWS, BRANCHES, RANK> shared;
  HcUpPreBody<T, ROWS, BRANCHES, RANK, false, false, PACKED>(streams, norm_w, w_up, partial, inv_rms_in,
                                     pre_out, gate_out, p, shared, blockIdx.x,
                                     static_cast<int64_t>(blockIdx.z) * ROWS);
}

template <typename T, bool kPost, int ROWS, int BRANCHES, int RANK, bool PACKED = false>
__global__ void __launch_bounds__(kThreads, 2)
    HcCooperativeKernel(HyperConnectionLowRankMixTensors<T> tensors, float* partial,
                         float* inv_rms, DownParams down, UpParams up) {
  constexpr bool kPrefetchUp = BRANCHES == 4 && RANK == 320;
  using Shared = std::conditional_t<kPrefetchUp, CooperativePrefetchShared<ROWS>,
                                    CooperativeShared<ROWS, BRANCHES, RANK>>;
  extern __shared__ __align__(16) unsigned char storage[];
  auto& shared = *reinterpret_cast<Shared*>(storage);
  const int down_columns = (down.rank + kColTile - 1) / kColTile + 1;
  const int block = blockIdx.x;
  GridTrigger();
  if (block < down_columns * down.splits) {
    const int j = block % down_columns;
    const int s = block / down_columns;
    const int c = s / down.pieces;
    const int h0 = (s % down.pieces) * down.slice;
    const int len = min(down.slice, down.hidden - h0);
    const int tid = threadIdx.x;
    const int64_t k0 = static_cast<int64_t>(c) * down.hidden + h0;
    if (j < down_columns - 1 && PACKED) {
      const uint4* tile = reinterpret_cast<const uint4*>(tensors.packed_down_weight) +
                          (s * 20 + j) * kWeightSlots * kThreads;
#pragma unroll
      for (int i = 0; i < kWeightSlots; ++i) {
        CpAsync16(&shared.down.wsm[i * kThreads + tid], tile + i * kThreads + tid);
      }
    } else if (j < down_columns - 1) {
      const int col = j * kColTile + (tid & 1) * 8;
      if (col < down.rank) {
#pragma unroll
        for (int i = 0; i < kWeightSlots; ++i) {
          const int row = (tid >> 1) + 128 * i;
          if (row < len) CpAsync16(&shared.down.wsm[i * kThreads + tid],
                                  tensors.down_weight + (k0 + row) * down.rank + col);
        }
      }
    } else {
      for (int q = tid; q < len * down.branches / 8; q += kThreads) {
        CpAsync16(&shared.down.wsm[q], tensors.inject_weight + k0 * down.branches + q * 8);
      }
    }
    CpCommit();
  }
  if constexpr (kPrefetchUp) {
    // All 320 rows fit in one weight chunk for four branches. Start both phases' reads
    // together, rather than putting the entire up-weight latency after the grid barrier.
    if (block < (up.hidden + kUpTile - 1) / kUpTile) {
      const int tid = threadIdx.x;
      const int col = block * kUpTile + (tid & 1) * 8;
      const int c = (tid % 8) >> 1;
      if constexpr (PACKED) {
        const uint4* tile = reinterpret_cast<const uint4*>(tensors.packed_up_weight) +
                            block * kWeightSlots * kThreads;
#pragma unroll
        for (int i = 0; i < kWeightSlots; ++i) {
          CpAsync16(&shared.up.wsm[i * kThreads + tid], tile + i * kThreads + tid);
        }
      } else if (col < up.hidden) {
        const int64_t K = static_cast<int64_t>(up.branches) * up.hidden;
        const T* base = tensors.up_weight + static_cast<int64_t>(c) * up.hidden + col;
#pragma unroll
        for (int i = 0; i < kWeightSlots; ++i) {
          const int row = tid / 8 + 32 * i;
          CpAsync16(&shared.up.wsm[i * kThreads + tid], base + static_cast<int64_t>(row) * K);
        }
      }
      CpCommit();
    }
  }
  // Only immutable weights were touched above. Volatile activation loads in the bodies
  // prevent readonly-load hoisting across this programmatic dependency wait.
  GridWait();
  if (block < down_columns * down.splits) {
    HcDownBody<T, kPost, ROWS, true>(
        tensors.streams, tensors.prev_block_output, tensors.prev_post_gate, tensors.norm_scale,
        tensors.down_weight, tensors.inject_weight, tensors.streams_out, partial, inv_rms, down,
        shared.down, block % down_columns, block / down_columns, 0);
  }
  // The down body normally drains this thread's asynchronous copies too. Idle down
  // CTAs must explicitly drain their up prefetch before the grid-wide rendezvous.
  if constexpr (kPrefetchUp) CpWaitAll();
  // All CTAs, including ones idle in either phase, participate. This also publishes partials,
  // inverse RMS and post-mixed streams before any up body reads them. No persistent state/reset.
  cooperative_groups::this_grid().sync();
  if (block < (up.hidden + kUpTile - 1) / kUpTile) {
    const T* streams = kPost ? tensors.streams_out : tensors.streams;
    HcUpPreBody<T, ROWS, BRANCHES, RANK, true, kPrefetchUp>(
        streams, tensors.norm_scale, tensors.up_weight, partial, inv_rms, tensors.pre_output,
        tensors.post_gate, up, shared.up, block, 0);
  }
}

// ----------------------------------------------------------------------------- unfused path kernels

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
  for (int h = threadIdx.x; h < hidden; h += kThreads) {
    float v = to_float<T>(streams[base + h]);
    if constexpr (kPost) {
      v = Rnd<T>(fmaf(gate_c, to_float<T>(prev_out[row * hidden + h]), v));
      streams_out[base + h] = from_float<T>(v);
    }
    sum_sq += v * v;
  }
  const float total = BlockReduce(storage).Sum(sum_sq);
  if (threadIdx.x == 0) inv_rms = rsqrtf(total / static_cast<float>(hidden) + epsilon);
  __syncthreads();
  for (int h = threadIdx.x; h < hidden; h += kThreads) {
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

template <typename T>
__global__ void PostGateKernel(const T* __restrict__ inject, T* __restrict__ gate, int64_t count, float gate_div,
                               float gate_mul) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float t = Rnd<T>(to_float<T>(inject[i]) / gate_div);
  const float u = Rnd<T>(SigmoidF32(t));
  gate[i] = from_float<T>(u * gate_mul);
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
    return CUDA_CALL(cudaLaunchKernelEx(&launch_config, kernel, args...));
  }
#else
  ORT_UNUSED_PARAMETER(use_pdl);
#endif
  kernel<<<grid, block, 0, stream>>>(args...);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T, bool kPost, int ROWS, int BRANCHES = 0, int RANK = 0, bool PACKED = false>
Status TryLaunchCooperative(const cudaDeviceProp& device_prop, cudaStream_t stream,
                            HyperConnectionLowRankMixTensors<T> tensors, const FusedGeometry& g,
                            float* partial, float* inv_rms, DownParams down, UpParams up,
                            bool weights_ready, bool& launched) {
  launched = false;
  if (!device_prop.cooperativeLaunch || down.rows != ROWS) return Status::OK();

  auto kernel = HcCooperativeKernel<T, kPost, ROWS, BRANCHES, RANK, PACKED>;
  constexpr bool kPrefetchUp = BRANCHES == 4 && RANK == 320;
  constexpr size_t shared_bytes = kPrefetchUp ? sizeof(CooperativePrefetchShared<ROWS>)
                                             : sizeof(CooperativeShared<ROWS, BRANCHES, RANK>);
  if (shared_bytes > static_cast<size_t>(device_prop.sharedMemPerBlockOptin)) return Status::OK();
  // The prefetched path needs opt-in shared memory (about 90 KiB for two rows).
  // Attribute setup and occupancy inspection are host-only and absent from graph replay.
  if constexpr (shared_bytes > 48 * 1024) {
    CUDA_RETURN_IF_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(shared_bytes)));
  }
  int blocks_per_sm = 0;
  // Query the exact instantiated kernel with its registers and requested dynamic shared memory.
  // This host-side query does not synchronize or allocate, and is safe during stream capture.
  CUDA_RETURN_IF_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, kThreads, shared_bytes));
  const int blocks = std::max((g.col_tiles + 1) * g.splits, (up.hidden + kUpTile - 1) / kUpTile);
  if (static_cast<int64_t>(blocks) >
      static_cast<int64_t>(blocks_per_sm) * device_prop.multiProcessorCount) return Status::OK();

  void* args[] = {&tensors, &partial, &inv_rms, &down, &up};
  // Do not turn a failed launch into a fallback: in particular a capture failure invalidates
  // the capture. Only an unsupported device/shape or insufficient residency selects fallback.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
  if (device_prop.major >= 9 && weights_ready) {
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id = cudaLaunchAttributeCooperative;
    attributes[0].val.cooperative = 1;
    attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[1].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t launch{};
    launch.gridDim = dim3(blocks);
    launch.blockDim = dim3(kThreads);
    launch.dynamicSmemBytes = shared_bytes;
    launch.stream = stream;
    launch.attrs = attributes;
    launch.numAttrs = 2;
    CUDA_RETURN_IF_ERROR(cudaLaunchKernelEx(&launch, kernel, tensors, partial, inv_rms, down, up));
    launched = true;
    return Status::OK();
  }
#endif
  ORT_UNUSED_PARAMETER(weights_ready);
  CUDA_RETURN_IF_ERROR(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel), dim3(blocks),
                                                   dim3(kThreads), args, shared_bytes, stream));
  launched = true;
  return Status::OK();
}

template <typename T, bool kPost, int ROWS>
Status TryLaunchCooperativeRows(const cudaDeviceProp& device_prop, cudaStream_t stream,
                                const HyperConnectionLowRankMixConfig& config,
                                const HyperConnectionLowRankMixTensors<T>& tensors, const FusedGeometry& g,
                                float* partial, float* inv_rms, DownParams down, UpParams up, bool& launched) {
  if (config.branches == 4 && config.rank == 320) {
    if (config.hidden == 2560 && tensors.packed_down_weight != nullptr &&
        tensors.packed_up_weight != nullptr && config.weights_ready) {
      return TryLaunchCooperative<T, kPost, ROWS, 4, 320, true>(
          device_prop, stream, tensors, g, partial, inv_rms, down, up, config.weights_ready, launched);
    }
    return TryLaunchCooperative<T, kPost, ROWS, 4, 320>(device_prop, stream, tensors, g, partial,
                                                      inv_rms, down, up, config.weights_ready, launched);
  }
  return TryLaunchCooperative<T, kPost, ROWS>(device_prop, stream, tensors, g, partial,
                                             inv_rms, down, up, config.weights_ready, launched);
}

template <typename T, int ROWS, bool PACKED = false>
Status LaunchFusedRowsImpl(const cudaDeviceProp& device_prop, bool use_pdl, cudaStream_t stream,
                       const HyperConnectionLowRankMixConfig& config,
                       const HyperConnectionLowRankMixTensors<T>& tensors, const FusedGeometry& g, float* partial,
                       float* inv_rms, const DownParams& down, const UpParams& up) {
  bool launched = false;
  if (tensors.prev_block_output != nullptr) {
    ORT_RETURN_IF_ERROR((TryLaunchCooperativeRows<T, true, ROWS>(
        device_prop, stream, config, tensors, g, partial, inv_rms, down, up, launched)));
  } else {
    ORT_RETURN_IF_ERROR((TryLaunchCooperativeRows<T, false, ROWS>(
        device_prop, stream, config, tensors, g, partial, inv_rms, down, up, launched)));
  }
  if (launched) return Status::OK();

  const unsigned row_groups = static_cast<unsigned>((config.rows + ROWS - 1) / ROWS);
  const dim3 down_grid(g.col_tiles + 1, g.splits, row_groups);
  const dim3 up_grid((config.hidden + kUpTile - 1) / kUpTile, 1, row_groups);
  const T* down_weight = PACKED ? tensors.packed_down_weight : tensors.down_weight;
  const T* up_weight = PACKED ? tensors.packed_up_weight : tensors.up_weight;
  if (tensors.prev_block_output != nullptr) {
    ORT_RETURN_IF_ERROR(LaunchWithProgrammaticSerialization(
        use_pdl, HcDownKernel<T, true, ROWS, PACKED>, down_grid, dim3(kThreads), stream, tensors.streams,
        tensors.prev_block_output, tensors.prev_post_gate, tensors.norm_scale, down_weight,
        tensors.inject_weight, tensors.streams_out, partial, inv_rms, down));
  } else {
    ORT_RETURN_IF_ERROR(LaunchWithProgrammaticSerialization(
        use_pdl, HcDownKernel<T, false, ROWS, PACKED>, down_grid, dim3(kThreads), stream, tensors.streams,
        static_cast<const T*>(nullptr), static_cast<const T*>(nullptr), tensors.norm_scale,
        down_weight, tensors.inject_weight, tensors.streams_out, partial, inv_rms, down));
  }

  const T* up_streams = tensors.prev_block_output != nullptr ? tensors.streams_out : tensors.streams;
  if (config.branches == 4 && config.rank == 320) {
    return LaunchWithProgrammaticSerialization(use_pdl, HcUpPreKernel<T, ROWS, 4, 320, PACKED>, up_grid, dim3(kThreads), stream,
                                               up_streams, tensors.norm_scale, up_weight,
                                               static_cast<const float*>(partial),
                                               static_cast<const float*>(inv_rms), tensors.pre_output,
                                               tensors.post_gate, up);
  }
  return LaunchWithProgrammaticSerialization(use_pdl, HcUpPreKernel<T, ROWS>, up_grid, dim3(kThreads), stream,
                                             up_streams, tensors.norm_scale, tensors.up_weight,
                                             static_cast<const float*>(partial),
                                             static_cast<const float*>(inv_rms), tensors.pre_output,
                                             tensors.post_gate, up);
}

template <typename T, int ROWS>
Status LaunchFusedRows(const cudaDeviceProp& device_prop, bool use_pdl, cudaStream_t stream,
                       const HyperConnectionLowRankMixConfig& config,
                       const HyperConnectionLowRankMixTensors<T>& tensors, const FusedGeometry& g, float* partial,
                       float* inv_rms, const DownParams& down, const UpParams& up) {
  if (config.branches == 4 && config.hidden == 2560 && config.rank == 320 && config.weights_ready &&
      tensors.packed_down_weight != nullptr && tensors.packed_up_weight != nullptr) {
    return LaunchFusedRowsImpl<T, ROWS, true>(device_prop, use_pdl, stream, config, tensors, g,
                                             partial, inv_rms, down, up);
  }
  return LaunchFusedRowsImpl<T, ROWS>(device_prop, use_pdl, stream, config, tensors, g,
                                     partial, inv_rms, down, up);
}

}  // namespace

Status LaunchPackHyperConnectionWeight(void* packed, const void* source, bool up, cudaStream_t stream) {
  PackHcWeightKernel<<<160, kThreads, 0, stream>>>(
      static_cast<uint4*>(packed), static_cast<const uint16_t*>(source), up);
  return CUDA_CALL(cudaGetLastError());
}

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
                                            float* scratch) {
  if constexpr (sizeof(T) != 2) {
    ORT_UNUSED_PARAMETER(device_prop);
    ORT_UNUSED_PARAMETER(stream);
    ORT_UNUSED_PARAMETER(config);
    ORT_UNUSED_PARAMETER(tensors);
    ORT_UNUSED_PARAMETER(scratch);
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "fused HyperConnectionLowRankMix requires a 16-bit type");
  } else {
    ORT_RETURN_IF_NOT(CanUseFusedHyperConnectionLowRankMix<T>(device_prop, config),
                      "unsupported problem for the fused HyperConnectionLowRankMix path");
    const FusedGeometry g = GetFusedGeometry(config);
    float* partial = scratch;
    float* inv_rms = scratch + static_cast<size_t>(config.rows) * g.splits * g.partial_stride;

    DownParams down{config.rows, config.branches, config.hidden, config.rank, g.pieces, g.slice,
                    g.splits, g.partial_stride, config.epsilon};
    UpParams up{config.rows, config.branches, config.hidden, config.rank, g.splits, g.partial_stride,
                config.silu_alpha, config.reduction_scale, config.gate_div, config.gate_mul};
    const bool use_pdl = device_prop.major >= 9 && config.weights_ready;
    if (config.rows == 1) {
      return LaunchFusedRows<T, 1>(device_prop, use_pdl, stream, config, tensors, g, partial, inv_rms, down, up);
    }
    static const int decode_rows = ParseEnvironmentVariableWithDefault<int>("ORT_HC_DECODE_ROWS", 2);
    ORT_ENFORCE(decode_rows == 1 || decode_rows == 2, "ORT_HC_DECODE_ROWS must be 1 or 2");
    if (config.rows == 4 && decode_rows == 1) {
      return LaunchFusedRows<T, 1>(device_prop, use_pdl, stream, config, tensors, g, partial, inv_rms, down, up);
    }
    return LaunchFusedRows<T, 2>(device_prop, use_pdl, stream, config, tensors, g, partial, inv_rms, down, up);
  }
}

template <typename T>
Status LaunchHyperConnectionLowRankMixUnfused(cublasHandle_t cublas, cudaStream_t stream,
                                              const HyperConnectionLowRankMixConfig& config,
                                              const HyperConnectionLowRankMixTensors<T>& tensors,
                                              T* hn, T* down, T* up, T* inject) {
  const int64_t rows = config.rows;
  const int64_t K = static_cast<int64_t>(config.branches) * config.hidden;
  const int64_t groups = rows * config.branches;
  ORT_RETURN_IF_NOT(groups <= std::numeric_limits<int>::max(), "CUDA launch requires too many blocks");

  if (tensors.prev_block_output != nullptr) {
    NormKernel<T, true><<<static_cast<int>(groups), kThreads, 0, stream>>>(
        tensors.streams, tensors.prev_block_output, tensors.prev_post_gate, tensors.norm_scale, hn,
        tensors.streams_out, config.branches, config.hidden, config.epsilon);
  } else {
    NormKernel<T, false><<<static_cast<int>(groups), kThreads, 0, stream>>>(
        tensors.streams, nullptr, nullptr, tensors.norm_scale, hn, nullptr, config.branches, config.hidden,
        config.epsilon);
  }
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  ORT_RETURN_IF_ERROR(Gemm<T>(cublas, rows, config.rank, K, hn, tensors.down_weight, down));
  ORT_RETURN_IF_ERROR(Gemm<T>(cublas, rows, config.branches, K, hn, tensors.inject_weight, inject));

  const int64_t down_count = rows * config.rank;
  ScaledSiluKernel<T><<<static_cast<int>(BlocksFor(down_count)), kThreads, 0, stream>>>(
      down, down_count, config.silu_alpha);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  ORT_RETURN_IF_ERROR(Gemm<T>(cublas, rows, K, config.rank, down, tensors.up_weight, up));

  const int64_t pre_count = rows * config.hidden;
  ORT_RETURN_IF_NOT(BlocksFor(pre_count) <= std::numeric_limits<int>::max(),
                    "CUDA launch requires too many blocks");
  PreMixKernel<T><<<static_cast<int>(BlocksFor(pre_count)), kThreads, 0, stream>>>(
      hn, up, tensors.pre_output, pre_count, config.branches, config.hidden, config.reduction_scale);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  PostGateKernel<T><<<static_cast<int>(BlocksFor(groups)), kThreads, 0, stream>>>(
      inject, tensors.post_gate, groups, config.gate_div, config.gate_mul);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                           \
  template bool CanUseFusedHyperConnectionLowRankMix<T>(const cudaDeviceProp&,                   \
                                                        const HyperConnectionLowRankMixConfig&); \
  template Status LaunchHyperConnectionLowRankMixFused<T>(                                       \
      const cudaDeviceProp&, cudaStream_t, const HyperConnectionLowRankMixConfig&,               \
      const HyperConnectionLowRankMixTensors<T>&, float*);                                       \
  template Status LaunchHyperConnectionLowRankMixUnfused<T>(                                     \
      cublasHandle_t, cudaStream_t, const HyperConnectionLowRankMixConfig&,                      \
      const HyperConnectionLowRankMixTensors<T>&, T*, T*, T*, T*);

INSTANTIATE(float)
INSTANTIATE(half)
INSTANTIATE(__nv_bfloat16)

#undef INSTANTIATE

}  // namespace onnxruntime::contrib::cuda
