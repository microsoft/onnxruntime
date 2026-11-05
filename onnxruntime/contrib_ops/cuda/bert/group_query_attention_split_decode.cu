// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/group_query_attention_split_decode.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kHead = kGqaSplitDecodeHeadSize;
constexpr int kGroup = kGqaSplitDecodeGroup;
constexpr int kWarps = 8;  // warps per CTA, each streaming its own 8-key steps
constexpr int kThreads = kWarps * 32;
constexpr int kStep = 8;            // keys per warp step (one mma n-tile)
constexpr int kRecord = kHead + 2;  // per split and head: partial output, max, sum

static_assert(kHead == 128 && kGroup == 8, "fragment layout below assumes 8 heads of 128 channels");

// grid (num_heads + 2 * kv_num_heads, batch), block 32. Lane l owns channels 4l..4l+3; with the
// half-split rotation the partner of channel c is c +- head_size / 2, i.e. lane l ^ 16.
__global__ void __launch_bounds__(32) GqaSplitDecodePrepKernel(
    const half* __restrict__ packed_qkv, const half* __restrict__ q_in, const half* __restrict__ k_in,
    const half* __restrict__ v_in, half* __restrict__ q_out, half* __restrict__ k_cache, half* __restrict__ v_cache,
    const int* __restrict__ past_seq_lens, const int64_t* __restrict__ position_ids,
    const half* __restrict__ cos_cache, const half* __restrict__ sin_cache, int rope_max_pos, int do_rotary,
    const half* __restrict__ q_norm, const half* __restrict__ k_norm, float q_eps, float k_eps,
    unsigned int* __restrict__ counters, int num_heads, int kv_num_heads, int capacity) {
  static_assert(kHead == 128, "lane l owns channels 4l..4l+3");
  const int head = blockIdx.x, b = blockIdx.y, lane = threadIdx.x;
  const int packed_width = (num_heads + 2 * kv_num_heads) * kHead;
  int type, n;
  const half* src;
  if (head < num_heads) {
    type = 0;
    n = head;
    src = packed_qkv ? packed_qkv + static_cast<int64_t>(b) * packed_width + n * kHead
                     : q_in + (static_cast<int64_t>(b) * num_heads + n) * kHead;
  } else if (head < num_heads + kv_num_heads) {
    type = 1;
    n = head - num_heads;
    src = packed_qkv ? packed_qkv + static_cast<int64_t>(b) * packed_width + (num_heads + n) * kHead
                     : k_in + (static_cast<int64_t>(b) * kv_num_heads + n) * kHead;
  } else {
    type = 2;
    n = head - num_heads - kv_num_heads;
    src = packed_qkv ? packed_qkv + static_cast<int64_t>(b) * packed_width + (num_heads + kv_num_heads + n) * kHead
                     : v_in + (static_cast<int64_t>(b) * kv_num_heads + n) * kHead;
  }
  const uint2 raw = reinterpret_cast<const uint2*>(src)[lane];
  half x[4];
  memcpy(x, &raw, sizeof(raw));
  const int past = past_seq_lens[b];
  if (type < 2) {
    const half* w = type == 0 ? q_norm : k_norm;
    if (w != nullptr) {
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float f = __half2float(x[i]);
        ss += f * f;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float inv_rms = rsqrtf(ss / kHead + (type == 0 ? q_eps : k_eps));
#pragma unroll
      for (int i = 0; i < 4; ++i) x[i] = __float2half(__half2float(x[i]) * inv_rms * __half2float(w[lane * 4 + i]));
    }
    if (do_rotary) {
      const int64_t pos = position_ids != nullptr ? position_ids[b] : static_cast<int64_t>(past);
      if (pos >= 0 && pos < rope_max_pos) {
        half pair[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          pair[i] = __ushort_as_half(__shfl_xor_sync(0xffffffffu, __half_as_ushort(x[i]), 16));
        }
        const float sign = lane < 16 ? -1.f : 1.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int c = (lane * 4 + i) % (kHead / 2);
          const float cf = __half2float(cos_cache[pos * (kHead / 2) + c]);
          const float sf = __half2float(sin_cache[pos * (kHead / 2) + c]);
          x[i] = __float2half(__half2float(x[i]) * cf + sign * __half2float(pair[i]) * sf);
        }
      }
    }
  }
  uint2 out;
  memcpy(&out, x, sizeof(out));
  if (type == 0) {
    reinterpret_cast<uint2*>(q_out + (static_cast<int64_t>(b) * num_heads + n) * kHead)[lane] = out;
  } else if (past >= 0 && past < capacity) {
    half* cache = type == 1 ? k_cache : v_cache;
    const int64_t row = (static_cast<int64_t>(b) * kv_num_heads + n) * capacity + past;
    reinterpret_cast<uint2*>(cache + row * kHead)[lane] = out;
  }
  // The attention kernel that follows on the same stream counts finished splits per (batch, kv head).
  if (type == 1 && lane == 0) counters[b * kv_num_heads + n] = 0;
}

// Number of splits used for a context of `total` keys; never more than the launched max_splits.
__device__ __forceinline__ int SplitCount(int total, int max_splits, int keys_per_split) {
  return min(max_splits, max(1, (total + keys_per_split - 1) / keys_per_split));
}

// D (rows 0-7) += A * B for m16n8k16 with rows 8-15 of A zero: a_lo = A[g][2r..2r+1], a_hi = A[g][2r+8..2r+9],
// b_lo = B[2r..2r+1][g], b_hi = B[2r+8..2r+9][g] (g = lane / 4, r = lane % 4); d = D[g][2r..2r+1].
__device__ __forceinline__ void MmaK16(float& d0, float& d1, uint32_t a_lo, uint32_t a_hi, uint32_t b_lo,
                                       uint32_t b_hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float d2 = 0.f, d3 = 0.f;
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a_lo), "r"(0u), "r"(a_hi), "r"(0u), "r"(b_lo), "r"(b_hi));
#endif
}

// D (rows 0-7) += A * B for m16n8k8 with rows 8-15 of A zero: a = A[g][2r..2r+1], b = B[2r..2r+1][g].
__device__ __forceinline__ void MmaK8(float& d0, float& d1, uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float d2 = 0.f, d3 = 0.f;
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a), "r"(0u), "r"(b));
#endif
}

// 16-byte global -> shared copy that bypasses L1; with `valid` false the destination is zero-filled.
__device__ __forceinline__ void CpAsync16(uint4* dst, const void* src, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src), "r"(valid ? 16 : 0)
               : "memory");
#endif
}

__device__ __forceinline__ void CpAsyncCommit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <int kPending>
__device__ __forceinline__ void CpAsyncWait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;" ::"n"(kPending) : "memory");
#endif
}

__device__ __forceinline__ uint32_t PackHalf2(float lo, float hi) {
  const half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// Output channel held in o[j][e] by a lane with r = lane % 4 (see the P * V layout below).
__device__ __forceinline__ int OutChannel(int j, int r, int e) {
  return (j < 8 ? 0 : 64) + 8 * (2 * r + e) + (j & 7);
}

// grid (kv_num_heads, max_splits, batch), block kThreads. The grid is fixed per capture; the CTAs
// beyond the split count of their batch (from its context length) exit at once.
//
// Each warp runs an online softmax over its own 8-key steps of the split with tensor-core mma; the
// 8 query heads of the KV head are rows 0-7 of the A operand (rows 8-15 are zero):
//  * Q * K^T: the reduction over channels may be permuted as long as Q and K agree, so lane (g, r)
//    (g = lane / 4, r = lane % 4) uses channels 32 * blk + 8 * r .. + 7 of query head g and of key g of
//    the step as two k16 slices.
//  * P * V: the score accumulators of lane (g, r) are P[g][2r..2r+1], which is the m16n8k8 A fragment.
//    The B fragment needs V[2r..2r+1][n] for output tile j; with n -> channel (j < 8 ? 0 : 64) + 8 * g +
//    (j & 7) lane (g, r) gets all of it from two 16-byte words per key row, so outputs come out permuted
//    (OutChannel).
// K and V reach the fragments through a per-warp ring of kStages steps in shared memory filled with
// cp.async, kStages - 1 steps ahead. The copy moves whole 256-byte key rows, 16 lanes per row: loading the
// fragments straight from global memory touches only 64 bytes of each 128-byte line of K per request, which
// caps the streaming rate of an SM well below its share of DRAM bandwidth.
template <int kStages>
__global__ void __launch_bounds__(kThreads, 1) GqaSplitDecodeKernel(const half* __restrict__ q,
                                                                    const half* __restrict__ k_cache,
                                                                    const half* __restrict__ v_cache,
                                                                    half* __restrict__ out,
                                                                    const int* __restrict__ total_seq_lens,
                                                                    float scale, int capacity,
                                                                    float* __restrict__ partials,
                                                                    unsigned int* __restrict__ counters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // The ring is [warp][stage][K, V][key row][16-byte chunk]; after the main loop the same memory holds the
  // per-warp outputs (s_red). The 128-byte alignment matters: at the
  // default offset after the static shared variables every cp.async row straddles two 128-byte lines, which
  // cost ~13% of the streaming rate on A100.
  extern __shared__ __align__(128) uint4 s_ring[];
  __shared__ float s_m[kWarps][kGroup], s_l[kWarps][kGroup];
  __shared__ float s_w[kGroup][kGqaSplitDecodeMaxSplits];
  __shared__ float s_inv[kGroup];
  __shared__ bool s_last;
  auto& s_red = *reinterpret_cast<float(*)[kWarps][kGroup][kHead]>(s_ring);

  const int kvh = blockIdx.x, split = blockIdx.y, b = blockIdx.z, tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32, g = lane / 4, r = lane % 4;
  const int kv_heads = gridDim.x, max_splits = gridDim.y;
  const int total = total_seq_lens[b];
  const int splits = SplitCount(total, max_splits, kGqaSplitDecodeMinKeysPerSplit);
  if (split >= splits) return;
  const int steps = (total + kStep - 1) / kStep;
  const int chunk = (steps + splits - 1) / splits * kStep;
  const int t0 = split * chunk, t1 = min(total, t0 + chunk);

  const int bh = b * kv_heads + kvh;
  const half* qrow = q + (static_cast<int64_t>(bh) * kGroup + g) * kHead;
  uint4 qf[4];
#pragma unroll
  for (int blk = 0; blk < 4; ++blk) qf[blk] = __ldg(reinterpret_cast<const uint4*>(qrow + 32 * blk + 8 * r));
  const int64_t cache_off = static_cast<int64_t>(bh) * capacity * kHead;
  const half* kb = k_cache + cache_off;
  const half* vb = v_cache + cache_off;
  uint4* ring = s_ring + warp * kStages * 2 * kStep * 16;

  // Copies the 8 key rows of K and of V from `base` into ring stage `stage`. Chunks are XOR-swizzled within
  // their 128-byte line so the fragment reads below are free of bank conflicts: K by (row & 1) << 2, V by
  // ((row >> 1) & 3) << 1. Rows past the context are zero-filled: uninitialized cache memory could hold NaN,
  // and 0 * NaN = NaN.
  auto issue = [&](int base, int stage) {
    uint4* dk = ring + stage * 2 * kStep * 16;
    uint4* dv = dk + kStep * 16;
#pragma unroll
    for (int j = 0; j < kStep * 16 / 32; ++j) {
      const int c = lane + 32 * j, row = c >> 4, col = c & 15;
      const bool ok = base + row < t1;
      const int64_t src = static_cast<int64_t>(ok ? base + row : t0) * kHead + 8 * col;
      CpAsync16(dk + row * 16 + (col ^ ((row & 1) << 2)), kb + src, ok);
      CpAsync16(dv + row * 16 + (col ^ (((row >> 1) & 3) << 1)), vb + src, ok);
    }
  };

  float o[16][2];
#pragma unroll
  for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = 0.f;
  float m = -FLT_MAX, l = 0.f;  // running max of row g (same in the 4 lanes of the row), lane-partial sum

  constexpr int kWarpStride = kWarps * kStep;
  const int base0 = t0 + warp * kStep;
  // One commit group per step, empty past the end: once step i + kStages - 1 is committed,
  // wait_group<kStages - 1> means step i has landed.
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (base0 + s * kWarpStride < t1) issue(base0 + s * kWarpStride, s);
    CpAsyncCommit();
  }
  int stage = 0;
  for (int base = base0; base < t1; base += kWarpStride) {
    __syncwarp();  // every lane is done reading the stage refilled below
    const int ahead = base + (kStages - 1) * kWarpStride;
    if (ahead < t1) issue(ahead, stage == 0 ? kStages - 1 : stage - 1);
    CpAsyncCommit();
    CpAsyncWait<kStages - 1>();
    __syncwarp();  // the step was copied by other lanes of the warp
    // kr[blk] = K[g][32 * blk + 8r ..]; vr[2e + i] = V[2r + e][(i ? 64 : 0) + 8g ..].
    uint4 kr[4], vr[4];
    const uint4* sk = ring + stage * 2 * kStep * 16;
    const uint4* sv = sk + kStep * 16;
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) kr[blk] = sk[g * 16 + ((4 * blk + r) ^ ((g & 1) << 2))];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
#pragma unroll
      for (int i = 0; i < 2; ++i) vr[2 * e + i] = sv[(2 * r + e) * 16 + ((8 * i + g) ^ (r << 1))];
    }
    stage = stage == kStages - 1 ? 0 : stage + 1;

    // Two accumulator chains halve the dependent mma latency.
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) {
      MmaK16(s0, s1, qf[blk].x, qf[blk].y, kr[blk].x, kr[blk].y);
      MmaK16(s2, s3, qf[blk].z, qf[blk].w, kr[blk].z, kr[blk].w);
    }
    s0 = base + 2 * r < t1 ? (s0 + s2) * scale : -INFINITY;
    s1 = base + 2 * r + 1 < t1 ? (s1 + s3) * scale : -INFINITY;
    float mx = fmaxf(s0, s1);
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float m_new = fmaxf(m, mx);  // finite: key `base` is in range
    const float corr = __expf(m - m_new);
    const float p0 = __expf(s0 - m_new), p1 = __expf(s1 - m_new);
    l = l * corr + p0 + p1;
    m = m_new;
    const uint32_t pa = PackHalf2(p0, p1);
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(vr);  // vr[2 * e + i] word w at v32[4 * (2e + i) + w]
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int i = j < 8 ? 0 : 1, w = (j & 7) / 2;
      const uint32_t bv = __byte_perm(v32[4 * i + w], v32[4 * (2 + i) + w], (j & 1) ? 0x7632 : 0x5410);
      o[j][0] *= corr;
      o[j][1] *= corr;
      MmaK8(o[j][0], o[j][1], pa, bv);
    }
  }
  CpAsyncWait<0>();
  l += __shfl_xor_sync(0xffffffffu, l, 1);
  l += __shfl_xor_sync(0xffffffffu, l, 2);

  // Combine the warps of the CTA at their common max.
  if (r == 0) {
    s_m[warp][g] = m;
    s_l[warp][g] = l;
  }
  __syncthreads();  // also: every warp is done with its ring, which s_red overwrites
  float m_all = -FLT_MAX;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) m_all = fmaxf(m_all, s_m[w][g]);
  const float f = m == -FLT_MAX ? 0.f : __expf(m - m_all);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) s_red[warp][g][OutChannel(j, r, e)] = o[j][e] * f;
  }
  __syncthreads();

  if (splits == 1) {
    // The CTA saw every key: normalize and write the output, with no partial, counter or merge.
    if (tid < kGroup) {
      float ma = -FLT_MAX, la = 0.f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) ma = fmaxf(ma, s_m[w][tid]);
#pragma unroll
      for (int w = 0; w < kWarps; ++w) la += s_m[w][tid] == -FLT_MAX ? 0.f : s_l[w][tid] * __expf(s_m[w][tid] - ma);
      s_inv[tid] = 1.f / la;
    }
    __syncthreads();
    half* ob = out + static_cast<int64_t>(bh) * kGroup * kHead;
    for (int i = tid; i < kGroup * kHead; i += kThreads) {
      float acc = 0.f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) acc += (&s_red[w][0][0])[i];
      ob[i] = __float2half(acc * s_inv[i / kHead]);
    }
    return;
  }

  // Publish this split's partial result, then let the last CTA of (b, kvh) merge all splits.
  // partials per (b, kvh): outputs [max_splits][kGroup][kHead], then stats [max_splits][kGroup][2].
  float* outs = partials + static_cast<int64_t>(bh) * max_splits * kGroup * kRecord;
  float* stats = outs + max_splits * kGroup * kHead;
  for (int i = tid; i < kGroup * kHead; i += kThreads) {
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) acc += (&s_red[w][0][0])[i];
    outs[split * kGroup * kHead + i] = acc;
  }
  if (tid < kGroup) {
    float ma = -FLT_MAX, la = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) ma = fmaxf(ma, s_m[w][tid]);
#pragma unroll
    for (int w = 0; w < kWarps; ++w) la += s_m[w][tid] == -FLT_MAX ? 0.f : s_l[w][tid] * __expf(s_m[w][tid] - ma);
    stats[(split * kGroup + tid) * 2] = ma;
    stats[(split * kGroup + tid) * 2 + 1] = la;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(&counters[bh], 1u) == static_cast<unsigned int>(splits - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  // Split weights: warp handles heads warp, warp + kWarps, ...; lane l splits l and l + 32.
  static_assert(kGqaSplitDecodeMaxSplits <= 64, "merge layout");
  for (int gm = warp; gm < kGroup; gm += kWarps) {
    float ms[2], ls[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int j = lane + 32 * i;
      ms[i] = j < splits ? __ldcg(stats + (j * kGroup + gm) * 2) : -FLT_MAX;
      ls[i] = j < splits ? __ldcg(stats + (j * kGroup + gm) * 2 + 1) : 0.f;
    }
    float mx = fmaxf(ms[0], ms[1]);
#pragma unroll
    for (int x = 16; x > 0; x >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, x));
    float wt[2], sum = 0.f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      wt[i] = ms[i] == -FLT_MAX ? 0.f : __expf(ms[i] - mx);
      sum += ls[i] * wt[i];
    }
#pragma unroll
    for (int x = 16; x > 0; x >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, x);
    const float inv = 1.f / sum;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int j = lane + 32 * i;
      if (j < splits) s_w[gm][j] = wt[i] * inv;
    }
  }
  __syncthreads();

  // Thread t produces outputs 4t .. 4t + 3 (and + 4 * kThreads, ...) with float4 loads.
  half* ob = out + static_cast<int64_t>(bh) * kGroup * kHead;
  constexpr int kMergeBatch = 16;
  for (int o4 = tid; o4 < kGroup * kHead / 4; o4 += kThreads) {
    const int gm = o4 * 4 / kHead;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int j0 = 0; j0 < splits; j0 += kMergeBatch) {
      float4 v[kMergeBatch];
#pragma unroll
      for (int jj = 0; jj < kMergeBatch; ++jj) {
        v[jj] = j0 + jj < splits ? __ldcg(reinterpret_cast<const float4*>(outs + (j0 + jj) * kGroup * kHead) + o4)
                                 : make_float4(0.f, 0.f, 0.f, 0.f);
      }
#pragma unroll
      for (int jj = 0; jj < kMergeBatch; ++jj) {
        const float wt = j0 + jj < splits ? s_w[gm][j0 + jj] : 0.f;
        acc.x = fmaf(v[jj].x, wt, acc.x);
        acc.y = fmaf(v[jj].y, wt, acc.y);
        acc.z = fmaf(v[jj].z, wt, acc.z);
        acc.w = fmaf(v[jj].w, wt, acc.w);
      }
    }
    half2* dst = reinterpret_cast<half2*>(ob + o4 * 4);
    dst[0] = __floats2half2_rn(acc.x, acc.y);
    dst[1] = __floats2half2_rn(acc.z, acc.w);
  }
#endif
}

constexpr size_t RingBytes(int stages) { return sizeof(uint4) * kWarps * stages * 2 * kStep * 16; }

// Ring depth for this device: 4 steps (128 KB) where the per-block opt-in shared memory allows it (sm_80,
// sm_87, sm_90), else 3 (96 KB, within the 99 KB of sm_86, sm_89 and sm_120).
int RingStages() {
  int device = 0, optin = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) {
    return 3;
  }
  return optin >= static_cast<int>(RingBytes(4)) + 4096 ? 4 : 3;  // + the static shared variables
}

template <int kStages>
Status Launch(cudaStream_t stream, dim3 grid, const half* q, const half* k_cache, const half* v_cache, half* output,
              const int* total_seq_lens, float scale, int capacity, float* partials, unsigned int* counters) {
  constexpr int kBytes = static_cast<int>(RingBytes(kStages));
  static_assert(sizeof(float) * kWarps * kGroup * kHead <= kBytes, "s_red must fit in the ring");
  ORT_RETURN_IF_ERROR(CUDA_CALL(
      cudaFuncSetAttribute(GqaSplitDecodeKernel<kStages>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes)));
  GqaSplitDecodeKernel<kStages><<<grid, kThreads, kBytes, stream>>>(q, k_cache, v_cache, output, total_seq_lens,
                                                                     scale, capacity, partials, counters);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace

int GqaSplitDecodeMaxSplits(int batch_size, int kv_num_heads, int capacity, int num_sms) {
  // One CTA per SM at a time. While the (batch, kv head) pairs fit in one wave, split them to fill it but not
  // beyond: a second wave or a merge costs more than it saves. With more pairs than SMs, one split per pair
  // unless the last wave would keep fewer than a third of the SMs busy (too few to saturate DRAM), in which
  // case two splits even out the waves.
  const int heads = batch_size * kv_num_heads;
  int splits = 1;
  if (heads <= num_sms) {
    splits = num_sms / heads;
  } else {
    const int tail = heads % num_sms;
    splits = tail != 0 && tail < num_sms / 3 ? 2 : 1;
  }
  const int by_capacity = (capacity + kGqaSplitDecodeMinKeysPerSplit - 1) / kGqaSplitDecodeMinKeysPerSplit;
  return std::max(1, std::min({kGqaSplitDecodeMaxSplits, splits, by_capacity}));
}

size_t GqaSplitDecodePartialsBytes(int batch_size, int kv_num_heads, int max_splits) {
  return sizeof(float) * static_cast<size_t>(batch_size) * kv_num_heads * max_splits * kGroup * kRecord;
}

size_t GqaSplitDecodeCountersBytes(int batch_size, int kv_num_heads) {
  return sizeof(unsigned int) * static_cast<size_t>(batch_size) * kv_num_heads;
}

Status LaunchGqaSplitDecodePrep(cudaStream_t stream, const half* packed_qkv, const half* q, const half* k,
                                const half* v, half* q_out, half* k_cache, half* v_cache, const int* past_seq_lens,
                                const int64_t* position_ids, const half* cos_cache, const half* sin_cache,
                                int rope_max_pos, bool do_rotary, const half* q_norm, const half* k_norm, float q_eps,
                                float k_eps, unsigned int* counters, int batch_size, int num_heads, int kv_num_heads,
                                int capacity) {
  GqaSplitDecodePrepKernel<<<dim3(num_heads + 2 * kv_num_heads, batch_size), 32, 0, stream>>>(
      packed_qkv, q, k, v, q_out, k_cache, v_cache, past_seq_lens, position_ids, cos_cache, sin_cache, rope_max_pos,
      do_rotary ? 1 : 0, q_norm, k_norm, q_eps, k_eps, counters, num_heads, kv_num_heads, capacity);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchGqaSplitDecode(cudaStream_t stream, const half* q, const half* k_cache, const half* v_cache, half* output,
                            const int* total_seq_lens, float scale, int batch_size, int kv_num_heads, int capacity,
                            int max_splits, float* partials, unsigned int* counters) {
  const int stages = RingStages();
  const dim3 grid(kv_num_heads, max_splits, batch_size);
  return stages == 4 ? Launch<4>(stream, grid, q, k_cache, v_cache, output, total_seq_lens, scale, capacity, partials,
                                 counters)
                     : Launch<3>(stream, grid, q, k_cache, v_cache, output, total_seq_lens, scale, capacity, partials,
                                 counters);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
