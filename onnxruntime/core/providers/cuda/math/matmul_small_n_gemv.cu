// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/matmul_small_n_gemv.h"

#include <algorithm>

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {
namespace {

constexpr int kRowsPerLaunch = 8;
constexpr int kMaxSupportedM = 64;
constexpr int kMaxN = 1024;
constexpr int kMinK = 128;
constexpr int kThreads = 256;
constexpr int kTx = 32;  // one warp-wide column tile -> fully coalesced B reads
constexpr int kMaxSplitK = 32;

// grid = (ceil(n / kTx), split_k). Block (x, y) accumulates the K-slice
// [y * k_per, (y+1) * k_per) for columns [x * kTx, x * kTx + kTx) and stores the
// fp32 partials in `ws`. The last block to finish a column tile reduces the
// partials in slice order (deterministic) and writes the half output.
template <int M>
__global__ void SmallNGemvSplitKKernel(const half* __restrict__ a, const half* __restrict__ b,
                                       half* __restrict__ c, int n, int k,
                                       volatile float* __restrict__ ws, unsigned int* __restrict__ counter,
                                       int split_k) {
  constexpr int TY = kThreads / kTx;
  const int tx = static_cast<int>(threadIdx.x) % kTx;
  const int ty = static_cast<int>(threadIdx.x) / kTx;
  const int col = static_cast<int>(blockIdx.x) * kTx + tx;
  const bool active = col < n;

  const int k_per = (k + split_k - 1) / split_k;
  const int k0 = static_cast<int>(blockIdx.y) * k_per;
  const int k1 = min(k, k0 + k_per);

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.0f;

  if (active) {
    for (int kk = k0 + ty; kk < k1; kk += TY) {
      const float bv = __half2float(b[static_cast<size_t>(kk) * n + col]);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        acc[m] = fmaf(__half2float(a[static_cast<size_t>(m) * k + kk]), bv, acc[m]);
      }
    }
  }

  __shared__ float smem[kThreads * M];
#pragma unroll
  for (int m = 0; m < M; ++m) smem[threadIdx.x * M + m] = acc[m];
  __syncthreads();
#pragma unroll
  for (int s = TY / 2; s > 0; s >>= 1) {
    if (ty < s) {
      const int partner = ((ty + s) * kTx + tx) * M;
#pragma unroll
      for (int m = 0; m < M; ++m) smem[threadIdx.x * M + m] += smem[partner + m];
    }
    __syncthreads();
  }

  if (ty == 0 && active) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      ws[(static_cast<size_t>(blockIdx.y) * M + m) * n + col] = smem[threadIdx.x * M + m];
    }
  }

  // Make the partials visible before announcing this block is done.
  __threadfence();
  __syncthreads();

  __shared__ bool is_last;
  if (threadIdx.x == 0) {
    is_last = (atomicAdd(&counter[blockIdx.x], 1u) == static_cast<unsigned int>(split_k - 1));
  }
  __syncthreads();
  if (!is_last) return;

  if (ty == 0 && active) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float sum = 0.0f;
      for (int s = 0; s < split_k; ++s) sum += ws[(static_cast<size_t>(s) * M + m) * n + col];
      c[static_cast<size_t>(m) * n + col] = __float2half(sum);
    }
  }
}

// Vectorized variant for even N, K % 8 == 0 and 16-byte aligned A rows. Each lane owns two adjacent
// columns (one half2 of B per K row) and eight consecutive K rows per step, so one 16-byte broadcast
// load of A feeds 16 FMAs per row. The eight warps of a block stride over the block's K slice and are
// reduced through shared memory before the same deterministic last-block split-K reduction.
constexpr int kVecCols = 64;  // 32 lanes x half2
constexpr int kVecWarps = kThreads / 32;
constexpr int kVecKStep = 8;

template <int M>
__global__ void __launch_bounds__(kThreads)
    SmallNGemvVecSplitKKernel(const half* __restrict__ a, const half* __restrict__ b, half* __restrict__ c,
                              int n, int k, int k_per, float* __restrict__ ws,
                              unsigned int* __restrict__ counter, int split_k) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int col = static_cast<int>(blockIdx.x) * kVecCols + lane * 2;
  const int k0 = static_cast<int>(blockIdx.y) * k_per;
  const int k1 = min(k, k0 + k_per);

  float acc[M][2];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    acc[m][0] = 0.0f;
    acc[m][1] = 0.0f;
  }

  if (col < n) {
#pragma unroll 2
    for (int kk = k0 + warp * kVecKStep; kk < k1; kk += kVecWarps * kVecKStep) {
      float2 bv[kVecKStep];
#pragma unroll
      for (int r = 0; r < kVecKStep; ++r) {
        bv[r] = __half22float2(*reinterpret_cast<const half2*>(b + static_cast<size_t>(kk + r) * n + col));
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint4 packed = *reinterpret_cast<const uint4*>(a + static_cast<size_t>(m) * k + kk);
        const half2* av = reinterpret_cast<const half2*>(&packed);
#pragma unroll
        for (int p = 0; p < kVecKStep / 2; ++p) {
          const float2 af = __half22float2(av[p]);
          acc[m][0] = fmaf(af.x, bv[2 * p].x, acc[m][0]);
          acc[m][1] = fmaf(af.x, bv[2 * p].y, acc[m][1]);
          acc[m][0] = fmaf(af.y, bv[2 * p + 1].x, acc[m][0]);
          acc[m][1] = fmaf(af.y, bv[2 * p + 1].y, acc[m][1]);
        }
      }
    }
  }

  __shared__ float smem[kVecWarps][M][kVecCols];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    smem[warp][m][lane * 2] = acc[m][0];
    smem[warp][m][lane * 2 + 1] = acc[m][1];
  }
  __syncthreads();

  for (int index = static_cast<int>(threadIdx.x); index < M * kVecCols; index += kThreads) {
    const int m = index / kVecCols;
    const int tile_col = index % kVecCols;
    const int out_col = static_cast<int>(blockIdx.x) * kVecCols + tile_col;
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kVecWarps; ++w) sum += smem[w][m][tile_col];
    if (out_col < n) {
      if (split_k == 1) {
        c[static_cast<size_t>(m) * n + out_col] = __float2half(sum);
      } else {
        // Partials bypass L1 in both directions (st.cg / ld.cg) so the last block reads them from L2.
        __stcg(ws + (static_cast<size_t>(blockIdx.y) * M + m) * n + out_col, sum);
      }
    }
  }
  if (split_k == 1) return;

  // Make the partials visible before announcing this block is done.
  __threadfence();
  __syncthreads();

  __shared__ bool is_last;
  if (threadIdx.x == 0) {
    is_last = (atomicAdd(&counter[blockIdx.x], 1u) == static_cast<unsigned int>(split_k - 1));
  }
  __syncthreads();
  if (!is_last) return;

  for (int index = static_cast<int>(threadIdx.x); index < M * kVecCols; index += kThreads) {
    const int m = index / kVecCols;
    const int out_col = static_cast<int>(blockIdx.x) * kVecCols + index % kVecCols;
    if (out_col < n) {
      // Four independent partial sums keep several L2 loads in flight; the order is fixed, so the
      // result stays deterministic. split_k is a power of two, so it is 1, 2 or a multiple of 4.
      const float* partial = ws + static_cast<size_t>(m) * n + out_col;
      const size_t slice_stride = static_cast<size_t>(M) * n;
      float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      int s = 0;
      for (; s + 4 <= split_k; s += 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) sum[j] += __ldcg(partial + static_cast<size_t>(s + j) * slice_stride);
      }
      for (; s < split_k; ++s) sum[0] += __ldcg(partial + static_cast<size_t>(s) * slice_stride);
      c[static_cast<size_t>(m) * n + out_col] = __float2half((sum[0] + sum[1]) + (sum[2] + sum[3]));
    }
  }
}

template <int M>
Status Launch(cudaStream_t stream, const half* a, const half* b, half* c, int n, int k,
              float* ws, unsigned int* counter, int split_k) {
  const dim3 grid(static_cast<unsigned>((n + kTx - 1) / kTx), static_cast<unsigned>(split_k));
  SmallNGemvSplitKKernel<M><<<grid, kThreads, 0, stream>>>(a, b, c, n, k, ws, counter, split_k);
  return CUDA_CALL(cudaGetLastError());
}

int VecSplitK(int n, int k) {
  const int tiles = (n + kVecCols - 1) / kVecCols;
  int split_k = 1;
  while (split_k < kMaxSplitK && split_k * tiles < 128) split_k <<= 1;
  // Every slice must give each warp at least one K step.
  while (split_k > 1 && k / split_k < kVecWarps * kVecKStep) split_k >>= 1;
  return split_k;
}

template <int M>
Status LaunchVec(cudaStream_t stream, const half* a, const half* b, half* c, int n, int k,
                 float* ws, unsigned int* counter, int split_k) {
  // Round each slice up to whole K steps; K itself is a multiple of the step.
  const int k_per = ((k + split_k - 1) / split_k + kVecKStep - 1) / kVecKStep * kVecKStep;
  const dim3 grid(static_cast<unsigned>((n + kVecCols - 1) / kVecCols), static_cast<unsigned>(split_k));
  SmallNGemvVecSplitKKernel<M><<<grid, kThreads, 0, stream>>>(a, b, c, n, k, k_per, ws, counter, split_k);
  return CUDA_CALL(cudaGetLastError());
}

bool CanUseVec(int n, int k, const half* a, const half* b) {
  return (n % 2) == 0 && (k % kVecKStep) == 0 &&
         (reinterpret_cast<uintptr_t>(a) % 16) == 0 && (reinterpret_cast<uintptr_t>(b) % 4) == 0;
}

}  // namespace

int SmallNGemvSplitK(int n, int k) {
  const int tiles = (n + kTx - 1) / kTx;
  int split_k = 1;
  while (split_k < kMaxSplitK && split_k * tiles < 128) split_k <<= 1;
  // Every slice must own at least one warp's worth of K rows.
  while (split_k > 1 && k / split_k < kThreads / kTx) split_k >>= 1;
  return split_k;
}

size_t SmallNGemvWorkspaceElements(int m, int n, int k) {
  const int rows = m < kRowsPerLaunch ? m : kRowsPerLaunch;
  // Sized for whichever kernel variant the launch picks.
  const int split_k = std::max(SmallNGemvSplitK(n, k), VecSplitK(n, k));
  return static_cast<size_t>(split_k) * rows * n;
}

size_t SmallNGemvCounterElements(int n) {
  return static_cast<size_t>((n + kTx - 1) / kTx);
}

bool CanUseSmallNGemv(int64_t m, int64_t n, int64_t k, const void* a, const void* b, const void* c) {
  if (m < 1 || m > kMaxSupportedM || n < 1 || n > kMaxN || k < kMinK) return false;
  if (k > (1 << 20)) return false;
  const uintptr_t align = reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
                          reinterpret_cast<uintptr_t>(c);
  return (align % 8) == 0;
}

Status LaunchSmallNGemv(cudaStream_t stream, const half* a, const half* b, half* c,
                        int m, int n, int k, float* ws, unsigned int* counter) {
  ORT_RETURN_IF_NOT(m >= 1 && m <= kMaxSupportedM,
                    "SmallNGemv supports M in [1, ", kMaxSupportedM, "], got ", m, ".");
  const bool vec = CanUseVec(n, k, a, b);
  const int split_k = vec ? VecSplitK(n, k) : SmallNGemvSplitK(n, k);
  for (int row = 0; row < m; row += kRowsPerLaunch) {
    const int rows = (m - row < kRowsPerLaunch) ? m - row : kRowsPerLaunch;
    // The scalar kernel always takes the completion counter; the vector kernel only when split.
    if (split_k > 1 || !vec) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
          counter, 0, SmallNGemvCounterElements(n) * sizeof(unsigned int), stream));
    }
    const half* chunk_a = a + static_cast<size_t>(row) * k;
    half* chunk_c = c + static_cast<size_t>(row) * n;
#define ORT_SMALL_N_GEMV_CASE(R)                                                                    \
  case R:                                                                                           \
    ORT_RETURN_IF_ERROR(vec ? LaunchVec<R>(stream, chunk_a, b, chunk_c, n, k, ws, counter, split_k) \
                            : Launch<R>(stream, chunk_a, b, chunk_c, n, k, ws, counter, split_k));  \
    break;
    switch (rows) {
      ORT_SMALL_N_GEMV_CASE(1)
      ORT_SMALL_N_GEMV_CASE(2)
      ORT_SMALL_N_GEMV_CASE(3)
      ORT_SMALL_N_GEMV_CASE(4)
      ORT_SMALL_N_GEMV_CASE(5)
      ORT_SMALL_N_GEMV_CASE(6)
      ORT_SMALL_N_GEMV_CASE(7)
      ORT_SMALL_N_GEMV_CASE(8)
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SmallNGemv: unsupported row chunk ", rows);
    }
#undef ORT_SMALL_N_GEMV_CASE
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
