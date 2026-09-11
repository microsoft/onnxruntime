// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/matmul_small_n_gemv.h"

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {
namespace {

constexpr int kRowsPerLaunch = 8;
constexpr int kMaxSupportedM = 64;
constexpr int kMaxDefaultDispatchM = 1;
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

template <int M>
Status Launch(cudaStream_t stream, const half* a, const half* b, half* c, int n, int k,
              float* ws, unsigned int* counter, int split_k) {
  const dim3 grid(static_cast<unsigned>((n + kTx - 1) / kTx), static_cast<unsigned>(split_k));
  SmallNGemvSplitKKernel<M><<<grid, kThreads, 0, stream>>>(a, b, c, n, k, ws, counter, split_k);
  return CUDA_CALL(cudaGetLastError());
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
  return static_cast<size_t>(SmallNGemvSplitK(n, k)) * rows * n;
}

size_t SmallNGemvCounterElements(int n) {
  return static_cast<size_t>((n + kTx - 1) / kTx);
}

bool CanUseSmallNGemv(int64_t m, int64_t n, int64_t k, const void* a, const void* b, const void* c) {
  if (m < 1 || m > kMaxDefaultDispatchM || n < 1 || n > kMaxN || k < kMinK) return false;
  if (k > (1 << 20)) return false;
  const uintptr_t align = reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
                          reinterpret_cast<uintptr_t>(c);
  return (align % 8) == 0;
}

Status LaunchSmallNGemv(cudaStream_t stream, const half* a, const half* b, half* c,
                        int m, int n, int k, float* ws, unsigned int* counter) {
  ORT_RETURN_IF_NOT(m >= 1 && m <= kMaxSupportedM,
                    "SmallNGemv supports M in [1, ", kMaxSupportedM, "], got ", m, ".");
  const int split_k = SmallNGemvSplitK(n, k);
  for (int row = 0; row < m; row += kRowsPerLaunch) {
    const int rows = (m - row < kRowsPerLaunch) ? m - row : kRowsPerLaunch;
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
        counter, 0, SmallNGemvCounterElements(n) * sizeof(unsigned int), stream));
    switch (rows) {
      case 1:
        ORT_RETURN_IF_ERROR(Launch<1>(stream, a + static_cast<size_t>(row) * k, b,
                                      c + static_cast<size_t>(row) * n, n, k, ws, counter, split_k));
        break;
      case 2:
        ORT_RETURN_IF_ERROR(Launch<2>(stream, a + static_cast<size_t>(row) * k, b,
                                      c + static_cast<size_t>(row) * n, n, k, ws, counter, split_k));
        break;
      case 3:
        ORT_RETURN_IF_ERROR(Launch<3>(stream, a + static_cast<size_t>(row) * k, b,
                                      c + static_cast<size_t>(row) * n, n, k, ws, counter, split_k));
        break;
      case 4:
        ORT_RETURN_IF_ERROR(Launch<4>(stream, a + static_cast<size_t>(row) * k, b,
                                      c + static_cast<size_t>(row) * n, n, k, ws, counter, split_k));
        break;
      case 5:
        ORT_RETURN_IF_ERROR(Launch<5>(stream, a + static_cast<size_t>(row) * k, b,
                                      c + static_cast<size_t>(row) * n, n, k, ws, counter, split_k));
        break;
      case 6:
        ORT_RETURN_IF_ERROR(Launch<6>(stream, a + static_cast<size_t>(row) * k, b,
                                      c + static_cast<size_t>(row) * n, n, k, ws, counter, split_k));
        break;
      case 7:
        ORT_RETURN_IF_ERROR(Launch<7>(stream, a + static_cast<size_t>(row) * k, b,
                                      c + static_cast<size_t>(row) * n, n, k, ws, counter, split_k));
        break;
      case 8:
        ORT_RETURN_IF_ERROR(Launch<8>(stream, a + static_cast<size_t>(row) * k, b,
                                      c + static_cast<size_t>(row) * n, n, k, ws, counter, split_k));
        break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SmallNGemv: unsupported row chunk ", rows);
    }
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
