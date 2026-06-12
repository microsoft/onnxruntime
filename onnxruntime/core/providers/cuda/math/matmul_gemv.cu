// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/matmul_gemv.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace cuda {

namespace {

// Tile transpose: dst[c, r] = src[r, c]. src is [rows, cols], dst is [cols, rows].
template <typename T, int TILE>
__global__ void TransposeKernel(T* __restrict__ dst, const T* __restrict__ src, int rows, int cols) {
  __shared__ T tile[TILE][TILE + 1];
  int r = blockIdx.y * TILE + threadIdx.y;
  int c = blockIdx.x * TILE + threadIdx.x;
  if (r < rows && c < cols) {
    tile[threadIdx.y][threadIdx.x] = src[static_cast<int64_t>(r) * cols + c];
  }
  __syncthreads();
  int rt = blockIdx.x * TILE + threadIdx.y;  // transposed row index (into cols)
  int ct = blockIdx.y * TILE + threadIdx.x;  // transposed col index (into rows)
  if (rt < cols && ct < rows) {
    dst[static_cast<int64_t>(rt) * rows + ct] = tile[threadIdx.x][threadIdx.y];
  }
}

// One block per output column n. Threads split K, reduce in shared, write Y[n].
// Bt is [N, K] row-major: row n (length K) is contiguous, so the per-step reads
// across threads are coalesced.
template <typename T, int THREADS>
__global__ void GemvM1Kernel(T* __restrict__ y, const T* __restrict__ a,
                             const T* __restrict__ b_transposed, int n, int k, float alpha) {
  const int col = blockIdx.x;
  if (col >= n) return;
  const int tid = threadIdx.x;
  const T* b_row = b_transposed + static_cast<int64_t>(col) * k;

  float acc = 0.0f;
  for (int i = tid; i < k; i += THREADS) {
    acc += static_cast<float>(a[i]) * static_cast<float>(b_row[i]);
  }

  __shared__ float sh[THREADS];
  sh[tid] = acc;
  __syncthreads();
#pragma unroll
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) sh[tid] += sh[tid + s];
    __syncthreads();
  }
  if (tid == 0) {
    y[col] = static_cast<T>(alpha * sh[0]);
  }
}

}  // namespace

template <typename T>
void TransposeForGemv(cudaStream_t stream, T* dst, const T* src, int rows, int cols) {
  constexpr int TILE = 32;
  dim3 block(TILE, TILE);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  TransposeKernel<T, TILE><<<grid, block, 0, stream>>>(dst, src, rows, cols);
}

template <typename T>
void MatMulGemvM1(cudaStream_t stream, T* y, const T* a, const T* b_transposed,
                  int n, int k, float alpha) {
  constexpr int THREADS = 128;
  GemvM1Kernel<T, THREADS><<<n, THREADS, 0, stream>>>(y, a, b_transposed, n, k, alpha);
}

#define INSTANTIATE_GEMV(T)                                                \
  template void TransposeForGemv<T>(cudaStream_t, T*, const T*, int, int); \
  template void MatMulGemvM1<T>(cudaStream_t, T*, const T*, const T*, int, int, float)

INSTANTIATE_GEMV(OrtToCudaType<MLFloat16>::type);
INSTANTIATE_GEMV(OrtToCudaType<BFloat16>::type);

#undef INSTANTIATE_GEMV

}  // namespace cuda
}  // namespace onnxruntime
