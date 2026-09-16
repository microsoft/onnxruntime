//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// Make generic operators for floating point types
/* This file contains:
   Generalized library calls
   kernels to be called for not supported data type
*/
// NV_TODO: optimize speed -- pass things needed in, optimize kernel speed, add half2
// NV_TODO: investigate cub support for half
#include "core/providers/cuda/curand_wrapper.h"
#include "core/providers/cuda/cu_inc/common.cuh"

#include <limits>

#define TRANS_TILE_DIM 32
#define BLOCK_ROWS 8
#define COPY_TILE_DIM 1024
#define COPY_BLOCK_DIM 256

// kernel(s) for half functions with no library support
namespace {

// TODO - refactor the function with similar logic in Transpose3DKernel using 16x16 Tile
__global__ void transposeNoOverlap(half* odata, const half* idata, const int m, const int n) {
  __shared__ half tile[TRANS_TILE_DIM][TRANS_TILE_DIM + 1];

  int x = blockIdx.x * TRANS_TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TRANS_TILE_DIM + threadIdx.y;

  if (x < m) {
    for (int j = 0; j < TRANS_TILE_DIM; j += BLOCK_ROWS) {
      if (j >= (n - y)) continue;
      const int64_t input_offset = static_cast<int64_t>(y + j) * m + x;
      tile[threadIdx.y + j][threadIdx.x] = idata[input_offset];
    }
  }

  __syncthreads();

  x = blockIdx.y * TRANS_TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TRANS_TILE_DIM + threadIdx.y;

  if (x >= n) return;

  for (int j = 0; j < TRANS_TILE_DIM; j += BLOCK_ROWS) {
    if ((y + j) >= m) return;
    const int64_t output_offset = static_cast<int64_t>(y + j) * n + x;
    odata[output_offset] = tile[threadIdx.x][threadIdx.y + j];
  }
}

__global__ void CopyVectorHalf(const half* x, int incx, half* y, int incy, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;
  y[id * incy] = x[id * incx];
}

__global__ void CopyVectorBFloat16(const onnxruntime::BFloat16* x, int incx, onnxruntime::BFloat16* y, int incy,
                                   int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;
  y[id * incy] = x[id * incx];
}

}  // namespace

dim3 cublasTransposeHelperDimGrid(int m, int n) {
  const auto grid_x = static_cast<unsigned int>((static_cast<int64_t>(n) + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM);
  const auto grid_y = static_cast<unsigned int>((static_cast<int64_t>(m) + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM);
  return dim3(grid_x, grid_y, 1);
}

// cublasTransposeHelper can only be used if it won't overflow the 65536 grid y dimension size
__host__ bool CanUse_cublasTransposeHelper_MLFloat16(int m, int n) {
  if (m <= 0 || n <= 0) {
    return false;
  }

  // transposeNoOverlap uses int64_t row * stride + col addressing in device code.
  // Keep fallback disabled when total element count would overflow 32-bit launch/indexing assumptions.
  if (static_cast<int64_t>(m) * static_cast<int64_t>(n) > std::numeric_limits<int>::max()) {
    return false;
  }

  dim3 dimGrid = cublasTransposeHelperDimGrid(m, n);
  return dimGrid.y < 65536;
}

cublasStatus_t cublasTransposeHelper(cudaStream_t stream, cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int) {
  ORT_ENFORCE(m > 0 && n > 0);
  if (C != A) {
    dim3 dimGrid = cublasTransposeHelperDimGrid(m, n);
    dim3 dimBlock(TRANS_TILE_DIM, BLOCK_ROWS, 1);

    ORT_ENFORCE(static_cast<int64_t>(m) * static_cast<int64_t>(n) <= std::numeric_limits<int>::max());
    ORT_ENFORCE(dimGrid.y < 65536);  // To prevent this, call CanUse_cublasTransposeHelper_MLFloat16 first
    transposeNoOverlap<<<dimGrid, dimBlock, 0, stream>>>(C, A, n, m);
  } else {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCopyHelper(cudaStream_t stream, cublasHandle_t, int n, const half* x, int incx, half* y, int incy) {
  dim3 dimGrid((unsigned int)(n + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
  dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
  CopyVectorHalf<<<dimGrid, dimBlock, 0, stream>>>(x, incx, y, incy, n);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCopyHelper(cudaStream_t stream, cublasHandle_t, int n, const onnxruntime::BFloat16* x, int incx,
                                onnxruntime::BFloat16* y, int incy) {
  dim3 dimGrid((unsigned int)(n + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
  dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
  CopyVectorBFloat16<<<dimGrid, dimBlock, 0, stream>>>(x, incx, y, incy, n);
  return CUBLAS_STATUS_SUCCESS;
}
