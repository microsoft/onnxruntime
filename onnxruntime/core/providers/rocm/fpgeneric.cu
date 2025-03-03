// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_runtime.h>
#include "core/providers/rocm/cu_inc/common.cuh"

#define TRANS_TILE_DIM 32
#define BLOCK_ROWS 8
#define COPY_TILE_DIM 1024
#define COPY_BLOCK_DIM 256

// kernel(s) for half functions with no library support
namespace {

__global__ void transposeNoOverlap(half* odata, const half* idata, const int m, const int n) {
  __shared__ half tile[TRANS_TILE_DIM][TRANS_TILE_DIM + 1];

  int x = blockIdx.x * TRANS_TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TRANS_TILE_DIM + threadIdx.y;

  if (x < m) {
    for (int j = 0; j < TRANS_TILE_DIM; j += BLOCK_ROWS) {
      if (j >= (n - y)) continue;
      tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * m + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * TRANS_TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TRANS_TILE_DIM + threadIdx.y;

  if (x >= n) return;

  for (int j = 0; j < TRANS_TILE_DIM; j += BLOCK_ROWS) {
    if ((y + j) >= m) return;
    odata[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
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

dim3 hipblasTransposeHelperDimGrid(int m, int n) {
  return dim3((n + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM, (m + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM, 1);
}

// hipblasTransposeHelper can only be used if it won't overflow the maxGridSize y dimension size
__host__ bool CanUse_hipblasTransposeHelper_MLFloat16(int m, int n) {
  dim3 dimGrid = hipblasTransposeHelperDimGrid(m, n);

  int deviceId;
  hipError_t hipError = hipGetDevice(&deviceId);
  if (hipError != 0) return false;

  hipDeviceProp_t deviceProp;
  hipError = hipGetDeviceProperties(&deviceProp, deviceId);
  if (hipError != 0) return false;

  return dimGrid.y < deviceProp.maxGridSize[1];
}

hipblasStatus_t hipblasTransposeHelper(hipStream_t stream, hipblasHandle_t, hipblasOperation_t , hipblasOperation_t , int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int) {
if (C != A) {
    dim3 dimGrid((n + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM, (m + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM, 1);
    dim3 dimBlock(TRANS_TILE_DIM, BLOCK_ROWS, 1);

    transposeNoOverlap<<<dim3(dimGrid), dim3(dimBlock), 0, stream>>>(C, A, n, m);
  } else {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
  return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasCopyHelper(hipStream_t stream, hipblasHandle_t, int n, const half* x, int incx, half* y, int incy) {
  dim3 dimGrid((unsigned int)(n + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
  dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
  CopyVectorHalf<<<dimGrid, dimBlock, 0, stream>>>(x, incx, y, incy, n);
  return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasCopyHelper(hipStream_t stream, hipblasHandle_t, int n, const onnxruntime::BFloat16* x, int incx,
                                 onnxruntime::BFloat16* y, int incy) {
  dim3 dimGrid((unsigned int)(n + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
  dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
  CopyVectorBFloat16<<<dimGrid, dimBlock, 0, stream>>>(x, incx, y, incy, n);
  return HIPBLAS_STATUS_SUCCESS;
}
