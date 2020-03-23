#include "hip/hip_runtime.h"
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

#include "core/providers/hip/cu_inc/common.cuh"
//#include <hiprand_kernel.h>

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

  for (int j = 0; j < TRANS_TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * m + x];

  __syncthreads();

  x = blockIdx.y * TRANS_TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TRANS_TILE_DIM + threadIdx.y;

  if (x >= n) return;

  for (int j = 0; j < TRANS_TILE_DIM; j += BLOCK_ROWS) {
    if ((y + j) >= m) return;
    odata[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}
// set up hiprand state, need to move up layer to remove calling for each generate call
// __global__ void setup_state(hiprandState* state, unsigned long long seed) {
//   hiprand_init(seed, 0, 0, state);
// }

// __global__ void GenerateUniformHalf(hiprandState* state, half* result, int n) {
//   int id = blockIdx.x * blockDim.x + threadIdx.x;
//   if (id >= n) return;

//   hiprandState localState = *state;

//   float x;
//   skipahead(id, &localState);
//   x = hiprand_uniform(&localState);

//   result[id] = x;
//   if (id == n - 1) *state = localState;
// }

// __global__ void GenerateNormalHalf(hiprandState* state, half* result, int n, half mean, half stddev) {
//   int id = blockIdx.x * blockDim.x + threadIdx.x;
//   if (id >= n) return;

//   hiprandState localState = *state;

//   float x;
//   skipahead(id, &localState);
//   x = hiprand_normal(&localState);

//   result[id] = (float)mean + (float)stddev * x;
//   if (id == n - 1) *state = localState;
// }

// kernels can convert matrix between half and float. speed currently not optimized, may need to add half2
/*
__global__ void copyHalf2Float(float *odata, const half *idata, const int n)
{
    float tmp[COPY_TILE_DIM/COPY_BLOCK_DIM];

    int x = blockIdx.x * COPY_TILE_DIM + threadIdx.x;

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        tmp[j] = (float) idata[x + j*COPY_BLOCK_DIM];

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        if(x + j*COPY_BLOCK_DIM < n) odata[x + j*COPY_BLOCK_DIM] = tmp[j];
}

__global__ void copyFloat2Half(half *odata, const float *idata, const int n)
{
    float tmp[COPY_TILE_DIM/COPY_BLOCK_DIM];

    int x = blockIdx.x * COPY_TILE_DIM + threadIdx.x;

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        tmp[j] = idata[x + j*COPY_BLOCK_DIM];

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        if(x + j*COPY_BLOCK_DIM < n) odata[x + j*COPY_BLOCK_DIM] = tmp[j];
}
*/

__global__ void CopyVectorHalf(const half* x, int incx, half* y, int incy, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;
  y[id * incy] = x[id * incx];
}

}  // namespace

hipblasStatus_t hipblasTransposeHelper(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int) {
  if (C != A) {
    dim3 dimGrid((n + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM, (m + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM, 1);
    dim3 dimBlock(TRANS_TILE_DIM, BLOCK_ROWS, 1);

    hipLaunchKernelGGL(transposeNoOverlap, dim3(dimGrid), dim3(dimBlock), 0, 0, C, A, n, m);
  } else {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
  return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasCopyHelper(hipblasHandle_t, int n, const half* x, int incx, half* y, int incy) {
  dim3 dimGrid((unsigned int)(n + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
  dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
  hipLaunchKernelGGL(CopyVectorHalf, dim3(dimGrid), dim3(dimBlock), 0, 0, x, incx, y, incy, n);
  return HIPBLAS_STATUS_SUCCESS;
}

// hiprandStatus_t curandGenerateUniformHelper(hiprandGenerator_t, half* outputPtr, size_t num) {
//   hiprandState* devStates;
//   hipMalloc((void**)&devStates, sizeof(hiprandState));
//   hipLaunchKernelGGL(setup_state, dim3(1), dim3(1), 0, 0, devStates, time(NULL));  // What does hiprandGenerateUniform actually doing? should also pass in state here

//   dim3 dimGrid((unsigned int)(num + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
//   dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
//   hipLaunchKernelGGL(GenerateUniformHalf, dim3(dimGrid), dim3(dimBlock), 0, 0, devStates, outputPtr, (int)num);

//   return (hiprandStatus_t)0;
// }

// hiprandStatus_t curandGenerateNormalHelper(hiprandGenerator_t, half* outputPtr, size_t n, half mean, half stddev) {
//   hiprandState* devStates;
//   hipMalloc((void**)&devStates, sizeof(hiprandState));
//   hipLaunchKernelGGL(setup_state, dim3(1), dim3(1), 0, 0, devStates, time(NULL));  // What does hiprandGenerateUniform actually doing? should also pass in state here

//   dim3 dimGrid((unsigned int)(n + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
//   dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
//   hipLaunchKernelGGL(GenerateNormalHalf, dim3(dimGrid), dim3(dimBlock), 0, 0, devStates, outputPtr, (int)n, mean, stddev);

//   return (hiprandStatus_t)0;
// }
