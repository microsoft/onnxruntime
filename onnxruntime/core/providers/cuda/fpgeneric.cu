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

#include "core/providers/cuda/cu_inc/common.cuh"
#include <curand_kernel.h>

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
// set up curand state, need to move up layer to remove calling for each generate call
__global__ void setup_state(curandState* state, unsigned long long seed) {
  curand_init(seed, 0, 0, state);
}

__global__ void GenerateUniformHalf(curandState* state, half* result, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;

  curandState localState = *state;

  float x;
  skipahead(id, &localState);
  x = curand_uniform(&localState);

  result[id] = x;
  if (id == n - 1) *state = localState;
}

__global__ void GenerateNormalHalf(curandState* state, half* result, int n, half mean, half stddev) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;

  curandState localState = *state;

  float x;
  skipahead(id, &localState);
  x = curand_normal(&localState);

  result[id] = (float)mean + (float)stddev * x;
  if (id == n - 1) *state = localState;
}

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

cublasStatus_t cublasTransposeHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int) {
  if (C != A) {
    dim3 dimGrid((n + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM, (m + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM, 1);
    dim3 dimBlock(TRANS_TILE_DIM, BLOCK_ROWS, 1);

    transposeNoOverlap<<<dimGrid, dimBlock>>>(C, A, n, m);
  } else {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCopyHelper(cublasHandle_t, int n, const half* x, int incx, half* y, int incy) {
  dim3 dimGrid((unsigned int)(n + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
  dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
  CopyVectorHalf<<<dimGrid, dimBlock>>>(x, incx, y, incy, n);
  return CUBLAS_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniformHelper(curandGenerator_t, half* outputPtr, size_t num) {
  curandState* devStates;
  cudaMalloc((void**)&devStates, sizeof(curandState));
  setup_state<<<1, 1>>>(devStates, time(NULL));  // What does curandGenerateUniform actually doing? should also pass in state here

  dim3 dimGrid((unsigned int)(num + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
  dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
  GenerateUniformHalf<<<dimGrid, dimBlock>>>(devStates, outputPtr, (int)num);

  return (curandStatus_t)0;
}

curandStatus_t curandGenerateNormalHelper(curandGenerator_t, half* outputPtr, size_t n, half mean, half stddev) {
  curandState* devStates;
  cudaMalloc((void**)&devStates, sizeof(curandState));
  setup_state<<<1, 1>>>(devStates, time(NULL));  // What does curandGenerateUniform actually doing? should also pass in state here

  dim3 dimGrid((unsigned int)(n + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
  dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
  GenerateNormalHalf<<<dimGrid, dimBlock>>>(devStates, outputPtr, (int)n, mean, stddev);

  return (curandStatus_t)0;
}

void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const int8_t* A,
                   int lda,
                   const int8_t* B,
                   int ldb,
                   int32_t* C,
                   int ldc) {
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatrixLayout_t a_desc = NULL, b_desc = NULL, c_desc = NULL;
  int32_t alpha = 1, beta = 0;
  cublasOperation_t op_transpose = CUBLAS_OP_T;

  // The tensor op igemm kernels require specialized memory order of
  // data.
  cublasLtMatrixTransformDesc_t transform_desc = NULL;
  int8_t *a_transform = NULL, *b_transform = NULL;
  int32_t* c_transform = NULL;
  cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
  float transformAlpha = 1.0f, transformBeta = 0.0f;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

  int ldatransform = 32 * m;
  int ldbtransform = 32 * roundoff(n, 8);
  int ldctransform = 32 * m;

  CUDA_CALL_THROW(cudaMalloc(&a_transform, sizeof(int8_t) * roundoff(k, 32) / 32 * ldatransform));
  CUDA_CALL_THROW(cudaMalloc(&b_transform, sizeof(int8_t) * roundoff(k, 32) / 32 * ldbtransform));
  CUDA_CALL_THROW(cudaMalloc(&c_transform, sizeof(int32_t) * roundoff(n, 32) / 32 * ldctransform));

  CUBLAS_CALL_THROW(cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F));

  // B matrix is non-transposed, but transposed matrix is needed - add transpose operation in matrix transform.
  CUBLAS_CALL_THROW(cublasLtMatrixTransformDescSetAttribute(transform_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &op_transpose, sizeof(op_transpose)));

  CUBLAS_CALL_THROW(cublasLtMatmulDescCreate(&matmulDesc, CUDA_R_32I));

  // Tensor op igemm kernels only support NT gemm
  CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_transpose, sizeof(op_transpose)));

  // Create descriptors for the original matrices
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8I, m, k, lda));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8I, k, n, ldb));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32I, m, n, ldc));

  // Create descriptors for the transformed matrices
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldatransform));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbtransform));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));

  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldctransform));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

  // Transforms and computation
  CUBLAS_CALL_THROW(cublasLtMatrixTransform(ltHandle, transform_desc, &transformAlpha, A, a_desc, &transformBeta, NULL, NULL, a_transform, AtransformDesc, 0));
  CUBLAS_CALL_THROW(cublasLtMatrixTransform(ltHandle,
                                            transform_desc,
                                            &transformAlpha,
                                            B,
                                            b_desc,
                                            &transformBeta,
                                            NULL,
                                            NULL,
                                            b_transform,
                                            BtransformDesc,
                                            0));

  // No need to transform C matrix as beta is assumed to be 0
  CUBLAS_CALL_THROW(cublasLtMatmul(ltHandle,
                                   matmulDesc,
                                   &alpha,
                                   a_transform,
                                   AtransformDesc,
                                   b_transform,
                                   BtransformDesc,
                                   &beta,
                                   c_transform,
                                   CtransformDesc,
                                   c_transform,
                                   CtransformDesc,
                                   NULL,
                                   NULL,
                                   0,
                                   0));

  // Transform the outputs to COL order
  CUBLAS_CALL_THROW(cublasLtMatrixTransform(ltHandle,
                                            transform_desc,
                                            &transformAlpha,
                                            c_transform,
                                            CtransformDesc,
                                            &transformBeta,
                                            NULL,
                                            NULL,
                                            C,
                                            c_desc,
                                            0));

  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  if (CtransformDesc) cublasLtMatrixLayoutDestroy(CtransformDesc);
  if (BtransformDesc) cublasLtMatrixLayoutDestroy(BtransformDesc);
  if (AtransformDesc) cublasLtMatrixLayoutDestroy(AtransformDesc);
  if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
  if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
  if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
  if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
  if (transform_desc) cublasLtMatrixTransformDescDestroy(transform_desc);

  // Wait until device is done before freeing transformed buffers
  if (c_transform) cudaFree(c_transform);
  if (b_transform) cudaFree(b_transform);
  if (a_transform) cudaFree(a_transform);
}
