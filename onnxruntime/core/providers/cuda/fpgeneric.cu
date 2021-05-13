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

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

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

#if CUDA_VERSION >= 11000
__global__ void CopyVectorBFloat16(const nv_bfloat16* x, int incx, nv_bfloat16* y, int incy, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;
  y[id * incy] = x[id * incx];
}
#endif

}  // namespace

cublasStatus_t cublasTransposeHelper(cudaStream_t stream, cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int) {
  if (C != A) {
    dim3 dimGrid((n + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM, (m + TRANS_TILE_DIM - 1) / TRANS_TILE_DIM, 1);
    dim3 dimBlock(TRANS_TILE_DIM, BLOCK_ROWS, 1);

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

#if CUDA_VERSION >= 11000
cublasStatus_t cublasCopyHelper(cudaStream_t stream, cublasHandle_t, int n, const nv_bfloat16* x, int incx, nv_bfloat16* y, int incy) {
  dim3 dimGrid((unsigned int)(n + COPY_BLOCK_DIM - 1) / COPY_BLOCK_DIM, 1, 1);
  dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
  CopyVectorBFloat16<<<dimGrid, dimBlock, 0, stream>>>(x, incx, y, incy, n);
  return CUBLAS_STATUS_SUCCESS;
}
#endif

// /*template<typename LayoutInputA, typename LayoutInputB>
// inline */cudaError_t cutlassGemmHelper(int m, int n, int k,
//                                      const float* alpha,
//                                      const float* A, int lda,
//                                      const float* B, int ldb,
//                                      const float* beta,
//                                      float* C, int ldc,
//                                      cudaStream_t stream) {
//   using ElementAccumulator = float;                   // <- data type of accumulator
//   using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
//   using ElementInputA = float;              // <- data type of elements in input matrix A
//   using ElementInputB = float;              // <- data type of elements in input matrix B
//   using ElementOutput = float;              // <- data type of elements in output matrix D

//   using LayoutInputA = cutlass::layout::ColumnMajor;
//   using LayoutInputB = cutlass::layout::RowMajor;
//   using LayoutOutput = cutlass::layout::RowMajor;

//   // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
//   using MMAOp = cutlass::arch::OpClassTensorOp;

//   // This code section describes CUDA SM architecture number
//   using SmArch = cutlass::arch::Sm70;

//   // This code section describes the tile size a thread block will compute
//   using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
//   // This code section describes tile size a warp will compute
//   using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
//   // This code section describes the size of MMA op
//   using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

//   // This code section describes how threadblocks are scheduled on GPU
//   using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

//   // This code section describes ?
//   using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
//       ElementOutput,                                     // <- data type of output matrix
//       128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- This is the number of elements per
//                                                         // vectorized memory access. For half
//                                                         // precision, it's 8 elements. This becomes
//                                                         // the vector width of math instructions in
//                                                         // epilogue too
//       ElementAccumulator,                                // <- data type of accumulator
//       ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

//   // Number of pipelines you want to use
//   constexpr int NumStages = 2;
//   // Put all the created template variables to create GemmSplitKParallel template variable
//   using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
//                                           LayoutInputA,
//                                           ElementInputB,
//                                           LayoutInputB,
//                                           ElementOutput,
//                                           LayoutOutput,
//                                           ElementAccumulator,
//                                           MMAOp,
//                                           SmArch,
//                                           ShapeMMAThreadBlock,
//                                           ShapeMMAWarp,
//                                           ShapeMMAOp,
//                                           EpilogueOp,
//                                           SwizzleThreadBlock,
//                                           NumStages>;

//   Gemm gemm_op;
//   cutlass::Status status = gemm_op({{m, n, k},
//                                     {A, lda},
//                                     strideA,
//                                     {B, ldb},
//                                     strideB,
//                                     {C, ldc},
//                                     strideC,
//                                     {C, ldc},
//                                     strideC,
//                                     {alpha, beta}});

//   if (status != cutlass::Status::kSuccess) {
//     return cudaErrorUnknown;
//   }

//   return cudaSuccess;
// }

template<typename LayoutInputA, typename LayoutInputB>
cudaError_t cutlassGemmHelper(int m, int n, int k,
  const half* A, int lda,
  const half* B, int ldb,
  half* C, int ldc,
  float alpha, // float beta,
  cudaStream_t stream) {
  using ElementAccumulator = float;                   // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
  using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
  using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

  // using LayoutInputA = cutlass::layout::RowMajor;
  // using LayoutInputB = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm70;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 32>;
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  // This code section describes the size of MMA op
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // This code section describes ?
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,                                     // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- This is the number of elements per
                                                        // vectorized memory access. For half
                                                        // precision, it's 8 elements. This becomes
                                                        // the vector width of math instructions in
                                                        // epilogue too
      ElementAccumulator,                                // <- data type of accumulator
      ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

  // Number of pipelines you want to use
  constexpr int NumStages = 2;
  // Put all the created template variables to create GemmSplitKParallel template variable
  using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                          LayoutInputA,
                                          ElementInputB,
                                          LayoutInputB,
                                          ElementOutput,
                                          LayoutOutput,
                                          ElementAccumulator,
                                          MMAOp,
                                          SmArch,
                                          ShapeMMAThreadBlock,
                                          ShapeMMAWarp,
                                          ShapeMMAOp,
                                          EpilogueOp,
                                          SwizzleThreadBlock,
                                          NumStages>;

  // using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
  //                                         LayoutInputA,
  //                                         ElementInputB,
  //                                         LayoutInputB,
  //                                         ElementOutput,
  //                                         LayoutOutput,
  //                                         ElementAccumulator,
  //                                         MMAOp,
  //                                         SmArch>;
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue epilogue_alpha = ElementComputeEpilogue(alpha);
  ElementComputeEpilogue epilogue_beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  // int split_k_slices = 1;

  Gemm gemm_op;
  cutlass::Status status = gemm_op({{m, n, k},
                                    {reinterpret_cast<const cutlass::half_t*>(A), lda},
                                    {reinterpret_cast<const cutlass::half_t*>(B), ldb},
                                    {reinterpret_cast<cutlass::half_t*>(C), ldc},
                                    {reinterpret_cast<cutlass::half_t*>(C), ldc},
                                    {epilogue_alpha, epilogue_beta}}, nullptr, stream);

  if (status != cutlass::Status::kSuccess) {
    std::cout << "cutlass error" << std::endl;
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

template cudaError_t cutlassGemmHelper<cutlass::layout::RowMajor, cutlass::layout::RowMajor>(int m, int n, int k,
  const half* A, int lda,
  const half* B, int ldb,
  half* C, int ldc,
  float alpha, // float beta,
  cudaStream_t stream);
template cudaError_t cutlassGemmHelper<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(int m, int n, int k,
  const half* A, int lda,
  const half* B, int ldb,
  half* C, int ldc,
  float alpha, // float beta,
  cudaStream_t stream);
template cudaError_t cutlassGemmHelper<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(int m, int n, int k,
  const half* A, int lda,
  const half* B, int ldb,
  half* C, int ldc,
  float alpha, // float beta,
  cudaStream_t stream);
template cudaError_t cutlassGemmHelper<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(int m, int n, int k,
  const half* A, int lda,
  const half* B, int ldb,
  half* C, int ldc,
  float alpha, // float beta,
  cudaStream_t stream);

template <typename LayoutInputA, typename LayoutInputB>
cudaError_t cutlassGemmStridedBatchedHelper(int m, int n, int k,
                                            const half* A, int lda,
                                            long long int strideA,
                                            const half* B, int ldb,
                                            long long int strideB,
                                            half* C, int ldc,
                                            long long int strideC,
                                            int batch_count,
                                            float alpha, // float beta,
                                            cudaStream_t stream) {
  using ElementAccumulator = float;                   // <- data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
  using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
  using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

  // using LayoutInputA = cutlass::layout::ColumnMajor;
  // using LayoutInputB = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm70;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
  // This code section describes the size of MMA op
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // This code section describes ?
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,                                     // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- This is the number of elements per
                                                        // vectorized memory access. For half
                                                        // precision, it's 8 elements. This becomes
                                                        // the vector width of math instructions in
                                                        // epilogue too
      ElementAccumulator,                                // <- data type of accumulator
      ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

  // Put all the created template variables to create GemmSplitKParallel template variable
  using Gemm = cutlass::gemm::device::GemmBatched<ElementInputA,
                                                  LayoutInputA,
                                                  ElementInputB,
                                                  LayoutInputB,
                                                  ElementOutput,
                                                  LayoutOutput,
                                                  ElementAccumulator,
                                                  MMAOp,
                                                  SmArch,
                                                  ShapeMMAThreadBlock,
                                                  ShapeMMAWarp,
                                                  ShapeMMAOp,
                                                  EpilogueOp>;

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue epilogue_alpha = ElementComputeEpilogue(alpha);
  ElementComputeEpilogue epilogue_beta = ElementComputeEpilogue(0);

  Gemm gemm_op;
  cutlass::Status status = gemm_op({{m, n, k},
                                    {reinterpret_cast<const cutlass::half_t*>(A), lda},
                                    strideA,
                                    {reinterpret_cast<const cutlass::half_t*>(B), ldb},
                                    strideB,
                                    {reinterpret_cast<cutlass::half_t*>(C), ldc},
                                    strideC,
                                    {reinterpret_cast<cutlass::half_t*>(C), ldc},
                                    strideC,
                                    {epilogue_alpha, epilogue_beta},
                                    batch_count}, nullptr, stream);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

template cudaError_t cutlassGemmStridedBatchedHelper<cutlass::layout::RowMajor, cutlass::layout::RowMajor>(int m, int n, int k,
  const half* A, int lda,
  long long int strideA,
  const half* B, int ldb,
  long long int strideB,
  half* C, int ldc,
  long long int strideC,
  int batch_count,
  float alpha, // float beta,
  cudaStream_t stream);

template cudaError_t cutlassGemmStridedBatchedHelper<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(int m, int n, int k,
  const half* A, int lda,
  long long int strideA,
  const half* B, int ldb,
  long long int strideB,
  half* C, int ldc,
  long long int strideC,
  int batch_count,
  float alpha, // float beta,
  cudaStream_t stream);

template cudaError_t cutlassGemmStridedBatchedHelper<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(int m, int n, int k,
  const half* A, int lda,
  long long int strideA,
  const half* B, int ldb,
  long long int strideB,
  half* C, int ldc,
  long long int strideC,
  int batch_count,
  float alpha, // float beta,
  cudaStream_t stream);

template cudaError_t cutlassGemmStridedBatchedHelper<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(int m, int n, int k,
  const half* A, int lda,
  long long int strideA,
  const half* B, int ldb,
  long long int strideB,
  half* C, int ldc,
  long long int strideC,
  int batch_count,
  float alpha, // float beta,
  cudaStream_t stream);