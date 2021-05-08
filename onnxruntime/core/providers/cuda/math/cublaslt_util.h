#pragma once

#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

#include "core/common/status.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {
namespace cuda {

auto constexpr algoCombinations = 6000;
constexpr int MAX_NUM_ALGO_IDS = 40;
constexpr int NUM_ITERATIONS = 10;

template <typename T>
struct TGemmTypes
{
};

template <>
struct TGemmTypes<half>
{
    static const cudaDataType_t cudaTypeI = CUDA_R_16F;
    using dataTypeI = half;
    static const cudaDataType_t cudaTypeO = CUDA_R_16F;
    using dataTypeO = half;
    static const cudaDataType_t cudaTypeS = CUDA_R_32F; // scale type, usually same as computeType
    using dataTypeS = float;
    static const cublasComputeType_t cudaTypeCom = CUBLAS_COMPUTE_32F;
};

template <>
struct TGemmTypes<nv_bfloat16>
{
    static const cudaDataType_t cudaTypeI = CUDA_R_16BF;
    using dataTypeI = nv_bfloat16;
    static const cudaDataType_t cudaTypeO = CUDA_R_16BF;
    using dataTypeO = nv_bfloat16;
    static const cudaDataType_t cudaTypeS = CUDA_R_32F; // scale type, usually same as computeType
    using dataTypeS = float;
    static const cublasComputeType_t cudaTypeCom = CUBLAS_COMPUTE_32F;
};

template <>
struct TGemmTypes<float>
{
    static const cudaDataType_t cudaTypeI = CUDA_R_32F;
    using dataTypeI = float;
    static const cudaDataType_t cudaTypeO = CUDA_R_32F;
    using dataTypeO = float;
    static const cudaDataType_t cudaTypeS = CUDA_R_32F; // scale type, usually same as computeType
    using dataTypeS = float;
    static const cublasComputeType_t cudaTypeCom = CUBLAS_COMPUTE_32F;
};

template <>
struct TGemmTypes<double>
{
    static const cudaDataType_t cudaTypeI = CUDA_R_64F;
    using dataTypeI = double;
    static const cudaDataType_t cudaTypeO = CUDA_R_64F;
    using dataTypeO = double;
    static const cudaDataType_t cudaTypeS = CUDA_R_64F; // scale type, usually same as computeType
    using dataTypeS = double;
    static const cublasComputeType_t cudaTypeCom = CUBLAS_COMPUTE_64F;
};

template <typename T>
struct TGemm
{
  int m, n, k, ldA, ldB, ldC, rA, rB, rC, cA, cB, cC;
  size_t elemA;
  size_t elemB;
  size_t elemC;
  
  size_t bytesA;
  size_t bytesB;
  size_t bytesC;

  using Types = TGemmTypes<T>;
  const typename Types::dataTypeI* A;
  const typename Types::dataTypeI* B;
  typename Types::dataTypeO* C;

  bool transA, transB;
  cublasOperation_t opA;
  cublasOperation_t opB;

  typename Types::dataTypeS alpha;
  typename Types::dataTypeS beta;

  TGemm() {}

  TGemm(int m_, int n_, int k_,
        const typename Types::dataTypeI* A_,
        const typename Types::dataTypeI* B_,
        typename Types::dataTypeO* C_,
        bool transA_ = false, bool transB_ = false) : A(A_), B(B_), C(C_) {
    m = m_;
    n = n_;
    k = k_;
    elemA = m * k;
    elemB = n * k;
    elemC = m * n;
    bytesA = sizeof(T) * elemA;
    bytesB = sizeof(T) * elemB;
    bytesC = sizeof(T) * elemC;
  
    transA = transA_;
    transB = transB_;
    ldA = transA ? k : m;
    ldB = transB ? n : k;
    ldC = m;

    rA = ldA;
    rB = ldB;
    rC = ldC;

    cA = transA ? m : k;
    cB = transB ? k : n;
    cC = n;

    opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    alpha = T(1.f);
    beta = T(0.f);
  }
};

/* Structure to store information about different run trials */
typedef struct
{
  cublasLtMatmulAlgo_t algo;
  cublasStatus_t status;
  float time{1000000};
  size_t workspaceSize; // actual memory workspace needed
  cublasMath_t mathMode;
  cublasLtReductionScheme_t reductionScheme;
  int customOption;
  float wavesCount;
} MatmulPerf_t;

// clang-format off
void LtGemmSearch(cublasLtHandle_t ltHandle,
                  cublasOperation_t transA,
                  cublasOperation_t transB,
                  int const &m,
                  int const &n,
                  int const &k,
                  void const *alpha,
                  void const *A,
                  int const &lda,
                  void const *B,
                  int const &ldb,
                  void const *beta,
                  void *C,
                  int const &ldc,
                  void *workSpace,
                  size_t workSpaceSize,
                  cublasComputeType_t computeType,
                  cudaDataType_t scaleType,
                  cudaDataType_t Atype,
                  cudaDataType_t Btype,
                  cudaDataType_t Ctype,
                  std::vector<MatmulPerf_t> &perfResults);
// clang-format on
template <typename T>
void LtGemmSearch(cublasLtHandle_t handle, const TGemm<T> &g,
                  void *workSpace, size_t workSpaceSize,
                  std::vector<MatmulPerf_t> &perfResults)
{
  // clang-format off
  LtGemmSearch(handle,
               g.opA, g.opB,
               g.m, g.n, g.k,
               &g.alpha,
               g.A, g.ldA,
               g.B, g.ldB,
               &g.beta,
               g.C, g.ldC,
               workSpace, workSpaceSize,
               TGemm<T>::Types::cudaTypeCom,
               TGemm<T>::Types::cudaTypeS,
               TGemm<T>::Types::cudaTypeI,
               TGemm<T>::Types::cudaTypeI,
               TGemm<T>::Types::cudaTypeO,
               perfResults);
    // clang-format on
}

/* CAUTION : must match cublasLtMatmulTile_t */
const char *const matmulTileName[] = {
    "UNDEF",
    "8x8",
    "8x16",
    "16x8",
    "8x32",
    "16x16",
    "32x8",
    "8x64",
    "16x32",
    "32x16",
    "64x8",
    "32x32",
    "32x64",
    "64x32",
    "32x128",
    "64x64",  // index = 15
    "128x32",
    "64x128",
    "128x64",
    "64x256",
    "128x128",
    "256x64",
    "64x512",
    "128x256",
    "256x128",
    "512x64",
};

}  // namespace cuda
}  // namespace onnxruntime