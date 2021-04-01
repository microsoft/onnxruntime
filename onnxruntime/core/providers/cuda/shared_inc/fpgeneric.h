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

#pragma once

#include "core/providers/cuda/cuda_common.h"

#include <cublasLt.h>

// Generalize library calls to be use in template functions
auto constexpr algoCombinations = 6000;
auto constexpr algoIds = 40;
auto constexpr printAlgos = 1;
auto constexpr kernelRepeats = 10;
auto constexpr threadsPerBlock = 1024;
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
} customMatmulPerf_t;

const char* const matmulTileName[] = {
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
    "64x64",
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

struct AlgoProps
{
    int algoId;
    int tile;
    int swizzle;
    int customOption;
    int numSplitsK;
    int reductionScheme;
    int mathMode;

    void populate(const cublasLtMatmulAlgo_t& algo)
    {
        const cublasLtMatmulAlgo_t* matmulAlgo = &algo;
        cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), nullptr);
        cublasLtMatmulAlgoCapGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CAP_MATHMODE_IMPL, &mathMode, sizeof(mathMode), nullptr);
    }
};

static inline bool time_compare(const customMatmulPerf_t& perf_a, const customMatmulPerf_t& perf_b)
{
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static void printPerfStructure(const customMatmulPerf_t& perf, int const& m, int const& n, int const& k)
{
    AlgoProps p;
    p.populate(perf.algo);
    /* Calculate GFLOPS */
    double timeAvg = (perf.time * 1e-3) / kernelRepeats; // Convert to seconds, then divide by loops
    double gflop = (2 * static_cast<unsigned long long int>(m * n) * k) * 1e-9; // Real

    std::cout << "algoId=" << p.algoId << " tile=" << p.tile << " (" << matmulTileName[p.tile] << ") numSplitsK=" << p.numSplitsK << " reductionScheme=" << p.reductionScheme << " swizzle=" << p.swizzle << " customOption=" << p.customOption << " mathMode=" << p.mathMode << " Stat=" <<  perf.status << " Time=" << perf.time << " WSbytes=" << perf.workspaceSize << " waves=" << perf.wavesCount << "GFlops=" << (gflop / timeAvg) << std::endl;
}

static cublasStatus_t myCustomMatmulRun(cublasLtHandle_t ltHandle, // to get the capabilities (required a GPU)
    cublasLtMatmulDesc_t operationDesc, float const* alpha,       /* host or device pointer */
    float const* A, cublasLtMatrixLayout_t Adesc, float const* B, cublasLtMatrixLayout_t Bdesc,
    float const* beta, /* host or device pointer */
    float const* C, cublasLtMatrixLayout_t Cdesc, float* D, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulAlgo_t const& algo, void* workSpace, size_t workSpaceSizeInBytes, customMatmulPerf_t& perfResults,
    cudaStream_t stream, cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{
    cublasLtMatmulHeuristicResult_t heurResult;

    /* Looping over the Algo */
    cublasStatus_t algoStatus
        = cublasLtMatmulAlgoCheck(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);

    if (algoStatus == CUBLAS_STATUS_SUCCESS)
    {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes)
        {
            cudaError_t err, err1, err2, err3;
            err = cudaEventRecord(startEvent, stream);
            for (int loop = 0; loop < kernelRepeats; loop++)
            {
                cublasStatus_t oneRunStatus
                    = cublasLtMatmul(ltHandle, operationDesc, alpha, /* host or device pointer */
                        A, Adesc, B, Bdesc, beta,                    /* host or device pointer */
                        C, Cdesc, D, Ddesc, &algo, workSpace, workSpaceSizeInBytes, stream);
                if (oneRunStatus != CUBLAS_STATUS_SUCCESS)
                {
                    algoStatus = oneRunStatus;
                    break;
                }
            }
            err1 = cudaEventRecord(stopEvent, stream);
            err2 = cudaEventSynchronize(stopEvent);
            float time;
            err3 = cudaEventElapsedTime(&time, startEvent, stopEvent);
            if ((err != cudaSuccess) || (err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess))
            {
                algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
            }
            // For the moment only add successful findings
            if (algoStatus == CUBLAS_STATUS_SUCCESS)
            {
                perfResults.algo = algo;
                perfResults.time = time / kernelRepeats; // Average time
                perfResults.workspaceSize = heurResult.workspaceSize;
                perfResults.wavesCount = heurResult.wavesCount;
            }
        }
        else
        {

            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED; // Not enough workspace
        }
    }
    return algoStatus;
}


static void LtGemmSearch(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int const& m,
    int const& n, int const& k, float const* alpha,                                  /* host pointer */
    float const* A, int const& lda, float const* B, int const& ldb, float const* beta, /* host pointer */
    float* C, int const& ldc, void* workSpace, size_t workSpaceSize,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype,
    std::vector<customMatmulPerf_t>& perfResults, cudaStream_t stream)
{

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    cudaEvent_t startEvent = nullptr, stopEvent = nullptr;

    // SplitK value that we are going to try when SplitK is supported for a given
    // algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};

    // Let try a fixed number of combinations
    int algoCount = 0;
    int nbAlgoIds = 0;
    int algoIdA[algoIds];
    // customMatmulPerf_t perfResults[algoCombinations];

    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSpaceSize, sizeof(workSpaceSize));

    const int mathMode = Ctype == CUDA_R_16F ? 1 : 0;
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MATH_MODE_MASK, &mathMode, sizeof(mathMode));
    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for details
    // about defaults; here we just need to set the transforms for A and B
    cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);

    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));

    // Create matrix descriptors. We are good with the details here so no need to
    // set any extra attributes
    cublasLtMatrixLayoutCreate(&Adesc, Atype, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, Btype, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc);

    // Request the 4 first AlgoId available for SGEMM ( computeType = scaleType =
    // Atype = Btype = Ctype = Dtype = CUDA_R_32F)
    cublasLtMatmulAlgoGetIds(
        ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIds, algoIdA, &nbAlgoIds);

    std::cout << "Number of algos " <<  nbAlgoIds << std::endl;

    // Create CUDA event to time the execution time of each algo
    cudaEventCreate(&startEvent, cudaEventBlockingSync);
    cudaEventCreate(&stopEvent, cudaEventBlockingSync);

    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (algoCount < algoCombinations); idx++)
    {
        cublasLtMatmulAlgo_t algo;
        size_t sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status
            = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            continue;
        }

        int mathMode = -1;
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_MATHMODE_IMPL, &mathMode, sizeof(mathMode), nullptr);
        // TODO is this the right way to check that it's SGEMM?
        if (Ctype == CUDA_R_32F && mathMode == 1)
        {
            // if mathMode is 1, cublasLt chooses automatically to run in mixed precision for certain sizes
            continue;
        }

        // Query the tiles enums supported by that algo
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sizeWritten);
        int nbTiles = int(sizeWritten / sizeof(int));
        int* tileA = new int[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0)
        {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
        }

        int splitkSupport, redMask, swizzlingMax, customOptionMax, epilogueMask;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the
        // different combinations
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int) * nbTiles, &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);

        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_EPILOGUE_MASK, &epilogueMask, sizeof(epilogueMask), &sizeWritten);

        /* Loop over the different tiles */
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++)
        {
            /* Loop over the different custom option if any */
            for (int customOption = 0; customOption <= customOptionMax; customOption++)
            {
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
                /* Loop over the CTAs swizzling support */
                for (int k = 0; k <= swizzlingMax; k++)
                {
                    int splitK_trial = 0;
                    if (splitkSupport)
                    {
                        splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                    }
                    // Loop over the splitK value over a fixed sequence splitKSequenceA in
                    // addition to the case where splitK is not enabled
                    for (int l = 0; (l < (1 + splitK_trial)) && (algoCount < algoCombinations); l++)
                    {
                        /* Setup attribute of the algo to run */
                        cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx]));
                        int splitK_val = 0;
                        int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                        cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val));
                        cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
                        cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int));

                        if (l > 0)
                        { // Split-K case
                            splitK_val = splitKSequenceA[l - 1];
                            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1]));
                            /* Going over all the reduction scheme  */
                            for (redScheme = 1; redScheme < static_cast<int>(CUBLASLT_REDUCTION_SCHEME_MASK)
                                 && (algoCount < algoCombinations);
                                 redScheme = redScheme << 1)
                            {
                                if (redScheme & redMask)
                                {
                                    cublasLtMatmulAlgoConfigSetAttribute(
                                        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(redScheme));

                                    status
                                        = myCustomMatmulRun(ltHandle, operationDesc, alpha, /* host or device pointer */
                                            A, Adesc, B, Bdesc, beta,                     /* host or device pointer */
                                            C, Cdesc, C, Cdesc, algo, workSpace, workSpaceSize, perfResults[algoCount],
                                            stream, startEvent, stopEvent);
                                    perfResults[algoCount].status = status;
                                    if (status == CUBLAS_STATUS_SUCCESS)
                                    {

                                        algoCount++;
                                    }
                                } // end if
                            }     // end for
                        }
                        else
                        { // Non-splitK case
                            /* if user preference is ok with workspace */
                            if (algoCount < algoCombinations)
                            {
                                status = myCustomMatmulRun(ltHandle, operationDesc, alpha, /* host or device pointer */
                                    A, Adesc, B, Bdesc, beta,                            /* host or device pointer */
                                    C, Cdesc, C, Cdesc, algo, workSpace, workSpaceSize, perfResults[algoCount], stream,
                                    startEvent, stopEvent);
                                perfResults[algoCount].status = status;
                                if (status == CUBLAS_STATUS_SUCCESS)
                                    algoCount++;
                            }
                        }
                    } // end l
                }     // end k
            }         // end customOption
        }             // end tileIdx
        delete[] tileA;
    } // end idx

    // Sort the results per run duration
    std::sort(perfResults.begin(), perfResults.end(), time_compare);
    // Print timing and perf details of the fastest combinations
    // for (int i = 0; i < perfResults.size(); i++){
    for (int i = 0; i < printAlgos; i++)
    {
        if (perfResults[i].time == 1000000.f)
            break;
        printPerfStructure(perfResults[i], m, n, k);
    }

    // Descriptors are no longer needed as all GPU work was already enqueued
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}


// gemm
inline cublasStatus_t cublasLtGemmHelperS(cublasLtHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         const float* alpha,
                                         const float* A, int lda,
                                         const float* B, int ldb,
                                         const float* beta,
                                         float* C, int ldc,
                                         void *workspace,
                                         size_t workspaceSize,
                                         cudaStream_t stream) {

    std::vector<customMatmulPerf_t> perfResults(algoCombinations);

    LtGemmSearch(handle,
                 transa, transb,
                 m, n, k,
                 alpha,                                  /* host pointer */
                 A, lda,
                 B, ldb,
                 beta,                                   /* host pointer */
                 C, ldc,
                 workspace, workspaceSize,
                 CUBLAS_COMPUTE_32F,
                 CUDA_R_32F,
                 CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                 perfResults,
                 stream);

    return CUBLAS_STATUS_SUCCESS;
}

static void cuBlasLtAlgoFill(cublasLtMatmulAlgo_t& algo,
                             int tile = 16,
                             int numSplitsK = 0,
                             int reductionScheme = 0,
                             int swizzle = 0,
                             int customOption = 0)
{
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme));
}


inline cublasStatus_t cublasLtGemmHelperI(cublasLtHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         const float* alpha,
                                         const float* A, int lda,
                                         const float* B, int ldb,
                                         const float* beta,
                                         float* C, int ldc,
                                         void *workspace,
                                         size_t workspaceSize,
                                         cudaStream_t stream) {

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    // check return status
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));

    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc);

    cublasLtMatmulAlgo_t algo;
    cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    cuBlasLtAlgoFill(algo, 20, 5, 1, 0);

    cublasStatus_t oneRunStatus = cublasLtMatmul(handle,
                                                 operationDesc,
                                                 alpha,
                                                 A,
                                                 Adesc,
                                                 B,
                                                 Bdesc,
                                                 beta,
                                                 C,
                                                 Cdesc,
                                                 C,
                                                 Cdesc,
                                                 &algo,
                                                 workspace,
                                                 workspaceSize,
                                                 stream);

    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);

    return oneRunStatus;
}


inline cublasStatus_t cublasLtGemmHelper(cublasLtHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         const float* alpha,
                                         const float* A, int lda,
                                         const float* B, int ldb,
                                         const float* beta,
                                         float* C, int ldc,
                                         void *workspace,
                                         size_t workspaceSize,
                                         cudaStream_t stream) {
    const bool search_mode = false;
    if (search_mode) {
        return cublasLtGemmHelperS(handle,
                                  transa,
                                  transb,
                                  m,  n,  k,
                                  alpha,
                                  A,  lda,
                                  B,  ldb,
                                  beta,
                                  C,  ldc,
                                  workspace,
                                  workspaceSize,
                                  stream);
    }
    return cublasLtGemmHelperI(handle,
                                  transa,
                                  transb,
                                  m,  n,  k,
                                  alpha,
                                  A,  lda,
                                  B,  ldb,
                                  beta,
                                  C,  ldc,
                                  workspace,
                                  workspaceSize,
                                  stream);
}

/////////////////////////////////////////////////////////////////////////////////

// gemm
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const float* alpha,
                                       const float* A, int lda,
                                       const float* B, int ldb,
                                       const float* beta,
                                       float* C, int ldc,
                                       const cudaDeviceProp& prop) {
#ifdef ENABLE_TRAINING
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TF32_TENSOR_OP_MATH);
#else
  ORT_UNUSED_PARAMETER(prop);
#endif
#else
  ORT_UNUSED_PARAMETER(prop);
#endif

  //return cublasSgemm(handle,
  //                   transa,
  //                   transb,
  //                   m, n, k,
  //                   alpha,
  //                   A, lda,
  //                   B, ldb,
  //                   beta,
  //                   C, ldc);
  return cublasGemmEx(handle,
                      transa,
                      transb,
                      m, n, k,
                      alpha,
                      A, CUDA_R_32F, lda,
                      B, CUDA_R_32F, ldb,
                      beta,
                      C, CUDA_R_32F, ldc,
                      CUDA_R_32F,
                      CUBLAS_GEMM_ALGO1_TENSOR_OP);
}
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const double* alpha,
                                       const double* A, int lda,
                                       const double* B, int ldb,
                                       const double* beta,
                                       double* C, int ldc,
                                       const cudaDeviceProp& /*prop*/) {
  return cublasDgemm(handle,
                     transa,
                     transb,
                     m, n, k,
                     alpha,
                     A, lda,
                     B, ldb,
                     beta,
                     C, ldc);
}
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const half* alpha,
                                       const half* A, int lda,
                                       const half* B, int ldb,
                                       const half* beta,
                                       half* C, int ldc,
                                       const cudaDeviceProp& prop) {
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TENSOR_OP_MATH);

#ifdef ENABLE_TRAINING
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));

  // accumulating in FP32
  return cublasGemmEx(handle,
                      transa,
                      transb,
                      m, n, k,
                      &h_a,
                      A, CUDA_R_16F, lda,
                      B, CUDA_R_16F, ldb,
                      &h_b,
                      C, CUDA_R_16F, ldc,
                      CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT);
#else
  // accumulating in FP16
  return cublasHgemm(handle,
                      transa,
                      transb,
                      m, n, k,
                      alpha,
                      A, lda,
                      B, ldb,
                      beta,
                      C, ldc);
#endif
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const nv_bfloat16* alpha,
                                       const nv_bfloat16* A, int lda,
                                       const nv_bfloat16* B, int ldb,
                                       const nv_bfloat16* beta,
                                       nv_bfloat16* C, int ldc,
                                       const cudaDeviceProp& prop) {
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_DEFAULT_MATH);

  float h_a = onnxruntime::BFloat16(*reinterpret_cast<const uint16_t*>(alpha)).ToFloat();
  float h_b = onnxruntime::BFloat16(*reinterpret_cast<const uint16_t*>(beta)).ToFloat();

  // accumulating in FP32
  return cublasGemmEx(handle,
                      transa,
                      transb,
                      m, n, k,
                      &h_a,
                      A, CUDA_R_16BF, lda,
                      B, CUDA_R_16BF, ldb,
                      &h_b,
                      C, CUDA_R_16BF, ldc,
                      CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT);
}
#endif

// batched gemm
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              const float* alpha,
                                              const float* Aarray[], int lda,
                                              const float* Barray[], int ldb,
                                              const float* beta,
                                              float* Carray[], int ldc,
                                              int batch_count,
                                              const cudaDeviceProp& prop) {
#ifdef ENABLE_TRAINING
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TF32_TENSOR_OP_MATH);
#else
  ORT_UNUSED_PARAMETER(prop);
#endif
#else
  ORT_UNUSED_PARAMETER(prop);
#endif
  return cublasSgemmBatched(handle,
                            transa,
                            transb,
                            m, n, k,
                            alpha,
                            Aarray, lda,
                            Barray, ldb,
                            beta,
                            Carray, ldc,
                            batch_count);
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              const double* alpha,
                                              const double* Aarray[], int lda,
                                              const double* Barray[], int ldb,
                                              const double* beta,
                                              double* Carray[], int ldc,
                                              int batch_count,
                                              const cudaDeviceProp& /*prop*/) {
  return cublasDgemmBatched(handle,
                            transa,
                            transb,
                            m, n, k,
                            alpha,
                            Aarray, lda,
                            Barray, ldb,
                            beta,
                            Carray, ldc,
                            batch_count);
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              const half* alpha,
                                              const half* Aarray[], int lda,
                                              const half* Barray[], int ldb,
                                              const half* beta,
                                              half* Carray[], int ldc,
                                              int batch_count,
                                              const cudaDeviceProp& prop) {
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TENSOR_OP_MATH);

#ifdef ENABLE_TRAINING
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));

  // accumulating in FP32
  return cublasGemmBatchedEx(handle,
                             transa,
                             transb,
                             m, n, k,
                             &h_a,
                             (const void**)Aarray, CUDA_R_16F, lda,
                             (const void**)Barray, CUDA_R_16F, ldb,
                             &h_b,
                             (void**)Carray, CUDA_R_16F, ldc,
                             batch_count,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT);
#else
  // accumulating in FP16
  return cublasHgemmBatched(handle,
                            transa,
                            transb,
                            m, n, k,
                            alpha,
                            (const __half**)Aarray, lda,
                            (const __half**)Barray, ldb,
                            beta,
                            (__half**)Carray, ldc,
                            batch_count);
#endif
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              const nv_bfloat16* alpha,
                                              const nv_bfloat16* Aarray[], int lda,
                                              const nv_bfloat16* Barray[], int ldb,
                                              const nv_bfloat16* beta,
                                              nv_bfloat16* Carray[], int ldc,
                                              int batch_count,
                                              const cudaDeviceProp& prop) {
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TENSOR_OP_MATH);
  float h_a = onnxruntime::BFloat16(*reinterpret_cast<const uint16_t*>(alpha)).ToFloat();
  float h_b = onnxruntime::BFloat16(*reinterpret_cast<const uint16_t*>(beta)).ToFloat();

  // accumulating in FP32
  return cublasGemmBatchedEx(handle,
                             transa,
                             transb,
                             m, n, k,
                             &h_a,
                             (const void**)Aarray, CUDA_R_16BF, lda,
                             (const void**)Barray, CUDA_R_16BF, ldb,
                             &h_b,
                             (void**)Carray, CUDA_R_16BF, ldc,
                             batch_count,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT);
}
#endif

// strided batched gemm
inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const float* alpha,
                                                     const float* A, int lda,
                                                     long long int strideA,
                                                     const float* B, int ldb,
                                                     long long int strideB,
                                                     const float* beta,
                                                     float* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& prop) {
#ifdef ENABLE_TRAINING
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TF32_TENSOR_OP_MATH);
#else
  ORT_UNUSED_PARAMETER(prop);
#endif
#else
  ORT_UNUSED_PARAMETER(prop);
#endif

 // return cublasSgemmStridedBatched(handle,
 //                                  transa,
 //                                  transb,
 //                                  m, n, k,
 //                                  alpha,
 //                                  A, lda, strideA,
 //                                  B, ldb, strideB,
 //                                  beta,
 //                                  C, ldc, strideC,
 //                                  batch_count);

  return cublasGemmStridedBatchedEx(handle,
                                   transa,
                                   transb,
                                   m, n, k,
                                   alpha,
                                   A, CUDA_R_32F, lda, strideA,
                                   B, CUDA_R_32F, ldb, strideB,
                                   beta,
                                   C, CUDA_R_32F, ldc, strideC,
                                   batch_count,
                                   CUDA_R_32F,
                                   CUBLAS_GEMM_ALGO0_TENSOR_OP);
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const double* alpha,
                                                     const double* A, int lda,
                                                     long long int strideA,
                                                     const double* B, int ldb,
                                                     long long int strideB,
                                                     const double* beta,
                                                     double* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& /*prop*/) {
  return cublasDgemmStridedBatched(handle,
                                   transa,
                                   transb,
                                   m, n, k,
                                   alpha,
                                   A, lda, strideA,
                                   B, ldb, strideB,
                                   beta,
                                   C, ldc, strideC,
                                   batch_count);
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const __half* alpha,
                                                     const __half* A, int lda,
                                                     long long int strideA,
                                                     const __half* B, int ldb,
                                                     long long int strideB,
                                                     const __half* beta,
                                                     __half* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& prop) {
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TENSOR_OP_MATH);

#ifdef ENABLE_TRAINING
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  // accumulating in FP32
  return cublasGemmStridedBatchedEx(handle,
                                    transa,
                                    transb,
                                    m, n, k,
                                    &h_a,
                                    A, CUDA_R_16F, lda, strideA,
                                    B, CUDA_R_16F, ldb, strideB,
                                    &h_b,
                                    C, CUDA_R_16F, ldc, strideC,
                                    batch_count,
                                    CUDA_R_32F,
                                    CUBLAS_GEMM_DEFAULT);
#else
  // accumulating in FP16
  return cublasHgemmStridedBatched(handle,
                                    transa,
                                    transb,
                                    m, n, k,
                                    alpha,
                                    A, lda, strideA,
                                    B, ldb, strideB,
                                    beta,
                                    C, ldc, strideC,
                                    batch_count);
#endif
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const nv_bfloat16* alpha,
                                                     const nv_bfloat16* A, int lda,
                                                     long long int strideA,
                                                     const nv_bfloat16* B, int ldb,
                                                     long long int strideB,
                                                     const nv_bfloat16* beta,
                                                     nv_bfloat16* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& prop) {
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, CUBLAS_TENSOR_OP_MATH);
  float h_a = onnxruntime::BFloat16(*reinterpret_cast<const uint16_t*>(alpha)).ToFloat();
  float h_b = onnxruntime::BFloat16(*reinterpret_cast<const uint16_t*>(beta)).ToFloat();
  // accumulating in FP32
  return cublasGemmStridedBatchedEx(handle,
                                    transa,
                                    transb,
                                    m, n, k,
                                    &h_a,
                                    A, CUDA_R_16BF, lda, strideA,
                                    B, CUDA_R_16BF, ldb, strideB,
                                    &h_b,
                                    C, CUDA_R_16BF, ldc, strideC,
                                    batch_count,
                                    CUDA_R_32F,
                                    CUBLAS_GEMM_DEFAULT);
}
#endif

// transpose using geam
inline cublasStatus_t cublasTransposeHelper(cudaStream_t, cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasTransposeHelper(cudaStream_t, cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
cublasStatus_t cublasTransposeHelper(cudaStream_t, cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int);

// copy
inline cublasStatus_t cublasCopyHelper(cudaStream_t, cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
  return cublasScopy(handle, n, x, incx, y, incy);
}
inline cublasStatus_t cublasCopyHelper(cudaStream_t, cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
  return cublasDcopy(handle, n, x, incx, y, incy);
}
cublasStatus_t cublasCopyHelper(cudaStream_t stream, cublasHandle_t handle, int n, const half* x, int incx, half* y, int incy);
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
cublasStatus_t cublasCopyHelper(cudaStream_t stream, cublasHandle_t handle, int n, const nv_bfloat16* x, int incx, nv_bfloat16* y, int incy);
#endif


// if (name == "MatMul_103") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1, &algo);
    //     cuBlasLtAlgoFill(algo, 13, 0, 0, 1);
    // } else if (name == "MatMul_117") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 2, 1, 0);
    // } else if (name == "MatMul_127") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 12, 1, 0);
    // } else if (name == "MatMul_197") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 3, 1, 0);
    // } else if (name == "MatMul_211") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 4, 1, 1);
    // } else if (name == "MatMul_221") { // problem
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo);
    // } else if (name == "MatMul_291") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 3, 1, 0);
    // } else if (name == "MatMul_305") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 2, 1, 0);
    // } else if (name == "MatMul_315") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 8, 1, 1);
    // } else if (name == "MatMul_385") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 6, 1, 1);
    // } else if (name == "MatMul_399") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 0, 0, 0);
    // } else if (name == "MatMul_409") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 12, 1, 0);
    // } else if (name == "MatMul_479") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1, &algo);
    //     cuBlasLtAlgoFill(algo, 13, 0, 0, 0);
    // } else if (name == "MatMul_493") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 0, 0, 0);
    // } else if (name == "MatMul_503") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 12, 1, 1);
    // } else if (name == "MatMul_573") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 20, 2, 1, 1);
    // } else if (name == "MatMul_587") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 3, 1, 1);
    // } else if (name == "MatMul_597") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 0, 0, 1);
    // } else if (name == "MatMul_667") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 3, 1, 0);
    // } else if (name == "MatMul_681") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 2, 1, 0);
    // } else if (name == "MatMul_691") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 0, 0, 1);
    // } else if (name == "MatMul_761") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 3, 1, 0);
    // } else if (name == "MatMul_775") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1, &algo);
    //     cuBlasLtAlgoFill(algo, 13, 0, 0, 1);
    // } else if (name == "MatMul_785") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 12, 1, 1);
    // } else if (name == "MatMul_855") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 3, 1, 0);
    // } else if (name == "MatMul_869") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 20, 2, 1, 1);
    // } else if (name == "MatMul_879") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 0, 0, 1);
    // } else if (name == "MatMul_949") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 3, 1, 0);
    // } else if (name == "MatMul_963") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 4, 1, 1);
    // } else if (name == "MatMul_973") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 20, 8, 1, 1);
    // } else if (name == "MatMul_1043") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 3, 1, 0);
    // } else if (name == "MatMul_1057") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 2, 1, 0);
    // } else if (name == "MatMul_1067") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 12, 1, 0);
    // } else if (name == "MatMul_1137") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1, &algo);
    //     cuBlasLtAlgoFill(algo, 13, 0, 0, 0);
    // } else if (name == "MatMul_1151") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1, &algo);
    //     cuBlasLtAlgoFill(algo, 16, 0, 0, 1);
    // } else if (name == "MatMul_1161") {
    //     cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo);
    //     cuBlasLtAlgoFill(algo, 18, 8, 1, 1);
    // }



