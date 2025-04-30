/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #if defined(__GNUC__)
 #pragma GCC diagnostic push
 #pragma GCC diagnostic ignored "-Wunused-parameter"
 #endif

#include "contrib_ops/cuda/llm/common/cublasMMWrapper.h"
#include "contrib_ops/cuda/llm/common/assert.h"
#include "contrib_ops/cuda/llm/common/cublasVersionCheck.h"
#include <algorithm>

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

namespace ort_llm
{
namespace common
{

CublasMMWrapper::CublasMMWrapper(std::shared_ptr<cublasHandle_t> cublasHandle,
    std::shared_ptr<cublasLtHandle_t> cublasltHandle, cudaStream_t stream, void* workspace)
    : mCublasHandle(cublasHandle)
    , mCublasLtHandle(cublasltHandle)
    , mStream(stream)
    , mCublasWorkspace(workspace)
{
}

CublasMMWrapper::~CublasMMWrapper() {}

CublasMMWrapper::CublasMMWrapper(CublasMMWrapper const& wrapper)
    : mCublasHandle(wrapper.mCublasHandle)
    , mCublasLtHandle(wrapper.mCublasLtHandle)
    , mStream(wrapper.mStream)
{
}

void CublasMMWrapper::createDescriptors(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n,
    int const k, int const lda, int const ldb, int const ldc, int8_t fastAcc)
{
    // --------------------------------------
    // Create descriptors for the original matrices
    check_cuda_error(
        cublasLtMatrixLayoutCreate(&mADesc, mAType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    check_cuda_error(
        cublasLtMatrixLayoutCreate(&mBDesc, mBType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    check_cuda_error(cublasLtMatrixLayoutCreate(&mCDesc, mCType, m, n, ldc));
    check_cuda_error(cublasLtMatmulDescCreate(&mOperationDesc, mComputeType, mScaleType));
    check_cuda_error(cublasLtMatmulDescSetAttribute(
        mOperationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    check_cuda_error(cublasLtMatmulDescSetAttribute(
        mOperationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(mOperationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAcc, sizeof(int8_t)));
}

void CublasMMWrapper::setScaleDescriptors(void* scale_a, void* scale_b)
{
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(mOperationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(void*)));
    check_cuda_error(
        cublasLtMatmulDescSetAttribute(mOperationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(void*)));
}

void CublasMMWrapper::destroyDescriptors()
{
    check_cuda_error(cublasLtMatmulDescDestroy(mOperationDesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(mADesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(mBDesc));
    check_cuda_error(cublasLtMatrixLayoutDestroy(mCDesc));
    mOperationDesc = NULL;
    mADesc = NULL;
    mBDesc = NULL;
    mCDesc = NULL;
}

void CublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
    void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc)
{
    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f);
}

void CublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
    void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc,
    std::optional<cublasLtMatmulHeuristicResult_t> const& heuristic)
{
    if (heuristic)
    {
        Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f, /* hasAlgo */ (*heuristic).algo,
            (*heuristic).state == CUBLAS_STATUS_SUCCESS && (*heuristic).workspaceSize < CUBLAS_WORKSPACE_SIZE,
            /* usingCublasLt */ true);
    }
    else
    {
        Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f, {}, /* hasAlgo */ false,
            /* usingCublasLt */ true);
    }
}

void CublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
    void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc, float f_alpha, float f_beta,
    std::optional<cublasLtMatmulHeuristicResult_t> const& heuristic)
{
    if (heuristic)
    {
        Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, f_alpha, f_beta, /* hasAlgo */ (*heuristic).algo,
            (*heuristic).state == CUBLAS_STATUS_SUCCESS && (*heuristic).workspaceSize < CUBLAS_WORKSPACE_SIZE,
            /* usingCublasLt */ true);
    }
    else
    {
        Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, f_alpha, f_beta, {}, /* hasAlgo */ false,
            /* usingCublasLt */ true);
    }
}

void CublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
    void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc, float f_alpha, float f_beta)
{
    bool usingCublasLt = mAType == CUDA_R_16F || mAType == CUDA_R_8F_E4M3;

    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, f_alpha, f_beta, {}, /* hasAlgo */ false,
        /* usingCublasLt */ usingCublasLt);
}

void CublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
    void const* A, int const lda, void const* B, int const ldb, void* C, int const ldc, float f_alpha, float f_beta,
    cublasLtMatmulAlgo_t const& algo, bool hasAlgo, bool usingCublasLt)
{
    half h_alpha = (half) (f_alpha);
    half h_beta = (half) (f_beta);

    // TODO: default cublas libs
    usingCublasLt = usingCublasLt && (mAType == CUDA_R_16F || mAType == CUDA_R_8F_E4M3 || mAType == CUDA_R_16BF);
    bool isFp16ComputeType = mComputeType == CUBLAS_COMPUTE_16F;
    // fp32 use cublas as default
    // fp16 use cublasLt as default
    void const* alpha = isFp16ComputeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    void const* beta = isFp16ComputeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);
    int workspaceSize = mCublasWorkspace == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

    if (usingCublasLt)
    {
        if (hasAlgo)
        {
            hasAlgo = checkTactic(transa, transb, m, n, k, lda, ldb, ldc, algo);
        }

        check_cuda_error(cublasLtMatmul(getCublasLtHandle(), mOperationDesc, alpha, A, mADesc, B, mBDesc, beta, C,
            mCDesc, C, mCDesc, (hasAlgo ? (&algo) : NULL), mCublasWorkspace, workspaceSize, mStream));

        sync_check_cuda_error(mStream);
    }
    else
    {
        check_cuda_error(cublasSetStream(getCublasHandle(), mStream));
        check_cuda_error(cublasSetWorkspace(getCublasHandle(), mCublasWorkspace, workspaceSize));
        // Go with default heuristic to choose tactic as cuBLAS does not allow to choose tactics in Ampere+
        cublasGemmAlgo_t cublasAlgo = CUBLAS_GEMM_DEFAULT;
        check_cuda_error(cublasGemmEx(getCublasHandle(), transa, transb, m, n, k, alpha, A, mAType, lda, B, mBType, ldb,
            beta, C, mCType, ldc, mComputeType, static_cast<cublasGemmAlgo_t>(cublasAlgo)));
        sync_check_cuda_error(mStream);
    }
}

void CublasMMWrapper::stridedBatchedGemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n,
    int const k, void const* A, int const lda, const int64_t strideA, void const* B, int const ldb,
    const int64_t strideB, void* C, int const ldc, const int64_t strideC, int const batchCount, float const f_alpha,
    float const f_beta)
{
    half h_alpha = (half) f_alpha;
    half h_beta = (half) f_beta;

    int isFp16ComputeType = mComputeType == CUBLAS_COMPUTE_16F ? 1 : 0;
    void const* alpha = isFp16ComputeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void const*>(&f_alpha);
    void const* beta = isFp16ComputeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void const*>(&f_beta);

    check_cuda_error(cublasGemmStridedBatchedEx(getCublasHandle(), transa, transb, m, n, k, alpha, A, mAType, lda,
        strideA, B, mBType, ldb, strideB, beta, C, mCType, ldc, strideC, batchCount, mComputeType,
        mAType == CUDA_R_32F ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void CublasMMWrapper::stridedBatchedGemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n,
    int const k, float const f_alpha, void const* A, cudaDataType_t AType, int const lda, const int64_t strideA,
    void const* B, cudaDataType_t BType, int const ldb, const int64_t strideB, float const f_beta, void* C,
    cudaDataType_t CType, int const ldc, const int64_t strideC, int const batchCount, cudaDataType_t computeType)
{
    half h_alpha = (half) f_alpha;
    half h_beta = (half) f_beta;

    bool isFp16ComputeType = mComputeType == CUBLAS_COMPUTE_16F ? 1 : 0;
    void const* alpha = isFp16ComputeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void const*>(&f_alpha);
    void const* beta = isFp16ComputeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void const*>(&f_beta);

    check_cuda_error(cublasGemmStridedBatchedEx(getCublasHandle(), transa, transb, m, n, k, alpha, A, AType, lda,
        strideA, B, BType, ldb, strideB, beta, C, CType, ldc, strideC, batchCount, computeType,
        mAType == CUDA_R_32F ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void CublasMMWrapper::setWorkspace(void* workspace)
{
    mCublasWorkspace = workspace;
}

void CublasMMWrapper::setFP32GemmConfig()
{
    setGemmConfig(CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F);
}

void CublasMMWrapper::setFP16GemmConfig(cudaDataType_t outputType)
{
    setGemmConfig(CUDA_R_16F, CUDA_R_16F, outputType, CUDA_R_32F);
}

#ifdef ENABLE_BF16
void CublasMMWrapper::setBF16GemmConfig(cudaDataType_t outputType)
{
    setGemmConfig(CUDA_R_16BF, CUDA_R_16BF, outputType, CUDA_R_32F);
}
#endif

#ifdef ENABLE_FP8
void CublasMMWrapper::setFP8GemmConfig(cudaDataType_t outputType)
{
    setGemmConfig(CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, outputType, CUDA_R_32F);
}
#endif

void CublasMMWrapper::setGemmConfig(
    cudaDataType_t aType, cudaDataType_t bType, cudaDataType_t cType, cudaDataType_t computeType)
{
    mAType = aType;
    mBType = bType;
    mCType = cType;
    bool isFp16ComputeType = computeType == CUDA_R_16F;
    if (isFp16ComputeType)
    {
        mComputeType = CUBLAS_COMPUTE_16F;
        mScaleType = CUDA_R_16F;
    }
    else
    {
        mComputeType = CUBLAS_COMPUTE_32F;
        mScaleType = CUDA_R_32F;
    }
}

CublasDataType CublasMMWrapper::getCublasDataType(cudaDataType_t data_type)
{
    if (data_type == CUDA_R_16F)
    {
        return HALF_DATATYPE;
    }
    else if (data_type == CUDA_R_32F)
    {
        return FLOAT_DATATYPE;
    }
    else if (data_type == CUDA_R_8I)
    {
        return INT8_DATATYPE;
    }
#ifdef ENABLE_BF16
    else if (data_type == CUDA_R_16BF)
    {
        return BFLOAT16_DATATYPE;
    }
#endif
    return FLOAT_DATATYPE;
}

void CublasMMWrapper::setStream(cudaStream_t stream)
{
    mStream = stream;
}

namespace
{

static inline char const* mmaToString(uint16_t mma)
{
    static char const* mmaStr[] = {
        "UNDEF", //
        "MMA884",
        "MMA1684",
        "MMA1688",
        "MMA16816",
    };

    static_assert(sizeof(mmaStr) / sizeof(mmaStr[0]) == CUBLASLT_MATMUL_INNER_SHAPE_END,
        "all mma configs must be listed in the metadata table");

    if (mma >= sizeof(mmaStr) / sizeof(mmaStr[0]))
        return "UNKNOWN";
    return mmaStr[mma];
}

static inline char const* cgaToString(uint16_t cga)
{
    // clang-format off
  static const char* cgaStr[] = {"AUTO",
                                 "ILLEGAL",
                                 "1x1x1",
                                 "1x2x1",
                                 "1x4x1",
                                 "2x1x1",
                                 "2x2x1",
                                 "2x4x1",
                                 "4x1x1",
                                 "4x2x1",
                                 "4x4x1",
                                 "1x8x1",
                                 "8x1x1",
                                 "2x8x1",
                                 "8x2x1",
                                 "1x16x1",
                                 "16x1x1",
                                 "1x3x1",
                                 "1x5x1",
                                 "1x6x1",
                                 "1x7x1",
                                 "1x9x1",
                                 "1x10x1",
                                 "1x11x1",
                                 "1x12x1",
                                 "1x13x1",
                                 "1x14x1",
                                 "1x15x1",
                                 "2x3x1",
                                 "2x5x1",
                                 "2x6x1",
                                 "2x7x1",
                                 "3x1x1",
                                 "3x2x1",
                                 "3x3x1",
                                 "3x4x1",
                                 "3x5x1",
                                 "4x3x1",
                                 "5x1x1",
                                 "5x2x1",
                                 "5x3x1",
                                 "6x1x1",
                                 "6x2x1",
                                 "7x1x1",
                                 "7x2x1",
                                 "9x1x1",
                                 "10x1x1",
                                 "11x1x1",
                                 "12x1x1",
                                 "13x1x1",
                                 "14x1x1",
                                 "15x1x1",
                                 };
    // clang-format on

    static_assert(sizeof(cgaStr) / sizeof(cgaStr[0]) == CUBLASLT_CLUSTER_SHAPE_END,
        "all cga configs must be listed in the metadata table");

    if (cga >= sizeof(cgaStr) / sizeof(cgaStr[0]))
        return "UNKNOWN";
    return cgaStr[cga];
}

static void print_algo(cublasLtMatmulAlgo_t const* matmulAlgo)
{
    int algoId, tile, stages, swizzle, customOption, numSplitsK, reductionScheme;
    uint16_t mma, cga;

    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(
        matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);

    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &mma, sizeof(mma), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cga, sizeof(cga), NULL);

    TLLM_LOG_DEBUG(
        "algo={ %d %d %d splitK=%d reduc=%d swizzle=%d custom=%d mma=%s cga=%s}"
        " [-algo%d -m_tile%d -m_stages%d -m_numsK%d -m_reduction%d -m_swizzle%d -m_custom%d -m_mma%d -m_cga%d "
        "\n",
        algoId, tile, stages, numSplitsK, reductionScheme, swizzle, customOption, mmaToString(mma), cgaToString(cga),
        algoId, tile, stages, numSplitsK, reductionScheme, swizzle, customOption, mma, cga);
}

} // namespace

bool CublasMMWrapper::checkTactic(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n,
    int const k, int const lda, int const ldb, int const ldc, cublasLtMatmulAlgo_t const& algo)
{
    TLLM_CHECK_WITH_INFO(
        descriptorsCreated(), "Descriptors are not created! Call createDescriptors before calling this function");

    cublasLtMatmulHeuristicResult_t heurResult;
    cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(
        getCublasLtHandle(), mOperationDesc, mADesc, mBDesc, mCDesc, mCDesc, &algo, &heurResult);

    if (algoStatus != CUBLAS_STATUS_SUCCESS || heurResult.state != CUBLAS_STATUS_SUCCESS
        || heurResult.workspaceSize > CUBLAS_WORKSPACE_SIZE)
    {
        TLLM_LOG_WARNING("CheckTactic failed with status: %d and heuristic status: %d with workspace size: %d.\n",
            algoStatus, heurResult.state, heurResult.workspaceSize);
        return false;
    }

    sync_check_cuda_error(mStream);

    return true;
}

std::vector<cublasLtMatmulHeuristicResult_t> CublasMMWrapper::getTactics(cublasOperation_t transa,
    cublasOperation_t transb, int const m, int const n, int const k, int const lda, int const ldb, int const ldc)
{
    TLLM_CHECK_WITH_INFO(
        descriptorsCreated(), "Descriptors are not created! Call createDescriptors before calling this function");

    auto const heuristics = getTactics(getCublasLtHandle(), mOperationDesc, mADesc, mBDesc, mCDesc, mCDesc);

    sync_check_cuda_error(mStream);

    return heuristics;
}

std::vector<cublasLtMatmulHeuristicResult_t> CublasMMWrapper::getTactics(cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc)
{
#if TLLM_CUBLAS_VER_LE(11, 4, 2)
    TLLM_CHECK_WITH_INFO(false, "CUBLAS version too low, must be > 11.4.2.");
    return {};
#else
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(200);
    cublasLtMatmulPreference_t preference;
    check_cuda_error(cublasLtMatmulPreferenceCreate(&preference));
    check_cuda_error(cublasLtMatmulPreferenceInit(preference));
    uint64_t workspace_size = CUBLAS_WORKSPACE_SIZE;
    check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
    // Restrict reduction algorithms for numerical stability and better determinism
    uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_MASK;
    check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_mask, sizeof(reduction_mask)));
#if TLLM_CUBLAS_VER_LT(12, 0, 0)
    uint32_t pointer_mode_mask = 0;
    check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK, &pointer_mode_mask, sizeof(pointer_mode_mask)));
#endif

    int return_count = 0;
    check_cuda_error(cublasLtMatmulAlgoGetHeuristic(lightHandle, computeDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
        heuristics.size(), heuristics.data(), &return_count));
    heuristics.resize(return_count);

    return heuristics;
#endif
}

} // namespace common

} // namespace ort_llm

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif