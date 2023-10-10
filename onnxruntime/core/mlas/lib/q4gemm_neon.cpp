/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4gemm_neon.cpp

Abstract:

    This module implements the fp32 matrix multiplication with compressed
    weight tensor (right hand side). The assumption is the right hand side
    tensor can be pre-packed and compressed using int-4 quantization to save
    memory.

    This implementation is for ARM NEON.

--*/

#include <arm_neon.h>

#include "q4gemm.h"

struct MLAS_FP_Q4_GEMM_KERNEL_NEON {
    // static constexpr size_t StrideM = 256;
};

//
// MlasQ4GemmKernel and related helper functions
//

template <typename Q4Type>
MLAS_FORCEINLINE size_t
MlasQ4GemmKernelNeon(const float* A,
                     const uint8_t* PackedB,
                     float* C,
                     size_t CountM,
                     size_t CountN,
                     size_t CountK,
                     size_t lda,
                     size_t ldb,
                     size_t ldc,
                     const float* Bias);

template <>
MLAS_FORCEINLINE size_t
MlasQ4GemmKernelNeon<MLAS_Q4TYPE_BLK0>(const float* A,
                                       const uint8_t* PackedB,
                                       float* C,
                                       size_t CountM,
                                       size_t CountN,
                                       size_t CountK,
                                       size_t lda,
                                       size_t ldb,
                                       size_t ldc,
                                       const float* Bias)
{
    static_cast<void>((A, PackedB, C, CountM, CountN, CountK, lda, ldb, ldc, Bias));
    return 1;  // TODO ...
}

template <>
MLAS_FORCEINLINE size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_NEON>(const float* A,
                                                                const uint8_t* PackedB,
                                                                float* C,
                                                                size_t CountM,
                                                                size_t CountN,
                                                                size_t CountK,
                                                                size_t lda,
                                                                size_t ldb,
                                                                size_t ldc,
                                                                const float* Bias)
{
    return MlasQ4GemmKernelNeon<MLAS_Q4TYPE_BLK0>(A, PackedB, C, CountM, CountN, CountK, lda, ldb,
                                                  ldc, Bias);
}

//
// MlasBlkQ4DequantB and related helper functions
//

template <typename Q4Type>
MLAS_FORCEINLINE void
MlasBlkQ4DequantBNeon(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    static_cast<void>((FpData, PackedB, CountN, CountK, ldb));
    // TODO ...
}

template <>
MLAS_FORCEINLINE void
MlasBlkQ4DequantB<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_NEON>(
    float* FpData, const uint8_t* PackedB, size_t CountN, size_t CountK, size_t ldb)
{
    MlasBlkQ4DequantBNeon<MLAS_Q4TYPE_BLK0>(FpData, PackedB, CountN, CountK, ldb);
}

//
// MlasFpQ4GemmDispatchNeon structure population
//

static MLAS_Q4GEMM_OPERATION* Q4Operations_neon[] = {
    MlasQ4GemmOperation<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_NEON>,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

const MLAS_FPQ4GEMM_DISPATCH MlasFpQ4GemmDispatchNeon = {Q4Operations_neon};
