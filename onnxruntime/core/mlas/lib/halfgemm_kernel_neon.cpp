/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemm_kernel_neon.cpp

Abstract:

    This module implements half precision GEMM kernel for neon.

--*/

#include "mlasi.h"
#include "halfgemm.h"

//
// Define the prototypes of the NEON routines written in assembly.
//
// N.B. The kernel has not been ported to build with the Windows ARM32 toolset.
//

extern "C" {

    size_t
    MLASCALL
    MlasHalfGemmKernelNeon(
        const size_t CountM,
        const size_t CountN,
        const size_t CountK,
        _mlas_fp16_* C,
        size_t ldc,
        const _mlas_fp16_* Bias,
        const _mlas_fp16_* A,
        const size_t lda,
        const _mlas_fp16_* B,
        const size_t ldb,
        const bool ZeroMode
        );

}


struct MLAS_HALF_GEMM_KERNEL_NEON {
    static constexpr bool PackNeeded = false;
    static constexpr size_t KernelMaxM = 6;  // max # rows the vectorized kernel can process
    static constexpr size_t PackedK = 1;

    static constexpr MLAS_HALF_GEMM_STRIDES Strides{24, 128, 16};
};

/**
 * @brief Convert a 2D matrix from float to fp16
*/
MLAS_FORCEINLINE
void
CvtFloat2Half2D(
    _mlas_fp16_* dest,
    const float* src,
    size_t stride,
    size_t CntRow,
    size_t CntCol
    )
{
    int64_t stride_gap = size_t(int64_t(stride) - int64_t(CntCol));
    if (0 == stride_gap) {
        const size_t len = CntRow * CntCol;
        for (size_t i = 0; i < len; i++) {
            *dest++ = MLAS_Float2Half(*(src++));
        }
        return;
    }
    while (CntRow > 0) {
        for (size_t k = 0; k < CntCol; k++) {
            *dest++ = MLAS_Float2Half(*(src++));
        }
        src += stride_gap;
        CntRow--;
    }
}

template<>
MLAS_FORCEINLINE
void
MlasHalfGemmConvertPackA<MLAS_HALF_GEMM_KERNEL_NEON>(
    _mlas_fp16_* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK
)
{
    CvtFloat2Half2D(D, A, lda, CountM, CountK);
}

template<>
MLAS_FORCEINLINE
void
MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_NEON>(
    _mlas_fp16_* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
)
{
    CvtFloat2Half2D(D, B, ldb, CountK, CountN); 
}


template<>
MLAS_FORCEINLINE
void
MlasHalfGemmKernel<MLAS_HALF_GEMM_KERNEL_NEON>(
    size_t CountM,
    size_t CountN,
    size_t CountK,
    _mlas_fp16_* C,
    size_t ldc,
    const _mlas_fp16_* Bias,
    const _mlas_fp16_* A,
    size_t lda,
    const _mlas_fp16_* B,
    size_t ldb,
    const bool ZeroMode)
{
    MlasHalfGemmKernelNeon(
        CountM,
        CountN,
        CountK,
        C,
        ldc,
        Bias,
        A,
        lda,
        B,
        ldb,
        ZeroMode);
}


const MLAS_HALFGEMM_DISPATCH MlasHalfGemmDispatchNeon = {
    MlasHalfGemmOperation<MLAS_HALF_GEMM_KERNEL_NEON>,
    nullptr, 
    MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_NEON>,
    MLAS_HALF_GEMM_KERNEL_NEON::PackedK,
    MLAS_HALF_GEMM_KERNEL_NEON::KernelMaxM,
    32 // kernel may read beyond buffer end by 32 bytes
};
