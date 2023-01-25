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
    for (size_t m = 0; m < CountM; m++) {
        for (size_t k = 0; k < CountK; k++) {
            *D++ = MLAS_Float2Half(*(A + m * lda + k));
        }
    }
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
    for (size_t k = 0; k < CountK; k++) {
        for (size_t n = 0; n < CountN; n++) {
            *D++ = MLAS_Float2Half(*(B + k * ldb + n));
        }
    }
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
