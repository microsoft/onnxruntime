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

#include "arm_neon.h"

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

    static constexpr MLAS_HALF_GEMM_STRIDES Strides{24, 128, 512};
};


MLAS_FORCEINLINE
void
CvtFloat2Half(
    _mlas_fp16_* dest,
    const float* src,
    size_t len
)
{
    while (len >= 4) {
        const auto* srcPtr = reinterpret_cast<const float32x4_t*>(src);
        auto* dstPtr = reinterpret_cast<float16x4_t*>(dest);
        *dstPtr = vcvt_f16_f32(*srcPtr);
        src += 4;
        dest += 4;
        len -= 4;
    }

    if (0 == len) {
        return;
    }

    float32x4_t buf;
    std::memcpy(&buf, src, len * sizeof(float));
    float16x4_t res = vcvt_f16_f32(buf);

    if ((len & 2) != 0) {
        auto wide = vreinterpret_f32_f16(res);
        vst1_lane_f32((float32_t*)dest, wide, 0);
        res = vreinterpret_f16_f32(vdup_lane_f32(wide, 1));
        dest += 2;
    }
    if ((len & 1) != 0) {
        vst1_lane_u16(dest, vreinterpret_u16_f16(res), 0);
    }
}

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
    if (stride == CntCol) {
        const size_t len = CntRow * CntCol;
        CvtFloat2Half(dest, src, len);
        return;
    }
    while (CntRow > 0) {
        CvtFloat2Half(dest, src, CntCol);
        src += stride;
        dest += CntCol;
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
