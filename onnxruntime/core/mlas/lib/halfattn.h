/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfattn.h

Abstract:

    This module includes the kernel dispatch structure and kernel declarations
    for the attention products over fp16 key/value caches that accumulate in
    fp32 (MlasHalfAttnQK, MlasHalfAttnPV).

--*/

#pragma once

#include "mlasi.h"

struct MLAS_HALF_ATTN_DISPATCH {
    /**
     * @brief C[m, n] = Alpha * sum_k A[m, k] * B[n, k], fp16 A and B, fp32 products and sums. See MlasHalfAttnQK.
     */
    typedef void(QK_Fn)(
        size_t M,
        size_t N,
        size_t K,
        float Alpha,
        const MLAS_FP16* A,
        size_t lda,
        const MLAS_FP16* B,
        size_t ldb,
        float* C,
        size_t ldc
    );

    QK_Fn* QK = nullptr;

    /**
     * @brief C[m, n] += sum_k A[m, k] * B[k, n], fp32 A, fp16 B, fp32 products and sums. See MlasHalfAttnPV.
     */
    typedef void(PV_Fn)(
        size_t M,
        size_t N,
        size_t K,
        const float* A,
        size_t lda,
        const MLAS_FP16* B,
        size_t ldb,
        float* C,
        size_t ldc
    );

    PV_Fn* PV = nullptr;
};

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

namespace halfattn_neon {

//
// Widen kernels (halfattn_kernel_neon_fp16.cpp): fp16 elements are widened to
// fp32 in registers (FCVTL) and multiplied with fp32 FMLA.
//

void QK_Kernel_Widen(size_t M, size_t N, size_t K, float Alpha, const MLAS_FP16* A, size_t lda,
                     const MLAS_FP16* B, size_t ldb, float* C, size_t ldc);

void PV_Kernel_Widen(size_t M, size_t N, size_t K, const float* A, size_t lda, const MLAS_FP16* B, size_t ldb,
                     float* C, size_t ldc);

//
// FEAT_FHM kernel (halfattn_kernel_neon_fhm.cpp): FMLAL/FMLAL2 multiply fp16
// operands and accumulate in fp32 without widening instructions.
//

void QK_Kernel_Fhm(size_t M, size_t N, size_t K, float Alpha, const MLAS_FP16* A, size_t lda,
                   const MLAS_FP16* B, size_t ldb, float* C, size_t ldc);

}  // namespace halfattn_neon

#endif
