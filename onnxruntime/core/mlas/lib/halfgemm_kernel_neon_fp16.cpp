/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemm_kernel_neon_fp16.cpp

Abstract:

    This module implements half precision GEMM kernel for neon.

--*/

#include <arm_neon.h>

#include "halfgemm.h"

namespace hgemm_neon {

void HGemm_TransposeB_Kernel(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    MLAS_FP16 alpha,
    MLAS_FP16 beta
) {

}

}  // namespace hgemm_neon
