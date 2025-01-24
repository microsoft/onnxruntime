/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    hgemm_kernel_neon.cpp

Abstract:

    This module implements half precision GEMM kernel for neon.

--*/

#include "mlasi.h"
#include "halfgemm.h"

const MLAS_HGEMM_DISPATCH MlasHGemmDispatchNeon = [](){
    MLAS_HGEMM_DISPATCH d;
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    d.HPackBKernel_TransposedB = hgemm_neon::HPackB_TransposedB_Kernel;
    d.HGemmKernel_TransposedB = hgemm_neon::HGemm_TransposedB_Kernel;
    d.HGemmKernel_TransposedPackedB = hgemm_neon::HGemm_TransposedPackedB_Kernel;
#endif
    return d;
}();
