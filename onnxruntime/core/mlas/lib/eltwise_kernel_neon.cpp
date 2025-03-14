/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    eltwise_kernel_neon.cpp

Abstract:

    This module implements the element-wise kernels for ARM NEON.

--*/

#include "eltwise.h"
#include "eltwise_kernel_neon.h"

//
// Kernel dispatch structure definition.
//
const MLAS_ELTWISE_DISPATCH MlasEltwiseDispatchNeon = []() {
    MLAS_ELTWISE_DISPATCH d;

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    if (MlasFp16AccelerationSupported()) {
        d.Add_Fp16 = eltwise_neon::Add_Kernel_Fp16;
    }
#endif
    return d;
}();
