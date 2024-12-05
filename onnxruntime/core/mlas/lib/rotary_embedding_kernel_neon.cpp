/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_neon.cpp

Abstract:

    This module implements the rotary embedding kernels for ARM NEON.

--*/

#include "rotary_embedding.h"
#include "rotary_embedding_kernel_neon.h"

//
// Kernel dispatch structure definition.
//
const MLAS_ROPE_DISPATCH MlasRopeDispatchNeon = []() {
    MLAS_ROPE_DISPATCH d;

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    if (MlasFp16AccelerationSupported()) {
        d.HRope = rope_neon::RopeKernel_Fp16;
    }
#endif
    return d;
}();
