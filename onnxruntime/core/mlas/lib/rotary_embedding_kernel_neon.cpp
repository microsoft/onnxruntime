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

namespace rope_neon {



}  // namespace rope_neon


//
// Kernel dispatch structure definition.
//
const MLAS_ROPE_DISPATCH MlasRopeDispatchNeon = []() {
    MLAS_ROPE_DISPATCH d;

    if (MlasFp16AccelerationSupported()) {
        d.HRope = nullptr;  // TODO
    }

    return d;
}();
