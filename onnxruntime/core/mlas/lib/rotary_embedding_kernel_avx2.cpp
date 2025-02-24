/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_avx2.cpp

Abstract:

    This module implements the rotary embedding kernels for AVX2 supported h/w.

--*/

#include "rotary_embedding.h"
#include "rotary_embedding_kernel_avx2.h"

//
// Kernel dispatch structure definition.
//
const MLAS_ROPE_DISPATCH MlasRopeDispatchAvx2 = []() {
    MLAS_ROPE_DISPATCH d;
    d.SRope = rope_avx2::RopeKernel_Avx2;
    return d;
}();
