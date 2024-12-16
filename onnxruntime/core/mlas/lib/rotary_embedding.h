/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding.h

Abstract:

    This module includes kernel function prototypes and helper functions for
    implementing rotary embedding.

--*/

#pragma once

#include "mlasi.h"

struct MLAS_ROPE_DISPATCH {
    // rotary embedding kernel for fp32
    typedef void(SRope_Fn)(
        const float* input,
        const float* sin,
        const float* cos,
        size_t dim,
        bool interleaved,
        float* output
    );

    SRope_Fn* SRope = nullptr;

    // rotary embedding kernel for fp16
    typedef void(HRope_Fn)(
        const MLAS_FP16* input,
        const MLAS_FP16* sin,
        const MLAS_FP16* cos,
        size_t dim,
        bool interleaved,
        MLAS_FP16* output
    );

    HRope_Fn* HRope = nullptr;
};
