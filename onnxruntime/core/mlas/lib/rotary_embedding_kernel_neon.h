/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_neon.h

Abstract:

    This module includes function declarations and common helper functions for
    rotary embedding on ARM cpu.

--*/

#pragma once

#include <arm_neon.h>

#include "mlasi.h"

namespace rope_neon {

// Rotary embedding kernel for fp16. Embed one hidden state vector.
void
RopeKernel_Fp16(
    const MLAS_FP16* input,
    const MLAS_FP16* sin,
    const MLAS_FP16* cos,
    size_t dim,
    bool interleaved,
    MLAS_FP16* output
);

}  // namespace rope_neon
