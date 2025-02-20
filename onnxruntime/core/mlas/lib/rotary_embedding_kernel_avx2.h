/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_avx2.h

Abstract:

    This module includes function declarations and common helper functions for
    rotary embedding on for AVX2 enabled h/w.

--*/

#pragma once



#include "mlasi.h"

namespace rope_avx2 {

// Rotary embedding kernel for FP32. Embed one hidden state vector.
void
RopeKernel_Avx2(
    const float* input,
    const float* sin_data,
    const float* cos_data,
    size_t dim,
    bool interleaved,
    float* output
);

}  // namespace rope_avx2
