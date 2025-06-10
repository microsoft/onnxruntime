/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    eltwise.h

Abstract:

    This module includes kernel function prototypes and helper functions for
    element-wise operations.

--*/
#pragma once

#include "mlasi.h"

struct MLAS_ELTWISE_DISPATCH {
    /**
     * @brief Compute the element-wise addition of the two given vectors
     * @param left          Address of the left operand
     * @param right         Address of the right operand
     * @param output        Address of the output array. Could be the same as the input array.
     * @param N             Number of elements in the input arrays
     */
    typedef void(Add_Fp16_Fn)(
        const MLAS_FP16* left,
        const MLAS_FP16* right,
        MLAS_FP16* output,
        size_t N
    );

    Add_Fp16_Fn* Add_Fp16 = nullptr;
};
