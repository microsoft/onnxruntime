/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    eltwise_kernel_neon.h

Abstract:

    This module includes function declarations and common helper functions for
    element-wise operations on ARM cpu.

--*/

#pragma once

#include <arm_neon.h>

#include "mlasi.h"

namespace eltwise_neon {

void Add_Kernel_Fp16(const MLAS_FP16* left, const MLAS_FP16* right, MLAS_FP16* output, size_t N);

}  // namespace eltwise_neon
