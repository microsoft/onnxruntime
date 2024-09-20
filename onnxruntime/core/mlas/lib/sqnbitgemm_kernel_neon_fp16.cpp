/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon_fp16.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON specific to
    input type T1 as float16.

--*/

#include <arm_neon.h>

#include "fp16_common.h"

// This file is enabled in cmake only if ARM64 is defined and not on Apple platforms
// The cmake condition is equivalent to MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64.
// Therefore omit the MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64 macro in this file.

void
MlasCastF16ToF32KernelNeon(const unsigned short* Source, float* Destination, size_t Count)
{

}
