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
    size_t i = 0;
    for (; i + 4 < Count; i += 4)
    {
        float16x4_t fp16_4 = vreinterpret_f16_u16(vld1_u16(Source + i));
        float32x4_t fp32_4 = vcvt_f32_f16(fp16_4);
        vst1q_f32(Destination + i, fp32_4);
    }

    if (i < Count)
    {
        float16x4_t fp16_4 = vreinterpret_f16_u16(vld1_u16(Source + i));
        float32x4_t fp32_4 = vcvt_f32_f16(fp16_4);
        for (size_t j = 0; i < Count; ++i, ++j)
        {
            Destination[i] = vgetq_lane_f32(fp32_4, j);
        }
    }
}
