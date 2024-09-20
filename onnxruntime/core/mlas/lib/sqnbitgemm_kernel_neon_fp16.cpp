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

#include "mlasi.h"
#include "mlas_float16.h"

// This file is enabled in cmake only if ARM64 is defined and not on Apple platforms
// The cmake condition is equivalent to MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64.
// Therefore omit the MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64 macro in this file.

void
MlasCastF16ToF32KernelNeon(const unsigned short* Source, float* Destination, size_t Count)
{
    size_t i = 0;
    // TODO(fajin): test unroll
    for (; i + 4 < Count; i += 4)
    {
        float16x4_t fp16v4_0 = vreinterpret_f16_u16(vld1_u16(Source + i));
        float32x4_t fp32v4_0 = vcvt_f32_f16(fp16v4_0);
        vst1q_f32(Destination + i, fp32v4_0);
    }

    if (i < Count)
    {
        float16x4_t fp16v4_0 = vreinterpret_f16_u16(vld1_u16(Source + i));
        float32x4_t fp32v4_0 = vcvt_f32_f16(fp16v4_0);
        for (size_t j = 0; i < Count; ++i, ++j)
        {
            Destination[i] = vgetq_lane_f32(fp32v4_0, j);
        }
    }
}

void
MlasCastF32ToF16KernelNeon(const float* Source, unsigned short* Destination, size_t Count)
{
    size_t i = 0;
    for (; i + 4 < Count; i += 4)
    {
        float32x4_t fp32v4_0 = vld1q_f32(Source + i);
        float16x4_t fp16v4_0 = vcvt_f16_f32(fp32v4_0);
        vst1_u16(Destination + i, vreinterpret_u16_f16(fp16v4_0));
    }

    if (i < Count)
    {
        float32x4_t fp32v4_0 = vld1q_f32(Source + i);
        float16x4_t fp16v4_0 = vcvt_f16_f32(fp32v4_0);
        uint16x4_t u16v4_0 = vreinterpret_u16_f16(fp16v4_0);
        for (size_t j = 0; i < Count; ++i, ++j)
        {
            Destination[i] = vget_lane_u16(u16v4_0, j);
        }
    }
}
