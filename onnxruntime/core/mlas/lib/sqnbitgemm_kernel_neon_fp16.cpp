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

#include "mlasi.h"

#include "arm_neon.h"

// This file is enabled in cmake only if ARM64 is defined and not on Apple platforms
// The cmake condition is equivalent to MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64.
// Therefore omit the MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64 macro in this file.

void
// TODO(fajin): test unrolling
MlasCastF16ToF32KernelNeon(const unsigned short* src, float* dest, size_t count)
{
    // 4 float16 alignment
    auto* src_aligned = reinterpret_cast<const unsigned short*>((reinterpret_cast<uintptr_t>(src) + 7) & ~7);
    auto pre_count = std::min(static_cast<size_t>(src_aligned - src), count);
    size_t i = 0;

    // Handle leading unaligned src
    if (pre_count > 0) {
        float16x4_t fp16v4;
        std::memcpy(&fp16v4, src, pre_count * sizeof(unsigned short));
        float32x4_t fp32v4 = vcvt_f32_f16(fp16v4);

        for (size_t j = 0; j < pre_count; ++i, ++j) {
            vst1q_lane_f32(dest + i, fp32v4, j);
        }
    }

    // aligned src
    for (; i + 4 < count; i += 4)
    {
        float16x4_t fp16v4_0 = vreinterpret_f16_u16(vld1_u16(src + i));
        float32x4_t fp32v4_0 = vcvt_f32_f16(fp16v4_0);
        vst1q_f32(dest + i, fp32v4_0);
    }

    // Handle trailing unaligned src
    auto post_count = count - i;
    if (post_count > 0)
    {
        float16x4_t fp16v4;
        std::memcpy(&fp16v4, src + i, post_count * sizeof(unsigned short));
        float32x4_t fp32v4 = vcvt_f32_f16(fp16v4);

        for (size_t j = 0; j < post_count; ++i, ++j)
        {
            vst1q_lane_f32(dest + i, fp32v4, j);
        }
    }
}

void
// TODO(fajin): test unrolling
MlasCastF32ToF16KernelNeon(const float* src, unsigned short* dest, size_t count)
{
    // 4 float32 alignment
    auto* src_aligned = reinterpret_cast<const float*>((reinterpret_cast<uintptr_t>(src) + 15) & ~15);
    auto pre_count = std::min(static_cast<size_t>(src_aligned - src), count);
    size_t i = 0;

    // Handle leading unaligned src
    if (pre_count > 0)
    {
        float32x4_t fp32v4;
        std::memcpy(&fp32v4, src, pre_count * sizeof(float));
        uint16x4_t u16v4 = vreinterpret_u16_f16(vcvt_f16_f32(fp32v4));
        for (size_t j = 0; j < pre_count; ++i, ++j)
        {
            vst1_lane_u16(dest + i, u16v4, j);
        }
    }

    // aligned src
    for (; i + 4 < count; i += 4)
    {
        float32x4_t fp32v4_0 = vld1q_f32(src + i);
        float16x4_t fp16v4_0 = vcvt_f16_f32(fp32v4_0);
        vst1_u16(dest + i, vreinterpret_u16_f16(fp16v4_0));
    }

    // Handle trailing unaligned src
    auto post_count = count - i;
    if (post_count > 0)
    {
        float32x4_t fp32v4;
        std::memcpy(&fp32v4, src + i, post_count * sizeof(float));
        uint16x4_t u16v4 = vreinterpret_u16_f16(vcvt_f16_f32(fp32v4));
        for (size_t j = 0; i < count; ++i, ++j)
        {
            vst1_lane_u16(dest + i, u16v4, j);
        }
    }
}
