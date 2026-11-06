/*++

SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT

Module Name:

    sgemv_kernel.h

Abstract:

    This module implements an inline ARM64 ASIMD GEMV kernel for single-row SGEMM
    optionally fuses bias into the output

--*/

#pragma once

#include "mlasi.h"

namespace mlas_sgemm_m1
{

template <int Lane>
MLAS_FORCEINLINE void
Accumulate(float32x4_t (&sums)[8], const float* b, size_t panel_stride, float32x4_t a)
{
    for (size_t vector = 0; vector < 8; ++vector) {
        const float* weights = b + (vector / 4) * panel_stride + (vector % 4) * 4;
        sums[vector] = vfmaq_laneq_f32(sums[vector], vld1q_f32(weights), a, Lane);
    }
}

// Pair two standard 16-column packed panels for eight independent accumulators.
MLAS_FORCEINLINE void
ComputeBlock(const float* a, const float* b, float* c, size_t count_k, float alpha, bool zero_mode,
    const float* bias, float beta)
{
    const size_t panel_stride = count_k * 16;
    float32x4_t sums[8];
    for (auto& sum : sums) {
        sum = vdupq_n_f32(0.0f);
    }

    size_t k = 0;
    for (; count_k - k >= 4; k += 4) {
        const float32x4_t av = vld1q_f32(a + k);
        Accumulate<0>(sums, b + k * 16, panel_stride, av);
        Accumulate<1>(sums, b + (k + 1) * 16, panel_stride, av);
        Accumulate<2>(sums, b + (k + 2) * 16, panel_stride, av);
        Accumulate<3>(sums, b + (k + 3) * 16, panel_stride, av);
    }
    for (; k < count_k; ++k) {
        Accumulate<0>(sums, b + k * 16, panel_stride, vdupq_n_f32(a[k]));
    }

    if (zero_mode) {
        for (auto& sum : sums) {
            sum = vmulq_n_f32(sum, alpha);
        }
    } else {
        float32x4_t initial[8];
        if (bias == nullptr) {
            for (size_t vector = 0; vector < 8; ++vector) {
                initial[vector] = vld1q_f32(c + vector * 4);
            }
        } else {
            for (size_t vector = 0; vector < 8; ++vector) {
                initial[vector] = vld1q_f32(bias + vector * 4);
            }
            if (beta != 1.0f) {
                for (auto& element : initial) {
                    element = vmulq_n_f32(element, beta);
                }
            }
        }
        for (size_t vector = 0; vector < 8; ++vector) {
            sums[vector] = vfmaq_n_f32(initial[vector], sums[vector], alpha);
        }
    }
    for (size_t vector = 0; vector < 8; ++vector) {
        vst1q_f32(c + vector * 4, sums[vector]);
    }
}

}  // namespace mlas_sgemm_m1

MLAS_FORCEINLINE void
MlasSgemmKernelM1(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    float alpha,
    bool ZeroMode,
    const float* Bias,
    float beta
)
{
    while (CountN >= 32) {
        mlas_sgemm_m1::ComputeBlock(A, B, C, CountK, alpha, ZeroMode, Bias, beta);
        if (Bias != nullptr) {
            Bias += 32;
        }
        B += CountK * 32;
        C += 32;
        CountN -= 32;
    }
    // Reuse the existing assembly for the remaining columns.
    if (CountN != 0) {
        if (Bias != nullptr && !ZeroMode) {
            for (size_t n = 0; n < CountN; ++n) {
                const float value = Bias[n];
                C[n] = beta == 1.0f ? value : value * beta;
            }
        }
        if (ZeroMode) {
            MlasSgemmKernelZero(A, B, C, CountK, 1, CountN, CountK, CountN, alpha);
        } else {
            MlasSgemmKernelAdd(A, B, C, CountK, 1, CountN, CountK, CountN, alpha);
        }
    }
}
