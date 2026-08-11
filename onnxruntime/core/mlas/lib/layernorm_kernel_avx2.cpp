/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    layernorm_kernel_avx2.cpp

Abstract:

    This module implements LayerNorm/RMSNorm kernels using x86-64 AVX2+FMA3
    intrinsics. Processes one normalization row at a time, matching the
    MLAS_LAYERNORM_F32_KERNEL signature dispatched from platform.cpp.

    The kernel vectorises the two passes (reduce → normalise) over a single
    row, processing 8 floats per iteration via 256-bit registers. A scalar
    tail handles lengths that are not a multiple of 8.

--*/

#include "mlasi.h"

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)

#include <immintrin.h>

#include <cassert>
#include <cmath>

void MLASCALL
MlasLayerNormKernelAvx2(
    const float* Input,
    const float* Scale,
    const float* Bias,
    float* Output,
    float* MeanOut,
    float* InvStdDevOut,
    size_t NormSize,
    float Epsilon,
    bool Simplified)
{
    assert(!Simplified || Bias == nullptr);

    const size_t n = NormSize;

    //
    // Pass 1: Compute sum and sum-of-squares in a single pass.
    //

    __m256 vsum = _mm256_setzero_ps();
    __m256 vsumsq = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(Input + i);
        vsum = _mm256_add_ps(vsum, vx);
        vsumsq = _mm256_fmadd_ps(vx, vx, vsumsq);
    }

    // Horizontal reduction: sum the 8 lanes.
    // vsum = [s0 s1 s2 s3 | s4 s5 s6 s7]
    __m128 hi_sum = _mm256_extractf128_ps(vsum, 1);
    __m128 lo_sum = _mm256_castps256_ps128(vsum);
    __m128 r_sum = _mm_add_ps(lo_sum, hi_sum);
    r_sum = _mm_add_ps(r_sum, _mm_movehl_ps(r_sum, r_sum));
    r_sum = _mm_add_ss(r_sum, _mm_movehdup_ps(r_sum));
    float sum_val = _mm_cvtss_f32(r_sum);

    __m128 hi_sq = _mm256_extractf128_ps(vsumsq, 1);
    __m128 lo_sq = _mm256_castps256_ps128(vsumsq);
    __m128 r_sq = _mm_add_ps(lo_sq, hi_sq);
    r_sq = _mm_add_ps(r_sq, _mm_movehl_ps(r_sq, r_sq));
    r_sq = _mm_add_ss(r_sq, _mm_movehdup_ps(r_sq));
    float sumsq_val = _mm_cvtss_f32(r_sq);

    // Scalar tail.
    for (; i < n; i++) {
        float x = Input[i];
        sum_val += x;
        sumsq_val += x * x;
    }

    //
    // Compute mean and inverse standard deviation.
    //

    float mean_val = sum_val / static_cast<float>(n);
    float denom;
    if (Simplified) {
        denom = sqrtf(sumsq_val / static_cast<float>(n) + Epsilon);
    } else {
        denom = sqrtf(sumsq_val / static_cast<float>(n) -
                       mean_val * mean_val + Epsilon);
    }
    float inv_denom = 1.0f / denom;

    //
    // Pass 2: Normalise and write output.
    //

    __m256 vmean = _mm256_set1_ps(mean_val);
    __m256 vinv = _mm256_set1_ps(inv_denom);

    i = 0;
    if (Simplified) {
        for (; i + 8 <= n; i += 8) {
            __m256 vx = _mm256_loadu_ps(Input + i);
            __m256 vs = _mm256_loadu_ps(Scale + i);
            __m256 vy = _mm256_mul_ps(vx, vinv);
            vy = _mm256_mul_ps(vy, vs);
            _mm256_storeu_ps(Output + i, vy);
        }
        for (; i < n; i++) {
            Output[i] = Input[i] * inv_denom * Scale[i];
        }
    } else if (Bias == nullptr) {
        for (; i + 8 <= n; i += 8) {
            __m256 vx = _mm256_loadu_ps(Input + i);
            __m256 vs = _mm256_loadu_ps(Scale + i);
            __m256 vy = _mm256_sub_ps(vx, vmean);
            vy = _mm256_mul_ps(vy, vinv);
            vy = _mm256_mul_ps(vy, vs);
            _mm256_storeu_ps(Output + i, vy);
        }
        for (; i < n; i++) {
            Output[i] = (Input[i] - mean_val) * inv_denom * Scale[i];
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 vx = _mm256_loadu_ps(Input + i);
            __m256 vs = _mm256_loadu_ps(Scale + i);
            __m256 vb = _mm256_loadu_ps(Bias + i);
            __m256 vy = _mm256_sub_ps(vx, vmean);
            vy = _mm256_mul_ps(vy, vinv);
            vy = _mm256_fmadd_ps(vy, vs, vb);
            _mm256_storeu_ps(Output + i, vy);
        }
        for (; i < n; i++) {
            Output[i] = (Input[i] - mean_val) * inv_denom * Scale[i] + Bias[i];
        }
    }

    if (MeanOut != nullptr) {
        *MeanOut = mean_val;
    }
    if (InvStdDevOut != nullptr) {
        *InvStdDevOut = inv_denom;
    }
}

#endif  // MLAS_TARGET_AMD64 || MLAS_TARGET_IX86
