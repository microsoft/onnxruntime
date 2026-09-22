/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    layernorm_kernel_avx2.cpp

Abstract:

    This module implements LayerNorm/RMSNorm kernels using x86 AVX2+FMA3
    intrinsics. Processes one normalization row at a time, matching the
    MLAS_LAYERNORM_F32_KERNEL signature dispatched from platform.cpp.

    RMSNorm uses a vectorised sum-of-squares accumulation (two-pass:
    reduce then normalise), processing 8 floats per iteration.

    Full LayerNorm uses a centered two-pass algorithm:
      Pass 1 — compute the mean via a double-precision sum (4 doubles
               per AVX2 iteration using vcvtps2pd + vaddpd). This keeps
               rounding in the mean from corrupting the centered variance
               for large-magnitude inputs.
      Pass 2 — accumulate sum((x - mean)^2) in fp32 (8 floats per
               iteration). Subtracting the (accurate) mean before
               squaring eliminates the catastrophic cancellation that
               plagues the uncentered E[x^2]-mean^2 formulation.

    A scalar tail handles lengths that are not a multiple of 8 (or 4
    for the double-precision mean pass).

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

    float mean_val;
    float inv_denom;

    if (Simplified) {
        //
        // RMSNorm: accumulate sum-of-squares for the inverse RMS
        // denominator. The mean is only needed when the caller requests
        // MeanOut — the normalisation itself never subtracts the mean.
        // Skip the sum accumulation when MeanOut is null to avoid ~1
        // extra vaddps per 8-element iteration.
        //

        __m256 vsumsq = _mm256_setzero_ps();
        size_t i = 0;
        float sumsq_val;

        if (MeanOut != nullptr) {
            //
            // Caller wants the mean: accumulate sum in double precision
            // alongside sum-of-squares in fp32.
            //
            __m256d vsumd = _mm256_setzero_pd();
            for (; i + 8 <= n; i += 8) {
                __m256 vx = _mm256_loadu_ps(Input + i);
                __m128 vx_lo = _mm256_castps256_ps128(vx);
                __m128 vx_hi = _mm256_extractf128_ps(vx, 1);
                vsumd = _mm256_add_pd(vsumd, _mm256_cvtps_pd(vx_lo));
                vsumd = _mm256_add_pd(vsumd, _mm256_cvtps_pd(vx_hi));
                vsumsq = _mm256_fmadd_ps(vx, vx, vsumsq);
            }

            // Horizontal reduce double sum.
            __m128d hi_d = _mm256_extractf128_pd(vsumd, 1);
            __m128d lo_d = _mm256_castpd256_pd128(vsumd);
            __m128d rd = _mm_add_pd(lo_d, hi_d);
            rd = _mm_add_sd(rd, _mm_unpackhi_pd(rd, rd));
            double dsum = _mm_cvtsd_f64(rd);

            // Horizontal reduce sum-of-squares.
            __m128 hi_sq = _mm256_extractf128_ps(vsumsq, 1);
            __m128 lo_sq = _mm256_castps256_ps128(vsumsq);
            __m128 r_sq = _mm_add_ps(lo_sq, hi_sq);
            r_sq = _mm_add_ps(r_sq, _mm_movehl_ps(r_sq, r_sq));
            r_sq = _mm_add_ss(r_sq, _mm_movehdup_ps(r_sq));
            sumsq_val = _mm_cvtss_f32(r_sq);

            for (; i < n; i++) {
                dsum += static_cast<double>(Input[i]);
                sumsq_val += Input[i] * Input[i];
            }

            mean_val = static_cast<float>(dsum / static_cast<double>(n));
        } else {
            //
            // No mean requested: sum-of-squares only.
            //
            for (; i + 8 <= n; i += 8) {
                __m256 vx = _mm256_loadu_ps(Input + i);
                vsumsq = _mm256_fmadd_ps(vx, vx, vsumsq);
            }

            // Horizontal reduce sum-of-squares.
            __m128 hi_sq = _mm256_extractf128_ps(vsumsq, 1);
            __m128 lo_sq = _mm256_castps256_ps128(vsumsq);
            __m128 r_sq = _mm_add_ps(lo_sq, hi_sq);
            r_sq = _mm_add_ps(r_sq, _mm_movehl_ps(r_sq, r_sq));
            r_sq = _mm_add_ss(r_sq, _mm_movehdup_ps(r_sq));
            sumsq_val = _mm_cvtss_f32(r_sq);

            for (; i < n; i++) {
                sumsq_val += Input[i] * Input[i];
            }

            mean_val = 0.0f;
        }

        inv_denom = 1.0f / sqrtf(sumsq_val / static_cast<float>(n) + Epsilon);
    } else {
        //
        // Full LayerNorm: centered two-pass algorithm.
        //
        // Pass 1 — Compute the mean using double-precision accumulation.
        // This prevents fp32 summation error for large-magnitude inputs
        // from corrupting the centered variance in the second pass.
        //

        __m256d vsumd = _mm256_setzero_pd();
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 vf = _mm_loadu_ps(Input + i);
            vsumd = _mm256_add_pd(vsumd, _mm256_cvtps_pd(vf));
        }

        // Horizontal reduce the 4 double lanes.
        __m128d hi_d = _mm256_extractf128_pd(vsumd, 1);
        __m128d lo_d = _mm256_castpd256_pd128(vsumd);
        __m128d rd = _mm_add_pd(lo_d, hi_d);
        rd = _mm_add_sd(rd, _mm_unpackhi_pd(rd, rd));
        double dsum = _mm_cvtsd_f64(rd);

        for (; i < n; i++) {
            dsum += static_cast<double>(Input[i]);
        }

        mean_val = static_cast<float>(dsum / static_cast<double>(n));

        //
        // Pass 2 — Accumulate centered sum-of-squared-deviations in fp32.
        // Subtracting the (accurate) mean before squaring removes the
        // catastrophic cancellation that plagues E[x^2] - mean^2.
        //

        __m256 vmean_acc = _mm256_set1_ps(mean_val);
        __m256 vvar = _mm256_setzero_ps();
        i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 vd = _mm256_sub_ps(_mm256_loadu_ps(Input + i), vmean_acc);
            vvar = _mm256_fmadd_ps(vd, vd, vvar);
        }

        // Horizontal reduce.
        __m128 hi = _mm256_extractf128_ps(vvar, 1);
        __m128 lo = _mm256_castps256_ps128(vvar);
        __m128 r = _mm_add_ps(lo, hi);
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_movehdup_ps(r));
        float var_val = _mm_cvtss_f32(r);

        for (; i < n; i++) {
            float d = Input[i] - mean_val;
            var_val += d * d;
        }

        inv_denom = 1.0f / sqrtf(var_val / static_cast<float>(n) + Epsilon);
    }

    //
    // Pass 2: Normalise and write output.
    //

    __m256 vmean = _mm256_set1_ps(mean_val);
    __m256 vinv = _mm256_set1_ps(inv_denom);

    size_t i = 0;
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
