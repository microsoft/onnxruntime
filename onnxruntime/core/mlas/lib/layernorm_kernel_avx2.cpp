/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    layernorm_kernel_avx2.cpp

Abstract:

    This module implements LayerNorm/RMSNorm kernels using x86-64 AVX2+FMA3
    intrinsics. Processes one normalization row at a time, matching the
    MLAS_LAYERNORM_F32_KERNEL signature dispatched from platform.cpp.

    RMSNorm uses a vectorised sum-of-squares accumulation (two-pass:
    reduce then normalise), processing 8 floats per iteration.

    Full LayerNorm uses Welford's online algorithm with 8 parallel
    accumulators (one per AVX2 lane), preserving the numerically stable
    single-pass variance formulation of the scalar baseline. The 8 partial
    accumulators are merged with the standard pairwise combine formula
    after the vector loop.

    A scalar tail handles lengths that are not a multiple of 8.

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
        // RMSNorm: accumulate sum and sum-of-squares. The sum is only
        // needed for the MeanOut optional output (the normalisation itself
        // does not subtract the mean), but the caller may request it.
        //

        __m256 vsum = _mm256_setzero_ps();
        __m256 vsumsq = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 vx = _mm256_loadu_ps(Input + i);
            vsum = _mm256_add_ps(vsum, vx);
            vsumsq = _mm256_fmadd_ps(vx, vx, vsumsq);
        }

        // Horizontal reduce sum.
        __m128 hi_sum = _mm256_extractf128_ps(vsum, 1);
        __m128 lo_sum = _mm256_castps256_ps128(vsum);
        __m128 r_sum = _mm_add_ps(lo_sum, hi_sum);
        r_sum = _mm_add_ps(r_sum, _mm_movehl_ps(r_sum, r_sum));
        r_sum = _mm_add_ss(r_sum, _mm_movehdup_ps(r_sum));
        float sum_val = _mm_cvtss_f32(r_sum);

        // Horizontal reduce sum-of-squares.
        __m128 hi_sq = _mm256_extractf128_ps(vsumsq, 1);
        __m128 lo_sq = _mm256_castps256_ps128(vsumsq);
        __m128 r_sq = _mm_add_ps(lo_sq, hi_sq);
        r_sq = _mm_add_ps(r_sq, _mm_movehl_ps(r_sq, r_sq));
        r_sq = _mm_add_ss(r_sq, _mm_movehdup_ps(r_sq));
        float sumsq_val = _mm_cvtss_f32(r_sq);

        for (; i < n; i++) {
            sum_val += Input[i];
            sumsq_val += Input[i] * Input[i];
        }

        mean_val = sum_val / static_cast<float>(n);
        inv_denom = 1.0f / sqrtf(sumsq_val / static_cast<float>(n) + Epsilon);
    } else {
        //
        // Full LayerNorm: Welford's online algorithm with 8 parallel
        // accumulators, preserving the same numerically stable single-pass
        // formulation used by the scalar baseline in layer_norm_impl.cc.
        //
        // Each AVX2 lane maintains an independent (count, mean, M2) triple.
        // After the vector loop the 8 partial results plus any scalar tail
        // elements are merged with the standard pairwise combine:
        //
        //   n_ab  = n_a + n_b
        //   delta = mean_b - mean_a
        //   mean  = mean_a + delta * n_b / n_ab
        //   M2    = M2_a + M2_b + delta^2 * n_a * n_b / n_ab
        //
        // This avoids the catastrophic cancellation risk of computing
        // Var = E[X^2] - E[X]^2 that a naïve two-pass or sum-of-squares
        // approach has when the mean is large relative to the spread.
        //

        __m256 vmean = _mm256_setzero_ps();
        __m256 vm2 = _mm256_setzero_ps();
        __m256 vcount = _mm256_setzero_ps();
        __m256 vone = _mm256_set1_ps(1.0f);

        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 vx = _mm256_loadu_ps(Input + i);
            vcount = _mm256_add_ps(vcount, vone);
            __m256 delta = _mm256_sub_ps(vx, vmean);
            vmean = _mm256_add_ps(vmean, _mm256_div_ps(delta, vcount));
            __m256 delta2 = _mm256_sub_ps(vx, vmean);
            vm2 = _mm256_fmadd_ps(delta, delta2, vm2);
        }

        // Merge the 8 lanes pairwise. Extract to two 128-bit halves first.
        // We need (count, mean, M2) per lane → merge 8 → 4 → 2 → 1.

        // Helper: pairwise-merge two sets of 4 Welford accumulators packed
        // in __m128 registers into one set of 4 combined accumulators.
        // Then we repeat in scalar until we have a single accumulator.

        // Extract the 8 lanes into arrays for the merge.
        alignas(32) float lane_count[8];
        alignas(32) float lane_mean[8];
        alignas(32) float lane_m2[8];
        _mm256_store_ps(lane_count, vcount);
        _mm256_store_ps(lane_mean, vmean);
        _mm256_store_ps(lane_m2, vm2);

        // Fold the scalar tail elements into lane 0's accumulator.
        float s_count = lane_count[0];
        float s_mean = lane_mean[0];
        float s_m2 = lane_m2[0];

        for (; i < n; i++) {
            s_count += 1.0f;
            float delta = Input[i] - s_mean;
            s_mean += delta / s_count;
            float delta2 = Input[i] - s_mean;
            s_m2 += delta * delta2;
        }

        // Now merge lanes 1..7 into (s_count, s_mean, s_m2).
        for (int lane = 1; lane < 8; lane++) {
            float n_b = lane_count[lane];
            if (n_b == 0.0f) continue;
            float n_ab = s_count + n_b;
            float delta = lane_mean[lane] - s_mean;
            s_mean += delta * n_b / n_ab;
            s_m2 += lane_m2[lane] + delta * delta * s_count * n_b / n_ab;
            s_count = n_ab;
        }

        mean_val = s_mean;
        inv_denom = 1.0f / sqrtf(s_m2 / static_cast<float>(n) + Epsilon);
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
