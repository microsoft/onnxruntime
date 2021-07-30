/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    vector_dot_prod_avx.cpp

Abstract:

    This module implements the vector dot product routine for avx.

--*/

#include "mlasi.h"


MLAS_FORCEINLINE
float
sum_256_horizontal(__m256 sums_256)
{
    __m256 t1_256 = _mm256_hadd_ps(sums_256, sums_256);
    __m256 t2_256 = _mm256_hadd_ps(t1_256, t1_256);
    __m128 t3_128 = _mm256_extractf128_ps(t2_256, 1);
    __m128 t4_128 = _mm256_castps256_ps128(t2_256);
    __m128 t5_128 = _mm_add_ss(t4_128, t3_128);
    return _mm_cvtss_f32(t5_128);
}

void
MLASCALL
MlasVectorDotProductF32KernelAvx(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N
    )
{
  size_t C_idx = 0;

  while (N > 0) {

    size_t cur_M = M;

    const float* cur_A = A;
    float cur_sum = 0.0f;

    while (cur_M >= 32) {
        __m256 a_0 = _mm256_loadu_ps(&cur_A[0]);
        __m256 a_1 = _mm256_loadu_ps(&cur_A[8]);
        __m256 a_2 = _mm256_loadu_ps(&cur_A[16]);
        __m256 a_3 = _mm256_loadu_ps(&cur_A[24]);

        __m256 b_0 = _mm256_loadu_ps(&B[0]);
        __m256 b_1 = _mm256_loadu_ps(&B[8]);
        __m256 b_2 = _mm256_loadu_ps(&B[16]);
        __m256 b_3 = _mm256_loadu_ps(&B[24]);

        __m256 ab_0 = _mm256_mul_ps(a_0, b_0);
        __m256 ab_1 = _mm256_mul_ps(a_1, b_1);
        __m256 ab_2 = _mm256_mul_ps(a_2, b_2);
        __m256 ab_3 = _mm256_mul_ps(a_3, b_3);

        __m256 sums_256 = _mm256_add_ps(ab_0, ab_1);
        sums_256 = _mm256_add_ps(sums_256, ab_2);
        sums_256 = _mm256_add_ps(sums_256, ab_3);

        cur_sum += sum_256_horizontal(sums_256);

        cur_A += 32;
        B += 32;

        cur_M -= 32;
    }

    while (cur_M >= 24) {
        __m256 a_0 = _mm256_loadu_ps(&cur_A[0]);
        __m256 a_1 = _mm256_loadu_ps(&cur_A[8]);
        __m256 a_2 = _mm256_loadu_ps(&cur_A[16]);

        __m256 b_0 = _mm256_loadu_ps(&B[0]);
        __m256 b_1 = _mm256_loadu_ps(&B[8]);
        __m256 b_2 = _mm256_loadu_ps(&B[16]);

        __m256 ab_0 = _mm256_mul_ps(a_0, b_0);
        __m256 ab_1 = _mm256_mul_ps(a_1, b_1);
        __m256 ab_2 = _mm256_mul_ps(a_2, b_2);

        __m256 sums_256 = _mm256_add_ps(ab_0, ab_1);
        sums_256 = _mm256_add_ps(sums_256, ab_2);

        cur_sum += sum_256_horizontal(sums_256);

        cur_A += 24;
        B += 24;

        cur_M -= 24;
    }

    while (cur_M >= 16) {
        __m256 a_0 = _mm256_loadu_ps(&cur_A[0]);
        __m256 a_1 = _mm256_loadu_ps(&cur_A[8]);

        __m256 b_0 = _mm256_loadu_ps(&B[0]);
        __m256 b_1 = _mm256_loadu_ps(&B[8]);

        __m256 ab_0 = _mm256_mul_ps(a_0, b_0);
        __m256 ab_1 = _mm256_mul_ps(a_1, b_1);

        __m256 sums_256 = _mm256_add_ps(ab_0, ab_1);

        cur_sum += sum_256_horizontal(sums_256);

        cur_A += 16;
        B += 16;

        cur_M -= 16;
    }

    while (cur_M >= 8) {
        __m256 a_0 = _mm256_loadu_ps(&cur_A[0]);

        __m256 b_0 = _mm256_loadu_ps(&B[0]);

        __m256 ab_0 = _mm256_mul_ps(a_0, b_0);

        cur_sum += sum_256_horizontal(ab_0);

        cur_A += 8;
        B += 8;

        cur_M -= 8;
    }

    while (cur_M > 0) {
        cur_sum += cur_A[0] * B[0];

        cur_A += 1;
        B += 1;

        cur_M -= 1;
    }

    C[C_idx] = cur_sum;
    C_idx++;

    N -= 1;
  }
}
