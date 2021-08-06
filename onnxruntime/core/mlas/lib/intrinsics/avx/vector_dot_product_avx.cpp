/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    vector_dot_prod_avx.cpp

Abstract:

    This module implements the vector dot product routine for avx.

--*/

#include "mlasi.h"

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
    size_t B_Row_Shift = N;

    while (N > 0) {

        while (N >= 32) {
            __m256 sums_0 = _mm256_setzero_ps();
            __m256 sums_1 = _mm256_setzero_ps();
            __m256 sums_2 = _mm256_setzero_ps();
            __m256 sums_3 = _mm256_setzero_ps();

            const float* cur_A = A;
            const float* cur_B = B;

            size_t cur_M = M;

            while (cur_M > 0) {
                __m256 a = _mm256_set1_ps(cur_A[0]);

                __m256 b_0 = _mm256_loadu_ps(&cur_B[0]);
                __m256 b_1 = _mm256_loadu_ps(&cur_B[8]);
                __m256 b_2 = _mm256_loadu_ps(&cur_B[16]);
                __m256 b_3 = _mm256_loadu_ps(&cur_B[24]);

                __m256 ab_0 = _mm256_mul_ps(a, b_0);
                __m256 ab_1 = _mm256_mul_ps(a, b_1);
                __m256 ab_2 = _mm256_mul_ps(a, b_2);
                __m256 ab_3 = _mm256_mul_ps(a, b_3);

                sums_0 = _mm256_add_ps(sums_0, ab_0);
                sums_1 = _mm256_add_ps(sums_1, ab_1);
                sums_2 = _mm256_add_ps(sums_2, ab_2);
                sums_3 = _mm256_add_ps(sums_3, ab_3);

                cur_A += 1;
                cur_B += B_Row_Shift;

                cur_M -= 1;
            }

            _mm256_storeu_ps(C, sums_0);
            C += 8;

            _mm256_storeu_ps(C, sums_1);
            C += 8;

            _mm256_storeu_ps(C, sums_2);
            C += 8;

            _mm256_storeu_ps(C, sums_3);
            C += 8;

            B += 32;

            N -= 32;
        }

        while (N >= 24) {
            __m256 sums_0 = _mm256_setzero_ps();
            __m256 sums_1 = _mm256_setzero_ps();
            __m256 sums_2 = _mm256_setzero_ps();

            const float* cur_A = A;
            const float* cur_B = B;

            size_t cur_M = M;

            while (cur_M > 0) {
                __m256 a = _mm256_set1_ps(cur_A[0]);

                __m256 b_0 = _mm256_loadu_ps(&cur_B[0]);
                __m256 b_1 = _mm256_loadu_ps(&cur_B[8]);
                __m256 b_2 = _mm256_loadu_ps(&cur_B[16]);

                __m256 ab_0 = _mm256_mul_ps(a, b_0);
                __m256 ab_1 = _mm256_mul_ps(a, b_1);
                __m256 ab_2 = _mm256_mul_ps(a, b_2);

                sums_0 = _mm256_add_ps(sums_0, ab_0);
                sums_1 = _mm256_add_ps(sums_1, ab_1);
                sums_2 = _mm256_add_ps(sums_2, ab_2);

                cur_A += 1;
                cur_B += B_Row_Shift;

                cur_M -= 1;
            }

            _mm256_storeu_ps(C, sums_0);
            C += 8;

            _mm256_storeu_ps(C, sums_1);
            C += 8;

            _mm256_storeu_ps(C, sums_2);
            C += 8;

            B += 24;

            N -= 24;
        }

        while (N >= 16) {
            __m256 sums_0 = _mm256_setzero_ps();
            __m256 sums_1 = _mm256_setzero_ps();

            const float* cur_A = A;
            const float* cur_B = B;

            size_t cur_M = M;

            while (cur_M > 0) {
                __m256 a = _mm256_set1_ps(cur_A[0]);

                __m256 b_0 = _mm256_loadu_ps(&cur_B[0]);
                __m256 b_1 = _mm256_loadu_ps(&cur_B[8]);

                __m256 ab_0 = _mm256_mul_ps(a, b_0);
                __m256 ab_1 = _mm256_mul_ps(a, b_1);

                sums_0 = _mm256_add_ps(sums_0, ab_0);
                sums_1 = _mm256_add_ps(sums_1, ab_1);

                cur_A += 1;
                cur_B += B_Row_Shift;

                cur_M -= 1;
            }

            _mm256_storeu_ps(C, sums_0);
            C += 8;

            _mm256_storeu_ps(C, sums_1);
            C += 8;

            B += 16;

            N -= 16;
        }

        while (N >= 8) {
            __m256 sums_0 = _mm256_setzero_ps();

            const float* cur_A = A;
            const float* cur_B = B;

            size_t cur_M = M;

            while (cur_M > 0) {
                __m256 a = _mm256_set1_ps(cur_A[0]);

                __m256 b = _mm256_loadu_ps(&cur_B[0]);

                __m256 ab_0 = _mm256_mul_ps(a, b);

                sums_0 = _mm256_add_ps(sums_0, ab_0);

                cur_A += 1;

                cur_B += B_Row_Shift;

                cur_M -= 1;
            }

            _mm256_storeu_ps(C, sums_0);
            C += 8;

            B += 8;

            N -= 8;
        }

        while (N > 0) {
            float sum = 0;

            const float* cur_A = A;
            const float* cur_B = B;

            size_t cur_M = M;

            while (cur_M > 0) {
                sum += cur_A[0] * cur_B[0];

                cur_A += 1;
                cur_B += B_Row_Shift;

                cur_M -= 1;
            }

            C[0] = sum;

            C += 1;
            B += 1;

            N -= 1;
        }
    }
}
