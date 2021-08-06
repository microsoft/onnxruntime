/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    vector_dot_prod_avx512f.cpp

Abstract:

    This module implements the vector dot product routine for avx512f.

--*/

#include "mlasi.h"

void MLASCALL
MlasVectorDotProductF32KernelAvx512F(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N
    )
{
    size_t B_Row_Shift = N;

    while (N > 0) {

        while (N >= 64) {
            __m512 sums_0 = _mm512_setzero_ps();
            __m512 sums_1 = _mm512_setzero_ps();
            __m512 sums_2 = _mm512_setzero_ps();
            __m512 sums_3 = _mm512_setzero_ps();

            const float* cur_A = A;
            const float* cur_B = B;

            size_t cur_M = M;

            while (cur_M > 0) {
                __m512 a = _mm512_set1_ps(cur_A[0]);

                __m512 b_0 = _mm512_loadu_ps(&cur_B[0]);
                __m512 b_1 = _mm512_loadu_ps(&cur_B[16]);
                __m512 b_2 = _mm512_loadu_ps(&cur_B[32]);
                __m512 b_3 = _mm512_loadu_ps(&cur_B[48]);

                __m512 ab_0 = _mm512_mul_ps(a, b_0);
                __m512 ab_1 = _mm512_mul_ps(a, b_1);
                __m512 ab_2 = _mm512_mul_ps(a, b_2);
                __m512 ab_3 = _mm512_mul_ps(a, b_3);

                sums_0 = _mm512_add_ps(sums_0, ab_0);
                sums_1 = _mm512_add_ps(sums_1, ab_1);
                sums_2 = _mm512_add_ps(sums_2, ab_2);
                sums_3 = _mm512_add_ps(sums_3, ab_3);

                cur_A += 1;
                cur_B += B_Row_Shift;

                cur_M -= 1;
            }

            _mm512_storeu_ps(C, sums_0);
            C += 16;

            _mm512_storeu_ps(C, sums_1);
            C += 16;

            _mm512_storeu_ps(C, sums_2);
            C += 16;

            _mm512_storeu_ps(C, sums_3);
            C += 16;

            B += 64;

            N -= 64;
        }

        while (N >= 48) {
            __m512 sums_0 = _mm512_setzero_ps();
            __m512 sums_1 = _mm512_setzero_ps();
            __m512 sums_2 = _mm512_setzero_ps();

            const float* cur_A = A;
            const float* cur_B = B;

            size_t cur_M = M;

            while (cur_M > 0) {
                __m512 a = _mm512_set1_ps(cur_A[0]);

                __m512 b_0 = _mm512_loadu_ps(&cur_B[0]);
                __m512 b_1 = _mm512_loadu_ps(&cur_B[16]);
                __m512 b_2 = _mm512_loadu_ps(&cur_B[32]);

                __m512 ab_0 = _mm512_mul_ps(a, b_0);
                __m512 ab_1 = _mm512_mul_ps(a, b_1);
                __m512 ab_2 = _mm512_mul_ps(a, b_2);

                sums_0 = _mm512_add_ps(sums_0, ab_0);
                sums_1 = _mm512_add_ps(sums_1, ab_1);
                sums_2 = _mm512_add_ps(sums_2, ab_2);

                cur_A += 1;
                cur_B += B_Row_Shift;

                cur_M -= 1;
            }

            _mm512_storeu_ps(C, sums_0);
            C += 16;

            _mm512_storeu_ps(C, sums_1);
            C += 16;

            _mm512_storeu_ps(C, sums_2);
            C += 16;

            B += 48;

            N -= 48;
        }

        while (N >= 32) {
            __m512 sums_0 = _mm512_setzero_ps();
            __m512 sums_1 = _mm512_setzero_ps();

            const float* cur_A = A;
            const float* cur_B = B;

            size_t cur_M = M;

            while (cur_M > 0) {
                __m512 a = _mm512_set1_ps(cur_A[0]);

                __m512 b_0 = _mm512_loadu_ps(&cur_B[0]);
                __m512 b_1 = _mm512_loadu_ps(&cur_B[16]);

                __m512 ab_0 = _mm512_mul_ps(a, b_0);
                __m512 ab_1 = _mm512_mul_ps(a, b_1);

                sums_0 = _mm512_add_ps(sums_0, ab_0);
                sums_1 = _mm512_add_ps(sums_1, ab_1);

                cur_A += 1;
                cur_B += B_Row_Shift;

                cur_M -= 1;
            }

            _mm512_storeu_ps(C, sums_0);
            C += 16;

            _mm512_storeu_ps(C, sums_1);
            C += 16;

            B += 32;

            N -= 32;
        }

        while (N >= 16) {
            __m512 sums_0 = _mm512_setzero_ps();

            const float* cur_A = A;
            const float* cur_B = B;

            size_t cur_M = M;

            while (cur_M > 0) {
                __m512 a = _mm512_set1_ps(cur_A[0]);

                __m512 b = _mm512_loadu_ps(&cur_B[0]);

                __m512 ab = _mm512_mul_ps(a, b);

                sums_0 = _mm512_add_ps(sums_0, ab);

                cur_A += 1;
                cur_B += B_Row_Shift;

                cur_M -= 1;
            }

            _mm512_storeu_ps(C, sums_0);
            C += 16;

            B += 16;

            N -= 16;
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

