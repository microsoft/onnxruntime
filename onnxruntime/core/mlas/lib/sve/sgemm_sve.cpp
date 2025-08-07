/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

    sgemm_sve.cpp

Abstract:

    This module contains the implementation of SVE-based sgemm operations
--*/

#ifdef __ARM_FEATURE_SVE
#include "mlasi_sve.h"

template <bool ZeroMode, bool Alpha1>
inline void
processrows_8(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl
)
{
    size_t k_step = 8 * vl;
    svfloat32_t zero_vec = MlasSveBroadcastFloat32(0.f);
    size_t col = 0;
    for (; col + 2 * vl <= n; col += 2 * vl) {
        svbool_t pg0 = svwhilelt_b32(col, n);
        svbool_t pg1 = svwhilelt_b32(col + vl, n);

        // Accumulators (8 rows × 2 vectors)
        svfloat32_t acc0_0 = zero_vec, acc0_1 = zero_vec;
        svfloat32_t acc1_0 = zero_vec, acc1_1 = zero_vec;
        svfloat32_t acc2_0 = zero_vec, acc2_1 = zero_vec;
        svfloat32_t acc3_0 = zero_vec, acc3_1 = zero_vec;
        svfloat32_t acc4_0 = zero_vec, acc4_1 = zero_vec;
        svfloat32_t acc5_0 = zero_vec, acc5_1 = zero_vec;
        svfloat32_t acc6_0 = zero_vec, acc6_1 = zero_vec;
        svfloat32_t acc7_0 = zero_vec, acc7_1 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b0_ptr = b + p * PACKED_B_BLOCK_WIDTH + col;
                const float* b1_ptr = b0_ptr + vl;
                svfloat32_t b0 = MlasSveLoadFloat32(pg0, b0_ptr);
                svfloat32_t b1 = MlasSveLoadFloat32(pg1, b1_ptr);
                float a0 = a[0 * lda + p];
                float a1 = a[1 * lda + p];
                float a2 = a[2 * lda + p];
                float a3 = a[3 * lda + p];
                float a4 = a[4 * lda + p];
                float a5 = a[5 * lda + p];
                float a6 = a[6 * lda + p];
                float a7 = a[7 * lda + p];

                if constexpr (!Alpha1) {
                    a0 *= alpha;
                    a1 *= alpha;
                    a2 *= alpha;
                    a3 *= alpha;
                    a4 *= alpha;
                    a5 *= alpha;
                    a6 *= alpha;
                    a7 *= alpha;
                }

                acc0_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a0), acc0_0);
                acc0_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a0), acc0_1);
                acc1_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a1), acc1_0);
                acc1_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a1), acc1_1);
                acc2_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a2), acc2_0);
                acc2_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a2), acc2_1);
                acc3_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a3), acc3_0);
                acc3_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a3), acc3_1);
                acc4_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a4), acc4_0);
                acc4_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a4), acc4_1);
                acc5_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a5), acc5_0);
                acc5_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a5), acc5_1);
                acc6_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a6), acc6_0);
                acc6_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a6), acc6_1);
                acc7_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a7), acc7_0);
                acc7_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a7), acc7_1);
            }
        }

        float* out0 = res + 0 * ldc + col;
        float* out1 = res + 1 * ldc + col;
        float* out2 = res + 2 * ldc + col;
        float* out3 = res + 3 * ldc + col;
        float* out4 = res + 4 * ldc + col;
        float* out5 = res + 5 * ldc + col;
        float* out6 = res + 6 * ldc + col;
        float* out7 = res + 7 * ldc + col;

        if constexpr (!ZeroMode) {
            acc0_0 = MlasSveAddFloat32(pg0, acc0_0, svld1(pg0, out0));
            acc1_0 = MlasSveAddFloat32(pg0, acc1_0, svld1(pg0, out1));
            acc2_0 = MlasSveAddFloat32(pg0, acc2_0, svld1(pg0, out2));
            acc3_0 = MlasSveAddFloat32(pg0, acc3_0, svld1(pg0, out3));
            acc4_0 = MlasSveAddFloat32(pg0, acc4_0, svld1(pg0, out4));
            acc5_0 = MlasSveAddFloat32(pg0, acc5_0, svld1(pg0, out5));
            acc6_0 = MlasSveAddFloat32(pg0, acc6_0, svld1(pg0, out6));
            acc7_0 = MlasSveAddFloat32(pg0, acc7_0, svld1(pg0, out7));

            acc0_1 = MlasSveAddFloat32(pg1, acc0_1, svld1(pg1, out0 + vl));
            acc1_1 = MlasSveAddFloat32(pg1, acc1_1, svld1(pg1, out1 + vl));
            acc2_1 = MlasSveAddFloat32(pg1, acc2_1, svld1(pg1, out2 + vl));
            acc3_1 = MlasSveAddFloat32(pg1, acc3_1, svld1(pg1, out3 + vl));
            acc4_1 = MlasSveAddFloat32(pg1, acc4_1, svld1(pg1, out4 + vl));
            acc5_1 = MlasSveAddFloat32(pg1, acc5_1, svld1(pg1, out5 + vl));
            acc6_1 = MlasSveAddFloat32(pg1, acc6_1, svld1(pg1, out6 + vl));
            acc7_1 = MlasSveAddFloat32(pg1, acc7_1, svld1(pg1, out7 + vl));
        }

        MlasSveStoreFloat32(pg0, out0, acc0_0);
        MlasSveStoreFloat32(pg0, out1, acc1_0);
        MlasSveStoreFloat32(pg0, out2, acc2_0);
        MlasSveStoreFloat32(pg0, out3, acc3_0);
        MlasSveStoreFloat32(pg0, out4, acc4_0);
        MlasSveStoreFloat32(pg0, out5, acc5_0);
        MlasSveStoreFloat32(pg0, out6, acc6_0);
        MlasSveStoreFloat32(pg0, out7, acc7_0);

        MlasSveStoreFloat32(pg1, out0 + vl, acc0_1);
        MlasSveStoreFloat32(pg1, out1 + vl, acc1_1);
        MlasSveStoreFloat32(pg1, out2 + vl, acc2_1);
        MlasSveStoreFloat32(pg1, out3 + vl, acc3_1);
        MlasSveStoreFloat32(pg1, out4 + vl, acc4_1);
        MlasSveStoreFloat32(pg1, out5 + vl, acc5_1);
        MlasSveStoreFloat32(pg1, out6 + vl, acc6_1);
        MlasSveStoreFloat32(pg1, out7 + vl, acc7_1);
    }

    for (; col < n; col += vl) {
        svbool_t pg = svwhilelt_b32(col, n);

        svfloat32_t acc0 = zero_vec, acc1 = zero_vec, acc2 = zero_vec, acc3 = zero_vec;
        svfloat32_t acc4 = zero_vec, acc5 = zero_vec, acc6 = zero_vec, acc7 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b_ptr = b + p * PACKED_B_BLOCK_WIDTH + col;
                svfloat32_t b0 = MlasSveLoadFloat32(pg, b_ptr);

                float a0 = a[0 * lda + p];
                float a1 = a[1 * lda + p];
                float a2 = a[2 * lda + p];
                float a3 = a[3 * lda + p];
                float a4 = a[4 * lda + p];
                float a5 = a[5 * lda + p];
                float a6 = a[6 * lda + p];
                float a7 = a[7 * lda + p];

                if constexpr (!Alpha1) {
                    a0 *= alpha;
                    a1 *= alpha;
                    a2 *= alpha;
                    a3 *= alpha;
                    a4 *= alpha;
                    a5 *= alpha;
                    a6 *= alpha;
                    a7 *= alpha;
                }
                acc0 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a0), acc0);
                acc1 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a1), acc1);
                acc2 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a2), acc2);
                acc3 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a3), acc3);
                acc4 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a4), acc4);
                acc5 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a5), acc5);
                acc6 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a6), acc6);
                acc7 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a7), acc7);
            }
        }

        float* out0 = res + 0 * ldc + col;
        float* out1 = res + 1 * ldc + col;
        float* out2 = res + 2 * ldc + col;
        float* out3 = res + 3 * ldc + col;
        float* out4 = res + 4 * ldc + col;
        float* out5 = res + 5 * ldc + col;
        float* out6 = res + 6 * ldc + col;
        float* out7 = res + 7 * ldc + col;

        if constexpr (!ZeroMode) {
            acc0 = MlasSveAddFloat32(pg, acc0, svld1(pg, out0));
            acc1 = MlasSveAddFloat32(pg, acc1, svld1(pg, out1));
            acc2 = MlasSveAddFloat32(pg, acc2, svld1(pg, out2));
            acc3 = MlasSveAddFloat32(pg, acc3, svld1(pg, out3));
            acc4 = MlasSveAddFloat32(pg, acc4, svld1(pg, out4));
            acc5 = MlasSveAddFloat32(pg, acc5, svld1(pg, out5));
            acc6 = MlasSveAddFloat32(pg, acc6, svld1(pg, out6));
            acc7 = MlasSveAddFloat32(pg, acc7, svld1(pg, out7));
        }

        MlasSveStoreFloat32(pg, out0, acc0);
        MlasSveStoreFloat32(pg, out1, acc1);
        MlasSveStoreFloat32(pg, out2, acc2);
        MlasSveStoreFloat32(pg, out3, acc3);
        MlasSveStoreFloat32(pg, out4, acc4);
        MlasSveStoreFloat32(pg, out5, acc5);
        MlasSveStoreFloat32(pg, out6, acc6);
        MlasSveStoreFloat32(pg, out7, acc7);
    }
}

template <bool ZeroMode, bool Alpha1>
inline void
processrows_6(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl
)
{
    size_t k_step = 8 * vl;
    svfloat32_t zero_vec = MlasSveBroadcastFloat32(0.f);
    size_t col = 0;
    for (; col + 2 * vl <= n; col += 2 * vl) {
        svbool_t pg0 = svwhilelt_b32(col, n);
        svbool_t pg1 = svwhilelt_b32(col + vl, n);

        float* out0_0 = res + 0 * ldc + col;
        float* out0_1 = res + 0 * ldc + col + vl;

        float* out1_0 = res + 1 * ldc + col;
        float* out1_1 = res + 1 * ldc + col + vl;

        float* out2_0 = res + 2 * ldc + col;
        float* out2_1 = res + 2 * ldc + col + vl;

        float* out3_0 = res + 3 * ldc + col;
        float* out3_1 = res + 3 * ldc + col + vl;

        float* out4_0 = res + 4 * ldc + col;
        float* out4_1 = res + 4 * ldc + col + vl;

        float* out5_0 = res + 5 * ldc + col;
        float* out5_1 = res + 5 * ldc + col + vl;

        svfloat32_t acc0_0 = zero_vec, acc0_1 = zero_vec;
        svfloat32_t acc1_0 = zero_vec, acc1_1 = zero_vec;
        svfloat32_t acc2_0 = zero_vec, acc2_1 = zero_vec;
        svfloat32_t acc3_0 = zero_vec, acc3_1 = zero_vec;
        svfloat32_t acc4_0 = zero_vec, acc4_1 = zero_vec;
        svfloat32_t acc5_0 = zero_vec, acc5_1 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b0_ptr = b + p * PACKED_B_BLOCK_WIDTH + col;
                const float* b1_ptr = b0_ptr + vl;

                svfloat32_t b0 = MlasSveLoadFloat32(pg0, b0_ptr);
                svfloat32_t b1 = MlasSveLoadFloat32(pg1, b1_ptr);

                float a0 = a[0 * lda + p];
                float a1 = a[1 * lda + p];
                float a2 = a[2 * lda + p];
                float a3 = a[3 * lda + p];
                float a4 = a[4 * lda + p];
                float a5 = a[5 * lda + p];

                if constexpr (!Alpha1) {
                    a0 *= alpha;
                    a1 *= alpha;
                    a2 *= alpha;
                    a3 *= alpha;
                    a4 *= alpha;
                    a5 *= alpha;
                }
                acc0_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a0), acc0_0);
                acc0_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a0), acc0_1);
                acc1_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a1), acc1_0);
                acc1_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a1), acc1_1);
                acc2_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a2), acc2_0);
                acc2_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a2), acc2_1);
                acc3_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a3), acc3_0);
                acc3_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a3), acc3_1);
                acc4_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a4), acc4_0);
                acc4_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a4), acc4_1);
                acc5_0 = MlasSveMultiplyAddFloat32(pg0, b0, MlasSveBroadcastFloat32(a5), acc5_0);
                acc5_1 = MlasSveMultiplyAddFloat32(pg1, b1, MlasSveBroadcastFloat32(a5), acc5_1);
            }
        }

        if constexpr (!ZeroMode) {
            acc0_0 = MlasSveAddFloat32(pg0, acc0_0, svld1(pg0, out0_0));
            acc0_1 = MlasSveAddFloat32(pg1, acc0_1, svld1(pg1, out0_1));
            acc1_0 = MlasSveAddFloat32(pg0, acc1_0, svld1(pg0, out1_0));
            acc1_1 = MlasSveAddFloat32(pg1, acc1_1, svld1(pg1, out1_1));
            acc2_0 = MlasSveAddFloat32(pg0, acc2_0, svld1(pg0, out2_0));
            acc2_1 = MlasSveAddFloat32(pg1, acc2_1, svld1(pg1, out2_1));
            acc3_0 = MlasSveAddFloat32(pg0, acc3_0, svld1(pg0, out3_0));
            acc3_1 = MlasSveAddFloat32(pg1, acc3_1, svld1(pg1, out3_1));
            acc4_0 = MlasSveAddFloat32(pg0, acc4_0, svld1(pg0, out4_0));
            acc4_1 = MlasSveAddFloat32(pg1, acc4_1, svld1(pg1, out4_1));
            acc5_0 = MlasSveAddFloat32(pg0, acc5_0, svld1(pg0, out5_0));
            acc5_1 = MlasSveAddFloat32(pg1, acc5_1, svld1(pg1, out5_1));
        }

        MlasSveStoreFloat32(pg0, out0_0, acc0_0);
        MlasSveStoreFloat32(pg1, out0_1, acc0_1);
        MlasSveStoreFloat32(pg0, out1_0, acc1_0);
        MlasSveStoreFloat32(pg1, out1_1, acc1_1);
        MlasSveStoreFloat32(pg0, out2_0, acc2_0);
        MlasSveStoreFloat32(pg1, out2_1, acc2_1);
        MlasSveStoreFloat32(pg0, out3_0, acc3_0);
        MlasSveStoreFloat32(pg1, out3_1, acc3_1);
        MlasSveStoreFloat32(pg0, out4_0, acc4_0);
        MlasSveStoreFloat32(pg1, out4_1, acc4_1);
        MlasSveStoreFloat32(pg0, out5_0, acc5_0);
        MlasSveStoreFloat32(pg1, out5_1, acc5_1);
    }

    for (; col < n; col += vl) {
        svbool_t pg = svwhilelt_b32(col, n);

        float* out0 = res + 0 * ldc + col;
        float* out1 = res + 1 * ldc + col;
        float* out2 = res + 2 * ldc + col;
        float* out3 = res + 3 * ldc + col;
        float* out4 = res + 4 * ldc + col;
        float* out5 = res + 5 * ldc + col;

        svfloat32_t acc0 = zero_vec;
        svfloat32_t acc1 = zero_vec;
        svfloat32_t acc2 = zero_vec;
        svfloat32_t acc3 = zero_vec;
        svfloat32_t acc4 = zero_vec;
        svfloat32_t acc5 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);

            for (size_t p = k_block; p < k_max; ++p) {
                const float* b_ptr = b + p * PACKED_B_BLOCK_WIDTH + col;
                svfloat32_t b0 = MlasSveLoadFloat32(pg, b_ptr);
                float a0 = a[0 * lda + p];
                float a1 = a[1 * lda + p];
                float a2 = a[2 * lda + p];
                float a3 = a[3 * lda + p];
                float a4 = a[4 * lda + p];
                float a5 = a[5 * lda + p];

                if constexpr (!Alpha1) {
                    a0 *= alpha;
                    a1 *= alpha;
                    a2 *= alpha;
                    a3 *= alpha;
                    a4 *= alpha;
                    a5 *= alpha;
                }
                acc0 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a0), acc0);
                acc1 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a1), acc1);
                acc2 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a2), acc2);
                acc3 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a3), acc3);
                acc4 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a4), acc4);
                acc5 = MlasSveMultiplyAddFloat32(pg, b0, MlasSveBroadcastFloat32(a5), acc5);
            }
        }

        if constexpr (!ZeroMode) {
            acc0 = MlasSveAddFloat32(pg, acc0, svld1(pg, out0));
            acc1 = MlasSveAddFloat32(pg, acc1, svld1(pg, out1));
            acc2 = MlasSveAddFloat32(pg, acc2, svld1(pg, out2));
            acc3 = MlasSveAddFloat32(pg, acc3, svld1(pg, out3));
            acc4 = MlasSveAddFloat32(pg, acc4, svld1(pg, out4));
            acc5 = MlasSveAddFloat32(pg, acc5, svld1(pg, out5));
        }

        MlasSveStoreFloat32(pg, out0, acc0);
        MlasSveStoreFloat32(pg, out1, acc1);
        MlasSveStoreFloat32(pg, out2, acc2);
        MlasSveStoreFloat32(pg, out3, acc3);
        MlasSveStoreFloat32(pg, out4, acc4);
        MlasSveStoreFloat32(pg, out5, acc5);
    }
}

template <bool ZeroMode, bool Alpha1>
inline void
processrows_4(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl
)
{
    size_t k_step = 8 * vl;

    svfloat32_t zero_vec = MlasSveBroadcastFloat32(0.f);

    size_t col = 0;
    for (; col + 2 * vl <= n; col += 2 * vl) {
        svbool_t pg0 = svwhilelt_b32(col, n);
        svbool_t pg1 = svwhilelt_b32(col + vl, n);

        float* out0_0 = res + 0 * ldc + col;
        float* out0_1 = res + 0 * ldc + col + vl;
        float* out1_0 = res + 1 * ldc + col;
        float* out1_1 = res + 1 * ldc + col + vl;
        float* out2_0 = res + 2 * ldc + col;
        float* out2_1 = res + 2 * ldc + col + vl;
        float* out3_0 = res + 3 * ldc + col;
        float* out3_1 = res + 3 * ldc + col + vl;

        svfloat32_t acc0_0 = zero_vec, acc0_1 = zero_vec;
        svfloat32_t acc1_0 = zero_vec, acc1_1 = zero_vec;
        svfloat32_t acc2_0 = zero_vec, acc2_1 = zero_vec;
        svfloat32_t acc3_0 = zero_vec, acc3_1 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);

            for (size_t p = k_block; p < k_max; ++p) {
                const float* b0_ptr = b + p * PACKED_B_BLOCK_WIDTH + col;
                const float* b1_ptr = b0_ptr + vl;
                svfloat32_t b0 = MlasSveLoadFloat32(pg0, b0_ptr);
                svfloat32_t b1 = MlasSveLoadFloat32(pg1, b1_ptr);

                float a0 = a[0 * lda + p];
                float a1 = a[1 * lda + p];
                float a2 = a[2 * lda + p];
                float a3 = a[3 * lda + p];

                if constexpr (!Alpha1) {
                    a0 *= alpha;
                    a1 *= alpha;
                    a2 *= alpha;
                    a3 *= alpha;
                }

                svfloat32_t va0 = MlasSveBroadcastFloat32(a0);
                svfloat32_t va1 = MlasSveBroadcastFloat32(a1);
                svfloat32_t va2 = MlasSveBroadcastFloat32(a2);
                svfloat32_t va3 = MlasSveBroadcastFloat32(a3);

                acc0_0 = MlasSveMultiplyAddFloat32(pg0, b0, va0, acc0_0);
                acc0_1 = MlasSveMultiplyAddFloat32(pg1, b1, va0, acc0_1);
                acc1_0 = MlasSveMultiplyAddFloat32(pg0, b0, va1, acc1_0);
                acc1_1 = MlasSveMultiplyAddFloat32(pg1, b1, va1, acc1_1);
                acc2_0 = MlasSveMultiplyAddFloat32(pg0, b0, va2, acc2_0);
                acc2_1 = MlasSveMultiplyAddFloat32(pg1, b1, va2, acc2_1);
                acc3_0 = MlasSveMultiplyAddFloat32(pg0, b0, va3, acc3_0);
                acc3_1 = MlasSveMultiplyAddFloat32(pg1, b1, va3, acc3_1);
            }
        }

        if constexpr (!ZeroMode) {
            acc0_0 = MlasSveAddFloat32(pg0, acc0_0, svld1(pg0, out0_0));
            acc0_1 = MlasSveAddFloat32(pg1, acc0_1, svld1(pg1, out0_1));
            acc1_0 = MlasSveAddFloat32(pg0, acc1_0, svld1(pg0, out1_0));
            acc1_1 = MlasSveAddFloat32(pg1, acc1_1, svld1(pg1, out1_1));
            acc2_0 = MlasSveAddFloat32(pg0, acc2_0, svld1(pg0, out2_0));
            acc2_1 = MlasSveAddFloat32(pg1, acc2_1, svld1(pg1, out2_1));
            acc3_0 = MlasSveAddFloat32(pg0, acc3_0, svld1(pg0, out3_0));
            acc3_1 = MlasSveAddFloat32(pg1, acc3_1, svld1(pg1, out3_1));
        }

        MlasSveStoreFloat32(pg0, out0_0, acc0_0);
        MlasSveStoreFloat32(pg1, out0_1, acc0_1);
        MlasSveStoreFloat32(pg0, out1_0, acc1_0);
        MlasSveStoreFloat32(pg1, out1_1, acc1_1);
        MlasSveStoreFloat32(pg0, out2_0, acc2_0);
        MlasSveStoreFloat32(pg1, out2_1, acc2_1);
        MlasSveStoreFloat32(pg0, out3_0, acc3_0);
        MlasSveStoreFloat32(pg1, out3_1, acc3_1);
    }

    for (; col < n; col += vl) {
        svbool_t pg = svwhilelt_b32(col, n);
        float* out0 = res + 0 * ldc + col;
        float* out1 = res + 1 * ldc + col;
        float* out2 = res + 2 * ldc + col;
        float* out3 = res + 3 * ldc + col;
        svfloat32_t acc0 = zero_vec;
        svfloat32_t acc1 = zero_vec;
        svfloat32_t acc2 = zero_vec;
        svfloat32_t acc3 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b_ptr = b + p * PACKED_B_BLOCK_WIDTH + col;
                svfloat32_t b0 = MlasSveLoadFloat32(pg, b_ptr);
                float a0 = a[0 * lda + p];
                float a1 = a[1 * lda + p];
                float a2 = a[2 * lda + p];
                float a3 = a[3 * lda + p];
                if constexpr (!Alpha1) {
                    a0 *= alpha;
                    a1 *= alpha;
                    a2 *= alpha;
                    a3 *= alpha;
                }
                svfloat32_t va0 = MlasSveBroadcastFloat32(a0);
                svfloat32_t va1 = MlasSveBroadcastFloat32(a1);
                svfloat32_t va2 = MlasSveBroadcastFloat32(a2);
                svfloat32_t va3 = MlasSveBroadcastFloat32(a3);
                acc0 = MlasSveMultiplyAddFloat32(pg, b0, va0, acc0);
                acc1 = MlasSveMultiplyAddFloat32(pg, b0, va1, acc1);
                acc2 = MlasSveMultiplyAddFloat32(pg, b0, va2, acc2);
                acc3 = MlasSveMultiplyAddFloat32(pg, b0, va3, acc3);
            }
        }

        if constexpr (!ZeroMode) {
            acc0 = MlasSveAddFloat32(pg, acc0, svld1(pg, out0));
            acc1 = MlasSveAddFloat32(pg, acc1, svld1(pg, out1));
            acc2 = MlasSveAddFloat32(pg, acc2, svld1(pg, out2));
            acc3 = MlasSveAddFloat32(pg, acc3, svld1(pg, out3));
        }

        MlasSveStoreFloat32(pg, out0, acc0);
        MlasSveStoreFloat32(pg, out1, acc1);
        MlasSveStoreFloat32(pg, out2, acc2);
        MlasSveStoreFloat32(pg, out3, acc3);
    }
}

template <bool ZeroMode, bool Alpha1>
inline void
processrows_2(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl
)
{
    size_t k_step = 8 * vl;
    svfloat32_t zero_vec = MlasSveBroadcastFloat32(0.f);
    size_t col = 0;
    for (; col + 2 * vl <= n; col += 2 * vl) {
        svbool_t pg0 = svwhilelt_b32(col, n);
        svbool_t pg1 = svwhilelt_b32(col + vl, n);

        float* out0_0 = res + 0 * ldc + col;
        float* out0_1 = res + 0 * ldc + col + vl;
        float* out1_0 = res + 1 * ldc + col;
        float* out1_1 = res + 1 * ldc + col + vl;

        svfloat32_t acc0_0 = zero_vec, acc0_1 = zero_vec;
        svfloat32_t acc1_0 = zero_vec, acc1_1 = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b0_ptr = b + p * PACKED_B_BLOCK_WIDTH + col;
                const float* b1_ptr = b0_ptr + vl;
                svfloat32_t b0 = MlasSveLoadFloat32(pg0, b0_ptr);
                svfloat32_t b1 = MlasSveLoadFloat32(pg1, b1_ptr);
                float a0 = a[0 * lda + p];
                float a1 = a[1 * lda + p];
                if constexpr (!Alpha1) {
                    a0 *= alpha;
                    a1 *= alpha;
                }
                svfloat32_t va0 = MlasSveBroadcastFloat32(a0);
                svfloat32_t va1 = MlasSveBroadcastFloat32(a1);
                acc0_0 = MlasSveMultiplyAddFloat32(pg0, b0, va0, acc0_0);
                acc0_1 = MlasSveMultiplyAddFloat32(pg1, b1, va0, acc0_1);
                acc1_0 = MlasSveMultiplyAddFloat32(pg0, b0, va1, acc1_0);
                acc1_1 = MlasSveMultiplyAddFloat32(pg1, b1, va1, acc1_1);
            }
        }
        if constexpr (!ZeroMode) {
            acc0_0 = MlasSveAddFloat32(pg0, acc0_0, svld1(pg0, out0_0));
            acc0_1 = MlasSveAddFloat32(pg1, acc0_1, svld1(pg1, out0_1));
            acc1_0 = MlasSveAddFloat32(pg0, acc1_0, svld1(pg0, out1_0));
            acc1_1 = MlasSveAddFloat32(pg1, acc1_1, svld1(pg1, out1_1));
        }
        MlasSveStoreFloat32(pg0, out0_0, acc0_0);
        MlasSveStoreFloat32(pg1, out0_1, acc0_1);
        MlasSveStoreFloat32(pg0, out1_0, acc1_0);
        MlasSveStoreFloat32(pg1, out1_1, acc1_1);
    }
    for (; col < n; col += vl) {
        svbool_t pg = svwhilelt_b32(col, n);
        float* out0 = res + 0 * ldc + col;
        float* out1 = res + 1 * ldc + col;
        svfloat32_t acc0 = zero_vec;
        svfloat32_t acc1 = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b_ptr = b + p * PACKED_B_BLOCK_WIDTH + col;
                svfloat32_t b0 = MlasSveLoadFloat32(pg, b_ptr);
                float a0 = a[0 * lda + p];
                float a1 = a[1 * lda + p];
                if constexpr (!Alpha1) {
                    a0 *= alpha;
                    a1 *= alpha;
                }
                svfloat32_t va0 = MlasSveBroadcastFloat32(a0);
                svfloat32_t va1 = MlasSveBroadcastFloat32(a1);
                acc0 = MlasSveMultiplyAddFloat32(pg, b0, va0, acc0);
                acc1 = MlasSveMultiplyAddFloat32(pg, b0, va1, acc1);
            }
        }
        if constexpr (!ZeroMode) {
            acc0 = MlasSveAddFloat32(pg, acc0, svld1(pg, out0));
            acc1 = MlasSveAddFloat32(pg, acc1, svld1(pg, out1));
        }
        MlasSveStoreFloat32(pg, out0, acc0);
        MlasSveStoreFloat32(pg, out1, acc1);
    }
}

template <bool ZeroMode, bool Alpha1>
inline void
processrows_1(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl
)
{
    size_t k_step = 8 * vl;

    MLAS_UNREFERENCED_PARAMETER(ldc);
    MLAS_UNREFERENCED_PARAMETER(lda);

    svfloat32_t zero_vec = MlasSveBroadcastFloat32(0.f);
    size_t col = 0;
    for (; col + 2 * vl <= n; col += 2 * vl) {
        svbool_t pg0 = svwhilelt_b32(col, n);
        svbool_t pg1 = svwhilelt_b32(col + vl, n);
        float* out0 = res + col;
        float* out1 = res + col + vl;
        svfloat32_t acc0 = zero_vec;
        svfloat32_t acc1 = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b0_ptr = b + p * PACKED_B_BLOCK_WIDTH + col;
                const float* b1_ptr = b0_ptr + vl;
                svfloat32_t b0 = MlasSveLoadFloat32(pg0, b0_ptr);
                svfloat32_t b1 = MlasSveLoadFloat32(pg1, b1_ptr);
                float a0 = a[p];
                if constexpr (!Alpha1)
                    a0 *= alpha;
                svfloat32_t va0 = MlasSveBroadcastFloat32(a0);
                acc0 = MlasSveMultiplyAddFloat32(pg0, b0, va0, acc0);
                acc1 = MlasSveMultiplyAddFloat32(pg1, b1, va0, acc1);
            }
        }
        if constexpr (!ZeroMode) {
            acc0 = MlasSveAddFloat32(pg0, acc0, svld1(pg0, out0));
            acc1 = MlasSveAddFloat32(pg1, acc1, svld1(pg1, out1));
        }
        MlasSveStoreFloat32(pg0, out0, acc0);
        MlasSveStoreFloat32(pg1, out1, acc1);
    }
    for (; col < n; col += vl) {
        svbool_t pg = svwhilelt_b32(col, n);
        float* out = res + col;
        svfloat32_t acc = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b_ptr = b + p * PACKED_B_BLOCK_WIDTH + col;
                svfloat32_t b = MlasSveLoadFloat32(pg, b_ptr);
                float a0 = a[p];
                if constexpr (!Alpha1)
                    a0 *= alpha;
                svfloat32_t va0 = MlasSveBroadcastFloat32(a0);
                acc = MlasSveMultiplyAddFloat32(pg, b, va0, acc);
            }
        }
        if constexpr (!ZeroMode) {
            acc = MlasSveAddFloat32(pg, acc, svld1(pg, out));
        }
        MlasSveStoreFloat32(pg, out, acc);
    }
}

template <auto ProcessFn>
inline void
ProcessRowsTemplate(

    const float* __restrict A,
    size_t lda,
    const float* __restrict B,
    float* __restrict C,
    size_t ldc,
    size_t K,
    size_t N,
    float alpha
)
{
    size_t n = 0;
    const size_t vl = svcntw();
    while (n < N) {
        int cols = (n + PACKED_B_BLOCK_WIDTH <= N) ? PACKED_B_BLOCK_WIDTH : (N - n);
        ProcessFn(A, B, C, K, cols, lda, ldc, alpha, vl);
        B += cols * K;
        C += cols;
        n += cols;
    }
}

size_t MLASCALL
MlasSgemmKernelZero_sve(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
)
{
    if (svcntw() == 4u) {
        size_t rows = MlasSgemmKernelZero(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        return rows;
    } else {
        if (alpha == 1.0f) {
            if (CountM >= 8) {
                ProcessRowsTemplate<processrows_8<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 8;
            } else if (CountM >= 6) {
                ProcessRowsTemplate<processrows_6<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 6;
            } else if (CountM >= 4) {
                ProcessRowsTemplate<processrows_4<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 4;
            } else if (CountM >= 2) {
                ProcessRowsTemplate<processrows_2<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 2;
            } else
                ProcessRowsTemplate<processrows_1<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 1;
        } else {
            if (CountM >= 8) {
                ProcessRowsTemplate<processrows_8<true, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 8;
            } else if (CountM >= 6) {
                ProcessRowsTemplate<processrows_6<true, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 6;
            } else if (CountM >= 4) {
                ProcessRowsTemplate<processrows_4<true, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 4;
            } else if (CountM >= 2) {
                ProcessRowsTemplate<processrows_2<true, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 2;
            } else
                ProcessRowsTemplate<processrows_1<true, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 1;
        }
    }
}

size_t MLASCALL
MlasSgemmKernelAdd_sve(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
)
{
    if (svcntw() == 4u) {
        size_t rows = MlasSgemmKernelAdd(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        return rows;
    } else {
        if (alpha == 1.0f) {
            if (CountM >= 8) {
                ProcessRowsTemplate<processrows_8<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 8;
            } else if (CountM >= 6) {
                ProcessRowsTemplate<processrows_6<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 6;
            } else if (CountM >= 4) {
                ProcessRowsTemplate<processrows_4<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 4;
            } else if (CountM >= 2) {
                ProcessRowsTemplate<processrows_2<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 2;
            } else
                ProcessRowsTemplate<processrows_1<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 1;
        } else {
            if (CountM >= 8) {
                ProcessRowsTemplate<processrows_8<false, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 8;
            } else if (CountM >= 6) {
                ProcessRowsTemplate<processrows_6<false, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 6;
            } else if (CountM >= 4) {
                ProcessRowsTemplate<processrows_4<false, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 4;
            } else if (CountM >= 2) {
                ProcessRowsTemplate<processrows_2<false, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
                return 2;
            } else
                ProcessRowsTemplate<processrows_1<false, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 1;
        }
    }
}

MLAS_SVE_TARGET
inline size_t
VL()
{
    static size_t fp32Lanes = svcntw();  // evaluated only once, the first time it's called
    return fp32Lanes;
}

MLAS_SVE_TARGET
void inline Transpose_SVE512_4x4(float* D, const float* B, size_t ldb)
{
    const static size_t VL = svcntw();
    MLAS_SVBOOL p = svwhilelt_b32(0ULL, VL / 4);
    MLAS_SVBOOL p3 = svwhilelt_b32(0ULL, VL / 2);
    MLAS_SVBOOL p1 = svnot_b_z(svwhilelt_b32(0ULL, VL), p);
    p1 = svand_b_z(p3, p3, p1);
    p3 = svrev_b32(p1);
    MLAS_SVBOOL p4 = svrev_b32(p);

    MLAS_SVFLOAT32 t0 = MlasSveLoadFloat32(p, &B[ldb * 0]);
    MLAS_SVFLOAT32 t1 = MlasSveLoadFloat32(p, &B[ldb * 1]);
    MLAS_SVFLOAT32 t2 = MlasSveLoadFloat32(p, &B[ldb * 2]);
    MLAS_SVFLOAT32 t3 = MlasSveLoadFloat32(p, &B[ldb * 3]);

    MLAS_SVFLOAT32 t02 = svzip1_f32(t0, t2);
    MLAS_SVFLOAT32 t13 = svzip1_f32(t1, t3);
    MLAS_SVFLOAT32 t0123 = svzip1_f32(t02, t13);  // This zips the first half together

    MlasSveStoreFloat32(p, D, t0123);
    MlasSveStoreFloat32(p1, &D[12], t0123);
    MlasSveStoreFloat32(p3, &D[24], t0123);
    MlasSveStoreFloat32(p4, &D[36], t0123);
}

MLAS_SVE_TARGET
void static inline Transpose_SVE256_4x4(float* D, const float* B, size_t ldb)
{
    const static size_t VL = svcntw();
    MLAS_SVBOOL p = svwhilelt_b32(0ULL, VL / 2);

    MLAS_SVFLOAT32 t0 = MlasSveLoadFloat32(p, &B[ldb * 0]);
    MLAS_SVFLOAT32 t1 = MlasSveLoadFloat32(p, &B[ldb * 1]);
    MLAS_SVFLOAT32 t2 = MlasSveLoadFloat32(p, &B[ldb * 2]);
    MLAS_SVFLOAT32 t3 = MlasSveLoadFloat32(p, &B[ldb * 3]);

    MLAS_SVBOOL p1 = svnot_b_z(svwhilelt_b32(0ULL, VL), p);
    MLAS_SVFLOAT32 t02 = svzip1_f32(t0, t2);
    MLAS_SVFLOAT32 t13 = svzip1_f32(t1, t3);
    MLAS_SVFLOAT32 first_t0123 = svzip1_f32(t02, t13);   // This zips the first half together
    MLAS_SVFLOAT32 second_t0123 = svzip2_f32(t02, t13);  // This zips the second half together

    MlasSveStoreFloat32(p, D, first_t0123);
    MlasSveStoreFloat32(p1, &D[12], first_t0123);
    MlasSveStoreFloat32(p, &D[32], second_t0123);
    MlasSveStoreFloat32(p1, &D[44], second_t0123);
}

MLAS_SVE_TARGET
void static inline Transpose_SVE128_4x4(float* D, const float* B, size_t ldb)
{
    const static size_t VL = svcntw();
    MLAS_SVBOOL p = svwhilelt_b32(0ULL, VL);

    MLAS_SVFLOAT32 v1 = MlasSveLoadFloat32(p, &B[ldb * 0]);
    MLAS_SVFLOAT32 v2 = MlasSveLoadFloat32(p, &B[ldb * 1]);
    MLAS_SVFLOAT32 v4 = MlasSveLoadFloat32(p, &B[ldb * 2]);
    MLAS_SVFLOAT32 v5 = MlasSveLoadFloat32(p, &B[ldb * 3]);

    MLAS_SVFLOAT32 v3 = svzip1_f32(v1, v4);
    v1 = svzip2_f32(v1, v4);

    v4 = svzip1_f32(v2, v5);
    v2 = svzip2_f32(v2, v5);

    v5 = svzip1_f32(v3, v4);
    v3 = svzip2_f32(v3, v4);

    v4 = svzip1_f32(v1, v2);
    v1 = svzip2_f32(v1, v2);

    MlasSveStoreFloat32(p, &D[0], v5);
    MlasSveStoreFloat32(p, &D[16], v3);
    MlasSveStoreFloat32(p, &D[32], v4);
    MlasSveStoreFloat32(p, &D[48], v1);
}

MLAS_SVE_TARGET
void static inline Transpose_SVE256_8x8(float* D, const float* B, size_t ldb)
{
    const static size_t VL = svcntw();

    MLAS_SVBOOL p = svwhilelt_b32(0LL, VL);

    MLAS_SVFLOAT32 v1 = MlasSveLoadFloat32(p, &B[ldb * 0]);
    MLAS_SVFLOAT32 v2 = MlasSveLoadFloat32(p, &B[ldb * 1]);
    MLAS_SVFLOAT32 v4 = MlasSveLoadFloat32(p, &B[ldb * 2]);
    MLAS_SVFLOAT32 v5 = MlasSveLoadFloat32(p, &B[ldb * 3]);

    MLAS_SVFLOAT32 v6 = MlasSveLoadFloat32(p, &B[ldb * 4]);
    MLAS_SVFLOAT32 v7 = MlasSveLoadFloat32(p, &B[ldb * 5]);
    MLAS_SVFLOAT32 v8 = MlasSveLoadFloat32(p, &B[ldb * 6]);
    MLAS_SVFLOAT32 v9 = MlasSveLoadFloat32(p, &B[ldb * 7]);

    // First mix
    MLAS_SVFLOAT32 v3 = svzip1_f32(v1, v6);
    v1 = svzip2_f32(v1, v6);

    v6 = svzip1_f32(v2, v7);
    v2 = svzip2_f32(v2, v7);

    v7 = svzip1_f32(v4, v8);
    v4 = svzip2_f32(v4, v8);

    v8 = svzip1_f32(v5, v9);

    v5 = svzip2_f32(v5, v9);

    // Second mix

    v9 = svzip1_f32(v3, v7);
    v3 = svzip2_f32(v3, v7);

    v7 = svzip1_f32(v6, v8);
    v6 = svzip2_f32(v6, v8);

    v8 = svzip1_f32(v1, v4);
    v1 = svzip2_f32(v1, v4);

    v4 = svzip1_f32(v2, v5);
    v2 = svzip2_f32(v2, v5);

    // Third mix

    v5 = svzip1_f32(v9, v7);
    v9 = svzip2_f32(v9, v7);

    v7 = svzip1_f32(v8, v4);
    v8 = svzip2_f32(v8, v4);

    v4 = svzip1_f32(v3, v6);
    v3 = svzip2_f32(v3, v6);

    v6 = svzip1_f32(v1, v2);
    v1 = svzip2_f32(v1, v2);

    MlasSveStoreFloat32(p, &D[0], v5);
    MlasSveStoreFloat32(p, &D[16], v9);
    MlasSveStoreFloat32(p, &D[32], v4);
    MlasSveStoreFloat32(p, &D[48], v3);
    MlasSveStoreFloat32(p, &D[64], v7);
    MlasSveStoreFloat32(p, &D[80], v8);
    MlasSveStoreFloat32(p, &D[96], v6);
    MlasSveStoreFloat32(p, &D[112], v1);
}

MLAS_SVE_TARGET
void static inline Transpose_SVE512_16x16(float* D, const float* B, size_t ldb)
{
    const static size_t VL = svcntw();
    MLAS_SVBOOL p = svwhilelt_b32(0LL, VL);

    MLAS_SVFLOAT32 v1 = MlasSveLoadFloat32(p, &B[ldb * 0]);
    MLAS_SVFLOAT32 v2 = MlasSveLoadFloat32(p, &B[ldb * 1]);
    MLAS_SVFLOAT32 v3 = MlasSveLoadFloat32(p, &B[ldb * 2]);
    MLAS_SVFLOAT32 v4 = MlasSveLoadFloat32(p, &B[ldb * 3]);

    MLAS_SVFLOAT32 v5 = MlasSveLoadFloat32(p, &B[ldb * 4]);
    MLAS_SVFLOAT32 v6 = MlasSveLoadFloat32(p, &B[ldb * 5]);
    MLAS_SVFLOAT32 v7 = MlasSveLoadFloat32(p, &B[ldb * 6]);
    MLAS_SVFLOAT32 v8 = MlasSveLoadFloat32(p, &B[ldb * 7]);

    MLAS_SVFLOAT32 v9 = MlasSveLoadFloat32(p, &B[ldb * 8]);
    MLAS_SVFLOAT32 v10 = MlasSveLoadFloat32(p, &B[ldb * 9]);
    MLAS_SVFLOAT32 v11 = MlasSveLoadFloat32(p, &B[ldb * 10]);
    MLAS_SVFLOAT32 v12 = MlasSveLoadFloat32(p, &B[ldb * 11]);

    MLAS_SVFLOAT32 v13 = MlasSveLoadFloat32(p, &B[ldb * 12]);
    MLAS_SVFLOAT32 v14 = MlasSveLoadFloat32(p, &B[ldb * 13]);
    MLAS_SVFLOAT32 v15 = MlasSveLoadFloat32(p, &B[ldb * 14]);
    MLAS_SVFLOAT32 v16 = MlasSveLoadFloat32(p, &B[ldb * 15]);

    MLAS_SVFLOAT32 v17 = svzip1_f32(v1, v9);
    MLAS_SVFLOAT32 v18 = svzip2_f32(v1, v9);

    MLAS_SVFLOAT32 v19 = svzip1_f32(v2, v10);
    MLAS_SVFLOAT32 v20 = svzip2_f32(v2, v10);

    MLAS_SVFLOAT32 v21 = svzip1_f32(v3, v11);
    MLAS_SVFLOAT32 v22 = svzip2_f32(v3, v11);

    MLAS_SVFLOAT32 v23 = svzip1_f32(v4, v12);
    MLAS_SVFLOAT32 v24 = svzip2_f32(v4, v12);

    MLAS_SVFLOAT32 v25 = svzip1_f32(v5, v13);
    MLAS_SVFLOAT32 v26 = svzip2_f32(v5, v13);

    MLAS_SVFLOAT32 v27 = svzip1_f32(v6, v14);
    MLAS_SVFLOAT32 v28 = svzip2_f32(v6, v14);

    MLAS_SVFLOAT32 v29 = svzip1_f32(v7, v15);
    MLAS_SVFLOAT32 v30 = svzip2_f32(v7, v15);

    MLAS_SVFLOAT32 v31 = svzip1_f32(v8, v16);
    MLAS_SVFLOAT32 v32 = svzip2_f32(v8, v16);

    v1 = svzip1_f32(v17, v25);
    v9 = svzip2_f32(v17, v25);

    v2 = svzip1_f32(v18, v26);
    v10 = svzip2_f32(v18, v26);

    v3 = svzip1_f32(v19, v27);
    v11 = svzip2_f32(v19, v27);

    v4 = svzip1_f32(v20, v28);
    v12 = svzip2_f32(v20, v28);

    v5 = svzip1_f32(v21, v29);
    v13 = svzip2_f32(v21, v29);

    v6 = svzip1_f32(v22, v30);
    v14 = svzip2_f32(v22, v30);

    v7 = svzip1_f32(v23, v31);
    v15 = svzip2_f32(v23, v31);

    v8 = svzip1_f32(v24, v32);
    v16 = svzip2_f32(v24, v32);

    v17 = svzip1_f32(v1, v5);
    v25 = svzip2_f32(v1, v5);

    v18 = svzip1_f32(v9, v13);
    v26 = svzip2_f32(v9, v13);

    v19 = svzip1_f32(v2, v6);
    v27 = svzip2_f32(v2, v6);

    v20 = svzip1_f32(v10, v14);
    v28 = svzip2_f32(v10, v14);

    v21 = svzip1_f32(v3, v7);
    v29 = svzip2_f32(v3, v7);

    v22 = svzip1_f32(v11, v15);
    v30 = svzip2_f32(v11, v15);

    v23 = svzip1_f32(v4, v8);
    v31 = svzip2_f32(v4, v8);

    v24 = svzip1_f32(v12, v16);
    v32 = svzip2_f32(v12, v16);

    v1 = svzip1_f32(v17, v21);
    v9 = svzip2_f32(v17, v21);

    v2 = svzip1_f32(v25, v29);
    v10 = svzip2_f32(v25, v29);

    v3 = svzip1_f32(v18, v22);
    v11 = svzip2_f32(v18, v22);

    v4 = svzip1_f32(v26, v30);
    v12 = svzip2_f32(v26, v30);

    v5 = svzip1_f32(v19, v23);
    v13 = svzip2_f32(v19, v23);

    v6 = svzip1_f32(v27, v31);
    v14 = svzip2_f32(v27, v31);

    v7 = svzip1_f32(v20, v24);
    v15 = svzip2_f32(v20, v24);

    v8 = svzip1_f32(v28, v32);
    v16 = svzip2_f32(v28, v32);

    MlasSveStoreFloat32(p, &D[0], v1);
    MlasSveStoreFloat32(p, &D[16], v9);
    MlasSveStoreFloat32(p, &D[32], v2);
    MlasSveStoreFloat32(p, &D[48], v10);

    MlasSveStoreFloat32(p, &D[64], v3);
    MlasSveStoreFloat32(p, &D[80], v11);
    MlasSveStoreFloat32(p, &D[96], v4);
    MlasSveStoreFloat32(p, &D[112], v12);

    MlasSveStoreFloat32(p, &D[128], v5);
    MlasSveStoreFloat32(p, &D[144], v13);
    MlasSveStoreFloat32(p, &D[160], v6);
    MlasSveStoreFloat32(p, &D[176], v14);

    MlasSveStoreFloat32(p, &D[192], v7);
    MlasSveStoreFloat32(p, &D[208], v15);
    MlasSveStoreFloat32(p, &D[224], v8);
    MlasSveStoreFloat32(p, &D[240], v16);
}

template <unsigned N>
void
TransposePackBNx8(
    float* D,
    const float* B,
    size_t ldb
)
{
    for (unsigned n = 0; n < N / 8; n++) {
        Transpose_SVE256_8x8(D, B, ldb);
        D += 8;
        B += ldb * 8;
    }
}

template <unsigned N>
void
MlasSveTransposePackBNx4(
    float* D,
    const float* B,
    size_t ldb
)
{
    for (unsigned n = 0; n < N / 4; n++) {
        if (VL() == 16) {
            Transpose_SVE512_4x4(&D[0], &B[0], ldb);
        } else if (VL() == 8) {
            Transpose_SVE256_4x4(&D[0], &B[0], ldb);
        } else if (VL() == 4) {
            Transpose_SVE128_4x4(&D[0], &B[0], ldb);
        }

        D += 4;
        B += ldb * 4;
    }
}

template void
MlasSveTransposePackBNx4<4>(
    float* D,
    const float* B,
    size_t ldb
);

template void
MlasSveTransposePackBNx4<8>(
    float* D,
    const float* B,
    size_t ldb
);

void MLAS_SVE_TARGET MLASCALL
SVE_TRANSPOSE(float*& D, const float*& b, size_t ldb, size_t& x)
{
    const static size_t VL = svcntw();
    if (VL == 16) {
        while (x >= 16) {
            Transpose_SVE512_16x16(&D[0], &b[0], ldb);
            D += 256;
            b += 16;
            x = x - 16;
        }
    } else if (VL == 8) {
        while (x >= 8) {
            TransposePackBNx8<16>(&D[0], &b[0], ldb);
            D += 128;
            b += 8;
            x = x - 8;
        }
    }
    while (x >= 4) {
        MlasSveTransposePackBNx4<16>(&D[0], &b[0], ldb);

        D += 16 * 4;
        b += 4;
        x = x - 4;
    }
}

void MLAS_SVE_TARGET MLASCALL
SCATTER_STORE(float* d, const float* b)
{
    MLAS_SVBOOL pb = svwhilelt_b32((int)0, 4);
    MLAS_SVFLOAT32 vec0 = MlasSveLoadFloat32(pb, b);

    svuint32_t idx = svindex_u32(0, 1);
    MLAS_SVBOOL pb_first_half = svcmpeq_u32(pb, idx, svdup_n_u32(0));
    MLAS_SVBOOL pb_second_half = svcmpeq_u32(pb, idx, svdup_n_u32(1));
    MLAS_SVBOOL pb_third_half = svcmpeq_u32(pb, idx, svdup_n_u32(2));
    MLAS_SVBOOL pb_fourth_half = svcmpeq_u32(pb, idx, svdup_n_u32(3));

    MlasSveStoreFloat32(pb_first_half, &d[0], vec0);
    MlasSveStoreFloat32(pb_second_half, &d[15], vec0);
    MlasSveStoreFloat32(pb_third_half, &d[30], vec0);
    MlasSveStoreFloat32(pb_fourth_half, &d[45], vec0);
}

void MLAS_SVE_TARGET MLASCALL
SVE_LOAD_STORE(float* D, const float* b)
{
    for (int i = 0; i < MLAS_SGEMM_STRIDEN_THREAD_ALIGN; i += VL()) {
        svfloat32_t vec0 = MlasSveLoadFloat32(svptrue_b32(), b + i);
        MlasSveStoreFloat32(svptrue_b32(), D + i, vec0);
    }
}

void MLAS_SVE_TARGET MLASCALL
SVE_ZERO_INITIALIZE(float* d)
{
    if (VL() == PACKED_B_BLOCK_WIDTH) {
        MlasSveStoreFloat32(svptrue_b32(), d, svdup_f32(0));
    } else {
        MlasSveStoreFloat32(svptrue_b32(), d, svdup_f32(0));
        MlasSveStoreFloat32(svptrue_b32(), d + VL(), svdup_f32(0));
    }
}

#endif