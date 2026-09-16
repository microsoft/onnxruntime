/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2025 FUJITSU LIMITED

Licensed under the MIT License.

Module Name:

    halfgemm_kernel_sve.cpp

Abstract:

    SVE implementation of the FP16 GEMM kernels used by hgemm.cpp. This is the
    source for aarch64/halfgemm_sve_asm.S (see sve/gen_sve_asm.py), so it must
    not make calls or reference global data.

--*/

#ifdef MLAS_USE_SVE

#ifndef __clang__
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#endif

#ifdef __clang__
#define MLAS_SVE_TARGET __attribute__((target("arch=armv8.2-a+sve+fp16")))
#else
#define MLAS_SVE_TARGET
#endif

#include <arm_sve.h>

#include <algorithm>

#include "halfgemm_sve.h"

using _mlas_fp16_ = mlas_sve_fp16_t;

#if !defined(MLAS_UNREFERENCED_PARAMETER)
#define MLAS_UNREFERENCED_PARAMETER(x) ((void)(x))
#endif
#if !defined(MLASCALL)
#define MLASCALL
#endif

// Helpers have to be inlined, gen_sve_asm.py rejects calls.
#if !defined(MLAS_FORCEINLINE)
#define MLAS_FORCEINLINE __attribute__((always_inline)) inline
#endif

//
// Define to accumulate in fp32 in the single-row remainder kernel.
//
// #define MLAS_HGEMM_ACCUMULATE_FP32

//
// Number of K iterations processed per pass over the column tiles.
//
const size_t hk_step = 64;

//
// ZeroMode overwrites C instead of adding to it. Alpha1 skips the alpha scale.
//
template <bool ZeroMode, bool Alpha1>
MLAS_SVE_TARGET MLAS_FORCEINLINE void
hprocessrows_8(
    const _mlas_fp16_* __restrict__ a,
    const _mlas_fp16_* __restrict__ b,
    _mlas_fp16_* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl
)
{
    const svfloat16_t zero_vec = svdup_n_f16((__fp16)0.f);
    const __fp16 halpha = (__fp16)alpha;

    size_t col = 0;
    for (; col + 2 * vl <= n; col += 2 * vl) {
        svbool_t pg0 = svwhilelt_b16(col, n);
        svbool_t pg1 = svwhilelt_b16(col + vl, n);

        svfloat16_t acc0_0 = zero_vec, acc0_1 = zero_vec;
        svfloat16_t acc1_0 = zero_vec, acc1_1 = zero_vec;
        svfloat16_t acc2_0 = zero_vec, acc2_1 = zero_vec;
        svfloat16_t acc3_0 = zero_vec, acc3_1 = zero_vec;
        svfloat16_t acc4_0 = zero_vec, acc4_1 = zero_vec;
        svfloat16_t acc5_0 = zero_vec, acc5_1 = zero_vec;
        svfloat16_t acc6_0 = zero_vec, acc6_1 = zero_vec;
        svfloat16_t acc7_0 = zero_vec, acc7_1 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += hk_step) {
            size_t k_max = std::min(k_block + hk_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const _mlas_fp16_* b0_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
                const _mlas_fp16_* b1_ptr = b0_ptr + vl;
                svfloat16_t b0 = svld1_f16(pg0, (const __fp16*)b0_ptr);
                svfloat16_t b1 = svld1_f16(pg1, (const __fp16*)b1_ptr);

                __fp16 a0 = reinterpret_cast<const __fp16*>(a)[0 * lda + p];
                __fp16 a1 = reinterpret_cast<const __fp16*>(a)[1 * lda + p];
                __fp16 a2 = reinterpret_cast<const __fp16*>(a)[2 * lda + p];
                __fp16 a3 = reinterpret_cast<const __fp16*>(a)[3 * lda + p];
                __fp16 a4 = reinterpret_cast<const __fp16*>(a)[4 * lda + p];
                __fp16 a5 = reinterpret_cast<const __fp16*>(a)[5 * lda + p];
                __fp16 a6 = reinterpret_cast<const __fp16*>(a)[6 * lda + p];
                __fp16 a7 = reinterpret_cast<const __fp16*>(a)[7 * lda + p];

                if constexpr (!Alpha1) {
                    a0 *= halpha; a1 *= halpha; a2 *= halpha; a3 *= halpha;
                    a4 *= halpha; a5 *= halpha; a6 *= halpha; a7 *= halpha;
                }

                acc0_0 = svmla_f16_m(pg0, acc0_0, b0, svdup_n_f16(a0));
                acc0_1 = svmla_f16_m(pg1, acc0_1, b1, svdup_n_f16(a0));
                acc1_0 = svmla_f16_m(pg0, acc1_0, b0, svdup_n_f16(a1));
                acc1_1 = svmla_f16_m(pg1, acc1_1, b1, svdup_n_f16(a1));
                acc2_0 = svmla_f16_m(pg0, acc2_0, b0, svdup_n_f16(a2));
                acc2_1 = svmla_f16_m(pg1, acc2_1, b1, svdup_n_f16(a2));
                acc3_0 = svmla_f16_m(pg0, acc3_0, b0, svdup_n_f16(a3));
                acc3_1 = svmla_f16_m(pg1, acc3_1, b1, svdup_n_f16(a3));
                acc4_0 = svmla_f16_m(pg0, acc4_0, b0, svdup_n_f16(a4));
                acc4_1 = svmla_f16_m(pg1, acc4_1, b1, svdup_n_f16(a4));
                acc5_0 = svmla_f16_m(pg0, acc5_0, b0, svdup_n_f16(a5));
                acc5_1 = svmla_f16_m(pg1, acc5_1, b1, svdup_n_f16(a5));
                acc6_0 = svmla_f16_m(pg0, acc6_0, b0, svdup_n_f16(a6));
                acc6_1 = svmla_f16_m(pg1, acc6_1, b1, svdup_n_f16(a6));
                acc7_0 = svmla_f16_m(pg0, acc7_0, b0, svdup_n_f16(a7));
                acc7_1 = svmla_f16_m(pg1, acc7_1, b1, svdup_n_f16(a7));
            }
        }

        _mlas_fp16_* out0 = res + 0 * ldc + col;
        _mlas_fp16_* out1 = res + 1 * ldc + col;
        _mlas_fp16_* out2 = res + 2 * ldc + col;
        _mlas_fp16_* out3 = res + 3 * ldc + col;
        _mlas_fp16_* out4 = res + 4 * ldc + col;
        _mlas_fp16_* out5 = res + 5 * ldc + col;
        _mlas_fp16_* out6 = res + 6 * ldc + col;
        _mlas_fp16_* out7 = res + 7 * ldc + col;

        if constexpr (!ZeroMode) {
            acc0_0 = svadd_f16_m(pg0, acc0_0, svld1_f16(pg0, (const __fp16*)out0));
            acc1_0 = svadd_f16_m(pg0, acc1_0, svld1_f16(pg0, (const __fp16*)out1));
            acc2_0 = svadd_f16_m(pg0, acc2_0, svld1_f16(pg0, (const __fp16*)out2));
            acc3_0 = svadd_f16_m(pg0, acc3_0, svld1_f16(pg0, (const __fp16*)out3));
            acc4_0 = svadd_f16_m(pg0, acc4_0, svld1_f16(pg0, (const __fp16*)out4));
            acc5_0 = svadd_f16_m(pg0, acc5_0, svld1_f16(pg0, (const __fp16*)out5));
            acc6_0 = svadd_f16_m(pg0, acc6_0, svld1_f16(pg0, (const __fp16*)out6));
            acc7_0 = svadd_f16_m(pg0, acc7_0, svld1_f16(pg0, (const __fp16*)out7));

            acc0_1 = svadd_f16_m(pg1, acc0_1, svld1_f16(pg1, (const __fp16*)out0 + vl));
            acc1_1 = svadd_f16_m(pg1, acc1_1, svld1_f16(pg1, (const __fp16*)out1 + vl));
            acc2_1 = svadd_f16_m(pg1, acc2_1, svld1_f16(pg1, (const __fp16*)out2 + vl));
            acc3_1 = svadd_f16_m(pg1, acc3_1, svld1_f16(pg1, (const __fp16*)out3 + vl));
            acc4_1 = svadd_f16_m(pg1, acc4_1, svld1_f16(pg1, (const __fp16*)out4 + vl));
            acc5_1 = svadd_f16_m(pg1, acc5_1, svld1_f16(pg1, (const __fp16*)out5 + vl));
            acc6_1 = svadd_f16_m(pg1, acc6_1, svld1_f16(pg1, (const __fp16*)out6 + vl));
            acc7_1 = svadd_f16_m(pg1, acc7_1, svld1_f16(pg1, (const __fp16*)out7 + vl));
        }

        svst1_f16(pg0, (__fp16*)out0, acc0_0);
        svst1_f16(pg0, (__fp16*)out1, acc1_0);
        svst1_f16(pg0, (__fp16*)out2, acc2_0);
        svst1_f16(pg0, (__fp16*)out3, acc3_0);
        svst1_f16(pg0, (__fp16*)out4, acc4_0);
        svst1_f16(pg0, (__fp16*)out5, acc5_0);
        svst1_f16(pg0, (__fp16*)out6, acc6_0);
        svst1_f16(pg0, (__fp16*)out7, acc7_0);

        svst1_f16(pg1, (__fp16*)out0 + vl, acc0_1);
        svst1_f16(pg1, (__fp16*)out1 + vl, acc1_1);
        svst1_f16(pg1, (__fp16*)out2 + vl, acc2_1);
        svst1_f16(pg1, (__fp16*)out3 + vl, acc3_1);
        svst1_f16(pg1, (__fp16*)out4 + vl, acc4_1);
        svst1_f16(pg1, (__fp16*)out5 + vl, acc5_1);
        svst1_f16(pg1, (__fp16*)out6 + vl, acc6_1);
        svst1_f16(pg1, (__fp16*)out7 + vl, acc7_1);
    }

    for (; col < n; col += vl) {
        svbool_t pg = svwhilelt_b16(col, n);

        svfloat16_t acc0 = zero_vec, acc1 = zero_vec, acc2 = zero_vec, acc3 = zero_vec;
        svfloat16_t acc4 = zero_vec, acc5 = zero_vec, acc6 = zero_vec, acc7 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += hk_step) {
            size_t k_max = std::min(k_block + hk_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const _mlas_fp16_* b_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
                svfloat16_t b0 = svld1_f16(pg, (const __fp16*)b_ptr);

                __fp16 a0 = reinterpret_cast<const __fp16*>(a)[0 * lda + p];
                __fp16 a1 = reinterpret_cast<const __fp16*>(a)[1 * lda + p];
                __fp16 a2 = reinterpret_cast<const __fp16*>(a)[2 * lda + p];
                __fp16 a3 = reinterpret_cast<const __fp16*>(a)[3 * lda + p];
                __fp16 a4 = reinterpret_cast<const __fp16*>(a)[4 * lda + p];
                __fp16 a5 = reinterpret_cast<const __fp16*>(a)[5 * lda + p];
                __fp16 a6 = reinterpret_cast<const __fp16*>(a)[6 * lda + p];
                __fp16 a7 = reinterpret_cast<const __fp16*>(a)[7 * lda + p];

                if constexpr (!Alpha1) {
                    a0 *= halpha; a1 *= halpha; a2 *= halpha; a3 *= halpha;
                    a4 *= halpha; a5 *= halpha; a6 *= halpha; a7 *= halpha;
                }
                acc0 = svmla_f16_m(pg, acc0, b0, svdup_n_f16(a0));
                acc1 = svmla_f16_m(pg, acc1, b0, svdup_n_f16(a1));
                acc2 = svmla_f16_m(pg, acc2, b0, svdup_n_f16(a2));
                acc3 = svmla_f16_m(pg, acc3, b0, svdup_n_f16(a3));
                acc4 = svmla_f16_m(pg, acc4, b0, svdup_n_f16(a4));
                acc5 = svmla_f16_m(pg, acc5, b0, svdup_n_f16(a5));
                acc6 = svmla_f16_m(pg, acc6, b0, svdup_n_f16(a6));
                acc7 = svmla_f16_m(pg, acc7, b0, svdup_n_f16(a7));
            }
        }

        _mlas_fp16_* out0 = res + 0 * ldc + col;
        _mlas_fp16_* out1 = res + 1 * ldc + col;
        _mlas_fp16_* out2 = res + 2 * ldc + col;
        _mlas_fp16_* out3 = res + 3 * ldc + col;
        _mlas_fp16_* out4 = res + 4 * ldc + col;
        _mlas_fp16_* out5 = res + 5 * ldc + col;
        _mlas_fp16_* out6 = res + 6 * ldc + col;
        _mlas_fp16_* out7 = res + 7 * ldc + col;

        if constexpr (!ZeroMode) {
            acc0 = svadd_f16_m(pg, acc0, svld1_f16(pg, (const __fp16*)out0));
            acc1 = svadd_f16_m(pg, acc1, svld1_f16(pg, (const __fp16*)out1));
            acc2 = svadd_f16_m(pg, acc2, svld1_f16(pg, (const __fp16*)out2));
            acc3 = svadd_f16_m(pg, acc3, svld1_f16(pg, (const __fp16*)out3));
            acc4 = svadd_f16_m(pg, acc4, svld1_f16(pg, (const __fp16*)out4));
            acc5 = svadd_f16_m(pg, acc5, svld1_f16(pg, (const __fp16*)out5));
            acc6 = svadd_f16_m(pg, acc6, svld1_f16(pg, (const __fp16*)out6));
            acc7 = svadd_f16_m(pg, acc7, svld1_f16(pg, (const __fp16*)out7));
        }

        svst1_f16(pg, (__fp16*)out0, acc0);
        svst1_f16(pg, (__fp16*)out1, acc1);
        svst1_f16(pg, (__fp16*)out2, acc2);
        svst1_f16(pg, (__fp16*)out3, acc3);
        svst1_f16(pg, (__fp16*)out4, acc4);
        svst1_f16(pg, (__fp16*)out5, acc5);
        svst1_f16(pg, (__fp16*)out6, acc6);
        svst1_f16(pg, (__fp16*)out7, acc7);
    }
}

template <bool ZeroMode, bool Alpha1>
MLAS_SVE_TARGET MLAS_FORCEINLINE void
hprocessrows_6(
    const _mlas_fp16_* __restrict__ a,
    const _mlas_fp16_* __restrict__ b,
    _mlas_fp16_* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl
)
{
    const svfloat16_t zero_vec = svdup_n_f16((__fp16)0.f);
    const __fp16 halpha = (__fp16)alpha;

    size_t col = 0;
    for (; col + 2 * vl <= n; col += 2 * vl) {
        svbool_t pg0 = svwhilelt_b16(col, n);
        svbool_t pg1 = svwhilelt_b16(col + vl, n);

        _mlas_fp16_* out0_0 = res + 0 * ldc + col;
        _mlas_fp16_* out0_1 = res + 0 * ldc + col + vl;
        _mlas_fp16_* out1_0 = res + 1 * ldc + col;
        _mlas_fp16_* out1_1 = res + 1 * ldc + col + vl;
        _mlas_fp16_* out2_0 = res + 2 * ldc + col;
        _mlas_fp16_* out2_1 = res + 2 * ldc + col + vl;
        _mlas_fp16_* out3_0 = res + 3 * ldc + col;
        _mlas_fp16_* out3_1 = res + 3 * ldc + col + vl;
        _mlas_fp16_* out4_0 = res + 4 * ldc + col;
        _mlas_fp16_* out4_1 = res + 4 * ldc + col + vl;
        _mlas_fp16_* out5_0 = res + 5 * ldc + col;
        _mlas_fp16_* out5_1 = res + 5 * ldc + col + vl;

        svfloat16_t acc0_0 = zero_vec, acc0_1 = zero_vec;
        svfloat16_t acc1_0 = zero_vec, acc1_1 = zero_vec;
        svfloat16_t acc2_0 = zero_vec, acc2_1 = zero_vec;
        svfloat16_t acc3_0 = zero_vec, acc3_1 = zero_vec;
        svfloat16_t acc4_0 = zero_vec, acc4_1 = zero_vec;
        svfloat16_t acc5_0 = zero_vec, acc5_1 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += hk_step) {
            size_t k_max = std::min(k_block + hk_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const _mlas_fp16_* b0_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
                const _mlas_fp16_* b1_ptr = b0_ptr + vl;
                svfloat16_t b0 = svld1_f16(pg0, (const __fp16*)b0_ptr);
                svfloat16_t b1 = svld1_f16(pg1, (const __fp16*)b1_ptr);

                __fp16 a0 = reinterpret_cast<const __fp16*>(a)[0 * lda + p];
                __fp16 a1 = reinterpret_cast<const __fp16*>(a)[1 * lda + p];
                __fp16 a2 = reinterpret_cast<const __fp16*>(a)[2 * lda + p];
                __fp16 a3 = reinterpret_cast<const __fp16*>(a)[3 * lda + p];
                __fp16 a4 = reinterpret_cast<const __fp16*>(a)[4 * lda + p];
                __fp16 a5 = reinterpret_cast<const __fp16*>(a)[5 * lda + p];

                if constexpr (!Alpha1) {
                    a0 *= halpha; a1 *= halpha; a2 *= halpha;
                    a3 *= halpha; a4 *= halpha; a5 *= halpha;
                }
                acc0_0 = svmla_f16_m(pg0, acc0_0, b0, svdup_n_f16(a0));
                acc0_1 = svmla_f16_m(pg1, acc0_1, b1, svdup_n_f16(a0));
                acc1_0 = svmla_f16_m(pg0, acc1_0, b0, svdup_n_f16(a1));
                acc1_1 = svmla_f16_m(pg1, acc1_1, b1, svdup_n_f16(a1));
                acc2_0 = svmla_f16_m(pg0, acc2_0, b0, svdup_n_f16(a2));
                acc2_1 = svmla_f16_m(pg1, acc2_1, b1, svdup_n_f16(a2));
                acc3_0 = svmla_f16_m(pg0, acc3_0, b0, svdup_n_f16(a3));
                acc3_1 = svmla_f16_m(pg1, acc3_1, b1, svdup_n_f16(a3));
                acc4_0 = svmla_f16_m(pg0, acc4_0, b0, svdup_n_f16(a4));
                acc4_1 = svmla_f16_m(pg1, acc4_1, b1, svdup_n_f16(a4));
                acc5_0 = svmla_f16_m(pg0, acc5_0, b0, svdup_n_f16(a5));
                acc5_1 = svmla_f16_m(pg1, acc5_1, b1, svdup_n_f16(a5));
            }
        }

        if constexpr (!ZeroMode) {
            acc0_0 = svadd_f16_m(pg0, acc0_0, svld1_f16(pg0, (const __fp16*)out0_0));
            acc0_1 = svadd_f16_m(pg1, acc0_1, svld1_f16(pg1, (const __fp16*)out0_1));
            acc1_0 = svadd_f16_m(pg0, acc1_0, svld1_f16(pg0, (const __fp16*)out1_0));
            acc1_1 = svadd_f16_m(pg1, acc1_1, svld1_f16(pg1, (const __fp16*)out1_1));
            acc2_0 = svadd_f16_m(pg0, acc2_0, svld1_f16(pg0, (const __fp16*)out2_0));
            acc2_1 = svadd_f16_m(pg1, acc2_1, svld1_f16(pg1, (const __fp16*)out2_1));
            acc3_0 = svadd_f16_m(pg0, acc3_0, svld1_f16(pg0, (const __fp16*)out3_0));
            acc3_1 = svadd_f16_m(pg1, acc3_1, svld1_f16(pg1, (const __fp16*)out3_1));
            acc4_0 = svadd_f16_m(pg0, acc4_0, svld1_f16(pg0, (const __fp16*)out4_0));
            acc4_1 = svadd_f16_m(pg1, acc4_1, svld1_f16(pg1, (const __fp16*)out4_1));
            acc5_0 = svadd_f16_m(pg0, acc5_0, svld1_f16(pg0, (const __fp16*)out5_0));
            acc5_1 = svadd_f16_m(pg1, acc5_1, svld1_f16(pg1, (const __fp16*)out5_1));
        }

        svst1_f16(pg0, (__fp16*)out0_0, acc0_0);
        svst1_f16(pg1, (__fp16*)out0_1, acc0_1);
        svst1_f16(pg0, (__fp16*)out1_0, acc1_0);
        svst1_f16(pg1, (__fp16*)out1_1, acc1_1);
        svst1_f16(pg0, (__fp16*)out2_0, acc2_0);
        svst1_f16(pg1, (__fp16*)out2_1, acc2_1);
        svst1_f16(pg0, (__fp16*)out3_0, acc3_0);
        svst1_f16(pg1, (__fp16*)out3_1, acc3_1);
        svst1_f16(pg0, (__fp16*)out4_0, acc4_0);
        svst1_f16(pg1, (__fp16*)out4_1, acc4_1);
        svst1_f16(pg0, (__fp16*)out5_0, acc5_0);
        svst1_f16(pg1, (__fp16*)out5_1, acc5_1);
    }

    for (; col < n; col += vl) {
        svbool_t pg = svwhilelt_b16(col, n);

        _mlas_fp16_* out0 = res + 0 * ldc + col;
        _mlas_fp16_* out1 = res + 1 * ldc + col;
        _mlas_fp16_* out2 = res + 2 * ldc + col;
        _mlas_fp16_* out3 = res + 3 * ldc + col;
        _mlas_fp16_* out4 = res + 4 * ldc + col;
        _mlas_fp16_* out5 = res + 5 * ldc + col;

        svfloat16_t acc0 = zero_vec, acc1 = zero_vec, acc2 = zero_vec;
        svfloat16_t acc3 = zero_vec, acc4 = zero_vec, acc5 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += hk_step) {
            size_t k_max = std::min(k_block + hk_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const _mlas_fp16_* b_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
                svfloat16_t b0 = svld1_f16(pg, (const __fp16*)b_ptr);

                __fp16 a0 = reinterpret_cast<const __fp16*>(a)[0 * lda + p];
                __fp16 a1 = reinterpret_cast<const __fp16*>(a)[1 * lda + p];
                __fp16 a2 = reinterpret_cast<const __fp16*>(a)[2 * lda + p];
                __fp16 a3 = reinterpret_cast<const __fp16*>(a)[3 * lda + p];
                __fp16 a4 = reinterpret_cast<const __fp16*>(a)[4 * lda + p];
                __fp16 a5 = reinterpret_cast<const __fp16*>(a)[5 * lda + p];

                if constexpr (!Alpha1) {
                    a0 *= halpha; a1 *= halpha; a2 *= halpha;
                    a3 *= halpha; a4 *= halpha; a5 *= halpha;
                }
                acc0 = svmla_f16_m(pg, acc0, b0, svdup_n_f16(a0));
                acc1 = svmla_f16_m(pg, acc1, b0, svdup_n_f16(a1));
                acc2 = svmla_f16_m(pg, acc2, b0, svdup_n_f16(a2));
                acc3 = svmla_f16_m(pg, acc3, b0, svdup_n_f16(a3));
                acc4 = svmla_f16_m(pg, acc4, b0, svdup_n_f16(a4));
                acc5 = svmla_f16_m(pg, acc5, b0, svdup_n_f16(a5));
            }
        }

        if constexpr (!ZeroMode) {
            acc0 = svadd_f16_m(pg, acc0, svld1_f16(pg, (const __fp16*)out0));
            acc1 = svadd_f16_m(pg, acc1, svld1_f16(pg, (const __fp16*)out1));
            acc2 = svadd_f16_m(pg, acc2, svld1_f16(pg, (const __fp16*)out2));
            acc3 = svadd_f16_m(pg, acc3, svld1_f16(pg, (const __fp16*)out3));
            acc4 = svadd_f16_m(pg, acc4, svld1_f16(pg, (const __fp16*)out4));
            acc5 = svadd_f16_m(pg, acc5, svld1_f16(pg, (const __fp16*)out5));
        }

        svst1_f16(pg, (__fp16*)out0, acc0);
        svst1_f16(pg, (__fp16*)out1, acc1);
        svst1_f16(pg, (__fp16*)out2, acc2);
        svst1_f16(pg, (__fp16*)out3, acc3);
        svst1_f16(pg, (__fp16*)out4, acc4);
        svst1_f16(pg, (__fp16*)out5, acc5);
    }
}

template <bool ZeroMode, bool Alpha1>
MLAS_SVE_TARGET MLAS_FORCEINLINE void
hprocessrows_4(
    const _mlas_fp16_* __restrict__ a,
    const _mlas_fp16_* __restrict__ b,
    _mlas_fp16_* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl
)
{
    const svfloat16_t zero_vec = svdup_n_f16((__fp16)0.f);
    const __fp16 halpha = (__fp16)alpha;

    size_t col = 0;
    for (; col + 2 * vl <= n; col += 2 * vl) {
        svbool_t pg0 = svwhilelt_b16(col, n);
        svbool_t pg1 = svwhilelt_b16(col + vl, n);

        _mlas_fp16_* out0_0 = res + 0 * ldc + col;
        _mlas_fp16_* out0_1 = res + 0 * ldc + col + vl;
        _mlas_fp16_* out1_0 = res + 1 * ldc + col;
        _mlas_fp16_* out1_1 = res + 1 * ldc + col + vl;
        _mlas_fp16_* out2_0 = res + 2 * ldc + col;
        _mlas_fp16_* out2_1 = res + 2 * ldc + col + vl;
        _mlas_fp16_* out3_0 = res + 3 * ldc + col;
        _mlas_fp16_* out3_1 = res + 3 * ldc + col + vl;

        svfloat16_t acc0_0 = zero_vec, acc0_1 = zero_vec;
        svfloat16_t acc1_0 = zero_vec, acc1_1 = zero_vec;
        svfloat16_t acc2_0 = zero_vec, acc2_1 = zero_vec;
        svfloat16_t acc3_0 = zero_vec, acc3_1 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += hk_step) {
            size_t k_max = std::min(k_block + hk_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const _mlas_fp16_* b0_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
                const _mlas_fp16_* b1_ptr = b0_ptr + vl;
                svfloat16_t b0 = svld1_f16(pg0, (const __fp16*)b0_ptr);
                svfloat16_t b1 = svld1_f16(pg1, (const __fp16*)b1_ptr);

                __fp16 a0 = reinterpret_cast<const __fp16*>(a)[0 * lda + p];
                __fp16 a1 = reinterpret_cast<const __fp16*>(a)[1 * lda + p];
                __fp16 a2 = reinterpret_cast<const __fp16*>(a)[2 * lda + p];
                __fp16 a3 = reinterpret_cast<const __fp16*>(a)[3 * lda + p];

                if constexpr (!Alpha1) {
                    a0 *= halpha; a1 *= halpha; a2 *= halpha; a3 *= halpha;
                }

                svfloat16_t va0 = svdup_n_f16(a0);
                svfloat16_t va1 = svdup_n_f16(a1);
                svfloat16_t va2 = svdup_n_f16(a2);
                svfloat16_t va3 = svdup_n_f16(a3);

                acc0_0 = svmla_f16_m(pg0, acc0_0, b0, va0);
                acc0_1 = svmla_f16_m(pg1, acc0_1, b1, va0);
                acc1_0 = svmla_f16_m(pg0, acc1_0, b0, va1);
                acc1_1 = svmla_f16_m(pg1, acc1_1, b1, va1);
                acc2_0 = svmla_f16_m(pg0, acc2_0, b0, va2);
                acc2_1 = svmla_f16_m(pg1, acc2_1, b1, va2);
                acc3_0 = svmla_f16_m(pg0, acc3_0, b0, va3);
                acc3_1 = svmla_f16_m(pg1, acc3_1, b1, va3);
            }
        }

        if constexpr (!ZeroMode) {
            acc0_0 = svadd_f16_m(pg0, acc0_0, svld1_f16(pg0, (const __fp16*)out0_0));
            acc0_1 = svadd_f16_m(pg1, acc0_1, svld1_f16(pg1, (const __fp16*)out0_1));
            acc1_0 = svadd_f16_m(pg0, acc1_0, svld1_f16(pg0, (const __fp16*)out1_0));
            acc1_1 = svadd_f16_m(pg1, acc1_1, svld1_f16(pg1, (const __fp16*)out1_1));
            acc2_0 = svadd_f16_m(pg0, acc2_0, svld1_f16(pg0, (const __fp16*)out2_0));
            acc2_1 = svadd_f16_m(pg1, acc2_1, svld1_f16(pg1, (const __fp16*)out2_1));
            acc3_0 = svadd_f16_m(pg0, acc3_0, svld1_f16(pg0, (const __fp16*)out3_0));
            acc3_1 = svadd_f16_m(pg1, acc3_1, svld1_f16(pg1, (const __fp16*)out3_1));
        }

        svst1_f16(pg0, (__fp16*)out0_0, acc0_0);
        svst1_f16(pg1, (__fp16*)out0_1, acc0_1);
        svst1_f16(pg0, (__fp16*)out1_0, acc1_0);
        svst1_f16(pg1, (__fp16*)out1_1, acc1_1);
        svst1_f16(pg0, (__fp16*)out2_0, acc2_0);
        svst1_f16(pg1, (__fp16*)out2_1, acc2_1);
        svst1_f16(pg0, (__fp16*)out3_0, acc3_0);
        svst1_f16(pg1, (__fp16*)out3_1, acc3_1);
    }

    for (; col < n; col += vl) {
        svbool_t pg = svwhilelt_b16(col, n);
        _mlas_fp16_* out0 = res + 0 * ldc + col;
        _mlas_fp16_* out1 = res + 1 * ldc + col;
        _mlas_fp16_* out2 = res + 2 * ldc + col;
        _mlas_fp16_* out3 = res + 3 * ldc + col;

        svfloat16_t acc0 = zero_vec, acc1 = zero_vec, acc2 = zero_vec, acc3 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += hk_step) {
            size_t k_max = std::min(k_block + hk_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const _mlas_fp16_* b_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
                svfloat16_t b0 = svld1_f16(pg, (const __fp16*)b_ptr);

                __fp16 a0 = reinterpret_cast<const __fp16*>(a)[0 * lda + p];
                __fp16 a1 = reinterpret_cast<const __fp16*>(a)[1 * lda + p];
                __fp16 a2 = reinterpret_cast<const __fp16*>(a)[2 * lda + p];
                __fp16 a3 = reinterpret_cast<const __fp16*>(a)[3 * lda + p];
                if constexpr (!Alpha1) {
                    a0 *= halpha; a1 *= halpha; a2 *= halpha; a3 *= halpha;
                }
                svfloat16_t va0 = svdup_n_f16(a0);
                svfloat16_t va1 = svdup_n_f16(a1);
                svfloat16_t va2 = svdup_n_f16(a2);
                svfloat16_t va3 = svdup_n_f16(a3);
                acc0 = svmla_f16_m(pg, acc0, b0, va0);
                acc1 = svmla_f16_m(pg, acc1, b0, va1);
                acc2 = svmla_f16_m(pg, acc2, b0, va2);
                acc3 = svmla_f16_m(pg, acc3, b0, va3);
            }
        }

        if constexpr (!ZeroMode) {
            acc0 = svadd_f16_m(pg, acc0, svld1_f16(pg, (const __fp16*)out0));
            acc1 = svadd_f16_m(pg, acc1, svld1_f16(pg, (const __fp16*)out1));
            acc2 = svadd_f16_m(pg, acc2, svld1_f16(pg, (const __fp16*)out2));
            acc3 = svadd_f16_m(pg, acc3, svld1_f16(pg, (const __fp16*)out3));
        }

        svst1_f16(pg, (__fp16*)out0, acc0);
        svst1_f16(pg, (__fp16*)out1, acc1);
        svst1_f16(pg, (__fp16*)out2, acc2);
        svst1_f16(pg, (__fp16*)out3, acc3);
    }
}

template <bool ZeroMode, bool Alpha1>
MLAS_SVE_TARGET MLAS_FORCEINLINE void
hprocessrows_2(
    const _mlas_fp16_* __restrict__ a,
    const _mlas_fp16_* __restrict__ b,
    _mlas_fp16_* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl
)
{
    const svfloat16_t zero_vec = svdup_n_f16((__fp16)0.f);
    const __fp16 halpha = (__fp16)alpha;

    size_t col = 0;
    for (; col + 2 * vl <= n; col += 2 * vl) {
        svbool_t pg0 = svwhilelt_b16(col, n);
        svbool_t pg1 = svwhilelt_b16(col + vl, n);

        _mlas_fp16_* out0_0 = res + 0 * ldc + col;
        _mlas_fp16_* out0_1 = res + 0 * ldc + col + vl;
        _mlas_fp16_* out1_0 = res + 1 * ldc + col;
        _mlas_fp16_* out1_1 = res + 1 * ldc + col + vl;

        svfloat16_t acc0_0 = zero_vec, acc0_1 = zero_vec;
        svfloat16_t acc1_0 = zero_vec, acc1_1 = zero_vec;

        for (size_t k_block = 0; k_block < k; k_block += hk_step) {
            size_t k_max = std::min(k_block + hk_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const _mlas_fp16_* b0_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
                const _mlas_fp16_* b1_ptr = b0_ptr + vl;
                svfloat16_t b0 = svld1_f16(pg0, (const __fp16*)b0_ptr);
                svfloat16_t b1 = svld1_f16(pg1, (const __fp16*)b1_ptr);

                __fp16 a0 = reinterpret_cast<const __fp16*>(a)[0 * lda + p];
                __fp16 a1 = reinterpret_cast<const __fp16*>(a)[1 * lda + p];
                if constexpr (!Alpha1) {
                    a0 *= halpha; a1 *= halpha;
                }
                svfloat16_t va0 = svdup_n_f16(a0);
                svfloat16_t va1 = svdup_n_f16(a1);
                acc0_0 = svmla_f16_m(pg0, acc0_0, b0, va0);
                acc0_1 = svmla_f16_m(pg1, acc0_1, b1, va0);
                acc1_0 = svmla_f16_m(pg0, acc1_0, b0, va1);
                acc1_1 = svmla_f16_m(pg1, acc1_1, b1, va1);
            }
        }
        if constexpr (!ZeroMode) {
            acc0_0 = svadd_f16_m(pg0, acc0_0, svld1_f16(pg0, (const __fp16*)out0_0));
            acc0_1 = svadd_f16_m(pg1, acc0_1, svld1_f16(pg1, (const __fp16*)out0_1));
            acc1_0 = svadd_f16_m(pg0, acc1_0, svld1_f16(pg0, (const __fp16*)out1_0));
            acc1_1 = svadd_f16_m(pg1, acc1_1, svld1_f16(pg1, (const __fp16*)out1_1));
        }
        svst1_f16(pg0, (__fp16*)out0_0, acc0_0);
        svst1_f16(pg1, (__fp16*)out0_1, acc0_1);
        svst1_f16(pg0, (__fp16*)out1_0, acc1_0);
        svst1_f16(pg1, (__fp16*)out1_1, acc1_1);
    }
    for (; col < n; col += vl) {
        svbool_t pg = svwhilelt_b16(col, n);
        _mlas_fp16_* out0 = res + 0 * ldc + col;
        _mlas_fp16_* out1 = res + 1 * ldc + col;
        svfloat16_t acc0 = zero_vec, acc1 = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += hk_step) {
            size_t k_max = std::min(k_block + hk_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const _mlas_fp16_* b_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
                svfloat16_t b0 = svld1_f16(pg, (const __fp16*)b_ptr);
                __fp16 a0 = reinterpret_cast<const __fp16*>(a)[0 * lda + p];
                __fp16 a1 = reinterpret_cast<const __fp16*>(a)[1 * lda + p];
                if constexpr (!Alpha1) {
                    a0 *= halpha; a1 *= halpha;
                }
                svfloat16_t va0 = svdup_n_f16(a0);
                svfloat16_t va1 = svdup_n_f16(a1);
                acc0 = svmla_f16_m(pg, acc0, b0, va0);
                acc1 = svmla_f16_m(pg, acc1, b0, va1);
            }
        }
        if constexpr (!ZeroMode) {
            acc0 = svadd_f16_m(pg, acc0, svld1_f16(pg, (const __fp16*)out0));
            acc1 = svadd_f16_m(pg, acc1, svld1_f16(pg, (const __fp16*)out1));
        }
        svst1_f16(pg, (__fp16*)out0, acc0);
        svst1_f16(pg, (__fp16*)out1, acc1);
    }
}

template <bool ZeroMode, bool Alpha1>
MLAS_SVE_TARGET MLAS_FORCEINLINE void
hprocessrows_1(
    const _mlas_fp16_* __restrict__ a,
    const _mlas_fp16_* __restrict__ b,
    _mlas_fp16_* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl
)
{
    MLAS_UNREFERENCED_PARAMETER(ldc);
    MLAS_UNREFERENCED_PARAMETER(lda);

    const svfloat16_t zero_vec = svdup_n_f16((__fp16)0.f);
    const __fp16 halpha = (__fp16)alpha;

    size_t col = 0;
    for (; col + 2 * vl <= n; col += 2 * vl) {
        svbool_t pg0 = svwhilelt_b16(col, n);
        svbool_t pg1 = svwhilelt_b16(col + vl, n);
        _mlas_fp16_* out0 = res + col;
        _mlas_fp16_* out1 = res + col + vl;
        svfloat16_t acc0 = zero_vec, acc1 = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += hk_step) {
            size_t k_max = std::min(k_block + hk_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const _mlas_fp16_* b0_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
                const _mlas_fp16_* b1_ptr = b0_ptr + vl;
                svfloat16_t b0 = svld1_f16(pg0, (const __fp16*)b0_ptr);
                svfloat16_t b1 = svld1_f16(pg1, (const __fp16*)b1_ptr);
                __fp16 a0 = reinterpret_cast<const __fp16*>(a)[p];
                if constexpr (!Alpha1)
                    a0 *= halpha;
                svfloat16_t va0 = svdup_n_f16(a0);
                acc0 = svmla_f16_m(pg0, acc0, b0, va0);
                acc1 = svmla_f16_m(pg1, acc1, b1, va0);
            }
        }
        if constexpr (!ZeroMode) {
            acc0 = svadd_f16_m(pg0, acc0, svld1_f16(pg0, (const __fp16*)out0));
            acc1 = svadd_f16_m(pg1, acc1, svld1_f16(pg1, (const __fp16*)out1));
        }
        svst1_f16(pg0, (__fp16*)out0, acc0);
        svst1_f16(pg1, (__fp16*)out1, acc1);
    }
    for (; col < n; col += vl) {
        svbool_t pg = svwhilelt_b16(col, n);
        _mlas_fp16_* out = res + col;
        svfloat16_t acc = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += hk_step) {
            size_t k_max = std::min(k_block + hk_step, k);
            for (size_t p = k_block; p < k_max; ++p) {
                const _mlas_fp16_* b_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
                svfloat16_t b0 = svld1_f16(pg, (const __fp16*)b_ptr);
                __fp16 a0 = reinterpret_cast<const __fp16*>(a)[p];
                if constexpr (!Alpha1)
                    a0 *= halpha;
                svfloat16_t va0 = svdup_n_f16(a0);
                acc = svmla_f16_m(pg, acc, b0, va0);
            }
        }
        if constexpr (!ZeroMode) {
            acc = svadd_f16_m(pg, acc, svld1_f16(pg, (const __fp16*)out));
        }
        svst1_f16(pg, (__fp16*)out, acc);
    }
}

#ifdef MLAS_HGEMM_ACCUMULATE_FP32
//
// Not used by default and not compile-tested.
//
template <bool ZeroMode, bool Alpha1>
MLAS_SVE_TARGET MLAS_FORCEINLINE void
hprocessrows_1_fp32acc(
    const _mlas_fp16_* __restrict__ a,
    const _mlas_fp16_* __restrict__ b,
    _mlas_fp16_* __restrict__ res,
    size_t k,
    size_t n,
    size_t lda,
    size_t ldc,
    float alpha,
    size_t vl   // fp16 lanes
)
{
    MLAS_UNREFERENCED_PARAMETER(ldc);
    MLAS_UNREFERENCED_PARAMETER(lda);

    const float falpha = alpha;

    for (size_t col = 0; col < n; col += vl) {
        svbool_t pg16 = svwhilelt_b16(col, n);
        _mlas_fp16_* out = res + col;

        svfloat32_t acc_lo = svdup_n_f32(0.f);
        svfloat32_t acc_hi = svdup_n_f32(0.f);

        for (size_t p = 0; p < k; ++p) {
            const _mlas_fp16_* b_ptr = b + p * PACKED_B_BLOCK_WIDTH_FP16 + col;
            svfloat16_t b0 = svld1_f16(pg16, (const __fp16*)b_ptr);

            float av = (float)reinterpret_cast<const __fp16*>(a)[p];
            if constexpr (!Alpha1) {
                av *= falpha;
            }
            svfloat32_t va = svdup_n_f32(av);

            svfloat32_t b_lo = svcvt_f32_f16_x(svptrue_b16(), svzip1_f16(b0, b0));
            svfloat32_t b_hi = svcvt_f32_f16_x(svptrue_b16(), svzip2_f16(b0, b0));

            acc_lo = svmla_f32_x(svptrue_b32(), acc_lo, b_lo, va);
            acc_hi = svmla_f32_x(svptrue_b32(), acc_hi, b_hi, va);
        }

        svfloat16_t res_lo = svcvt_f16_f32_x(svptrue_b32(), acc_lo);
        svfloat16_t res_hi = svcvt_f16_f32_x(svptrue_b32(), acc_hi);
        svfloat16_t result = svuzp1_f16(res_lo, res_hi);

        if constexpr (!ZeroMode) {
            result = svadd_f16_m(pg16, result, svld1_f16(pg16, (const __fp16*)out));
        }
        svst1_f16(pg16, (__fp16*)out, result);
    }
}
#endif // MLAS_HGEMM_ACCUMULATE_FP32

//
// The packer pads every N block to PACKED_B_BLOCK_WIDTH_FP16, so B advances by the full
// block width.
//
template <auto ProcessFn>
MLAS_SVE_TARGET MLAS_FORCEINLINE void
HProcessRowsTemplate(
    const _mlas_fp16_* __restrict__ A,
    size_t lda,
    const _mlas_fp16_* __restrict__ B,
    _mlas_fp16_* __restrict__ C,
    size_t ldc,
    size_t K,
    size_t N,
    float alpha
)
{
    size_t n = 0;
    const size_t vl = svcnth();
    while (n < N) {
        size_t cols = (n + PACKED_B_BLOCK_WIDTH_FP16 <= N)
                          ? (size_t)PACKED_B_BLOCK_WIDTH_FP16
                          : (N - n);
        ProcessFn(A, B, C, K, cols, lda, ldc, alpha, vl);
        B += (size_t)PACKED_B_BLOCK_WIDTH_FP16 * K;
        C += cols;
        n += cols;
    }
}

extern "C" size_t MLAS_SVE_TARGET MLASCALL
MlasHgemmKernelZero_sve(
    const _mlas_fp16_* A,
    const _mlas_fp16_* B,
    _mlas_fp16_* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
)
{
    if (alpha == 1.0f) {
        if (CountM >= 8) {
            HProcessRowsTemplate<hprocessrows_8<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 8;
        } else if (CountM >= 6) {
            HProcessRowsTemplate<hprocessrows_6<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 6;
        } else if (CountM >= 4) {
            HProcessRowsTemplate<hprocessrows_4<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 4;
        } else if (CountM >= 2) {
            HProcessRowsTemplate<hprocessrows_2<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 2;
        } else {
            HProcessRowsTemplate<hprocessrows_1<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 1;
        }
    } else {
        if (CountM >= 8) {
            HProcessRowsTemplate<hprocessrows_8<true, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 8;
        } else if (CountM >= 6) {
            HProcessRowsTemplate<hprocessrows_6<true, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 6;
        } else if (CountM >= 4) {
            HProcessRowsTemplate<hprocessrows_4<true, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 4;
        } else if (CountM >= 2) {
            HProcessRowsTemplate<hprocessrows_2<true, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 2;
        } else {
            HProcessRowsTemplate<hprocessrows_1<true, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 1;
        }
    }
}

extern "C" size_t MLAS_SVE_TARGET MLASCALL
MlasHgemmKernelAdd_sve(
    const _mlas_fp16_* A,
    const _mlas_fp16_* B,
    _mlas_fp16_* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
)
{
    if (alpha == 1.0f) {
        if (CountM >= 8) {
            HProcessRowsTemplate<hprocessrows_8<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 8;
        } else if (CountM >= 6) {
            HProcessRowsTemplate<hprocessrows_6<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 6;
        } else if (CountM >= 4) {
            HProcessRowsTemplate<hprocessrows_4<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 4;
        } else if (CountM >= 2) {
            HProcessRowsTemplate<hprocessrows_2<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 2;
        } else {
            HProcessRowsTemplate<hprocessrows_1<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 1;
        }
    } else {
        if (CountM >= 8) {
            HProcessRowsTemplate<hprocessrows_8<false, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 8;
        } else if (CountM >= 6) {
            HProcessRowsTemplate<hprocessrows_6<false, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 6;
        } else if (CountM >= 4) {
            HProcessRowsTemplate<hprocessrows_4<false, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 4;
        } else if (CountM >= 2) {
            HProcessRowsTemplate<hprocessrows_2<false, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 2;
        } else {
            HProcessRowsTemplate<hprocessrows_1<false, false>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 1;
        }
    }
}

extern "C" void MLAS_SVE_TARGET MLASCALL
MlasHgemmCopyPackB_sve(
    _mlas_fp16_* D,
    const _mlas_fp16_* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
)
{
    const size_t vl = svcnth();

    while (CountX >= PACKED_B_BLOCK_WIDTH_FP16) {
        const _mlas_fp16_* b = B;
        size_t y = CountY;
        do {
            for (size_t i = 0; i < PACKED_B_BLOCK_WIDTH_FP16; i += vl) {
                svbool_t pg = svwhilelt_b16(i, (size_t)PACKED_B_BLOCK_WIDTH_FP16);
                svfloat16_t v = svld1_f16(pg, (const __fp16*)b + i);
                svst1_f16(pg, (__fp16*)D + i, v);
            }
            D += PACKED_B_BLOCK_WIDTH_FP16;
            b += ldb;
            y--;
        } while (y > 0);

        B += PACKED_B_BLOCK_WIDTH_FP16;
        CountX -= PACKED_B_BLOCK_WIDTH_FP16;
    }

    if (CountX > 0) {
        size_t y = CountY;
        do {
            for (size_t i = 0; i < PACKED_B_BLOCK_WIDTH_FP16; i += vl) {
                svst1_f16(svptrue_b16(), (__fp16*)D + i, svdup_n_f16((__fp16)0.f));
            }
            for (size_t i = 0; i < CountX; i += vl) {
                svbool_t pg = svwhilelt_b16(i, CountX);
                svfloat16_t v = svld1_f16(pg, (const __fp16*)B + i);
                svst1_f16(pg, (__fp16*)D + i, v);
            }
            D += PACKED_B_BLOCK_WIDTH_FP16;
            B += ldb;
            y--;
        } while (y > 0);
    }
}

extern "C" void MLAS_SVE_TARGET MLASCALL
MlasHgemvFloat16Kernel_sve(
    const _mlas_fp16_* A,    // length-K row vector
    const _mlas_fp16_* B,    // K x N, row-major, row stride = ldb
    _mlas_fp16_* C,          // length-N output
    size_t CountK,
    size_t CountN,
    size_t ldb,
    bool ZeroMode
)
{
    const size_t vl = svcnth();
    size_t col = 0;

    const svbool_t pall = svptrue_b16();
    for (; col + 4 * vl <= CountN; col += 4 * vl) {
        svfloat16_t acc0, acc1, acc2, acc3;
        if (ZeroMode) {
            acc0 = svdup_n_f16((__fp16)0.f);
            acc1 = acc0;
            acc2 = acc0;
            acc3 = acc0;
        } else {
            acc0 = svld1_f16(pall, (const __fp16*)C + col);
            acc1 = svld1_f16(pall, (const __fp16*)C + col + vl);
            acc2 = svld1_f16(pall, (const __fp16*)C + col + 2 * vl);
            acc3 = svld1_f16(pall, (const __fp16*)C + col + 3 * vl);
        }

        const __fp16* b_row = reinterpret_cast<const __fp16*>(B) + col;
        for (size_t p = 0; p < CountK; ++p, b_row += ldb) {
            const svfloat16_t avec = svdup_n_f16(reinterpret_cast<const __fp16*>(A)[p]);
            acc0 = svmla_f16_x(pall, acc0, svld1_f16(pall, b_row), avec);
            acc1 = svmla_f16_x(pall, acc1, svld1_f16(pall, b_row + vl), avec);
            acc2 = svmla_f16_x(pall, acc2, svld1_f16(pall, b_row + 2 * vl), avec);
            acc3 = svmla_f16_x(pall, acc3, svld1_f16(pall, b_row + 3 * vl), avec);
        }

        svst1_f16(pall, (__fp16*)C + col, acc0);
        svst1_f16(pall, (__fp16*)C + col + vl, acc1);
        svst1_f16(pall, (__fp16*)C + col + 2 * vl, acc2);
        svst1_f16(pall, (__fp16*)C + col + 3 * vl, acc3);
    }

    for (; col < CountN; col += vl) {
        svbool_t pg = svwhilelt_b16(col, CountN);

        svfloat16_t acc;
        if (ZeroMode) {
            acc = svdup_n_f16((__fp16)0.f);
        } else {
            acc = svld1_f16(pg, (const __fp16*)C + col);
        }

        for (size_t p = 0; p < CountK; ++p) {
            const _mlas_fp16_* b_row = B + p * ldb + col;
            svfloat16_t bvec = svld1_f16(pg, (const __fp16*)b_row);
            svfloat16_t avec = svdup_n_f16(reinterpret_cast<const __fp16*>(A)[p]);
            acc = svmla_f16_m(pg, acc, bvec, avec);
        }

        svst1_f16(pg, (__fp16*)C + col, acc);
    }
}

extern "C" void MLAS_SVE_TARGET MLASCALL
MlasHgemv2Float16Kernel_sve(
    const _mlas_fp16_* A,    // 2 x K, row stride lda
    size_t lda,
    const _mlas_fp16_* B,    // K x N, row-major, row stride = ldb
    _mlas_fp16_* C,          // 2 x N, row stride ldc
    size_t ldc,
    size_t CountK,
    size_t CountN,
    size_t ldb,
    bool ZeroMode
)
{
    const size_t vl = svcnth();
    const svbool_t pall = svptrue_b16();
    const __fp16* a0 = reinterpret_cast<const __fp16*>(A);
    const __fp16* a1 = a0 + lda;
    __fp16* c0 = reinterpret_cast<__fp16*>(C);
    __fp16* c1 = c0 + ldc;
    size_t col = 0;

    for (; col + 4 * vl <= CountN; col += 4 * vl) {
        svfloat16_t r00, r01, r02, r03, r10, r11, r12, r13;
        if (ZeroMode) {
            r00 = svdup_n_f16((__fp16)0.f);
            r01 = r00; r02 = r00; r03 = r00;
            r10 = r00; r11 = r00; r12 = r00; r13 = r00;
        } else {
            r00 = svld1_f16(pall, c0 + col);
            r01 = svld1_f16(pall, c0 + col + vl);
            r02 = svld1_f16(pall, c0 + col + 2 * vl);
            r03 = svld1_f16(pall, c0 + col + 3 * vl);
            r10 = svld1_f16(pall, c1 + col);
            r11 = svld1_f16(pall, c1 + col + vl);
            r12 = svld1_f16(pall, c1 + col + 2 * vl);
            r13 = svld1_f16(pall, c1 + col + 3 * vl);
        }

        const __fp16* b_row = reinterpret_cast<const __fp16*>(B) + col;
        for (size_t p = 0; p < CountK; ++p, b_row += ldb) {
            const svfloat16_t b0 = svld1_f16(pall, b_row);
            const svfloat16_t b1 = svld1_f16(pall, b_row + vl);
            const svfloat16_t b2 = svld1_f16(pall, b_row + 2 * vl);
            const svfloat16_t b3 = svld1_f16(pall, b_row + 3 * vl);
            const svfloat16_t x0 = svdup_n_f16(a0[p]);
            const svfloat16_t x1 = svdup_n_f16(a1[p]);
            r00 = svmla_f16_x(pall, r00, b0, x0);
            r01 = svmla_f16_x(pall, r01, b1, x0);
            r02 = svmla_f16_x(pall, r02, b2, x0);
            r03 = svmla_f16_x(pall, r03, b3, x0);
            r10 = svmla_f16_x(pall, r10, b0, x1);
            r11 = svmla_f16_x(pall, r11, b1, x1);
            r12 = svmla_f16_x(pall, r12, b2, x1);
            r13 = svmla_f16_x(pall, r13, b3, x1);
        }

        svst1_f16(pall, c0 + col, r00);
        svst1_f16(pall, c0 + col + vl, r01);
        svst1_f16(pall, c0 + col + 2 * vl, r02);
        svst1_f16(pall, c0 + col + 3 * vl, r03);
        svst1_f16(pall, c1 + col, r10);
        svst1_f16(pall, c1 + col + vl, r11);
        svst1_f16(pall, c1 + col + 2 * vl, r12);
        svst1_f16(pall, c1 + col + 3 * vl, r13);
    }

    for (; col < CountN; col += vl) {
        const svbool_t pg = svwhilelt_b16(col, CountN);
        svfloat16_t r0, r1;
        if (ZeroMode) {
            r0 = svdup_n_f16((__fp16)0.f);
            r1 = r0;
        } else {
            r0 = svld1_f16(pg, c0 + col);
            r1 = svld1_f16(pg, c1 + col);
        }
        const __fp16* b_row = reinterpret_cast<const __fp16*>(B) + col;
        for (size_t p = 0; p < CountK; ++p, b_row += ldb) {
            const svfloat16_t b = svld1_f16(pg, b_row);
            r0 = svmla_f16_m(pg, r0, b, svdup_n_f16(a0[p]));
            r1 = svmla_f16_m(pg, r1, b, svdup_n_f16(a1[p]));
        }
        svst1_f16(pg, c0 + col, r0);
        svst1_f16(pg, c1 + col, r1);
    }
}

// Requires svcnth() == 8.
MLAS_SVE_TARGET MLAS_FORCEINLINE void
MlasSveTranspose8x8Float16(_mlas_fp16_* dst, const _mlas_fp16_* src, size_t ldb)
{
    const svbool_t p = svptrue_b16();
    const __fp16* B = reinterpret_cast<const __fp16*>(src);

    svfloat16_t v1=svld1_f16(p,B+ldb*0), v2=svld1_f16(p,B+ldb*1), v4=svld1_f16(p,B+ldb*2), v5=svld1_f16(p,B+ldb*3);
    svfloat16_t v6=svld1_f16(p,B+ldb*4), v7=svld1_f16(p,B+ldb*5), v8=svld1_f16(p,B+ldb*6), v9=svld1_f16(p,B+ldb*7);

    svfloat16_t v3=svzip1_f16(v1,v6); v1=svzip2_f16(v1,v6);
    v6=svzip1_f16(v2,v7);             v2=svzip2_f16(v2,v7);
    v7=svzip1_f16(v4,v8);             v4=svzip2_f16(v4,v8);
    v8=svzip1_f16(v5,v9);             v5=svzip2_f16(v5,v9);

    v9=svzip1_f16(v3,v7);             v3=svzip2_f16(v3,v7);
    v7=svzip1_f16(v6,v8);             v6=svzip2_f16(v6,v8);
    v8=svzip1_f16(v1,v4);             v1=svzip2_f16(v1,v4);
    v4=svzip1_f16(v2,v5);             v2=svzip2_f16(v2,v5);

    v5=svzip1_f16(v9,v7);             v9=svzip2_f16(v9,v7);
    v7=svzip1_f16(v8,v4);             v8=svzip2_f16(v8,v4);
    v4=svzip1_f16(v3,v6);             v3=svzip2_f16(v3,v6);
    v6=svzip1_f16(v1,v2);             v1=svzip2_f16(v1,v2);

    __fp16* d = reinterpret_cast<__fp16*>(dst);
    const size_t s = PACKED_B_BLOCK_WIDTH_FP16;
    svst1_f16(p, d + 0*s, v5);  svst1_f16(p, d + 1*s, v9);
    svst1_f16(p, d + 2*s, v4);  svst1_f16(p, d + 3*s, v3);
    svst1_f16(p, d + 4*s, v7);  svst1_f16(p, d + 5*s, v8);
    svst1_f16(p, d + 6*s, v6);  svst1_f16(p, d + 7*s, v1);
}

// Requires svcnth() == 16.
MLAS_SVE_TARGET MLAS_FORCEINLINE void
MlasSveTranspose16x16Float16(_mlas_fp16_* dst, const _mlas_fp16_* src, size_t ldb)
{
    const svbool_t p = svptrue_b16();
    const __fp16* B = reinterpret_cast<const __fp16*>(src);

    svfloat16_t v1=svld1_f16(p,B+ldb*0),  v2=svld1_f16(p,B+ldb*1),  v3=svld1_f16(p,B+ldb*2),  v4=svld1_f16(p,B+ldb*3);
    svfloat16_t v5=svld1_f16(p,B+ldb*4),  v6=svld1_f16(p,B+ldb*5),  v7=svld1_f16(p,B+ldb*6),  v8=svld1_f16(p,B+ldb*7);
    svfloat16_t v9=svld1_f16(p,B+ldb*8),  v10=svld1_f16(p,B+ldb*9), v11=svld1_f16(p,B+ldb*10),v12=svld1_f16(p,B+ldb*11);
    svfloat16_t v13=svld1_f16(p,B+ldb*12),v14=svld1_f16(p,B+ldb*13),v15=svld1_f16(p,B+ldb*14),v16=svld1_f16(p,B+ldb*15);

    svfloat16_t v17=svzip1_f16(v1,v9),  v18=svzip2_f16(v1,v9);
    svfloat16_t v19=svzip1_f16(v2,v10), v20=svzip2_f16(v2,v10);
    svfloat16_t v21=svzip1_f16(v3,v11), v22=svzip2_f16(v3,v11);
    svfloat16_t v23=svzip1_f16(v4,v12), v24=svzip2_f16(v4,v12);
    svfloat16_t v25=svzip1_f16(v5,v13), v26=svzip2_f16(v5,v13);
    svfloat16_t v27=svzip1_f16(v6,v14), v28=svzip2_f16(v6,v14);
    svfloat16_t v29=svzip1_f16(v7,v15), v30=svzip2_f16(v7,v15);
    svfloat16_t v31=svzip1_f16(v8,v16), v32=svzip2_f16(v8,v16);

    v1=svzip1_f16(v17,v25);  v9=svzip2_f16(v17,v25);
    v2=svzip1_f16(v18,v26);  v10=svzip2_f16(v18,v26);
    v3=svzip1_f16(v19,v27);  v11=svzip2_f16(v19,v27);
    v4=svzip1_f16(v20,v28);  v12=svzip2_f16(v20,v28);
    v5=svzip1_f16(v21,v29);  v13=svzip2_f16(v21,v29);
    v6=svzip1_f16(v22,v30);  v14=svzip2_f16(v22,v30);
    v7=svzip1_f16(v23,v31);  v15=svzip2_f16(v23,v31);
    v8=svzip1_f16(v24,v32);  v16=svzip2_f16(v24,v32);

    v17=svzip1_f16(v1,v5);   v25=svzip2_f16(v1,v5);
    v18=svzip1_f16(v9,v13);  v26=svzip2_f16(v9,v13);
    v19=svzip1_f16(v2,v6);   v27=svzip2_f16(v2,v6);
    v20=svzip1_f16(v10,v14); v28=svzip2_f16(v10,v14);
    v21=svzip1_f16(v3,v7);   v29=svzip2_f16(v3,v7);
    v22=svzip1_f16(v11,v15); v30=svzip2_f16(v11,v15);
    v23=svzip1_f16(v4,v8);   v31=svzip2_f16(v4,v8);
    v24=svzip1_f16(v12,v16); v32=svzip2_f16(v12,v16);

    v1=svzip1_f16(v17,v21);  v9=svzip2_f16(v17,v21);
    v2=svzip1_f16(v25,v29);  v10=svzip2_f16(v25,v29);
    v3=svzip1_f16(v18,v22);  v11=svzip2_f16(v18,v22);
    v4=svzip1_f16(v26,v30);  v12=svzip2_f16(v26,v30);
    v5=svzip1_f16(v19,v23);  v13=svzip2_f16(v19,v23);
    v6=svzip1_f16(v27,v31);  v14=svzip2_f16(v27,v31);
    v7=svzip1_f16(v20,v24);  v15=svzip2_f16(v20,v24);
    v8=svzip1_f16(v28,v32);  v16=svzip2_f16(v28,v32);

    __fp16* d = reinterpret_cast<__fp16*>(dst);
    const size_t s = PACKED_B_BLOCK_WIDTH_FP16;
    svst1_f16(p, d + 0*s,  v1);  svst1_f16(p, d + 1*s,  v9);
    svst1_f16(p, d + 2*s,  v2);  svst1_f16(p, d + 3*s,  v10);
    svst1_f16(p, d + 4*s,  v3);  svst1_f16(p, d + 5*s,  v11);
    svst1_f16(p, d + 6*s,  v4);  svst1_f16(p, d + 7*s,  v12);
    svst1_f16(p, d + 8*s,  v5);  svst1_f16(p, d + 9*s,  v13);
    svst1_f16(p, d + 10*s, v6);  svst1_f16(p, d + 11*s, v14);
    svst1_f16(p, d + 12*s, v7);  svst1_f16(p, d + 13*s, v15);
    svst1_f16(p, d + 14*s, v8);  svst1_f16(p, d + 15*s, v16);
}

// Requires svcnth() == 32.
MLAS_SVE_TARGET MLAS_FORCEINLINE void
MlasSveTranspose32x32Float16(_mlas_fp16_* dst, const _mlas_fp16_* src, size_t ldb)
{
    const svbool_t p = svptrue_b16();
    const __fp16* B = reinterpret_cast<const __fp16*>(src);
    svfloat16_t v0 = svld1_f16(p, B + ldb * 0);
    svfloat16_t v1 = svld1_f16(p, B + ldb * 1);
    svfloat16_t v2 = svld1_f16(p, B + ldb * 2);
    svfloat16_t v3 = svld1_f16(p, B + ldb * 3);
    svfloat16_t v4 = svld1_f16(p, B + ldb * 4);
    svfloat16_t v5 = svld1_f16(p, B + ldb * 5);
    svfloat16_t v6 = svld1_f16(p, B + ldb * 6);
    svfloat16_t v7 = svld1_f16(p, B + ldb * 7);
    svfloat16_t v8 = svld1_f16(p, B + ldb * 8);
    svfloat16_t v9 = svld1_f16(p, B + ldb * 9);
    svfloat16_t v10 = svld1_f16(p, B + ldb * 10);
    svfloat16_t v11 = svld1_f16(p, B + ldb * 11);
    svfloat16_t v12 = svld1_f16(p, B + ldb * 12);
    svfloat16_t v13 = svld1_f16(p, B + ldb * 13);
    svfloat16_t v14 = svld1_f16(p, B + ldb * 14);
    svfloat16_t v15 = svld1_f16(p, B + ldb * 15);
    svfloat16_t v16 = svld1_f16(p, B + ldb * 16);
    svfloat16_t v17 = svld1_f16(p, B + ldb * 17);
    svfloat16_t v18 = svld1_f16(p, B + ldb * 18);
    svfloat16_t v19 = svld1_f16(p, B + ldb * 19);
    svfloat16_t v20 = svld1_f16(p, B + ldb * 20);
    svfloat16_t v21 = svld1_f16(p, B + ldb * 21);
    svfloat16_t v22 = svld1_f16(p, B + ldb * 22);
    svfloat16_t v23 = svld1_f16(p, B + ldb * 23);
    svfloat16_t v24 = svld1_f16(p, B + ldb * 24);
    svfloat16_t v25 = svld1_f16(p, B + ldb * 25);
    svfloat16_t v26 = svld1_f16(p, B + ldb * 26);
    svfloat16_t v27 = svld1_f16(p, B + ldb * 27);
    svfloat16_t v28 = svld1_f16(p, B + ldb * 28);
    svfloat16_t v29 = svld1_f16(p, B + ldb * 29);
    svfloat16_t v30 = svld1_f16(p, B + ldb * 30);
    svfloat16_t v31 = svld1_f16(p, B + ldb * 31);

    svfloat16_t a0 = svzip1_f16(v0, v16);
    v16 = svzip2_f16(v0, v16);
    v0 = a0;
    svfloat16_t a1 = svzip1_f16(v1, v17);
    v17 = svzip2_f16(v1, v17);
    v1 = a1;
    svfloat16_t a2 = svzip1_f16(v2, v18);
    v18 = svzip2_f16(v2, v18);
    v2 = a2;
    svfloat16_t a3 = svzip1_f16(v3, v19);
    v19 = svzip2_f16(v3, v19);
    v3 = a3;
    svfloat16_t a4 = svzip1_f16(v4, v20);
    v20 = svzip2_f16(v4, v20);
    v4 = a4;
    svfloat16_t a5 = svzip1_f16(v5, v21);
    v21 = svzip2_f16(v5, v21);
    v5 = a5;
    svfloat16_t a6 = svzip1_f16(v6, v22);
    v22 = svzip2_f16(v6, v22);
    v6 = a6;
    svfloat16_t a7 = svzip1_f16(v7, v23);
    v23 = svzip2_f16(v7, v23);
    v7 = a7;
    svfloat16_t a8 = svzip1_f16(v8, v24);
    v24 = svzip2_f16(v8, v24);
    v8 = a8;
    svfloat16_t a9 = svzip1_f16(v9, v25);
    v25 = svzip2_f16(v9, v25);
    v9 = a9;
    svfloat16_t a10 = svzip1_f16(v10, v26);
    v26 = svzip2_f16(v10, v26);
    v10 = a10;
    svfloat16_t a11 = svzip1_f16(v11, v27);
    v27 = svzip2_f16(v11, v27);
    v11 = a11;
    svfloat16_t a12 = svzip1_f16(v12, v28);
    v28 = svzip2_f16(v12, v28);
    v12 = a12;
    svfloat16_t a13 = svzip1_f16(v13, v29);
    v29 = svzip2_f16(v13, v29);
    v13 = a13;
    svfloat16_t a14 = svzip1_f16(v14, v30);
    v30 = svzip2_f16(v14, v30);
    v14 = a14;
    svfloat16_t a15 = svzip1_f16(v15, v31);
    v31 = svzip2_f16(v15, v31);
    v15 = a15;
    svfloat16_t a16 = svzip1_f16(v0, v8);
    v8 = svzip2_f16(v0, v8);
    v0 = a16;
    svfloat16_t a17 = svzip1_f16(v1, v9);
    v9 = svzip2_f16(v1, v9);
    v1 = a17;
    svfloat16_t a18 = svzip1_f16(v2, v10);
    v10 = svzip2_f16(v2, v10);
    v2 = a18;
    svfloat16_t a19 = svzip1_f16(v3, v11);
    v11 = svzip2_f16(v3, v11);
    v3 = a19;
    svfloat16_t a20 = svzip1_f16(v4, v12);
    v12 = svzip2_f16(v4, v12);
    v4 = a20;
    svfloat16_t a21 = svzip1_f16(v5, v13);
    v13 = svzip2_f16(v5, v13);
    v5 = a21;
    svfloat16_t a22 = svzip1_f16(v6, v14);
    v14 = svzip2_f16(v6, v14);
    v6 = a22;
    svfloat16_t a23 = svzip1_f16(v7, v15);
    v15 = svzip2_f16(v7, v15);
    v7 = a23;
    svfloat16_t a24 = svzip1_f16(v16, v24);
    v24 = svzip2_f16(v16, v24);
    v16 = a24;
    svfloat16_t a25 = svzip1_f16(v17, v25);
    v25 = svzip2_f16(v17, v25);
    v17 = a25;
    svfloat16_t a26 = svzip1_f16(v18, v26);
    v26 = svzip2_f16(v18, v26);
    v18 = a26;
    svfloat16_t a27 = svzip1_f16(v19, v27);
    v27 = svzip2_f16(v19, v27);
    v19 = a27;
    svfloat16_t a28 = svzip1_f16(v20, v28);
    v28 = svzip2_f16(v20, v28);
    v20 = a28;
    svfloat16_t a29 = svzip1_f16(v21, v29);
    v29 = svzip2_f16(v21, v29);
    v21 = a29;
    svfloat16_t a30 = svzip1_f16(v22, v30);
    v30 = svzip2_f16(v22, v30);
    v22 = a30;
    svfloat16_t a31 = svzip1_f16(v23, v31);
    v31 = svzip2_f16(v23, v31);
    v23 = a31;
    svfloat16_t a32 = svzip1_f16(v0, v4);
    v4 = svzip2_f16(v0, v4);
    v0 = a32;
    svfloat16_t a33 = svzip1_f16(v1, v5);
    v5 = svzip2_f16(v1, v5);
    v1 = a33;
    svfloat16_t a34 = svzip1_f16(v2, v6);
    v6 = svzip2_f16(v2, v6);
    v2 = a34;
    svfloat16_t a35 = svzip1_f16(v3, v7);
    v7 = svzip2_f16(v3, v7);
    v3 = a35;
    svfloat16_t a36 = svzip1_f16(v8, v12);
    v12 = svzip2_f16(v8, v12);
    v8 = a36;
    svfloat16_t a37 = svzip1_f16(v9, v13);
    v13 = svzip2_f16(v9, v13);
    v9 = a37;
    svfloat16_t a38 = svzip1_f16(v10, v14);
    v14 = svzip2_f16(v10, v14);
    v10 = a38;
    svfloat16_t a39 = svzip1_f16(v11, v15);
    v15 = svzip2_f16(v11, v15);
    v11 = a39;
    svfloat16_t a40 = svzip1_f16(v16, v20);
    v20 = svzip2_f16(v16, v20);
    v16 = a40;
    svfloat16_t a41 = svzip1_f16(v17, v21);
    v21 = svzip2_f16(v17, v21);
    v17 = a41;
    svfloat16_t a42 = svzip1_f16(v18, v22);
    v22 = svzip2_f16(v18, v22);
    v18 = a42;
    svfloat16_t a43 = svzip1_f16(v19, v23);
    v23 = svzip2_f16(v19, v23);
    v19 = a43;
    svfloat16_t a44 = svzip1_f16(v24, v28);
    v28 = svzip2_f16(v24, v28);
    v24 = a44;
    svfloat16_t a45 = svzip1_f16(v25, v29);
    v29 = svzip2_f16(v25, v29);
    v25 = a45;
    svfloat16_t a46 = svzip1_f16(v26, v30);
    v30 = svzip2_f16(v26, v30);
    v26 = a46;
    svfloat16_t a47 = svzip1_f16(v27, v31);
    v31 = svzip2_f16(v27, v31);
    v27 = a47;
    svfloat16_t a48 = svzip1_f16(v0, v2);
    v2 = svzip2_f16(v0, v2);
    v0 = a48;
    svfloat16_t a49 = svzip1_f16(v1, v3);
    v3 = svzip2_f16(v1, v3);
    v1 = a49;
    svfloat16_t a50 = svzip1_f16(v4, v6);
    v6 = svzip2_f16(v4, v6);
    v4 = a50;
    svfloat16_t a51 = svzip1_f16(v5, v7);
    v7 = svzip2_f16(v5, v7);
    v5 = a51;
    svfloat16_t a52 = svzip1_f16(v8, v10);
    v10 = svzip2_f16(v8, v10);
    v8 = a52;
    svfloat16_t a53 = svzip1_f16(v9, v11);
    v11 = svzip2_f16(v9, v11);
    v9 = a53;
    svfloat16_t a54 = svzip1_f16(v12, v14);
    v14 = svzip2_f16(v12, v14);
    v12 = a54;
    svfloat16_t a55 = svzip1_f16(v13, v15);
    v15 = svzip2_f16(v13, v15);
    v13 = a55;
    svfloat16_t a56 = svzip1_f16(v16, v18);
    v18 = svzip2_f16(v16, v18);
    v16 = a56;
    svfloat16_t a57 = svzip1_f16(v17, v19);
    v19 = svzip2_f16(v17, v19);
    v17 = a57;
    svfloat16_t a58 = svzip1_f16(v20, v22);
    v22 = svzip2_f16(v20, v22);
    v20 = a58;
    svfloat16_t a59 = svzip1_f16(v21, v23);
    v23 = svzip2_f16(v21, v23);
    v21 = a59;
    svfloat16_t a60 = svzip1_f16(v24, v26);
    v26 = svzip2_f16(v24, v26);
    v24 = a60;
    svfloat16_t a61 = svzip1_f16(v25, v27);
    v27 = svzip2_f16(v25, v27);
    v25 = a61;
    svfloat16_t a62 = svzip1_f16(v28, v30);
    v30 = svzip2_f16(v28, v30);
    v28 = a62;
    svfloat16_t a63 = svzip1_f16(v29, v31);
    v31 = svzip2_f16(v29, v31);
    v29 = a63;
    svfloat16_t a64 = svzip1_f16(v0, v1);
    v1 = svzip2_f16(v0, v1);
    v0 = a64;
    svfloat16_t a65 = svzip1_f16(v2, v3);
    v3 = svzip2_f16(v2, v3);
    v2 = a65;
    svfloat16_t a66 = svzip1_f16(v4, v5);
    v5 = svzip2_f16(v4, v5);
    v4 = a66;
    svfloat16_t a67 = svzip1_f16(v6, v7);
    v7 = svzip2_f16(v6, v7);
    v6 = a67;
    svfloat16_t a68 = svzip1_f16(v8, v9);
    v9 = svzip2_f16(v8, v9);
    v8 = a68;
    svfloat16_t a69 = svzip1_f16(v10, v11);
    v11 = svzip2_f16(v10, v11);
    v10 = a69;
    svfloat16_t a70 = svzip1_f16(v12, v13);
    v13 = svzip2_f16(v12, v13);
    v12 = a70;
    svfloat16_t a71 = svzip1_f16(v14, v15);
    v15 = svzip2_f16(v14, v15);
    v14 = a71;
    svfloat16_t a72 = svzip1_f16(v16, v17);
    v17 = svzip2_f16(v16, v17);
    v16 = a72;
    svfloat16_t a73 = svzip1_f16(v18, v19);
    v19 = svzip2_f16(v18, v19);
    v18 = a73;
    svfloat16_t a74 = svzip1_f16(v20, v21);
    v21 = svzip2_f16(v20, v21);
    v20 = a74;
    svfloat16_t a75 = svzip1_f16(v22, v23);
    v23 = svzip2_f16(v22, v23);
    v22 = a75;
    svfloat16_t a76 = svzip1_f16(v24, v25);
    v25 = svzip2_f16(v24, v25);
    v24 = a76;
    svfloat16_t a77 = svzip1_f16(v26, v27);
    v27 = svzip2_f16(v26, v27);
    v26 = a77;
    svfloat16_t a78 = svzip1_f16(v28, v29);
    v29 = svzip2_f16(v28, v29);
    v28 = a78;
    svfloat16_t a79 = svzip1_f16(v30, v31);
    v31 = svzip2_f16(v30, v31);
    v30 = a79;

    __fp16* d = reinterpret_cast<__fp16*>(dst);
    const size_t s = PACKED_B_BLOCK_WIDTH_FP16;
    svst1_f16(p, d + 0*s, v0);
    svst1_f16(p, d + 1*s, v1);
    svst1_f16(p, d + 2*s, v2);
    svst1_f16(p, d + 3*s, v3);
    svst1_f16(p, d + 4*s, v4);
    svst1_f16(p, d + 5*s, v5);
    svst1_f16(p, d + 6*s, v6);
    svst1_f16(p, d + 7*s, v7);
    svst1_f16(p, d + 8*s, v8);
    svst1_f16(p, d + 9*s, v9);
    svst1_f16(p, d + 10*s, v10);
    svst1_f16(p, d + 11*s, v11);
    svst1_f16(p, d + 12*s, v12);
    svst1_f16(p, d + 13*s, v13);
    svst1_f16(p, d + 14*s, v14);
    svst1_f16(p, d + 15*s, v15);
    svst1_f16(p, d + 16*s, v16);
    svst1_f16(p, d + 17*s, v17);
    svst1_f16(p, d + 18*s, v18);
    svst1_f16(p, d + 19*s, v19);
    svst1_f16(p, d + 20*s, v20);
    svst1_f16(p, d + 21*s, v21);
    svst1_f16(p, d + 22*s, v22);
    svst1_f16(p, d + 23*s, v23);
    svst1_f16(p, d + 24*s, v24);
    svst1_f16(p, d + 25*s, v25);
    svst1_f16(p, d + 26*s, v26);
    svst1_f16(p, d + 27*s, v27);
    svst1_f16(p, d + 28*s, v28);
    svst1_f16(p, d + 29*s, v29);
    svst1_f16(p, d + 30*s, v30);
    svst1_f16(p, d + 31*s, v31);
}

template <size_t VL>
MLAS_SVE_TARGET MLAS_FORCEINLINE static void
HTransposePackBImpl(
    _mlas_fp16_* D, const _mlas_fp16_* B, size_t ldb, size_t CountN, size_t CountK)
{
    const size_t BW = PACKED_B_BLOCK_WIDTH_FP16;
    for (size_t n0 = 0; n0 < CountN; n0 += BW) {
        const size_t block = std::min(BW, CountN - n0);
        _mlas_fp16_* Dblk = D;

        for (size_t h = 0; h * VL < block; ++h) {
            const size_t nb = n0 + h * VL;
            const size_t nrows = std::min(VL, block - h * VL);
            size_t k = 0;
            if (nrows == VL) {
                for (; k + VL <= CountK; k += VL) {
                    _mlas_fp16_* dst = Dblk + k * BW + h * VL;
                    const _mlas_fp16_* src = B + nb * ldb + k;
                    if constexpr (VL == 8) {
                        MlasSveTranspose8x8Float16(dst, src, ldb);
                    } else if constexpr (VL == 16) {
                        MlasSveTranspose16x16Float16(dst, src, ldb);
                    } else if constexpr (VL == 32) {
                        MlasSveTranspose32x32Float16(dst, src, ldb);
                    }
                }
            }
            for (; k < CountK; ++k) {
                _mlas_fp16_* d = Dblk + k * BW + h * VL;
                for (size_t j = 0; j < nrows; ++j) {
                    d[j] = B[(nb + j) * ldb + k];
                }
            }
        }

        for (size_t k = 0; k < CountK; ++k) {
            _mlas_fp16_* d = Dblk + k * BW;
            for (size_t j = block; j < BW; ++j) {
                d[j] = (_mlas_fp16_)0;
            }
        }
        D += BW * CountK;
    }
}

MLAS_SVE_TARGET MLAS_FORCEINLINE static void
HTransposePackBScalar(
    _mlas_fp16_* D, const _mlas_fp16_* B, size_t ldb, size_t CountN, size_t CountK)
{
    const size_t BW = PACKED_B_BLOCK_WIDTH_FP16;
    for (size_t n0 = 0; n0 < CountN; n0 += BW) {
        const size_t block = std::min(BW, CountN - n0);
        for (size_t k = 0; k < CountK; ++k) {
            _mlas_fp16_* d = D + k * BW;
            for (size_t j = 0; j < block; ++j) {
                d[j] = B[(n0 + j) * ldb + k];
            }
            for (size_t j = block; j < BW; ++j) {
                d[j] = (_mlas_fp16_)0;
            }
        }
        D += BW * CountK;
    }
}

extern "C" void MLAS_SVE_TARGET MLASCALL
MlasHgemmTransposePackB_sve(
    _mlas_fp16_* D,
    const _mlas_fp16_* B,
    size_t ldb,
    size_t CountY,  // CountN
    size_t CountX  // CountK
)
{
    switch (svcnth()) {
        case 8:
            HTransposePackBImpl<8>(D, B, ldb, CountY, CountX);
            break;
        case 16:
            HTransposePackBImpl<16>(D, B, ldb, CountY, CountX);
            break;
        case 32:
            HTransposePackBImpl<32>(D, B, ldb, CountY, CountX);
            break;
        default:
            HTransposePackBScalar(D, B, ldb, CountY, CountX);
            break;
    }
}

// The M loads are predicated to rows, so lda can be smaller than VL.
MLAS_SVE_TARGET MLAS_FORCEINLINE void
HTransposeATile8(__fp16* d, const __fp16* a, size_t lda, size_t CountX, size_t rows)
{
    const svbool_t pM = svwhilelt_b16((uint64_t)0, (uint64_t)rows);
    const svbool_t pK = svptrue_b16();
    svfloat16_t v1=svld1_f16(pM,a+lda*0), v2=svld1_f16(pM,a+lda*1), v4=svld1_f16(pM,a+lda*2), v5=svld1_f16(pM,a+lda*3);
    svfloat16_t v6=svld1_f16(pM,a+lda*4), v7=svld1_f16(pM,a+lda*5), v8=svld1_f16(pM,a+lda*6), v9=svld1_f16(pM,a+lda*7);
    svfloat16_t v3=svzip1_f16(v1,v6); v1=svzip2_f16(v1,v6);
    v6=svzip1_f16(v2,v7);             v2=svzip2_f16(v2,v7);
    v7=svzip1_f16(v4,v8);             v4=svzip2_f16(v4,v8);
    v8=svzip1_f16(v5,v9);             v5=svzip2_f16(v5,v9);
    v9=svzip1_f16(v3,v7);             v3=svzip2_f16(v3,v7);
    v7=svzip1_f16(v6,v8);             v6=svzip2_f16(v6,v8);
    v8=svzip1_f16(v1,v4);             v1=svzip2_f16(v1,v4);
    v4=svzip1_f16(v2,v5);             v2=svzip2_f16(v2,v5);
    v5=svzip1_f16(v9,v7);             v9=svzip2_f16(v9,v7);
    v7=svzip1_f16(v8,v4);             v8=svzip2_f16(v8,v4);
    v4=svzip1_f16(v3,v6);             v3=svzip2_f16(v3,v6);
    v6=svzip1_f16(v1,v2);             v1=svzip2_f16(v1,v2);
    if(rows>0) svst1_f16(pK, d+0*CountX, v5);
    if(rows>1) svst1_f16(pK, d+1*CountX, v9);
    if(rows>2) svst1_f16(pK, d+2*CountX, v4);
    if(rows>3) svst1_f16(pK, d+3*CountX, v3);
    if(rows>4) svst1_f16(pK, d+4*CountX, v7);
    if(rows>5) svst1_f16(pK, d+5*CountX, v8);
    if(rows>6) svst1_f16(pK, d+6*CountX, v6);
    if(rows>7) svst1_f16(pK, d+7*CountX, v1);
}

MLAS_SVE_TARGET MLAS_FORCEINLINE void
HTransposeATile16(__fp16* d, const __fp16* a, size_t lda, size_t CountX, size_t rows)
{
    const svbool_t pM = svwhilelt_b16((uint64_t)0, (uint64_t)rows);
    const svbool_t pK = svptrue_b16();
    svfloat16_t v1=svld1_f16(pM,a+lda*0),  v2=svld1_f16(pM,a+lda*1),  v3=svld1_f16(pM,a+lda*2),  v4=svld1_f16(pM,a+lda*3);
    svfloat16_t v5=svld1_f16(pM,a+lda*4),  v6=svld1_f16(pM,a+lda*5),  v7=svld1_f16(pM,a+lda*6),  v8=svld1_f16(pM,a+lda*7);
    svfloat16_t v9=svld1_f16(pM,a+lda*8),  v10=svld1_f16(pM,a+lda*9), v11=svld1_f16(pM,a+lda*10),v12=svld1_f16(pM,a+lda*11);
    svfloat16_t v13=svld1_f16(pM,a+lda*12),v14=svld1_f16(pM,a+lda*13),v15=svld1_f16(pM,a+lda*14),v16=svld1_f16(pM,a+lda*15);

    svfloat16_t v17=svzip1_f16(v1,v9),  v18=svzip2_f16(v1,v9);
    svfloat16_t v19=svzip1_f16(v2,v10), v20=svzip2_f16(v2,v10);
    svfloat16_t v21=svzip1_f16(v3,v11), v22=svzip2_f16(v3,v11);
    svfloat16_t v23=svzip1_f16(v4,v12), v24=svzip2_f16(v4,v12);
    svfloat16_t v25=svzip1_f16(v5,v13), v26=svzip2_f16(v5,v13);
    svfloat16_t v27=svzip1_f16(v6,v14), v28=svzip2_f16(v6,v14);
    svfloat16_t v29=svzip1_f16(v7,v15), v30=svzip2_f16(v7,v15);
    svfloat16_t v31=svzip1_f16(v8,v16), v32=svzip2_f16(v8,v16);

    v1=svzip1_f16(v17,v25);  v9=svzip2_f16(v17,v25);
    v2=svzip1_f16(v18,v26);  v10=svzip2_f16(v18,v26);
    v3=svzip1_f16(v19,v27);  v11=svzip2_f16(v19,v27);
    v4=svzip1_f16(v20,v28);  v12=svzip2_f16(v20,v28);
    v5=svzip1_f16(v21,v29);  v13=svzip2_f16(v21,v29);
    v6=svzip1_f16(v22,v30);  v14=svzip2_f16(v22,v30);
    v7=svzip1_f16(v23,v31);  v15=svzip2_f16(v23,v31);
    v8=svzip1_f16(v24,v32);  v16=svzip2_f16(v24,v32);

    v17=svzip1_f16(v1,v5);   v25=svzip2_f16(v1,v5);
    v18=svzip1_f16(v9,v13);  v26=svzip2_f16(v9,v13);
    v19=svzip1_f16(v2,v6);   v27=svzip2_f16(v2,v6);
    v20=svzip1_f16(v10,v14); v28=svzip2_f16(v10,v14);
    v21=svzip1_f16(v3,v7);   v29=svzip2_f16(v3,v7);
    v22=svzip1_f16(v11,v15); v30=svzip2_f16(v11,v15);
    v23=svzip1_f16(v4,v8);   v31=svzip2_f16(v4,v8);
    v24=svzip1_f16(v12,v16); v32=svzip2_f16(v12,v16);

    v1=svzip1_f16(v17,v21);  v9=svzip2_f16(v17,v21);
    v2=svzip1_f16(v25,v29);  v10=svzip2_f16(v25,v29);
    v3=svzip1_f16(v18,v22);  v11=svzip2_f16(v18,v22);
    v4=svzip1_f16(v26,v30);  v12=svzip2_f16(v26,v30);
    v5=svzip1_f16(v19,v23);  v13=svzip2_f16(v19,v23);
    v6=svzip1_f16(v27,v31);  v14=svzip2_f16(v27,v31);
    v7=svzip1_f16(v20,v24);  v15=svzip2_f16(v20,v24);
    v8=svzip1_f16(v28,v32);  v16=svzip2_f16(v28,v32);

    if(rows>0)  svst1_f16(pK, d+0*CountX,  v1);
    if(rows>1)  svst1_f16(pK, d+1*CountX,  v9);
    if(rows>2)  svst1_f16(pK, d+2*CountX,  v2);
    if(rows>3)  svst1_f16(pK, d+3*CountX,  v10);
    if(rows>4)  svst1_f16(pK, d+4*CountX,  v3);
    if(rows>5)  svst1_f16(pK, d+5*CountX,  v11);
    if(rows>6)  svst1_f16(pK, d+6*CountX,  v4);
    if(rows>7)  svst1_f16(pK, d+7*CountX,  v12);
    if(rows>8)  svst1_f16(pK, d+8*CountX,  v5);
    if(rows>9)  svst1_f16(pK, d+9*CountX,  v13);
    if(rows>10) svst1_f16(pK, d+10*CountX, v6);
    if(rows>11) svst1_f16(pK, d+11*CountX, v14);
    if(rows>12) svst1_f16(pK, d+12*CountX, v7);
    if(rows>13) svst1_f16(pK, d+13*CountX, v15);
    if(rows>14) svst1_f16(pK, d+14*CountX, v8);
    if(rows>15) svst1_f16(pK, d+15*CountX, v16);
}

MLAS_SVE_TARGET MLAS_FORCEINLINE void
HTransposeATile32(__fp16* d, const __fp16* a, size_t lda, size_t CountX, size_t rows)
{
    const svbool_t pM = svwhilelt_b16((uint64_t)0, (uint64_t)rows);
    const svbool_t pK = svptrue_b16();
    svfloat16_t v0=svld1_f16(pM,a+lda*0);
    svfloat16_t v1=svld1_f16(pM,a+lda*1);
    svfloat16_t v2=svld1_f16(pM,a+lda*2);
    svfloat16_t v3=svld1_f16(pM,a+lda*3);
    svfloat16_t v4=svld1_f16(pM,a+lda*4);
    svfloat16_t v5=svld1_f16(pM,a+lda*5);
    svfloat16_t v6=svld1_f16(pM,a+lda*6);
    svfloat16_t v7=svld1_f16(pM,a+lda*7);
    svfloat16_t v8=svld1_f16(pM,a+lda*8);
    svfloat16_t v9=svld1_f16(pM,a+lda*9);
    svfloat16_t v10=svld1_f16(pM,a+lda*10);
    svfloat16_t v11=svld1_f16(pM,a+lda*11);
    svfloat16_t v12=svld1_f16(pM,a+lda*12);
    svfloat16_t v13=svld1_f16(pM,a+lda*13);
    svfloat16_t v14=svld1_f16(pM,a+lda*14);
    svfloat16_t v15=svld1_f16(pM,a+lda*15);
    svfloat16_t v16=svld1_f16(pM,a+lda*16);
    svfloat16_t v17=svld1_f16(pM,a+lda*17);
    svfloat16_t v18=svld1_f16(pM,a+lda*18);
    svfloat16_t v19=svld1_f16(pM,a+lda*19);
    svfloat16_t v20=svld1_f16(pM,a+lda*20);
    svfloat16_t v21=svld1_f16(pM,a+lda*21);
    svfloat16_t v22=svld1_f16(pM,a+lda*22);
    svfloat16_t v23=svld1_f16(pM,a+lda*23);
    svfloat16_t v24=svld1_f16(pM,a+lda*24);
    svfloat16_t v25=svld1_f16(pM,a+lda*25);
    svfloat16_t v26=svld1_f16(pM,a+lda*26);
    svfloat16_t v27=svld1_f16(pM,a+lda*27);
    svfloat16_t v28=svld1_f16(pM,a+lda*28);
    svfloat16_t v29=svld1_f16(pM,a+lda*29);
    svfloat16_t v30=svld1_f16(pM,a+lda*30);
    svfloat16_t v31=svld1_f16(pM,a+lda*31);
    svfloat16_t a0 = svzip1_f16(v0, v16);
    v16 = svzip2_f16(v0, v16);
    v0 = a0;
    svfloat16_t a1 = svzip1_f16(v1, v17);
    v17 = svzip2_f16(v1, v17);
    v1 = a1;
    svfloat16_t a2 = svzip1_f16(v2, v18);
    v18 = svzip2_f16(v2, v18);
    v2 = a2;
    svfloat16_t a3 = svzip1_f16(v3, v19);
    v19 = svzip2_f16(v3, v19);
    v3 = a3;
    svfloat16_t a4 = svzip1_f16(v4, v20);
    v20 = svzip2_f16(v4, v20);
    v4 = a4;
    svfloat16_t a5 = svzip1_f16(v5, v21);
    v21 = svzip2_f16(v5, v21);
    v5 = a5;
    svfloat16_t a6 = svzip1_f16(v6, v22);
    v22 = svzip2_f16(v6, v22);
    v6 = a6;
    svfloat16_t a7 = svzip1_f16(v7, v23);
    v23 = svzip2_f16(v7, v23);
    v7 = a7;
    svfloat16_t a8 = svzip1_f16(v8, v24);
    v24 = svzip2_f16(v8, v24);
    v8 = a8;
    svfloat16_t a9 = svzip1_f16(v9, v25);
    v25 = svzip2_f16(v9, v25);
    v9 = a9;
    svfloat16_t a10 = svzip1_f16(v10, v26);
    v26 = svzip2_f16(v10, v26);
    v10 = a10;
    svfloat16_t a11 = svzip1_f16(v11, v27);
    v27 = svzip2_f16(v11, v27);
    v11 = a11;
    svfloat16_t a12 = svzip1_f16(v12, v28);
    v28 = svzip2_f16(v12, v28);
    v12 = a12;
    svfloat16_t a13 = svzip1_f16(v13, v29);
    v29 = svzip2_f16(v13, v29);
    v13 = a13;
    svfloat16_t a14 = svzip1_f16(v14, v30);
    v30 = svzip2_f16(v14, v30);
    v14 = a14;
    svfloat16_t a15 = svzip1_f16(v15, v31);
    v31 = svzip2_f16(v15, v31);
    v15 = a15;
    svfloat16_t a16 = svzip1_f16(v0, v8);
    v8 = svzip2_f16(v0, v8);
    v0 = a16;
    svfloat16_t a17 = svzip1_f16(v1, v9);
    v9 = svzip2_f16(v1, v9);
    v1 = a17;
    svfloat16_t a18 = svzip1_f16(v2, v10);
    v10 = svzip2_f16(v2, v10);
    v2 = a18;
    svfloat16_t a19 = svzip1_f16(v3, v11);
    v11 = svzip2_f16(v3, v11);
    v3 = a19;
    svfloat16_t a20 = svzip1_f16(v4, v12);
    v12 = svzip2_f16(v4, v12);
    v4 = a20;
    svfloat16_t a21 = svzip1_f16(v5, v13);
    v13 = svzip2_f16(v5, v13);
    v5 = a21;
    svfloat16_t a22 = svzip1_f16(v6, v14);
    v14 = svzip2_f16(v6, v14);
    v6 = a22;
    svfloat16_t a23 = svzip1_f16(v7, v15);
    v15 = svzip2_f16(v7, v15);
    v7 = a23;
    svfloat16_t a24 = svzip1_f16(v16, v24);
    v24 = svzip2_f16(v16, v24);
    v16 = a24;
    svfloat16_t a25 = svzip1_f16(v17, v25);
    v25 = svzip2_f16(v17, v25);
    v17 = a25;
    svfloat16_t a26 = svzip1_f16(v18, v26);
    v26 = svzip2_f16(v18, v26);
    v18 = a26;
    svfloat16_t a27 = svzip1_f16(v19, v27);
    v27 = svzip2_f16(v19, v27);
    v19 = a27;
    svfloat16_t a28 = svzip1_f16(v20, v28);
    v28 = svzip2_f16(v20, v28);
    v20 = a28;
    svfloat16_t a29 = svzip1_f16(v21, v29);
    v29 = svzip2_f16(v21, v29);
    v21 = a29;
    svfloat16_t a30 = svzip1_f16(v22, v30);
    v30 = svzip2_f16(v22, v30);
    v22 = a30;
    svfloat16_t a31 = svzip1_f16(v23, v31);
    v31 = svzip2_f16(v23, v31);
    v23 = a31;
    svfloat16_t a32 = svzip1_f16(v0, v4);
    v4 = svzip2_f16(v0, v4);
    v0 = a32;
    svfloat16_t a33 = svzip1_f16(v1, v5);
    v5 = svzip2_f16(v1, v5);
    v1 = a33;
    svfloat16_t a34 = svzip1_f16(v2, v6);
    v6 = svzip2_f16(v2, v6);
    v2 = a34;
    svfloat16_t a35 = svzip1_f16(v3, v7);
    v7 = svzip2_f16(v3, v7);
    v3 = a35;
    svfloat16_t a36 = svzip1_f16(v8, v12);
    v12 = svzip2_f16(v8, v12);
    v8 = a36;
    svfloat16_t a37 = svzip1_f16(v9, v13);
    v13 = svzip2_f16(v9, v13);
    v9 = a37;
    svfloat16_t a38 = svzip1_f16(v10, v14);
    v14 = svzip2_f16(v10, v14);
    v10 = a38;
    svfloat16_t a39 = svzip1_f16(v11, v15);
    v15 = svzip2_f16(v11, v15);
    v11 = a39;
    svfloat16_t a40 = svzip1_f16(v16, v20);
    v20 = svzip2_f16(v16, v20);
    v16 = a40;
    svfloat16_t a41 = svzip1_f16(v17, v21);
    v21 = svzip2_f16(v17, v21);
    v17 = a41;
    svfloat16_t a42 = svzip1_f16(v18, v22);
    v22 = svzip2_f16(v18, v22);
    v18 = a42;
    svfloat16_t a43 = svzip1_f16(v19, v23);
    v23 = svzip2_f16(v19, v23);
    v19 = a43;
    svfloat16_t a44 = svzip1_f16(v24, v28);
    v28 = svzip2_f16(v24, v28);
    v24 = a44;
    svfloat16_t a45 = svzip1_f16(v25, v29);
    v29 = svzip2_f16(v25, v29);
    v25 = a45;
    svfloat16_t a46 = svzip1_f16(v26, v30);
    v30 = svzip2_f16(v26, v30);
    v26 = a46;
    svfloat16_t a47 = svzip1_f16(v27, v31);
    v31 = svzip2_f16(v27, v31);
    v27 = a47;
    svfloat16_t a48 = svzip1_f16(v0, v2);
    v2 = svzip2_f16(v0, v2);
    v0 = a48;
    svfloat16_t a49 = svzip1_f16(v1, v3);
    v3 = svzip2_f16(v1, v3);
    v1 = a49;
    svfloat16_t a50 = svzip1_f16(v4, v6);
    v6 = svzip2_f16(v4, v6);
    v4 = a50;
    svfloat16_t a51 = svzip1_f16(v5, v7);
    v7 = svzip2_f16(v5, v7);
    v5 = a51;
    svfloat16_t a52 = svzip1_f16(v8, v10);
    v10 = svzip2_f16(v8, v10);
    v8 = a52;
    svfloat16_t a53 = svzip1_f16(v9, v11);
    v11 = svzip2_f16(v9, v11);
    v9 = a53;
    svfloat16_t a54 = svzip1_f16(v12, v14);
    v14 = svzip2_f16(v12, v14);
    v12 = a54;
    svfloat16_t a55 = svzip1_f16(v13, v15);
    v15 = svzip2_f16(v13, v15);
    v13 = a55;
    svfloat16_t a56 = svzip1_f16(v16, v18);
    v18 = svzip2_f16(v16, v18);
    v16 = a56;
    svfloat16_t a57 = svzip1_f16(v17, v19);
    v19 = svzip2_f16(v17, v19);
    v17 = a57;
    svfloat16_t a58 = svzip1_f16(v20, v22);
    v22 = svzip2_f16(v20, v22);
    v20 = a58;
    svfloat16_t a59 = svzip1_f16(v21, v23);
    v23 = svzip2_f16(v21, v23);
    v21 = a59;
    svfloat16_t a60 = svzip1_f16(v24, v26);
    v26 = svzip2_f16(v24, v26);
    v24 = a60;
    svfloat16_t a61 = svzip1_f16(v25, v27);
    v27 = svzip2_f16(v25, v27);
    v25 = a61;
    svfloat16_t a62 = svzip1_f16(v28, v30);
    v30 = svzip2_f16(v28, v30);
    v28 = a62;
    svfloat16_t a63 = svzip1_f16(v29, v31);
    v31 = svzip2_f16(v29, v31);
    v29 = a63;
    svfloat16_t a64 = svzip1_f16(v0, v1);
    v1 = svzip2_f16(v0, v1);
    v0 = a64;
    svfloat16_t a65 = svzip1_f16(v2, v3);
    v3 = svzip2_f16(v2, v3);
    v2 = a65;
    svfloat16_t a66 = svzip1_f16(v4, v5);
    v5 = svzip2_f16(v4, v5);
    v4 = a66;
    svfloat16_t a67 = svzip1_f16(v6, v7);
    v7 = svzip2_f16(v6, v7);
    v6 = a67;
    svfloat16_t a68 = svzip1_f16(v8, v9);
    v9 = svzip2_f16(v8, v9);
    v8 = a68;
    svfloat16_t a69 = svzip1_f16(v10, v11);
    v11 = svzip2_f16(v10, v11);
    v10 = a69;
    svfloat16_t a70 = svzip1_f16(v12, v13);
    v13 = svzip2_f16(v12, v13);
    v12 = a70;
    svfloat16_t a71 = svzip1_f16(v14, v15);
    v15 = svzip2_f16(v14, v15);
    v14 = a71;
    svfloat16_t a72 = svzip1_f16(v16, v17);
    v17 = svzip2_f16(v16, v17);
    v16 = a72;
    svfloat16_t a73 = svzip1_f16(v18, v19);
    v19 = svzip2_f16(v18, v19);
    v18 = a73;
    svfloat16_t a74 = svzip1_f16(v20, v21);
    v21 = svzip2_f16(v20, v21);
    v20 = a74;
    svfloat16_t a75 = svzip1_f16(v22, v23);
    v23 = svzip2_f16(v22, v23);
    v22 = a75;
    svfloat16_t a76 = svzip1_f16(v24, v25);
    v25 = svzip2_f16(v24, v25);
    v24 = a76;
    svfloat16_t a77 = svzip1_f16(v26, v27);
    v27 = svzip2_f16(v26, v27);
    v26 = a77;
    svfloat16_t a78 = svzip1_f16(v28, v29);
    v29 = svzip2_f16(v28, v29);
    v28 = a78;
    svfloat16_t a79 = svzip1_f16(v30, v31);
    v31 = svzip2_f16(v30, v31);
    v30 = a79;
    if(rows>0) svst1_f16(pK, d+0*CountX, v0);
    if(rows>1) svst1_f16(pK, d+1*CountX, v1);
    if(rows>2) svst1_f16(pK, d+2*CountX, v2);
    if(rows>3) svst1_f16(pK, d+3*CountX, v3);
    if(rows>4) svst1_f16(pK, d+4*CountX, v4);
    if(rows>5) svst1_f16(pK, d+5*CountX, v5);
    if(rows>6) svst1_f16(pK, d+6*CountX, v6);
    if(rows>7) svst1_f16(pK, d+7*CountX, v7);
    if(rows>8) svst1_f16(pK, d+8*CountX, v8);
    if(rows>9) svst1_f16(pK, d+9*CountX, v9);
    if(rows>10) svst1_f16(pK, d+10*CountX, v10);
    if(rows>11) svst1_f16(pK, d+11*CountX, v11);
    if(rows>12) svst1_f16(pK, d+12*CountX, v12);
    if(rows>13) svst1_f16(pK, d+13*CountX, v13);
    if(rows>14) svst1_f16(pK, d+14*CountX, v14);
    if(rows>15) svst1_f16(pK, d+15*CountX, v15);
    if(rows>16) svst1_f16(pK, d+16*CountX, v16);
    if(rows>17) svst1_f16(pK, d+17*CountX, v17);
    if(rows>18) svst1_f16(pK, d+18*CountX, v18);
    if(rows>19) svst1_f16(pK, d+19*CountX, v19);
    if(rows>20) svst1_f16(pK, d+20*CountX, v20);
    if(rows>21) svst1_f16(pK, d+21*CountX, v21);
    if(rows>22) svst1_f16(pK, d+22*CountX, v22);
    if(rows>23) svst1_f16(pK, d+23*CountX, v23);
    if(rows>24) svst1_f16(pK, d+24*CountX, v24);
    if(rows>25) svst1_f16(pK, d+25*CountX, v25);
    if(rows>26) svst1_f16(pK, d+26*CountX, v26);
    if(rows>27) svst1_f16(pK, d+27*CountX, v27);
    if(rows>28) svst1_f16(pK, d+28*CountX, v28);
    if(rows>29) svst1_f16(pK, d+29*CountX, v29);
    if(rows>30) svst1_f16(pK, d+30*CountX, v30);
    if(rows>31) svst1_f16(pK, d+31*CountX, v31);
}

extern "C" void MLAS_SVE_TARGET MLASCALL
MlasHgemmTransposeA_sve(_mlas_fp16_* D, const _mlas_fp16_* A, size_t lda, size_t CountY, size_t CountX)
{
    const __fp16* Af = reinterpret_cast<const __fp16*>(A);
    __fp16* Df = reinterpret_cast<__fp16*>(D);
    const size_t VL = svcnth();
    if (VL == 8 || VL == 16 || VL == 32) {
        size_t k = 0;
        for (; k + VL <= CountX; k += VL) {
            const __fp16* aK = Af + k * lda;
            __fp16* dK = Df + k;
            for (size_t mb = 0; mb < CountY; mb += VL) {
                const size_t rows = std::min(VL, CountY - mb);
                if (VL == 8)       HTransposeATile8 (dK + mb * CountX, aK + mb, lda, CountX, rows);
                else if (VL == 16) HTransposeATile16(dK + mb * CountX, aK + mb, lda, CountX, rows);
                else               HTransposeATile32(dK + mb * CountX, aK + mb, lda, CountX, rows);
            }
        }
        for (; k < CountX; ++k)
            for (size_t m = 0; m < CountY; ++m)
                Df[m * CountX + k] = Af[k * lda + m];
        return;
    }
    for (size_t m = 0; m < CountY; ++m)
        for (size_t k = 0; k < CountX; ++k)
            Df[m * CountX + k] = Af[k * lda + m];
}

#ifndef __clang__
#pragma GCC pop_options
#endif

#endif // MLAS_USE_SVE
