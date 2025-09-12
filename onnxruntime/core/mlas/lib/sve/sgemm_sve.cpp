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
    constexpr size_t k_step = 64;  // Tunable tile size
    svfloat32_t zero_vec = MlasSvedupFloat32(0.f);
    for (size_t col = 0; col < n; col += vl) {
        svbool_t pg = svwhilelt_b32(col, n);
        // Output pointers
        float* out0 = res + 0 * ldc + col;
        float* out1 = res + 1 * ldc + col;
        float* out2 = res + 2 * ldc + col;
        float* out3 = res + 3 * ldc + col;
        float* out4 = res + 4 * ldc + col;
        float* out5 = res + 5 * ldc + col;
        float* out6 = res + 6 * ldc + col;
        float* out7 = res + 7 * ldc + col;
        // Accumulators initialized to zero
        svfloat32_t acc0 = zero_vec;
        svfloat32_t acc1 = zero_vec;
        svfloat32_t acc2 = zero_vec;
        svfloat32_t acc3 = zero_vec;
        svfloat32_t acc4 = zero_vec;
        svfloat32_t acc5 = zero_vec;
        svfloat32_t acc6 = zero_vec;
        svfloat32_t acc7 = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            // Temporary partial sums
            svfloat32_t partial0 = zero_vec;
            svfloat32_t partial1 = zero_vec;
            svfloat32_t partial2 = zero_vec;
            svfloat32_t partial3 = zero_vec;
            svfloat32_t partial4 = zero_vec;
            svfloat32_t partial5 = zero_vec;
            svfloat32_t partial6 = zero_vec;
            svfloat32_t partial7 = zero_vec;
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b_vec = b + p * PACKED_B_BLOCK_WIDTH + col;
                svfloat32_t bvals = MlasSveLoadFloat32(pg, b_vec);
                svfloat32_t a0, a1, a2, a3, a4, a5, a6, a7;
                if constexpr (!Alpha1) {
                    a0 = MlasSvedupFloat32(a[0 * lda + p] * alpha);
                    a1 = MlasSvedupFloat32(a[1 * lda + p] * alpha);
                    a2 = MlasSvedupFloat32(a[2 * lda + p] * alpha);
                    a3 = MlasSvedupFloat32(a[3 * lda + p] * alpha);
                    a4 = MlasSvedupFloat32(a[4 * lda + p] * alpha);
                    a5 = MlasSvedupFloat32(a[5 * lda + p] * alpha);
                    a6 = MlasSvedupFloat32(a[6 * lda + p] * alpha);
                    a7 = MlasSvedupFloat32(a[7 * lda + p] * alpha);
                } else {
                    a0 = MlasSvedupFloat32(a[0 * lda + p]);
                    a1 = MlasSvedupFloat32(a[1 * lda + p]);
                    a2 = MlasSvedupFloat32(a[2 * lda + p]);
                    a3 = MlasSvedupFloat32(a[3 * lda + p]);
                    a4 = MlasSvedupFloat32(a[4 * lda + p]);
                    a5 = MlasSvedupFloat32(a[5 * lda + p]);
                    a6 = MlasSvedupFloat32(a[6 * lda + p]);
                    a7 = MlasSvedupFloat32(a[7 * lda + p]);
                }
                partial0 = MlasSveMultiplyAddFloat32(pg, partial0, bvals, a0);
                partial1 = MlasSveMultiplyAddFloat32(pg, partial1, bvals, a1);
                partial2 = MlasSveMultiplyAddFloat32(pg, partial2, bvals, a2);
                partial3 = MlasSveMultiplyAddFloat32(pg, partial3, bvals, a3);
                partial4 = MlasSveMultiplyAddFloat32(pg, partial4, bvals, a4);
                partial5 = MlasSveMultiplyAddFloat32(pg, partial5, bvals, a5);
                partial6 = MlasSveMultiplyAddFloat32(pg, partial6, bvals, a6);
                partial7 = MlasSveMultiplyAddFloat32(pg, partial7, bvals, a7);
            }
            // Accumulate partials into accumulators
            acc0 = MlasSveAddFloat32(pg, acc0, partial0);
            acc1 = MlasSveAddFloat32(pg, acc1, partial1);
            acc2 = MlasSveAddFloat32(pg, acc2, partial2);
            acc3 = MlasSveAddFloat32(pg, acc3, partial3);
            acc4 = MlasSveAddFloat32(pg, acc4, partial4);
            acc5 = MlasSveAddFloat32(pg, acc5, partial5);
            acc6 = MlasSveAddFloat32(pg, acc6, partial6);
            acc7 = MlasSveAddFloat32(pg, acc7, partial7);
        }
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
        // Store results
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
    constexpr size_t k_step = 64;  // Can be tuned per architecture
    svfloat32_t zero_vec = MlasSvedupFloat32(0.f);
    for (size_t col = 0; col < n; col += vl) {
        svbool_t pg = svwhilelt_b32(col, n);
        float* out0 = res + 0 * ldc + col;
        float* out1 = res + 1 * ldc + col;
        float* out2 = res + 2 * ldc + col;
        float* out3 = res + 3 * ldc + col;
        float* out4 = res + 4 * ldc + col;
        float* out5 = res + 5 * ldc + col;
        // Initialize accumulators to zero
        svfloat32_t acc0 = zero_vec;
        svfloat32_t acc1 = zero_vec;
        svfloat32_t acc2 = zero_vec;
        svfloat32_t acc3 = zero_vec;
        svfloat32_t acc4 = zero_vec;
        svfloat32_t acc5 = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            svfloat32_t partial0 = zero_vec;
            svfloat32_t partial1 = zero_vec;
            svfloat32_t partial2 = zero_vec;
            svfloat32_t partial3 = zero_vec;
            svfloat32_t partial4 = zero_vec;
            svfloat32_t partial5 = zero_vec;
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b_vec = b + p * PACKED_B_BLOCK_WIDTH + col;
                svfloat32_t bvals = MlasSveLoadFloat32(pg, b_vec);
                svfloat32_t a0, a1, a2, a3, a4, a5;
                if constexpr (!Alpha1) {
                    a0 = MlasSvedupFloat32(a[0 * lda + p] * alpha);
                    a1 = MlasSvedupFloat32(a[1 * lda + p] * alpha);
                    a2 = MlasSvedupFloat32(a[2 * lda + p] * alpha);
                    a3 = MlasSvedupFloat32(a[3 * lda + p] * alpha);
                    a4 = MlasSvedupFloat32(a[4 * lda + p] * alpha);
                    a5 = MlasSvedupFloat32(a[5 * lda + p] * alpha);
                } else {
                    a0 = MlasSvedupFloat32(a[0 * lda + p]);
                    a1 = MlasSvedupFloat32(a[1 * lda + p]);
                    a2 = MlasSvedupFloat32(a[2 * lda + p]);
                    a3 = MlasSvedupFloat32(a[3 * lda + p]);
                    a4 = MlasSvedupFloat32(a[4 * lda + p]);
                    a5 = MlasSvedupFloat32(a[5 * lda + p]);
                }
                partial0 = MlasSveMultiplyAddFloat32(pg, partial0, bvals, a0);
                partial1 = MlasSveMultiplyAddFloat32(pg, partial1, bvals, a1);
                partial2 = MlasSveMultiplyAddFloat32(pg, partial2, bvals, a2);
                partial3 = MlasSveMultiplyAddFloat32(pg, partial3, bvals, a3);
                partial4 = MlasSveMultiplyAddFloat32(pg, partial4, bvals, a4);
                partial5 = MlasSveMultiplyAddFloat32(pg, partial5, bvals, a5);
            }
            acc0 = MlasSveAddFloat32(pg, acc0, partial0);
            acc1 = MlasSveAddFloat32(pg, acc1, partial1);
            acc2 = MlasSveAddFloat32(pg, acc2, partial2);
            acc3 = MlasSveAddFloat32(pg, acc3, partial3);
            acc4 = MlasSveAddFloat32(pg, acc4, partial4);
            acc5 = MlasSveAddFloat32(pg, acc5, partial5);
        }
        // Add existing result values at the end (if not ZeroMode)
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
    constexpr size_t k_step = 64;  // Tunable tile size
    svfloat32_t zero_vec = MlasSvedupFloat32(0.f);
    for (size_t col = 0; col < n; col += vl) {
        svbool_t pg = svwhilelt_b32(col, n);
        float* out0 = res + 0 * ldc + col;
        float* out1 = res + 1 * ldc + col;
        float* out2 = res + 2 * ldc + col;
        float* out3 = res + 3 * ldc + col;
        // Start with clean zeroed accumulators
        svfloat32_t acc0 = zero_vec;
        svfloat32_t acc1 = zero_vec;
        svfloat32_t acc2 = zero_vec;
        svfloat32_t acc3 = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            svfloat32_t partial0 = zero_vec;
            svfloat32_t partial1 = zero_vec;
            svfloat32_t partial2 = zero_vec;
            svfloat32_t partial3 = zero_vec;
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b_vec = b + p * PACKED_B_BLOCK_WIDTH + col;
                svfloat32_t bvals = MlasSveLoadFloat32(pg, b_vec);
                svfloat32_t a0, a1, a2, a3;
                if constexpr (!Alpha1) {
                    a0 = MlasSvedupFloat32(a[0 * lda + p] * alpha);
                    a1 = MlasSvedupFloat32(a[1 * lda + p] * alpha);
                    a2 = MlasSvedupFloat32(a[2 * lda + p] * alpha);
                    a3 = MlasSvedupFloat32(a[3 * lda + p] * alpha);
                } else {
                    a0 = MlasSvedupFloat32(a[0 * lda + p]);
                    a1 = MlasSvedupFloat32(a[1 * lda + p]);
                    a2 = MlasSvedupFloat32(a[2 * lda + p]);
                    a3 = MlasSvedupFloat32(a[3 * lda + p]);
                }
                partial0 = MlasSveMultiplyAddFloat32(pg, partial0, bvals, a0);
                partial1 = MlasSveMultiplyAddFloat32(pg, partial1, bvals, a1);
                partial2 = MlasSveMultiplyAddFloat32(pg, partial2, bvals, a2);
                partial3 = MlasSveMultiplyAddFloat32(pg, partial3, bvals, a3);
            }
            acc0 = MlasSveAddFloat32(pg, acc0, partial0);
            acc1 = MlasSveAddFloat32(pg, acc1, partial1);
            acc2 = MlasSveAddFloat32(pg, acc2, partial2);
            acc3 = MlasSveAddFloat32(pg, acc3, partial3);
        }
        // Final addition of existing result (if ZeroMode == false)
        if constexpr (!ZeroMode) {
            acc0 = MlasSveAddFloat32(pg, acc0, svld1(pg, out0));
            acc1 = MlasSveAddFloat32(pg, acc1, svld1(pg, out1));
            acc2 = MlasSveAddFloat32(pg, acc2, svld1(pg, out2));
            acc3 = MlasSveAddFloat32(pg, acc3, svld1(pg, out3));
        }
        // Store the final accumulated results
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
    int lda,
    int ldc,
    float alpha,
    size_t vl
)
{
    constexpr size_t k_step = 64;  // Tune this value as needed
    svfloat32_t zero_vec = MlasSvedupFloat32(0.f);
    for (size_t col = 0; col < n; col += vl) {
        svbool_t pg = svwhilelt_b32(col, n);
        float* out0 = res + 0 * ldc + col;
        float* out1 = res + 1 * ldc + col;
        // Always start with zero accumulators
        svfloat32_t acc0 = zero_vec;
        svfloat32_t acc1 = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            svfloat32_t partial0 = zero_vec;
            svfloat32_t partial1 = zero_vec;
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b_vec = b + p * PACKED_B_BLOCK_WIDTH + col;
                svfloat32_t bvals = MlasSveLoadFloat32(pg, b_vec);
                svfloat32_t a0, a1;
                if constexpr (!Alpha1) {
                    a0 = MlasSvedupFloat32(a[0 * lda + p] * alpha);
                    a1 = MlasSvedupFloat32(a[1 * lda + p] * alpha);
                } else {
                    a0 = MlasSvedupFloat32(a[0 * lda + p]);
                    a1 = MlasSvedupFloat32(a[1 * lda + p]);
                }
                partial0 = MlasSveMultiplyAddFloat32(pg, partial0, bvals, a0);
                partial1 = MlasSveMultiplyAddFloat32(pg, partial1, bvals, a1);
            }
            acc0 = MlasSveAddFloat32(pg, acc0, partial0);
            acc1 = MlasSveAddFloat32(pg, acc1, partial1);
        }
        // Add existing values at the end (if ZeroMode == false)
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
    constexpr size_t k_step = 64;  // Tune based on target hardware
    svfloat32_t zero_vec = MlasSvedupFloat32(0.f);
    for (size_t col = 0; col < n; col += vl) {
        svbool_t pg = svwhilelt_b32(col, n);
        float* out0 = res + 0 * ldc + col;
        // Always start with zero accumulator
        svfloat32_t acc0 = zero_vec;
        for (size_t k_block = 0; k_block < k; k_block += k_step) {
            size_t k_max = std::min(k_block + k_step, k);
            svfloat32_t partial = zero_vec;
            for (size_t p = k_block; p < k_max; ++p) {
                const float* b_vec = b + p * PACKED_B_BLOCK_WIDTH + col;
                svfloat32_t bvals = MlasSveLoadFloat32(pg, b_vec);
                svfloat32_t a0;
                if constexpr (!Alpha1) {
                    a0 = MlasSvedupFloat32(a[p + 0 * lda] * alpha);
                } else {
                    a0 = MlasSvedupFloat32(a[p + 0 * lda]);
                }
                partial = MlasSveMultiplyAddFloat32(pg, partial, bvals, a0);
            }
            acc0 = MlasSveAddFloat32(pg, acc0, partial);
        }
        // In Add mode (ZeroMode == false), add existing res at the end
        if constexpr (!ZeroMode) {
            svfloat32_t prev = svld1(pg, out0);
            acc0 = MlasSveAddFloat32(pg, acc0, prev);
        }
        // Store final result
        MlasSveStoreFloat32(pg, out0, acc0);
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
    if (alpha == 1.0f) {
        if (CountM >= 8) {
            ProcessRowsTemplate<processrows_8<true, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 8;
        }

        else if (CountM >= 6) {
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
        }

        else if (CountM >= 6) {
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
    if (alpha == 1.0f) {
        if (CountM >= 8) {
            ProcessRowsTemplate<processrows_8<false, true>>(A, lda, B, C, ldc, CountK, CountN, alpha);
            return 8;
        }

        else if (CountM >= 6) {
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
        }

        else if (CountM >= 6) {
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

#endif
