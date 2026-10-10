/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sgemm_kernel_rvv.cpp

Abstract:

    This module implements an RVV kernel for the single precision matrix/matrix
    multiply operation (SGEMM) on riscv64.

    Optimizations applied:
    1. Use vfloat32m1_t to allow more accumulators without register pressure
    2. 4 accumulators per row to hide FMACC latency
    3. 8x K-loop unrolling for better ILP
    4. Software prefetching for next iteration

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

namespace {

// The packed B layout stays 16 columns wide to match MLAS.
// We process 4 columns at a time using vfloat32m1_t.
constexpr size_t kPackedCountN = 16;
constexpr size_t kBlockSize = 4;  // Process 4 columns at a time

template<bool ZeroMode, bool AlphaIsOne, bool ProcessTwoRows>
MLAS_FORCEINLINE
size_t
MlasSgemmKernelRvvOptimized(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
{
#if defined(_WIN32)
    if (!ProcessTwoRows) {
        UNREFERENCED_PARAMETER(lda);
        UNREFERENCED_PARAMETER(ldc);
    }
    if constexpr (AlphaIsOne) {
        UNREFERENCED_PARAMETER(alpha);
    }
#endif

    // Precompute the B stride adjustment
    size_t k_shift = CountK * kPackedCountN - kPackedCountN;
    int countb = 0;

    // Set vector length to 4 for processing 4 floats per vector register
    size_t vl = __riscv_vsetvl_e32m1(kBlockSize);

    do {
        // Clear the block accumulators - 4 accumulators per row for latency hiding
        vfloat32m1_t Row0Acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t Row0Acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t Row0Acc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t Row0Acc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        vfloat32m1_t Row1Acc0, Row1Acc1, Row1Acc2, Row1Acc3;
        if (ProcessTwoRows) {
            Row1Acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            Row1Acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            Row1Acc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            Row1Acc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        }

        const float* a = A;
        const float* b = B;
        size_t k = CountK;

        //
        // Main K loop - 8x unroll for better ILP
        //
        while (k >= 8) {
            // Prefetch next iteration data
            __builtin_prefetch(a + 8, 0, 3);
            __builtin_prefetch(b + 8 * kPackedCountN, 0, 3);

            // --- K elements 0, 1, 2, 3 ---
            float a0_0 = a[0], a0_1 = a[1], a0_2 = a[2], a0_3 = a[3];

            vfloat32m1_t b0 = __riscv_vle32_v_f32m1(b, vl);
            vfloat32m1_t b1 = __riscv_vle32_v_f32m1(b + kPackedCountN, vl);
            vfloat32m1_t b2 = __riscv_vle32_v_f32m1(b + kPackedCountN * 2, vl);
            vfloat32m1_t b3 = __riscv_vle32_v_f32m1(b + kPackedCountN * 3, vl);

            Row0Acc0 = __riscv_vfmacc_vf_f32m1(Row0Acc0, a0_0, b0, vl);
            Row0Acc1 = __riscv_vfmacc_vf_f32m1(Row0Acc1, a0_1, b1, vl);
            Row0Acc2 = __riscv_vfmacc_vf_f32m1(Row0Acc2, a0_2, b2, vl);
            Row0Acc3 = __riscv_vfmacc_vf_f32m1(Row0Acc3, a0_3, b3, vl);

            // --- K elements 4, 5, 6, 7 ---
            float a0_4 = a[4], a0_5 = a[5], a0_6 = a[6], a0_7 = a[7];

            vfloat32m1_t b4 = __riscv_vle32_v_f32m1(b + kPackedCountN * 4, vl);
            vfloat32m1_t b5 = __riscv_vle32_v_f32m1(b + kPackedCountN * 5, vl);
            vfloat32m1_t b6 = __riscv_vle32_v_f32m1(b + kPackedCountN * 6, vl);
            vfloat32m1_t b7 = __riscv_vle32_v_f32m1(b + kPackedCountN * 7, vl);

            Row0Acc0 = __riscv_vfmacc_vf_f32m1(Row0Acc0, a0_4, b4, vl);
            Row0Acc1 = __riscv_vfmacc_vf_f32m1(Row0Acc1, a0_5, b5, vl);
            Row0Acc2 = __riscv_vfmacc_vf_f32m1(Row0Acc2, a0_6, b6, vl);
            Row0Acc3 = __riscv_vfmacc_vf_f32m1(Row0Acc3, a0_7, b7, vl);

            if (ProcessTwoRows) {
                float a1_0 = a[lda], a1_1 = a[lda + 1], a1_2 = a[lda + 2], a1_3 = a[lda + 3];
                float a1_4 = a[lda + 4], a1_5 = a[lda + 5], a1_6 = a[lda + 6], a1_7 = a[lda + 7];

                Row1Acc0 = __riscv_vfmacc_vf_f32m1(Row1Acc0, a1_0, b0, vl);
                Row1Acc1 = __riscv_vfmacc_vf_f32m1(Row1Acc1, a1_1, b1, vl);
                Row1Acc2 = __riscv_vfmacc_vf_f32m1(Row1Acc2, a1_2, b2, vl);
                Row1Acc3 = __riscv_vfmacc_vf_f32m1(Row1Acc3, a1_3, b3, vl);
                Row1Acc0 = __riscv_vfmacc_vf_f32m1(Row1Acc0, a1_4, b4, vl);
                Row1Acc1 = __riscv_vfmacc_vf_f32m1(Row1Acc1, a1_5, b5, vl);
                Row1Acc2 = __riscv_vfmacc_vf_f32m1(Row1Acc2, a1_6, b6, vl);
                Row1Acc3 = __riscv_vfmacc_vf_f32m1(Row1Acc3, a1_7, b7, vl);
            }

            a += 8;
            b += kPackedCountN * 8;
            k -= 8;
        }

        //
        // Handle remaining K elements (4 at a time)
        //
        while (k >= 4) {
            float a0_0 = a[0], a0_1 = a[1], a0_2 = a[2], a0_3 = a[3];

            __builtin_prefetch(a + 4, 0, 3);
            __builtin_prefetch(b + kPackedCountN * 4, 0, 3);

            vfloat32m1_t b0 = __riscv_vle32_v_f32m1(b, vl);
            vfloat32m1_t b1 = __riscv_vle32_v_f32m1(b + kPackedCountN, vl);
            vfloat32m1_t b2 = __riscv_vle32_v_f32m1(b + kPackedCountN * 2, vl);
            vfloat32m1_t b3 = __riscv_vle32_v_f32m1(b + kPackedCountN * 3, vl);

            Row0Acc0 = __riscv_vfmacc_vf_f32m1(Row0Acc0, a0_0, b0, vl);
            Row0Acc1 = __riscv_vfmacc_vf_f32m1(Row0Acc1, a0_1, b1, vl);
            Row0Acc2 = __riscv_vfmacc_vf_f32m1(Row0Acc2, a0_2, b2, vl);
            Row0Acc3 = __riscv_vfmacc_vf_f32m1(Row0Acc3, a0_3, b3, vl);

            if (ProcessTwoRows) {
                float a1_0 = a[lda], a1_1 = a[lda + 1], a1_2 = a[lda + 2], a1_3 = a[lda + 3];
                Row1Acc0 = __riscv_vfmacc_vf_f32m1(Row1Acc0, a1_0, b0, vl);
                Row1Acc1 = __riscv_vfmacc_vf_f32m1(Row1Acc1, a1_1, b1, vl);
                Row1Acc2 = __riscv_vfmacc_vf_f32m1(Row1Acc2, a1_2, b2, vl);
                Row1Acc3 = __riscv_vfmacc_vf_f32m1(Row1Acc3, a1_3, b3, vl);
            }

            a += 4;
            b += kPackedCountN * 4;
            k -= 4;
        }

        //
        // Handle remaining K elements (2 at a time)
        //
        while (k >= 2) {
            float a0_0 = a[0], a0_1 = a[1];

            vfloat32m1_t b0 = __riscv_vle32_v_f32m1(b, vl);
            vfloat32m1_t b1 = __riscv_vle32_v_f32m1(b + kPackedCountN, vl);

            Row0Acc0 = __riscv_vfmacc_vf_f32m1(Row0Acc0, a0_0, b0, vl);
            Row0Acc1 = __riscv_vfmacc_vf_f32m1(Row0Acc1, a0_1, b1, vl);

            if (ProcessTwoRows) {
                float a1_0 = a[lda], a1_1 = a[lda + 1];
                Row1Acc0 = __riscv_vfmacc_vf_f32m1(Row1Acc0, a1_0, b0, vl);
                Row1Acc1 = __riscv_vfmacc_vf_f32m1(Row1Acc1, a1_1, b1, vl);
            }

            a += 2;
            b += kPackedCountN * 2;
            k -= 2;
        }

        //
        // Handle remaining K element
        //
        if (k > 0) {
            float a0_0 = a[0];
            vfloat32m1_t b0 = __riscv_vle32_v_f32m1(b, vl);
            Row0Acc0 = __riscv_vfmacc_vf_f32m1(Row0Acc0, a0_0, b0, vl);

            if (ProcessTwoRows) {
                float a1_0 = a[lda];
                Row1Acc0 = __riscv_vfmacc_vf_f32m1(Row1Acc0, a1_0, b0, vl);
            }
        }

        //
        // Merge accumulators
        //
        vfloat32m1_t Row0Block = __riscv_vfadd_vv_f32m1(
            __riscv_vfadd_vv_f32m1(Row0Acc0, Row0Acc1, vl),
            __riscv_vfadd_vv_f32m1(Row0Acc2, Row0Acc3, vl), vl);

        vfloat32m1_t Row1Block;
        if (ProcessTwoRows) {
            Row1Block = __riscv_vfadd_vv_f32m1(
                __riscv_vfadd_vv_f32m1(Row1Acc0, Row1Acc1, vl),
                __riscv_vfadd_vv_f32m1(Row1Acc2, Row1Acc3, vl), vl);
        }

        //
        // Multiply by alpha
        //
        if constexpr (!AlphaIsOne) {
            Row0Block = __riscv_vfmul_vf_f32m1(Row0Block, alpha, vl);
            if (ProcessTwoRows) {
                Row1Block = __riscv_vfmul_vf_f32m1(Row1Block, alpha, vl);
            }
        }

        //
        // Store the output block
        //
        if (CountN >= kBlockSize) {
            if (!ZeroMode) {
                vfloat32m1_t CVec = __riscv_vle32_v_f32m1(C, vl);
                Row0Block = __riscv_vfadd_vv_f32m1(Row0Block, CVec, vl);
            }
            __riscv_vse32_v_f32m1(C, Row0Block, vl);

            if (ProcessTwoRows) {
                if (!ZeroMode) {
                    vfloat32m1_t CVec = __riscv_vle32_v_f32m1(C + ldc, vl);
                    Row1Block = __riscv_vfadd_vv_f32m1(Row1Block, CVec, vl);
                }
                __riscv_vse32_v_f32m1(C + ldc, Row1Block, vl);
            }

        } else {
            // Store partial output block
            float Row0Block_arr[4];
            __riscv_vse32_v_f32m1(Row0Block_arr, Row0Block, vl);

            float Row1Block_arr[4];
            if (ProcessTwoRows) {
                __riscv_vse32_v_f32m1(Row1Block_arr, Row1Block, vl);
            }

            if ((CountN & 2) != 0) {
                if (!ZeroMode) {
                    Row0Block_arr[0] += C[0];
                    Row0Block_arr[1] += C[1];
                }
                C[0] = Row0Block_arr[0];
                C[1] = Row0Block_arr[1];
                Row0Block_arr[0] = Row0Block_arr[2];
                Row0Block_arr[1] = Row0Block_arr[3];

                if (ProcessTwoRows) {
                    if (!ZeroMode) {
                        Row1Block_arr[0] += C[ldc];
                        Row1Block_arr[1] += C[ldc + 1];
                    }
                    C[ldc] = Row1Block_arr[0];
                    C[ldc + 1] = Row1Block_arr[1];
                    Row1Block_arr[0] = Row1Block_arr[2];
                    Row1Block_arr[1] = Row1Block_arr[3];
                }
                C += 2;
            }

            if ((CountN & 1) != 0) {
                if (!ZeroMode) {
                    Row0Block_arr[0] += C[0];
                }
                C[0] = Row0Block_arr[0];

                if (ProcessTwoRows) {
                    if (!ZeroMode) {
                        Row1Block_arr[0] += C[ldc];
                    }
                    C[ldc] = Row1Block_arr[0];
                }
            }

            break;
        }

        // Advance to the next block of 4 output columns
        countb = (countb + 1) % 4;
        B += kBlockSize;
        C += kBlockSize;
        CountN -= kBlockSize;

        if (countb == 0) {
            B += k_shift;
        }

    } while (CountN > 0);

    return ProcessTwoRows ? 2 : 1;
}

template<bool ZeroMode, bool AlphaIsOne>
MLAS_FORCEINLINE
size_t
MlasGemmFloatKernelRvvDispatchRows(
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
    // Process 2 rows at a time for best performance
    size_t rows_handled = 0;

    while (CountM >= 2) {
        size_t handled = MlasSgemmKernelRvvOptimized<ZeroMode, AlphaIsOne, true>(
            A, B, C, CountK, CountN, lda, ldc, alpha);
        C += ldc * handled;
        A += lda * handled;
        CountM -= handled;
        rows_handled += handled;
    }

    if (CountM == 1) {
        size_t handled = MlasSgemmKernelRvvOptimized<ZeroMode, AlphaIsOne, false>(
            A, B, C, CountK, CountN, lda, ldc, alpha);
        rows_handled += handled;
    }

    return rows_handled;
}

template<bool ZeroMode>
MLAS_FORCEINLINE
size_t
MlasGemmFloatKernelRvvDispatch(
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
        return MlasGemmFloatKernelRvvDispatchRows<ZeroMode, true>(
            A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
    }
    return MlasGemmFloatKernelRvvDispatchRows<ZeroMode, false>(
        A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}

}  // namespace

size_t
MLASCALL
MlasGemmFloatKernelRvv(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
    )
{
    if (ZeroMode) {
        return MlasGemmFloatKernelRvvDispatch<true>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
    }
    return MlasGemmFloatKernelRvvDispatch<false>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}

#endif  // defined(MLAS_USE_RVV)
