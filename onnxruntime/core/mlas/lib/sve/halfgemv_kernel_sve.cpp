/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2025 FUJITSU LIMITED

Licensed under the MIT License.

Module Name:

    halfgemv_kernel_sve.cpp

Abstract:

    SVE intrinsics implementation of the FP16 matrix-vector kernel (N == 1),
    and the regeneration source for the portable machine-code variant
    (aarch64/halfgemv_sve_asm.S, script: sve/gen_sve_asm.py).

    Vector length agnostic: the K loop steps by svcnth() and the tail is
    predicated with svwhilelt_b16, so the same code runs on 128-, 256- and
    512-bit SVE without a width switch.

    Four rows of A are processed per pass so the B vector is loaded once per
    row group rather than once per row -- B is the reused operand here, and at
    N == 1 the kernel is bandwidth bound on A.

    Self-containment contract (verified by the generator): no calls, no global
    data, no literal pools -- every input arrives through the argument
    registers, so the frozen machine code is position independent.

--*/

#include <arm_sve.h>

#include "halfgemv_sve.h"

using _mlas_fp16_ = mlas_sve_fp16_t;

#if !defined(MLAS_FORCEINLINE)
#define MLAS_FORCEINLINE __attribute__((always_inline)) inline
#endif

//
// Accumulate acc += a[k] * b[k] over one predicated vector of K.
//
MLAS_FORCEINLINE svfloat16_t
HgemvStep(svbool_t pg, svfloat16_t acc, const _mlas_fp16_* a, svfloat16_t bvec)
{
    return svmla_f16_m(pg, acc, svld1_f16(pg, (const __fp16*)a), bvec);
}

extern "C" void
MlasHgemvNKernel_sve(
    const _mlas_fp16_* A,
    const _mlas_fp16_* B,
    _mlas_fp16_* C,
    size_t CountM,
    size_t CountK,
    size_t lda,
    size_t ldc_elem,
    float alpha,
    bool ZeroMode
    )
{
    const size_t vl = svcnth();
    const __fp16 alpha_h = (__fp16)alpha;

    size_t m = 0;

    //
    // Four rows at a time: one B load feeds four independent FMA chains, which
    // both amortises the B traffic and gives the FMA pipeline four independent
    // accumulators to work on.
    //
    for (; m + 4 <= CountM; m += 4) {
        svfloat16_t acc0 = svdup_n_f16((__fp16)0.f);
        svfloat16_t acc1 = acc0, acc2 = acc0, acc3 = acc0;

        const _mlas_fp16_* a0 = A + (m + 0) * lda;
        const _mlas_fp16_* a1 = A + (m + 1) * lda;
        const _mlas_fp16_* a2 = A + (m + 2) * lda;
        const _mlas_fp16_* a3 = A + (m + 3) * lda;

        for (size_t k = 0; k < CountK; k += vl) {
            const svbool_t pg = svwhilelt_b16(k, CountK);
            const svfloat16_t bvec = svld1_f16(pg, (const __fp16*)(B + k));
            acc0 = HgemvStep(pg, acc0, a0 + k, bvec);
            acc1 = HgemvStep(pg, acc1, a1 + k, bvec);
            acc2 = HgemvStep(pg, acc2, a2 + k, bvec);
            acc3 = HgemvStep(pg, acc3, a3 + k, bvec);
        }

        const svbool_t pall = svptrue_b16();
        __fp16 r0 = svaddv_f16(pall, acc0) * alpha_h;
        __fp16 r1 = svaddv_f16(pall, acc1) * alpha_h;
        __fp16 r2 = svaddv_f16(pall, acc2) * alpha_h;
        __fp16 r3 = svaddv_f16(pall, acc3) * alpha_h;

        __fp16* c = (__fp16*)C;
        if (ZeroMode) {
            c[(m + 0) * ldc_elem] = r0;
            c[(m + 1) * ldc_elem] = r1;
            c[(m + 2) * ldc_elem] = r2;
            c[(m + 3) * ldc_elem] = r3;
        } else {
            c[(m + 0) * ldc_elem] += r0;
            c[(m + 1) * ldc_elem] += r1;
            c[(m + 2) * ldc_elem] += r2;
            c[(m + 3) * ldc_elem] += r3;
        }
    }

    //
    // Remaining rows, one at a time.
    //
    for (; m < CountM; ++m) {
        svfloat16_t acc = svdup_n_f16((__fp16)0.f);
        const _mlas_fp16_* a = A + m * lda;

        for (size_t k = 0; k < CountK; k += vl) {
            const svbool_t pg = svwhilelt_b16(k, CountK);
            const svfloat16_t bvec = svld1_f16(pg, (const __fp16*)(B + k));
            acc = HgemvStep(pg, acc, a + k, bvec);
        }

        __fp16 r = svaddv_f16(svptrue_b16(), acc) * alpha_h;
        __fp16* c = (__fp16*)C;
        if (ZeroMode) {
            c[m * ldc_elem] = r;
        } else {
            c[m * ldc_elem] += r;
        }
    }
}
