/*++

Copyright 2025 FUJITSU LIMITED
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    elementwise_sve.cpp

Abstract:

    This module implements the float32 elementwise kernel bodies using SVE
    intrinsics: Erf, Logistic, Exp, SumExp, ReduceMaximum,
    ReduceMinimumMaximum and the Softmax/LogSoftmax output steps.

    Only the vector loops live here, as extern "C" *Impl functions that
    receive their constant tables by pointer and therefore contain no data
    relocations: this lets the machine code be captured verbatim into the
    portable assembly variant (elementwise_sve_asm.S), which exports the
    same symbols. The public MlasSve* entry points live in
    elementwise_sve_dispatch.cpp, which links against exactly one of the two
    implementations.

    The kernels share one loop idiom: a whilelt-governed predicate covers the
    vector body and the tail in the same loop, and every constant is
    broadcast once before the loop. The polynomial constants are the shared
    tables from erf.cpp / logistic.cpp / compute.cpp — the values and the
    operation order match the scalar/NEON kernels.

    This file is compiled with SVE enabled via its per-file -march compile
    flag (see onnxruntime_mlas.cmake), so no pragma/attribute targeting is
    needed.

--*/

#include "mlasi_sve.h"

#include <arm_sve.h>

extern "C" void
MLASCALL
MlasSveErfKernelImpl(
    const float* Input,
    float* Output,
    size_t N,
    const MLAS_ERF_CONSTANTS* Constants
    )
/*++

Routine Description:

    This routine implements the error function using SVE.

    erf(x) = sign(x) * P_small(|x|) for |x| <= 0.921875, else
    sign(x) * (1 - exp(P_big(min(|x|, 3.925)))), with the same split
    polynomials as the scalar/NEON kernels.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    const svfloat32_t NegZero = svdup_n_f32(Constants->ErfNegZero);
    const svfloat32_t UpperAbsRange = svdup_n_f32(Constants->ErfUpperAbsRange);
    const svfloat32_t SplitBoundary = svdup_n_f32(Constants->ErfSplitBoundary);
    const svfloat32_t One = svdup_n_f32(Constants->ErfOne);

    const svfloat32_t SmallP0 = svdup_n_f32(Constants->ErfSMALL_P0);
    const svfloat32_t SmallP1 = svdup_n_f32(Constants->ErfSMALL_P1);
    const svfloat32_t SmallP2 = svdup_n_f32(Constants->ErfSMALL_P2);
    const svfloat32_t SmallP3 = svdup_n_f32(Constants->ErfSMALL_P3);
    const svfloat32_t SmallP4 = svdup_n_f32(Constants->ErfSMALL_P4);
    const svfloat32_t SmallP5 = svdup_n_f32(Constants->ErfSMALL_P5_Minus_One);

    const svfloat32_t BigP0 = svdup_n_f32(Constants->ErfBIG_P0);
    const svfloat32_t BigP1 = svdup_n_f32(Constants->ErfBIG_P1);
    const svfloat32_t BigP2 = svdup_n_f32(Constants->ErfBIG_P2);
    const svfloat32_t BigP3 = svdup_n_f32(Constants->ErfBIG_P3);
    const svfloat32_t BigP4 = svdup_n_f32(Constants->ErfBIG_P4);
    const svfloat32_t BigP5 = svdup_n_f32(Constants->ErfBIG_P5);
    const svfloat32_t BigP6 = svdup_n_f32(Constants->ErfBIG_P6_Minus_One);

    const svfloat32_t ExpLowerRange = svdup_n_f32(Constants->Exp_LowerRange);
    const svfloat32_t ExpLog2Reciprocal = svdup_n_f32(Constants->Exp_Log2Reciprocal);
    const svfloat32_t ExpLog2Hi = svdup_n_f32(Constants->Exp_log2_hi);
    const svfloat32_t ExpLog2Lo = svdup_n_f32(Constants->Exp_log2_lo);
    const svfloat32_t ExpC = svdup_n_f32(Constants->Exp_C);
    const svfloat32_t ExpP0 = svdup_n_f32(Constants->Exp_P0);
    const svfloat32_t ExpP1 = svdup_n_f32(Constants->Exp_P1);
    const svfloat32_t ExpP2 = svdup_n_f32(Constants->Exp_P2);
    const svfloat32_t ExpP3 = svdup_n_f32(Constants->Exp_P3);
    const svfloat32_t ExpP4 = svdup_n_f32(Constants->Exp_P4);
    const svfloat32_t ExpP5 = svdup_n_f32(Constants->Exp_P5);
    const svfloat32_t ExpP6 = svdup_n_f32(Constants->Exp_P6);
    const svint32_t ExpX7F = svdup_n_s32(Constants->Exp_X7F);

    for (size_t i = 0; i < N; i += svcntw()) {

        const svbool_t pg = svwhilelt_b32_u64(i, N);

        const svfloat32_t Value = svld1_f32(pg, Input + i);

        // Split off the sign so the polynomials operate on |x|. The sign is
        // reapplied at the end with a bitwise OR, which also preserves the
        // sign of zero.
        const svuint32_t SignMask =
            svand_u32_x(pg, svreinterpret_u32_f32(Value), svreinterpret_u32_f32(NegZero));
        svfloat32_t AbsValue =
            svreinterpret_f32_u32(svbic_u32_x(pg, svreinterpret_u32_f32(Value), svreinterpret_u32_f32(NegZero)));
        AbsValue = svmin_f32_x(pg, UpperAbsRange, AbsValue);

        // Small-input polynomial in x^2.
        const svfloat32_t SquareValue = svmul_f32_x(pg, AbsValue, AbsValue);
        svfloat32_t r_small = svmla_f32_x(pg, SmallP1, SmallP0, SquareValue);
        r_small = svmla_f32_x(pg, SmallP2, r_small, SquareValue);
        r_small = svmla_f32_x(pg, SmallP3, r_small, SquareValue);
        r_small = svmla_f32_x(pg, SmallP4, r_small, SquareValue);
        r_small = svmla_f32_x(pg, SmallP5, r_small, SquareValue);
        r_small = svmla_f32_x(pg, AbsValue, r_small, AbsValue);

        // Large-input polynomial, then 1 - exp(-poly).
        svfloat32_t r_big = svmla_f32_x(pg, BigP1, BigP0, AbsValue);
        r_big = svmla_f32_x(pg, BigP2, r_big, AbsValue);
        r_big = svmla_f32_x(pg, BigP3, r_big, AbsValue);
        r_big = svmla_f32_x(pg, BigP4, r_big, AbsValue);
        r_big = svmla_f32_x(pg, BigP5, r_big, AbsValue);
        r_big = svmla_f32_x(pg, BigP6, r_big, AbsValue);
        r_big = svmla_f32_x(pg, AbsValue, r_big, AbsValue);
        r_big = svreinterpret_f32_u32(
            sveor_u32_x(pg, svreinterpret_u32_f32(r_big), svreinterpret_u32_f32(NegZero)));

        // exp(-poly) via log2 range reduction, matching the scalar kernel's
        // Erf exp step.
        r_big = svmax_f32_x(pg, ExpLowerRange, r_big);

        svfloat32_t r_exp = svmla_f32_x(pg, ExpC, r_big, ExpLog2Reciprocal);
        r_exp = svsub_f32_x(pg, r_exp, ExpC);

        svfloat32_t fx = svmla_f32_x(pg, r_big, r_exp, ExpLog2Hi);
        fx = svmla_f32_x(pg, fx, r_exp, ExpLog2Lo);

        svfloat32_t y_big = svmla_f32_x(pg, ExpP1, ExpP0, fx);
        y_big = svmla_f32_x(pg, ExpP2, y_big, fx);
        y_big = svmla_f32_x(pg, ExpP3, y_big, fx);
        y_big = svmla_f32_x(pg, ExpP4, y_big, fx);
        y_big = svmla_f32_x(pg, ExpP5, y_big, fx);
        y_big = svmla_f32_x(pg, ExpP6, y_big, fx);

        // Scale by 2^r_exp: convert to integer, bias by 127 and shift into
        // the float exponent field.
        const svint32_t emm0 = svadd_s32_x(pg, svcvt_s32_f32_x(pg, r_exp), ExpX7F);
        const svfloat32_t scale = svreinterpret_f32_s32(svlsl_n_s32_x(pg, emm0, 23));

        y_big = svmul_f32_x(pg, y_big, scale);
        y_big = svsub_f32_x(pg, One, y_big);

        // Select the branch per lane, then reapply the sign.
        const svbool_t BigLanes = svcmpgt_f32(pg, AbsValue, SplitBoundary);
        svfloat32_t y = svsel_f32(BigLanes, y_big, r_small);
        y = svreinterpret_f32_u32(svorr_u32_x(pg, svreinterpret_u32_f32(y), SignMask));

        svst1_f32(pg, Output + i, y);
    }
}

extern "C" void
MLASCALL
MlasSveLogisticKernelImpl(
    const float* Input,
    float* Output,
    size_t N,
    const MLAS_LOGISTIC_CONSTANTS* Constants
    )
/*++

Routine Description:

    This routine implements the logistic function using SVE, with the same
    odd/even rational polynomial as the scalar/NEON kernels.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    const svfloat32_t LowerRange = svdup_n_f32(Constants->LowerRange);
    const svfloat32_t UpperRange = svdup_n_f32(Constants->UpperRange);

    const svfloat32_t Alpha9 = svdup_n_f32(Constants->alpha_9);
    const svfloat32_t Alpha7 = svdup_n_f32(Constants->alpha_7);
    const svfloat32_t Alpha5 = svdup_n_f32(Constants->alpha_5);
    const svfloat32_t Alpha3 = svdup_n_f32(Constants->alpha_3);
    const svfloat32_t Alpha1 = svdup_n_f32(Constants->alpha_1);

    const svfloat32_t Beta10 = svdup_n_f32(Constants->beta_10);
    const svfloat32_t Beta8 = svdup_n_f32(Constants->beta_8);
    const svfloat32_t Beta6 = svdup_n_f32(Constants->beta_6);
    const svfloat32_t Beta4 = svdup_n_f32(Constants->beta_4);
    const svfloat32_t Beta2 = svdup_n_f32(Constants->beta_2);
    const svfloat32_t Beta0 = svdup_n_f32(Constants->beta_0);

    const svfloat32_t OneHalf = svdup_n_f32(Constants->one_half);
    const svfloat32_t Zero = svdup_n_f32(0.0f);
    const svfloat32_t One = svdup_n_f32(1.0f);

    for (size_t i = 0; i < N; i += svcntw()) {

        const svbool_t pg = svwhilelt_b32_u64(i, N);

        svfloat32_t Value = svld1_f32(pg, Input + i);
        Value = svmax_f32_x(pg, LowerRange, Value);
        Value = svmin_f32_x(pg, UpperRange, Value);

        const svfloat32_t ValueSquared = svmul_f32_x(pg, Value, Value);

        svfloat32_t p = svmla_f32_x(pg, Alpha7, ValueSquared, Alpha9);
        p = svmla_f32_x(pg, Alpha5, p, ValueSquared);
        p = svmla_f32_x(pg, Alpha3, p, ValueSquared);
        p = svmla_f32_x(pg, Alpha1, p, ValueSquared);
        p = svmul_f32_x(pg, p, Value);

        svfloat32_t q = svmla_f32_x(pg, Beta8, ValueSquared, Beta10);
        q = svmla_f32_x(pg, Beta6, q, ValueSquared);
        q = svmla_f32_x(pg, Beta4, q, ValueSquared);
        q = svmla_f32_x(pg, Beta2, q, ValueSquared);
        q = svmla_f32_x(pg, Beta0, q, ValueSquared);

        svfloat32_t y = svadd_f32_x(pg, svdiv_f32_x(pg, p, q), OneHalf);
        y = svmax_f32_x(pg, Zero, y);
        y = svmin_f32_x(pg, One, y);

        svst1_f32(pg, Output + i, y);
    }
}

extern "C" void
MLASCALL
MlasSveComputeExpF32KernelImpl(
    const float* Input,
    float* Output,
    size_t N,
    const MLAS_SVE_EXP_CONSTANTS* Constants
    )
/*++

Routine Description:

    This routine implements exp(x) using SVE. The polynomial approximation is
    taken from the ARM Compute Library:
    https://github.com/ARM-software/ComputeLibrary/blob/9f7a1fb06bc0435d989a9a6a3c0fd2cebfedbf5f/src/core/NEON/SVEMath.inl#L105

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    const svfloat32_t C1 = svreinterpret_f32_u32(svdup_n_u32(Constants->C1));
    const svfloat32_t C2 = svreinterpret_f32_u32(svdup_n_u32(Constants->C2));
    const svfloat32_t C3 = svreinterpret_f32_u32(svdup_n_u32(Constants->C3));
    const svfloat32_t C4 = svreinterpret_f32_u32(svdup_n_u32(Constants->C4));
    const svfloat32_t C5 = svreinterpret_f32_u32(svdup_n_u32(Constants->C5));

    const svfloat32_t Shift = svreinterpret_f32_u32(svdup_n_u32(Constants->Shift));
    const svfloat32_t InvLn2 = svreinterpret_f32_u32(svdup_n_u32(Constants->InvLn2));
    const svfloat32_t NegLn2Hi = svreinterpret_f32_u32(svdup_n_u32(Constants->NegLn2Hi));
    const svfloat32_t NegLn2Lo = svreinterpret_f32_u32(svdup_n_u32(Constants->NegLn2Lo));

    const svfloat32_t Inf = svreinterpret_f32_u32(svdup_n_u32(Constants->Inf));
    const svfloat32_t MaxInput = svreinterpret_f32_u32(svdup_n_u32(Constants->MaxInput));
    const svfloat32_t MinInput = svreinterpret_f32_u32(svdup_n_u32(Constants->MinInput));
    const svfloat32_t Zero = svdup_n_f32(0.0f);

    for (size_t i = 0; i < N; i += svcntw()) {

        const svbool_t pg = svwhilelt_b32_u64(i, N);

        const svfloat32_t Vector = svld1_f32(pg, Input + i);

        // Range reduction: e^x = 2^n * e^r with n = floor(x / ln(2)). Adding
        // the magic shift constant pushes the fractional part of x / ln(2)
        // out of the fp32 mantissa, leaving n + 127 in the low mantissa bits
        // ready to become the exponent of 2^n.
        const svfloat32_t z = svmla_f32_x(pg, Shift, Vector, InvLn2);
        const svfloat32_t n = svsub_f32_x(pg, z, Shift);
        const svfloat32_t Scale =
            svreinterpret_f32_u32(svlsl_n_u32_x(pg, svreinterpret_u32_f32(z), 23));  // 2^n

        // r = x - n * ln(2), computed in two steps for extra accuracy.
        const svfloat32_t r_hi = svmla_f32_x(pg, Vector, n, NegLn2Hi);
        const svfloat32_t r = svmla_f32_x(pg, r_hi, n, NegLn2Lo);

        // Truncated Taylor series of e^r.
        const svfloat32_t r2 = svmul_f32_x(pg, r, r);

        const svfloat32_t p1 = svmul_f32_x(pg, C1, r);
        const svfloat32_t p23 = svmla_f32_x(pg, C2, C3, r);
        const svfloat32_t p45 = svmla_f32_x(pg, C4, C5, r);
        const svfloat32_t p2345 = svmla_f32_x(pg, p23, p45, r2);
        const svfloat32_t p12345 = svmla_f32_x(pg, p1, p2345, r2);

        svfloat32_t poly = svmla_f32_x(pg, Scale, p12345, Scale);

        // Handle underflow and overflow.
        poly = svsel_f32(svcmplt_f32(pg, Vector, MinInput), Zero, poly);
        poly = svsel_f32(svcmpgt_f32(pg, Vector, MaxInput), Inf, poly);

        svst1_f32(pg, Output + i, poly);
    }
}

extern "C" float
MLASCALL
MlasSveComputeSumExpF32KernelImpl(
    const float* Input,
    float* Output,
    size_t N,
    const float* NegativeMaximum,
    const MLAS_EXP_CONSTANTS* Constants,
    const MLAS_SVE_EXP_CONSTANTS* AclConstants
    )
/*++

Routine Description:

    This routine computes the sum of exp(Input[i] + NegativeMaximum), and
    optionally stores the individual exponentials.

Arguments:

    Input - Supplies the input buffer.

    Output - Optionally supplies the output buffer for the exponentials.

    N - Supplies the number of elements to process.

    NegativeMaximum - Supplies the negated maximum of the input values.

Return Value:

    Returns the sum of the exponentials.

--*/
{
#if defined(MLAS_SVE_SUMEXP_FEXPA)
    //
    // EXPERIMENT: exp() via the FEXPA hardware accelerator (as in glibc's
    // SVE expf): FEXPA looks up 2^(n/64) from the low bits of the rounded
    // scaled input, leaving only a 2-term correction polynomial for the
    // residual |r| <= ln2/128. Measured max error 1.16 ulp over
    // [-88.5, 88.5] (280M-point sweep) - well inside the 1e-6 test
    // tolerance. Inputs below the FEXPA subnormal threshold (-87.34) yield
    // ~0 with ~6e-39 absolute error, which is inert in a softmax sum.
    //
    // FEXPA-safe lower clamp: below -127*ln2 (-88.0297) the FEXPA index
    // underflows and wraps into the exponent field, producing NaN (glibc
    // special-cases this zone). Clamping at -88.0 instead of the shared
    // -88.376 changes the true exp() by at most 2e-39 - inert in any sum.
    const svfloat32_t LowerRangeSumExp = svreinterpret_f32_u32(svdup_n_u32(AclConstants->FexpaLowerRange));
    const svfloat32_t InvLn2 = svreinterpret_f32_u32(svdup_n_u32(AclConstants->InvLn2));
    const svfloat32_t NegLn2Hi = svreinterpret_f32_u32(svdup_n_u32(AclConstants->NegLn2Hi));
    const svfloat32_t NegLn2Lo = svreinterpret_f32_u32(svdup_n_u32(AclConstants->NegLn2Lo));
    const svfloat32_t Shift = svreinterpret_f32_u32(svdup_n_u32(AclConstants->FexpaShift));
    const svfloat32_t Half = svreinterpret_f32_u32(svdup_n_u32(AclConstants->OneHalf));
#else
    const svfloat32_t LowerRangeSumExp = svdup_n_f32(Constants->LowerRangeSumExp);
    const svfloat32_t RoundingBias = svdup_n_f32(Constants->RoundingBias);
    const svfloat32_t Log2Reciprocal = svdup_n_f32(Constants->Log2Reciprocal);
    const svfloat32_t Log2High = svdup_n_f32(Constants->Log2High);
    const svfloat32_t Log2Low = svdup_n_f32(Constants->Log2Low);
    const svfloat32_t Poly0 = svdup_n_f32(Constants->poly_0);
    const svfloat32_t Poly1 = svdup_n_f32(Constants->poly_1);
    const svfloat32_t Poly2 = svdup_n_f32(Constants->poly_2);
    const svfloat32_t Poly3 = svdup_n_f32(Constants->poly_3);
    const svfloat32_t Poly4 = svdup_n_f32(Constants->poly_4);
    const svfloat32_t Poly56 = svdup_n_f32(Constants->poly_56);
    const svint32_t MaximumExponent = svdup_n_s32(Constants->MaximumExponent);
#endif

    const svfloat32_t NegativeMaximumVector = svdup_n_f32(*NegativeMaximum);

    //
    // Two vectors per iteration, and per-vector accumulators instead of a
    // horizontal reduction per iteration (a single reduction happens at the
    // end). The second predicate is all-false when only one vector remains;
    // the merge-predicated accumulate keeps its lanes unchanged.
    //

    svfloat32_t Accumulator0 = svdup_n_f32(0.0f);
    svfloat32_t Accumulator1 = svdup_n_f32(0.0f);

    const size_t VL = svcntw();

    for (size_t i = 0; i < N; i += 2 * VL) {

        const svbool_t pg0 = svwhilelt_b32_u64(i, N);
        const svbool_t pg1 = svwhilelt_b32_u64(i + VL, N);

        svfloat32_t Vector0 = svld1_f32(pg0, Input + i);
        svfloat32_t Vector1 = svld1_f32(pg1, Input + i + VL);

        Vector0 = svadd_f32_x(pg0, Vector0, NegativeMaximumVector);
        Vector1 = svadd_f32_x(pg1, Vector1, NegativeMaximumVector);
        Vector0 = svmax_f32_x(pg0, LowerRangeSumExp, Vector0);
        Vector1 = svmax_f32_x(pg1, LowerRangeSumExp, Vector1);

#if defined(MLAS_SVE_SUMEXP_FEXPA)
        // exp(x) = FEXPA(z) * (1 + r + r^2/2): the shift constant rounds
        // x/ln2 to multiples of 1/64 with the FEXPA exponent bias baked into
        // its low mantissa bits.
        const svfloat32_t z0 = svmla_f32_x(pg0, Shift, Vector0, InvLn2);
        const svfloat32_t z1 = svmla_f32_x(pg1, Shift, Vector1, InvLn2);
        const svfloat32_t n0 = svsub_f32_x(pg0, z0, Shift);
        const svfloat32_t n1 = svsub_f32_x(pg1, z1, Shift);

        svfloat32_t r0 = svmla_f32_x(pg0, Vector0, n0, NegLn2Hi);
        svfloat32_t r1 = svmla_f32_x(pg1, Vector1, n1, NegLn2Hi);
        r0 = svmla_f32_x(pg0, r0, n0, NegLn2Lo);
        r1 = svmla_f32_x(pg1, r1, n1, NegLn2Lo);

        const svfloat32_t scale0 = svexpa_f32(svreinterpret_u32_f32(z0));
        const svfloat32_t scale1 = svexpa_f32(svreinterpret_u32_f32(z1));

        const svfloat32_t r2_0 = svmul_f32_x(pg0, r0, r0);
        const svfloat32_t r2_1 = svmul_f32_x(pg1, r1, r1);
        const svfloat32_t poly0 = svmla_f32_x(pg0, r0, r2_0, Half);
        const svfloat32_t poly1 = svmla_f32_x(pg1, r1, r2_1, Half);

        svfloat32_t p0 = svmla_f32_x(pg0, scale0, scale0, poly0);
        svfloat32_t p1 = svmla_f32_x(pg1, scale1, scale1, poly1);
#else
        // exp(x + NegativeMaximum), matching the scalar kernel's operation
        // order (including the historical double application of poly_56).
        const svfloat32_t biased0 = svmla_f32_x(pg0, RoundingBias, Vector0, Log2Reciprocal);
        const svfloat32_t biased1 = svmla_f32_x(pg1, RoundingBias, Vector1, Log2Reciprocal);
        const svfloat32_t m0 = svsub_f32_x(pg0, biased0, RoundingBias);
        const svfloat32_t m1 = svsub_f32_x(pg1, biased1, RoundingBias);

        Vector0 = svmla_f32_x(pg0, Vector0, m0, Log2High);
        Vector1 = svmla_f32_x(pg1, Vector1, m1, Log2High);
        Vector0 = svmla_f32_x(pg0, Vector0, m0, Log2Low);
        Vector1 = svmla_f32_x(pg1, Vector1, m1, Log2Low);

        svint32_t normal0 = svlsl_n_s32_x(pg0, svreinterpret_s32_f32(biased0), 23);
        svint32_t normal1 = svlsl_n_s32_x(pg1, svreinterpret_s32_f32(biased1), 23);
        normal0 = svadd_s32_x(pg0, normal0, MaximumExponent);
        normal1 = svadd_s32_x(pg1, normal1, MaximumExponent);

        svfloat32_t p0 = svmla_f32_x(pg0, Poly1, Poly0, Vector0);
        svfloat32_t p1 = svmla_f32_x(pg1, Poly1, Poly0, Vector1);
        p0 = svmla_f32_x(pg0, Poly2, p0, Vector0);
        p1 = svmla_f32_x(pg1, Poly2, p1, Vector1);
        p0 = svmla_f32_x(pg0, Poly3, p0, Vector0);
        p1 = svmla_f32_x(pg1, Poly3, p1, Vector1);
        p0 = svmla_f32_x(pg0, Poly4, p0, Vector0);
        p1 = svmla_f32_x(pg1, Poly4, p1, Vector1);
        p0 = svmla_f32_x(pg0, Poly56, p0, Vector0);
        p1 = svmla_f32_x(pg1, Poly56, p1, Vector1);
        p0 = svmla_f32_x(pg0, Poly56, p0, Vector0);
        p1 = svmla_f32_x(pg1, Poly56, p1, Vector1);

        p0 = svmul_f32_x(pg0, p0, svreinterpret_f32_s32(normal0));
        p1 = svmul_f32_x(pg1, p1, svreinterpret_f32_s32(normal1));
#endif

        if (Output != nullptr) {
            svst1_f32(pg0, Output + i, p0);
            svst1_f32(pg1, Output + i + VL, p1);
        }

        Accumulator0 = svadd_f32_m(pg0, Accumulator0, p0);
        Accumulator1 = svadd_f32_m(pg1, Accumulator1, p1);
    }

    const svbool_t ptrue = svptrue_b32();
    return svaddv_f32(ptrue, svadd_f32_x(ptrue, Accumulator0, Accumulator1));
}

extern "C" void
MLASCALL
MlasSveTanhKernelImpl(
    const float* Input,
    float* Output,
    size_t N,
    const MLAS_TANH_CONSTANTS* Constants
    )
/*++

Routine Description:

    This routine implements the SVE kernel for the hyperbolic tangent
    function: the same clamped P13/Q6 rational polynomial as the generic
    kernel, evaluated per whilelt-predicated vector.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Constants - Supplies the coefficient table (passed by pointer so the
        frozen machine-code variant needs no data relocations).

Return Value:

    None.

--*/
{
    const svfloat32_t LowerRange = svdup_n_f32(Constants->LowerRange);
    const svfloat32_t UpperRange = svdup_n_f32(Constants->UpperRange);

    const svfloat32_t Alpha13 = svdup_n_f32(Constants->alpha_13);
    const svfloat32_t Alpha11 = svdup_n_f32(Constants->alpha_11);
    const svfloat32_t Alpha9 = svdup_n_f32(Constants->alpha_9);
    const svfloat32_t Alpha7 = svdup_n_f32(Constants->alpha_7);
    const svfloat32_t Alpha5 = svdup_n_f32(Constants->alpha_5);
    const svfloat32_t Alpha3 = svdup_n_f32(Constants->alpha_3);
    const svfloat32_t Alpha1 = svdup_n_f32(Constants->alpha_1);

    const svfloat32_t Beta6 = svdup_n_f32(Constants->beta_6);
    const svfloat32_t Beta4 = svdup_n_f32(Constants->beta_4);
    const svfloat32_t Beta2 = svdup_n_f32(Constants->beta_2);
    const svfloat32_t Beta0 = svdup_n_f32(Constants->beta_0);

    for (size_t i = 0; i < N; i += svcntw()) {

        const svbool_t pg = svwhilelt_b32_u64(i, N);

        svfloat32_t Value = svld1_f32(pg, Input + i);
        Value = svmax_f32_x(pg, LowerRange, Value);
        Value = svmin_f32_x(pg, UpperRange, Value);

        const svfloat32_t ValueSquared = svmul_f32_x(pg, Value, Value);

        svfloat32_t p = svmla_f32_x(pg, Alpha11, ValueSquared, Alpha13);
        p = svmla_f32_x(pg, Alpha9, p, ValueSquared);
        p = svmla_f32_x(pg, Alpha7, p, ValueSquared);
        p = svmla_f32_x(pg, Alpha5, p, ValueSquared);
        p = svmla_f32_x(pg, Alpha3, p, ValueSquared);
        p = svmla_f32_x(pg, Alpha1, p, ValueSquared);
        p = svmul_f32_x(pg, p, Value);

        svfloat32_t q = svmla_f32_x(pg, Beta4, ValueSquared, Beta6);
        q = svmla_f32_x(pg, Beta2, q, ValueSquared);
        q = svmla_f32_x(pg, Beta0, q, ValueSquared);

        svst1_f32(pg, Output + i, svdiv_f32_x(pg, p, q));
    }
}

extern "C" float
MLASCALL
MlasSveReduceMaximumF32KernelImpl(
    const float* Input,
    size_t N,
    float MinimumValue
    )
/*++

Routine Description:

    This routine computes the maximum value of the supplied buffer.

Arguments:

    Input - Supplies the input buffer.

    N - Supplies the number of elements to process.

Return Value:

    Returns the maximum value of the supplied buffer.

--*/
{
    const svbool_t ptrue = svptrue_b32();
    const size_t veclen = svcntw();

    svfloat32_t MaximumVector0 = svdup_n_f32(MinimumValue);

    //
    // Unrolled main loop: four independent maximum accumulators.
    //

    if (N >= veclen * 4) {

        svfloat32_t MaximumVector1 = MaximumVector0;
        svfloat32_t MaximumVector2 = MaximumVector0;
        svfloat32_t MaximumVector3 = MaximumVector0;

        while (N >= veclen * 4) {

            MaximumVector0 = svmax_f32_x(ptrue, MaximumVector0, svld1_f32(ptrue, Input));
            MaximumVector1 = svmax_f32_x(ptrue, MaximumVector1, svld1_f32(ptrue, Input + veclen));
            MaximumVector2 = svmax_f32_x(ptrue, MaximumVector2, svld1_f32(ptrue, Input + 2 * veclen));
            MaximumVector3 = svmax_f32_x(ptrue, MaximumVector3, svld1_f32(ptrue, Input + 3 * veclen));

            Input += veclen * 4;
            N -= veclen * 4;
        }

        MaximumVector0 = svmax_f32_x(ptrue, MaximumVector0, MaximumVector1);
        MaximumVector2 = svmax_f32_x(ptrue, MaximumVector2, MaximumVector3);
        MaximumVector0 = svmax_f32_x(ptrue, MaximumVector0, MaximumVector2);
    }

    for (size_t i = 0; i < N; i += veclen) {

        const svbool_t pg = svwhilelt_b32_u64(i, N);
        MaximumVector0 = svmax_f32_m(pg, MaximumVector0, svld1_f32(pg, Input + i));
    }

    return svmaxv_f32(ptrue, MaximumVector0);
}

extern "C" void
MLASCALL
MlasSveReduceMinimumMaximumF32KernelImpl(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
    )
/*++

Routine Description:

    This routine computes the minimum and maximum values of the supplied
    buffer.

Arguments:

    Input - Supplies the input buffer.

    Min - Receives the minimum value of the supplied buffer.

    Max - Receives the maximum value of the supplied buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    const svbool_t ptrue = svptrue_b32();
    const size_t veclen = svcntw();

    svfloat32_t MinimumVector0 = svdup_n_f32(std::numeric_limits<float>::max());
    svfloat32_t MaximumVector0 = svdup_n_f32(std::numeric_limits<float>::lowest());

    //
    // Unrolled main loop: four independent accumulator pairs with
    // unpredicated operations, matching the generic kernel's structure (a
    // single predicated accumulator loop measured 2-3.9x slower than the
    // generic NEON kernel on Cortex-A725/X925).
    //

    if (N >= veclen * 4) {

        svfloat32_t MinimumVector1 = MinimumVector0;
        svfloat32_t MinimumVector2 = MinimumVector0;
        svfloat32_t MinimumVector3 = MinimumVector0;

        svfloat32_t MaximumVector1 = MaximumVector0;
        svfloat32_t MaximumVector2 = MaximumVector0;
        svfloat32_t MaximumVector3 = MaximumVector0;

        while (N >= veclen * 4) {

            const svfloat32_t InputVector0 = svld1_f32(ptrue, Input);
            const svfloat32_t InputVector1 = svld1_f32(ptrue, Input + veclen);
            const svfloat32_t InputVector2 = svld1_f32(ptrue, Input + 2 * veclen);
            const svfloat32_t InputVector3 = svld1_f32(ptrue, Input + 3 * veclen);

            MinimumVector0 = svmin_f32_x(ptrue, MinimumVector0, InputVector0);
            MinimumVector1 = svmin_f32_x(ptrue, MinimumVector1, InputVector1);
            MinimumVector2 = svmin_f32_x(ptrue, MinimumVector2, InputVector2);
            MinimumVector3 = svmin_f32_x(ptrue, MinimumVector3, InputVector3);

            MaximumVector0 = svmax_f32_x(ptrue, MaximumVector0, InputVector0);
            MaximumVector1 = svmax_f32_x(ptrue, MaximumVector1, InputVector1);
            MaximumVector2 = svmax_f32_x(ptrue, MaximumVector2, InputVector2);
            MaximumVector3 = svmax_f32_x(ptrue, MaximumVector3, InputVector3);

            Input += veclen * 4;
            N -= veclen * 4;
        }

        MinimumVector0 = svmin_f32_x(ptrue, MinimumVector0, MinimumVector1);
        MinimumVector2 = svmin_f32_x(ptrue, MinimumVector2, MinimumVector3);
        MinimumVector0 = svmin_f32_x(ptrue, MinimumVector0, MinimumVector2);

        MaximumVector0 = svmax_f32_x(ptrue, MaximumVector0, MaximumVector1);
        MaximumVector2 = svmax_f32_x(ptrue, MaximumVector2, MaximumVector3);
        MaximumVector0 = svmax_f32_x(ptrue, MaximumVector0, MaximumVector2);
    }

    for (size_t i = 0; i < N; i += veclen) {

        const svbool_t pg = svwhilelt_b32_u64(i, N);
        const svfloat32_t Vector = svld1_f32(pg, Input + i);

        MinimumVector0 = svmin_f32_m(pg, MinimumVector0, Vector);
        MaximumVector0 = svmax_f32_m(pg, MaximumVector0, Vector);
    }

    *Min = svminv_f32(ptrue, MinimumVector0);
    *Max = svmaxv_f32(ptrue, MaximumVector0);
}

extern "C" void
MLASCALL
MlasSveComputeSoftmaxOutputF32KernelImpl(
    float* Output,
    size_t N,
    const float* Parameters
    )
/*++

Routine Description:

    This routine scales the softmax exponentials in place by the reciprocal
    of their sum.

Arguments:

    Output - Supplies the exponentials, and receives the scaled values.

    N - Supplies the number of elements to process.

    Parameters - Supplies the scale in element 0.

Return Value:

    None.

--*/
{
    const svfloat32_t ScaleVector = svdup_n_f32(Parameters[0]);

    for (size_t i = 0; i < N; i += svcntw()) {

        const svbool_t pg = svwhilelt_b32_u64(i, N);
        const svfloat32_t Vector = svmul_f32_x(pg, ScaleVector, svld1_f32(pg, Output + i));
        svst1_f32(pg, Output + i, Vector);
    }
}

extern "C" void
MLASCALL
MlasSveComputeLogSoftmaxOutputF32KernelImpl(
    const float* Input,
    float* Output,
    size_t N,
    const float* Parameters
    )
/*++

Routine Description:

    This routine computes Input + NegativeMaximum - log(SumExp) as the final
    log-softmax output step.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Parameters - Supplies NegativeMaximum in element 0 and log(SumExp) in
        element 1.

Return Value:

    None.

--*/
{
    const svfloat32_t NegativeMaximumVector = svdup_n_f32(Parameters[0]);
    const svfloat32_t LogarithmVector = svdup_n_f32(Parameters[1]);

    for (size_t i = 0; i < N; i += svcntw()) {

        const svbool_t pg = svwhilelt_b32_u64(i, N);

        svfloat32_t Vector = svld1_f32(pg, Input + i);
        Vector = svadd_f32_x(pg, Vector, NegativeMaximumVector);
        Vector = svsub_f32_x(pg, Vector, LogarithmVector);

        svst1_f32(pg, Output + i, Vector);
    }
}
