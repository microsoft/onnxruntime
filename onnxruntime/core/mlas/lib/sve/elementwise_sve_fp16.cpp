/*++

Copyright 2025 FUJITSU LIMITED
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    elementwise_sve_fp16.cpp

Abstract:

    This module implements the float16 elementwise kernel bodies using SVE
    intrinsics: Tanh, Erf and the Gelu passes.

    Only the vector loops live here, as extern "C" *Impl functions with no
    data relocations (constants are materialized from immediates), so the
    machine code can be captured verbatim into the portable assembly variant
    (elementwise_sve_asm.S), which exports the same symbols. The public
    MlasSve* entry points live in elementwise_sve_dispatch.cpp.

    The kernels share one loop idiom: a whilelt-governed predicate covers the
    vector body and the tail in the same loop, and every constant is
    broadcast once before the loop. Each kernel divides the way the kernel it
    replaces did, so the results stay bit-identical: Erf and Gelu use the
    reciprocal-estimate + two Newton-Raphson steps idiom, while Tanh uses the
    hardware divide.

    This file is compiled with SVE and fp16 enabled via its per-file -march
    compile flag (see onnxruntime_mlas.cmake).

--*/

#include "mlasi_sve.h"

#include <cmath>  // M_SQRT1_2

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)

#include <arm_sve.h>

namespace {

//
// Coefficients of the rational tanh approximation.
//

constexpr struct {
    __fp16 LowerRange;
    __fp16 UpperRange;
    __fp16 alpha_7;
    __fp16 alpha_5;
    __fp16 alpha_3;
    __fp16 alpha_1;
    __fp16 beta_6;
    __fp16 beta_4;
    __fp16 beta_2;
    __fp16 beta_0;
} MlasSveTanhConstantsFp16 = {
    -3.515625f,
    3.515625f,
    5.960464477539063e-08f,
    1.4841556549072266e-05f,
    0.000637054443359375f,
    0.004894256591796875f,
    1.1920928955078125e-06f,
    0.00011855363845825195f,
    0.0022678375244140625f,
    0.004894256591796875f,
};

//
// Coefficients of the Abramowitz-Stegun erf approximation:
// erf(x) ~= 1 - (a1*t + a2*t^2 + ... + a5*t^5) * exp(-x^2), t = 1/(1 + p*x),
// with inputs saturated to +-1 beyond |x| = 4.
//

constexpr struct {
    __fp16 p;
    __fp16 a1;
    __fp16 a2;
    __fp16 a3;
    __fp16 a4;
    __fp16 a5;
    __fp16 SaturationThreshold;
} MlasSveErfConstantsFp16 = {
    0.328f,
    0.2505f,
    -0.2881f,
    1.4102f,
    -1.423f,
    1.0547f,
    4.0f,
};

//
// Coefficients of the rational approximation of exp(-x) over [0, 6], used by
// the erf kernel for the exp(-x^2) factor.
//

constexpr struct {
    __fp16 UpperRange;
    __fp16 c0;
    __fp16 c1;
    __fp16 c2;
    __fp16 d0;
    __fp16 d1;
    __fp16 d2;
} MlasSveExpNegConstantsFp16 = {
    6.0f,
    1.330f,
    -0.390f,
    0.0288f,
    1.338f,
    0.848f,
    0.467f,
};

//
// Gelu constants: sqrt(1/2) for the erf form, and the tanh-form inner
// polynomial coefficients sqrt(2/pi) and 0.044715 * sqrt(2/pi), with the
// tanh argument clamped to +-5.
//

constexpr struct {
    __fp16 OneHalf;
    __fp16 One;
    __fp16 Sqrt1_2;
    __fp16 TanhB;
    __fp16 TanhC;
    __fp16 TanhArgLowerRange;
    __fp16 TanhArgUpperRange;
} MlasSveGeluConstantsFp16 = {
    0.5f,
    1.0f,
    static_cast<float>(M_SQRT1_2),
    0.7978845608028654f,
    0.035677408136300125f,
    -5.0f,
    5.0f,
};

//
// Computes 1/x with reciprocal estimate plus two Newton-Raphson steps.
//

MLAS_FORCEINLINE
svfloat16_t
MlasSveReciprocalFp16(
    svbool_t pg,
    svfloat16_t x
    )
{
    svfloat16_t recip = svrecpe_f16(x);
    recip = svmul_f16_x(pg, recip, svrecps_f16(x, recip));
    recip = svmul_f16_x(pg, recip, svrecps_f16(x, recip));
    return recip;
}

}  // namespace

extern "C" void
MLASCALL
MlasSveTanhFP16KernelImpl(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements tanh(x) for fp16 using the rational polynomial
    approximation p(x^2)*x / q(x^2) on the clamped input.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    const __fp16* input = reinterpret_cast<const __fp16*>(Input);
    __fp16* output = reinterpret_cast<__fp16*>(Output);

    const svfloat16_t LowerRange = svdup_n_f16(MlasSveTanhConstantsFp16.LowerRange);
    const svfloat16_t UpperRange = svdup_n_f16(MlasSveTanhConstantsFp16.UpperRange);
    const svfloat16_t Alpha7 = svdup_n_f16(MlasSveTanhConstantsFp16.alpha_7);
    const svfloat16_t Alpha5 = svdup_n_f16(MlasSveTanhConstantsFp16.alpha_5);
    const svfloat16_t Alpha3 = svdup_n_f16(MlasSveTanhConstantsFp16.alpha_3);
    const svfloat16_t Alpha1 = svdup_n_f16(MlasSveTanhConstantsFp16.alpha_1);
    const svfloat16_t Beta6 = svdup_n_f16(MlasSveTanhConstantsFp16.beta_6);
    const svfloat16_t Beta4 = svdup_n_f16(MlasSveTanhConstantsFp16.beta_4);
    const svfloat16_t Beta2 = svdup_n_f16(MlasSveTanhConstantsFp16.beta_2);
    const svfloat16_t Beta0 = svdup_n_f16(MlasSveTanhConstantsFp16.beta_0);

    for (size_t i = 0; i < N; i += svcnth()) {

        const svbool_t pg = svwhilelt_b16_u64(i, N);

        svfloat16_t x = svld1_f16(pg, input + i);
        x = svmin_f16_x(pg, x, UpperRange);
        x = svmax_f16_x(pg, x, LowerRange);

        const svfloat16_t x2 = svmul_f16_x(pg, x, x);

        svfloat16_t p = svmla_f16_x(pg, Alpha5, Alpha7, x2);
        p = svmla_f16_x(pg, Alpha3, p, x2);
        p = svmla_f16_x(pg, Alpha1, p, x2);
        p = svmul_f16_x(pg, p, x);

        svfloat16_t q = svmla_f16_x(pg, Beta4, Beta6, x2);
        q = svmla_f16_x(pg, Beta2, q, x2);
        q = svmla_f16_x(pg, Beta0, q, x2);

        svst1_f16(pg, output + i, svdiv_f16_x(pg, p, q));
    }
}

extern "C" void
MLASCALL
MlasSveErfFP16KernelImpl(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements erf(x) for fp16 using the Abramowitz-Stegun
    approximation, saturating to sign(x) beyond |x| = 4.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    const __fp16* input = reinterpret_cast<const __fp16*>(Input);
    __fp16* output = reinterpret_cast<__fp16*>(Output);

    const svfloat16_t P = svdup_n_f16(MlasSveErfConstantsFp16.p);
    const svfloat16_t A1 = svdup_n_f16(MlasSveErfConstantsFp16.a1);
    const svfloat16_t A2 = svdup_n_f16(MlasSveErfConstantsFp16.a2);
    const svfloat16_t A3 = svdup_n_f16(MlasSveErfConstantsFp16.a3);
    const svfloat16_t A4 = svdup_n_f16(MlasSveErfConstantsFp16.a4);
    const svfloat16_t A5 = svdup_n_f16(MlasSveErfConstantsFp16.a5);
    const svfloat16_t Threshold = svdup_n_f16(MlasSveErfConstantsFp16.SaturationThreshold);

    const svfloat16_t ExpUpperRange = svdup_n_f16(MlasSveExpNegConstantsFp16.UpperRange);
    const svfloat16_t ExpC0 = svdup_n_f16(MlasSveExpNegConstantsFp16.c0);
    const svfloat16_t ExpC1 = svdup_n_f16(MlasSveExpNegConstantsFp16.c1);
    const svfloat16_t ExpC2 = svdup_n_f16(MlasSveExpNegConstantsFp16.c2);
    const svfloat16_t ExpD0 = svdup_n_f16(MlasSveExpNegConstantsFp16.d0);
    const svfloat16_t ExpD1 = svdup_n_f16(MlasSveExpNegConstantsFp16.d1);
    const svfloat16_t ExpD2 = svdup_n_f16(MlasSveExpNegConstantsFp16.d2);

    const svfloat16_t One = svdup_n_f16(__fp16(1.0f));
    const svfloat16_t NegOne = svdup_n_f16(__fp16(-1.0f));
    const svfloat16_t Zero = svdup_n_f16(__fp16(0.0f));

    for (size_t i = 0; i < N; i += svcnth()) {

        const svbool_t pg = svwhilelt_b16_u64(i, N);

        const svfloat16_t x = svld1_f16(pg, input + i);

        const svbool_t NegativeLanes = svcmplt_f16(pg, x, Zero);
        const svfloat16_t Sign = svsel_f16(NegativeLanes, NegOne, One);

        const svfloat16_t AbsX = svabs_f16_x(pg, x);
        const svbool_t ApproxLanes = svcmplt_f16(pg, AbsX, Threshold);
        const svfloat16_t AbsXClamped = svmin_f16_x(pg, AbsX, Threshold);

        // t = 1 / (1 + p * |x|)
        const svfloat16_t Denominator = svmla_f16_x(pg, One, P, AbsXClamped);
        const svfloat16_t t = MlasSveReciprocalFp16(pg, Denominator);

        const svfloat16_t t2 = svmul_f16_x(pg, t, t);
        const svfloat16_t t3 = svmul_f16_x(pg, t2, t);
        const svfloat16_t t4 = svmul_f16_x(pg, t3, t);
        const svfloat16_t t5 = svmul_f16_x(pg, t4, t);

        svfloat16_t Poly = svmul_f16_x(pg, A1, t);
        Poly = svmla_f16_x(pg, Poly, A2, t2);
        Poly = svmla_f16_x(pg, Poly, A3, t3);
        Poly = svmla_f16_x(pg, Poly, A4, t4);
        Poly = svmla_f16_x(pg, Poly, A5, t5);

        // exp(-x^2) via the rational approximation over [0, 6].
        svfloat16_t x2 = svmul_f16_x(pg, AbsXClamped, AbsXClamped);
        x2 = svmin_f16_x(pg, x2, ExpUpperRange);

        const svfloat16_t x4 = svmul_f16_x(pg, x2, x2);

        svfloat16_t Numerator = svmla_f16_x(pg, ExpC0, ExpC1, x2);
        Numerator = svmla_f16_x(pg, Numerator, ExpC2, x4);

        svfloat16_t Denominator2 = svmla_f16_x(pg, ExpD0, ExpD1, x2);
        Denominator2 = svmla_f16_x(pg, Denominator2, ExpD2, x4);

        const svfloat16_t ExpNegX2 =
            svmul_f16_x(pg, Numerator, MlasSveReciprocalFp16(pg, Denominator2));

        // erf ~= sign * (1 - poly * exp(-x^2)), clamped to [-1, 1]; saturated
        // lanes produce sign(x) directly.
        svfloat16_t Erf = svmul_f16_x(pg, Sign, svsub_f16_x(pg, One, svmul_f16_x(pg, Poly, ExpNegX2)));
        Erf = svmin_f16_x(pg, Erf, One);
        Erf = svmax_f16_x(pg, Erf, NegOne);

        const svfloat16_t Result = svsel_f16(ApproxLanes, Erf, Sign);

        svst1_f16(pg, output + i, Result);
    }
}

//
// Gelu is orchestrated from elementwise_sve_dispatch.cpp as three vector
// passes (argument preparation, tanh/erf, combine) so that each pass is a
// single self-contained loop.
//

extern "C" void
MLASCALL
MlasSveGeluTanhArgFP16KernelImpl(
    const MLAS_FP16* Input,
    MLAS_FP16* Temp,
    size_t N
    )
/*++

Routine Description:

    This routine computes the clamped tanh-form Gelu inner argument
    sqrt(2/pi) * (x + 0.044715 * x^3).

--*/
{
    const __fp16* input = reinterpret_cast<const __fp16*>(Input);
    __fp16* temp = reinterpret_cast<__fp16*>(Temp);

    const svfloat16_t TanhB = svdup_n_f16(MlasSveGeluConstantsFp16.TanhB);
    const svfloat16_t TanhC = svdup_n_f16(MlasSveGeluConstantsFp16.TanhC);
    const svfloat16_t ArgLower = svdup_n_f16(MlasSveGeluConstantsFp16.TanhArgLowerRange);
    const svfloat16_t ArgUpper = svdup_n_f16(MlasSveGeluConstantsFp16.TanhArgUpperRange);

    for (size_t i = 0; i < N; i += svcnth()) {

        const svbool_t pg = svwhilelt_b16_u64(i, N);

        const svfloat16_t x = svld1_f16(pg, input + i);
        const svfloat16_t x2 = svmul_f16_x(pg, x, x);

        svfloat16_t TanhArg = svmul_f16_x(pg, x, svmla_f16_x(pg, TanhB, TanhC, x2));
        TanhArg = svmax_f16_x(pg, ArgLower, svmin_f16_x(pg, TanhArg, ArgUpper));

        svst1_f16(pg, temp + i, TanhArg);
    }
}

extern "C" void
MLASCALL
MlasSveGeluScaleFP16KernelImpl(
    const MLAS_FP16* Input,
    MLAS_FP16* Temp,
    size_t N
    )
/*++

Routine Description:

    This routine scales the input by sqrt(1/2) for the exact (erf) Gelu form.

--*/
{
    const __fp16* input = reinterpret_cast<const __fp16*>(Input);
    __fp16* temp = reinterpret_cast<__fp16*>(Temp);

    const svfloat16_t Sqrt1_2 = svdup_n_f16(MlasSveGeluConstantsFp16.Sqrt1_2);

    for (size_t i = 0; i < N; i += svcnth()) {

        const svbool_t pg = svwhilelt_b16_u64(i, N);

        const svfloat16_t x = svld1_f16(pg, input + i);
        svst1_f16(pg, temp + i, svmul_f16_x(pg, x, Sqrt1_2));
    }
}

extern "C" void
MLASCALL
MlasSveGeluCombineFP16KernelImpl(
    const MLAS_FP16* Input,
    const MLAS_FP16* Inner,
    MLAS_FP16* Output,
    size_t N
    )
/*++

Routine Description:

    This routine combines the Gelu factors: 0.5 * x * (1 + tanh/erf value).

--*/
{
    const __fp16* input = reinterpret_cast<const __fp16*>(Input);
    const __fp16* inner = reinterpret_cast<const __fp16*>(Inner);
    __fp16* output = reinterpret_cast<__fp16*>(Output);

    const svfloat16_t OneHalf = svdup_n_f16(MlasSveGeluConstantsFp16.OneHalf);
    const svfloat16_t One = svdup_n_f16(MlasSveGeluConstantsFp16.One);

    for (size_t i = 0; i < N; i += svcnth()) {

        const svbool_t pg = svwhilelt_b16_u64(i, N);

        const svfloat16_t x = svld1_f16(pg, input + i);
        const svfloat16_t InnerValue = svld1_f16(pg, inner + i);
        const svfloat16_t Result =
            svmul_f16_x(pg, OneHalf, svmul_f16_x(pg, x, svadd_f16_x(pg, One, InnerValue)));

        svst1_f16(pg, output + i, Result);
    }
}

#endif  // fp16 vector intrinsics supported
