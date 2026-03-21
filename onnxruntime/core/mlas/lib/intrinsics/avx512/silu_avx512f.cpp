/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    silu_avx512f.cpp

Abstract:

    This module implements routines to compute the SiLU function with AVX512F
    intrinsics.

--*/

#include <cstdint>
#include <limits>

#include "mlasi.h"

namespace {

struct SiluAvx512Constants {
    static constexpr int32_t SignBitMask = INT32_MIN;
    static constexpr int32_t PositiveMask = INT32_MAX;
    static constexpr float Half = 0.5f;
    static constexpr float One = 1.0f;
    static constexpr float Two = 2.0f;
    static constexpr float Ln2 = 0.693147182f;
    static constexpr float Log2EF = 1.44269502f;
    static constexpr float ExpLnFltMax = 88.3762589f;
    static constexpr float ExpLnFltMin = -87.3365479f;

    static constexpr float P1 = 0.999999701f;
    static constexpr float P2 = 0.499991506f;
    static constexpr float P3 = 0.166676521f;
    static constexpr float P4 = 0.0418978221f;
    static constexpr float P5 = 0.00828929059f;
};

MLAS_FORCEINLINE __m512
MlasExpApproxAvx512(
    __m512 Value
    )
{
    const __m512 Half = _mm512_set1_ps(SiluAvx512Constants::Half);
    const __m512 One = _mm512_set1_ps(SiluAvx512Constants::One);
    const __m512 Two = _mm512_set1_ps(SiluAvx512Constants::Two);
    const __m512 Ln2 = _mm512_set1_ps(SiluAvx512Constants::Ln2);
    const __m512 Log2EF = _mm512_set1_ps(SiluAvx512Constants::Log2EF);
    const __m512 ExpLnFltMax = _mm512_set1_ps(SiluAvx512Constants::ExpLnFltMax);
    const __m512 ExpLnFltMin = _mm512_set1_ps(SiluAvx512Constants::ExpLnFltMin);
    const __m512 P1 = _mm512_set1_ps(SiluAvx512Constants::P1);
    const __m512 P2 = _mm512_set1_ps(SiluAvx512Constants::P2);
    const __m512 P3 = _mm512_set1_ps(SiluAvx512Constants::P3);
    const __m512 P4 = _mm512_set1_ps(SiluAvx512Constants::P4);
    const __m512 P5 = _mm512_set1_ps(SiluAvx512Constants::P5);
    const __m512i ExponentBias = _mm512_set1_epi32(127);

    const __mmask16 UnderflowMask = _mm512_cmp_ps_mask(Value, ExpLnFltMin, _CMP_LT_OQ);

    Value = _mm512_min_ps(Value, ExpLnFltMax);
    Value = _mm512_max_ps(Value, ExpLnFltMin);

    __m512 Fx = _mm512_fmadd_ps(Value, Log2EF, Half);
    Fx = _mm512_floor_ps(Fx);

    const __m512 R = _mm512_fnmadd_ps(Fx, Ln2, Value);

    const __m512 NMinusOne = _mm512_sub_ps(Fx, One);
    __m512i Exponent = _mm512_cvttps_epi32(NMinusOne);
    Exponent = _mm512_add_epi32(Exponent, ExponentBias);
    Exponent = _mm512_slli_epi32(Exponent, 23);
    Exponent = _mm512_mask_mov_epi32(Exponent, UnderflowMask, _mm512_setzero_si512());
    const __m512 Pow2NMinusOne = _mm512_castsi512_ps(Exponent);

    __m512 Y = P5;
    Y = _mm512_fmadd_ps(Y, R, P4);
    Y = _mm512_fmadd_ps(Y, R, P3);
    Y = _mm512_fmadd_ps(Y, R, P2);
    Y = _mm512_fmadd_ps(Y, R, P1);
    Y = _mm512_fmadd_ps(Y, R, One);

    Y = _mm512_mul_ps(Y, Pow2NMinusOne);
    Y = _mm512_mul_ps(Y, Two);
    return Y;
}

MLAS_FORCEINLINE __m512
MlasLogisticApproxAvx512(
    __m512 Value
    )
{
    const __m512 One = _mm512_set1_ps(1.0f);
    const __m512 Zero = _mm512_setzero_ps();
    const __m512 SignMask = _mm512_castsi512_ps(_mm512_set1_epi32(SiluAvx512Constants::SignBitMask));
    const __m512 PositiveMask = _mm512_castsi512_ps(_mm512_set1_epi32(SiluAvx512Constants::PositiveMask));

    const __m512 XAbs = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(Value), _mm512_castps_si512(PositiveMask)));
    const __m512 XNeg = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(XAbs), _mm512_castps_si512(SignMask)));

    const __m512 E = MlasExpApproxAvx512(XNeg);
    const __m512 Y = _mm512_div_ps(E, _mm512_add_ps(E, One));
    const __m512 OneMinusY = _mm512_sub_ps(One, Y);
    const __mmask16 NegativeMask = _mm512_cmp_ps_mask(Value, Zero, _CMP_LT_OQ);

    return _mm512_mask_blend_ps(NegativeMask, OneMinusY, Y);
}

MLAS_FORCEINLINE __m512
MlasComputeSiluVectorAvx512(
    __m512 X
    )
{
    __m512 Result = _mm512_mul_ps(X, MlasLogisticApproxAvx512(X));

    const __mmask16 NaNMask = _mm512_cmp_ps_mask(X, X, _CMP_UNORD_Q);
    Result = _mm512_mask_mov_ps(Result, NaNMask, X);

    return Result;
}

}  // namespace

void
MLASCALL
MlasSiluKernelAvx512F(
    const float* Input,
    float* Output,
    size_t N
    )
{
    size_t Offset = 0;

    while (Offset + 16 <= N) {
        const __m512 X = _mm512_loadu_ps(Input + Offset);
        const __m512 Result = MlasComputeSiluVectorAvx512(X);
        _mm512_storeu_ps(Output + Offset, Result);
        Offset += 16;
    }

    if (Offset < N) {
        const __mmask16 TailMask = static_cast<__mmask16>((1u << (N - Offset)) - 1u);
        const __m512 X = _mm512_maskz_loadu_ps(TailMask, Input + Offset);
        const __m512 Result = MlasComputeSiluVectorAvx512(X);
        _mm512_mask_storeu_ps(Output + Offset, TailMask, Result);
    }
}
