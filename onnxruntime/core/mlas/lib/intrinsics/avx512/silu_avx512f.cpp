/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    silu_avx512f.cpp

Abstract:

    This module implements routines to compute the SiLU function with AVX512F
    intrinsics.

--*/

#include "mlasi.h"

namespace {

struct SiluAvx512Constants {
    static constexpr float LogisticLowerRange = -18.0f;
    static constexpr float LogisticUpperRange = 18.0f;
    static constexpr float Alpha9 = 4.37031012579801e-11f;
    static constexpr float Alpha7 = 1.15627324459942e-07f;
    static constexpr float Alpha5 = 6.08574864600143e-05f;
    static constexpr float Alpha3 = 8.51377133304701e-03f;
    static constexpr float Alpha1 = 2.48287947061529e-01f;
    static constexpr float Beta10 = 6.10247389755681e-13f;
    static constexpr float Beta8 = 5.76102136993427e-09f;
    static constexpr float Beta6 = 6.29106785017040e-06f;
    static constexpr float Beta4 = 1.70198817374094e-03f;
    static constexpr float Beta2 = 1.16817656904453e-01f;
    static constexpr float Beta0 = 9.93151921023180e-01f;
    static constexpr float OneHalf = 0.5f;
};

struct SiluAvx512BroadcastConstants {
    const __m512 LogisticLowerRange = _mm512_set1_ps(SiluAvx512Constants::LogisticLowerRange);
    const __m512 LogisticUpperRange = _mm512_set1_ps(SiluAvx512Constants::LogisticUpperRange);
    const __m512 Alpha9 = _mm512_set1_ps(SiluAvx512Constants::Alpha9);
    const __m512 Alpha7 = _mm512_set1_ps(SiluAvx512Constants::Alpha7);
    const __m512 Alpha5 = _mm512_set1_ps(SiluAvx512Constants::Alpha5);
    const __m512 Alpha3 = _mm512_set1_ps(SiluAvx512Constants::Alpha3);
    const __m512 Alpha1 = _mm512_set1_ps(SiluAvx512Constants::Alpha1);
    const __m512 Beta10 = _mm512_set1_ps(SiluAvx512Constants::Beta10);
    const __m512 Beta8 = _mm512_set1_ps(SiluAvx512Constants::Beta8);
    const __m512 Beta6 = _mm512_set1_ps(SiluAvx512Constants::Beta6);
    const __m512 Beta4 = _mm512_set1_ps(SiluAvx512Constants::Beta4);
    const __m512 Beta2 = _mm512_set1_ps(SiluAvx512Constants::Beta2);
    const __m512 Beta0 = _mm512_set1_ps(SiluAvx512Constants::Beta0);
    const __m512 OneHalf = _mm512_set1_ps(SiluAvx512Constants::OneHalf);
    const __m512 Zero = _mm512_setzero_ps();
    const __m512 One = _mm512_set1_ps(1.0f);
};

MLAS_FORCEINLINE __m512
MlasLogisticApproxAvx512(
    __m512 Value,
    const SiluAvx512BroadcastConstants& Constants
    )
{
    // Mirror MlasComputeLogistic by evaluating the same clamped rational
    // approximation in-register and then multiplying by x for SiLU.
    const __m512 ClampedValue = _mm512_max_ps(_mm512_min_ps(Value, Constants.LogisticUpperRange), Constants.LogisticLowerRange);
    const __m512 ValueSquared = _mm512_mul_ps(ClampedValue, ClampedValue);

    __m512 P = _mm512_fmadd_ps(ValueSquared, Constants.Alpha9, Constants.Alpha7);
    P = _mm512_fmadd_ps(P, ValueSquared, Constants.Alpha5);
    P = _mm512_fmadd_ps(P, ValueSquared, Constants.Alpha3);
    P = _mm512_fmadd_ps(P, ValueSquared, Constants.Alpha1);
    P = _mm512_mul_ps(P, ClampedValue);

    __m512 Q = _mm512_fmadd_ps(ValueSquared, Constants.Beta10, Constants.Beta8);
    Q = _mm512_fmadd_ps(Q, ValueSquared, Constants.Beta6);
    Q = _mm512_fmadd_ps(Q, ValueSquared, Constants.Beta4);
    Q = _mm512_fmadd_ps(Q, ValueSquared, Constants.Beta2);
    Q = _mm512_fmadd_ps(Q, ValueSquared, Constants.Beta0);

    __m512 Logistic = _mm512_add_ps(_mm512_div_ps(P, Q), Constants.OneHalf);
    Logistic = _mm512_min_ps(_mm512_max_ps(Logistic, Constants.Zero), Constants.One);

    return Logistic;
}

MLAS_FORCEINLINE __m512
MlasComputeSiluVectorAvx512(
    __m512 X,
    const SiluAvx512BroadcastConstants& Constants
    )
{
    __m512 Result = _mm512_mul_ps(X, MlasLogisticApproxAvx512(X, Constants));

    // Preserve NaN payload/sign behavior explicitly because the clamped
    // logistic approximation uses min/max operations that do not reliably
    // propagate NaNs the same way as the existing MLAS SiLU semantics.
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
    const SiluAvx512BroadcastConstants Constants;
    size_t Offset = 0;

    while (Offset + 32 <= N) {
        const __m512 X0 = _mm512_loadu_ps(Input + Offset);
        const __m512 X1 = _mm512_loadu_ps(Input + Offset + 16);
        const __m512 Result0 = MlasComputeSiluVectorAvx512(X0, Constants);
        const __m512 Result1 = MlasComputeSiluVectorAvx512(X1, Constants);
        _mm512_storeu_ps(Output + Offset, Result0);
        _mm512_storeu_ps(Output + Offset + 16, Result1);
        Offset += 32;
    }

    while (Offset + 16 <= N) {
        const __m512 X = _mm512_loadu_ps(Input + Offset);
        const __m512 Result = MlasComputeSiluVectorAvx512(X, Constants);
        _mm512_storeu_ps(Output + Offset, Result);
        Offset += 16;
    }

    if (Offset < N) {
        const __mmask16 TailMask = static_cast<__mmask16>((1u << (N - Offset)) - 1u);
        const __m512 X = _mm512_maskz_loadu_ps(TailMask, Input + Offset);
        const __m512 Result = MlasComputeSiluVectorAvx512(X, Constants);
        _mm512_mask_storeu_ps(Output + Offset, TailMask, Result);
    }
}
