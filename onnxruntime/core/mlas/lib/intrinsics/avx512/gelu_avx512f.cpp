/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gelu_avx512f.cpp

Abstract:

    This module implements routines to compute exact Gelu with AVX512F
    intrinsics.

--*/

#include <cstdint>

#include "mlasi.h"

namespace {

struct GeluAvx512Constants {
    static constexpr int32_t SignBitMask = INT32_MIN;
    static constexpr float InvSqrt2 = 0.70710678118654752440f;
    static constexpr float Half = 0.5f;
    static constexpr float One = 1.0f;

    static constexpr float ErfUpperAbsRange = 3.925f;
    static constexpr float ErfSplitBoundary = 0.921875f;
    static constexpr float ErfSMALL_P0 = -5.99104969e-4f;
    static constexpr float ErfSMALL_P1 = 4.99339588e-3f;
    static constexpr float ErfSMALL_P2 = -2.67667342e-2f;
    static constexpr float ErfSMALL_P3 = 1.12818025e-1f;
    static constexpr float ErfSMALL_P4 = -3.76124859e-1f;
    static constexpr float ErfSMALL_P5_Minus_One = 1.28379151e-1f;
    static constexpr float ErfBIG_P0 = 1.72948930e-5f;
    static constexpr float ErfBIG_P1 = -3.83208680e-4f;
    static constexpr float ErfBIG_P2 = 3.88393435e-3f;
    static constexpr float ErfBIG_P3 = -2.42545605e-2f;
    static constexpr float ErfBIG_P4 = 1.06777847e-1f;
    static constexpr float ErfBIG_P5 = 6.34846687e-1f;
    static constexpr float ErfBIG_P6_Minus_One = 1.28717512e-1f;
    static constexpr float ErfOne = 1.0f;
    static constexpr float ExpLowerRange = -88.3762626647949f;
    static constexpr float ExpLog2Reciprocal = 1.44269504088896341f;
    static constexpr float ExpLog2Hi = -6.93145752e-1f;
    static constexpr float ExpLog2Lo = -1.42860677e-6f;
    static constexpr float ExpP0 = 1.38319808e-3f;
    static constexpr float ExpP1 = 8.37550033e-3f;
    static constexpr float ExpP2 = 4.16689515e-2f;
    static constexpr float ExpP3 = 1.66664466e-1f;
    static constexpr float ExpP4 = 4.99999851e-1f;
    static constexpr float ExpP5 = 1.0f;
    static constexpr float ExpP6 = 1.0f;
    static constexpr float ExpC = 1.25829120e+7f;
};

struct GeluAvx512BroadcastConstants {
    const __m512 NegZero = _mm512_castsi512_ps(_mm512_set1_epi32(GeluAvx512Constants::SignBitMask));
    const __m512 Zero = _mm512_setzero_ps();
    const __m512 InvSqrt2 = _mm512_set1_ps(GeluAvx512Constants::InvSqrt2);
    const __m512 Half = _mm512_set1_ps(GeluAvx512Constants::Half);
    const __m512 One = _mm512_set1_ps(GeluAvx512Constants::One);
    const __m512 ErfUpperAbsRange = _mm512_set1_ps(GeluAvx512Constants::ErfUpperAbsRange);
    const __m512 ErfSplitBoundary = _mm512_set1_ps(GeluAvx512Constants::ErfSplitBoundary);
    const __m512 ErfSmallP0 = _mm512_set1_ps(GeluAvx512Constants::ErfSMALL_P0);
    const __m512 ErfSmallP1 = _mm512_set1_ps(GeluAvx512Constants::ErfSMALL_P1);
    const __m512 ErfSmallP2 = _mm512_set1_ps(GeluAvx512Constants::ErfSMALL_P2);
    const __m512 ErfSmallP3 = _mm512_set1_ps(GeluAvx512Constants::ErfSMALL_P3);
    const __m512 ErfSmallP4 = _mm512_set1_ps(GeluAvx512Constants::ErfSMALL_P4);
    const __m512 ErfSmallP5MinusOne = _mm512_set1_ps(GeluAvx512Constants::ErfSMALL_P5_Minus_One);
    const __m512 ErfBigP0 = _mm512_set1_ps(GeluAvx512Constants::ErfBIG_P0);
    const __m512 ErfBigP1 = _mm512_set1_ps(GeluAvx512Constants::ErfBIG_P1);
    const __m512 ErfBigP2 = _mm512_set1_ps(GeluAvx512Constants::ErfBIG_P2);
    const __m512 ErfBigP3 = _mm512_set1_ps(GeluAvx512Constants::ErfBIG_P3);
    const __m512 ErfBigP4 = _mm512_set1_ps(GeluAvx512Constants::ErfBIG_P4);
    const __m512 ErfBigP5 = _mm512_set1_ps(GeluAvx512Constants::ErfBIG_P5);
    const __m512 ErfBigP6MinusOne = _mm512_set1_ps(GeluAvx512Constants::ErfBIG_P6_Minus_One);
    const __m512 ErfOne = _mm512_set1_ps(GeluAvx512Constants::ErfOne);
    const __m512 ExpLowerRange = _mm512_set1_ps(GeluAvx512Constants::ExpLowerRange);
    const __m512 ExpLog2Reciprocal = _mm512_set1_ps(GeluAvx512Constants::ExpLog2Reciprocal);
    const __m512 ExpLog2Hi = _mm512_set1_ps(GeluAvx512Constants::ExpLog2Hi);
    const __m512 ExpLog2Lo = _mm512_set1_ps(GeluAvx512Constants::ExpLog2Lo);
    const __m512 ExpP0 = _mm512_set1_ps(GeluAvx512Constants::ExpP0);
    const __m512 ExpP1 = _mm512_set1_ps(GeluAvx512Constants::ExpP1);
    const __m512 ExpP2 = _mm512_set1_ps(GeluAvx512Constants::ExpP2);
    const __m512 ExpP3 = _mm512_set1_ps(GeluAvx512Constants::ExpP3);
    const __m512 ExpP4 = _mm512_set1_ps(GeluAvx512Constants::ExpP4);
    const __m512 ExpP5 = _mm512_set1_ps(GeluAvx512Constants::ExpP5);
    const __m512 ExpP6 = _mm512_set1_ps(GeluAvx512Constants::ExpP6);
    const __m512 ExpC = _mm512_set1_ps(GeluAvx512Constants::ExpC);
};

MLAS_FORCEINLINE __m512
MlasGeluErfExpVectorAvx512(
    __m512 Value,
    const GeluAvx512BroadcastConstants& Constants
    )
{
    __m512 R = _mm512_fmadd_ps(Constants.ExpLog2Reciprocal, Value, Constants.ExpC);
    R = _mm512_sub_ps(R, Constants.ExpC);

    __m512 Fx = _mm512_fmadd_ps(R, Constants.ExpLog2Hi, Value);
    Fx = _mm512_fmadd_ps(R, Constants.ExpLog2Lo, Fx);

    __m512 Y = Constants.ExpP0;
    Y = _mm512_fmadd_ps(Y, Fx, Constants.ExpP1);
    Y = _mm512_fmadd_ps(Y, Fx, Constants.ExpP2);
    Y = _mm512_fmadd_ps(Y, Fx, Constants.ExpP3);
    Y = _mm512_fmadd_ps(Y, Fx, Constants.ExpP4);
    Y = _mm512_fmadd_ps(Y, Fx, Constants.ExpP5);
    Y = _mm512_fmadd_ps(Y, Fx, Constants.ExpP6);
    Y = _mm512_scalef_ps(Y, R);

    return Y;
}

MLAS_FORCEINLINE __m512
MlasGeluErfAvx512(
    __m512 Value,
    const GeluAvx512BroadcastConstants& Constants
    )
{
    const __m512 SignMask = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(Value), _mm512_castps_si512(Constants.NegZero)));
    __m512 AbsValue = _mm512_castsi512_ps(_mm512_andnot_si512(_mm512_castps_si512(Constants.NegZero), _mm512_castps_si512(Value)));
    AbsValue = _mm512_min_ps(Constants.ErfUpperAbsRange, AbsValue);

    const __m512 SquareValue = _mm512_mul_ps(AbsValue, AbsValue);

    __m512 SmallResult = Constants.ErfSmallP0;
    SmallResult = _mm512_fmadd_ps(SmallResult, SquareValue, Constants.ErfSmallP1);
    SmallResult = _mm512_fmadd_ps(SmallResult, SquareValue, Constants.ErfSmallP2);
    SmallResult = _mm512_fmadd_ps(SmallResult, SquareValue, Constants.ErfSmallP3);
    SmallResult = _mm512_fmadd_ps(SmallResult, SquareValue, Constants.ErfSmallP4);
    SmallResult = _mm512_fmadd_ps(SmallResult, SquareValue, Constants.ErfSmallP5MinusOne);
    SmallResult = _mm512_fmadd_ps(SmallResult, AbsValue, AbsValue);

    const __mmask16 SplitMask = _mm512_cmp_ps_mask(AbsValue, Constants.ErfSplitBoundary, _CMP_GT_OQ);
    const __m512 BigInput = _mm512_mask_blend_ps(SplitMask, Constants.Zero, AbsValue);

    __m512 BigResult = Constants.ErfBigP0;
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, Constants.ErfBigP1);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, Constants.ErfBigP2);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, Constants.ErfBigP3);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, Constants.ErfBigP4);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, Constants.ErfBigP5);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, Constants.ErfBigP6MinusOne);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, BigInput);

    BigResult = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(BigResult), _mm512_castps_si512(Constants.NegZero)));
    BigResult = _mm512_max_ps(Constants.ExpLowerRange, BigResult);
    BigResult = _mm512_sub_ps(Constants.ErfOne, MlasGeluErfExpVectorAvx512(BigResult, Constants));

    __m512 Result = _mm512_mask_blend_ps(SplitMask, SmallResult, BigResult);
    Result = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(Result), _mm512_castps_si512(SignMask)));
    return Result;
}

MLAS_FORCEINLINE __m512
MlasComputeGeluVectorExactAvx512(
    __m512 X,
    const GeluAvx512BroadcastConstants& Constants
    )
{
    const __m512 ErfInput = _mm512_mul_ps(X, Constants.InvSqrt2);
    const __m512 ErfValue = MlasGeluErfAvx512(ErfInput, Constants);
    __m512 Result = _mm512_mul_ps(_mm512_mul_ps(Constants.Half, X), _mm512_add_ps(ErfValue, Constants.One));

    // Preserve NaN payload/sign behavior explicitly because the erf
    // approximation uses min/max style range limiting that is not guaranteed to
    // preserve NaNs the same way as the existing MLAS GELU semantics.
    const __mmask16 NaNMask = _mm512_cmp_ps_mask(X, X, _CMP_UNORD_Q);
    Result = _mm512_mask_mov_ps(Result, NaNMask, X);

    return Result;
}

void
MlasGeluErfKernelAvx512FExactImpl(
    const float* Input,
    float* Output,
    size_t N
    )
{
    const GeluAvx512BroadcastConstants Constants;
    while (N >= 16) {
        const __m512 X = _mm512_loadu_ps(Input);
        const __m512 Result = MlasComputeGeluVectorExactAvx512(X, Constants);

        _mm512_storeu_ps(Output, Result);

        Input += 16;
        Output += 16;
        N -= 16;
    }

    if (N > 0) {
        const __mmask16 TailMask = __mmask16((1u << static_cast<unsigned>(N)) - 1u);
        const __m512 X = _mm512_maskz_loadu_ps(TailMask, Input);
        const __m512 Result = MlasComputeGeluVectorExactAvx512(X, Constants);

        _mm512_mask_storeu_ps(Output, TailMask, Result);
    }
}

}  // namespace

void
MLASCALL
MlasGeluErfKernelAvx512F(
    const float* Input,
    float* Output,
    size_t N
    )
{
    MlasGeluErfKernelAvx512FExactImpl(Input, Output, N);
}
