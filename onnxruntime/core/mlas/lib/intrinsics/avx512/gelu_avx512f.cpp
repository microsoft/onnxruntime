/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gelu_avx512f.cpp

Abstract:

    This module implements routines to compute exact Gelu with AVX512F
    intrinsics.

    Idea and code credit for the minimax approximation: OneDNN library

--*/

#include <cmath>
#include <limits>

#include "mlasi.h"

namespace {

struct GeluAvx512Constants {
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

alignas(64) static const uint32_t MlasGeluErfMinimaxTable[6][32] = {
    {
        0xa6f2cb94, 0x32827792, 0x3381cc0c, 0x34523d4a,
        0x351ac44d, 0x35f36d88, 0x36ee8229, 0x37b8a3bb,
        0x3867a213, 0x3940033b, 0x3a2a5a1d, 0x3ae35863,
        0x3b7828f2, 0x3c08b14b, 0x3c515ed3, 0xbb503236,
        0xbd8d8e5e, 0xbe8abcd9, 0xbf0c19a2, 0xbeccb328,
        0x3e176ced, 0x3f470d99, 0x3f7abb28, 0x3f800000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
    },
    {
        0x3f4c422a, 0x3f4c421f, 0x3f4c4207, 0x3f4c41cb,
        0x3f4c413b, 0x3f4c3fad, 0x3f4c3a2f, 0x3f4c2d40,
        0x3f4c146a, 0x3f4bc341, 0x3f4ad08c, 0x3f48f8cf,
        0x3f45fac7, 0x3f404e07, 0x3f3b980f, 0x3f48dff3,
        0x3f78b21b, 0x3fbb0704, 0x40019c32, 0x3fe536d6,
        0x3f81331e, 0x3e6c8684, 0x3c98f936, 0x00000000,
        0x3f800000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
    },
    {
        0xb62173f4, 0x3735e4cf, 0x37f2ff89, 0x388c23be,
        0x3917535c, 0x39ab2ab0, 0x3a60fadb, 0x3af9b960,
        0x3b6e5491, 0x3c0a4ec5, 0x3ca5aa8c, 0x3d2138d9,
        0x3d8737d4, 0x3ddfb660, 0x3e0f27ab, 0x3d94004b,
        0xbe0efdeb, 0xbf1d96c3, 0xbf89db58, 0xbf6d9897,
        0xbef69fb8, 0xbdc4f8a8, 0xbbde6422, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
    },
    {
        0xbe081a19, 0xbe084570, 0xbe08639b, 0xbe089837,
        0xbe08f409, 0xbe09ab95, 0xbe0b66d0, 0xbe0e400a,
        0xbe124df8, 0xbe1bde02, 0xbe2f19c9, 0xbe4931bf,
        0xbe685fbc, 0xbe89c95f, 0xbe96cbca, 0xbe8044aa,
        0xbe0550f2, 0x3dcfd6a1, 0x3e94c826, 0x3e79345f,
        0x3decec91, 0x3ca46568, 0x3aa1e00a, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
    },
    {
        0xba3d61db, 0x39f097a3, 0x3a5845dc, 0x3ab1fa35,
        0x3b0cefb8, 0x3b653ab6, 0x3bcae527, 0x3c221712,
        0x3c6c5840, 0x3cc0a703, 0x3d1dcc19, 0x3d63656d,
        0x3d955907, 0x3dbf9910, 0x3dd53f69, 0x3db7dcef,
        0x3d639ebe, 0xba6ede48, 0xbd22be69, 0xbd041cf1,
        0xbc64f5ab, 0xbb097a32, 0xb8ebf380, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
    },
    {
        0x3cb7d80c, 0x3c9b6050, 0x3c978d11, 0x3c92e850,
        0x3c8d058b, 0x3c848454, 0x3c6cd623, 0x3c4c824b,
        0x3c2a7935, 0x3be0b390, 0x3b0651ac, 0xbb232f53,
        0xbbd42fa0, 0xbc2c5366, 0xbc492c9e, 0xbc2a7aa6,
        0xbbd55d04, 0xba823a76, 0x3b102aa8, 0x3ae25a7e,
        0x3a31f792, 0x38b84375, 0x3689bb5a, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
    }
};

MLAS_FORCEINLINE __m512
MlasLoadMinimaxTable(
    const uint32_t* Table
    )
{
    return _mm512_castsi512_ps(_mm512_load_si512(reinterpret_cast<const __m512i*>(Table)));
}

MLAS_FORCEINLINE __m512
MlasGatherMinimaxCoeff(
    int Degree,
    __m512i Index
    )
{
    const uint32_t* Base = MlasGeluErfMinimaxTable[Degree];
    const __m512 Lo = MlasLoadMinimaxTable(Base);
    const __m512 Hi = MlasLoadMinimaxTable(Base + 16);
    return _mm512_permutex2var_ps(Lo, Index, Hi);
}

MLAS_FORCEINLINE __m512
MlasGeluErfExpVectorAvx512(
    __m512 Value
    )
{
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

    __m512 R = _mm512_fmadd_ps(ExpLog2Reciprocal, Value, ExpC);
    R = _mm512_sub_ps(R, ExpC);

    __m512 Fx = _mm512_fmadd_ps(R, ExpLog2Hi, Value);
    Fx = _mm512_fmadd_ps(R, ExpLog2Lo, Fx);

    __m512 Y = ExpP0;
    Y = _mm512_fmadd_ps(Y, Fx, ExpP1);
    Y = _mm512_fmadd_ps(Y, Fx, ExpP2);
    Y = _mm512_fmadd_ps(Y, Fx, ExpP3);
    Y = _mm512_fmadd_ps(Y, Fx, ExpP4);
    Y = _mm512_fmadd_ps(Y, Fx, ExpP5);
    Y = _mm512_fmadd_ps(Y, Fx, ExpP6);
    Y = _mm512_scalef_ps(Y, R);

    return Y;
}

MLAS_FORCEINLINE __m512
MlasGeluErfAvx512(
    __m512 Value
    )
{
    const __m512 NegZero = _mm512_castsi512_ps(_mm512_set1_epi32(int(0x80000000u)));
    const __m512 Zero = _mm512_setzero_ps();
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

    const __m512 SignMask = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(Value), _mm512_castps_si512(NegZero)));
    __m512 AbsValue = _mm512_castsi512_ps(_mm512_andnot_si512(_mm512_castps_si512(NegZero), _mm512_castps_si512(Value)));
    AbsValue = _mm512_min_ps(ErfUpperAbsRange, AbsValue);

    const __m512 SquareValue = _mm512_mul_ps(AbsValue, AbsValue);

    __m512 SmallResult = ErfSmallP0;
    SmallResult = _mm512_fmadd_ps(SmallResult, SquareValue, ErfSmallP1);
    SmallResult = _mm512_fmadd_ps(SmallResult, SquareValue, ErfSmallP2);
    SmallResult = _mm512_fmadd_ps(SmallResult, SquareValue, ErfSmallP3);
    SmallResult = _mm512_fmadd_ps(SmallResult, SquareValue, ErfSmallP4);
    SmallResult = _mm512_fmadd_ps(SmallResult, SquareValue, ErfSmallP5MinusOne);
    SmallResult = _mm512_fmadd_ps(SmallResult, AbsValue, AbsValue);

    const __mmask16 SplitMask = _mm512_cmp_ps_mask(AbsValue, ErfSplitBoundary, _CMP_GT_OQ);
    const __m512 BigInput = _mm512_mask_blend_ps(SplitMask, Zero, AbsValue);

    __m512 BigResult = ErfBigP0;
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, ErfBigP1);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, ErfBigP2);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, ErfBigP3);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, ErfBigP4);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, ErfBigP5);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, ErfBigP6MinusOne);
    BigResult = _mm512_fmadd_ps(BigResult, BigInput, BigInput);

    BigResult = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(BigResult), _mm512_castps_si512(NegZero)));
    BigResult = _mm512_max_ps(ExpLowerRange, BigResult);
    BigResult = _mm512_sub_ps(ErfOne, MlasGeluErfExpVectorAvx512(BigResult));

    __m512 Result = _mm512_mask_blend_ps(SplitMask, SmallResult, BigResult);
    Result = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(Result), _mm512_castps_si512(SignMask)));
    return Result;
}

MLAS_FORCEINLINE __m512
MlasComputeGeluVectorMinimaxAvx512(
    __m512 X
    )
{
    const __m512 PositiveInfinity = _mm512_set1_ps(std::numeric_limits<float>::infinity());
    const __m512 SignMask = _mm512_castsi512_ps(_mm512_set1_epi32(int(0x80000000u)));
    const __m512 PositiveMask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffffu));
    const __m512 One = _mm512_set1_ps(1.0f);
    const __m512 Half = _mm512_set1_ps(0.5f);
    const __m512 NegativeInfinity = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    const __m512 NegativeZero = _mm512_castsi512_ps(_mm512_set1_epi32(int(0x80000000u)));
    const __m512 RightBound = _mm512_castsi512_ps(_mm512_set1_epi32(int(0x40b15ceeu)));

    const __m512i IndexBias = _mm512_set1_epi32(static_cast<int32_t>(0xc21fffff));
    const __m512i OneI = _mm512_set1_epi32(1);
    const __m512i TwentyThreeI = _mm512_set1_epi32(23);
    const __m512i TwentyFourI = _mm512_set1_epi32(24);

    const __m512 XPositive = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(X), _mm512_castps_si512(PositiveMask)));

    __m512i Index = _mm512_castps_si512(XPositive);
    Index = _mm512_add_epi32(Index, IndexBias);
    Index = _mm512_srai_epi32(Index, 21);
    Index = _mm512_max_epi32(Index, OneI);
    Index = _mm512_min_epi32(Index, TwentyFourI);

    const __mmask16 GreaterThanRightBoundMask = _mm512_cmp_ps_mask(XPositive, RightBound, _CMP_GT_OQ);
    Index = _mm512_mask_blend_epi32(GreaterThanRightBoundMask, Index, TwentyThreeI);

    __m512 Polynomial = MlasGatherMinimaxCoeff(5, Index);
    Polynomial = _mm512_fmadd_ps(Polynomial, XPositive, MlasGatherMinimaxCoeff(4, Index));
    Polynomial = _mm512_fmadd_ps(Polynomial, XPositive, MlasGatherMinimaxCoeff(3, Index));
    Polynomial = _mm512_fmadd_ps(Polynomial, XPositive, MlasGatherMinimaxCoeff(2, Index));
    Polynomial = _mm512_fmadd_ps(Polynomial, XPositive, MlasGatherMinimaxCoeff(1, Index));
    Polynomial = _mm512_fmadd_ps(Polynomial, XPositive, MlasGatherMinimaxCoeff(0, Index));

    const __m512 Sign = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(X), _mm512_castps_si512(SignMask)));
    const __m512 ErfPart = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(Polynomial), _mm512_castps_si512(Sign)));
    __m512 Result = _mm512_mul_ps(_mm512_mul_ps(X, _mm512_add_ps(ErfPart, One)), Half);

    const __mmask16 PositiveInfinityMask = _mm512_cmp_ps_mask(X, PositiveInfinity, _CMP_EQ_OQ);
    Result = _mm512_mask_mov_ps(Result, PositiveInfinityMask, PositiveInfinity);

    const __mmask16 NegativeInfinityMask = _mm512_cmp_ps_mask(X, NegativeInfinity, _CMP_EQ_OQ);
    Result = _mm512_mask_mov_ps(Result, NegativeInfinityMask, NegativeZero);

    return Result;
}

MLAS_FORCEINLINE __m512
MlasComputeGeluVectorExactAvx512(
    __m512 X
    )
{
    const __m512 InvSqrt2 = _mm512_set1_ps(GeluAvx512Constants::InvSqrt2);
    const __m512 Half = _mm512_set1_ps(GeluAvx512Constants::Half);
    const __m512 One = _mm512_set1_ps(GeluAvx512Constants::One);
    const __m512 ErfInput = _mm512_mul_ps(X, InvSqrt2);
    const __m512 ErfValue = MlasGeluErfAvx512(ErfInput);
    return _mm512_mul_ps(_mm512_mul_ps(Half, X), _mm512_add_ps(ErfValue, One));
}

void
MlasGeluKernelAvx512FExactImpl(
    const float* Input,
    float* Output,
    size_t N
    )
{
    while (N >= 16) {
        const __m512 X = _mm512_loadu_ps(Input);
        const __m512 Result = MlasComputeGeluVectorExactAvx512(X);

        _mm512_storeu_ps(Output, Result);

        Input += 16;
        Output += 16;
        N -= 16;
    }

    while (N > 0) {
        const float X = *Input++;
        *Output++ = GeluAvx512Constants::Half * X * (std::erff(X * GeluAvx512Constants::InvSqrt2) + GeluAvx512Constants::One);
        N -= 1;
    }
}

void
MlasGeluKernelAvx512FMinimaxApproxImpl(
    const float* Input,
    float* Output,
    size_t N
    )
{
    size_t Offset = 0;

    while (Offset + 16 <= N) {
        const __m512 X = _mm512_loadu_ps(Input + Offset);
        const __m512 Result = MlasComputeGeluVectorMinimaxAvx512(X);
        _mm512_storeu_ps(Output + Offset, Result);
        Offset += 16;
    }

    if (Offset < N) {
        const __mmask16 TailMask = static_cast<__mmask16>((1u << (N - Offset)) - 1u);
        const __m512 X = _mm512_maskz_loadu_ps(TailMask, Input + Offset);
        const __m512 Result = MlasComputeGeluVectorMinimaxAvx512(X);
        _mm512_mask_storeu_ps(Output + Offset, TailMask, Result);
    }
}

}  // namespace

void
MLASCALL
MlasGeluKernelAvx512F(
    const float* Input,
    float* Output,
    size_t N
    )
{
    MlasGeluKernelAvx512FExactImpl(Input, Output, N);
}

void
MLASCALL
MlasGeluKernelAvx512FMinimaxApprox(
    const float* Input,
    float* Output,
    size_t N
    )
{
    MlasGeluKernelAvx512FMinimaxApproxImpl(Input, Output, N);
}
