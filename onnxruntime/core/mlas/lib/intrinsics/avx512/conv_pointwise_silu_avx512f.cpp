/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    conv_pointwise_silu_avx512f.cpp

Abstract:

    This module implements a fused pointwise (1x1) NCHWc convolution kernel
    with SiLU activation for AVX-512F.

    The kernel has the same interface as MLAS_CONV_POINTWISE_FLOAT_KERNEL and
    replaces the separate conv-kernel + DoActivation(SiLU) pair with a single
    pass that applies SiLU in the post-processing stage, keeping partial sums
    in zmm registers throughout.

    Data layout (BlockSize = 16):
      Input  : [InputChannels][H*W][16]   (NCHWc)
      Filter : [FilterCount][Cin][16]     (OIHWBo pointwise, one block per call)
      Output : [FilterCount][H*W][16]     (NCHWc)

    The hot inner loop is:
      for each IC block  (InputChannels iterations):
        for j in 0..15:                             (16 scalar input channels)
          Acc[fc] += broadcast(Input[ic][j]) * Filter[fc][ic*16 + j][0..15]

    Post-processing (only on the last input-channel batch, i.e. when
    MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION is set):
      1. Accumulate prior partial output if MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
      2. Add bias vector (16 floats per output block)
      3. Apply SiLU: y = x * sigmoid(x)

    Caller (snchwc.cpp) must suppress the subsequent DoActivation() call when
    this kernel is selected.

--*/

#include <limits>
#include "mlasi.h"

#ifndef MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT 0x00000001
#endif

#ifndef MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION     0x00000002
#endif

#ifndef MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION   0x00000004
#endif

#ifndef MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION  0x00000008
#endif

//
// SiLU helpers (private to this translation unit).
//

namespace {

// Five-term polynomial approximation for exp(x), numerically stable for
// all representable float32 values through range reduction to [0, ln2].
MLAS_FORCEINLINE __m512
PwSiluExpApprox(
    __m512 Value
    )
{
    const __m512 Half          = _mm512_set1_ps(0.5f);
    const __m512 One           = _mm512_set1_ps(1.0f);
    const __m512 Two           = _mm512_set1_ps(2.0f);
    const __m512 Ln2           = _mm512_set1_ps(0.693147182f);
    const __m512 Log2EF        = _mm512_set1_ps(1.44269502f);
    const __m512 ExpLnFltMax   = _mm512_set1_ps(88.3762589f);
    const __m512 ExpLnFltMin   = _mm512_set1_ps(-87.3365479f);
    const __m512 P1            = _mm512_set1_ps(0.999999701f);
    const __m512 P2            = _mm512_set1_ps(0.499991506f);
    const __m512 P3            = _mm512_set1_ps(0.166676521f);
    const __m512 P4            = _mm512_set1_ps(0.0418978221f);
    const __m512 P5            = _mm512_set1_ps(0.00828929059f);
    const __m512i ExponentBias = _mm512_set1_epi32(127);

    const __mmask16 UnderflowMask = _mm512_cmp_ps_mask(Value, ExpLnFltMin, _CMP_LT_OS);

    Value = _mm512_min_ps(Value, ExpLnFltMax);
    Value = _mm512_max_ps(Value, ExpLnFltMin);

    __m512 Fx = _mm512_fmadd_ps(Value, Log2EF, Half);
    Fx = _mm512_floor_ps(Fx);

    const __m512 R = _mm512_fnmadd_ps(Fx, Ln2, Value);

    __m512i Exponent = _mm512_cvttps_epi32(_mm512_sub_ps(Fx, One));
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

    return _mm512_mul_ps(_mm512_mul_ps(Y, Pow2NMinusOne), Two);
}

// Numerically stable sigmoid: always computes exp on the negative side,
// so the denominator never underflows.
MLAS_FORCEINLINE __m512
PwSiluSigmoid(
    __m512 X
    )
{
    const __m512 One         = _mm512_set1_ps(1.0f);
    const __m512 SignMask    = _mm512_castsi512_ps(_mm512_set1_epi32(int(0x80000000u)));
    const __m512 PositiveMask= _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffffu));

    const __m512 XNeg = _mm512_or_ps(_mm512_and_ps(X, PositiveMask), SignMask);
    const __m512 E    = PwSiluExpApprox(XNeg);
    const __m512 Y    = _mm512_div_ps(E, _mm512_add_ps(E, One));

    // For negative X:  sigmoid = E/(E+1)  (computed above, correct)
    // For positive X:  sigmoid = 1 - E/(E+1) = 1/(1+E') where E'=1/E
    const __m512 OneMinusY = _mm512_sub_ps(One, Y);
    const __mmask16 NegMask = _mm512_cmp_ps_mask(X, _mm512_setzero_ps(), _CMP_LT_OQ);
    return _mm512_mask_blend_ps(NegMask, OneMinusY, Y);
}

// y = x * sigmoid(x),  with special-case handling for NaN and +-inf.
MLAS_FORCEINLINE __m512
PwSiluActivate(
    __m512 X
    )
{
    const __m512 PosInf  = _mm512_set1_ps( std::numeric_limits<float>::infinity());
    const __m512 NegInf  = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    const __m512 NegZero = _mm512_castsi512_ps(_mm512_set1_epi32(int(0x80000000u)));

    __m512 Result = _mm512_mul_ps(X, PwSiluSigmoid(X));

    // NaN passthrough
    const __mmask16 NaNMask  = _mm512_cmp_ps_mask(X, X, _CMP_UNORD_Q);
    Result = _mm512_mask_mov_ps(Result, NaNMask, X);

    // +inf -> +inf
    const __mmask16 PosInfMask = _mm512_cmp_ps_mask(X, PosInf, _CMP_EQ_OQ);
    Result = _mm512_mask_mov_ps(Result, PosInfMask, PosInf);

    // -inf -> -0
    const __mmask16 NegInfMask = _mm512_cmp_ps_mask(X, NegInf, _CMP_EQ_OQ);
    Result = _mm512_mask_mov_ps(Result, NegInfMask, NegZero);

    return Result;
}

template<size_t OutputCount, size_t FilterCount>
MLAS_FORCEINLINE void
PwFmaAccumulateTile(
    const float* __restrict InputBase,
    const float* __restrict Filter,
    size_t InputChannels,
    size_t StrideWidth,
    size_t InputStride,
    size_t FilterStride,
    __m512 (&Acc)[FilterCount][OutputCount]
    )
{
    const size_t InStride = InputStride / sizeof(float);
    const size_t InPosStride = StrideWidth / sizeof(float);
    const size_t FltStride = FilterStride / sizeof(float);

    const float* F0 = Filter;
    const float* F1 = F0 + FltStride;
    const float* F2 = F1 + FltStride;
    const float* F3 = F2 + FltStride;

    for (size_t ic = 0; ic < InputChannels; ++ic) {
        const float* I0 = InputBase + ic * InStride;
        const float* I1 = I0 + InPosStride;
        const float* I2 = I1 + InPosStride;
        const float* I3 = I2 + InPosStride;
        const float* I4 = I3 + InPosStride;
        const float* I5 = I4 + InPosStride;

        const float* Fic0 = F0 + ic * 16 * 16;
        const float* Fic1 = F1 + ic * 16 * 16;
        const float* Fic2 = F2 + ic * 16 * 16;
        const float* Fic3 = F3 + ic * 16 * 16;

#define MLAS_PW_TILE_FMA_J(j) \
        { \
            const __m512 W0 = _mm512_loadu_ps(Fic0 + (j) * 16); \
            __m512 W1, W2, W3; \
            if constexpr (FilterCount > 1) W1 = _mm512_loadu_ps(Fic1 + (j) * 16); \
            if constexpr (FilterCount > 2) W2 = _mm512_loadu_ps(Fic2 + (j) * 16); \
            if constexpr (FilterCount > 3) W3 = _mm512_loadu_ps(Fic3 + (j) * 16); \
            if constexpr (OutputCount > 0) { const __m512 V = _mm512_set1_ps(I0[j]); Acc[0][0] = _mm512_fmadd_ps(V, W0, Acc[0][0]); if constexpr (FilterCount > 1) Acc[1][0] = _mm512_fmadd_ps(V, W1, Acc[1][0]); if constexpr (FilterCount > 2) Acc[2][0] = _mm512_fmadd_ps(V, W2, Acc[2][0]); if constexpr (FilterCount > 3) Acc[3][0] = _mm512_fmadd_ps(V, W3, Acc[3][0]); } \
            if constexpr (OutputCount > 1) { const __m512 V = _mm512_set1_ps(I1[j]); Acc[0][1] = _mm512_fmadd_ps(V, W0, Acc[0][1]); if constexpr (FilterCount > 1) Acc[1][1] = _mm512_fmadd_ps(V, W1, Acc[1][1]); if constexpr (FilterCount > 2) Acc[2][1] = _mm512_fmadd_ps(V, W2, Acc[2][1]); if constexpr (FilterCount > 3) Acc[3][1] = _mm512_fmadd_ps(V, W3, Acc[3][1]); } \
            if constexpr (OutputCount > 2) { const __m512 V = _mm512_set1_ps(I2[j]); Acc[0][2] = _mm512_fmadd_ps(V, W0, Acc[0][2]); if constexpr (FilterCount > 1) Acc[1][2] = _mm512_fmadd_ps(V, W1, Acc[1][2]); if constexpr (FilterCount > 2) Acc[2][2] = _mm512_fmadd_ps(V, W2, Acc[2][2]); if constexpr (FilterCount > 3) Acc[3][2] = _mm512_fmadd_ps(V, W3, Acc[3][2]); } \
            if constexpr (OutputCount > 3) { const __m512 V = _mm512_set1_ps(I3[j]); Acc[0][3] = _mm512_fmadd_ps(V, W0, Acc[0][3]); if constexpr (FilterCount > 1) Acc[1][3] = _mm512_fmadd_ps(V, W1, Acc[1][3]); if constexpr (FilterCount > 2) Acc[2][3] = _mm512_fmadd_ps(V, W2, Acc[2][3]); if constexpr (FilterCount > 3) Acc[3][3] = _mm512_fmadd_ps(V, W3, Acc[3][3]); } \
            if constexpr (OutputCount > 4) { const __m512 V = _mm512_set1_ps(I4[j]); Acc[0][4] = _mm512_fmadd_ps(V, W0, Acc[0][4]); if constexpr (FilterCount > 1) Acc[1][4] = _mm512_fmadd_ps(V, W1, Acc[1][4]); if constexpr (FilterCount > 2) Acc[2][4] = _mm512_fmadd_ps(V, W2, Acc[2][4]); if constexpr (FilterCount > 3) Acc[3][4] = _mm512_fmadd_ps(V, W3, Acc[3][4]); } \
            if constexpr (OutputCount > 5) { const __m512 V = _mm512_set1_ps(I5[j]); Acc[0][5] = _mm512_fmadd_ps(V, W0, Acc[0][5]); if constexpr (FilterCount > 1) Acc[1][5] = _mm512_fmadd_ps(V, W1, Acc[1][5]); if constexpr (FilterCount > 2) Acc[2][5] = _mm512_fmadd_ps(V, W2, Acc[2][5]); if constexpr (FilterCount > 3) Acc[3][5] = _mm512_fmadd_ps(V, W3, Acc[3][5]); } \
        }

        MLAS_PW_TILE_FMA_J(0)  MLAS_PW_TILE_FMA_J(1)  MLAS_PW_TILE_FMA_J(2)  MLAS_PW_TILE_FMA_J(3)
        MLAS_PW_TILE_FMA_J(4)  MLAS_PW_TILE_FMA_J(5)  MLAS_PW_TILE_FMA_J(6)  MLAS_PW_TILE_FMA_J(7)
        MLAS_PW_TILE_FMA_J(8)  MLAS_PW_TILE_FMA_J(9)  MLAS_PW_TILE_FMA_J(10) MLAS_PW_TILE_FMA_J(11)
        MLAS_PW_TILE_FMA_J(12) MLAS_PW_TILE_FMA_J(13) MLAS_PW_TILE_FMA_J(14) MLAS_PW_TILE_FMA_J(15)

#undef MLAS_PW_TILE_FMA_J
    }
}

template<size_t OutputCount, size_t FilterCount>
MLAS_FORCEINLINE void
PwSiluPostProcessTile(
    float* __restrict Output,
    const float* __restrict Bias,
    size_t OutputStride,
    unsigned Flags,
    __m512 (&Acc)[FilterCount][OutputCount]
    )
{
    const size_t OutStride = OutputStride / sizeof(float);

    __m512 BiasVec[FilterCount];
    if ((Flags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) && Bias != nullptr) {
        if constexpr (FilterCount > 0) BiasVec[0] = _mm512_loadu_ps(Bias);
        if constexpr (FilterCount > 1) BiasVec[1] = _mm512_loadu_ps(Bias + 16);
        if constexpr (FilterCount > 2) BiasVec[2] = _mm512_loadu_ps(Bias + 32);
        if constexpr (FilterCount > 3) BiasVec[3] = _mm512_loadu_ps(Bias + 48);
    }

    for (size_t o = 0; o < OutputCount; ++o) {
        float* Out0 = Output + o * 16;
        float* Out1 = Out0 + OutStride;
        float* Out2 = Out1 + OutStride;
        float* Out3 = Out2 + OutStride;

        if (Flags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) {
            if constexpr (FilterCount > 0) Acc[0][o] = _mm512_add_ps(Acc[0][o], _mm512_loadu_ps(Out0));
            if constexpr (FilterCount > 1) Acc[1][o] = _mm512_add_ps(Acc[1][o], _mm512_loadu_ps(Out1));
            if constexpr (FilterCount > 2) Acc[2][o] = _mm512_add_ps(Acc[2][o], _mm512_loadu_ps(Out2));
            if constexpr (FilterCount > 3) Acc[3][o] = _mm512_add_ps(Acc[3][o], _mm512_loadu_ps(Out3));
        }

        if (Flags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) {
            if constexpr (FilterCount > 0) Acc[0][o] = _mm512_add_ps(Acc[0][o], BiasVec[0]);
            if constexpr (FilterCount > 1) Acc[1][o] = _mm512_add_ps(Acc[1][o], BiasVec[1]);
            if constexpr (FilterCount > 2) Acc[2][o] = _mm512_add_ps(Acc[2][o], BiasVec[2]);
            if constexpr (FilterCount > 3) Acc[3][o] = _mm512_add_ps(Acc[3][o], BiasVec[3]);
        }

        if (Flags & MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION) {
            if constexpr (FilterCount > 0) Acc[0][o] = PwSiluActivate(Acc[0][o]);
            if constexpr (FilterCount > 1) Acc[1][o] = PwSiluActivate(Acc[1][o]);
            if constexpr (FilterCount > 2) Acc[2][o] = PwSiluActivate(Acc[2][o]);
            if constexpr (FilterCount > 3) Acc[3][o] = PwSiluActivate(Acc[3][o]);
        }

        if constexpr (FilterCount > 0) _mm512_storeu_ps(Out0, Acc[0][o]);
        if constexpr (FilterCount > 1) _mm512_storeu_ps(Out1, Acc[1][o]);
        if constexpr (FilterCount > 2) _mm512_storeu_ps(Out2, Acc[2][o]);
        if constexpr (FilterCount > 3) _mm512_storeu_ps(Out3, Acc[3][o]);
    }
}

template<size_t OutputCount, size_t FilterCount>
MLAS_FORCEINLINE void
PwKernelTile(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t InputChannels,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    const float* Bias,
    unsigned KernelFlags
    )
{
    __m512 Acc[FilterCount][OutputCount];

    for (size_t f = 0; f < FilterCount; ++f) {
        for (size_t o = 0; o < OutputCount; ++o) {
            Acc[f][o] = _mm512_setzero_ps();
        }
    }

    PwFmaAccumulateTile<OutputCount, FilterCount>(Input, Filter, InputChannels,
        StrideWidth, InputStride, FilterStride, Acc);

    PwSiluPostProcessTile<OutputCount, FilterCount>(Output, Bias, OutputStride,
        KernelFlags, Acc);
}

template<size_t OutputCount>
MLAS_FORCEINLINE void
PwDispatchFilterCount(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t InputChannels,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    const float* Bias,
    unsigned KernelFlags
    )
{
    switch (FilterCount) {
        case 1:
            PwKernelTile<OutputCount, 1>(Input, Filter, Output, StrideWidth, InputChannels,
                InputStride, FilterStride, OutputStride, Bias, KernelFlags);
            break;
        case 2:
            PwKernelTile<OutputCount, 2>(Input, Filter, Output, StrideWidth, InputChannels,
                InputStride, FilterStride, OutputStride, Bias, KernelFlags);
            break;
        case 3:
            PwKernelTile<OutputCount, 3>(Input, Filter, Output, StrideWidth, InputChannels,
                InputStride, FilterStride, OutputStride, Bias, KernelFlags);
            break;
        default:
            PwKernelTile<OutputCount, 4>(Input, Filter, Output, StrideWidth, InputChannels,
                InputStride, FilterStride, OutputStride, Bias, KernelFlags);
            break;
    }
}

} // namespace

//
// Public entry point — same ABI as MlasConvPointwiseFloatKernelAvx512F.
//
// Processes up to 6 output positions per iteration (matching the assembly
// kernel's tile width) to keep the filter panels hot in L1.
//

void
MLASCALL
MlasConvPointwiseSiluFloatKernelAvx512F(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,     // bytes between adjacent spatial positions in Input
    size_t InputChannels,   // number of 16-float input-channel blocks
    size_t FilterCount,     // output-channel blocks to process (1..4)
    size_t InputStride,     // bytes from IC block n to block n+1 in Input
    size_t FilterStride,    // bytes from output block n to block n+1 in Filter
    size_t OutputStride,    // bytes from output block n to block n+1 in Output
    size_t OutputCount,     // number of spatial output positions
    const float* Bias,      // [FilterCount * 16] floats, or nullptr
    unsigned KernelFlags
    )
{
    // StrideWidth is in bytes; convert once.
    const size_t InPosDelta = StrideWidth / sizeof(float);   // floats per input position step
    const size_t OutPosDelta = 16;                           // NCHWc: 16 floats per output position

    // Process in batches matching the 6-wide tile of the assembly baseline;
    // this keeps the filter panels warm in L1 for the common 13×13–26×26 cases.
    size_t Remaining = OutputCount;

    while (Remaining >= 6) {
        PwDispatchFilterCount<6>(Input, Filter, Output, StrideWidth, InputChannels,
            FilterCount, InputStride, FilterStride, OutputStride, Bias, KernelFlags);
        Input  += 6 * InPosDelta;
        Output += 6 * OutPosDelta;
        Remaining -= 6;
    }
    if (Remaining >= 3) {
        PwDispatchFilterCount<3>(Input, Filter, Output, StrideWidth, InputChannels,
            FilterCount, InputStride, FilterStride, OutputStride, Bias, KernelFlags);
        Input  += 3 * InPosDelta;
        Output += 3 * OutPosDelta;
        Remaining -= 3;
    }
    if (Remaining >= 2) {
        PwDispatchFilterCount<2>(Input, Filter, Output, StrideWidth, InputChannels,
            FilterCount, InputStride, FilterStride, OutputStride, Bias, KernelFlags);
        Input  += 2 * InPosDelta;
        Output += 2 * OutPosDelta;
        Remaining -= 2;
    }
    if (Remaining >= 1) {
        PwDispatchFilterCount<1>(Input, Filter, Output, StrideWidth, InputChannels,
            FilterCount, InputStride, FilterStride, OutputStride, Bias, KernelFlags);
    }
}
