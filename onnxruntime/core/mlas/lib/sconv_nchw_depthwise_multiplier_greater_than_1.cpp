/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_nchw_depthwise_multiplier_greater_than_1.cpp

Abstract:

    This module implements specialized single precision NCHW depthwise convolution
    kernels for depth multiplier greater than 1.

--*/

#include "mlasi.h"
#include <cassert>

// Specialized depthwise-with-multiplier kernels are currently disabled.
// Keep the previous implementation here for reference while excluding it from
// the active MLAS build.
#if 0

// Specialized depthwise-with-multiplier kernel for 7x7 stride-2 depth_multiplier=2.

static
void
MlasConv2dSingleChannel_CHW_Kernel7x7_PadAny_Stride2_Dilation1_DepthMultiplier2(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output
    )
/*++

Routine Description:

    This routine is an inner kernel to compute depthwise convolution on one channel input with depth multiplier 2.

Arguments:

    Parameters - conv parameters calculated based on conv parameters like padding, strides, dilations, etc.

    Input - input channel data start. Input is NCHW, so this pointer points to single H x W image data.

    Filter - whole filters are F x CpG x FH x FW. This filter points to two FH x FW filter blocks.

    Output - whole output is N x F x OH x OW. This pointer points to two OH x OW output blocks.

--*/
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;
    constexpr size_t SpecializedKernelSize = 7;

    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];
    const size_t OutputSize = Parameters->OutputSize;

    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

    const size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    const size_t PaddingTop = Parameters->Padding[HeightShapeIndex];
    const size_t PaddingLeft = Parameters->Padding[WidthShapeIndex];

    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    const size_t KernelSize = KernelHeight * KernelWidth;
    const float* Filter0 = Filter;
    const float* Filter1 = Filter + KernelSize;

    float* Output0 = Output;
    float* Output1 = Output + OutputSize;
    const float Beta = Parameters->Beta;

    auto ComputeScalarPoint = [&](size_t oh, size_t ow) {
        const ptrdiff_t InputOriginY = ptrdiff_t(oh * StrideHeight) - ptrdiff_t(PaddingTop);
        const ptrdiff_t InputOriginX = ptrdiff_t(ow * StrideWidth) - ptrdiff_t(PaddingLeft);

        float Sum0 = 0.0f;
        float Sum1 = 0.0f;

        for (size_t kh = 0; kh < KernelHeight; ++kh) {
            const ptrdiff_t ih = InputOriginY + ptrdiff_t(kh * DilationHeight);
            if (ih < 0 || size_t(ih) >= InputHeight) {
                continue;
            }

            const float* InputRow = Input + size_t(ih) * InputWidth;
            const float* FilterRow0 = Filter0 + kh * KernelWidth;
            const float* FilterRow1 = Filter1 + kh * KernelWidth;

            for (size_t kw = 0; kw < KernelWidth; ++kw) {
                const ptrdiff_t iw = InputOriginX + ptrdiff_t(kw * DilationWidth);
                if (iw < 0 || size_t(iw) >= InputWidth) {
                    continue;
                }

                const float InputValue = InputRow[size_t(iw)];
                Sum0 += InputValue * FilterRow0[kw];
                Sum1 += InputValue * FilterRow1[kw];
            }
        }

        const size_t OutputOffset = oh * OutputWidth + ow;
        if (Beta == 0.0f) {
            Output0[OutputOffset] = Sum0;
            Output1[OutputOffset] = Sum1;
        } else {
            Output0[OutputOffset] = Output0[OutputOffset] * Beta + Sum0;
            Output1[OutputOffset] = Output1[OutputOffset] * Beta + Sum1;
        }
    };

    const ptrdiff_t InputHeightLimit = ptrdiff_t(InputHeight) - ptrdiff_t(KernelHeight);
    const size_t MinOwForInterior = (PaddingLeft + StrideWidth - 1) / StrideWidth;
    const bool HasInteriorWidth = (PaddingLeft + InputWidth >= KernelWidth);
    const size_t MaxOwForInterior = HasInteriorWidth ? ((PaddingLeft + InputWidth - KernelWidth) / StrideWidth) : 0;

    for (size_t oh = 0; oh < OutputHeight; ++oh) {
        const ptrdiff_t InputOriginY = ptrdiff_t(oh * StrideHeight) - ptrdiff_t(PaddingTop);

        size_t InteriorBegin = 0;
        size_t InteriorEnd = 0;

        if ((InputOriginY >= 0) && (InputOriginY <= InputHeightLimit) && HasInteriorWidth && (MaxOwForInterior >= MinOwForInterior)) {
            InteriorBegin = MinOwForInterior;
            InteriorEnd = std::min(OutputWidth, MaxOwForInterior + 1);
        }

        size_t ow = 0;

        for (; ow < InteriorBegin; ++ow) {
            ComputeScalarPoint(oh, ow);
        }

        for (; ow + 1 < InteriorEnd; ow += 2) {
            const ptrdiff_t InputOriginX = ptrdiff_t(ow * StrideWidth) - ptrdiff_t(PaddingLeft);

            const float* InputRows[SpecializedKernelSize] = {
                Input + (size_t(InputOriginY + 0) * InputWidth) + size_t(InputOriginX),
                Input + (size_t(InputOriginY + 1) * InputWidth) + size_t(InputOriginX),
                Input + (size_t(InputOriginY + 2) * InputWidth) + size_t(InputOriginX),
                Input + (size_t(InputOriginY + 3) * InputWidth) + size_t(InputOriginX),
                Input + (size_t(InputOriginY + 4) * InputWidth) + size_t(InputOriginX),
                Input + (size_t(InputOriginY + 5) * InputWidth) + size_t(InputOriginX),
                Input + (size_t(InputOriginY + 6) * InputWidth) + size_t(InputOriginX),
            };

            const float* FilterRows0[SpecializedKernelSize] = {
                Filter0 + 0 * SpecializedKernelSize,
                Filter0 + 1 * SpecializedKernelSize,
                Filter0 + 2 * SpecializedKernelSize,
                Filter0 + 3 * SpecializedKernelSize,
                Filter0 + 4 * SpecializedKernelSize,
                Filter0 + 5 * SpecializedKernelSize,
                Filter0 + 6 * SpecializedKernelSize,
            };

            const float* FilterRows1[SpecializedKernelSize] = {
                Filter1 + 0 * SpecializedKernelSize,
                Filter1 + 1 * SpecializedKernelSize,
                Filter1 + 2 * SpecializedKernelSize,
                Filter1 + 3 * SpecializedKernelSize,
                Filter1 + 4 * SpecializedKernelSize,
                Filter1 + 5 * SpecializedKernelSize,
                Filter1 + 6 * SpecializedKernelSize,
            };

            float Sum00 = 0.0f;
            float Sum01 = 0.0f;
            float Sum10 = 0.0f;
            float Sum11 = 0.0f;

            for (size_t kh = 0; kh < SpecializedKernelSize; ++kh) {
                const float* InputRow = InputRows[kh];
                const float* FilterRow0 = FilterRows0[kh];
                const float* FilterRow1 = FilterRows1[kh];

                for (size_t kw = 0; kw < SpecializedKernelSize; ++kw) {
                    const float InputValue0 = InputRow[kw];
                    const float InputValue1 = InputRow[kw + StrideWidth];

                    Sum00 += InputValue0 * FilterRow0[kw];
                    Sum10 += InputValue0 * FilterRow1[kw];
                    Sum01 += InputValue1 * FilterRow0[kw];
                    Sum11 += InputValue1 * FilterRow1[kw];
                }
            }

            const size_t OutputOffset = oh * OutputWidth + ow;
            if (Beta == 0.0f) {
                Output0[OutputOffset] = Sum00;
                Output0[OutputOffset + 1] = Sum01;
                Output1[OutputOffset] = Sum10;
                Output1[OutputOffset + 1] = Sum11;
            } else {
                Output0[OutputOffset] = Output0[OutputOffset] * Beta + Sum00;
                Output0[OutputOffset + 1] = Output0[OutputOffset + 1] * Beta + Sum01;
                Output1[OutputOffset] = Output1[OutputOffset] * Beta + Sum10;
                Output1[OutputOffset + 1] = Output1[OutputOffset + 1] * Beta + Sum11;
            }
        }

        for (; ow < InteriorEnd; ++ow) {
            const ptrdiff_t InputOriginX = ptrdiff_t(ow * StrideWidth) - ptrdiff_t(PaddingLeft);

                const float* InputRow0 = Input + (size_t(InputOriginY + 0) * InputWidth) + size_t(InputOriginX);
                const float* InputRow1 = Input + (size_t(InputOriginY + 1) * InputWidth) + size_t(InputOriginX);
                const float* InputRow2 = Input + (size_t(InputOriginY + 2) * InputWidth) + size_t(InputOriginX);
                const float* InputRow3 = Input + (size_t(InputOriginY + 3) * InputWidth) + size_t(InputOriginX);
                const float* InputRow4 = Input + (size_t(InputOriginY + 4) * InputWidth) + size_t(InputOriginX);
                const float* InputRow5 = Input + (size_t(InputOriginY + 5) * InputWidth) + size_t(InputOriginX);
                const float* InputRow6 = Input + (size_t(InputOriginY + 6) * InputWidth) + size_t(InputOriginX);

                const float* FilterRow00 = Filter0 + 0 * KernelWidth;
                const float* FilterRow01 = Filter0 + 1 * KernelWidth;
                const float* FilterRow02 = Filter0 + 2 * KernelWidth;
                const float* FilterRow03 = Filter0 + 3 * KernelWidth;
                const float* FilterRow04 = Filter0 + 4 * KernelWidth;
                const float* FilterRow05 = Filter0 + 5 * KernelWidth;
                const float* FilterRow06 = Filter0 + 6 * KernelWidth;

                const float* FilterRow10 = Filter1 + 0 * KernelWidth;
                const float* FilterRow11 = Filter1 + 1 * KernelWidth;
                const float* FilterRow12 = Filter1 + 2 * KernelWidth;
                const float* FilterRow13 = Filter1 + 3 * KernelWidth;
                const float* FilterRow14 = Filter1 + 4 * KernelWidth;
                const float* FilterRow15 = Filter1 + 5 * KernelWidth;
                const float* FilterRow16 = Filter1 + 6 * KernelWidth;

                float Sum0 = 0.0f;
                float Sum1 = 0.0f;

                Sum0 += InputRow0[0] * FilterRow00[0] + InputRow0[1] * FilterRow00[1] + InputRow0[2] * FilterRow00[2] + InputRow0[3] * FilterRow00[3] + InputRow0[4] * FilterRow00[4] + InputRow0[5] * FilterRow00[5] + InputRow0[6] * FilterRow00[6];
                Sum1 += InputRow0[0] * FilterRow10[0] + InputRow0[1] * FilterRow10[1] + InputRow0[2] * FilterRow10[2] + InputRow0[3] * FilterRow10[3] + InputRow0[4] * FilterRow10[4] + InputRow0[5] * FilterRow10[5] + InputRow0[6] * FilterRow10[6];
                Sum0 += InputRow1[0] * FilterRow01[0] + InputRow1[1] * FilterRow01[1] + InputRow1[2] * FilterRow01[2] + InputRow1[3] * FilterRow01[3] + InputRow1[4] * FilterRow01[4] + InputRow1[5] * FilterRow01[5] + InputRow1[6] * FilterRow01[6];
                Sum1 += InputRow1[0] * FilterRow11[0] + InputRow1[1] * FilterRow11[1] + InputRow1[2] * FilterRow11[2] + InputRow1[3] * FilterRow11[3] + InputRow1[4] * FilterRow11[4] + InputRow1[5] * FilterRow11[5] + InputRow1[6] * FilterRow11[6];
                Sum0 += InputRow2[0] * FilterRow02[0] + InputRow2[1] * FilterRow02[1] + InputRow2[2] * FilterRow02[2] + InputRow2[3] * FilterRow02[3] + InputRow2[4] * FilterRow02[4] + InputRow2[5] * FilterRow02[5] + InputRow2[6] * FilterRow02[6];
                Sum1 += InputRow2[0] * FilterRow12[0] + InputRow2[1] * FilterRow12[1] + InputRow2[2] * FilterRow12[2] + InputRow2[3] * FilterRow12[3] + InputRow2[4] * FilterRow12[4] + InputRow2[5] * FilterRow12[5] + InputRow2[6] * FilterRow12[6];
                Sum0 += InputRow3[0] * FilterRow03[0] + InputRow3[1] * FilterRow03[1] + InputRow3[2] * FilterRow03[2] + InputRow3[3] * FilterRow03[3] + InputRow3[4] * FilterRow03[4] + InputRow3[5] * FilterRow03[5] + InputRow3[6] * FilterRow03[6];
                Sum1 += InputRow3[0] * FilterRow13[0] + InputRow3[1] * FilterRow13[1] + InputRow3[2] * FilterRow13[2] + InputRow3[3] * FilterRow13[3] + InputRow3[4] * FilterRow13[4] + InputRow3[5] * FilterRow13[5] + InputRow3[6] * FilterRow13[6];
                Sum0 += InputRow4[0] * FilterRow04[0] + InputRow4[1] * FilterRow04[1] + InputRow4[2] * FilterRow04[2] + InputRow4[3] * FilterRow04[3] + InputRow4[4] * FilterRow04[4] + InputRow4[5] * FilterRow04[5] + InputRow4[6] * FilterRow04[6];
                Sum1 += InputRow4[0] * FilterRow14[0] + InputRow4[1] * FilterRow14[1] + InputRow4[2] * FilterRow14[2] + InputRow4[3] * FilterRow14[3] + InputRow4[4] * FilterRow14[4] + InputRow4[5] * FilterRow14[5] + InputRow4[6] * FilterRow14[6];
                Sum0 += InputRow5[0] * FilterRow05[0] + InputRow5[1] * FilterRow05[1] + InputRow5[2] * FilterRow05[2] + InputRow5[3] * FilterRow05[3] + InputRow5[4] * FilterRow05[4] + InputRow5[5] * FilterRow05[5] + InputRow5[6] * FilterRow05[6];
                Sum1 += InputRow5[0] * FilterRow15[0] + InputRow5[1] * FilterRow15[1] + InputRow5[2] * FilterRow15[2] + InputRow5[3] * FilterRow15[3] + InputRow5[4] * FilterRow15[4] + InputRow5[5] * FilterRow15[5] + InputRow5[6] * FilterRow15[6];
                Sum0 += InputRow6[0] * FilterRow06[0] + InputRow6[1] * FilterRow06[1] + InputRow6[2] * FilterRow06[2] + InputRow6[3] * FilterRow06[3] + InputRow6[4] * FilterRow06[4] + InputRow6[5] * FilterRow06[5] + InputRow6[6] * FilterRow06[6];
                Sum1 += InputRow6[0] * FilterRow16[0] + InputRow6[1] * FilterRow16[1] + InputRow6[2] * FilterRow16[2] + InputRow6[3] * FilterRow16[3] + InputRow6[4] * FilterRow16[4] + InputRow6[5] * FilterRow16[5] + InputRow6[6] * FilterRow16[6];

            const size_t OutputOffset = oh * OutputWidth + ow;
            if (Beta == 0.0f) {
                Output0[OutputOffset] = Sum0;
                Output1[OutputOffset] = Sum1;
            } else {
                Output0[OutputOffset] = Output0[OutputOffset] * Beta + Sum0;
                Output1[OutputOffset] = Output1[OutputOffset] * Beta + Sum1;
            }
        }

        for (; ow < OutputWidth; ++ow) {
            ComputeScalarPoint(oh, ow);
        }
    }
}

void
MlasConvDepthwiseWithMultiplierFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output
    )
/*++

Routine Description:

    This routine is an inner kernel to compute depthwise convolution for one input channel
    with depth multiplier 2.

Arguments:

    Parameters - conv parameters calculated based on conv parameters like padding, strides, dilations, etc.

    Input - input channel data start. Input is NCHW, so this pointer points to single H x W image data.

    Filter - whole filters are F x CpG x FH x FW. This filter points to two FH x FW filter blocks.

    Output - whole output is N x F x OH x OW. This pointer points to two OH x OW output blocks.

Note:
    No checking here as it is inner loop. Logic in generating Parameters controls the check.

    Current specialized constraints are 2D kernel 7x7 with strides=2, dilations=1, and generic padding.

    Will add general case and more special case if needed later.
--*/
{
    assert(Parameters->KernelShape[0] == 7);
    assert(Parameters->KernelShape[1] == 7);
    assert(Parameters->StrideShape[0] == 2);
    assert(Parameters->StrideShape[1] == 2);
    assert(Parameters->DilationShape[0] == 1);
    assert(Parameters->DilationShape[1] == 1);

    // Kernel dispatch
#if defined(MLAS_TARGET_AMD64)
    if (GetMlasPlatform().Avx512Supported_) {
        MlasConvDepthwiseWithMultiplierFloatCHWKernel7x7Stride2DepthMultiplier2Avx512F(
            Parameters, Input, Filter, Output);
        return;
    }

    MlasConv2dSingleChannel_CHW_Kernel7x7_PadAny_Stride2_Dilation1_DepthMultiplier2(
        Parameters, Input, Filter, Output);
}
#endif

#endif
