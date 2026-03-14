/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_nchw_depthwise_multiplier_greater_than_1_avx512f.cpp

Abstract:

    This module implements an AVX512F hot-path micro-kernel for the specialized
    single precision NCHW depthwise convolution kernel with depth multiplier 2.

--*/

#include "mlasi.h"

// Specialized AVX512F hot path for depthwise-with-multiplier disabled for A/B testing.
#if 0

namespace {

constexpr size_t SpecializedKernelSize = 7;
constexpr size_t SpecializedStride = 2;
constexpr __mmask16 SpecializedKernelMask = (__mmask16)((1u << SpecializedKernelSize) - 1);

template<size_t OutputCount>
MLAS_FORCEINLINE void
MlasConv2dSingleChannelCHWKernel7x7PadAnyStride2Dilation1DepthMultiplier2HotPathAvx512F(
    const float* const* InputRows,
    const float* const* FilterRows0,
    const float* const* FilterRows1,
    float* Output0,
    float* Output1,
    size_t OutputOffset,
    float Beta
    )
{
    __m512 Sum0[OutputCount];
    __m512 Sum1[OutputCount];

    for (size_t index = 0; index < OutputCount; ++index) {
        Sum0[index] = _mm512_setzero_ps();
        Sum1[index] = _mm512_setzero_ps();
    }

    for (size_t kh = 0; kh < SpecializedKernelSize; ++kh) {
        const __m512 FilterVector0 = _mm512_maskz_loadu_ps(SpecializedKernelMask, FilterRows0[kh]);
        const __m512 FilterVector1 = _mm512_maskz_loadu_ps(SpecializedKernelMask, FilterRows1[kh]);

        if constexpr (OutputCount > 0) {
            const __m512 InputVector = _mm512_maskz_loadu_ps(SpecializedKernelMask, InputRows[kh]);
            Sum0[0] = _mm512_fmadd_ps(InputVector, FilterVector0, Sum0[0]);
            Sum1[0] = _mm512_fmadd_ps(InputVector, FilterVector1, Sum1[0]);
        }

        if constexpr (OutputCount > 1) {
            const __m512 InputVector = _mm512_maskz_loadu_ps(SpecializedKernelMask, InputRows[kh] + SpecializedStride);
            Sum0[1] = _mm512_fmadd_ps(InputVector, FilterVector0, Sum0[1]);
            Sum1[1] = _mm512_fmadd_ps(InputVector, FilterVector1, Sum1[1]);
        }

        if constexpr (OutputCount > 2) {
            const __m512 InputVector = _mm512_maskz_loadu_ps(SpecializedKernelMask, InputRows[kh] + 2 * SpecializedStride);
            Sum0[2] = _mm512_fmadd_ps(InputVector, FilterVector0, Sum0[2]);
            Sum1[2] = _mm512_fmadd_ps(InputVector, FilterVector1, Sum1[2]);
        }

        if constexpr (OutputCount > 3) {
            const __m512 InputVector = _mm512_maskz_loadu_ps(SpecializedKernelMask, InputRows[kh] + 3 * SpecializedStride);
            Sum0[3] = _mm512_fmadd_ps(InputVector, FilterVector0, Sum0[3]);
            Sum1[3] = _mm512_fmadd_ps(InputVector, FilterVector1, Sum1[3]);
        }
    }

    for (size_t index = 0; index < OutputCount; ++index) {
        const float ReducedSum0 = _mm512_reduce_add_ps(Sum0[index]);
        const float ReducedSum1 = _mm512_reduce_add_ps(Sum1[index]);

        if (Beta == 0.0f) {
            Output0[OutputOffset + index] = ReducedSum0;
            Output1[OutputOffset + index] = ReducedSum1;
        } else {
            Output0[OutputOffset + index] = Output0[OutputOffset + index] * Beta + ReducedSum0;
            Output1[OutputOffset + index] = Output1[OutputOffset + index] * Beta + ReducedSum1;
        }
    }
}

}  // namespace

void
MlasConvDepthwiseWithMultiplierFloatCHWKernel7x7Stride2DepthMultiplier2Avx512F(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output
    )
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

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

        const size_t ScalarOutputOffset = oh * OutputWidth + ow;
        if (Beta == 0.0f) {
            Output0[ScalarOutputOffset] = Sum0;
            Output1[ScalarOutputOffset] = Sum1;
        } else {
            Output0[ScalarOutputOffset] = Output0[ScalarOutputOffset] * Beta + Sum0;
            Output1[ScalarOutputOffset] = Output1[ScalarOutputOffset] * Beta + Sum1;
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

        for (; ow + 3 < InteriorEnd; ow += 4) {
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

            const size_t OutputOffset = oh * OutputWidth + ow;
            MlasConv2dSingleChannelCHWKernel7x7PadAnyStride2Dilation1DepthMultiplier2HotPathAvx512F<4>(
                InputRows, FilterRows0, FilterRows1, Output0, Output1, OutputOffset, Beta);
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

            const size_t OutputOffset = oh * OutputWidth + ow;
            MlasConv2dSingleChannelCHWKernel7x7PadAnyStride2Dilation1DepthMultiplier2HotPathAvx512F<2>(
                InputRows, FilterRows0, FilterRows1, Output0, Output1, OutputOffset, Beta);
        }

        for (; ow < InteriorEnd; ++ow) {
            ComputeScalarPoint(oh, ow);
        }

        for (; ow < OutputWidth; ++ow) {
            ComputeScalarPoint(oh, ow);
        }
    }
}
#endif
