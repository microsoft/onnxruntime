/*++
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
Module Name:
    sconv_nchw_depthwise_multiplier_greater_than_1_avx512f.cpp
Abstract:
        This module implements the AVX512F kernel for the exact MobileClip grouped
        projection case:

            - CHW input/output layout per group slice
            - input channels per group = 1
            - output channels per group = 2
            - kernel = 7x7
            - stride = 2x2
            - padding = 3,3,3,3
            - dilation = 1x1

        The outer dispatch is expected to guarantee these constraints.
--*/

#include "mlasi.h"

#if defined(MLAS_TARGET_AMD64)

namespace {

MLAS_FORCEINLINE
void
MlasConv2dSingleChannelCHWKernel7x7Pad3Stride2Dilation1DepthMultiplier2Scalar(
    const float* Input,
    size_t InputHeight,
    size_t InputWidth,
    const float* Filter0,
    const float* Filter1,
    float* Output0,
    float* Output1,
    size_t OutputWidth,
    size_t oh,
    size_t ow,
    float Beta
    )
/*++

Routine Description:

    Computes one border output point for the exact MobileClip
    7x7/pad-3/stride-2/dilation-1, multiplier-2 case.

    This helper is only used by the AVX512 implementation for border handling;
    it is not a generic fallback dispatch path despite the scalar
    implementation.

--*/
{
    const ptrdiff_t input_origin_y = static_cast<ptrdiff_t>(oh * 2) - 3;
    const ptrdiff_t input_origin_x = static_cast<ptrdiff_t>(ow * 2) - 3;
    const size_t output_index = oh * OutputWidth + ow;

    float acc0 = (Beta == 0.0f) ? 0.0f : Output0[output_index] * Beta;
    float acc1 = (Beta == 0.0f) ? 0.0f : Output1[output_index] * Beta;

    for (size_t kh = 0; kh < 7; ++kh) {
        const ptrdiff_t ih = input_origin_y + static_cast<ptrdiff_t>(kh);
        if (ih < 0 || ih >= static_cast<ptrdiff_t>(InputHeight)) {
            continue;
        }

        const float* input_row = Input + static_cast<size_t>(ih) * InputWidth;
        const float* filter0_row = Filter0 + kh * 7;
        const float* filter1_row = Filter1 + kh * 7;

        for (size_t kw = 0; kw < 7; ++kw) {
            const ptrdiff_t iw = input_origin_x + static_cast<ptrdiff_t>(kw);
            if (iw < 0 || iw >= static_cast<ptrdiff_t>(InputWidth)) {
                continue;
            }

            const float input_value = input_row[static_cast<size_t>(iw)];
            acc0 += input_value * filter0_row[kw];
            acc1 += input_value * filter1_row[kw];
        }
    }

    Output0[output_index] = acc0;
    Output1[output_index] = acc1;
}

}  // namespace

void
MlasConvDepthwiseMultiplier2CHWKernel7x7S2Avx512F(
    const float* Input,
    size_t InputHeight,
    size_t InputWidth,
    const float* Filter,
    float* Output,
    size_t OutputHeight,
    size_t OutputWidth,
    float Beta
    )
/*++

Routine Description:

    Computes one group slice of the exact MobileClip grouped projection case.

Assumptions:

    - Input and output are CHW tensors for a single group slice.
    - Filter is OIHW for a single group slice with exactly two output channels.
    - Kernel = 7x7, stride = 2, padding = 3, dilation = 1.
    - OutputHeight and OutputWidth match the supplied input geometry.

Return Value:

    None.

--*/
{
    constexpr size_t KernelSize = 7;
    constexpr __mmask16 ValidKernelMask = 0x007F;

    const float* Filter0 = Filter;
    const float* Filter1 = Filter + KernelSize * KernelSize;
    float* Output0 = Output;
    float* Output1 = Output + (OutputHeight * OutputWidth);

    for (size_t oh = 0; oh < OutputHeight; ++oh) {
        const ptrdiff_t input_origin_y = static_cast<ptrdiff_t>(oh * 2) - 3;
        const bool interior_y = input_origin_y >= 0 &&
                                (input_origin_y + static_cast<ptrdiff_t>(KernelSize)) <= static_cast<ptrdiff_t>(InputHeight);

        for (size_t ow = 0; ow < OutputWidth; ++ow) {
            const ptrdiff_t input_origin_x = static_cast<ptrdiff_t>(ow * 2) - 3;
            const bool interior_x = input_origin_x >= 0 &&
                                    (input_origin_x + static_cast<ptrdiff_t>(KernelSize)) <= static_cast<ptrdiff_t>(InputWidth);

            if (!(interior_y && interior_x)) {
                MlasConv2dSingleChannelCHWKernel7x7Pad3Stride2Dilation1DepthMultiplier2Scalar(
                    Input, InputHeight, InputWidth, Filter0, Filter1, Output0, Output1, OutputWidth, oh, ow, Beta);
                continue;
            }

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            for (size_t kh = 0; kh < KernelSize; ++kh) {
                const float* input_row = Input + (static_cast<size_t>(input_origin_y) + kh) * InputWidth + static_cast<size_t>(input_origin_x);
                const __m512 input_vec = _mm512_maskz_loadu_ps(ValidKernelMask, input_row);
                const __m512 filter0_vec = _mm512_maskz_loadu_ps(ValidKernelMask, Filter0 + kh * KernelSize);
                const __m512 filter1_vec = _mm512_maskz_loadu_ps(ValidKernelMask, Filter1 + kh * KernelSize);

                acc0 = _mm512_fmadd_ps(input_vec, filter0_vec, acc0);
                acc1 = _mm512_fmadd_ps(input_vec, filter1_vec, acc1);
            }

            const size_t output_index = oh * OutputWidth + ow;
            float acc0_scalar = _mm512_reduce_add_ps(acc0);
            float acc1_scalar = _mm512_reduce_add_ps(acc1);

            if (Beta != 0.0f) {
                acc0_scalar += Output0[output_index] * Beta;
                acc1_scalar += Output1[output_index] * Beta;
            }

            Output0[output_index] = acc0_scalar;
            Output1[output_index] = acc1_scalar;
        }
    }
}

#endif
