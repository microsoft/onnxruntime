/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qdwconv.cpp

Abstract:

    This module implements the quantized integer depthwise convolution routines.

--*/

#include "mlasi.h"

template <typename InputType, typename FilterType>
void
MLASCALL
MlasConvDepthwiseKernel(
    const InputType* const* Input,
    InputType InputZeroPoint,
    const FilterType* Filter,
    FilterType FilterZeroPoint,
    int32_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    )
{
    //
    // TODO Modify MlasConvDepthwiseGetKernelOutputCnt() function if this kernel
    // is further optimized.
    //
#if defined(MLAS_SSE2_INTRINSICS)
    const __m128i ZeroVector = _mm_setzero_si128();
    const __m128i InputZeroPointVector = _mm_set1_epi16(InputZeroPoint);
    const __m128i FilterZeroPointVector = _mm_set1_epi16(FilterZeroPoint);
#elif defined(MLAS_NEON_INTRINSICS)
    const uint8x8_t InputZeroPointVector = vdup_n_u8(uint8_t(InputZeroPoint));
    const uint8x8_t FilterZeroPointVector = vdup_n_u8(uint8_t(FilterZeroPoint));
#elif defined(MLAS_LSX_INTRINSICS)
    const __m128i ZeroVector = __lsx_vldi(0);
    const __m128i InputZeroPointVector = __lsx_vreplgr2vr_h(InputZeroPoint);
    const __m128i FilterZeroPointVector = __lsx_vreplgr2vr_h(FilterZeroPoint);
#endif

    while (OutputCount > 0) {
        size_t ChannelOffset = 0;
        size_t c = Channels;

#if defined(MLAS_SSE2_INTRINSICS)

        while (c >= 8) {
            __m128i Accumulator0 = _mm_setzero_si128();
            __m128i Accumulator1 = _mm_setzero_si128();
            size_t ChannelKernelOffset = ChannelOffset;

            for (size_t k = 0; k < KernelSize; k++) {
                __m128i InputVector = _mm_loadl_epi64((const __m128i*)&Input[k][ChannelOffset]);
                __m128i FilterVector =
                    _mm_loadl_epi64((const __m128i*)&Filter[ChannelKernelOffset]);

                if (std::is_signed<InputType>::value) {
                    InputVector = _mm_srai_epi16(_mm_unpacklo_epi8(ZeroVector, InputVector), 8);
                } else {
                    InputVector = _mm_unpacklo_epi8(InputVector, ZeroVector);
                }

                if (std::is_signed<FilterType>::value) {
                    FilterVector = _mm_srai_epi16(_mm_unpacklo_epi8(ZeroVector, FilterVector), 8);
                } else {
                    FilterVector = _mm_unpacklo_epi8(FilterVector, ZeroVector);
                }

                InputVector = _mm_sub_epi16(InputVector, InputZeroPointVector);
                FilterVector = _mm_sub_epi16(FilterVector, FilterZeroPointVector);

                // N.B. Emulate PMULLD functionality on SSE2 by computing the low
                // and high parts of the result and interleaving the results.
                __m128i MultiplyLowWords = _mm_mullo_epi16(InputVector, FilterVector);
                __m128i MultiplyHighWords = _mm_mulhi_epi16(InputVector, FilterVector);
                __m128i Multiply0 = _mm_unpacklo_epi16(MultiplyLowWords, MultiplyHighWords);
                __m128i Multiply1 = _mm_unpackhi_epi16(MultiplyLowWords, MultiplyHighWords);

                Accumulator0 = _mm_add_epi32(Accumulator0, Multiply0);
                Accumulator1 = _mm_add_epi32(Accumulator1, Multiply1);
                ChannelKernelOffset += Channels;
            }

            _mm_storeu_si128((__m128i*)&Output[0], Accumulator0);
            _mm_storeu_si128((__m128i*)&Output[4], Accumulator1);
            Output += 8;

            ChannelOffset += 8;
            c -= 8;
        }

#elif defined(MLAS_NEON_INTRINSICS)

        while (c >= 8) {
            int32x4_t Accumulator0 = vdupq_n_s32(0);
            int32x4_t Accumulator1 = vdupq_n_s32(0);
            size_t ChannelKernelOffset = ChannelOffset;

            for (size_t k = 0; k < KernelSize; k++) {
                uint8x8_t InputVector =
                    vld1_u8(reinterpret_cast<const uint8_t*>(&Input[k][ChannelOffset]));
                uint8x8_t FilterVector =
                    vld1_u8(reinterpret_cast<const uint8_t*>(&Filter[ChannelKernelOffset]));

                int16x8_t InputVector16;
                if (std::is_signed<InputType>::value) {
                    InputVector16 = vsubl_s8(vreinterpret_s8_u8(InputVector),
                                             vreinterpret_s8_u8(InputZeroPointVector));
                } else {
                    InputVector16 =
                        vreinterpretq_s16_u16(vsubl_u8(InputVector, InputZeroPointVector));
                }

                int16x8_t FilterVector16;
                if (std::is_signed<FilterType>::value) {
                    FilterVector16 = vsubl_s8(vreinterpret_s8_u8(FilterVector),
                                              vreinterpret_s8_u8(FilterZeroPointVector));
                } else {
                    FilterVector16 =
                        vreinterpretq_s16_u16(vsubl_u8(FilterVector, FilterZeroPointVector));
                }

                Accumulator0 = vmlal_s16(Accumulator0, vget_low_s16(InputVector16),
                                         vget_low_s16(FilterVector16));
#if defined(MLAS_NEON64_INTRINSICS)
                Accumulator1 = vmlal_high_s16(Accumulator1, InputVector16, FilterVector16);
#else
                Accumulator1 = vmlal_s16(Accumulator1, vget_high_s16(InputVector16),
                                         vget_high_s16(FilterVector16));
#endif

                ChannelKernelOffset += Channels;
            }

            vst1q_s32(&Output[0], Accumulator0);
            vst1q_s32(&Output[4], Accumulator1);
            Output += 8;

            ChannelOffset += 8;
            c -= 8;
        }
#elif defined(MLAS_LSX_INTRINSICS)

        while (c >= 8) {
            __m128i Accumulator0 = __lsx_vldi(0);
            __m128i Accumulator1 = __lsx_vldi(0);
            size_t ChannelKernelOffset = ChannelOffset;

            for (size_t k = 0; k < KernelSize; k++) {
                __m128i InputVector = __lsx_vld((const __m128i*)&Input[k][ChannelOffset], 0);
                __lsx_vinsgr2vr_d(InputVector, 0, 1);
                __m128i FilterVector =
                    __lsx_vld((const __m128i*)&Filter[ChannelKernelOffset], 0);
                __lsx_vinsgr2vr_d(FilterVector, 0, 1);

                if (std::is_signed<InputType>::value) {
                    InputVector = __lsx_vsrai_h(__lsx_vilvl_b(InputVector, ZeroVector), 8);
                } else {
                    InputVector = __lsx_vilvl_b(ZeroVector, InputVector );
                }

                if (std::is_signed<FilterType>::value) {
                    FilterVector = __lsx_vsrai_h(__lsx_vilvl_b(FilterVector, ZeroVector), 8);
                } else {
                    FilterVector = __lsx_vilvl_b(ZeroVector, FilterVector);
                }

                InputVector = __lsx_vsub_h(InputVector, InputZeroPointVector);
                FilterVector = __lsx_vsub_h(FilterVector, FilterZeroPointVector);

                // N.B. Emulate PMULLD functionality on LSX by computing the low
                // and high parts of the result and interleaving the results.
                __m128i MultiplyLowWords = __lsx_vmul_h(InputVector, FilterVector);
                __m128i MultiplyHighWords = __lsx_vmuh_h(InputVector, FilterVector);
                __m128i Multiply0 = __lsx_vilvl_h(MultiplyHighWords, MultiplyLowWords);
                __m128i Multiply1 = __lsx_vilvh_h(MultiplyHighWords, MultiplyLowWords);

                Accumulator0 = __lsx_vadd_w(Accumulator0, Multiply0);
                Accumulator1 = __lsx_vadd_w(Accumulator1, Multiply1);
                ChannelKernelOffset += Channels;
            }

            __lsx_vst(Accumulator0, (__m128i*)&Output[0], 0);
            __lsx_vst(Accumulator1, (__m128i*)&Output[4], 0);
            Output += 8;

            ChannelOffset += 8;
            c -= 8;
        }

#endif

        while (c > 0) {
            int32_t Accumulator = 0;
            size_t ChannelKernelOffset = ChannelOffset;

            for (size_t k = 0; k < KernelSize; k++) {
                int32_t InputValue = int32_t(Input[k][ChannelOffset]) - InputZeroPoint;
                int32_t FilterValue = int32_t(Filter[ChannelKernelOffset]) - FilterZeroPoint;

                Accumulator += InputValue * FilterValue;
                ChannelKernelOffset += Channels;
            }

            *Output++ = Accumulator;

            ChannelOffset += 1;
            c -= 1;
        }

        Input += KernelSize;
        OutputCount -= 1;
    }
}

template
void
MLASCALL
MlasConvDepthwiseKernel(
    const uint8_t* const* Input,
    uint8_t InputZeroPoint,
    const int8_t* Filter,
    int8_t FilterZeroPoint,
    int32_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    );

template
void
MLASCALL
MlasConvDepthwiseKernel(
    const uint8_t* const* Input,
    uint8_t InputZeroPoint,
    const uint8_t* Filter,
    uint8_t FilterZeroPoint,
    int32_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    );

template
void
MLASCALL
MlasConvDepthwiseKernel(
    const int8_t* const* Input,
    int8_t InputZeroPoint,
    const int8_t* Filter,
    int8_t FilterZeroPoint,
    int32_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    );

template
void
MLASCALL
MlasConvDepthwiseKernel(
    const int8_t* const* Input,
    int8_t InputZeroPoint,
    const uint8_t* Filter,
    uint8_t FilterZeroPoint,
    int32_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    );

void
MLASCALL
MlasConvDepthwise(
    const void* const* Input,
    int32_t InputZeroPoint,
    bool InputIsSigned,
    const void* Filter,
    int32_t FilterZeroPoint,
    bool FilterIsSigned,
    int32_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    )
/*++

Routine Description:

    This routine implements the depthwise convolution operation.

    The input is supplied as an indirection buffer. Every pointer in the
    indirection buffer points at a Channels length vector (either from the
    input tensor or a vector of padding values). These are grouped in batches
    of length KernelSize that are processed by the kernel to produce a single
    output of length Channels. These batches are then repeated OutputCount
    times.

    The filter tensor is organized in HW1O format, so the length of each row of
    the filter tensor is Channels. The number of columns of the filter tensor
    is KernelSize.

Arguments:

    Input - Supplies an indirection buffer to the elements of the input tensor.

    InputZeroPoint - Supplies the zero point offset of the input tensor.

    InputIsSigned - Supplies true if the input tensor is signed data, else
        false if the input tensor is unsigned data.

    Filter - Supplies the filter tensor.

    FilterZeroPoint - Supplies the zero point offset of the filter tensor.

    FilterIsSigned - Supplies true if the filter tensor is signed data, else
        false if the filter tensor is unsigned data.

    Output - Supplies the output tensor in channels last format.

    Channels - Supplies the number of channels.

    OutputCount - Supplies the number of channel sized output elements to
        produce.

    KernelSize - Supplies the total number of channel sized kernel elements to
        consume.

Return Value:

    None.

--*/
{
    if (InputIsSigned) {
        if (FilterIsSigned) {

            GetMlasPlatform().ConvDepthwiseS8S8Kernel(
                reinterpret_cast<const int8_t* const*>(Input), static_cast<int8_t>(InputZeroPoint),
                reinterpret_cast<const int8_t*>(Filter), static_cast<int8_t>(FilterZeroPoint),
                Output, Channels, OutputCount, KernelSize
                );
        } else {

            GetMlasPlatform().ConvDepthwiseS8U8Kernel(
                reinterpret_cast<const int8_t* const*>(Input), static_cast<int8_t>(InputZeroPoint),
                reinterpret_cast<const uint8_t*>(Filter), static_cast<uint8_t>(FilterZeroPoint),
                Output, Channels, OutputCount, KernelSize
                );
        }
    } else {
        if (FilterIsSigned) {

            GetMlasPlatform().ConvDepthwiseU8S8Kernel(
                reinterpret_cast<const uint8_t* const*>(Input), static_cast<uint8_t>(InputZeroPoint),
                reinterpret_cast<const int8_t*>(Filter), static_cast<int8_t>(FilterZeroPoint),
                Output, Channels, OutputCount, KernelSize
                );
        } else {

            GetMlasPlatform().ConvDepthwiseU8U8Kernel(
                reinterpret_cast<const uint8_t* const*>(Input), static_cast<uint8_t>(InputZeroPoint),
                reinterpret_cast<const uint8_t*>(Filter), static_cast<uint8_t>(FilterZeroPoint),
                Output, Channels, OutputCount, KernelSize
                );
        }
    }
}
