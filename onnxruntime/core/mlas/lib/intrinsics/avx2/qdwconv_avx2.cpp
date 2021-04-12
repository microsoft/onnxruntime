/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qdwconv_avx2.cpp

Abstract:

    This module implements the quantized integer depthwise convolution kernels.

    This implementation uses AVX2 instructions.

--*/

#include "mlasi.h"

template<typename FilterType>
void
MLASCALL
MlasConvDepthwiseKernelAvx2(
    const uint8_t* const* Input,
    uint8_t InputZeroPoint,
    const FilterType* Filter,
    FilterType FilterZeroPoint,
    int32_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    )
{
    const __m256i InputZeroPointVector = _mm256_set1_epi16(InputZeroPoint);
    const __m256i FilterZeroPointVector = _mm256_set1_epi16(FilterZeroPoint);

    while (OutputCount > 0) {

        size_t ChannelOffset = 0;
        size_t c = Channels;

        while (c >= 16) {

            __m256i Accumulator0 = _mm256_setzero_si256();
            __m256i Accumulator1 = _mm256_setzero_si256();
            size_t ChannelKernelOffset = ChannelOffset;

            for (size_t k = 0; k < KernelSize; k++) {

                __m256i InputVector = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&Input[k][ChannelOffset]));
                __m256i FilterVector;

                if (std::is_signed<FilterType>::value) {
                    FilterVector = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)&Filter[ChannelKernelOffset]));
                } else {
                    FilterVector = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&Filter[ChannelKernelOffset]));
                }

                InputVector = _mm256_sub_epi16(InputVector, InputZeroPointVector);
                FilterVector = _mm256_sub_epi16(FilterVector, FilterZeroPointVector);

                // N.B. The original SSE2 implementation used PMULLW/PMULHW in
                // order to emulate the SSE 4.1 PMULLD instruction, however this
                // implementation ends up being faster for some CPUs than
                // extending to 32-bits and using PMULLD.
                __m256i MultiplyLowWords = _mm256_mullo_epi16(InputVector, FilterVector);
                __m256i MultiplyHighWords = _mm256_mulhi_epi16(InputVector, FilterVector);
                __m256i Multiply0 = _mm256_unpacklo_epi16(MultiplyLowWords, MultiplyHighWords);
                __m256i Multiply1 = _mm256_unpackhi_epi16(MultiplyLowWords, MultiplyHighWords);

                Accumulator0 = _mm256_add_epi32(Accumulator0, Multiply0);
                Accumulator1 = _mm256_add_epi32(Accumulator1, Multiply1);
                ChannelKernelOffset += Channels;
            }

            // N.B. The above interleaving of the intermediate results leaves
            // the accumulators in a swizzled layout, because the interleaving
            // is per 128-bit half of the __m256i register. Reorder the results
            // now to get the expected sequential order.
            __m256i Reorder0 = _mm256_permute2x128_si256(Accumulator0, Accumulator1, 0x20);
            __m256i Reorder1 = _mm256_permute2x128_si256(Accumulator0, Accumulator1, 0x31);

            _mm256_storeu_si256((__m256i*)&Output[0], Reorder0);
            _mm256_storeu_si256((__m256i*)&Output[8], Reorder1);
            Output += 16;

            ChannelOffset += 16;
            c -= 16;
        }

        if (c >= 8) {

            __m128i Accumulator0 = _mm_setzero_si128();
            __m128i Accumulator1 = _mm_setzero_si128();
            size_t ChannelKernelOffset = ChannelOffset;

            for (size_t k = 0; k < KernelSize; k++) {

                __m128i InputVector = _mm_loadl_epi64((const __m128i*)&Input[k][ChannelOffset]);
                __m128i FilterVector = _mm_loadl_epi64((const __m128i*)&Filter[ChannelKernelOffset]);

                InputVector = _mm_cvtepu8_epi16(InputVector);

                if (std::is_signed<FilterType>::value) {
                    FilterVector = _mm_cvtepi8_epi16(FilterVector);
                } else {
                    FilterVector = _mm_cvtepu8_epi16(FilterVector);
                }

                InputVector = _mm_sub_epi16(InputVector, _mm256_castsi256_si128(InputZeroPointVector));
                FilterVector = _mm_sub_epi16(FilterVector, _mm256_castsi256_si128(FilterZeroPointVector));

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
MlasConvDepthwiseKernelAvx2<int8_t>(
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
MlasConvDepthwiseKernelAvx2<uint8_t>(
    const uint8_t* const* Input,
    uint8_t InputZeroPoint,
    const uint8_t* Filter,
    uint8_t FilterZeroPoint,
    int32_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    );
