/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qlgavgpool.cpp

Abstract:

    This module implements routines for quantized linear global average pool.

--*/

#include "mlasi.h"

size_t
MLASCALL
MlasQLinearSafePaddingElementCount(
    size_t ElementSize,
    size_t ElementCount
    )
{
    if (!(ElementSize == 1 || ElementSize == 2 || ElementSize == 4 || ElementSize == 8 ||
          ElementSize == 16)) {
        MLAS_THROW_EX(std::invalid_argument,
                      "ElementSize must be power of 2 and less or equal than 16!");
    }
    return ElementCount + (size_t{256} / ElementSize - 1);
}

MLAS_FORCEINLINE
float
CheckQLinearGlobalAveragePoolScaleAndSize(
    float ScaleInput,
    float ScaleOutput,
    size_t ImageSize
    )
{
    if (ImageSize >= 0x1000000) {
        MLAS_THROW_EX(std::invalid_argument, "QLinearGlobalAveragePool ImageSize too large!");
    }

    float scale = ScaleInput / (ScaleOutput * static_cast<float>(ImageSize));
    if (scale < 0x1.0p-32f || scale >= 256.0f) {
        // In first case, the scale is too small, ScaleInput/ScaleOutput < 1/256 no matter what
        // ImageSize In second case, the scale is too large, ScaleInput/ScaleOutput >= 256 no matter
        // what Image Size both case make output value constant, and hence not meaningful.
        MLAS_THROW_EX(std::invalid_argument,
                      "QLinearGlobalAveragePool parameter out of computation range!");
    }
    return scale;
}

#if defined(MLAS_NEON_INTRINSICS)

template <typename T8Bits>
void
MLASCALL
MlasQLinearGlobalAveragePoolNchw(
    const T8Bits* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    T8Bits* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize,
    int32_t* AccumulateBuffer
    )
{
    float scale = CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    int32_t bias[] = {-ZeroPointInput * static_cast<int32_t>(ImageSize), 0, 0, 0};
    const int32x4_t vbias = vld1q_s32(bias);
    const int32x4_t vzero = vmovq_n_s32(0);
    const uint8_t* InputU8 = (const uint8_t*)(Input);

    int32_t* sum_buffer = AccumulateBuffer;
    uint8_t tail_buffer[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t c = Channels; c > 0; c--) {

        int32x4_t vacc_lo = vbias;
        int32x4_t vacc_hi = vzero;
        auto Len = ImageSize;
        for (; Len >= 32; Len -= 32) {

            const uint8x8_t vi0 = vld1_u8(InputU8);
            const uint8x8_t vi1 = vld1_u8(InputU8 + 8);
            const uint8x8_t vi2 = vld1_u8(InputU8 + 16);
            const uint8x8_t vi3 = vld1_u8(InputU8 + 24);

            int16x8_t vsum;
            if constexpr (std::is_signed<T8Bits>::value) {

                const int16x8_t vs01 = vaddl_s8(vreinterpret_s8_u8(vi0), vreinterpret_s8_u8(vi1));
                const int16x8_t vs23 = vaddl_s8(vreinterpret_s8_u8(vi2), vreinterpret_s8_u8(vi3));
                vsum = vaddq_s16(vs01, vs23);
            } else {

                const uint16x8_t vs01 = vaddl_u8(vi0, vi1);
                const uint16x8_t vs23 = vaddl_u8(vi2, vi3);
                vsum = vreinterpretq_s16_u16(vaddq_u16(vs01, vs23));
            }

            vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));
            vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum));
            InputU8 += 32;
        }
        for (; Len >= 8; Len -= 8) {

            int16x8_t vsum;
            if constexpr (std::is_signed<T8Bits>::value) {
                vsum = vmovl_s8(vreinterpret_s8_u8(vld1_u8(InputU8)));
            } else {
                vsum = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(InputU8)));
            }
            vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));
            vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum));
            InputU8 += 8;
        }

        if (Len > 0) {

            memcpy(tail_buffer, InputU8, Len);
            int16x8_t vsum;
            if constexpr (std::is_signed<T8Bits>::value) {
                vsum = vmovl_s8(vreinterpret_s8_u8(vld1_u8(tail_buffer)));
            } else {
                vsum = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(tail_buffer)));
            }

            vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));
            vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum));
            InputU8 += Len;
        }

        vacc_lo = vaddq_s32(vacc_lo, vacc_hi);
        int32x2_t vacc = vadd_s32(vget_high_s32(vacc_lo), vget_low_s32(vacc_lo));
        *sum_buffer++ = vget_lane_s32(vpadd_s32(vacc, vacc), 0);
    }

    MlasRequantizeOutput(AccumulateBuffer, Channels, Output, Channels, nullptr, &scale, false,
                         static_cast<T8Bits>(ZeroPointOutput), 0, 0, 1, Channels);
}

template <typename T8Bits>
MLAS_FORCEINLINE
void
MlasQLinearGlobalAveragePoolNhwcSingleBatch(
    const T8Bits* Input,
    T8Bits* Output,
    const T8Bits* LastOf8,
    size_t ImageSize,
    size_t Channels,
    size_t Stride,
    int32_t Bias,
    float Scale,
    T8Bits Output_zero_point,
    int32_t* AccumulateBuffer,
    const T8Bits* ZeroBuffer
    )
{
#define LOAD_FULL_CHANNELS()           \
    const uint8x8_t vi0 = vld1_u8(i0); \
    i0 += 8;                           \
    const uint8x8_t vi1 = vld1_u8(i1); \
    i1 += 8;                           \
    const uint8x8_t vi2 = vld1_u8(i2); \
    i2 += 8;                           \
    const uint8x8_t vi3 = vld1_u8(i3); \
    i3 += 8;                           \
    const uint8x8_t vi4 = vld1_u8(i4); \
    i4 += 8;                           \
    const uint8x8_t vi5 = vld1_u8(i5); \
    i5 += 8;                           \
    const uint8x8_t vi6 = vld1_u8(i6); \
    i6 += 8

#define CALCULATE_ACCUMULATE_VECTORS()                                                         \
    int32x4_t vacc_lo = finish_one_pass ? vld1q_s32(acc) : vbias;                              \
    int32x4_t vacc_hi = finish_one_pass ? vld1q_s32(acc + 4) : vbias;                          \
    int16x8_t vsum;                                                                            \
    if constexpr (std::is_signed<T8Bits>::value) {                                             \
        const int16x8_t vsum01 = vaddl_s8(vreinterpret_s8_u8(vi0), vreinterpret_s8_u8(vi1));   \
        const int16x8_t vsum23 = vaddl_s8(vreinterpret_s8_u8(vi2), vreinterpret_s8_u8(vi3));   \
        const int16x8_t vsum45 = vaddl_s8(vreinterpret_s8_u8(vi4), vreinterpret_s8_u8(vi5));   \
        const int16x8_t vsum016 = vaddw_s8(vsum01, vreinterpret_s8_u8(vi6));                   \
        const int16x8_t vsum2345 = vaddq_s16(vsum23, vsum45);                                  \
        vsum = vaddq_s16(vsum016, vsum2345);                                                   \
    } else {                                                                                   \
        const uint16x8_t vsum01 = vaddl_u8(vi0, vi1);                                          \
        const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);                                          \
        const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);                                          \
        const uint16x8_t vsum016 = vaddw_u8(vsum01, vi6);                                      \
        const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);                                 \
        vsum = vreinterpretq_s16_u16(vaddq_u16(vsum016, vsum2345));                            \
    }                                                                                          \
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));                                          \
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum))

    uint8_t tail[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int32x4_t vbias = vld1q_dup_s32(&Bias);
    bool finish_one_pass = false;
    const size_t step_next_group = 7 * Stride - (Channels & ~size_t{7});

    const uint8_t* LastOf8U8 = (const uint8_t*)LastOf8;
    const uint8_t* i0 = (const uint8_t*)Input;
    const uint8_t* i1 = i0 + Stride;
    const uint8_t* i4 = i0 + Stride * 4;
    const uint8_t* i2 = i1 + Stride;
    const uint8_t* i5 = i4 + Stride;
    const uint8_t* i3 = i2 + Stride;
    const uint8_t* i6 = i5 + Stride;

    for (; ImageSize > 7; ImageSize -= 7) {

        int32_t* acc = AccumulateBuffer;
        size_t c = Channels;
        for (; c >= 8; c -= 8) {

            LOAD_FULL_CHANNELS();

            CALCULATE_ACCUMULATE_VECTORS();

            vst1q_s32(acc, vacc_lo);
            vst1q_s32(acc + 4, vacc_hi);
            acc += 8;
        }
        if (c > 0) {

            const uint8x8_t vi0 = vld1_u8(((i0 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i0, c) : i0));
            const uint8x8_t vi1 = vld1_u8(((i1 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i1, c) : i1));
            const uint8x8_t vi2 = vld1_u8(((i2 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i2, c) : i2));
            const uint8x8_t vi3 = vld1_u8(((i3 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i3, c) : i3));
            const uint8x8_t vi4 = vld1_u8(((i4 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i4, c) : i4));
            const uint8x8_t vi5 = vld1_u8(((i5 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i5, c) : i5));
            const uint8x8_t vi6 = vld1_u8(((i6 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i6, c) : i6));

            CALCULATE_ACCUMULATE_VECTORS();

            vst1q_s32(acc, vacc_lo);
            vst1q_s32(acc + 4, vacc_hi);
        }
        finish_one_pass = true;

        i0 += step_next_group;
        i1 += step_next_group;
        i2 += step_next_group;
        i3 += step_next_group;
        i4 += step_next_group;
        i5 += step_next_group;
        i6 += step_next_group;
    }

    if (ImageSize > 0) {

        switch (ImageSize) {
            case 1:
                i1 = (const uint8_t*)ZeroBuffer; /* fall through */
            case 2:
                i2 = (const uint8_t*)ZeroBuffer; /* fall through */
            case 3:
                i3 = (const uint8_t*)ZeroBuffer; /* fall through */
            case 4:
                i4 = (const uint8_t*)ZeroBuffer; /* fall through */
            case 5:
                i5 = (const uint8_t*)ZeroBuffer; /* fall through */
            case 6:
                i6 = (const uint8_t*)ZeroBuffer; /* fall through */
            default:
                break;
        }

        int32_t* acc = AccumulateBuffer;
        size_t c = Channels;
        for (; c >= 8; c -= 8) {

            LOAD_FULL_CHANNELS();

            CALCULATE_ACCUMULATE_VECTORS();

            vst1q_s32(acc, vacc_lo);
            vst1q_s32(acc + 4, vacc_hi);
            acc += 8;
        }

        if (c > 0) {

            const uint8x8_t vi0 =
                vld1_u8(((i0 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i0, c) : i0));
            const uint8x8_t vi1 = vld1_u8(
                ((1 < ImageSize && i1 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i1, c) : i1));
            const uint8x8_t vi2 = vld1_u8(
                ((2 < ImageSize && i2 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i2, c) : i2));
            const uint8x8_t vi3 = vld1_u8(
                ((3 < ImageSize && i3 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i3, c) : i3));
            const uint8x8_t vi4 = vld1_u8(
                ((4 < ImageSize && i4 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i4, c) : i4));
            const uint8x8_t vi5 = vld1_u8(
                ((5 < ImageSize && i5 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i5, c) : i5));
            const uint8x8_t vi6 = vld1_u8(
                ((6 < ImageSize && i6 >= LastOf8U8) ? (const uint8_t*)memcpy(tail, i6, c) : i6));

            CALCULATE_ACCUMULATE_VECTORS();

            vst1q_s32(acc, vacc_lo);
            vst1q_s32(acc + 4, vacc_hi);
        }
    }
    MlasRequantizeOutput(AccumulateBuffer, Channels, Output, Channels, nullptr, &Scale, false,
                         Output_zero_point, 0, 0, 1, Channels);
}

#elif defined(MLAS_SSE2_INTRINSICS)

template <typename T8Bits>
void MLASCALL
MlasQLinearGlobalAveragePoolNchw(
    const T8Bits* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    T8Bits* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize,
    int32_t* AccumulateBuffer
    )
{
    float scale = CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    const int32_t bias[] = {-ZeroPointInput * static_cast<int32_t>(ImageSize), 0, 0, 0};
    const auto vbias = _mm_loadu_si128((const __m128i*)&bias);
    const auto vzero = _mm_setzero_si128();
    uint8_t buffer[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    int32_t* sum_buffer = AccumulateBuffer;
    for (size_t c = Channels; c > 0; c--) {

        __m128i vacc_lo = vbias;
        __m128i vacc_hi = vzero;
        auto Len = ImageSize;
        for (; Len >= 32; Len -= 32) {

            const __m128i vi0 = _mm_loadl_epi64((const __m128i*)Input);
            const __m128i vi1 = _mm_loadl_epi64((const __m128i*)(Input + 8));
            const __m128i vi2 = _mm_loadl_epi64((const __m128i*)(Input + 16));
            const __m128i vi3 = _mm_loadl_epi64((const __m128i*)(Input + 24));

            if constexpr (std::is_signed<T8Bits>::value) {

                const __m128i vxi0 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi0), 8);
                const __m128i vxi1 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi1), 8);
                const __m128i vxi2 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi2), 8);
                const __m128i vxi3 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi3), 8);
                const __m128i vsum = _mm_add_epi16(_mm_add_epi16(vxi0, vxi1),
                                                   _mm_add_epi16(vxi2, vxi3));
                vacc_lo = _mm_add_epi32(vacc_lo, _mm_srai_epi32(_mm_unpacklo_epi16(vzero, vsum), 16));
                vacc_hi = _mm_add_epi32(vacc_hi, _mm_srai_epi32(_mm_unpackhi_epi16(vzero, vsum), 16));
            } else {

                const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
                const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
                const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
                const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
                const __m128i vsum = _mm_add_epi16(_mm_add_epi16(vxi0, vxi1),
                                                   _mm_add_epi16(vxi2, vxi3));
                vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
                vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));
            }

            Input += 32;
        }
        for (; Len >= 8; Len -= 8) {

            if constexpr (std::is_signed<T8Bits>::value) {

                const __m128i vsum = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, _mm_loadl_epi64((const __m128i*)Input)), 8);
                vacc_lo = _mm_add_epi32(vacc_lo, _mm_srai_epi32(_mm_unpacklo_epi16(vzero, vsum), 16));
                vacc_hi = _mm_add_epi32(vacc_hi, _mm_srai_epi32(_mm_unpackhi_epi16(vzero, vsum), 16));
            } else {

                const __m128i vsum = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)Input), vzero);
                vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
                vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));
            }

            Input += 8;
        }
        if (Len > 0) {

            memcpy(buffer, Input, Len);

            if constexpr (std::is_signed<T8Bits>::value) {

                const __m128i vsum = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, _mm_loadl_epi64((const __m128i*)buffer)), 8);
                vacc_lo = _mm_add_epi32(vacc_lo, _mm_srai_epi32(_mm_unpacklo_epi16(vzero, vsum), 16));
                vacc_hi = _mm_add_epi32(vacc_hi, _mm_srai_epi32(_mm_unpackhi_epi16(vzero, vsum), 16));
            } else {

                const __m128i vsum = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)buffer), vzero);
                vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
                vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));
            }

            Input += Len;
        }

        __m128i vacc = _mm_add_epi32(vacc_lo, vacc_hi);                    // [ D C | B A ]
        __m128i vshuf = _mm_shuffle_epi32(vacc, _MM_SHUFFLE(2, 3, 0, 1));  // [ C D | A B ]
        __m128i vsums = _mm_add_epi32(vacc, vshuf);                        // [ D+C C+D | B+A A+B ]
        vshuf = _mm_shuffle_epi32(vsums, _MM_SHUFFLE(1, 0, 3, 2));         // [ B+A A+B | D+C C+D ]
        vsums = _mm_add_epi32(vsums, vshuf);
        *sum_buffer++ = _mm_cvtsi128_si32(vsums);
    }

    MlasRequantizeOutput(AccumulateBuffer, Channels, Output, Channels, nullptr, &scale, false,
                         static_cast<T8Bits>(ZeroPointOutput), 0, 0, 1, Channels);
}

template <typename T8Bits>
MLAS_FORCEINLINE
void
MlasQLinearGlobalAveragePoolNhwcSingleBatch(
    const T8Bits* Input,
    T8Bits* Output,
    const T8Bits* LastOf8,
    size_t ImageSize,
    size_t Channels,
    size_t Stride,
    int32_t Bias,
    float Scale,
    T8Bits Output_zero_point,
    int32_t* AccumulateBuffer,
    const T8Bits* ZeroBuffer
    )
{
#if defined(MLAS_TARGET_IX86)

    constexpr size_t PixelsPerIteration = 4;

#define LOAD_FULL_CHANNELS()                                 \
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0); \
    i0 += 8;                                                 \
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1); \
    i1 += 8;                                                 \
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2); \
    i2 += 8;                                                 \
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3); \
    i3 += 8;

#define CALCULATE_ACCUMULATE_VECTORS()                                                         \
    __m128i vacc_lo = finish_one_pass ? _mm_loadu_si128((__m128i*)acc) : vbias;                \
    __m128i vacc_hi = finish_one_pass ? _mm_loadu_si128(((__m128i*)acc) + 1) : vbias;          \
    __m128i vxi0;                                                                              \
    __m128i vxi1;                                                                              \
    __m128i vxi2;                                                                              \
    __m128i vxi3;                                                                              \
    if constexpr (std::is_signed<T8Bits>::value) {                                             \
        vxi0 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi0), 8);                               \
        vxi1 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi1), 8);                               \
        vxi2 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi2), 8);                               \
        vxi3 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi3), 8);                               \
    } else {                                                                                   \
        vxi0 = _mm_unpacklo_epi8(vi0, vzero);                                                  \
        vxi1 = _mm_unpacklo_epi8(vi1, vzero);                                                  \
        vxi2 = _mm_unpacklo_epi8(vi2, vzero);                                                  \
        vxi3 = _mm_unpacklo_epi8(vi3, vzero);                                                  \
    }                                                                                          \
    __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);                                                \
    __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);                                                \
    __m128i vsum = _mm_add_epi16(vsum01, vsum23);                                              \
                                                                                               \
    if constexpr (std::is_signed<T8Bits>::value) {                                             \
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_srai_epi32(_mm_unpacklo_epi16(vzero, vsum), 16)); \
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_srai_epi32(_mm_unpackhi_epi16(vzero, vsum), 16)); \
    } else {                                                                                   \
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));                     \
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));                     \
    }

#else

    constexpr size_t PixelsPerIteration = 7;
#define LOAD_FULL_CHANNELS()                                 \
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*)i0); \
    i0 += 8;                                                 \
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*)i1); \
    i1 += 8;                                                 \
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*)i2); \
    i2 += 8;                                                 \
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*)i3); \
    i3 += 8;                                                 \
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*)i4); \
    i4 += 8;                                                 \
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*)i5); \
    i5 += 8;                                                 \
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*)i6); \
    i6 += 8

#define CALCULATE_ACCUMULATE_VECTORS()                                                         \
    __m128i vacc_lo = finish_one_pass ? _mm_loadu_si128((__m128i*)acc) : vbias;                \
    __m128i vacc_hi = finish_one_pass ? _mm_loadu_si128(((__m128i*)acc) + 1) : vbias;          \
    __m128i vxi0;                                                                              \
    __m128i vxi1;                                                                              \
    __m128i vxi2;                                                                              \
    __m128i vxi3;                                                                              \
    __m128i vxi4;                                                                              \
    __m128i vxi5;                                                                              \
    __m128i vxi6;                                                                              \
    if constexpr (std::is_signed<T8Bits>::value) {                                             \
        vxi0 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi0), 8);                               \
        vxi1 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi1), 8);                               \
        vxi2 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi2), 8);                               \
        vxi3 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi3), 8);                               \
        vxi4 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi4), 8);                               \
        vxi5 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi5), 8);                               \
        vxi6 = _mm_srai_epi16(_mm_unpacklo_epi8(vzero, vi6), 8);                               \
    } else {                                                                                   \
        vxi0 = _mm_unpacklo_epi8(vi0, vzero);                                                  \
        vxi1 = _mm_unpacklo_epi8(vi1, vzero);                                                  \
        vxi2 = _mm_unpacklo_epi8(vi2, vzero);                                                  \
        vxi3 = _mm_unpacklo_epi8(vi3, vzero);                                                  \
        vxi4 = _mm_unpacklo_epi8(vi4, vzero);                                                  \
        vxi5 = _mm_unpacklo_epi8(vi5, vzero);                                                  \
        vxi6 = _mm_unpacklo_epi8(vi6, vzero);                                                  \
    }                                                                                          \
    const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);                                          \
    const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);                                          \
    const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);                                          \
    const __m128i vsum016 = _mm_add_epi16(vsum01, vxi6);                                       \
    const __m128i vsum2345 = _mm_add_epi16(vsum23, vsum45);                                    \
    const __m128i vsum = _mm_add_epi16(vsum016, vsum2345);                                     \
    if constexpr (std::is_signed<T8Bits>::value) {                                             \
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_srai_epi32(_mm_unpacklo_epi16(vzero, vsum), 16)); \
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_srai_epi32(_mm_unpackhi_epi16(vzero, vsum), 16)); \
    } else {                                                                                   \
        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));                     \
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));                     \
    }

#endif

    T8Bits tail[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool finish_one_pass = false;
    const __m128i vbias = _mm_set1_epi32(Bias);
    const __m128i vzero = _mm_setzero_si128();
    size_t step_next_group = PixelsPerIteration * Stride - (Channels & ~size_t{7});

    const T8Bits* i0 = Input;
    const T8Bits* i1 = i0 + Stride;
    const T8Bits* i2 = i1 + Stride;
    const T8Bits* i3 = i2 + Stride;
#if !defined(MLAS_TARGET_IX86)
    const T8Bits* i4 = i0 + Stride * 4;
    const T8Bits* i5 = i4 + Stride;
    const T8Bits* i6 = i5 + Stride;
#endif

    for (; ImageSize > PixelsPerIteration; ImageSize -= PixelsPerIteration) {

        int32_t* acc = AccumulateBuffer;
        size_t c = Channels;
        for (; c >= 8; c -= 8) {

            LOAD_FULL_CHANNELS();

            CALCULATE_ACCUMULATE_VECTORS();

            _mm_storeu_si128((__m128i*)acc, vacc_lo);
            _mm_storeu_si128(((__m128i*)acc) + 1, vacc_hi);
            acc += 8;
        }
        if (c > 0) {
            const __m128i vi0 =
                _mm_loadl_epi64((const __m128i*)(i0 >= LastOf8 ? memcpy(tail, i0, c) : i0));
            const __m128i vi1 =
                _mm_loadl_epi64((const __m128i*)(i1 >= LastOf8 ? memcpy(tail, i1, c) : i1));
            const __m128i vi2 =
                _mm_loadl_epi64((const __m128i*)(i2 >= LastOf8 ? memcpy(tail, i2, c) : i2));
            const __m128i vi3 =
                _mm_loadl_epi64((const __m128i*)(i3 >= LastOf8 ? memcpy(tail, i3, c) : i3));
#if !defined(MLAS_TARGET_IX86)
            const __m128i vi4 =
                _mm_loadl_epi64((const __m128i*)(i4 >= LastOf8 ? memcpy(tail, i4, c) : i4));
            const __m128i vi5 =
                _mm_loadl_epi64((const __m128i*)(i5 >= LastOf8 ? memcpy(tail, i5, c) : i5));
            const __m128i vi6 =
                _mm_loadl_epi64((const __m128i*)(i6 >= LastOf8 ? memcpy(tail, i6, c) : i6));
#endif

            CALCULATE_ACCUMULATE_VECTORS();

            _mm_storeu_si128((__m128i*)acc, vacc_lo);
            _mm_storeu_si128(((__m128i*)acc) + 1, vacc_hi);
        }
        finish_one_pass = true;

        i0 += step_next_group;
        i1 += step_next_group;
        i2 += step_next_group;
        i3 += step_next_group;
#if !defined(MLAS_TARGET_IX86)
        i4 += step_next_group;
        i5 += step_next_group;
        i6 += step_next_group;
#endif
    }

    if (ImageSize > 0) {
#if defined(MLAS_TARGET_IX86)
        switch (ImageSize) {
            case 1:
                i1 = ZeroBuffer;
                [[fallthrough]];
            case 2:
                i2 = ZeroBuffer;
                [[fallthrough]];
            case 3:
                i3 = ZeroBuffer;
                [[fallthrough]];
            default:
                break;
        }
#else
        switch (ImageSize) {
            case 1:
                i1 = ZeroBuffer;
                [[fallthrough]];
            case 2:
                i2 = ZeroBuffer;
                [[fallthrough]];
            case 3:
                i3 = ZeroBuffer;
                [[fallthrough]];
            case 4:
                i4 = ZeroBuffer;
                [[fallthrough]];
            case 5:
                i5 = ZeroBuffer;
                [[fallthrough]];
            case 6:
                i6 = ZeroBuffer;
                [[fallthrough]];
            default:
                break;
        }
#endif

        int32_t* acc = AccumulateBuffer;
        size_t c = Channels;
        for (; c >= 8; c -= 8) {

            LOAD_FULL_CHANNELS();

            CALCULATE_ACCUMULATE_VECTORS();

            _mm_storeu_si128((__m128i*)acc, vacc_lo);
            _mm_storeu_si128(((__m128i*)acc) + 1, vacc_hi);
            acc += 8;
        }

        if (c > 0) {
            const __m128i vi0 =
                _mm_loadl_epi64((const __m128i*)(i0 >= LastOf8 ? memcpy(tail, i0, c) : i0));
            const __m128i vi1 = _mm_loadl_epi64(
                (const __m128i*)(1 < ImageSize && i1 >= LastOf8 ? memcpy(tail, i1, c) : i1));
            const __m128i vi2 = _mm_loadl_epi64(
                (const __m128i*)(2 < ImageSize && i2 >= LastOf8 ? memcpy(tail, i2, c) : i2));
            const __m128i vi3 = _mm_loadl_epi64(
                (const __m128i*)(3 < ImageSize && i3 >= LastOf8 ? memcpy(tail, i3, c) : i3));
#if !defined(MLAS_TARGET_IX86)
            const __m128i vi4 = _mm_loadl_epi64(
                (const __m128i*)(4 < ImageSize && i4 >= LastOf8 ? memcpy(tail, i4, c) : i4));
            const __m128i vi5 = _mm_loadl_epi64(
                (const __m128i*)(5 < ImageSize && i5 >= LastOf8 ? memcpy(tail, i5, c) : i5));
            const __m128i vi6 = _mm_loadl_epi64(
                (const __m128i*)(6 < ImageSize && i6 >= LastOf8 ? memcpy(tail, i6, c) : i6));
#endif

            CALCULATE_ACCUMULATE_VECTORS();

            _mm_storeu_si128((__m128i*)acc, vacc_lo);
            _mm_storeu_si128(((__m128i*)acc) + 1, vacc_hi);
        }
    }
    MlasRequantizeOutput(AccumulateBuffer, Channels, Output, Channels, nullptr, &Scale, false,
                         Output_zero_point, 0, 0, 1, Channels);
}

#elif defined(MLAS_LSX_INTRINSICS)

template <typename T8Bits>
void MLASCALL
MlasQLinearGlobalAveragePoolNchw(
    const T8Bits* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    T8Bits* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize,
    int32_t* AccumulateBuffer
    )
{
    float scale = CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    const int32_t bias[] = {-ZeroPointInput * static_cast<int32_t>(ImageSize), 0, 0, 0};
    const auto vbias = __lsx_vld((const __m128i*)&bias, 0);
    const auto vzero = __lsx_vldi(0);
    uint8_t buffer[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    int32_t* sum_buffer = AccumulateBuffer;
    for (size_t c = Channels; c > 0; c--) {

        __m128i vacc_lo = vbias;
        __m128i vacc_hi = vzero;
        auto Len = ImageSize;
        for (; Len >= 32; Len -= 32) {

            const __m128i vi0 = __lsx_vld((const __m128i*)Input, 0);
            __lsx_vinsgr2vr_d(vi0, 0, 1);
            const __m128i vi1 = __lsx_vld((const __m128i*)(Input + 8), 0);
            __lsx_vinsgr2vr_d(vi1, 0, 1);
            const __m128i vi2 = __lsx_vld((const __m128i*)(Input + 16), 0);
            __lsx_vinsgr2vr_d(vi2, 0, 1);
            const __m128i vi3 = __lsx_vld((const __m128i*)(Input + 24), 0);
            __lsx_vinsgr2vr_d(vi3, 0, 1);

            if constexpr (std::is_signed<T8Bits>::value) {

                const __m128i vxi0 = __lsx_vsrai_h(__lsx_vilvl_b(vi0, vzero), 8);
                const __m128i vxi1 = __lsx_vsrai_h(__lsx_vilvl_b(vi1, vzero), 8);
                const __m128i vxi2 = __lsx_vsrai_h(__lsx_vilvl_b(vi2, vzero), 8);
                const __m128i vxi3 = __lsx_vsrai_h(__lsx_vilvl_b(vi3, vzero), 8);
                const __m128i vsum = __lsx_vadd_h(__lsx_vadd_h(vxi0, vxi1),
                                                   __lsx_vadd_h(vxi2, vxi3));
                vacc_lo = __lsx_vadd_w(vacc_lo, __lsx_vsrai_w(__lsx_vilvl_h(vsum, vzero), 16));
                vacc_hi = __lsx_vadd_w(vacc_hi, __lsx_vsrai_w(__lsx_vilvh_h(vsum, vzero), 16));
            } else {

                const __m128i vxi0 = __lsx_vilvl_b(vzero, vi0);
                const __m128i vxi1 = __lsx_vilvl_b(vzero, vi1);
                const __m128i vxi2 = __lsx_vilvl_b(vzero, vi2);
                const __m128i vxi3 = __lsx_vilvl_b(vzero, vi3);
                const __m128i vsum = __lsx_vadd_h(__lsx_vadd_h(vxi0, vxi1),
                                                   __lsx_vadd_h(vxi2, vxi3));
                vacc_lo = __lsx_vadd_w(vacc_lo, __lsx_vilvl_h(vzero, vsum));
                vacc_hi = __lsx_vadd_w(vacc_hi, __lsx_vilvh_h(vzero, vsum));
            }

            Input += 32;
        }
        for (; Len >= 8; Len -= 8) {

            if constexpr (std::is_signed<T8Bits>::value) {

                const __m128i vsum = __lsx_vsrai_h(__lsx_vilvl_b(__lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)Input, 0), 0, 1), vzero), 8);
                vacc_lo = __lsx_vadd_w(vacc_lo, __lsx_vsrai_w(__lsx_vilvl_h(vsum, vzero), 16));
                vacc_hi = __lsx_vadd_w(vacc_hi, __lsx_vsrai_w(__lsx_vilvh_h(vsum, vzero), 16));
            } else {

                const __m128i vsum = __lsx_vilvl_b(vzero, __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)Input, 0), 0, 1));
                vacc_lo = __lsx_vadd_w(vacc_lo, __lsx_vilvl_h(vzero, vsum));
                vacc_hi = __lsx_vadd_w(vacc_hi, __lsx_vilvh_h(vzero, vsum));
            }

            Input += 8;
        }
        if (Len > 0) {

            memcpy(buffer, Input, Len);

            if constexpr (std::is_signed<T8Bits>::value) {

                const __m128i vsum = __lsx_vsrai_h(__lsx_vilvl_b(__lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)buffer, 0), 0, 1), vzero), 8);
                vacc_lo = __lsx_vadd_w(vacc_lo, __lsx_vsrai_w(__lsx_vilvl_h(vsum, vzero), 16));
                vacc_hi = __lsx_vadd_w(vacc_hi, __lsx_vsrai_w(__lsx_vilvh_h(vsum, vzero), 16));
            } else {

                const __m128i vsum = __lsx_vilvl_b(vzero, __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)buffer, 0), 0, 1));
                vacc_lo = __lsx_vadd_w(vacc_lo, __lsx_vilvl_h(vzero, vsum));
                vacc_hi = __lsx_vadd_w(vacc_hi, __lsx_vilvh_h(vzero, vsum));
            }

            Input += Len;
        }

        __m128i vacc = __lsx_vadd_w(vacc_lo, vacc_hi);                    // [ D C | B A ]
        __m128i vshuf = __lsx_vshuf4i_w(vacc, 0xb1);  // [ C D | A B ] _MM_SHUFFLE(2, 3, 0, 1)
        __m128i vsums = __lsx_vadd_w(vacc, vshuf);                        // [ D+C C+D | B+A A+B ]
        vshuf = __lsx_vshuf4i_w(vsums, 0x4e);         // [ B+A A+B | D+C C+D ] _MM_SHUFFLE(1, 0, 3, 2)
        vsums = __lsx_vadd_w(vsums, vshuf);
        __lsx_vstelm_w(vsums, sum_buffer++, 0 , 0);
    }

    MlasRequantizeOutput(AccumulateBuffer, Channels, Output, Channels, nullptr, &scale, false,
                         static_cast<T8Bits>(ZeroPointOutput), 0, 0, 1, Channels);
}

template <typename T8Bits>
MLAS_FORCEINLINE
void
MlasQLinearGlobalAveragePoolNhwcSingleBatch(
    const T8Bits* Input,
    T8Bits* Output,
    const T8Bits* LastOf8,
    size_t ImageSize,
    size_t Channels,
    size_t Stride,
    int32_t Bias,
    float Scale,
    T8Bits Output_zero_point,
    int32_t* AccumulateBuffer,
    const T8Bits* ZeroBuffer
    )
{

    constexpr size_t PixelsPerIteration = 7;
#define LOAD_FULL_CHANNELS()                                 \
    const __m128i vi0 = __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)i0, 0), 0 , 1); \
    i0 += 8;                                                 \
    const __m128i vi1 = __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)i1, 0), 0 , 1); \
    i1 += 8;                                                 \
    const __m128i vi2 = __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)i2, 0), 0 , 1); \
    i2 += 8;                                                 \
    const __m128i vi3 = __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)i3, 0), 0 , 1); \
    i3 += 8;                                                 \
    const __m128i vi4 = __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)i4, 0), 0 , 1); \
    i4 += 8;                                                 \
    const __m128i vi5 = __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)i5, 0), 0 , 1); \
    i5 += 8;                                                 \
    const __m128i vi6 = __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)i6, 0), 0 , 1); \
    i6 += 8

#define CALCULATE_ACCUMULATE_VECTORS()                                                         \
    __m128i vacc_lo = finish_one_pass ? __lsx_vld((__m128i*)acc, 0) : vbias;                \
    __m128i vacc_hi = finish_one_pass ? __lsx_vld(((__m128i*)acc) + 1, 0) : vbias;          \
    __m128i vxi0;                                                                              \
    __m128i vxi1;                                                                              \
    __m128i vxi2;                                                                              \
    __m128i vxi3;                                                                              \
    __m128i vxi4;                                                                              \
    __m128i vxi5;                                                                              \
    __m128i vxi6;                                                                              \
    if constexpr (std::is_signed<T8Bits>::value) {                                             \
        vxi0 = __lsx_vsrai_h(__lsx_vilvl_b(vi0, vzero), 8);                               \
        vxi1 = __lsx_vsrai_h(__lsx_vilvl_b(vi1, vzero), 8);                               \
        vxi2 = __lsx_vsrai_h(__lsx_vilvl_b(vi2, vzero), 8);                               \
        vxi3 = __lsx_vsrai_h(__lsx_vilvl_b(vi3, vzero), 8);                               \
        vxi4 = __lsx_vsrai_h(__lsx_vilvl_b(vi4, vzero), 8);                               \
        vxi5 = __lsx_vsrai_h(__lsx_vilvl_b(vi5, vzero), 8);                               \
        vxi6 = __lsx_vsrai_h(__lsx_vilvl_b(vi6, vzero), 8);                               \
    } else {                                                                                   \
        vxi0 = __lsx_vilvl_b(vzero, vi0);                                                  \
        vxi1 = __lsx_vilvl_b(vzero, vi1);                                                  \
        vxi2 = __lsx_vilvl_b(vzero, vi2);                                                  \
        vxi3 = __lsx_vilvl_b(vzero, vi3);                                                  \
        vxi4 = __lsx_vilvl_b(vzero, vi4);                                                  \
        vxi5 = __lsx_vilvl_b(vzero, vi5);                                                  \
        vxi6 = __lsx_vilvl_b(vzero, vi6);                                                  \
    }                                                                                          \
    const __m128i vsum01 = __lsx_vadd_h(vxi0, vxi1);                                          \
    const __m128i vsum23 = __lsx_vadd_h(vxi2, vxi3);                                          \
    const __m128i vsum45 = __lsx_vadd_h(vxi4, vxi5);                                          \
    const __m128i vsum016 = __lsx_vadd_h(vsum01, vxi6);                                       \
    const __m128i vsum2345 = __lsx_vadd_h(vsum23, vsum45);                                    \
    const __m128i vsum = __lsx_vadd_h(vsum016, vsum2345);                                     \
    if constexpr (std::is_signed<T8Bits>::value) {                                             \
        vacc_lo = __lsx_vadd_w(vacc_lo, __lsx_vsrai_w(__lsx_vilvl_h(vsum, vzero), 16)); \
        vacc_hi = __lsx_vadd_w(vacc_hi, __lsx_vsrai_w(__lsx_vilvh_h(vsum, vzero), 16)); \
    } else {                                                                                   \
        vacc_lo = __lsx_vadd_w(vacc_lo, __lsx_vilvl_h(vzero, vsum));                     \
        vacc_hi = __lsx_vadd_w(vacc_hi, __lsx_vilvh_h(vzero, vsum));                     \
    }


    T8Bits tail[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool finish_one_pass = false;
    const __m128i vbias = __lsx_vreplgr2vr_w(Bias);
    const __m128i vzero = __lsx_vldi(0);
    size_t step_next_group = PixelsPerIteration * Stride - (Channels & ~size_t{7});

    const T8Bits* i0 = Input;
    const T8Bits* i1 = i0 + Stride;
    const T8Bits* i2 = i1 + Stride;
    const T8Bits* i3 = i2 + Stride;
    const T8Bits* i4 = i0 + Stride * 4;
    const T8Bits* i5 = i4 + Stride;
    const T8Bits* i6 = i5 + Stride;

    for (; ImageSize > PixelsPerIteration; ImageSize -= PixelsPerIteration) {

        int32_t* acc = AccumulateBuffer;
        size_t c = Channels;
        for (; c >= 8; c -= 8) {

            LOAD_FULL_CHANNELS();

            CALCULATE_ACCUMULATE_VECTORS();

            __lsx_vst(vacc_lo, (__m128i*)acc, 0);
            __lsx_vst(vacc_hi, ((__m128i*)acc) + 1, 0);
            acc += 8;
        }
        if (c > 0) {
            const __m128i vi0 =
                __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)(i0 >= LastOf8 ? memcpy(tail, i0, c) : i0), 0), 0 ,1);
            const __m128i vi1 =
                __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)(i1 >= LastOf8 ? memcpy(tail, i1, c) : i1), 0), 0 ,1);
            const __m128i vi2 =
                __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)(i2 >= LastOf8 ? memcpy(tail, i2, c) : i2), 0), 0 ,1);
            const __m128i vi3 =
                __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)(i3 >= LastOf8 ? memcpy(tail, i3, c) : i3), 0), 0 ,1);
            const __m128i vi4 =
                __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)(i4 >= LastOf8 ? memcpy(tail, i4, c) : i4), 0), 0 ,1);
            const __m128i vi5 =
                __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)(i5 >= LastOf8 ? memcpy(tail, i5, c) : i5), 0), 0 ,1);
            const __m128i vi6 =
                __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)(i6 >= LastOf8 ? memcpy(tail, i6, c) : i6), 0), 0 ,1);

            CALCULATE_ACCUMULATE_VECTORS();

            __lsx_vst(vacc_lo, (__m128i*)acc, 0);
            __lsx_vst(vacc_hi, ((__m128i*)acc) + 1, 0);
        }
        finish_one_pass = true;

        i0 += step_next_group;
        i1 += step_next_group;
        i2 += step_next_group;
        i3 += step_next_group;
        i4 += step_next_group;
        i5 += step_next_group;
        i6 += step_next_group;
    }

    if (ImageSize > 0) {
        switch (ImageSize) {
            case 1:
                i1 = ZeroBuffer;
                [[fallthrough]];
            case 2:
                i2 = ZeroBuffer;
                [[fallthrough]];
            case 3:
                i3 = ZeroBuffer;
                [[fallthrough]];
            case 4:
                i4 = ZeroBuffer;
                [[fallthrough]];
            case 5:
                i5 = ZeroBuffer;
                [[fallthrough]];
            case 6:
                i6 = ZeroBuffer;
                [[fallthrough]];
            default:
                break;
        }

        int32_t* acc = AccumulateBuffer;
        size_t c = Channels;
        for (; c >= 8; c -= 8) {

            LOAD_FULL_CHANNELS();

            CALCULATE_ACCUMULATE_VECTORS();

            __lsx_vst(vacc_lo, (__m128i*)acc, 0);
            __lsx_vst(vacc_hi, ((__m128i*)acc) + 1, 0);
            acc += 8;
        }

        if (c > 0) {
            const __m128i vi0 =
                __lsx_vinsgr2vr_d(__lsx_vld((const __m128i*)(i0 >= LastOf8 ? memcpy(tail, i0, c) : i0), 0), 0 ,1);
            const __m128i vi1 = __lsx_vinsgr2vr_d(__lsx_vld(
                (const __m128i*)(1 < ImageSize && i1 >= LastOf8 ? memcpy(tail, i1, c) : i1), 0), 0, 1);
            const __m128i vi2 = __lsx_vinsgr2vr_d(__lsx_vld(
                (const __m128i*)(2 < ImageSize && i2 >= LastOf8 ? memcpy(tail, i2, c) : i2), 0), 0, 1);
            const __m128i vi3 = __lsx_vinsgr2vr_d(__lsx_vld(
                (const __m128i*)(3 < ImageSize && i3 >= LastOf8 ? memcpy(tail, i3, c) : i3), 0), 0, 1);
            const __m128i vi4 = __lsx_vinsgr2vr_d(__lsx_vld(
                (const __m128i*)(4 < ImageSize && i4 >= LastOf8 ? memcpy(tail, i4, c) : i4), 0), 0, 1);
            const __m128i vi5 = __lsx_vinsgr2vr_d(__lsx_vld(
                (const __m128i*)(5 < ImageSize && i5 >= LastOf8 ? memcpy(tail, i5, c) : i5), 0), 0, 1);
            const __m128i vi6 = __lsx_vinsgr2vr_d(__lsx_vld(
                (const __m128i*)(6 < ImageSize && i6 >= LastOf8 ? memcpy(tail, i6, c) : i6), 0), 0, 1);

            CALCULATE_ACCUMULATE_VECTORS();

            __lsx_vst(vacc_lo, (__m128i*)acc, 0);
            __lsx_vst(vacc_hi, ((__m128i*)acc) + 1, 0);
        }
    }
    MlasRequantizeOutput(AccumulateBuffer, Channels, Output, Channels, nullptr, &Scale, false,
                         Output_zero_point, 0, 0, 1, Channels);
}

#else

// Pure C++ Implementation

template <typename T8Bits>
void
MLASCALL
MlasQLinearGlobalAveragePoolNchw(
    const T8Bits* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    T8Bits* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize,
    int32_t* /* AccumulateBuffer */
    )
{
    float scale = CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    int32_t bias = -ZeroPointInput * static_cast<int32_t>(ImageSize);
    for (; Channels > 0; Channels--) {

        int32_t acc = bias;
        for (size_t i = 0; i < ImageSize; ++i) {
            acc += static_cast<int32_t>(*Input++);
        }
        int32_t v = static_cast<int32_t>(std::nearbyintf(acc * scale)) + ZeroPointOutput;
        v = std::min(static_cast<int32_t>(std::numeric_limits<T8Bits>::max()), v);
        v = std::max(static_cast<int32_t>(std::numeric_limits<T8Bits>::lowest()), v);
        *Output++ = static_cast<T8Bits>(v);
    }
}

template <typename T8Bits>
void
MLASCALL
MlasQLinearGlobalAveragePoolNhwc(
    const T8Bits* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    T8Bits* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Batch,
    size_t ImageSize,
    size_t Stride,
    size_t Channels,
    int32_t* AccumulateBuffer,
    const T8Bits* /*ZeroBuffer*/
    )
{
    float scale = CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    int32_t bias = -ZeroPointInput * static_cast<int32_t>(ImageSize);
    for (; Batch > 0; Batch--) {

        const T8Bits* batch_input = Input;
        T8Bits* batch_output = Output;
        Input += Stride * ImageSize;
        Output += Stride;
        std::fill_n(AccumulateBuffer, Channels, bias);
        for (size_t i = 0; i < ImageSize; ++i) {

            for (size_t c = 0; c < Channels; ++c) {
                AccumulateBuffer[c] += static_cast<int>(batch_input[c]);
            }

            batch_input += Stride;
        }

        for (size_t c = 0; c < Channels; ++c) {

            int32_t v = static_cast<int32_t>(std::nearbyintf(AccumulateBuffer[c] * scale)) + ZeroPointOutput;
            v = std::min(static_cast<int32_t>(std::numeric_limits<T8Bits>::max()), v);
            v = std::max(static_cast<int32_t>(std::numeric_limits<T8Bits>::lowest()), v);
            *batch_output++ = static_cast<T8Bits>(v);
        }
    }
}

#endif

#if defined(MLAS_NEON_INTRINSICS) || defined(MLAS_SSE2_INTRINSICS) || defined(MLAS_LSX_INTRINSICS)

template <typename T8Bits>
void
MLASCALL
MlasQLinearGlobalAveragePoolNhwc(
    const T8Bits* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    T8Bits* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Batch,
    size_t ImageSize,
    size_t Stride,
    size_t Channels,
    int32_t* AccumulateBuffer,
    const T8Bits* ZeroBuffer
    )
{
    float scale = CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    const int32_t bias = -ZeroPointInput * static_cast<int32_t>(ImageSize);
    const T8Bits* inputLastOf8 = Input + (Batch * ImageSize * Stride - Stride + Channels) - 8;

    for (; Batch > 0; Batch--) {
        MlasQLinearGlobalAveragePoolNhwcSingleBatch(
            Input, Output, inputLastOf8, ImageSize, Channels, Stride, bias, scale,
            static_cast<T8Bits>(ZeroPointOutput), AccumulateBuffer, ZeroBuffer);
        Input += ImageSize * Stride;
        Output += Stride;
    }
}

#endif

template
void
MLASCALL
MlasQLinearGlobalAveragePoolNchw<int8_t>(
    const int8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    int8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize,
    int32_t* AccumulateBuffer
    );

template
void
MLASCALL
MlasQLinearGlobalAveragePoolNchw<uint8_t>(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize,
    int32_t* AccumulateBuffer
    );

template
void
MLASCALL
MlasQLinearGlobalAveragePoolNhwc<int8_t>(
    const int8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    int8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Batch,
    size_t ImageSize,
    size_t Stride,
    size_t Channels,
    int32_t* AccumulateBuffer,
    const int8_t* ZeroBuffer
    );

template
void
MLASCALL
MlasQLinearGlobalAveragePoolNhwc<uint8_t>(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Batch,
    size_t ImageSize,
    size_t Stride,
    size_t Channels,
    int32_t* AccumulateBuffer,
    const uint8_t* ZeroBuffer
    );
