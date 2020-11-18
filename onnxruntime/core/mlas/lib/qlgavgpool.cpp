/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize.cpp

Abstract:

    This module implements routines to quantize buffers.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "quantize.h"

size_t
MLASCALL
MlasQLinearSafePaddingElementCount(
    size_t ElementSize,
    size_t ElementCount
    )
{
    if (!(ElementSize == 1 || ElementSize == 2 || ElementSize == 4 || ElementSize == 8 || ElementSize == 16)) {
#ifdef MLAS_NO_EXCEPTION
        abort();
#else
        throw std::invalid_argument("ElementSize must be power of 2 and less or equal than 16!");
#endif
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
    float scale = ScaleInput / (ScaleOutput * (float)gsl::narrow_cast<int>(ImageSize));
    if (ImageSize >= 0x1000000 || scale < 0x1.0p-32f || scale >= 256.0f) {
#ifdef MLAS_NO_EXCEPTION
        abort();
#else
        throw std::invalid_argument("QLinearGlobalAveragePool parameter out of computation range!");
#endif
    }
    return scale;
}

#if defined(MLAS_NEON_INTRINSICS)

#if defined(MLAS_NEON64_INTRINSICS)

typedef int32x4_t VMultiplierType;
#define MultiplierFromValueAddress(multiplier_address) vld1q_dup_s32(multiplier_address)

#else

typedef int32x2_t VMultiplierType;
#define MultiplierFromValueAddress(multiplier_address) vld1_dup_s32(multiplier_address)

#endif

void
MlasCalculateParametersForGloabalAveragePool(
    size_t ImageSize,
    float ScaleInput,
    int32_t ZeroPointInput,
    float ScaleOutput,
    int32_t& Bias,
    int32_t& Multiplier,
    int64_t& LeftShift
    )
{
    Bias = -ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize);
    float scale = ScaleInput / (ScaleOutput * (float)ImageSize);

    const uint32_t scale_bits = MlasBitsOfFp32(scale);
    Multiplier = (int32_t)(scale_bits & 0x007FFFFF | 0x00800000);

    // Shift is in [16, 55] range.
    int64_t Shift = 127 + 23 - (scale_bits >> 23);
    LeftShift = -(int64_t)Shift;
}

MLAS_FORCEINLINE
uint8x8_t
RequantizeBiasedSum(
    int32x4_t vacc_lo,
    int32x4_t vacc_hi,
    const VMultiplierType vmultiplier,
    const int64x2_t vleft_shift,
    const int16x8_t voutput_zero_point
    )
{
    const int32x4_t vneg_mask_lo = vreinterpretq_s32_u32(vcltq_s32(vacc_lo, vmovq_n_s32(0)));
    const int32x4_t vneg_mask_hi = vreinterpretq_s32_u32(vcltq_s32(vacc_hi, vmovq_n_s32(0)));

#if defined(MLAS_NEON64_INTRINSICS)

    const int64x2_t vproduct01 = vmull_s32(vget_low_s32(vacc_lo), vget_low_s32(vmultiplier));
    const int64x2_t vproduct23 = vmull_high_s32(vacc_lo, vmultiplier);
    const int64x2_t vproduct45 = vmull_s32(vget_low_s32(vacc_hi), vget_low_s32(vmultiplier));
    const int64x2_t vproduct67 = vmull_high_s32(vacc_hi, vmultiplier);

    const int64x2_t vadjusted_product01 = vaddw_s32(vproduct01, vget_low_s32(vneg_mask_lo));
    const int64x2_t vadjusted_product23 = vaddw_high_s32(vproduct23, vneg_mask_lo);
    const int64x2_t vadjusted_product45 = vaddw_s32(vproduct45, vget_low_s32(vneg_mask_hi));
    const int64x2_t vadjusted_product67 = vaddw_high_s32(vproduct67, vneg_mask_hi);

#else

    const int64x2_t vproduct01 = vmull_s32(vget_low_s32(vacc_lo), vmultiplier);
    const int64x2_t vproduct23 = vmull_s32(vget_high_s32(vacc_lo), vmultiplier);
    const int64x2_t vproduct45 = vmull_s32(vget_low_s32(vacc_hi), vmultiplier);
    const int64x2_t vproduct67 = vmull_s32(vget_high_s32(vacc_hi), vmultiplier);

    const int64x2_t vadjusted_product01 = vaddw_s32(vproduct01, vget_low_s32(vneg_mask_lo));
    const int64x2_t vadjusted_product23 = vaddw_s32(vproduct23, vget_high_s32(vneg_mask_lo));
    const int64x2_t vadjusted_product45 = vaddw_s32(vproduct45, vget_low_s32(vneg_mask_hi));
    const int64x2_t vadjusted_product67 = vaddw_s32(vproduct67, vget_high_s32(vneg_mask_hi));

#endif

    const int64x2_t vscaled_acc01 = vrshlq_s64(vadjusted_product01, vleft_shift);
    const int64x2_t vscaled_acc23 = vrshlq_s64(vadjusted_product23, vleft_shift);
    const int64x2_t vscaled_acc45 = vrshlq_s64(vadjusted_product45, vleft_shift);
    const int64x2_t vscaled_acc67 = vrshlq_s64(vadjusted_product67, vleft_shift);

#if defined(MLAS_NEON64_INTRINSICS)

    vacc_lo = vuzp1q_s32(vreinterpretq_s32_s64(vscaled_acc01), vreinterpretq_s32_s64(vscaled_acc23));
    vacc_hi = vuzp1q_s32(vreinterpretq_s32_s64(vscaled_acc45), vreinterpretq_s32_s64(vscaled_acc67));
    const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);

#else

    vacc_lo = vcombine_s32(vmovn_s64(vscaled_acc01), vmovn_s64(vscaled_acc23));
    vacc_hi = vcombine_s32(vmovn_s64(vscaled_acc45), vmovn_s64(vscaled_acc67));
    const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), voutput_zero_point);

#endif

    uint8x8_t vout = vqmovun_s16(vacc);
    return vout;
}

#if !defined(_MSC_VER)

#define vst1_lane_u32_ex(pD, D, lane, align) vst1_lane_u32((uint32_t*)__builtin_assume_aligned(pD, ((align) / 8)), D, lane)
#define vst1_lane_u16_ex(pD, D, lane, align) vst1_lane_u16((uint16_t*)__builtin_assume_aligned(pD, ((align) / 8)), D, lane)

#endif

MLAS_FORCEINLINE
void
StoreTailBytes(
    uint8_t* Output,
    uint8x8_t vresult,
    size_t Len
    )
{
    if (Len & 4) {
        vst1_lane_u32_ex((uint32_t*)Output, vreinterpret_u32_u8(vresult), 0, 8);
        Output += 4;
        vresult = vext_u8(vresult, vresult, 4);
    }
    if (Len & 2) {
        vst1_lane_u16_ex((uint16_t*)Output, vreinterpret_u16_u8(vresult), 0, 8);
        Output += 2;
        vresult = vext_u8(vresult, vresult, 2);
    }
    if (Len & 1) {
        vst1_lane_u8(Output, vresult, 0);
    }
}

void
MLASCALL
MlasQLinearGlobalAveragePool(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize
    )
{
    int32_t Bias, Multiplier;
    int64_t LeftShift;
    CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    MlasCalculateParametersForGloabalAveragePool(
        ImageSize, ScaleInput, ZeroPointInput, ScaleOutput, Bias, Multiplier, LeftShift);

    const auto vbias = vmovq_n_s32(Bias);
    const VMultiplierType vmultiplier = MultiplierFromValueAddress(&Multiplier);
    const int16x8_t voutput_zero_point = vld1q_dup_s16((const int16_t*)&ZeroPointOutput);
    const int64x2_t vleft_shift = vmovq_n_s64(LeftShift);
    const int32x4_t vzero = vmovq_n_s32(0);

    size_t sum_count = 0;
    int32_t sum_buffer[8] = { 0, 0, 0, 0 };
    uint8_t tail_buffer[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (; Channels > 0; Channels--) {
        int32x4_t vacc_lo = vzero;
        int32x4_t vacc_hi = vzero;
        auto Len = ImageSize;
        for (; Len >= 32; Len -= 32) {
            const uint8x8_t vi0 = vld1_u8(Input);
            const uint8x8_t vi1 = vld1_u8(Input + 8);
            const uint8x8_t vi2 = vld1_u8(Input + 16);
            const uint8x8_t vi3 = vld1_u8(Input + 24);

            const uint16x8_t vs01 = vaddl_u8(vi0, vi1);
            const uint16x8_t vs23 = vaddl_u8(vi2, vi3);
            const int16x8_t vsum = vreinterpretq_s16_u16(vaddq_u16(vs01, vs23));
            vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));
            vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum));
            Input += 32;
        }
        for (; Len >= 8; Len -= 8) {
            const int16x8_t vsum = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(Input)));
            vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));
            vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum));
            Input += 8;
        }
        if (Len > 0) {
            MlasCopyTailBytes(tail_buffer, Input, Len);
            const int16x8_t vsum = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(tail_buffer)));
            vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));
            vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum));
            Input += Len;
        }

        vacc_lo = vaddq_s32(vacc_lo, vacc_hi);
        int32x2_t vacc = vadd_s32(vget_high_s32(vacc_lo), vget_low_s32(vacc_lo));
        sum_buffer[sum_count++] = vget_lane_s32(vpadd_s32(vacc, vacc), 0);

        if (sum_count == 8) {
            vacc_lo = vaddq_s32(vbias, vld1q_s32(sum_buffer));
            vacc_hi = vaddq_s32(vbias, vld1q_s32(sum_buffer + 4));
            uint8x8_t vresult = RequantizeBiasedSum(vacc_lo, vacc_hi, vmultiplier, vleft_shift, voutput_zero_point);
            vst1_u8(Output, vresult);
            Output += 8;
            sum_count = 0;
        }
    }
    if (sum_count > 0) {
        const int32x4_t vacc_lo = vaddq_s32(vbias, vld1q_s32(sum_buffer));
        const int32x4_t vacc_hi = vaddq_s32(vbias, vld1q_s32(sum_buffer + 4));
        uint8x8_t vresult = RequantizeBiasedSum(vacc_lo, vacc_hi, vmultiplier, vleft_shift, voutput_zero_point);
        StoreTailBytes(Output, vresult, sum_count);
    }
}

MLAS_FORCEINLINE
void
MLASCALL
MlasNhwcQLinearGlobalAveragePoolSingleBatch(
    const uint8_t* Input,
    uint8_t* Output,
    const uint8_t* LastOf8,
    size_t ImageSize,
    size_t Channels,
    size_t Stride,
    const int32x4_t vbias,
    const VMultiplierType vmultiplier,
    const int16x8_t voutput_zero_point,
    const int64x2_t vleft_shift,
    int32_t* AccumulateBuffer,
    const uint8_t* ZeroBuffer)
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
    i6 += 8;

#define CACULATE_ACCUMULATE_VECTORS()                                                \
    int32x4_t vacc_lo = finish_one_pass ? vld1q_s32(acc) : vbias;                    \
    int32x4_t vacc_hi = finish_one_pass ? vld1q_s32(acc + 4) : vbias;                \
    const uint16x8_t vsum01 = vaddl_u8(vi0, vi1);                                    \
    const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);                                    \
    const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);                                    \
    const uint16x8_t vsum016 = vaddw_u8(vsum01, vi6);                                \
    const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);                           \
    const int16x8_t vsum = vreinterpretq_s16_u16(vaddq_u16(vsum016, vsum2345));      \
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));                                \
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum))

    uint8_t tail[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    bool finish_one_pass = false;
    const size_t step_next_group = 7 * Stride - (Channels & ~size_t{7});

    const uint8_t* i0 = Input;
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

            CACULATE_ACCUMULATE_VECTORS();

            vst1q_s32(acc, vacc_lo);
            vst1q_s32(acc + 4, vacc_hi);
            acc += 8;
        }
        if (c > 0) {
          const uint8x8_t vi0 = vld1_u8(i0 >= LastOf8 ? MlasCopyTailBytes(tail, i0, c) : i0);
          const uint8x8_t vi1 = vld1_u8(i1 >= LastOf8 ? MlasCopyTailBytes(tail, i1, c) : i1);
          const uint8x8_t vi2 = vld1_u8(i2 >= LastOf8 ? MlasCopyTailBytes(tail, i2, c) : i2);
          const uint8x8_t vi3 = vld1_u8(i3 >= LastOf8 ? MlasCopyTailBytes(tail, i3, c) : i3);
          const uint8x8_t vi4 = vld1_u8(i4 >= LastOf8 ? MlasCopyTailBytes(tail, i4, c) : i4);
          const uint8x8_t vi5 = vld1_u8(i5 >= LastOf8 ? MlasCopyTailBytes(tail, i5, c) : i5);
          const uint8x8_t vi6 = vld1_u8(i6 >= LastOf8 ? MlasCopyTailBytes(tail, i6, c) : i6);

          CACULATE_ACCUMULATE_VECTORS();

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
        case 1: i1 = ZeroBuffer; /* fall through */
        case 2: i2 = ZeroBuffer; /* fall through */
        case 3: i3 = ZeroBuffer; /* fall through */
        case 4: i4 = ZeroBuffer; /* fall through */
        case 5: i5 = ZeroBuffer; /* fall through */
        case 6: i6 = ZeroBuffer; /* fall through */
        default: break;
        }

        int32_t* acc = AccumulateBuffer;
        for (; Channels >= 8; Channels -= 8) {
            LOAD_FULL_CHANNELS();

            CACULATE_ACCUMULATE_VECTORS();

            uint8x8_t vresult = RequantizeBiasedSum(vacc_lo, vacc_hi, vmultiplier, vleft_shift, voutput_zero_point);
            vst1_u8(Output, vresult);
            acc += 8;
            Output += 8;
        }

        if (Channels > 0) {
          const uint8x8_t vi0 = vld1_u8(i0 >= LastOf8 ? MlasCopyTailBytes(tail, i0, Channels) : i0);
          const uint8x8_t vi1 = vld1_u8(1 < ImageSize && i1 >= LastOf8 ? MlasCopyTailBytes(tail, i1, Channels) : i1);
          const uint8x8_t vi2 = vld1_u8(2 < ImageSize && i2 >= LastOf8 ? MlasCopyTailBytes(tail, i2, Channels) : i2);
          const uint8x8_t vi3 = vld1_u8(3 < ImageSize && i3 >= LastOf8 ? MlasCopyTailBytes(tail, i3, Channels) : i3);
          const uint8x8_t vi4 = vld1_u8(4 < ImageSize && i4 >= LastOf8 ? MlasCopyTailBytes(tail, i4, Channels) : i4);
          const uint8x8_t vi5 = vld1_u8(5 < ImageSize && i5 >= LastOf8 ? MlasCopyTailBytes(tail, i5, Channels) : i5);
          const uint8x8_t vi6 = vld1_u8(6 < ImageSize && i6 >= LastOf8 ? MlasCopyTailBytes(tail, i6, Channels) : i6);

          CACULATE_ACCUMULATE_VECTORS();

          uint8x8_t vresult = RequantizeBiasedSum(vacc_lo, vacc_hi, vmultiplier, vleft_shift, voutput_zero_point);
          StoreTailBytes(Output, vresult, Channels);
        }
    }
}

void
MLASCALL
MlasNhwcQLinearGlobalAveragePool(
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
    )
{
    if (Stride == 1) {
        MlasQLinearGlobalAveragePool(Input, ScaleInput, ZeroPointInput, 
                                    Output, ScaleOutput, ZeroPointOutput, Batch, ImageSize);
        return;
    }

    int32_t Bias, Multiplier;
    int64_t LeftShift;
    CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    MlasCalculateParametersForGloabalAveragePool(
        ImageSize, ScaleInput, ZeroPointInput, ScaleOutput, Bias, Multiplier, LeftShift);

    const auto vbias = vmovq_n_s32(Bias);
    const VMultiplierType vmultiplier = MultiplierFromValueAddress(&Multiplier);
    const int16x8_t voutput_zero_point = vld1q_dup_s16((const int16_t*)&ZeroPointOutput);
    const int64x2_t vleft_shift = vmovq_n_s64(LeftShift);
    const uint8_t* InputLastOf8 = Input + (Batch * ImageSize * Stride - Stride + Channels) - 8;

    for (; Batch > 0; Batch--) {
        MlasNhwcQLinearGlobalAveragePoolSingleBatch(
            Input, Output, InputLastOf8, ImageSize, Channels, Stride,
            vbias, vmultiplier, voutput_zero_point, vleft_shift,
            AccumulateBuffer, ZeroBuffer);
        Input += ImageSize * Stride;
        Output += Stride;
    }
}

#elif defined(MLAS_SSE2_INTRINSICS)

MLAS_FORCEINLINE
void StoreTailBytes(
    uint8_t* Output,
    __m128i vresult,
    size_t Len
    )
{
    if (Len >= 4) {
        *(int32_t*)Output = _mm_cvtsi128_si32(vresult);
        Output += 4;
        Len -= 4;
        vresult = _mm_shuffle_epi32(vresult, _MM_SHUFFLE(0, 3, 2, 1));
    }
    if (Len > 0) {
        uint32_t tail_values = (uint32_t)_mm_cvtsi128_si32(vresult);
        for (; Len > 0; --Len) {
            *Output++ = (uint8_t)tail_values;
            tail_values = tail_values >> 8;
        }
    }
}

void
MLASCALL
MlasQLinearGlobalAveragePool(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize
    )
{
    float scale = CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    const auto vscale = _mm_set1_ps(scale);
    const auto vbias = _mm_set1_epi32(-ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize));
    const auto vmin_value = _mm_set1_ps(float(0 - ZeroPointOutput));
    const auto vmax_value = _mm_set1_ps(float(255 - ZeroPointOutput));
    const auto vzero_point = _mm_set1_epi32(ZeroPointOutput);
    const auto vzero = _mm_setzero_si128();
    uint8_t buffer[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    int32_t sum_buffer[8];
    int sum_count = 0;
    for (; Channels > 0; Channels--) {
        __m128i vacc_lo = _mm_setzero_si128();
        __m128i vacc_hi = _mm_setzero_si128();
        auto Len = ImageSize;
        for (; Len >= 32; Len -= 32) {
            const __m128i vi0 = _mm_loadl_epi64((const __m128i*)Input);
            const __m128i vi1 = _mm_loadl_epi64((const __m128i*)(Input + 8));
            const __m128i vi2 = _mm_loadl_epi64((const __m128i*)(Input + 16));
            const __m128i vi3 = _mm_loadl_epi64((const __m128i*)(Input + 24));

            const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
            const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
            const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
            const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);

            const __m128i vsum = _mm_add_epi16(_mm_add_epi16(vxi0, vxi1), _mm_add_epi16(vxi2, vxi3));
            vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
            vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));
            Input += 32;
        }
        for (; Len >= 8; Len -= 8) {
            const __m128i vsum = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)Input), vzero);
            vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
            vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));
            Input += 8;
        }
        if (Len > 0) {
            MlasCopyTailBytes(buffer, Input, Len);
            const __m128i vsum = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)buffer), vzero);
            vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
            vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));
            Input += Len;
        }

        __m128i vacc = _mm_add_epi32(vacc_lo, vacc_hi);                    // [ D C | B A ]
        __m128i vshuf = _mm_shuffle_epi32(vacc, _MM_SHUFFLE(2, 3, 0, 1));  // [ C D | A B ]
        __m128i vsums = _mm_add_epi32(vacc, vshuf);                        // [ D+C C+D | B+A A+B ]
        vshuf = _mm_shuffle_epi32(vsums, _MM_SHUFFLE(1, 0, 3, 2));         // [ B+A A+B | D+C C+D ]
        vsums = _mm_add_epi32(vsums, vshuf);
        sum_buffer[sum_count++] = _mm_cvtsi128_si32(vsums);

        if (sum_count == 8) {
            auto vresult0 = _mm_load_si128((const __m128i*)sum_buffer);
            auto vresult1 = _mm_load_si128(((const __m128i*)sum_buffer) + 1);
            vresult0 = MlasRequantizeOutputVector(vresult0, vbias, vscale, vmin_value, vmax_value, vzero_point);
            vresult1 = MlasRequantizeOutputVector(vresult1, vbias, vscale, vmin_value, vmax_value, vzero_point);
            vresult0 = _mm_packus_epi16(vresult0, vresult1);
            vresult0 = _mm_packus_epi16(vresult0, vresult0);
            _mm_storel_epi64((__m128i*)Output, vresult0);
            Output += 8;
            sum_count = 0;
        }
    }
    if (sum_count > 0) {
        auto vresult0 = _mm_load_si128((const __m128i*)sum_buffer);
        auto vresult1 = _mm_load_si128(((const __m128i*)sum_buffer) + 1);
        vresult0 = MlasRequantizeOutputVector(vresult0, vbias, vscale, vmin_value, vmax_value, vzero_point);
        vresult1 = MlasRequantizeOutputVector(vresult1, vbias, vscale, vmin_value, vmax_value, vzero_point);
        vresult0 = _mm_packus_epi16(vresult0, vresult1);
        vresult0 = _mm_packus_epi16(vresult0, vresult0);
        StoreTailBytes(Output, vresult0, sum_count);
    }
}

MLAS_FORCEINLINE
void
MlasNhwcQLinearGlobalAveragePoolSingleBatch(
    const uint8_t* Input,
    uint8_t* Output,
    const uint8_t* LastOf8,
    size_t ImageSize,
    size_t Channels,
    size_t Stride,
    const __m128i vbias,
    const __m128 vscale,
    const __m128 vmin_value,
    const __m128 vmax_value,
    const __m128i vzero_point,
    int32_t* AccumulateBuffer,
    const uint8_t* ZeroBuffer)
{
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

#define CACULATE_ACCUMULATE_VECTORS()                                                                  \
    __m128i vacc_lo = finish_one_pass ? _mm_load_si128((__m128i*)acc) : vbias;                         \
    __m128i vacc_hi = finish_one_pass ? _mm_load_si128(((__m128i*)acc) + 1) : vbias;                   \
    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);                                                \
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);                                                \
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);                                                \
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);                                                \
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);                                                \
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);                                                \
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);                                                \
    const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);                                                  \
    const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);                                                  \
    const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);                                                  \
    const __m128i vsum016 = _mm_add_epi16(vsum01, vxi6);                                               \
    const __m128i vsum2345 = _mm_add_epi16(vsum23, vsum45);                                            \
    const __m128i vsum = _mm_add_epi16(vsum016, vsum2345);                                             \
    vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));                                 \
    vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero))

    uint8_t tail[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    bool finish_one_pass = false;
    const __m128i vzero = _mm_setzero_si128();
    size_t step_next_group = 7 * Stride - (Channels & ~size_t{7});

    const uint8_t* i0 = Input;
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

            CACULATE_ACCUMULATE_VECTORS();

            _mm_store_si128((__m128i*)acc, vacc_lo);
            _mm_store_si128(((__m128i*)acc) + 1, vacc_hi);
            acc += 8;
        }
        if (c > 0) {
            const __m128i vi0 = _mm_loadl_epi64((const __m128i*)(i0 >= LastOf8 ? MlasCopyTailBytes(tail, i0, c) : i0));
            const __m128i vi1 = _mm_loadl_epi64((const __m128i*)(i1 >= LastOf8 ? MlasCopyTailBytes(tail, i1, c) : i1));
            const __m128i vi2 = _mm_loadl_epi64((const __m128i*)(i2 >= LastOf8 ? MlasCopyTailBytes(tail, i2, c) : i2));
            const __m128i vi3 = _mm_loadl_epi64((const __m128i*)(i3 >= LastOf8 ? MlasCopyTailBytes(tail, i3, c) : i3));
            const __m128i vi4 = _mm_loadl_epi64((const __m128i*)(i4 >= LastOf8 ? MlasCopyTailBytes(tail, i4, c) : i4));
            const __m128i vi5 = _mm_loadl_epi64((const __m128i*)(i5 >= LastOf8 ? MlasCopyTailBytes(tail, i5, c) : i5));
            const __m128i vi6 = _mm_loadl_epi64((const __m128i*)(i6 >= LastOf8 ? MlasCopyTailBytes(tail, i6, c) : i6));

            CACULATE_ACCUMULATE_VECTORS();

            _mm_store_si128((__m128i*)acc, vacc_lo);
            _mm_store_si128(((__m128i*)acc) + 1, vacc_hi);
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
        case 1: i1 = ZeroBuffer; /* fall through */
        case 2: i2 = ZeroBuffer; /* fall through */
        case 3: i3 = ZeroBuffer; /* fall through */
        case 4: i4 = ZeroBuffer; /* fall through */
        case 5: i5 = ZeroBuffer; /* fall through */
        case 6: i6 = ZeroBuffer; /* fall through */
        default: break;
        }

        int32_t* acc = AccumulateBuffer;
        for (; Channels >= 8; Channels -= 8) {
            LOAD_FULL_CHANNELS();

            CACULATE_ACCUMULATE_VECTORS();

            vacc_lo = MlasRequantizeOutputVector(vacc_lo, vzero, vscale, vmin_value, vmax_value, vzero_point);
            vacc_hi = MlasRequantizeOutputVector(vacc_hi, vzero, vscale, vmin_value, vmax_value, vzero_point);
            __m128i vresult = _mm_packus_epi16(vacc_lo, vacc_hi);
            vresult = _mm_packus_epi16(vresult, vresult);
            _mm_storel_epi64((__m128i*)Output, vresult);
            acc += 8;
            Output += 8;
        }

        if (Channels > 0) {
            const __m128i vi0 = _mm_loadl_epi64((const __m128i*)(i0 >= LastOf8 ? MlasCopyTailBytes(tail, i0, Channels) : i0));
            const __m128i vi1 = _mm_loadl_epi64((const __m128i*)(1 < ImageSize && i1 >= LastOf8 ? MlasCopyTailBytes(tail, i1, Channels) : i1));
            const __m128i vi2 = _mm_loadl_epi64((const __m128i*)(2 < ImageSize && i2 >= LastOf8 ? MlasCopyTailBytes(tail, i2, Channels) : i2));
            const __m128i vi3 = _mm_loadl_epi64((const __m128i*)(3 < ImageSize && i3 >= LastOf8 ? MlasCopyTailBytes(tail, i3, Channels) : i3));
            const __m128i vi4 = _mm_loadl_epi64((const __m128i*)(4 < ImageSize && i4 >= LastOf8 ? MlasCopyTailBytes(tail, i4, Channels) : i4));
            const __m128i vi5 = _mm_loadl_epi64((const __m128i*)(5 < ImageSize && i5 >= LastOf8 ? MlasCopyTailBytes(tail, i5, Channels) : i5));
            const __m128i vi6 = _mm_loadl_epi64((const __m128i*)(6 < ImageSize && i6 >= LastOf8 ? MlasCopyTailBytes(tail, i6, Channels) : i6));

            CACULATE_ACCUMULATE_VECTORS();

            vacc_lo = MlasRequantizeOutputVector(vacc_lo, vzero, vscale, vmin_value, vmax_value, vzero_point);
            vacc_hi = MlasRequantizeOutputVector(vacc_hi, vzero, vscale, vmin_value, vmax_value, vzero_point);
            __m128i vresult = _mm_packus_epi16(vacc_lo, vacc_hi);
            vresult = _mm_packus_epi16(vresult, vresult);
            StoreTailBytes(Output, vresult, Channels);
        }
    }
}

void
MLASCALL
MlasNhwcQLinearGlobalAveragePool(
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
    )
{
    if (Stride == 1) {
        MlasQLinearGlobalAveragePool(Input, ScaleInput, ZeroPointInput, 
                                    Output, ScaleOutput, ZeroPointOutput, Batch, ImageSize);
        return;
    }

    float scale = CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    const auto vscale = _mm_set1_ps(scale);
    const int32_t Bias = -ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize);
    const auto vbias = _mm_set1_epi32(Bias);
    const auto vmin_value = _mm_set1_ps(float(0 - ZeroPointOutput));
    const auto vmax_value = _mm_set1_ps(float(255 - ZeroPointOutput));
    const auto vzero_point = _mm_set1_epi32(ZeroPointOutput);
    const uint8_t* InputLastOf8 = Input + (Batch * ImageSize * Stride - Stride + Channels) - 8;

    for (; Batch > 0; Batch--) {
        MlasNhwcQLinearGlobalAveragePoolSingleBatch(
            Input, Output, InputLastOf8, ImageSize, Channels, Stride,
            vbias, vscale, vmin_value, vmax_value, vzero_point,
            AccumulateBuffer, ZeroBuffer);
        Input += ImageSize * Stride;
        Output += Stride;
    }
}

#else

// Pure C++ Implementation

void
MLASCALL
MlasQLinearGlobalAveragePool(
    const uint8_t* Input,
    float ScaleInput,
    int32_t ZeroPointInput,
    uint8_t* Output,
    float ScaleOutput,
    int32_t ZeroPointOutput,
    size_t Channels,
    size_t ImageSize
    )
{
    float scale = CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    int32_t bias = -ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize);
    for (; Channels > 0; Channels--) {
        int32_t acc = bias;
        for (size_t i = 0; i < ImageSize; ++i) {
            acc += static_cast<int>(*Input++);
        }
        int32_t v = static_cast<int>(std::nearbyintf(acc * scale)) + ZeroPointOutput;
        *Output++ = std::max(std::min(255, v), 0);
    }
}

void
MLASCALL
MlasNhwcQLinearGlobalAveragePool(
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
    const uint8_t* /* ZeroBuffer */
    )
{
    float scale = CheckQLinearGlobalAveragePoolScaleAndSize(ScaleInput, ScaleOutput, ImageSize);
    int32_t bias = -ZeroPointInput * gsl::narrow_cast<int32_t>(ImageSize);
    for (; Batch > 0; Batch--) {
        const uint8_t* batch_input = Input;
        uint8_t* batch_output = Output;
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
            int32_t v = static_cast<int>(std::nearbyintf(AccumulateBuffer[c] * scale)) + ZeroPointOutput;
            *batch_output++ = std::max(std::min(255, v), 0);
        }
    }
}

#endif
