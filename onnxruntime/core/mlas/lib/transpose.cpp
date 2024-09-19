/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    transpose.cpp

Abstract:

    This module implements the transpose operation.

--*/

#include "mlasi.h"

#if defined(MLAS_SSE2_INTRINSICS)

MLAS_FORCEINLINE
void
MlasTranspose4x4Block(
    const uint32_t* Input,
    size_t InputStride,
    uint32_t* Output,
    size_t OutputStride
    )
{
    __m128i a0 = _mm_loadu_si128((const __m128i*)&Input[InputStride * 0]);
    __m128i a1 = _mm_loadu_si128((const __m128i*)&Input[InputStride * 1]);
    __m128i a2 = _mm_loadu_si128((const __m128i*)&Input[InputStride * 2]);
    __m128i a3 = _mm_loadu_si128((const __m128i*)&Input[InputStride * 3]);

    __m128i b0 = _mm_unpacklo_epi32(a0, a2);
    __m128i b1 = _mm_unpackhi_epi32(a0, a2);
    __m128i b2 = _mm_unpacklo_epi32(a1, a3);
    __m128i b3 = _mm_unpackhi_epi32(a1, a3);

    __m128i c0 = _mm_unpacklo_epi32(b0, b2);
    __m128i c1 = _mm_unpackhi_epi32(b0, b2);
    __m128i c2 = _mm_unpacklo_epi32(b1, b3);
    __m128i c3 = _mm_unpackhi_epi32(b1, b3);

    _mm_storeu_si128((__m128i*)&Output[OutputStride * 0], c0);
    _mm_storeu_si128((__m128i*)&Output[OutputStride * 1], c1);
    _mm_storeu_si128((__m128i*)&Output[OutputStride * 2], c2);
    _mm_storeu_si128((__m128i*)&Output[OutputStride * 3], c3);
}

MLAS_FORCEINLINE
void
MlasTranspose4x4Block(
    const uint16_t* Input,
    size_t InputStride,
    uint16_t* Output,
    size_t OutputStride
    )
{
    __m128i a0 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 0]);
    __m128i a1 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 1]);
    __m128i a2 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 2]);
    __m128i a3 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 3]);

    __m128i b0 = _mm_unpacklo_epi16(a0, a2);
    __m128i b1 = _mm_unpacklo_epi16(a1, a3);

    __m128i c0 = _mm_unpacklo_epi16(b0, b1);
    __m128i c1 = _mm_unpackhi_epi16(b0, b1);

    _mm_storel_pi((__m64*)&Output[OutputStride * 0], _mm_castsi128_ps(c0));
    _mm_storeh_pi((__m64*)&Output[OutputStride * 1], _mm_castsi128_ps(c0));
    _mm_storel_pi((__m64*)&Output[OutputStride * 2], _mm_castsi128_ps(c1));
    _mm_storeh_pi((__m64*)&Output[OutputStride * 3], _mm_castsi128_ps(c1));
}

MLAS_FORCEINLINE
void
MlasTranspose8x8Block(
    const uint8_t* Input,
    size_t InputStride,
    uint8_t* Output,
    size_t OutputStride
    )
{
    __m128i a0 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 0]);
    __m128i a1 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 1]);
    __m128i b0 = _mm_unpacklo_epi8(a0, a1);

    __m128i a2 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 2]);
    __m128i a3 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 3]);
    __m128i b1 = _mm_unpacklo_epi8(a2, a3);

    __m128i a4 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 4]);
    __m128i a5 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 5]);
    __m128i b2 = _mm_unpacklo_epi8(a4, a5);

    __m128i a6 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 6]);
    __m128i a7 = _mm_loadl_epi64((const __m128i*)&Input[InputStride * 7]);
    __m128i b3 = _mm_unpacklo_epi8(a6, a7);

    __m128i c0 = _mm_unpacklo_epi16(b0, b1);
    __m128i c1 = _mm_unpackhi_epi16(b0, b1);
    __m128i c2 = _mm_unpacklo_epi16(b2, b3);
    __m128i c3 = _mm_unpackhi_epi16(b2, b3);

    __m128 d0 = _mm_castsi128_ps(_mm_unpacklo_epi32(c0, c2));
    _mm_storel_pi((__m64*)&Output[OutputStride * 0], d0);
    _mm_storeh_pi((__m64*)&Output[OutputStride * 1], d0);

    __m128 d1 = _mm_castsi128_ps(_mm_unpackhi_epi32(c0, c2));
    _mm_storel_pi((__m64*)&Output[OutputStride * 2], d1);
    _mm_storeh_pi((__m64*)&Output[OutputStride * 3], d1);

    __m128 d2 = _mm_castsi128_ps(_mm_unpacklo_epi32(c1, c3));
    _mm_storel_pi((__m64*)&Output[OutputStride * 4], d2);
    _mm_storeh_pi((__m64*)&Output[OutputStride * 5], d2);

    __m128 d3 = _mm_castsi128_ps(_mm_unpackhi_epi32(c1, c3));
    _mm_storel_pi((__m64*)&Output[OutputStride * 6], d3);
    _mm_storeh_pi((__m64*)&Output[OutputStride * 7], d3);
}

#elif defined(MLAS_NEON_INTRINSICS)

MLAS_FORCEINLINE
void
MlasTranspose4x4Block(
    const uint32_t* Input,
    size_t InputStride,
    uint32_t* Output,
    size_t OutputStride
    )
{
    uint32x4_t a0 = vld1q_u32(&Input[InputStride * 0]);
    uint32x4_t a1 = vld1q_u32(&Input[InputStride * 1]);
    uint32x4_t a2 = vld1q_u32(&Input[InputStride * 2]);
    uint32x4_t a3 = vld1q_u32(&Input[InputStride * 3]);

    uint32x4x2_t b0 = vzipq_u32(a0, a2);
    uint32x4x2_t b1 = vzipq_u32(a1, a3);

    uint32x4x2_t c0 = vzipq_u32(b0.val[0], b1.val[0]);
    uint32x4x2_t c1 = vzipq_u32(b0.val[1], b1.val[1]);

    vst1q_u32(&Output[OutputStride * 0], c0.val[0]);
    vst1q_u32(&Output[OutputStride * 1], c0.val[1]);
    vst1q_u32(&Output[OutputStride * 2], c1.val[0]);
    vst1q_u32(&Output[OutputStride * 3], c1.val[1]);
}

MLAS_FORCEINLINE
void
MlasTranspose4x4Block(
    const uint16_t* Input,
    size_t InputStride,
    uint16_t* Output,
    size_t OutputStride
    )
{
    uint16x4_t a0 = vld1_u16(&Input[InputStride * 0]);
    uint16x4_t a1 = vld1_u16(&Input[InputStride * 1]);
    uint16x4_t a2 = vld1_u16(&Input[InputStride * 2]);
    uint16x4_t a3 = vld1_u16(&Input[InputStride * 3]);

    uint16x4x2_t b0 = vzip_u16(a0, a2);
    uint16x4x2_t b1 = vzip_u16(a1, a3);

    uint16x4x2_t c0 = vzip_u16(b0.val[0], b1.val[0]);
    uint16x4x2_t c1 = vzip_u16(b0.val[1], b1.val[1]);

    vst1_u16(&Output[OutputStride * 0], c0.val[0]);
    vst1_u16(&Output[OutputStride * 1], c0.val[1]);
    vst1_u16(&Output[OutputStride * 2], c1.val[0]);
    vst1_u16(&Output[OutputStride * 3], c1.val[1]);
}

MLAS_FORCEINLINE
void
MlasTranspose8x8Block(
    const uint8_t* Input,
    size_t InputStride,
    uint8_t* Output,
    size_t OutputStride
    )
{
    uint8x8_t a0 = vld1_u8(&Input[InputStride * 0]);
    uint8x8_t a1 = vld1_u8(&Input[InputStride * 1]);
    uint8x8x2_t b0 = vzip_u8(a0, a1);

    uint8x8_t a2 = vld1_u8(&Input[InputStride * 2]);
    uint8x8_t a3 = vld1_u8(&Input[InputStride * 3]);
    uint8x8x2_t b1 = vzip_u8(a2, a3);

    uint8x8_t a4 = vld1_u8(&Input[InputStride * 4]);
    uint8x8_t a5 = vld1_u8(&Input[InputStride * 5]);
    uint8x8x2_t b2 = vzip_u8(a4, a5);

    uint8x8_t a6 = vld1_u8(&Input[InputStride * 6]);
    uint8x8_t a7 = vld1_u8(&Input[InputStride * 7]);
    uint8x8x2_t b3 = vzip_u8(a6, a7);

    uint16x4x2_t c0 = vzip_u16(vreinterpret_u16_u8(b0.val[0]), vreinterpret_u16_u8(b1.val[0]));
    uint16x4x2_t c1 = vzip_u16(vreinterpret_u16_u8(b0.val[1]), vreinterpret_u16_u8(b1.val[1]));
    uint16x4x2_t c2 = vzip_u16(vreinterpret_u16_u8(b2.val[0]), vreinterpret_u16_u8(b3.val[0]));
    uint16x4x2_t c3 = vzip_u16(vreinterpret_u16_u8(b2.val[1]), vreinterpret_u16_u8(b3.val[1]));

    uint32x2x2_t d0 = vzip_u32(vreinterpret_u32_u16(c0.val[0]), vreinterpret_u32_u16(c2.val[0]));
    uint32x2x2_t d1 = vzip_u32(vreinterpret_u32_u16(c0.val[1]), vreinterpret_u32_u16(c2.val[1]));
    uint32x2x2_t d2 = vzip_u32(vreinterpret_u32_u16(c1.val[0]), vreinterpret_u32_u16(c3.val[0]));
    uint32x2x2_t d3 = vzip_u32(vreinterpret_u32_u16(c1.val[1]), vreinterpret_u32_u16(c3.val[1]));

    vst1_u8(&Output[OutputStride * 0], vreinterpret_u8_u32(d0.val[0]));
    vst1_u8(&Output[OutputStride * 1], vreinterpret_u8_u32(d0.val[1]));
    vst1_u8(&Output[OutputStride * 2], vreinterpret_u8_u32(d1.val[0]));
    vst1_u8(&Output[OutputStride * 3], vreinterpret_u8_u32(d1.val[1]));
    vst1_u8(&Output[OutputStride * 4], vreinterpret_u8_u32(d2.val[0]));
    vst1_u8(&Output[OutputStride * 5], vreinterpret_u8_u32(d2.val[1]));
    vst1_u8(&Output[OutputStride * 6], vreinterpret_u8_u32(d3.val[0]));
    vst1_u8(&Output[OutputStride * 7], vreinterpret_u8_u32(d3.val[1]));
}

#elif defined(MLAS_TARGET_POWER)

MLAS_FORCEINLINE
void
MlasTranspose4x4Block(
    const uint32_t* Input,
    size_t InputStride,
    uint32_t* Output,
    size_t OutputStride
    )
{
    __vector unsigned int a0 = vec_vsx_ld(0, Input);
    __vector unsigned int a1 = vec_vsx_ld(0, &Input[InputStride]);
    __vector unsigned int a2 = vec_vsx_ld(0, &Input[InputStride * 2]);
    __vector unsigned int a3 = vec_vsx_ld(0, &Input[InputStride * 3]);

    __vector unsigned int b0 = vec_mergeh(a0, a1);
    __vector unsigned int b1 = vec_mergeh(a2, a3);
    __vector unsigned int b2 = vec_mergel(a0, a1);
    __vector unsigned int b3 = vec_mergel(a2, a3);

    __vector unsigned int c0 = vec_xxpermdi(b0, b1, 0);
    __vector unsigned int c1 = vec_xxpermdi(b0, b1, 3);
    __vector unsigned int c2 = vec_xxpermdi(b2, b3, 0);
    __vector unsigned int c3 = vec_xxpermdi(b2, b3, 3);

    vec_vsx_st(c0, 0, Output);
    vec_vsx_st(c1, 0, &Output[OutputStride]);
    vec_vsx_st(c2, 0, &Output[OutputStride * 2]);
    vec_vsx_st(c3, 0, &Output[OutputStride * 3]);
}

MLAS_FORCEINLINE
void
MlasTranspose16x16Block(
    const uint8_t* Input,
    size_t InputStride,
    uint8_t* Output,
    size_t OutputStride
    )
{
    __vector unsigned char a0 = vec_vsx_ld(0, Input);
    __vector unsigned char a1 = vec_vsx_ld(0, &Input[InputStride]);
    __vector unsigned char a2 = vec_vsx_ld(0, &Input[InputStride * 2]);
    __vector unsigned char a3 = vec_vsx_ld(0, &Input[InputStride * 3]);
    __vector unsigned char a4 = vec_vsx_ld(0, &Input[InputStride * 4]);
    __vector unsigned char a5 = vec_vsx_ld(0, &Input[InputStride * 5]);
    __vector unsigned char a6 = vec_vsx_ld(0, &Input[InputStride * 6]);
    __vector unsigned char a7 = vec_vsx_ld(0, &Input[InputStride * 7]);
    __vector unsigned char a8 = vec_vsx_ld(0, &Input[InputStride * 8]);
    __vector unsigned char a9 = vec_vsx_ld(0, &Input[InputStride * 9]);
    __vector unsigned char a10 = vec_vsx_ld(0, &Input[InputStride * 10]);
    __vector unsigned char a11 = vec_vsx_ld(0, &Input[InputStride * 11]);
    __vector unsigned char a12 = vec_vsx_ld(0, &Input[InputStride * 12]);
    __vector unsigned char a13 = vec_vsx_ld(0, &Input[InputStride * 13]);
    __vector unsigned char a14 = vec_vsx_ld(0, &Input[InputStride * 14]);
    __vector unsigned char a15 = vec_vsx_ld(0, &Input[InputStride * 15]);

    __vector unsigned char b0 = vec_mergeh(a0, a1);
    __vector unsigned char b1 = vec_mergeh(a2, a3);
    __vector unsigned char b2 = vec_mergeh(a4, a5);
    __vector unsigned char b3 = vec_mergeh(a6, a7);
    __vector unsigned char b4 = vec_mergeh(a8, a9);
    __vector unsigned char b5 = vec_mergeh(a10, a11);
    __vector unsigned char b6 = vec_mergeh(a12, a13);
    __vector unsigned char b7 = vec_mergeh(a14, a15);
    __vector unsigned char c0 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned short>(b0), reinterpret_cast<__vector unsigned short>(b1)));
    __vector unsigned char c1 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned short>(b2), reinterpret_cast<__vector unsigned short>(b3)));
    __vector unsigned char c2 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned short>(b4), reinterpret_cast<__vector unsigned short>(b5)));
    __vector unsigned char c3 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned short>(b6), reinterpret_cast<__vector unsigned short>(b7)));

    __vector unsigned char d0 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned int>(c0), reinterpret_cast<__vector unsigned int>(c1)));
    __vector unsigned char d1 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned int>(c2), reinterpret_cast<__vector unsigned int>(c3)));
    __vector unsigned char e0 = vec_xxpermdi(d0, d1, 0);
    __vector unsigned char e1 = vec_xxpermdi(d0, d1, 3);
    vec_vsx_st(e0, 0, &Output[0]);
    vec_vsx_st(e1, 0, &Output[OutputStride]);

    d0 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned int>(c0), reinterpret_cast<__vector unsigned int>(c1)));
    d1 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned int>(c2), reinterpret_cast<__vector unsigned int>(c3)));
    e0 = vec_xxpermdi(d0, d1, 0);
    e1 = vec_xxpermdi(d0, d1, 3);
    vec_vsx_st(e0, 0, &Output[OutputStride * 2]);
    vec_vsx_st(e1, 0, &Output[OutputStride * 3]);

    c0 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned short>(b0), reinterpret_cast<__vector unsigned short>(b1)));
    c1 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned short>(b2), reinterpret_cast<__vector unsigned short>(b3)));
    c2 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned short>(b4), reinterpret_cast<__vector unsigned short>(b5)));
    c3 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned short>(b6), reinterpret_cast<__vector unsigned short>(b7)));

    d0 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned int>(c0), reinterpret_cast<__vector unsigned int>(c1)));
    d1 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned int>(c2), reinterpret_cast<__vector unsigned int>(c3)));
    e0 = vec_xxpermdi(d0, d1, 0);
    e1 = vec_xxpermdi(d0, d1, 3);
    vec_vsx_st(e0, 0, &Output[OutputStride * 4]);
    vec_vsx_st(e1, 0, &Output[OutputStride * 5]);

    d0 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned int>(c0), reinterpret_cast<__vector unsigned int>(c1)));
    d1 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned int>(c2), reinterpret_cast<__vector unsigned int>(c3)));
    e0 = vec_xxpermdi(d0, d1, 0);
    e1 = vec_xxpermdi(d0, d1, 3);
    vec_vsx_st(e0, 0, &Output[OutputStride * 6]);
    vec_vsx_st(e1, 0, &Output[OutputStride * 7]);

    b0 = vec_mergel(a0, a1);
    b1 = vec_mergel(a2, a3);
    b2 = vec_mergel(a4, a5);
    b3 = vec_mergel(a6, a7);
    b4 = vec_mergel(a8, a9);
    b5 = vec_mergel(a10, a11);
    b6 = vec_mergel(a12, a13);
    b7 = vec_mergel(a14, a15);

    c0 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned short>(b0), reinterpret_cast<__vector unsigned short>(b1)));
    c1 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned short>(b2), reinterpret_cast<__vector unsigned short>(b3)));
    c2 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned short>(b4), reinterpret_cast<__vector unsigned short>(b5)));
    c3 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned short>(b6), reinterpret_cast<__vector unsigned short>(b7)));

    d0 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned int>(c0), reinterpret_cast<__vector unsigned int>(c1)));
    d1 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned int>(c2), reinterpret_cast<__vector unsigned int>(c3)));
    e0 = vec_xxpermdi(d0, d1, 0);
    e1 = vec_xxpermdi(d0, d1, 3);
    vec_vsx_st(e0, 0, &Output[OutputStride * 8]);
    vec_vsx_st(e1, 0, &Output[OutputStride * 9]);

    d0 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned int>(c0), reinterpret_cast<__vector unsigned int>(c1)));
    d1 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned int>(c2), reinterpret_cast<__vector unsigned int>(c3)));
    e0 = vec_xxpermdi(d0, d1, 0);
    e1 = vec_xxpermdi(d0, d1, 3);
    vec_vsx_st(e0, 0, &Output[OutputStride * 10]);
    vec_vsx_st(e1, 0, &Output[OutputStride * 11]);

    c0 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned short>(b0), reinterpret_cast<__vector unsigned short>(b1)));
    c1 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned short>(b2), reinterpret_cast<__vector unsigned short>(b3)));
    c2 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned short>(b4), reinterpret_cast<__vector unsigned short>(b5)));
    c3 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned short>(b6), reinterpret_cast<__vector unsigned short>(b7)));

    d0 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned int>(c0), reinterpret_cast<__vector unsigned int>(c1)));
    d1 = reinterpret_cast<__vector unsigned char>(vec_mergeh(reinterpret_cast<__vector unsigned int>(c2), reinterpret_cast<__vector unsigned int>(c3)));
    e0 = vec_xxpermdi(d0, d1, 0);
    e1 = vec_xxpermdi(d0, d1, 3);
    vec_vsx_st(e0, 0, &Output[OutputStride * 12]);
    vec_vsx_st(e1, 0, &Output[OutputStride * 13]);

    d0 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned int>(c0), reinterpret_cast<__vector unsigned int>(c1)));
    d1 = reinterpret_cast<__vector unsigned char>(vec_mergel(reinterpret_cast<__vector unsigned int>(c2), reinterpret_cast<__vector unsigned int>(c3)));
    e0 = vec_xxpermdi(d0, d1, 0);
    e1 = vec_xxpermdi(d0, d1, 3);
    vec_vsx_st(e0, 0, &Output[OutputStride * 14]);
    vec_vsx_st(e1, 0, &Output[OutputStride * 15]);
}

#elif defined(MLAS_LSX_INTRINSICS)

MLAS_FORCEINLINE
void
MlasTranspose4x4Block(
    const uint32_t* Input,
    size_t InputStride,
    uint32_t* Output,
    size_t OutputStride
    )
{
    __m128i a0 = __lsx_vld((const __m128i*)&Input[InputStride * 0], 0);
    __m128i a1 = __lsx_vld((const __m128i*)&Input[InputStride * 1], 0);
    __m128i a2 = __lsx_vld((const __m128i*)&Input[InputStride * 2], 0);
    __m128i a3 = __lsx_vld((const __m128i*)&Input[InputStride * 3], 0);

    __m128i b0 = __lsx_vilvl_w(a2, a0);
    __m128i b1 = __lsx_vilvh_w(a2, a0);
    __m128i b2 = __lsx_vilvl_w(a3, a1);
    __m128i b3 = __lsx_vilvh_w(a3, a1);
    __m128i c0 = __lsx_vilvl_w(b2, b0);
    __m128i c1 = __lsx_vilvh_w(b2, b0);
    __m128i c2 = __lsx_vilvl_w(b3, b1);
    __m128i c3 = __lsx_vilvh_w(b3, b1);

    __lsx_vst(c0, (__m128i*)&Output[OutputStride * 0], 0);
    __lsx_vst(c1, (__m128i*)&Output[OutputStride * 1], 0);
    __lsx_vst(c2, (__m128i*)&Output[OutputStride * 2], 0);
    __lsx_vst(c3, (__m128i*)&Output[OutputStride * 3], 0);
}

MLAS_FORCEINLINE
void
MlasTranspose4x4Block(
    const uint16_t* Input,
    size_t InputStride,
    uint16_t* Output,
    size_t OutputStride
    )
{
    __m128i a0 = __lsx_vld((const __m128i*)&Input[InputStride * 0], 0);
    __lsx_vinsgr2vr_d(a0, 0 , 1);
    __m128i a1 = __lsx_vld((const __m128i*)&Input[InputStride * 1], 0);
    __lsx_vinsgr2vr_d(a1, 0 , 1);
    __m128i a2 = __lsx_vld((const __m128i*)&Input[InputStride * 2], 0);
    __lsx_vinsgr2vr_d(a2, 0 , 1);
    __m128i a3 = __lsx_vld((const __m128i*)&Input[InputStride * 3], 0);
    __lsx_vinsgr2vr_d(a3, 0 , 1);

    __m128i b0 = __lsx_vilvl_h(a2, a0);
    __m128i b1 = __lsx_vilvl_h(a3, a1);
    __m128i c0 = __lsx_vilvl_h(b1, b0);
    __m128i c1 = __lsx_vilvh_h(b1, b0);

    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 0], 0), __lsx_vpickve2gr_d(c0, 0), 0), (__m128i *)&Output[OutputStride * 0], 0);
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 1], 0), __lsx_vpickve2gr_d(c0, 1), 0), (__m128i *)&Output[OutputStride * 1], 0);
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 2], 0), __lsx_vpickve2gr_d(c1, 0), 0), (__m128i *)&Output[OutputStride * 2], 0);
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 3], 0), __lsx_vpickve2gr_d(c1, 1), 0), (__m128i *)&Output[OutputStride * 3], 0);
}

MLAS_FORCEINLINE
void
MlasTranspose8x8Block(
    const uint8_t* Input,
    size_t InputStride,
    uint8_t* Output,
    size_t OutputStride
    )
{
    __m128i a0 = __lsx_vld((const __m128i*)&Input[InputStride * 0], 0);
    __lsx_vinsgr2vr_d(a0, 0, 1);
    __m128i a1 = __lsx_vld((const __m128i*)&Input[InputStride * 1], 0);
    __lsx_vinsgr2vr_d(a1, 0, 1);
    __m128i b0 = __lsx_vilvl_b(a1, a0);

    __m128i a2 = __lsx_vld((const __m128i*)&Input[InputStride * 2], 0);
    __lsx_vinsgr2vr_d(a2, 0, 1);
    __m128i a3 = __lsx_vld((const __m128i*)&Input[InputStride * 3], 0);
    __lsx_vinsgr2vr_d(a3, 0, 1);
    __m128i b1 = __lsx_vilvl_b(a3, a2);

    __m128i a4 = __lsx_vld((const __m128i*)&Input[InputStride * 4], 0);
    __lsx_vinsgr2vr_d(a4, 0, 1);
    __m128i a5 = __lsx_vld((const __m128i*)&Input[InputStride * 5], 0);
    __lsx_vinsgr2vr_d(a5, 0, 1);
    __m128i b2 = __lsx_vilvl_b(a5, a4);

    __m128i a6 = __lsx_vld((const __m128i*)&Input[InputStride * 6], 0);
    __lsx_vinsgr2vr_d(a6, 0, 1);
    __m128i a7 = __lsx_vld((const __m128i*)&Input[InputStride * 7], 0);
    __lsx_vinsgr2vr_d(a7, 0, 1);
    __m128i b3 = __lsx_vilvl_b(a7, a6);
    __m128i c0 = __lsx_vilvl_h(b1, b0);
    __m128i c1 = __lsx_vilvh_h(b1, b0);
    __m128i c2 = __lsx_vilvl_h(b3, b2);
    __m128i c3 = __lsx_vilvh_h(b3, b2);

    __m128 d0 = (__m128)(__lsx_vilvl_w(c2, c0));
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 0], 0), __lsx_vpickve2gr_d(d0, 0), 0), (__m128i *)&Output[OutputStride * 0], 0);
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 1], 0), __lsx_vpickve2gr_d(d0, 1), 0), (__m128i *)&Output[OutputStride * 1], 0);

    __m128 d1 = (__m128)(__lsx_vilvh_w(c2, c0));
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 2], 0), __lsx_vpickve2gr_d(d1, 0), 0), (__m128i *)&Output[OutputStride * 2], 0);
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 3], 0), __lsx_vpickve2gr_d(d1, 1), 0), (__m128i *)&Output[OutputStride * 3], 0);

    __m128 d2 = (__m128)(__lsx_vilvl_w(c3, c1));
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 4], 0), __lsx_vpickve2gr_d(d2, 0), 0), (__m128i *)&Output[OutputStride * 4], 0);
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 5], 0), __lsx_vpickve2gr_d(d2, 1), 0), (__m128i *)&Output[OutputStride * 5], 0);

    __m128 d3 = (__m128)(__lsx_vilvh_w(c3, c1));
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 6], 0), __lsx_vpickve2gr_d(d3, 0), 0), (__m128i *)&Output[OutputStride * 6], 0);
    __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((__m128i *)&Output[OutputStride * 7], 0), __lsx_vpickve2gr_d(d3, 1), 0), (__m128i *)&Output[OutputStride * 7], 0);
}

#endif

template<typename ElementType>
MLAS_FORCEINLINE
void
MlasTranspose4xNVector(
    const ElementType* Input,
    size_t InputStride,
    ElementType* Output,
    size_t OutputStride
    )
{
    ElementType a0 = Input[InputStride * 0];
    ElementType a1 = Input[InputStride * 1];
    ElementType a2 = Input[InputStride * 2];
    ElementType a3 = Input[InputStride * 3];

    Output[OutputStride * 0] = a0;
    Output[OutputStride * 1] = a1;
    Output[OutputStride * 2] = a2;
    Output[OutputStride * 3] = a3;
}

#if defined(MLAS_TARGET_POWER)
template<typename ElementType>
MLAS_FORCEINLINE
void
MlasTranspose16xNVector(
    const ElementType* Input,
    size_t InputStride,
    ElementType* Output,
    size_t OutputStride
    )
{
    MlasTranspose4xNVector(&Input[InputStride * 0], InputStride, &Output[OutputStride * 0], OutputStride);
    MlasTranspose4xNVector(&Input[InputStride * 4], InputStride, &Output[OutputStride * 4], OutputStride);
    MlasTranspose4xNVector(&Input[InputStride * 8], InputStride, &Output[OutputStride * 8], OutputStride);
    MlasTranspose4xNVector(&Input[InputStride * 12], InputStride, &Output[OutputStride * 12], OutputStride);
}
#endif

template<typename ElementType>
MLAS_FORCEINLINE
void
MlasTranspose8xNVector(
    const ElementType* Input,
    size_t InputStride,
    ElementType* Output,
    size_t OutputStride
    )
{
    MlasTranspose4xNVector(&Input[InputStride * 0], InputStride, &Output[OutputStride * 0], OutputStride);
    MlasTranspose4xNVector(&Input[InputStride * 4], InputStride, &Output[OutputStride * 4], OutputStride);
}

void
MLASCALL
MlasTranspose(
    const uint32_t* Input,
    uint32_t* Output,
    size_t M,
    size_t N
    )
/*++

Routine Description:

    This routine transposes the input matrix (M rows by N columns) to the
    output matrix (N rows by M columns).

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    M - Supplies the number of rows for the input matrix and the number of
        columns for the output matrix.

    N - Supplies the number of columns for the input matrix and the number of
        rows for the output matrix.

Return Value:

    None.

--*/
{
    size_t n = N;

    //
    // Transpose elements from the input matrix to the output matrix 4 columns
    // at a time.
    //

    while (n >= 4) {

        const uint32_t* s = Input;
        uint32_t* d = Output;
        size_t m = M;

#if defined(MLAS_SSE2_INTRINSICS) || defined(MLAS_NEON_INTRINSICS) || defined(MLAS_TARGET_POWER) || \
    defined(MLAS_LSX_INTRINSICS)

        while (m >= 4) {

            MlasTranspose4x4Block(s, N, d, M);

            s += N * 4;
            d += 4;
            m -= 4;
        }

#endif

        while (m > 0) {

            MlasTranspose4xNVector(s, 1, d, M);

            s += N;
            d += 1;
            m -= 1;
        }

        Input += 4;
        Output += M * 4;
        n -= 4;
    }

    //
    // Transpose elements from the input matrix to the output matrix for the
    // remaining columns.
    //

    while (n > 0) {

        const uint32_t* s = Input;
        uint32_t* d = Output;
        size_t m = M;

        while (m >= 4) {

            MlasTranspose4xNVector(s, N, d, 1);

            s += N * 4;
            d += 4;
            m -= 4;
        }

        while (m > 0) {

            d[0] = s[0];

            s += N;
            d += 1;
            m -= 1;
        }

        Input += 1;
        Output += M;
        n -= 1;
    }
}

void
MLASCALL
MlasTranspose(
    const float* Input,
    float* Output,
    size_t M,
    size_t N
    )
{
    MlasTranspose(
        reinterpret_cast<const uint32_t*>(Input),
        reinterpret_cast<uint32_t*>(Output),
        M,
        N);
}


void
MLASCALL
MlasTranspose(
    const uint16_t* Input,
    uint16_t* Output,
    size_t M,
    size_t N
    )
/*++

Routine Description:

    This routine transposes the input matrix (M rows by N columns) to the
    output matrix (N rows by M columns).

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    M - Supplies the number of rows for the input matrix and the number of
        columns for the output matrix.

    N - Supplies the number of columns for the input matrix and the number of
        rows for the output matrix.

Return Value:

    None.

--*/
{
    size_t n = N;

    //
    // Transpose elements from the input matrix to the output matrix 4 columns
    // at a time.
    //

    while (n >= 4) {

        const uint16_t* s = Input;
        uint16_t* d = Output;
        size_t m = M;

#if defined(MLAS_SSE2_INTRINSICS) || defined(MLAS_NEON_INTRINSICS)  || defined(MLAS_LSX_INTRINSICS)

        while (m >= 4) {

            MlasTranspose4x4Block(s, N, d, M);

            s += N * 4;
            d += 4;
            m -= 4;
        }

#endif

        while (m > 0) {

            MlasTranspose4xNVector(s, 1, d, M);

            s += N;
            d += 1;
            m -= 1;
        }

        Input += 4;
        Output += M * 4;
        n -= 4;
    }

    //
    // Transpose elements from the input matrix to the output matrix for the
    // remaining columns.
    //

    while (n > 0) {

        const uint16_t* s = Input;
        uint16_t* d = Output;
        size_t m = M;

        while (m >= 4) {

            MlasTranspose4xNVector(s, N, d, 1);

            s += N * 4;
            d += 4;
            m -= 4;
        }

        while (m > 0) {

            d[0] = s[0];

            s += N;
            d += 1;
            m -= 1;
        }

        Input += 1;
        Output += M;
        n -= 1;
    }
}


void
MLASCALL
MlasTranspose(
    const uint8_t* Input,
    uint8_t* Output,
    size_t M,
    size_t N
    )
/*++

Routine Description:

    This routine transposes the input matrix (M rows by N columns) to the
    output matrix (N rows by M columns).

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    M - Supplies the number of rows for the input matrix and the number of
        columns for the output matrix.

    N - Supplies the number of columns for the input matrix and the number of
        rows for the output matrix.

Return Value:

    None.

--*/
{
    size_t n = N;

    //
    // Transpose elements from the input matrix to the output matrix 8 columns
    // at a time.
    //
#if defined(MLAS_TARGET_POWER)
    while (n >= 16) {

        const uint8_t* s = Input;
        uint8_t* d = Output;
        size_t m = M;
        while (m >= 16) {

            MlasTranspose16x16Block(s, N, d, M);

            s += N * 16;
            d += 16;
            m -= 16;
        }

        while (m > 0) {

            MlasTranspose16xNVector(s, 1, d, M);

            s += N;
            d += 1;
            m -= 1;
        }

        Input += 16;
        Output += M * 16;
        n -= 16;
    }
#endif
    while (n >= 8) {

        const uint8_t* s = Input;
        uint8_t* d = Output;
        size_t m = M;

#if defined(MLAS_SSE2_INTRINSICS) || defined(MLAS_NEON_INTRINSICS)  || defined(MLAS_LSX_INTRINSICS)

        while (m >= 8) {

            MlasTranspose8x8Block(s, N, d, M);

            s += N * 8;
            d += 8;
            m -= 8;
        }

#endif

        while (m > 0) {

            MlasTranspose8xNVector(s, 1, d, M);

            s += N;
            d += 1;
            m -= 1;
        }

        Input += 8;
        Output += M * 8;
        n -= 8;
    }

    //
    // Transpose elements from the input matrix to the output matrix for the
    // remaining columns.
    //

    while (n > 0) {

        const uint8_t* s = Input;
        uint8_t* d = Output;
        size_t m = M;

        while (m >= 8) {

            MlasTranspose8xNVector(s, N, d, 1);

            s += N * 8;
            d += 8;
            m -= 8;
        }

        while (m > 0) {

            d[0] = s[0];

            s += N;
            d += 1;
            m -= 1;
        }

        Input += 1;
        Output += M;
        n -= 1;
    }
}

void
MLASCALL
MlasTranspose(
    const int8_t* Input,
    int8_t* Output,
    size_t M,
    size_t N)
{
    MlasTranspose(
        reinterpret_cast<const uint8_t*>(Input),
        reinterpret_cast<uint8_t*>(Output),
        M,
        N);
}
