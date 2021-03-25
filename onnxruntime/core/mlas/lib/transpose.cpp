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

#if defined(MLAS_SSE2_INTRINSICS) || defined(MLAS_NEON_INTRINSICS)

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

    while (n >= 8) {

        const uint8_t* s = Input;
        uint8_t* d = Output;
        size_t m = M;

#if defined(MLAS_SSE2_INTRINSICS) || defined(MLAS_NEON_INTRINSICS)

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
