/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    transpose.cpp

Abstract:

    This module implements the transpose operation.

--*/

#include "mlasi.h"

#ifdef MLAS_TARGET_AMD64_IX86

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

        while (m >= 8) {

            __m128i a0 = _mm_loadl_epi64((const __m128i*)&s[N * 0]);
            __m128i a1 = _mm_loadl_epi64((const __m128i*)&s[N * 1]);
            __m128i b0 = _mm_unpacklo_epi8(a0, a1);

            __m128i a2 = _mm_loadl_epi64((const __m128i*)&s[N * 2]);
            __m128i a3 = _mm_loadl_epi64((const __m128i*)&s[N * 3]);
            __m128i b1 = _mm_unpacklo_epi8(a2, a3);

            __m128i a4 = _mm_loadl_epi64((const __m128i*)&s[N * 4]);
            __m128i a5 = _mm_loadl_epi64((const __m128i*)&s[N * 5]);
            __m128i b2 = _mm_unpacklo_epi8(a4, a5);

            __m128i a6 = _mm_loadl_epi64((const __m128i*)&s[N * 6]);
            __m128i a7 = _mm_loadl_epi64((const __m128i*)&s[N * 7]);
            __m128i b3 = _mm_unpacklo_epi8(a6, a7);

            __m128i c0 = _mm_unpacklo_epi16(b0, b1);
            __m128i c1 = _mm_unpackhi_epi16(b0, b1);
            __m128i c2 = _mm_unpacklo_epi16(b2, b3);
            __m128i c3 = _mm_unpackhi_epi16(b2, b3);

            __m128 d0 = _mm_castsi128_ps(_mm_unpacklo_epi32(c0, c2));
            _mm_storel_pi((__m64*)&d[M * 0], d0);
            _mm_storeh_pi((__m64*)&d[M * 1], d0);

            __m128 d1 = _mm_castsi128_ps(_mm_unpackhi_epi32(c0, c2));
            _mm_storel_pi((__m64*)&d[M * 2], d1);
            _mm_storeh_pi((__m64*)&d[M * 3], d1);

            __m128 d2 = _mm_castsi128_ps(_mm_unpacklo_epi32(c1, c3));
            _mm_storel_pi((__m64*)&d[M * 4], d2);
            _mm_storeh_pi((__m64*)&d[M * 5], d2);

            __m128 d3 = _mm_castsi128_ps(_mm_unpackhi_epi32(c1, c3));
            _mm_storel_pi((__m64*)&d[M * 6], d3);
            _mm_storeh_pi((__m64*)&d[M * 7], d3);

            s += N * 8;
            d += 8;
            m -= 8;
        }

        while (m > 0) {

            d[M * 0] = s[0];
            d[M * 1] = s[1];
            d[M * 2] = s[2];
            d[M * 3] = s[3];
            d[M * 4] = s[4];
            d[M * 5] = s[5];
            d[M * 6] = s[6];
            d[M * 7] = s[7];

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

            d[0] = s[N * 0];
            d[1] = s[N * 1];
            d[2] = s[N * 2];
            d[3] = s[N * 3];
            d[4] = s[N * 4];
            d[5] = s[N * 5];
            d[6] = s[N * 6];
            d[7] = s[N * 7];

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

#endif
