/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    reorder_avx512f.cpp

Abstract:

    This module implements AVX-512 accelerated helpers for the NCHW<->NCHWc
    reorder routines when the NCHWc block size is 16. The baseline reorder path
    transposes 4x4 tiles with SSE2, requiring four transposes per 16-wide block;
    these routines transpose full 16x16 tiles in one pass.

    The transposes are pure data-movement: for a tile,
    Output[p*DstStride + c] = Input[c*SrcStride + p].

--*/

#include <immintrin.h>

#include "mlasi.h"

namespace {

//
// Transpose a 16x16 float tile.
//
//  Input  row c (0..15): 16 contiguous floats at Input + c*SrcStride.
//  Output row p (0..15): 16 contiguous floats at Output + p*DstStride, with
//                        Output[p*DstStride + c] = Input[c*SrcStride + p].
//
MLAS_FORCEINLINE
void
MlasReorderTranspose16x16Avx512F(
    const float* Input,
    float* Output,
    size_t SrcStride,
    size_t DstStride
    )
{
    __m512 r[16];
    for (int i = 0; i < 16; i++) {
        r[i] = _mm512_loadu_ps(Input + i * SrcStride);
    }

    __m512 t[16];
    for (int i = 0; i < 8; i++) {
        t[2 * i + 0] = _mm512_unpacklo_ps(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_ps(r[2 * i], r[2 * i + 1]);
    }

    __m512 u[16];
    for (int i = 0; i < 4; i++) {
        u[4 * i + 0] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[4 * i + 0]), _mm512_castps_pd(t[4 * i + 2])));
        u[4 * i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[4 * i + 0]), _mm512_castps_pd(t[4 * i + 2])));
        u[4 * i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[4 * i + 1]), _mm512_castps_pd(t[4 * i + 3])));
        u[4 * i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[4 * i + 1]), _mm512_castps_pd(t[4 * i + 3])));
    }

    __m512 v[16];
    for (int i = 0; i < 4; i++) {
        v[i + 0]  = _mm512_shuffle_f32x4(u[i], u[i + 4], 0x88);
        v[i + 4]  = _mm512_shuffle_f32x4(u[i], u[i + 4], 0xDD);
        v[i + 8]  = _mm512_shuffle_f32x4(u[i + 8], u[i + 12], 0x88);
        v[i + 12] = _mm512_shuffle_f32x4(u[i + 8], u[i + 12], 0xDD);
    }

    for (int i = 0; i < 8; i++) {
        _mm512_storeu_ps(Output + (i + 0) * DstStride, _mm512_shuffle_f32x4(v[i], v[i + 8], 0x88));
        _mm512_storeu_ps(Output + (i + 8) * DstStride, _mm512_shuffle_f32x4(v[i], v[i + 8], 0xDD));
    }
}

}  // namespace

//
// Reorder one full 16-channel NCHW block into NCHWc (block size 16).
//
//  Source S: 16 channels, each InputSize contiguous spatial floats (stride InputSize).
//  Dest   D: InputSize spatial rows, each 16 contiguous channel floats (stride 16).
//  D[p*16 + c] = S[c*InputSize + p].
//
// Equivalent to the InputChannelsThisIteration == BlockSize == 16 case of the
// scalar MlasReorderInputNchw inner loops.
//
void
MLASCALL
MlasReorderInputNchwBlock16Avx512F(
    const float* S,
    float* D,
    size_t InputSize
    )
{
    size_t p = 0;
    for (; p + 16 <= InputSize; p += 16) {
        MlasReorderTranspose16x16Avx512F(S + p, D + p * 16, InputSize, 16);
    }
    for (; p < InputSize; p++) {
        float* d = D + p * 16;
        const float* s = S + p;
        for (int c = 0; c < 16; c++) {
            d[c] = s[c * InputSize];
        }
    }
}

//
// Reorder one full 16-channel NCHWc block into NCHW (block size 16), for a
// contiguous run of OutputSize spatial positions.
//
//  Source S: OutputSize spatial rows, each 16 contiguous channel floats (stride 16).
//  Dest   D: 16 channels, each OutputSize contiguous spatial floats (stride OutputSize).
//  D[c*OutputSize + p] = S[p*16 + c].
//
// Equivalent to the OutputChannelsThisIteration == BlockSize == 16 case of the
// scalar MlasReorderOutputNchwThreaded inner loops.
//
void
MLASCALL
MlasReorderOutputNchwBlock16Avx512F(
    const float* S,
    float* D,
    size_t OutputSize
    )
{
    size_t p = 0;
    for (; p + 16 <= OutputSize; p += 16) {
        // Source rows are spatial positions (stride 16 channels); dest rows are
        // channels (stride OutputSize). Transpose with SrcStride=16,
        // DstStride=OutputSize maps Output[c*OutputSize + p] = Input[p*16 + c].
        MlasReorderTranspose16x16Avx512F(S + p * 16, D + p, 16, OutputSize);
    }
    for (; p < OutputSize; p++) {
        const float* s = S + p * 16;
        float* d = D + p;
        for (int c = 0; c < 16; c++) {
            d[c * OutputSize] = s[c];
        }
    }
}
