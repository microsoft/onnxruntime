/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    snchwc_winograd_conv_avx512f.cpp

Abstract:

    This module implements Winograd helper routines for NCHWc convolution.

--*/

#include "mlasi.h"
#include "snchwc_winograd_conv_avx512f.h"

thread_local std::unique_ptr<float[]> MlasWinogradThreadedScratchBuffer;
thread_local size_t MlasWinogradThreadedScratchBufferSize = 0;

float*
MlasWinogradGetThreadedScratchBuffer(
    size_t FloatCount
    )
{
    if (MlasWinogradThreadedScratchBufferSize < FloatCount) {
        MlasWinogradThreadedScratchBuffer = std::make_unique<float[]>(FloatCount);
        MlasWinogradThreadedScratchBufferSize = FloatCount;
    }

    return MlasWinogradThreadedScratchBuffer.get();
}

void
MlasWinogradTransformInputBlockF2x2_3x3Avx512(
    const float* InputChannelBlock,
    size_t InputHeight,
    size_t InputWidth,
    size_t TileOutputY,
    size_t TileOutputX,
    float* TransformedInputBlock
    )
{
    __m512 D[MLAS_WINOGRAD_TRANSFORM_SIZE][MLAS_WINOGRAD_TRANSFORM_SIZE];
    const __m512 Zero = _mm512_setzero_ps();

    for (size_t iy = 0; iy < MLAS_WINOGRAD_TRANSFORM_SIZE; iy++) {
        for (size_t ix = 0; ix < MLAS_WINOGRAD_TRANSFORM_SIZE; ix++) {
            const ptrdiff_t InputY = ptrdiff_t(TileOutputY + iy) - 1;
            const ptrdiff_t InputX = ptrdiff_t(TileOutputX + ix) - 1;

            if (InputY >= 0 && InputY < ptrdiff_t(InputHeight) && InputX >= 0 && InputX < ptrdiff_t(InputWidth)) {
                D[iy][ix] = _mm512_loadu_ps(InputChannelBlock + (size_t(InputY) * InputWidth + size_t(InputX)) * 16);
            } else {
                D[iy][ix] = Zero;
            }
        }
    }

    __m512 T[MLAS_WINOGRAD_TRANSFORM_SIZE][MLAS_WINOGRAD_TRANSFORM_SIZE];

    for (size_t c = 0; c < MLAS_WINOGRAD_TRANSFORM_SIZE; c++) {
        T[0][c] = _mm512_sub_ps(D[0][c], D[2][c]);
        T[1][c] = _mm512_add_ps(D[1][c], D[2][c]);
        T[2][c] = _mm512_sub_ps(D[2][c], D[1][c]);
        T[3][c] = _mm512_sub_ps(D[1][c], D[3][c]);
    }

    __m512 V[MLAS_WINOGRAD_TRANSFORM_SIZE][MLAS_WINOGRAD_TRANSFORM_SIZE];

    for (size_t r = 0; r < MLAS_WINOGRAD_TRANSFORM_SIZE; r++) {
        V[r][0] = _mm512_sub_ps(T[r][0], T[r][2]);
        V[r][1] = _mm512_add_ps(T[r][1], T[r][2]);
        V[r][2] = _mm512_sub_ps(T[r][2], T[r][1]);
        V[r][3] = _mm512_sub_ps(T[r][1], T[r][3]);
    }

    for (size_t k = 0; k < MLAS_WINOGRAD_TRANSFORM_COUNT; k++) {
        _mm512_storeu_ps(TransformedInputBlock + k * 16, V[k / MLAS_WINOGRAD_TRANSFORM_SIZE][k % MLAS_WINOGRAD_TRANSFORM_SIZE]);
    }
}

void
MlasWinogradTransformOutputBlockF2x2_3x3Avx512(
    const float* TransformedOutput,
    const float* Bias,
    float* OutputBlockBase,
    size_t OutputWidth,
    size_t TileOutputY,
    size_t TileOutputX,
    size_t ValidRows,
    size_t ValidCols,
    bool ZeroMode
    )
{
    __m512 M[MLAS_WINOGRAD_TRANSFORM_SIZE][MLAS_WINOGRAD_TRANSFORM_SIZE];
    for (size_t k = 0; k < MLAS_WINOGRAD_TRANSFORM_COUNT; k++) {
        M[k / MLAS_WINOGRAD_TRANSFORM_SIZE][k % MLAS_WINOGRAD_TRANSFORM_SIZE] =
            _mm512_loadu_ps(TransformedOutput + k * 16);
    }

    __m512 T[MLAS_WINOGRAD_TILE_SIZE][MLAS_WINOGRAD_TRANSFORM_SIZE];
    for (size_t c = 0; c < MLAS_WINOGRAD_TRANSFORM_SIZE; c++) {
        T[0][c] = _mm512_add_ps(_mm512_add_ps(M[0][c], M[1][c]), M[2][c]);
        T[1][c] = _mm512_sub_ps(_mm512_sub_ps(M[1][c], M[2][c]), M[3][c]);
    }

    __m512 Y00 = _mm512_add_ps(_mm512_add_ps(T[0][0], T[0][1]), T[0][2]);
    __m512 Y01 = _mm512_sub_ps(_mm512_sub_ps(T[0][1], T[0][2]), T[0][3]);
    __m512 Y10 = _mm512_add_ps(_mm512_add_ps(T[1][0], T[1][1]), T[1][2]);
    __m512 Y11 = _mm512_sub_ps(_mm512_sub_ps(T[1][1], T[1][2]), T[1][3]);

    if (Bias != nullptr) {
        const __m512 BiasVector = _mm512_loadu_ps(Bias);
        Y00 = _mm512_add_ps(Y00, BiasVector);
        Y01 = _mm512_add_ps(Y01, BiasVector);
        Y10 = _mm512_add_ps(Y10, BiasVector);
        Y11 = _mm512_add_ps(Y11, BiasVector);
    }

    if (!ZeroMode) {
        if (ValidRows >= 1 && ValidCols >= 1) {
            Y00 = _mm512_add_ps(Y00, _mm512_loadu_ps(OutputBlockBase + (TileOutputY * OutputWidth + TileOutputX) * 16));
        }
        if (ValidRows >= 1 && ValidCols >= 2) {
            Y01 = _mm512_add_ps(Y01, _mm512_loadu_ps(OutputBlockBase + (TileOutputY * OutputWidth + TileOutputX + 1) * 16));
        }
        if (ValidRows >= 2 && ValidCols >= 1) {
            Y10 = _mm512_add_ps(Y10, _mm512_loadu_ps(OutputBlockBase + ((TileOutputY + 1) * OutputWidth + TileOutputX) * 16));
        }
        if (ValidRows >= 2 && ValidCols >= 2) {
            Y11 = _mm512_add_ps(Y11, _mm512_loadu_ps(OutputBlockBase + ((TileOutputY + 1) * OutputWidth + TileOutputX + 1) * 16));
        }
    }

    if (ValidRows >= 1 && ValidCols >= 1) {
        _mm512_storeu_ps(OutputBlockBase + (TileOutputY * OutputWidth + TileOutputX) * 16, Y00);
    }
    if (ValidRows >= 1 && ValidCols >= 2) {
        _mm512_storeu_ps(OutputBlockBase + (TileOutputY * OutputWidth + TileOutputX + 1) * 16, Y01);
    }
    if (ValidRows >= 2 && ValidCols >= 1) {
        _mm512_storeu_ps(OutputBlockBase + ((TileOutputY + 1) * OutputWidth + TileOutputX) * 16, Y10);
    }
    if (ValidRows >= 2 && ValidCols >= 2) {
        _mm512_storeu_ps(OutputBlockBase + ((TileOutputY + 1) * OutputWidth + TileOutputX + 1) * 16, Y11);
    }
}

void
MlasWinogradAccumulateOutputBlocksAvx512(
    size_t InputChannels,
    const float* Filter0,
    const float* Filter1,
    const float* Filter2,
    const float* Filter3,
    const float* TransformedInput,
    float* Accumulator0,
    float* Accumulator1,
    float* Accumulator2,
    float* Accumulator3,
    size_t OutputBlockCountThisIteration
    )
{
    constexpr size_t BlockSize = 16;

    for (size_t k = 0; k < MLAS_WINOGRAD_TRANSFORM_COUNT; k++) {
        __m512 Acc0 = _mm512_setzero_ps();
        __m512 Acc1 = _mm512_setzero_ps();
        __m512 Acc2 = _mm512_setzero_ps();
        __m512 Acc3 = _mm512_setzero_ps();

        for (size_t ic = 0; ic < InputChannels; ic += BlockSize) {
            const float* InputTransform = TransformedInput + (ic / BlockSize) * MLAS_WINOGRAD_TRANSFORM_COUNT * BlockSize + k * BlockSize;
            const float* FilterTransform0 = Filter0 + ic * MLAS_WINOGRAD_TRANSFORM_COUNT * BlockSize + k * BlockSize * BlockSize;
            const float* FilterTransform1 = OutputBlockCountThisIteration > 1
                ? Filter1 + ic * MLAS_WINOGRAD_TRANSFORM_COUNT * BlockSize + k * BlockSize * BlockSize
                : nullptr;
            const float* FilterTransform2 = OutputBlockCountThisIteration > 2
                ? Filter2 + ic * MLAS_WINOGRAD_TRANSFORM_COUNT * BlockSize + k * BlockSize * BlockSize
                : nullptr;
            const float* FilterTransform3 = OutputBlockCountThisIteration > 3
                ? Filter3 + ic * MLAS_WINOGRAD_TRANSFORM_COUNT * BlockSize + k * BlockSize * BlockSize
                : nullptr;

            for (size_t bi = 0; bi < BlockSize; bi++) {
                const __m512 InputValue = _mm512_set1_ps(InputTransform[bi]);
                Acc0 = _mm512_fmadd_ps(InputValue, _mm512_loadu_ps(FilterTransform0 + bi * BlockSize), Acc0);
                if (OutputBlockCountThisIteration > 1) {
                    Acc1 = _mm512_fmadd_ps(InputValue, _mm512_loadu_ps(FilterTransform1 + bi * BlockSize), Acc1);
                }
                if (OutputBlockCountThisIteration > 2) {
                    Acc2 = _mm512_fmadd_ps(InputValue, _mm512_loadu_ps(FilterTransform2 + bi * BlockSize), Acc2);
                }
                if (OutputBlockCountThisIteration > 3) {
                    Acc3 = _mm512_fmadd_ps(InputValue, _mm512_loadu_ps(FilterTransform3 + bi * BlockSize), Acc3);
                }
            }
        }

        _mm512_storeu_ps(Accumulator0 + k * BlockSize, Acc0);
        if (OutputBlockCountThisIteration > 1) {
            _mm512_storeu_ps(Accumulator1 + k * BlockSize, Acc1);
        }
        if (OutputBlockCountThisIteration > 2) {
            _mm512_storeu_ps(Accumulator2 + k * BlockSize, Acc2);
        }
        if (OutputBlockCountThisIteration > 3) {
            _mm512_storeu_ps(Accumulator3 + k * BlockSize, Acc3);
        }
    }
}
