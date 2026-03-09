/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    snchwc_winograd.h

Abstract:

    This module declares Winograd helper routines for NCHWc convolution.

--*/

#pragma once

#include <cstddef>

inline constexpr size_t MLAS_WINOGRAD_TILE_SIZE = 2;
inline constexpr size_t MLAS_WINOGRAD_TRANSFORM_SIZE = 4;
inline constexpr size_t MLAS_WINOGRAD_TRANSFORM_COUNT = MLAS_WINOGRAD_TRANSFORM_SIZE * MLAS_WINOGRAD_TRANSFORM_SIZE;

bool
MlasNchwcShouldPreferWinograd3x3(
    size_t OutputHeight,
    size_t OutputWidth
    );

bool
MlasNchwcIsWinograd3x3Supported(
    bool UseWinograd,
    size_t GroupCount,
    const size_t KernelShape[2],
    const size_t DilationShape[2],
    const size_t Padding[4],
    const size_t StrideShape[2],
    size_t InputChannels,
    size_t BlockSize,
    size_t OutputHeight,
    size_t OutputWidth
    );

float*
MlasWinogradGetThreadedScratchBuffer(
    size_t FloatCount
    );

void
MlasWinogradTransformInputF2x2_3x3(
    const float D[4][4],
    float V[4][4]
    );

void
MlasWinogradTransformOutputF2x2_3x3(
    const float M[4][4],
    float Y[2][2]
    );

#if defined(MLAS_TARGET_AMD64) && (defined(_MSC_VER) || defined(__AVX512F__))
void
MlasWinogradTransformInputBlockF2x2_3x3Avx512(
    const float* InputChannelBlock,
    size_t InputHeight,
    size_t InputWidth,
    size_t TileOutputY,
    size_t TileOutputX,
    float* TransformedInputBlock
    );

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
    );

void
MlasWinogradAccumulateOutputBlocksAvx512(
    size_t InputChannels,
    const float* Filter0,
    const float* Filter1,
    const float* TransformedInput,
    float* Accumulator0,
    float* Accumulator1,
    size_t OutputBlockCountThisIteration
    );
#endif

void
MlasWinogradAccumulateOutputBlockScalar(
    size_t BlockSize,
    size_t InputChannels,
    const float* Filter,
    const float* TransformedInput,
    float* Accumulator
    );
