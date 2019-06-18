// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>


#include <immintrin.h>
#include <inttypes.h>


namespace onnxruntime {

void AVX2IntGemvS8U8S32R(
    int8_t* matrix,
    uint8_t* vec,
    int matrixRowDimension,
    int paddedRowDimension,
    int matrixColumnDimension,
    int32_t* output);

void AVX2IntGemvS16S16S32R(
    int16_t* matrix,
    int16_t* vec,
    int matrixRowDimension,
    int paddedRowDimension,
    int matrixColumnDimension,
    int32_t* output);

void AVX2IntGemvS8U8S32REx(
    int8_t* matrix,
    uint8_t* vec,
    int matrixRowDimension,
    int matrixColumnDimension,
    int32_t* output);

void AVX2IntGemvS16S16S32REx(
    int16_t* matrix,
    int16_t* vec,
    int matrixRowDimension,
    int matrixColumnDimension,
    int32_t* output);

}  // namespace onnxruntime
