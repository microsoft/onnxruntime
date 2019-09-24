// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>

#ifdef NUPHAR_USE_MKL
// Need to build with USE_MKLML
#include <mkl_cblas.h>
#endif  // NUPHAR_USE_MKL

namespace onnxruntime {
#ifdef NUPHAR_USE_MKL
void MKLIntGemvS16S16S32R(
    int16_t* matrixA,
    int16_t* matrixB,
    int M,
    int N,
    int K,
    int32_t* output);

void MKLIntGemvS8U8S32R(
    int8_t* matrixA,
    uint8_t* matrixB,
    int M,
    int N,
    int K,
    int32_t* output);
#endif
}  // namespace onnxruntime
