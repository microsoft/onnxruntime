/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_lut_kernel_avx2.h

Abstract:

    This module implements x64 AVX2 kernel functions for LUT-based n-bit
    quantized integer matrix multiplication.
--*/

#pragma once
#include "qnbitgemm.h"

void
GenerateLUT_avx2(
    int32_t group_size,
    int8_t lut,
    const float* b,
    float* scales,
    float* biases,
    int K
);

void
TMACComputeGemm_avx2(
    const void* A,
    const void* a_scales,
    const void* LUT,
    const void* LUT_Scales,
    const void* LUT_Biases,
    void* C,
    int bm,
    int K,
    int M,
    int N,
    size_t BlkLen
);
