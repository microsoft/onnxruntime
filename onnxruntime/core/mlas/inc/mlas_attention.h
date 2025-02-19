/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_attention.h

Abstract:

    This module contains the public data structures and procedure prototypes
    for attention related ops


--*/

#pragma once


#include "mlas.h"
#include "mlas_gemm_postprocessor.h"

template <typename T>
void MLASCALL
MlasRotaryEmbedOneRow_FallBack(
    const T* input_data,
    const T* sin_data,
    const T* cos_data,
    size_t rotary_emb_dim,
    bool interleaved,
    T* output_data
);

template <typename T>
void MLASCALL
MlasRotaryEmbedOneRow(
    const float* input,
    const float* sin,
    const float* cos,
    size_t dim,
    bool interleaved,
    float* output
);
