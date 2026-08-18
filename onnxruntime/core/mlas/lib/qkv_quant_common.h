/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qkv_quant_common.h

Abstract:

    Shared inline helpers for the quantized KV-cache GEMM kernel implementations.
    These utilities are used by the scalar reference (qkv_quant.cpp) and
    all SIMD-optimized backends (AVX2, AVX512-VNNI, NEON).

--*/

#pragma once

#include "mlas_qkv_quant.h"

namespace MlasKVQuantInternal {

constexpr int kInt4Bias = 8;

inline bool
IsInt4Mode(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::S4_PerTensor ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel;
}

inline bool
IsPerChannelMode(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::S8_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel;
}

}  // namespace MlasKVQuantInternal
