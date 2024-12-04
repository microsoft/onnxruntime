/*++

Copyright (c) Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding.cpp

Abstract:

    This module implements rotary embedding kernels for fp32/16.

--*/

#include "rotary_embedding.h"

namespace {

template <typename T>
void
MLASCALL
MlasRotaryEmbedOneRow_FallBack(
    const T* input,
    const T* sin,
    const T* cos,
    size_t dim,
    bool interleaved,
    T* output
) {

}

}  // namespace


template <>
void
MLASCALL
MlasRotaryEmbedOneRow<float>(
    const float* input,
    const float* sin,
    const float* cos,
    size_t dim,
    bool interleaved,
    float* output
) {
    const auto* dispatch = GetMlasPlatform().RopeDispatch;

    if (dispatch == nullptr || dispatch->SRope == nullptr) {
        MlasRotaryEmbedOneRow_FallBack(input, sin, cos, dim, interleaved, output);
        return;
    }

    dispatch->SRope(input, sin, cos, dim, interleaved, output);
}

template <>
void
MLASCALL
MlasRotaryEmbedOneRow<MLAS_FP16>(
    const MLAS_FP16* input,
    const MLAS_FP16* sin,
    const MLAS_FP16* cos,
    size_t dim,
    bool interleaved,
    MLAS_FP16* output
) {
    const auto* dispatch = GetMlasPlatform().RopeDispatch;

    if (dispatch == nullptr || dispatch->HRope == nullptr) {
        MlasRotaryEmbedOneRow_FallBack(input, sin, cos, dim, interleaved, output);
        return;
    }

    dispatch->HRope(input, sin, cos, dim, interleaved, output);
}
