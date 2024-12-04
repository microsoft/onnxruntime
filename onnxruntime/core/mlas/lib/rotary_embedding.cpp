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
    const T* input_data,
    const T* sin_data,
    const T* cos_data,
    size_t rotary_emb_dim,
    bool interleaved,
    T* output_data
) {
    const size_t half_rotary_emb_dim = rotary_emb_dim / 2;
    size_t cache_idx = 0;
    bool sign = false;
    size_t j = 0;
    for (size_t i = 0; i < rotary_emb_dim; i++) {
        if (interleaved) {
            cache_idx = (i / 2) % half_rotary_emb_dim;
            sign = i & 1;
            j = sign ? i - 1 : i + 1;  // i - sign
        } else {
            cache_idx = i % half_rotary_emb_dim;
            sign = (i >= half_rotary_emb_dim);
            j = (i + half_rotary_emb_dim) % rotary_emb_dim;
        }
        float output_data_i = static_cast<float>(input_data[i]) * static_cast<float>(cos_data[cache_idx]);
        float input_data_j = static_cast<float>(input_data[j]);
        float sin_data_cache_idx = static_cast<float>(sin_data[cache_idx]);
        if (sign) {
            output_data_i += input_data_j * sin_data_cache_idx;
        } else {
            output_data_i -= input_data_j * sin_data_cache_idx;
        }
        output_data[i] = static_cast<T>(output_data_i);
    }
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
        MlasRotaryEmbedOneRow_FallBack<float>(input, sin, cos, dim, interleaved, output);
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
        MlasRotaryEmbedOneRow_FallBack<MLAS_FP16>(input, sin, cos, dim, interleaved, output);
        return;
    }

    dispatch->HRope(input, sin, cos, dim, interleaved, output);
}
