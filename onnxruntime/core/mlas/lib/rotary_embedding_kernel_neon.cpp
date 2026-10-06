/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_neon.cpp

Abstract:

    This module implements the rotary embedding kernels for ARM NEON.

--*/

#include <cassert>

#include "rotary_embedding.h"
#include "rotary_embedding_kernel_neon.h"

namespace {

template <bool interleaved>
void
RopeKernel_Fp32_Impl(
    const float* input,
    const float* sin,
    const float* cos,
    size_t dim,
    float* output
);

template <>
void
RopeKernel_Fp32_Impl<false>(
    const float* input,
    const float* sin,
    const float* cos,
    size_t dim,
    float* output
) {
    const size_t half_dim = dim >> 1;
    size_t i = 0, j = half_dim;

    for (; i + 3 < half_dim; i += 4, j += 4) {
        float32x4_t real = vld1q_f32(input + i);
        float32x4_t imag = vld1q_f32(input + j);
        float32x4_t sin_val = vld1q_f32(sin + i);
        float32x4_t cos_val = vld1q_f32(cos + i);

        float32x4_t real_out = vfmsq_f32(vmulq_f32(real, cos_val), imag, sin_val);
        float32x4_t imag_out = vfmaq_f32(vmulq_f32(real, sin_val), imag, cos_val);

        vst1q_f32(output + i, real_out);
        vst1q_f32(output + j, imag_out);
    }

    for (; i < half_dim; i++, j++) {
        float real = input[i];
        float imag = input[j];
        float sin_val = sin[i];
        float cos_val = cos[i];
        output[i] = real * cos_val - imag * sin_val;
        output[j] = real * sin_val + imag * cos_val;
    }
}

template <>
void
RopeKernel_Fp32_Impl<true>(
    const float* input,
    const float* sin,
    const float* cos,
    size_t dim,
    float* output
) {
    size_t i = 0;

    for (; i + 7 < dim; i += 8) {
        float32x4x2_t v = vld2q_f32(input + i);
        float32x4_t real = v.val[0];
        float32x4_t imag = v.val[1];

        float32x4_t sin_val = vld1q_f32(sin + i / 2);
        float32x4_t cos_val = vld1q_f32(cos + i / 2);

        float32x4_t real_out = vfmsq_f32(vmulq_f32(real, cos_val), imag, sin_val);
        float32x4_t imag_out = vfmaq_f32(vmulq_f32(real, sin_val), imag, cos_val);

        float32x4x2_t out;
        out.val[0] = real_out;
        out.val[1] = imag_out;
        vst2q_f32(output + i, out);
    }

    for (; i + 1 < dim; i += 2) {
        size_t cache_idx = i / 2;
        float in0 = input[i];
        float in1 = input[i + 1];
        float sin_val = sin[cache_idx];
        float cos_val = cos[cache_idx];
        output[i] = in0 * cos_val - in1 * sin_val;
        output[i + 1] = in0 * sin_val + in1 * cos_val;
    }
}

}  // namespace

namespace rope_neon {

void
RopeKernel_Fp32(
    const float* input,
    const float* sin,
    const float* cos,
    size_t dim,
    bool interleaved,
    float* output
) {
    assert(dim % 2 == 0);
    if (interleaved) {
        RopeKernel_Fp32_Impl<true>(input, sin, cos, dim, output);
    } else {
        RopeKernel_Fp32_Impl<false>(input, sin, cos, dim, output);
    }
}

}  // namespace rope_neon

//
// Kernel dispatch structure definition.
//
const MLAS_ROPE_DISPATCH MlasRopeDispatchNeon = []() {
    MLAS_ROPE_DISPATCH d;

    d.SRope = rope_neon::RopeKernel_Fp32;

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    if (MlasFp16AccelerationSupported()) {
        d.HRope = rope_neon::RopeKernel_Fp16;
    }
#endif
    return d;
}();
