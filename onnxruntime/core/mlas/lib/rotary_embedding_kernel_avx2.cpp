/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_avx2.cpp

Abstract:

    This module implements the rotary embedding kernels for AVX2 supported h/w.

--*/


#include <cassert>

#include "rotary_embedding.h"
#include "rotary_embedding_kernel_avx2.h"

namespace rope_avx2 {

namespace {

typedef __m256 float32x8_t;

template <bool interleaved>
void
RopeKernel_Avx2_fp16_Impl(
    const MLAS_FP16* input,
    const MLAS_FP16* sin_data,
    const MLAS_FP16* cos_data,
    size_t dim,
    MLAS_FP16* output
);

float32x8_t
load_fp16_and_convert_to_fp32(const MLAS_FP16* input)
{
    __m128i fp16 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(input));
    return _mm256_cvtph_ps(fp16);
}

void
convert_to_fp16_and_store(MLAS_FP16* dst_fp16, const __m256 output)
{
    __m128i fp16_chunk = _mm256_cvtps_ph(output, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_fp16), fp16_chunk);
}

template <>
void
RopeKernel_Avx2_fp16_Impl<false>(
    const MLAS_FP16* input,
    const MLAS_FP16* sin_data,
    const MLAS_FP16* cos_data,
    size_t dim,
    MLAS_FP16* output
)
{
  // ?cast input -> const unsigned short*
    const size_t half_dim = dim >> 1;
    size_t i = 0, j = half_dim;
    for (; i + 7 < half_dim; i += 8, j += 8) {
        float32x8_t real = load_fp16_and_convert_to_fp32(input + i);
        float32x8_t imag = load_fp16_and_convert_to_fp32(input + j);
        float32x8_t sin_val = load_fp16_and_convert_to_fp32(sin_data + i);
        float32x8_t cos_val = load_fp16_and_convert_to_fp32(cos_data + i);
        // Compute Real and Imaginary output values
        float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
        float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));
        // Store back into non interleaved format
        convert_to_fp16_and_store(output + i, real_out);
        convert_to_fp16_and_store(output + j, imag_out);
    }
    for (; i < half_dim; i++, j++) {
        float real = input[i].ToFloat();
        float imag = input[j].ToFloat();
        float sin_val = sin_data[i];
        float cos_val = cos_data[i];
        output[i] = MLAS_FP16(real * cos_val - imag * sin_val);
        output[j] = MLAS_FP16(real * sin_val + imag * cos_val);
    }
}

template <>
void
RopeKernel_Avx2_fp16_Impl<true>(
    const MLAS_FP16* input,
    const MLAS_FP16* sin_data,
    const MLAS_FP16* cos_data,
    size_t dim,
    MLAS_FP16* output
)
{
    // ?cast input -> const unsigned short*
    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        float32x8_t x0 = load_fp16_and_convert_to_fp32(input + i);
        float32x8_t x1 = load_fp16_and_convert_to_fp32(input + i + 8);
        float32x8_t real_s = _mm256_shuffle_ps(x0, x1, 0b10001000);
        float32x8_t imag_s = _mm256_shuffle_ps(x0, x1, 0b11011101);
        __m256i in_mask_vec = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
        float32x8_t real = _mm256_permutevar8x32_ps(real_s, in_mask_vec);
        float32x8_t imag = _mm256_permutevar8x32_ps(imag_s, in_mask_vec);
        float32x8_t sin_val = load_fp16_and_convert_to_fp32(sin_data + i / 2);
        float32x8_t cos_val = load_fp16_and_convert_to_fp32(cos_data + i / 2);
        // Compute Real and Imaginary output values
        float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
        float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));
        // Store back into interleaved format
        __m256i out_mask_vec = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
        float32x8_t real_out_s = _mm256_permutevar8x32_ps(real_out, out_mask_vec);
        float32x8_t imag_out_s = _mm256_permutevar8x32_ps(imag_out, out_mask_vec);
        float32x8_t y0 = _mm256_unpacklo_ps(real_out_s, imag_out_s);
        float32x8_t y1 = _mm256_unpackhi_ps(real_out_s, imag_out_s);

        // Store back into non interleaved format
        convert_to_fp16_and_store(output + i, y0);
        convert_to_fp16_and_store(output + i + 8, y1);
    }

    // Scalar remainder loop to safely handle trailing elements in pairs.
    for (; i + 1 < dim; i += 2) {
        size_t cache_idx = i / 2;
        float input0 = input[i].ToFloat();
        float input1 = input[i + 1].ToFloat();
        float sin_val = sin_data[cache_idx].ToFloat();
        float cos_val = cos_data[cache_idx].ToFloat();
        output[i] = MLAS_FP16(input0 * cos_val - input1 * sin_val);
        output[i + 1] = MLAS_FP16(input0 * sin_val + input1 * cos_val);
    }
}

template <bool interleaved>
void
RopeKernel_Avx2_fp32_Impl(
    const float* input,
    const float* sin_data,
    const float* cos_data,
    size_t dim,
    float* output
);

template <>
void
RopeKernel_Avx2_fp32_Impl<false>(
    const float* input,
    const float* sin_data,
    const float* cos_data,
    size_t dim,
    float* output
) {
    const size_t half_dim = dim >> 1;
    size_t i = 0, j = half_dim;
    for (; i + 7 < half_dim; i += 8, j += 8) {
        float32x8_t real = _mm256_loadu_ps(input + i);
        float32x8_t imag = _mm256_loadu_ps(input + j);
        float32x8_t sin_val = _mm256_loadu_ps(sin_data + i);
        float32x8_t cos_val = _mm256_loadu_ps(cos_data + i);
        //Compute Real and Imaginary output values
        float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
        float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));
        //Store back into non interleaved format
        _mm256_storeu_ps(output + i, real_out);
        _mm256_storeu_ps(output + j, imag_out);
    }

    // Scalar remainder loop to safely handle trailing elements
    for (; i < half_dim; i++, j++) {
        float real = input[i];
        float imag = input[j];
        float sin_val = sin_data[i];
        float cos_val = cos_data[i];
        output[i] = real * cos_val - imag * sin_val;
        output[j] = real * sin_val + imag * cos_val;
    }
}

template <>
void
RopeKernel_Avx2_fp32_Impl<true>(
    const float* input,
    const float* sin_data,
    const float* cos_data,
    size_t dim,
    float* output
) {
    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        float32x8_t x0 = _mm256_loadu_ps(input + i);
        float32x8_t x1 = _mm256_loadu_ps(input + i + 8);
        //Load imaginary and real values to separate non-interleaved vectors
        float32x8_t real_s = _mm256_shuffle_ps(x0, x1, 0b10001000);
        float32x8_t imag_s = _mm256_shuffle_ps(x0, x1, 0b11011101);
        __m256i in_mask_vec = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
        float32x8_t real = _mm256_permutevar8x32_ps(real_s, in_mask_vec);
        float32x8_t imag = _mm256_permutevar8x32_ps(imag_s, in_mask_vec);
        float32x8_t sin_val = _mm256_loadu_ps(sin_data + i / 2);
        float32x8_t cos_val = _mm256_loadu_ps(cos_data + i / 2);
        //Compute Real and Imaginary output values
        float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
        float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));
        //Store back into interleaved format
        __m256i out_mask_vec = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
        float32x8_t real_out_s = _mm256_permutevar8x32_ps(real_out, out_mask_vec);
        float32x8_t imag_out_s = _mm256_permutevar8x32_ps(imag_out, out_mask_vec);
        float32x8_t y0 = _mm256_unpacklo_ps(real_out_s, imag_out_s);
        float32x8_t y1 = _mm256_unpackhi_ps(real_out_s, imag_out_s);
        _mm256_storeu_ps(output + i, y0);
        _mm256_storeu_ps(output + i + 8, y1);
    }

    // Scalar remainder loop to safely handle trailing elements in pairs
    for (; i + 1 < dim; i += 2) {
        size_t cache_idx = i / 2;
        float input0 = input[i];
        float input1 = input[i + 1];
        float sin_val = sin_data[cache_idx];
        float cos_val = cos_data[cache_idx];
        output[i]     = input0 * cos_val - input1 * sin_val;
        output[i + 1] = input0 * sin_val + input1 * cos_val;
    }
}

}  // rope_avx2 namespace

void
RopeKernel_Avx2_fp32(
    const float* input,
    const float* sin_data,
    const float* cos_data,
    size_t dim,
    bool interleaved,
    float* output
) {
    // real part and imaginary part must be paired
    assert(dim % 2 == 0);
    const auto* input_impl = reinterpret_cast<const float*>(input);
    const auto* sin_impl = reinterpret_cast<const float*>(sin_data);
    const auto* cos_impl = reinterpret_cast<const float*>(cos_data);
    auto* output_impl = reinterpret_cast<float*>(output);

    if (interleaved) {
        RopeKernel_Avx2_fp32_Impl<true>(input_impl, sin_impl, cos_impl, dim, output_impl);
    } else {
        RopeKernel_Avx2_fp32_Impl<false>(input_impl, sin_impl, cos_impl, dim, output_impl);
    }
}

void
RopeKernel_Avx2_fp16(
    const MLAS_FP16* input,
    const MLAS_FP16* sin_data,
    const MLAS_FP16* cos_data,
    size_t dim,
    bool interleaved,
    MLAS_FP16* output
)
{
    // real part and imaginary part must be paired
    assert(dim % 2 == 0);

    if (interleaved) {
        RopeKernel_Avx2_fp16_Impl<true>(input, sin_data, cos_data, dim, output);
    } else {
        RopeKernel_Avx2_fp16_Impl<false>(input, sin_data, cos_data, dim, output);
    }
}
}

//
// Kernel dispatch structure definition.
//
const MLAS_ROPE_DISPATCH MlasRopeDispatchAvx2 = []() {
    MLAS_ROPE_DISPATCH d;
    d.SRope = rope_avx2::RopeKernel_Avx2_fp32;
    d.HRope = rope_avx2::RopeKernel_Avx2_fp16;
    return d;
}();
