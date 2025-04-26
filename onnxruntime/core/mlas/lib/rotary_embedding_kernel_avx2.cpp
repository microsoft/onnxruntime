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
static constexpr int32_t mask_buffer[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

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

    for (; i < dim; i++) {
        size_t cache_idx = i / 2;
        bool sign = i & 1;
        size_t j = sign ? i - 1 : i + 1;

        float output_data_i = input[i].ToFloat() * cos_data[cache_idx].ToFloat();
        float input_data_j = input[j].ToFloat();
        float sin_data_cache_idx = sin_data[cache_idx].ToFloat();
        if (sign) {
            output_data_i += input_data_j * sin_data_cache_idx;
        } else {
            output_data_i -= input_data_j * sin_data_cache_idx;
        }
        output[i] = MLAS_FP16(output_data_i);
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
    if (half_dim - i != 0) {
        size_t rem = half_dim - i;
        const __m256i mask = _mm256_loadu_si256((const __m256i*)(mask_buffer + 8 - rem));
        //Use a mask to load the remaining input values
        float32x8_t real = _mm256_maskload_ps(input + i, mask);
        float32x8_t imag = _mm256_maskload_ps(input + j, mask);
        float32x8_t sin_val = _mm256_maskload_ps(sin_data + i, mask);
        float32x8_t cos_val = _mm256_maskload_ps(cos_data + i, mask);
        //Compute Real and Imaginary output values
        float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
        float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));
        //Store back into non interleaved format
        _mm256_maskstore_ps(output + i, mask, real_out);
        _mm256_maskstore_ps(output + j, mask, imag_out);
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
    if (dim - i != 0) {
        size_t rem = dim - i;
        const __m256i mask0 = _mm256_loadu_si256((const __m256i*)(mask_buffer + 8 - (rem>8?8:rem)));
        const __m256i mask1 = _mm256_loadu_si256((const __m256i*)(mask_buffer + 8 - (rem>8?(rem-8):0)));
        float32x8_t x0 = _mm256_maskload_ps(input + i, mask0);   //Load the first set of data using mask
        float32x8_t x1 = _mm256_maskload_ps(input + i + 8, mask1); //Load the reminder of data using a second mask
        //Load imaginary and real values to separate non-interleaved vectors
        float32x8_t real_s = _mm256_shuffle_ps(x0, x1, 0b10001000);
        float32x8_t imag_s = _mm256_shuffle_ps(x0, x1, 0b11011101);
        __m256i in_mask_vec = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
        float32x8_t real = _mm256_permutevar8x32_ps(real_s, in_mask_vec);
        float32x8_t imag = _mm256_permutevar8x32_ps(imag_s, in_mask_vec);
        float32x8_t sin_val = _mm256_loadu_ps(sin_data+ i / 2);
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
        _mm256_maskstore_ps(output + i, mask0, y0);
        _mm256_maskstore_ps(output + i + 8, mask1, y1);
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
