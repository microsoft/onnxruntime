/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_avx2_fp32.cpp

Abstract:

    This module implements the fp32 rotary embedding kernels using AVX2.

--*/

#include <cassert>

#include "rotary_embedding.h"
#include "rotary_embedding_kernel_avx2.h"

namespace rope_avx2 {

namespace {

typedef __m256 float32x8_t;

template <bool interleaved>
void
RopeKernel_Avx2_Impl(
    const float* input,
    const float* sin_data,
    const float* cos_data,
    size_t dim,
    float* output
);

template <>
void
RopeKernel_Avx2_Impl<false>(
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
        static constexpr int32_t mask_buffer[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
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
RopeKernel_Avx2_Impl<true>(
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
        static constexpr int32_t mask_buffer[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
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

}  // namespace

void
RopeKernel_Avx2(
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
        RopeKernel_Avx2_Impl<true>(input_impl, sin_impl, cos_impl, dim, output_impl);
    } else {
        RopeKernel_Avx2_Impl<false>(input_impl, sin_impl, cos_impl, dim, output_impl);
    }
}

}
