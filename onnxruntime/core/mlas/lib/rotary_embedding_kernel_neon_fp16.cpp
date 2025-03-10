/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_neon_fp16.cpp

Abstract:

    This module implements the fp16 rotary embedding kernels for ARM NEON.

--*/

#include <arm_neon.h>
#include <cassert>

#include "fp16_common.h"
#include "rotary_embedding.h"
#include "rotary_embedding_kernel_neon.h"

namespace rope_neon {

namespace {

template <bool interleaved>
void
RopeKernel_Fp16_Impl(
    const _mlas_fp16_* input,
    const _mlas_fp16_* sin,
    const _mlas_fp16_* cos,
    size_t dim,
    _mlas_fp16_* output
);

template <>
void
RopeKernel_Fp16_Impl<false>(
    const _mlas_fp16_* input,
    const _mlas_fp16_* sin,
    const _mlas_fp16_* cos,
    size_t dim,
    _mlas_fp16_* output
) {
    const size_t half_dim = dim >> 1;
    size_t i = 0, j = half_dim;
    for (; i + 7 < half_dim; i += 8, j += 8) {
        float16x8_t real = MlasLoadFloat16x8(input + i);
        float16x8_t imag = MlasLoadFloat16x8(input + j);
        float16x8_t sin_val = MlasLoadFloat16x8(sin + i);
        float16x8_t cos_val = MlasLoadFloat16x8(cos + i);
        float16x8_t real_out = vfmsq_f16(vmulq_f16(real, cos_val), imag, sin_val);
        float16x8_t imag_out = vfmaq_f16(vmulq_f16(real, sin_val), imag, cos_val);
        MlasStoreFloat16x8(output + i, real_out);
        MlasStoreFloat16x8(output + j, imag_out);
    }
    for (; i + 3 < half_dim; i += 4, j += 4) {
        float16x4_t real = MlasLoadFloat16x4(input + i);
        float16x4_t imag = MlasLoadFloat16x4(input + j);
        float16x4_t sin_val = MlasLoadFloat16x4(sin + i);
        float16x4_t cos_val = MlasLoadFloat16x4(cos + i);
        float16x4_t real_out = vfms_f16(vmul_f16(real, cos_val), imag, sin_val);
        float16x4_t imag_out = vfma_f16(vmul_f16(real, sin_val), imag, cos_val);
        MlasStoreFloat16x4(output + i, real_out);
        MlasStoreFloat16x4(output + j, imag_out);
    }
    if (half_dim - i == 3) {
        float16x4_t real = MlasZeroFloat16x4();
        float16x4_t imag = MlasZeroFloat16x4();
        float16x4_t sin_val = MlasZeroFloat16x4();
        float16x4_t cos_val = MlasZeroFloat16x4();
        real = MlasLoadLaneFloat16x4<0>(input + i, real);
        real = MlasLoadLaneFloat16x4<1>(input + i + 1, real);
        real = MlasLoadLaneFloat16x4<2>(input + i + 2, real);
        imag = MlasLoadLaneFloat16x4<0>(input + j, imag);
        imag = MlasLoadLaneFloat16x4<1>(input + j + 1, imag);
        imag = MlasLoadLaneFloat16x4<2>(input + j + 2, imag);
        sin_val = MlasLoadLaneFloat16x4<0>(sin + i, sin_val);
        sin_val = MlasLoadLaneFloat16x4<1>(sin + i + 1, sin_val);
        sin_val = MlasLoadLaneFloat16x4<2>(sin + i + 2, sin_val);
        cos_val = MlasLoadLaneFloat16x4<0>(cos + i, cos_val);
        cos_val = MlasLoadLaneFloat16x4<1>(cos + i + 1, cos_val);
        cos_val = MlasLoadLaneFloat16x4<2>(cos + i + 2, cos_val);
        float16x4_t real_out = vfms_f16(vmul_f16(real, cos_val), imag, sin_val);
        float16x4_t imag_out = vfma_f16(vmul_f16(real, sin_val), imag, cos_val);
        MlasStoreLaneFloat16x4<0>(output + i, real_out);
        MlasStoreLaneFloat16x4<1>(output + i + 1, real_out);
        MlasStoreLaneFloat16x4<2>(output + i + 2, real_out);
        MlasStoreLaneFloat16x4<0>(output + j, imag_out);
        MlasStoreLaneFloat16x4<1>(output + j + 1, imag_out);
        MlasStoreLaneFloat16x4<2>(output + j + 2, imag_out);
    } else if (half_dim - i == 2) {
        float16x4_t real = MlasZeroFloat16x4();
        float16x4_t imag = MlasZeroFloat16x4();
        float16x4_t sin_val = MlasZeroFloat16x4();
        float16x4_t cos_val = MlasZeroFloat16x4();
        real = MlasLoadLaneFloat16x4<0>(input + i, real);
        real = MlasLoadLaneFloat16x4<1>(input + i + 1, real);
        imag = MlasLoadLaneFloat16x4<0>(input + j, imag);
        imag = MlasLoadLaneFloat16x4<1>(input + j + 1, imag);
        sin_val = MlasLoadLaneFloat16x4<0>(sin + i, sin_val);
        sin_val = MlasLoadLaneFloat16x4<1>(sin + i + 1, sin_val);
        cos_val = MlasLoadLaneFloat16x4<0>(cos + i, cos_val);
        cos_val = MlasLoadLaneFloat16x4<1>(cos + i + 1, cos_val);
        float16x4_t real_out = vfms_f16(vmul_f16(real, cos_val), imag, sin_val);
        float16x4_t imag_out = vfma_f16(vmul_f16(real, sin_val), imag, cos_val);
        MlasStoreLaneFloat16x4<0>(output + i, real_out);
        MlasStoreLaneFloat16x4<1>(output + i + 1, real_out);
        MlasStoreLaneFloat16x4<0>(output + j, imag_out);
        MlasStoreLaneFloat16x4<1>(output + j + 1, imag_out);
    } else if (half_dim - i == 1) {
        float16x4_t real = MlasZeroFloat16x4();
        float16x4_t imag = MlasZeroFloat16x4();
        float16x4_t sin_val = MlasZeroFloat16x4();
        float16x4_t cos_val = MlasZeroFloat16x4();
        real = MlasLoadLaneFloat16x4<0>(input + i, real);
        imag = MlasLoadLaneFloat16x4<0>(input + j, imag);
        sin_val = MlasLoadLaneFloat16x4<0>(sin + i, sin_val);
        cos_val = MlasLoadLaneFloat16x4<0>(cos + i, cos_val);
        float16x4_t real_out = vfms_f16(vmul_f16(real, cos_val), imag, sin_val);
        float16x4_t imag_out = vfma_f16(vmul_f16(real, sin_val), imag, cos_val);
        MlasStoreLaneFloat16x4<0>(output + i, real_out);
        MlasStoreLaneFloat16x4<0>(output + j, imag_out);
    }
}

template <>
void
RopeKernel_Fp16_Impl<true>(
    const _mlas_fp16_* input,
    const _mlas_fp16_* sin,
    const _mlas_fp16_* cos,
    size_t dim,
    _mlas_fp16_* output
) {
    size_t i = 0;
    for (; i + 15 < dim; i += 16) {
        float16x8_t x0 = MlasLoadFloat16x8(input + i);
        float16x8_t x1 = MlasLoadFloat16x8(input + i + 8);
        float16x8_t real = vuzp1q_f16(x0, x1);
        float16x8_t imag = vuzp2q_f16(x0, x1);
        float16x8_t sin_val = MlasLoadFloat16x8(sin + i);
        float16x8_t cos_val = MlasLoadFloat16x8(cos + i);
        float16x8_t real_out = vfmsq_f16(vmulq_f16(real, cos_val), imag, sin_val);
        float16x8_t imag_out = vfmaq_f16(vmulq_f16(real, sin_val), imag, cos_val);
        float16x8_t y0 = vzip1q_f16(real_out, imag_out);
        float16x8_t y1 = vzip2q_f16(real_out, imag_out);
        MlasStoreFloat16x8(output + i, y0);
        MlasStoreFloat16x8(output + i + 8, y1);
    }
    for (; i + 7 < dim; i += 8) {
        float16x4_t x0 = MlasLoadFloat16x4(input + i);
        float16x4_t x1 = MlasLoadFloat16x4(input + i + 4);
        float16x4_t real = vuzp1_f16(x0, x1);
        float16x4_t imag = vuzp2_f16(x0, x1);
        float16x4_t sin_val = MlasLoadFloat16x4(sin + i);
        float16x4_t cos_val = MlasLoadFloat16x4(cos + i);
        float16x4_t real_out = vfms_f16(vmul_f16(real, cos_val), imag, sin_val);
        float16x4_t imag_out = vfma_f16(vmul_f16(real, sin_val), imag, cos_val);
        float16x4_t y0 = vzip1_f16(real_out, imag_out);
        float16x4_t y1 = vzip2_f16(real_out, imag_out);
        MlasStoreFloat16x4(output + i, y0);
        MlasStoreFloat16x4(output + i + 4, y1);
    }
    if (dim - i == 6) {
        float16x4_t real = MlasZeroFloat16x4();
        float16x4_t imag = MlasZeroFloat16x4();
        float16x4_t sin_val = MlasZeroFloat16x4();
        float16x4_t cos_val = MlasZeroFloat16x4();
        real = MlasLoadLaneFloat16x4<0>(input + i, real);
        imag = MlasLoadLaneFloat16x4<0>(input + i + 1, imag);
        real = MlasLoadLaneFloat16x4<1>(input + i + 2, real);
        imag = MlasLoadLaneFloat16x4<1>(input + i + 3, imag);
        real = MlasLoadLaneFloat16x4<2>(input + i + 4, real);
        imag = MlasLoadLaneFloat16x4<2>(input + i + 5, imag);
        sin_val = MlasLoadLaneFloat16x4<0>(sin + i, sin_val);
        sin_val = MlasLoadLaneFloat16x4<1>(sin + i + 1, sin_val);
        sin_val = MlasLoadLaneFloat16x4<2>(sin + i + 2, sin_val);
        cos_val = MlasLoadLaneFloat16x4<0>(cos + i, cos_val);
        cos_val = MlasLoadLaneFloat16x4<1>(cos + i + 1, cos_val);
        cos_val = MlasLoadLaneFloat16x4<2>(cos + i + 2, cos_val);
        float16x4_t real_out = vfms_f16(vmul_f16(real, cos_val), imag, sin_val);
        float16x4_t imag_out = vfma_f16(vmul_f16(real, sin_val), imag, cos_val);
        MlasStoreLaneFloat16x4<0>(output + i, real_out);
        MlasStoreLaneFloat16x4<0>(output + i + 1, imag_out);
        MlasStoreLaneFloat16x4<1>(output + i + 2, real_out);
        MlasStoreLaneFloat16x4<1>(output + i + 3, imag_out);
        MlasStoreLaneFloat16x4<2>(output + i + 4, real_out);
        MlasStoreLaneFloat16x4<2>(output + i + 5, imag_out);
    } else if (dim - i == 4) {
        float16x4_t real = MlasZeroFloat16x4();
        float16x4_t imag = MlasZeroFloat16x4();
        float16x4_t sin_val = MlasZeroFloat16x4();
        float16x4_t cos_val = MlasZeroFloat16x4();
        real = MlasLoadLaneFloat16x4<0>(input + i, real);
        imag = MlasLoadLaneFloat16x4<0>(input + i + 1, imag);
        real = MlasLoadLaneFloat16x4<1>(input + i + 2, real);
        imag = MlasLoadLaneFloat16x4<1>(input + i + 3, imag);
        sin_val = MlasLoadLaneFloat16x4<0>(sin + i, sin_val);
        sin_val = MlasLoadLaneFloat16x4<1>(sin + i + 1, sin_val);
        cos_val = MlasLoadLaneFloat16x4<0>(cos + i, cos_val);
        cos_val = MlasLoadLaneFloat16x4<1>(cos + i + 1, cos_val);
        float16x4_t real_out = vfms_f16(vmul_f16(real, cos_val), imag, sin_val);
        float16x4_t imag_out = vfma_f16(vmul_f16(real, sin_val), imag, cos_val);
        MlasStoreLaneFloat16x4<0>(output + i, real_out);
        MlasStoreLaneFloat16x4<0>(output + i + 1, imag_out);
        MlasStoreLaneFloat16x4<1>(output + i + 2, real_out);
        MlasStoreLaneFloat16x4<1>(output + i + 3, imag_out);
    } else if (dim - i == 2) {
        float16x4_t real = MlasZeroFloat16x4();
        float16x4_t imag = MlasZeroFloat16x4();
        float16x4_t sin_val = MlasZeroFloat16x4();
        float16x4_t cos_val = MlasZeroFloat16x4();
        real = MlasLoadLaneFloat16x4<0>(input + i, real);
        imag = MlasLoadLaneFloat16x4<0>(input + i + 1, imag);
        sin_val = MlasLoadLaneFloat16x4<0>(sin + i, sin_val);
        cos_val = MlasLoadLaneFloat16x4<0>(cos + i, cos_val);
        float16x4_t real_out = vfms_f16(vmul_f16(real, cos_val), imag, sin_val);
        float16x4_t imag_out = vfma_f16(vmul_f16(real, sin_val), imag, cos_val);
        MlasStoreLaneFloat16x4<0>(output + i, real_out);
        MlasStoreLaneFloat16x4<0>(output + i + 1, imag_out);
    }
}

}  // namespace

void
RopeKernel_Fp16(
    const MLAS_FP16* input,
    const MLAS_FP16* sin,
    const MLAS_FP16* cos,
    size_t dim,
    bool interleaved,
    MLAS_FP16* output
) {
    // real part and imaginary part must be paired
    assert(dim % 2 == 0);

    const auto* input_impl = reinterpret_cast<const _mlas_fp16_*>(input);
    const auto* sin_impl = reinterpret_cast<const _mlas_fp16_*>(sin);
    const auto* cos_impl = reinterpret_cast<const _mlas_fp16_*>(cos);
    auto* output_impl = reinterpret_cast<_mlas_fp16_*>(output);

    if (interleaved) {
        RopeKernel_Fp16_Impl<true>(input_impl, sin_impl, cos_impl, dim, output_impl);
    } else {
        RopeKernel_Fp16_Impl<false>(input_impl, sin_impl, cos_impl, dim, output_impl);
    }
}

}  // namespace rope_neon
