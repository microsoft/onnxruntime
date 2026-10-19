//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>

namespace ArmKleidiAI
{

constexpr size_t kAlphaBetaNeonMinElements = 32;

inline void
MlasGemmApplyAlphaBetaStrided(
    const float* product, size_t count, float alpha, float beta, float* output, size_t output_stride, bool allow_memcpy
)
{
    if (count == 0) {
        return;
    }
    if (alpha == 1.0f && beta == 0.0f && allow_memcpy && output_stride == 1) {
        std::memcpy(output, product, count * sizeof(float));
        return;
    }

    size_t index = 0;
    if (output_stride == 1 && count >= kAlphaBetaNeonMinElements) {
        const float32x4_t alpha_vector = vdupq_n_f32(alpha);
        const float32x4_t beta_vector = vdupq_n_f32(beta);
        if (alpha == 1.0f && beta == 0.0f) {
            for (; index + 4 <= count; index += 4) {
                vst1q_f32(output + index, vld1q_f32(product + index));
            }
        } else if (alpha == 1.0f) {
            for (; index + 4 <= count; index += 4) {
                vst1q_f32(
                    output + index, vmlaq_f32(vld1q_f32(product + index), beta_vector, vld1q_f32(output + index))
                );
            }
        } else if (beta == 0.0f) {
            for (; index + 4 <= count; index += 4) {
                vst1q_f32(output + index, vmulq_f32(alpha_vector, vld1q_f32(product + index)));
            }
        } else {
            for (; index + 4 <= count; index += 4) {
                vst1q_f32(
                    output + index,
                    vmlaq_f32(
                        vmulq_f32(alpha_vector, vld1q_f32(product + index)), beta_vector, vld1q_f32(output + index)
                    )
                );
            }
        }
    }
    for (; index < count; ++index) {
        float value = alpha == 1.0f ? product[index] : alpha * product[index];
        if (beta != 0.0f) {
            value += beta * output[index * output_stride];
        }
        output[index * output_stride] = value;
    }
}

inline void
MlasGemmApplyAlphaBeta(
    const float* product,
    size_t rows,
    size_t columns,
    float alpha,
    float beta,
    float* output,
    size_t output_ldc,
    bool flatten_contiguous = true
)
{
    if (rows == 0 || columns == 0) {
        return;
    }
    // Preserve each adapter's SIMD/tail boundaries: legacy flattens contiguous
    // tiles, while Ops applies scaling row by row.
    if (flatten_contiguous && output_ldc == columns && rows <= std::numeric_limits<size_t>::max() / columns) {
        MlasGemmApplyAlphaBetaStrided(product, rows * columns, alpha, beta, output, 1, true);
        return;
    }
    for (size_t row = 0; row < rows; ++row) {
        MlasGemmApplyAlphaBetaStrided(
            product + row * columns, columns, alpha, beta, output + row * output_ldc, 1, true
        );
    }
}

inline void
MlasGemmApplyBeta(float* output, size_t rows, size_t columns, size_t output_ldc, float beta)
{
    if (rows == 0 || columns == 0 || beta == 1.0f) {
        return;
    }
    for (size_t row = 0; row < rows; ++row) {
        float* destination = output + row * output_ldc;
        if (beta == 0.0f) {
            std::fill_n(destination, columns, 0.0f);
            continue;
        }
        const float32x4_t beta_vector = vdupq_n_f32(beta);
        size_t column = 0;
        for (; column + 4 <= columns; column += 4) {
            vst1q_f32(destination + column, vmulq_f32(beta_vector, vld1q_f32(destination + column)));
        }
        for (; column < columns; ++column) {
            destination[column] *= beta;
        }
    }
}

}  // namespace ArmKleidiAI
