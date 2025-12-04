/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_nchw_kernel_neon.cpp

Abstract:

    This module implements the single precision NCHW convolution kernels for ARM NEON.

--*/


#include "mlasi.h"
#include <arm_neon.h>

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#else
#define C10_LIKELY(expr) (expr)
#endif


#if defined(_MSC_VER) && defined(MLAS_TARGET_ARM64)
MLAS_FORCEINLINE float32x4_t operator+(float32x4_t lhs, float32x4_t rhs)
{
    return vaddq_f32(lhs, rhs);
}

MLAS_FORCEINLINE float32x4_t operator-(float32x4_t lhs, float32x4_t rhs)
{
    return vsubq_f32(lhs, rhs);
}

MLAS_FORCEINLINE float32x4_t operator*(float32x4_t lhs, float32x4_t rhs)
{
    return vmulq_f32(lhs, rhs);
}

MLAS_FORCEINLINE float32x4_t operator-(float32x4_t value)
{
    return vnegq_f32(value);
}
#endif

struct Arguments final {
    // Input layer dimensions
    int64_t in_rows;
    int64_t in_cols;

    // Output layer dimensions
    int64_t out_rows;
    int64_t out_cols;

    // Padding info
    int64_t pad_rows;
    int64_t pad_cols;
};

inline void winograd_f2k3_input_transform_inplace__neon(
    float32x4_t* const d0,
    float32x4_t* const d1,
    float32x4_t* const d2,
    float32x4_t* const d3
)
{
    const float32x4_t wd0 = *d0 - *d2;
    const float32x4_t wd1 = *d1 + *d2;
    const float32x4_t wd2 = -*d1 + *d2;
    const float32x4_t wd3 = *d1 - *d3;
    *d0 = wd0;
    *d1 = wd1;
    *d2 = wd2;
    *d3 = wd3;
}

inline void winograd_f2k3_output_transform_inplace__neon(
    float32x4_t* const m0,
    float32x4_t* const m1,
    const float32x4_t* const m2,
    const float32x4_t* const m3
)
{
    *m0 = *m0 + *m1 + *m2;
    *m1 = *m1 - *m2 - *m3;
}

inline float32x4_t vmuladdq_f32(const float32x4_t c, const float32x4_t a, const float32x4_t b)
{
    return vfmaq_f32(c, a, b);
    // TODO: Support ARMv7
}

inline float32x4_t vmulsubq_f32(const float32x4_t c, const float32x4_t a, const float32x4_t b)
{
    return vfmsq_f32(c, a, b);
    // TODO: Support ARMv7
}

inline void winograd_f2k3_kernel_transform__neon(
    const float32x4_t g0,
    const float32x4_t g1,
    const float32x4_t g2,
    float32x4_t* const transform0,
    float32x4_t* const transform1,
    float32x4_t* const transform2,
    float32x4_t* const transform3
)
{
    const float32x4_t const_half = vdupq_n_f32(0.5f);
    float32x4_t half_g0_plus_g2 = const_half * (g0 + g2);
    *transform0 = g0;
    *transform1 = vmuladdq_f32(half_g0_plus_g2, const_half, g1);
    *transform2 = vmulsubq_f32(half_g0_plus_g2, const_half, g1);
    *transform3 = g2;
}

inline float32x4x4_t v4f_transpose4x4__neon(const float32x4x4_t m)
{
    float32x4x4_t ret;
    vst4q_f32((float*)(&ret), m);
    return ret;
}

void convolution_depthwise3x3_winograd_impl(
    const Arguments& args,
    const float* const input,
    const float* const kernel,
    float* const output
)
{
    //const float32x4_t vbias = vsetq_lane_f32(*bias, vdupq_n_f32(0.0), 1);
    float32x4x4_t kernel_tile;

    {
        const float32x4_t g0 = vld1q_f32(kernel);
        const float32x4_t g1 = vld1q_f32(kernel + 3);
        // g2[3] is junk
        const float32x4_t g2 =
            vextq_f32(vld1q_f32(kernel + 5), vld1q_f32(kernel + 5), 1);
        float32x4x4_t w;
        winograd_f2k3_kernel_transform__neon(
            g0, g1, g2, &w.val[0], &w.val[1], &w.val[2], &w.val[3]
        );
        w = v4f_transpose4x4__neon(w);

        winograd_f2k3_kernel_transform__neon(
            w.val[0],
            w.val[1],
            w.val[2],
            &kernel_tile.val[0],
            &kernel_tile.val[1],
            &kernel_tile.val[2],
            &kernel_tile.val[3]
        );
    }

    #define TILE                                                      \
        winograd_f2k3_input_transform_inplace__neon(                  \
            &input_tile.val[0],                                       \
            &input_tile.val[1],                                       \
            &input_tile.val[2],                                       \
            &input_tile.val[3]                                        \
        );                                                            \
        input_tile = v4f_transpose4x4__neon(input_tile);              \
        winograd_f2k3_input_transform_inplace__neon(                  \
            &input_tile.val[0],                                       \
            &input_tile.val[1],                                       \
            &input_tile.val[2],                                       \
            &input_tile.val[3]                                        \
        );                                                            \
                                                                      \
        for (size_t row = 0; row < 4; ++row) {                        \
            input_tile.val[row] =                                     \
                vmulq_f32(input_tile.val[row], kernel_tile.val[row]); \
        }                                                             \
                                                                      \
        winograd_f2k3_output_transform_inplace__neon(                 \
            &input_tile.val[0],                                       \
            &input_tile.val[1],                                       \
            &input_tile.val[2],                                       \
            &input_tile.val[3]                                        \
        );                                                            \
        input_tile = v4f_transpose4x4__neon(input_tile);              \
        winograd_f2k3_output_transform_inplace__neon(                 \
            &input_tile.val[0],                                       \
            &input_tile.val[1],                                       \
            &input_tile.val[2],                                       \
            &input_tile.val[3]                                        \
        )

  // Non-padded regime.

    // Iterate over non-padded output tiles.
    // TODO: avoid spilling W by breaking out the non-padded vs padded case.
    for (int64_t oth = 0; oth < (args.out_rows + 1) / 2; ++oth) {
        for (int64_t otw = 0; otw < (args.out_cols + 1) / 2; ++otw) {
            // load input tile for [oth, otw];
            int64_t ih = oth * 2 - args.pad_rows;
            int64_t iw = otw * 2 - args.pad_cols;
            // fast-path, all accesses in-bounds
            if (C10_LIKELY(
                    ih >= 0 && iw >= 0 && ih + 3 < args.in_rows &&
                    iw + 3 < args.in_cols && 2 * oth + 1 < args.out_rows &&
                    2 * otw + 1 < args.out_cols
                )) {
                float32x4x4_t input_tile;
                for (int64_t row = 0; row < 4; ++row) {
                    input_tile.val[row] =
                        vld1q_f32(input + (ih + row) * args.in_cols + iw);
                }

                TILE;

                for (int64_t row = 0; row < 2; ++row) {
                    vst1_f32(
                        output + (oth * 2 + row) * args.out_cols + otw * 2,
                        vget_low_f32(input_tile.val[row])
                    );
                }
            } else {
                float block[4][4];
                for (int64_t row = 0; row < 4; ++row) {
                    for (int64_t col = 0; col < 4; ++col) {
                        if (ih + row >= 0 && iw + col >= 0 && ih + row < args.in_rows &&
                            iw + col < args.in_cols) {
                            block[row][col] = input[(ih + row) * args.in_cols + iw + col];
                        } else {
                            block[row][col] = 0.0;
                        }
                    }
                }

                float32x4x4_t input_tile;
                for (int64_t row = 0; row < 4; ++row) {
                    input_tile.val[row] = vld1q_f32(&block[row][0]);
                }

                TILE;

                float oblock[2][2];
                for (int64_t row = 0; row < 2; ++row) {
                    vst1_f32(&oblock[row][0], vget_low_f32(input_tile.val[row]));
                }
                for (int64_t row = 0; row < 2; ++row) {
                    for (int64_t col = 0; col < 2; ++col) {
                        if (2 * oth + row < args.out_rows &&
                            2 * otw + col < args.out_cols) {
                            output[(2 * oth + row) * args.out_cols + 2 * otw + col] =
                                oblock[row][col];
                        }
                    }
                }
            }
        }
    }
}

MLAS_FORCEINLINE float DepthwiseSampleValue(
    const float* row,
    ptrdiff_t col,
    size_t width
)
{
    if (row == nullptr || col < 0 || col >= static_cast<ptrdiff_t>(width)) {
        return 0.0f;
    }
    return row[col];
}

MLAS_FORCEINLINE float DepthwiseAccumulateRowScalar(
    float acc,
    const float* row,
    size_t base,
    float w0,
    float w1,
    float w2
)
{
    if (row == nullptr) {
        return acc;
    }

    acc += row[base] * w0;
    acc += row[base + 1] * w1;
    acc += row[base + 2] * w2;
    return acc;
}

MLAS_FORCEINLINE void DepthwiseAccumulateRowVector(
    float32x4_t& acc,
    const float* row,
    size_t base,
    float w0,
    float w1,
    float w2
)
{
    if (row == nullptr) {
        return;
    }

    const float* r = row + base;
    const float32x4_t c0 = vld1q_f32(r);
    const float32x4_t c1 = vld1q_f32(r + 1);
    const float32x4_t c2 = vld1q_f32(r + 2);

    acc = vmlaq_n_f32(acc, c0, w0);
    acc = vmlaq_n_f32(acc, c1, w1);
    acc = vmlaq_n_f32(acc, c2, w2);
}

MLAS_FORCEINLINE float DepthwiseComputeEdge(
    const float* row0,
    const float* row1,
    const float* row2,
    ptrdiff_t iw,
    size_t width,
    const float w00,
    const float w01,
    const float w02,
    const float w10,
    const float w11,
    const float w12,
    const float w20,
    const float w21,
    const float w22
)
{
    float acc = 0.0f;
    const ptrdiff_t c0 = iw;
    const ptrdiff_t c1 = iw + 1;
    const ptrdiff_t c2 = iw + 2;

    acc += DepthwiseSampleValue(row0, c0, width) * w00;
    acc += DepthwiseSampleValue(row0, c1, width) * w01;
    acc += DepthwiseSampleValue(row0, c2, width) * w02;
    acc += DepthwiseSampleValue(row1, c0, width) * w10;
    acc += DepthwiseSampleValue(row1, c1, width) * w11;
    acc += DepthwiseSampleValue(row1, c2, width) * w12;
    acc += DepthwiseSampleValue(row2, c0, width) * w20;
    acc += DepthwiseSampleValue(row2, c1, width) * w21;
    acc += DepthwiseSampleValue(row2, c2, width) * w22;

    return acc;
}

static void DepthwiseConv3x3Stride1PadLe1Neon(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
)
{
    const size_t H = Parameters->InputShape[0];
    const size_t W = Parameters->InputShape[1];
    const size_t out_rows = Parameters->OutputShape[0];
    const size_t out_cols = Parameters->OutputShape[1];

    const size_t pad_top = Parameters->Padding[0];
    const size_t pad_left = Parameters->Padding[1];
    const size_t pad_right = Parameters->Padding[3];

    const float beta = Parameters->Beta;
    const bool accumulate_output = beta != 0.0f;

    const float w00 = Filter[0];
    const float w01 = Filter[1];
    const float w02 = Filter[2];
    const float w10 = Filter[3];
    const float w11 = Filter[4];
    const float w12 = Filter[5];
    const float w20 = Filter[6];
    const float w21 = Filter[7];
    const float w22 = Filter[8];

    for (size_t oh = 0; oh < out_rows; ++oh) {
        const ptrdiff_t ih = static_cast<ptrdiff_t>(oh) - static_cast<ptrdiff_t>(pad_top);

        const ptrdiff_t row0_index = ih;
        const ptrdiff_t row1_index = ih + 1;
        const ptrdiff_t row2_index = ih + 2;

        const float* row0 = nullptr;
        const float* row1 = nullptr;
        const float* row2 = nullptr;

        if (row0_index >= 0 && row0_index < static_cast<ptrdiff_t>(H)) {
            row0 = Input + static_cast<size_t>(row0_index) * W;
        }
        if (row1_index >= 0 && row1_index < static_cast<ptrdiff_t>(H)) {
            row1 = Input + static_cast<size_t>(row1_index) * W;
        }
        if (row2_index >= 0 && row2_index < static_cast<ptrdiff_t>(H)) {
            row2 = Input + static_cast<size_t>(row2_index) * W;
        }

        float* out_row = Output + oh * out_cols;
        size_t ow = 0;

        if (pad_left && ow < out_cols) {
            const ptrdiff_t iw = static_cast<ptrdiff_t>(ow) - static_cast<ptrdiff_t>(pad_left);
            float acc = DepthwiseComputeEdge(
                row0, row1, row2, iw, W,
                w00, w01, w02, w10, w11, w12, w20, w21, w22
            );
            if (accumulate_output) {
                acc += beta * out_row[ow];
            }
            out_row[ow++] = acc;
        }

        size_t interior_cols = 0;
        if (out_cols > pad_left + pad_right) {
            interior_cols = out_cols - pad_left - pad_right;
        }

        size_t processed = 0;
        while (processed + 4 <= interior_cols) {
            const ptrdiff_t iw = static_cast<ptrdiff_t>(ow) - static_cast<ptrdiff_t>(pad_left);
            if ((iw + 5) >= static_cast<ptrdiff_t>(W)) {
                break;
            }

            const size_t base = static_cast<size_t>(iw);
            float32x4_t acc = vdupq_n_f32(0.0f);

            DepthwiseAccumulateRowVector(acc, row0, base, w00, w01, w02);
            DepthwiseAccumulateRowVector(acc, row1, base, w10, w11, w12);
            DepthwiseAccumulateRowVector(acc, row2, base, w20, w21, w22);

            if (accumulate_output) {
                const float32x4_t prev = vld1q_f32(out_row + ow);
                acc = vmlaq_n_f32(acc, prev, beta);
            }

            vst1q_f32(out_row + ow, acc);
            ow += 4;
            processed += 4;
        }

        for (; processed < interior_cols; ++processed) {
            const ptrdiff_t iw = static_cast<ptrdiff_t>(ow) - static_cast<ptrdiff_t>(pad_left);
            const size_t base = static_cast<size_t>(iw);

            float acc = 0.0f;
            acc = DepthwiseAccumulateRowScalar(acc, row0, base, w00, w01, w02);
            acc = DepthwiseAccumulateRowScalar(acc, row1, base, w10, w11, w12);
            acc = DepthwiseAccumulateRowScalar(acc, row2, base, w20, w21, w22);

            if (accumulate_output) {
                acc += beta * out_row[ow];
            }
            out_row[ow++] = acc;
        }

        if (pad_right && ow < out_cols) {
            const ptrdiff_t iw = static_cast<ptrdiff_t>(ow) - static_cast<ptrdiff_t>(pad_left);
            float acc = DepthwiseComputeEdge(
                row0, row1, row2, iw, W,
                w00, w01, w02, w10, w11, w12, w20, w21, w22
            );
            if (accumulate_output) {
                acc += beta * out_row[ow];
            }
            out_row[ow++] = acc;
        }
    }
}

/*
    static void MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
++

Routine Description:

    This routine is an inner kernel to compute convolution on one channel input with one filter channel.

Arguments:

    Parameters - conv parameters calculated based on conv parameters like padding, strides, dilations, etc.

    Input - input channel data start. Input is NCHW, so this pointer point to single H x W image data.

    Filter - Whole filters are of F x CpG x FH x FW, this filter point to single FH x FW filter data.

    Output - whole output are of N x F x OH x OW. This pointer point to single OH x OW output image data.

    Zeroes - Point to working buffer where all 0.0f are filled.

--
{
     MLAS_UNREFERENCED_PARAMETER(Zeros);

     Arguments args;
     args.in_rows = Parameters->InputShape[0];
     args.in_cols = Parameters->InputShape[1];

     args.out_rows = Parameters->OutputShape[0];
     args.out_cols = Parameters->OutputShape[1];

     args.pad_rows = Parameters->Padding[0];
     args.pad_cols = Parameters->Padding[1];

     convolution_depthwise3x3_winograd_impl(args, Input, Filter, Output);
}
*/

static
void
MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output
    )
/*++

Routine Description:

    This routine is an inner kernel to compute convolution on one channel input with one filter channel.

Arguments:

    Parameters - conv parameters calculated based on conv parameters like padding, strides, dilations, etc.

    Input - input channel data start. Input is NCHW, so this pointer point to single H x W image data.

    Filter - Whole filters are of F x CpG x FH x FW, this filter point to single FH x FW filter data.

    Output - whole output are of N x F x OH x OW. This pointer point to single OH x OW output image data.

--*/
{
        DepthwiseConv3x3Stride1PadLe1Neon(Parameters, Input, Filter, Output);
}

void MlasConvDepthwiseFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
/*++

Routine Description:

    This routine is an inner kernel to compute depthwise convolution for one filter channel on one input channel.

Arguments:

    Parameters - conv parameters calculated based on conv parameters like padding, strides, dilations, etc.

    Input - input channel data start. Input is NCHW, so this pointer point to single H x W image data.

    Filter - Whole filters are of F x CpG x FH x FW, this filter point to single FH x FW filter data.

    Output - whole output are of N x F x OH x OW. This pointer point to single OH x OW output image data.

    Zeroes - Point to working buffer where all 0.0f are filled.

Note:
    No checking here as it is inner loop. Logic in generating Parameters controls the check.

    Currently only support 2d kernel 3x3 with strides=1, dilations=1, pads<=1.
    Will add general case and more special case if needed later.

--*/
{
    MLAS_UNREFERENCED_PARAMETER(Zeros);
    MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(Parameters, Input, Filter, Output);
}
