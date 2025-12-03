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

    static void MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
/*++

Routine Description:

    This routine is an inner kernel to compute convolution on one channel input with one filter channel.

Arguments:

    Parameters - conv parameters calculated based on conv parameters like padding, strides, dilations, etc.

    Input - input channel data start. Input is NCHW, so this pointer point to single H x W image data.

    Filter - Whole filters are of F x CpG x FH x FW, this filter point to single FH x FW filter data.

    Output - whole output are of N x F x OH x OW. This pointer point to single OH x OW output image data.

    Zeroes - Point to working buffer where all 0.0f are filled.

--*/
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

    Currently only support 2d kernel 3x3.
    Will add general case and more special case if needed later.

--*/
{
    MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(Parameters, Input, Filter, Output, Zeros);
}

