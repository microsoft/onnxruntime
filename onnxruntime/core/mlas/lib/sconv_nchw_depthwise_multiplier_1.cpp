/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_nchw_depthwise_multiplier_1.cpp

Abstract:

    This module implements the single precision NCHW depthwise convolution
    kernel entry point for the currently supported depth-multiplier-1 case.

    At present, the implementation is intentionally narrow and only supports:

      - 2D convolution
      - input channels per group = 1
      - output channels per group = 1
      - kernel = 3x3
      - stride = 1x1
      - dilation = 1x1
      - padding <= 1 on each side

--*/


#include "mlasi.h"
#include <cassert>

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
    MLAS_FLOAT32X4& acc,
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
    const MLAS_FLOAT32X4 c0 = MlasLoadFloat32x4(r);
    const MLAS_FLOAT32X4 c1 = MlasLoadFloat32x4(r + 1);
    const MLAS_FLOAT32X4 c2 = MlasLoadFloat32x4(r + 2);

    acc = MlasMultiplyAddFloat32x4(c0, w0, acc);
    acc = MlasMultiplyAddFloat32x4(c1, w1, acc);
    acc = MlasMultiplyAddFloat32x4(c2, w2, acc);
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

    Computes one single-channel 3x3 depthwise convolution slice.

Arguments:

    Parameters - Supplies the prepared convolution parameters.

    Input - Supplies one input channel slice in CHW layout.

    Filter - Supplies one filter slice in FH x FW layout.

    Output - Supplies one output channel slice in OH x OW layout.

--*/
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
            MLAS_FLOAT32X4 acc = MlasZeroFloat32x4();

            DepthwiseAccumulateRowVector(acc, row0, base, w00, w01, w02);
            DepthwiseAccumulateRowVector(acc, row1, base, w10, w11, w12);
            DepthwiseAccumulateRowVector(acc, row2, base, w20, w21, w22);

            if (accumulate_output) {
                const MLAS_FLOAT32X4 prev = MlasLoadFloat32x4(out_row + ow);
                acc = MlasMultiplyAddFloat32x4(prev, beta, acc);
            }

            MlasStoreFloat32x4(out_row + ow, acc);
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

void MlasConvDepthwiseFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
/*++

Routine Description:

    Dispatches the currently supported depth-multiplier-1 implementation.

Arguments:

    Parameters - Supplies the prepared convolution parameters. The following
        constraints are required:
          * Dimensions == 2
          * InputChannels == 1
          * FilterCount == 1
          * KernelShape == {3, 3}
          * StrideShape == {1, 1}
          * DilationShape == {1, 1}
          * Padding <= 1 on each side

    Input - Supplies one batch/group input slice in CHW layout.

    Filter - Supplies one group filter block in OIHW layout. For this path,
        only a single 3x3 filter is consumed.

    Output - Supplies one batch/group output slice in CHW layout.

    Zeros - Supplies a zero-filled working buffer. This implementation does not
        currently consume it.

Note:
    The routine asserts its narrow contract locally. Selection logic in
    MlasConvPrepare() is expected to route only matching shapes here.

    This is not a generic depthwise path. It currently supports only 2D
    kernel 3x3, stride 1, dilation 1, and padding <= 1.

--*/
{
    assert(Parameters->Dimensions == 2);
    assert(Parameters->FilterCount == 1);
    assert(Parameters->InputChannels == 1);
    assert(Parameters->KernelShape[0] == 3);
    assert(Parameters->KernelShape[1] == 3);
    assert(Parameters->StrideShape[0] == 1);
    assert(Parameters->StrideShape[1] == 1);
    assert(Parameters->DilationShape[0] == 1);
    assert(Parameters->DilationShape[1] == 1);
    assert(Parameters->Padding[0] <= 1);
    assert(Parameters->Padding[1] <= 1);
    assert(Parameters->Padding[2] <= 1);
    assert(Parameters->Padding[3] <= 1);

    MLAS_UNREFERENCED_PARAMETER(Zeros);

    // Kernel dispatch
    MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(Parameters, Input, Filter, Output);
}
