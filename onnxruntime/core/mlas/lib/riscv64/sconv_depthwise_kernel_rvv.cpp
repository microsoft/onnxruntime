/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_depthwise_kernel_rvv.cpp

Abstract:

    This module implements an RVV kernel for the single precision depthwise
    convolution operation (3x3 kernel, stride 1, dilation 1, padding <= 1)
    on riscv64.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>
#include <cassert>

namespace {

MLAS_FORCEINLINE
void
DepthwiseAccumulateRowRvv(
    vfloat32m4_t& acc,
    const float* row,
    size_t base,
    float w0,
    float w1,
    float w2,
    size_t vl
    )
{
    if (row == nullptr) {
        return;
    }

    const float* r = row + base;
    vfloat32m4_t c0 = __riscv_vle32_v_f32m4(r, vl);
    vfloat32m4_t c1 = __riscv_vle32_v_f32m4(r + 1, vl);
    vfloat32m4_t c2 = __riscv_vle32_v_f32m4(r + 2, vl);

    acc = __riscv_vfmacc_vf_f32m4(acc, w0, c0, vl);
    acc = __riscv_vfmacc_vf_f32m4(acc, w1, c1, vl);
    acc = __riscv_vfmacc_vf_f32m4(acc, w2, c2, vl);
}

MLAS_FORCEINLINE
float
DepthwiseSampleValue(
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

MLAS_FORCEINLINE
float
DepthwiseComputeEdge(
    const float* row0,
    const float* row1,
    const float* row2,
    ptrdiff_t iw,
    size_t width,
    float w00, float w01, float w02,
    float w10, float w11, float w12,
    float w20, float w21, float w22
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

MLAS_FORCEINLINE
float
DepthwiseAccumulateRowScalar(
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

}  // namespace

static
void
MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output
    )
{
    const size_t H = Parameters->InputShape[0];
    const size_t W = Parameters->InputShape[1];
    const size_t out_rows = Parameters->OutputShape[0];
    const size_t out_cols = Parameters->OutputShape[1];

    const size_t pad_top = Parameters->Padding[0];
    const size_t pad_left = Parameters->Padding[1];
    [[maybe_unused]] const size_t pad_bottom = Parameters->Padding[2];
    const size_t pad_right = Parameters->Padding[3];

    assert(pad_top <= 1);
    assert(pad_bottom <= 1);
    assert(pad_left <= 1);
    assert(pad_right <= 1);
    MLAS_UNREFERENCED_PARAMETER(pad_bottom);

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
        while (processed < interior_cols) {
            const ptrdiff_t iw = static_cast<ptrdiff_t>(ow) - static_cast<ptrdiff_t>(pad_left);
            const size_t base = static_cast<size_t>(iw);

            size_t remaining = interior_cols - processed;
            if ((base + remaining + 2) > W) {
                remaining = (W > base + 2) ? W - base - 2 : 0;
            }

            if (remaining == 0) {
                break;
            }

            size_t vl = __riscv_vsetvl_e32m4(remaining);

            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            DepthwiseAccumulateRowRvv(acc, row0, base, w00, w01, w02, vl);
            DepthwiseAccumulateRowRvv(acc, row1, base, w10, w11, w12, vl);
            DepthwiseAccumulateRowRvv(acc, row2, base, w20, w21, w22, vl);

            if (accumulate_output) {
                vfloat32m4_t prev = __riscv_vle32_v_f32m4(out_row + ow, vl);
                acc = __riscv_vfmacc_vf_f32m4(acc, beta, prev, vl);
            }

            __riscv_vse32_v_f32m4(out_row + ow, acc, vl);
            ow += vl;
            processed += vl;
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
{
    MLAS_UNREFERENCED_PARAMETER(Zeros);

    assert(Parameters->Dimensions == 2);
    assert(Parameters->FilterCount == 1);
    assert(Parameters->InputChannels == 1);
    assert(Parameters->KernelShape[0] == 3 && Parameters->KernelShape[1] == 3);
    assert(Parameters->StrideShape[0] == 1 && Parameters->StrideShape[1] == 1);
    assert(Parameters->DilationShape[0] == 1 && Parameters->DilationShape[1] == 1);

    MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(Parameters, Input, Filter, Output);
}

#endif  // MLAS_USE_RVV
