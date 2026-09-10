/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_nchwc_kernel_rvv.cpp

Abstract:

    This module implements RVV kernels for the single precision NCHWc
    convolution operations on riscv64: direct NCHW, direct NCHWc,
    depthwise, and pointwise convolution.

    BlockSize is fixed at 16, matching the ARM64 NEON NCHWc layout.
    With VLEN>=128 and LMUL=4, a single vfloat32m4_t holds 16 floats.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>
#include <cassert>
#include <limits>

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT     0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION         0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION       0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION      0x00000008

namespace {

constexpr size_t BlockSize = 16;

//
// Number of output positions a kernel accumulates at once.
//
// A block of channels fills a register group, so no parallelism is left inside
// one: the multiply accumulates for an output position form a chain. It comes
// from the output positions instead, and they share one filter load. Four of
// them plus that filter occupy twenty of the thirty two registers at LMUL 4.
//
// A register group cannot be held in an array, so the positions are spelled out,
// and a group holding fewer than four repeats the first position in the slots it
// does not use. Those slots are not stored, which lets one body serve both a
// full group and the remainder.
//
constexpr size_t OutputStep = 4;

MLAS_FORCEINLINE
void
ApplyPostProcessing(
    vfloat32m4_t& acc,
    const float* Output,
    const float* Bias,
    unsigned KernelFlags,
    size_t vl
    )
{
    if (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) {
        vfloat32m4_t old_output = __riscv_vle32_v_f32m4(Output, vl);
        acc = __riscv_vfadd_vv_f32m4(acc, old_output, vl);
    }

    if (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) {
        assert(Bias != nullptr);
        vfloat32m4_t bias_vec = __riscv_vle32_v_f32m4(Bias, vl);
        acc = __riscv_vfadd_vv_f32m4(acc, bias_vec, vl);
    }

    if (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) {
        vfloat32m4_t zero = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        acc = __riscv_vfmax_vv_f32m4(acc, zero, vl);
    }
}

MLAS_FORCEINLINE
float
LoadInRange(
    const float* Address,
    const float* RowStart,
    const float* RowEnd
    )
{
    return (Address >= RowStart && Address < RowEnd) ? *Address : 0.0f;
}

//
// Accumulates a group of output positions of the direct NCHW convolution. Input
// and Output address the first of them; the filter is already at its block.
//

MLAS_FORCEINLINE
void
MlasConvNchwFloatGroupRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidthElements,
    size_t DilationWidthElements,
    size_t InputWidthElements,
    size_t DilatedInputWidthElements,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    const float* Bias,
    unsigned KernelFlags,
    size_t vl,
    size_t OutputCountThisIteration
    )
{
    const size_t offset1 = (OutputCountThisIteration > 1 ? 1 : 0) * StrideWidthElements;
    const size_t offset2 = (OutputCountThisIteration > 2 ? 2 : 0) * StrideWidthElements;
    const size_t offset3 = (OutputCountThisIteration > 3 ? 3 : 0) * StrideWidthElements;

    const size_t offset_last = (OutputCountThisIteration - 1) * StrideWidthElements;

    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t acc1 = acc0;
    vfloat32m4_t acc2 = acc0;
    vfloat32m4_t acc3 = acc0;

    for (size_t kh = 0; kh < KernelHeight; kh++) {

        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
        const float* input_row_end = input_row_start + InputWidthElements;

        for (size_t kw = 0; kw < KernelWidth; kw++) {

            const float* input_pos =
                Input + kh * DilatedInputWidthElements + kw * DilationWidthElements;

            vfloat32m4_t filt =
                __riscv_vle32_v_f32m4(&Filter[(kh * KernelWidth + kw) * BlockSize], vl);

            if (input_pos >= input_row_start && (input_pos + offset_last) < input_row_end) {
                acc0 = __riscv_vfmacc_vf_f32m4(acc0, input_pos[0], filt, vl);
                acc1 = __riscv_vfmacc_vf_f32m4(acc1, input_pos[offset1], filt, vl);
                acc2 = __riscv_vfmacc_vf_f32m4(acc2, input_pos[offset2], filt, vl);
                acc3 = __riscv_vfmacc_vf_f32m4(acc3, input_pos[offset3], filt, vl);
            } else {
                acc0 = __riscv_vfmacc_vf_f32m4(
                    acc0, LoadInRange(input_pos, input_row_start, input_row_end), filt, vl);
                acc1 = __riscv_vfmacc_vf_f32m4(
                    acc1, LoadInRange(input_pos + offset1, input_row_start, input_row_end),
                    filt, vl);
                acc2 = __riscv_vfmacc_vf_f32m4(
                    acc2, LoadInRange(input_pos + offset2, input_row_start, input_row_end),
                    filt, vl);
                acc3 = __riscv_vfmacc_vf_f32m4(
                    acc3, LoadInRange(input_pos + offset3, input_row_start, input_row_end),
                    filt, vl);
            }
        }
    }

    ApplyPostProcessing(acc0, &Output[0], Bias, KernelFlags, vl);
    __riscv_vse32_v_f32m4(&Output[0], acc0, vl);

    if (OutputCountThisIteration > 1) {
        ApplyPostProcessing(acc1, &Output[BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[BlockSize], acc1, vl);
    }

    if (OutputCountThisIteration > 2) {
        ApplyPostProcessing(acc2, &Output[2 * BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[2 * BlockSize], acc2, vl);
    }

    if (OutputCountThisIteration > 3) {
        ApplyPostProcessing(acc3, &Output[3 * BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[3 * BlockSize], acc3, vl);
    }
}

}  // namespace

//
// Direct NCHW convolution kernel.
//
// Input is in NCHW format (single channel per kernel position).
// Filter is laid out as [KH][KW][BlockSize] — one scalar input is
// broadcast and multiplied with BlockSize filter values.
// Output is in NCHWc (BlockSize channels interleaved).
//

void
MLASCALL
MlasConvNchwFloatKernelRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad,
    const float* Bias,
    unsigned KernelFlags
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    const size_t vl = __riscv_vsetvl_e32m4(BlockSize);
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t filterSetBlock = 0; filterSetBlock < FilterCount; filterSetBlock++) {

        const float* filter = Filter + filterSetBlock * FilterStrideElements;
        float* output = Output + filterSetBlock * OutputStrideElements;
        const float* bias = (Bias != nullptr) ? &Bias[filterSetBlock * BlockSize] : nullptr;

        size_t output_idx = 0;

        for (; output_idx + OutputStep <= TotalOutputCount; output_idx += OutputStep) {
            MlasConvNchwFloatGroupRvv(
                Input + output_idx * StrideWidthElements, filter, &output[output_idx * BlockSize],
                StrideWidthElements, DilationWidthElements, InputWidthElements,
                DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, bias,
                KernelFlags, vl, OutputStep);
        }

        if (output_idx < TotalOutputCount) {
            MlasConvNchwFloatGroupRvv(
                Input + output_idx * StrideWidthElements, filter, &output[output_idx * BlockSize],
                StrideWidthElements, DilationWidthElements, InputWidthElements,
                DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, bias,
                KernelFlags, vl, TotalOutputCount - output_idx);
        }
    }
}

namespace {

//
// Accumulates a group of output positions of the direct NCHWc convolution. Input
// and Output address the first of them; the filter is already at its block.
//

MLAS_FORCEINLINE
void
MlasConvNchwcFloatGroupRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidthElements,
    size_t DilationWidthElements,
    size_t InputWidthElements,
    size_t DilatedInputWidthElements,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    const float* Bias,
    unsigned KernelFlags,
    size_t vl,
    size_t OutputCountThisIteration
    )
{
    const size_t offset1 = (OutputCountThisIteration > 1 ? 1 : 0) * StrideWidthElements;
    const size_t offset2 = (OutputCountThisIteration > 2 ? 2 : 0) * StrideWidthElements;
    const size_t offset3 = (OutputCountThisIteration > 3 ? 3 : 0) * StrideWidthElements;

    // Furthest position the group stores; the slots it does not use sit at zero.
    const size_t offset_last = (OutputCountThisIteration - 1) * StrideWidthElements;

    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t acc1 = acc0;
    vfloat32m4_t acc2 = acc0;
    vfloat32m4_t acc3 = acc0;

    for (size_t kh = 0; kh < KernelHeight; kh++) {

        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
        const float* input_row_end = input_row_start + InputWidthElements;

        for (size_t kw = 0; kw < KernelWidth; kw++) {

            const float* filter_position =
                Filter + (kh * KernelWidth + kw) * BlockSize * BlockSize;

            const float* input_base =
                Input + kh * DilatedInputWidthElements + kw * DilationWidthElements;
            const float* input_base0 = input_base;
            const float* input_base1 = input_base + offset1;
            const float* input_base2 = input_base + offset2;
            const float* input_base3 = input_base + offset3;

            const bool all_in_bounds = (input_base >= input_row_start) &&
                                     ((input_base + offset_last + BlockSize) <= input_row_end);

            if (all_in_bounds) {
                for (size_t ic = 0; ic < BlockSize; ic++) {
                    vfloat32m4_t filt =
                        __riscv_vle32_v_f32m4(&filter_position[ic * BlockSize], vl);
                    acc0 = __riscv_vfmacc_vf_f32m4(acc0, input_base0[ic], filt, vl);
                    acc1 = __riscv_vfmacc_vf_f32m4(acc1, input_base1[ic], filt, vl);
                    acc2 = __riscv_vfmacc_vf_f32m4(acc2, input_base2[ic], filt, vl);
                    acc3 = __riscv_vfmacc_vf_f32m4(acc3, input_base3[ic], filt, vl);
                }
            } else {
                for (size_t ic = 0; ic < BlockSize; ic++) {
                    vfloat32m4_t filt =
                        __riscv_vle32_v_f32m4(&filter_position[ic * BlockSize], vl);
                    acc0 = __riscv_vfmacc_vf_f32m4(
                        acc0, LoadInRange(input_base0 + ic, input_row_start, input_row_end),
                        filt, vl);
                    acc1 = __riscv_vfmacc_vf_f32m4(
                        acc1, LoadInRange(input_base1 + ic, input_row_start, input_row_end),
                        filt, vl);
                    acc2 = __riscv_vfmacc_vf_f32m4(
                        acc2, LoadInRange(input_base2 + ic, input_row_start, input_row_end),
                        filt, vl);
                    acc3 = __riscv_vfmacc_vf_f32m4(
                        acc3, LoadInRange(input_base3 + ic, input_row_start, input_row_end),
                        filt, vl);
                }
            }
        }
    }

    ApplyPostProcessing(acc0, &Output[0], Bias, KernelFlags, vl);
    __riscv_vse32_v_f32m4(&Output[0], acc0, vl);

    if (OutputCountThisIteration > 1) {
        ApplyPostProcessing(acc1, &Output[BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[BlockSize], acc1, vl);
    }

    if (OutputCountThisIteration > 2) {
        ApplyPostProcessing(acc2, &Output[2 * BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[2 * BlockSize], acc2, vl);
    }

    if (OutputCountThisIteration > 3) {
        ApplyPostProcessing(acc3, &Output[3 * BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[3 * BlockSize], acc3, vl);
    }
}

}  // namespace

//
// Direct NCHWc convolution kernel.
//
// Input is in NCHWc format (BlockSize channels interleaved per spatial position).
// Filter layout: [KH][KW][BlockSize_in][BlockSize_out].
// For each kernel position and input channel, one input scalar is broadcast
// and multiplied with BlockSize output filter values.
//

void
MLASCALL
MlasConvNchwcFloatKernelRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad,
    const float* Bias,
    unsigned KernelFlags
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    const size_t vl = __riscv_vsetvl_e32m4(BlockSize);
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t filterSetBlock = 0; filterSetBlock < FilterCount; filterSetBlock++) {

        const float* filter = Filter + filterSetBlock * FilterStrideElements;
        float* output = Output + filterSetBlock * OutputStrideElements;
        const float* bias = (Bias != nullptr) ? &Bias[filterSetBlock * BlockSize] : nullptr;

        size_t output_idx = 0;

        for (; output_idx + OutputStep <= TotalOutputCount; output_idx += OutputStep) {
            MlasConvNchwcFloatGroupRvv(
                Input + output_idx * StrideWidthElements, filter, &output[output_idx * BlockSize],
                StrideWidthElements, DilationWidthElements, InputWidthElements,
                DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, bias,
                KernelFlags, vl, OutputStep);
        }

        if (output_idx < TotalOutputCount) {
            MlasConvNchwcFloatGroupRvv(
                Input + output_idx * StrideWidthElements, filter, &output[output_idx * BlockSize],
                StrideWidthElements, DilationWidthElements, InputWidthElements,
                DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, bias,
                KernelFlags, vl, TotalOutputCount - output_idx);
        }
    }
}

namespace {

MLAS_FORCEINLINE
vfloat32m4_t
LoadBlockInRange(
    const float* Address,
    const float* RowStart,
    const float* RowEnd,
    size_t vl
    )
{
    if (Address >= RowStart && (Address + BlockSize - 1) < RowEnd) {
        return __riscv_vle32_v_f32m4(Address, vl);
    }

    return __riscv_vfmv_v_f_f32m4(0.0f, vl);
}

//
// Accumulates a group of output positions of the depthwise convolution.
//

MLAS_FORCEINLINE
void
MlasConvDepthwiseFloatGroupRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidthElements,
    size_t DilationWidthElements,
    size_t InputWidthElements,
    size_t DilatedInputWidthElements,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    const float* Bias,
    unsigned KernelFlags,
    size_t vl,
    size_t OutputCountThisIteration
    )
{
    const size_t offset1 = (OutputCountThisIteration > 1 ? 1 : 0) * StrideWidthElements;
    const size_t offset2 = (OutputCountThisIteration > 2 ? 2 : 0) * StrideWidthElements;
    const size_t offset3 = (OutputCountThisIteration > 3 ? 3 : 0) * StrideWidthElements;

    const size_t offset_last = (OutputCountThisIteration - 1) * StrideWidthElements;

    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t acc1 = acc0;
    vfloat32m4_t acc2 = acc0;
    vfloat32m4_t acc3 = acc0;

    for (size_t kh = 0; kh < KernelHeight; kh++) {

        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
        const float* input_row_end = input_row_start + InputWidthElements;

        for (size_t kw = 0; kw < KernelWidth; kw++) {

            const float* input_base =
                Input + kh * DilatedInputWidthElements + kw * DilationWidthElements;

            vfloat32m4_t filt =
                __riscv_vle32_v_f32m4(&Filter[(kh * KernelWidth + kw) * BlockSize], vl);

            if (input_base >= input_row_start &&
                (input_base + offset_last + BlockSize - 1) < input_row_end) {
                acc0 = __riscv_vfmacc_vv_f32m4(
                    acc0, __riscv_vle32_v_f32m4(input_base, vl), filt, vl);
                acc1 = __riscv_vfmacc_vv_f32m4(
                    acc1, __riscv_vle32_v_f32m4(input_base + offset1, vl), filt, vl);
                acc2 = __riscv_vfmacc_vv_f32m4(
                    acc2, __riscv_vle32_v_f32m4(input_base + offset2, vl), filt, vl);
                acc3 = __riscv_vfmacc_vv_f32m4(
                    acc3, __riscv_vle32_v_f32m4(input_base + offset3, vl), filt, vl);
            } else {
                acc0 = __riscv_vfmacc_vv_f32m4(
                    acc0, LoadBlockInRange(input_base, input_row_start, input_row_end, vl),
                    filt, vl);
                acc1 = __riscv_vfmacc_vv_f32m4(
                    acc1,
                    LoadBlockInRange(input_base + offset1, input_row_start, input_row_end, vl),
                    filt, vl);
                acc2 = __riscv_vfmacc_vv_f32m4(
                    acc2,
                    LoadBlockInRange(input_base + offset2, input_row_start, input_row_end, vl),
                    filt, vl);
                acc3 = __riscv_vfmacc_vv_f32m4(
                    acc3,
                    LoadBlockInRange(input_base + offset3, input_row_start, input_row_end, vl),
                    filt, vl);
            }
        }
    }

    ApplyPostProcessing(acc0, &Output[0], Bias, KernelFlags, vl);
    __riscv_vse32_v_f32m4(&Output[0], acc0, vl);

    if (OutputCountThisIteration > 1) {
        ApplyPostProcessing(acc1, &Output[BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[BlockSize], acc1, vl);
    }

    if (OutputCountThisIteration > 2) {
        ApplyPostProcessing(acc2, &Output[2 * BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[2 * BlockSize], acc2, vl);
    }

    if (OutputCountThisIteration > 3) {
        ApplyPostProcessing(acc3, &Output[3 * BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[3 * BlockSize], acc3, vl);
    }
}

}  // namespace

//
// Depthwise NCHWc convolution kernel.
//
// Each channel is convolved with its own filter (element-wise).
// Input is NCHWc, filter is [KH][KW][BlockSize].
//

void
MLASCALL
MlasConvDepthwiseFloatKernelRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t InputStride,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad,
    const float* Bias,
    unsigned KernelFlags
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    const size_t vl = __riscv_vsetvl_e32m4(BlockSize);
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    size_t output_idx = 0;

    for (; output_idx + OutputStep <= TotalOutputCount; output_idx += OutputStep) {
        MlasConvDepthwiseFloatGroupRvv(
            Input + output_idx * StrideWidthElements, Filter, &Output[output_idx * BlockSize],
            StrideWidthElements, DilationWidthElements, InputWidthElements,
            DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, Bias, KernelFlags,
            vl, OutputStep);
    }

    if (output_idx < TotalOutputCount) {
        MlasConvDepthwiseFloatGroupRvv(
            Input + output_idx * StrideWidthElements, Filter, &Output[output_idx * BlockSize],
            StrideWidthElements, DilationWidthElements, InputWidthElements,
            DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, Bias, KernelFlags,
            vl, TotalOutputCount - output_idx);
    }
}

namespace {

//
// Accumulates a group of output positions of the pointwise convolution. Nothing
// is padded here, so no position needs a range test.
//

MLAS_FORCEINLINE
void
MlasConvPointwiseFloatGroupRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidthElements,
    size_t InputStrideElements,
    size_t InputChannels,
    const float* Bias,
    unsigned KernelFlags,
    size_t vl,
    size_t OutputCountThisIteration
    )
{
    const size_t offset1 = (OutputCountThisIteration > 1 ? 1 : 0) * StrideWidthElements;
    const size_t offset2 = (OutputCountThisIteration > 2 ? 2 : 0) * StrideWidthElements;
    const size_t offset3 = (OutputCountThisIteration > 3 ? 3 : 0) * StrideWidthElements;

    vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t acc1 = acc0;
    vfloat32m4_t acc2 = acc0;
    vfloat32m4_t acc3 = acc0;

    for (size_t ic = 0; ic < InputChannels; ic++) {

        const float* input = Input + ic * InputStrideElements;
        const float* filter = Filter + ic * BlockSize * BlockSize;

        for (size_t j = 0; j < BlockSize; j++) {
            vfloat32m4_t filt = __riscv_vle32_v_f32m4(&filter[j * BlockSize], vl);
            acc0 = __riscv_vfmacc_vf_f32m4(acc0, input[j], filt, vl);
            acc1 = __riscv_vfmacc_vf_f32m4(acc1, input[offset1 + j], filt, vl);
            acc2 = __riscv_vfmacc_vf_f32m4(acc2, input[offset2 + j], filt, vl);
            acc3 = __riscv_vfmacc_vf_f32m4(acc3, input[offset3 + j], filt, vl);
        }
    }

    ApplyPostProcessing(acc0, &Output[0], Bias, KernelFlags, vl);
    __riscv_vse32_v_f32m4(&Output[0], acc0, vl);

    if (OutputCountThisIteration > 1) {
        ApplyPostProcessing(acc1, &Output[BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[BlockSize], acc1, vl);
    }

    if (OutputCountThisIteration > 2) {
        ApplyPostProcessing(acc2, &Output[2 * BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[2 * BlockSize], acc2, vl);
    }

    if (OutputCountThisIteration > 3) {
        ApplyPostProcessing(acc3, &Output[3 * BlockSize], Bias, KernelFlags, vl);
        __riscv_vse32_v_f32m4(&Output[3 * BlockSize], acc3, vl);
    }
}

}  // namespace

//
// Pointwise (1x1) NCHWc convolution kernel.
//
// No padding, kernel size = 1.
// Processes OutputCount output positions, accumulating over InputChannels
// (counted in blocks of BlockSize).
//

void
MLASCALL
MlasConvPointwiseFloatKernelRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t InputChannels,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    size_t OutputCount,
    const float* Bias,
    unsigned KernelFlags
    )
{
    const size_t vl = __riscv_vsetvl_e32m4(BlockSize);
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    for (size_t f = 0; f < FilterCount; f++) {

        const float* filter = Filter + f * FilterStrideElements;
        float* output = Output + f * OutputStrideElements;
        const float* bias = (Bias != nullptr) ? &Bias[f * BlockSize] : nullptr;

        size_t out = 0;

        for (; out + OutputStep <= OutputCount; out += OutputStep) {
            MlasConvPointwiseFloatGroupRvv(
                Input + out * StrideWidthElements, filter, &output[out * BlockSize],
                StrideWidthElements, InputStrideElements, InputChannels, bias, KernelFlags, vl,
                OutputStep);
        }

        if (out < OutputCount) {
            MlasConvPointwiseFloatGroupRvv(
                Input + out * StrideWidthElements, filter, &output[out * BlockSize],
                StrideWidthElements, InputStrideElements, InputChannels, bias, KernelFlags, vl,
                OutputCount - out);
        }
    }
}

//
// Max pooling kernel for NCHWc format.
//

void
MLASCALL
MlasPoolMaximumFloatKernelRvv(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t InputStride,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad
    )
{
    MLAS_UNREFERENCED_PARAMETER(ActualKernelSize);
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    const size_t vl = __riscv_vsetvl_e32m4(BlockSize);
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);
    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    const float PadValue = std::numeric_limits<float>::lowest();

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {

        vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(PadValue, vl);

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            const float* row_start = InputBase + kh * DilatedInputWidthElements;
            const float* row_end = row_start + InputWidthElements;

            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_ptr = Input + output_idx * StrideWidthElements +
                                         kh * DilatedInputWidthElements + kw * DilationWidthElements;

                if (input_ptr >= row_start && (input_ptr + BlockSize) <= row_end) {
                    vfloat32m4_t inp = __riscv_vle32_v_f32m4(input_ptr, vl);
                    max_vec = __riscv_vfmax_vv_f32m4(max_vec, inp, vl);
                } else {
                    float values[BlockSize];
                    for (size_t i = 0; i < BlockSize; i++) {
                        const float* ep = input_ptr + i;
                        values[i] = (ep >= row_start && ep < row_end) ? *ep : PadValue;
                    }
                    vfloat32m4_t inp = __riscv_vle32_v_f32m4(values, vl);
                    max_vec = __riscv_vfmax_vv_f32m4(max_vec, inp, vl);
                }
            }
        }

        __riscv_vse32_v_f32m4(&Output[output_idx * BlockSize], max_vec, vl);
    }
}

//
// Average pooling kernel (shared implementation).
//

namespace {

MLAS_FORCEINLINE
void
MlasPoolAverageFloatKernelRvvImpl(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad,
    bool ExcludePad
    )
{
    const size_t vl = __riscv_vsetvl_e32m4(BlockSize);
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);
    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {

        vfloat32m4_t sum_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        uint32_t valid_count[BlockSize];

        if (ExcludePad) {
            for (size_t i = 0; i < BlockSize; i++) {
                valid_count[i] = 0;
            }
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            const float* row_start = InputBase + kh * DilatedInputWidthElements;
            const float* row_end = row_start + InputWidthElements;

            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_ptr = Input + output_idx * StrideWidthElements +
                                         kh * DilatedInputWidthElements + kw * DilationWidthElements;

                if (input_ptr >= row_start && (input_ptr + BlockSize) <= row_end) {
                    vfloat32m4_t inp = __riscv_vle32_v_f32m4(input_ptr, vl);
                    sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, inp, vl);

                    if (ExcludePad) {
                        for (size_t i = 0; i < BlockSize; i++) {
                            valid_count[i]++;
                        }
                    }
                } else {
                    float values[BlockSize];
                    for (size_t i = 0; i < BlockSize; i++) {
                        const float* ep = input_ptr + i;
                        if (ep >= row_start && ep < row_end) {
                            values[i] = *ep;
                            if (ExcludePad) {
                                valid_count[i]++;
                            }
                        } else {
                            values[i] = 0.0f;
                        }
                    }
                    vfloat32m4_t inp = __riscv_vle32_v_f32m4(values, vl);
                    sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, inp, vl);
                }
            }
        }

        if (ExcludePad) {
            float results[BlockSize];
            __riscv_vse32_v_f32m4(results, sum_vec, vl);
            for (size_t i = 0; i < BlockSize; i++) {
                results[i] = (valid_count[i] > 0)
                    ? results[i] / static_cast<float>(valid_count[i])
                    : 0.0f;
            }
            vfloat32m4_t result_vec = __riscv_vle32_v_f32m4(results, vl);
            __riscv_vse32_v_f32m4(&Output[output_idx * BlockSize], result_vec, vl);
        } else {
            vfloat32m4_t divisor = __riscv_vfmv_v_f_f32m4(
                static_cast<float>(ActualKernelSize), vl);
            vfloat32m4_t result_vec = __riscv_vfdiv_vv_f32m4(sum_vec, divisor, vl);
            __riscv_vse32_v_f32m4(&Output[output_idx * BlockSize], result_vec, vl);
        }
    }
}

}  // namespace

void
MLASCALL
MlasPoolAverageExcludePadFloatKernelRvv(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t InputStride,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    MlasPoolAverageFloatKernelRvvImpl(
        Input, Output, StrideWidth, DilationWidth, ActualKernelSize,
        KernelHeight, KernelWidth, InputBase, InputWidth, DilatedInputWidth,
        OutputCountLeftPad, OutputCount, OutputCountRightPad, true);
}

void
MLASCALL
MlasPoolAverageIncludePadFloatKernelRvv(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t InputStride,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    MlasPoolAverageFloatKernelRvvImpl(
        Input, Output, StrideWidth, DilationWidth, ActualKernelSize,
        KernelHeight, KernelWidth, InputBase, InputWidth, DilatedInputWidth,
        OutputCountLeftPad, OutputCount, OutputCountRightPad, false);
}

#endif  // MLAS_USE_RVV
