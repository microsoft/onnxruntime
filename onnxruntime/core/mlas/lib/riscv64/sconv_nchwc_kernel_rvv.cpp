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

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {

        for (size_t filterSetBlock = 0; filterSetBlock < FilterCount; filterSetBlock++) {
            const float* filter = Filter + filterSetBlock * FilterStrideElements;
            float* output = Output + filterSetBlock * OutputStrideElements;

            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            for (size_t kh = 0; kh < KernelHeight; kh++) {
                for (size_t kw = 0; kw < KernelWidth; kw++) {
                    const float* input_pos = Input + output_idx * StrideWidthElements +
                                             kh * DilatedInputWidthElements + kw * DilationWidthElements;

                    const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                    const float* input_row_end = input_row_start + InputWidthElements;

                    float input_value = 0.0f;
                    if (input_pos >= input_row_start && input_pos < input_row_end) {
                        input_value = *input_pos;
                    }

                    size_t kernel_base_pos = kh * KernelWidth + kw;
                    vfloat32m4_t filt = __riscv_vle32_v_f32m4(&filter[kernel_base_pos * BlockSize], vl);
                    acc = __riscv_vfmacc_vf_f32m4(acc, input_value, filt, vl);
                }
            }

            ApplyPostProcessing(acc, &output[output_idx * BlockSize],
                                Bias ? &Bias[filterSetBlock * BlockSize] : nullptr,
                                KernelFlags, vl);

            __riscv_vse32_v_f32m4(&output[output_idx * BlockSize], acc, vl);
        }
    }
}

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

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {

        for (size_t filterSetBlock = 0; filterSetBlock < FilterCount; filterSetBlock++) {
            const float* filter = Filter + filterSetBlock * FilterStrideElements;
            float* output = Output + filterSetBlock * OutputStrideElements;

            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            for (size_t kh = 0; kh < KernelHeight; kh++) {
                for (size_t kw = 0; kw < KernelWidth; kw++) {
                    const float* input_base = Input + output_idx * StrideWidthElements +
                                              kh * DilatedInputWidthElements + kw * DilationWidthElements;

                    const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                    const float* input_row_end = input_row_start + InputWidthElements;

                    size_t kernel_pos = kh * KernelWidth + kw;

                    bool in_bounds = (input_base >= input_row_start) &&
                                     ((input_base + BlockSize) <= input_row_end);

                    if (in_bounds) {
                        for (size_t ic = 0; ic < BlockSize; ic++) {
                            float input_value = input_base[ic];
                            size_t filter_offset = kernel_pos * BlockSize * BlockSize + ic * BlockSize;
                            vfloat32m4_t filt = __riscv_vle32_v_f32m4(&filter[filter_offset], vl);
                            acc = __riscv_vfmacc_vf_f32m4(acc, input_value, filt, vl);
                        }
                    } else {
                        for (size_t ic = 0; ic < BlockSize; ic++) {
                            const float* input_element = input_base + ic;
                            float input_value = 0.0f;
                            if (input_element >= input_row_start && input_element < input_row_end) {
                                input_value = *input_element;
                            }
                            size_t filter_offset = kernel_pos * BlockSize * BlockSize + ic * BlockSize;
                            vfloat32m4_t filt = __riscv_vle32_v_f32m4(&filter[filter_offset], vl);
                            acc = __riscv_vfmacc_vf_f32m4(acc, input_value, filt, vl);
                        }
                    }
                }
            }

            ApplyPostProcessing(acc, &output[output_idx * BlockSize],
                                Bias ? &Bias[filterSetBlock * BlockSize] : nullptr,
                                KernelFlags, vl);

            __riscv_vse32_v_f32m4(&output[output_idx * BlockSize], acc, vl);
        }
    }
}

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
    const size_t KernelSize = KernelHeight * KernelWidth;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {

        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        for (size_t kpos = 0; kpos < KernelSize; kpos++) {
            size_t kh = kpos / KernelWidth;
            size_t kw = kpos % KernelWidth;

            const float* input_base = Input + output_idx * StrideWidthElements +
                                      kh * DilatedInputWidthElements + kw * DilationWidthElements;

            const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
            const float* input_row_end = input_row_start + InputWidthElements;

            vfloat32m4_t input_vec;
            if (input_base >= input_row_start && (input_base + BlockSize - 1) < input_row_end) {
                input_vec = __riscv_vle32_v_f32m4(input_base, vl);
            } else {
                input_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            }

            vfloat32m4_t filt = __riscv_vle32_v_f32m4(&Filter[kpos * BlockSize], vl);
            acc = __riscv_vfmacc_vv_f32m4(acc, input_vec, filt, vl);
        }

        ApplyPostProcessing(acc, &Output[output_idx * BlockSize], Bias, KernelFlags, vl);

        __riscv_vse32_v_f32m4(&Output[output_idx * BlockSize], acc, vl);
    }
}

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

        for (size_t out = 0; out < OutputCount; out++) {

            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            const float* input = Input + out * StrideWidthElements;

            for (size_t ic = 0; ic < InputChannels; ic++) {
                for (size_t j = 0; j < BlockSize; j++) {
                    float input_value = input[ic * InputStrideElements + j];
                    vfloat32m4_t filt = __riscv_vle32_v_f32m4(
                        &filter[ic * BlockSize * BlockSize + j * BlockSize], vl);
                    acc = __riscv_vfmacc_vf_f32m4(acc, input_value, filt, vl);
                }
            }

            ApplyPostProcessing(acc, &output[out * BlockSize],
                                Bias ? &Bias[f * BlockSize] : nullptr,
                                KernelFlags, vl);

            __riscv_vse32_v_f32m4(&output[out * BlockSize], acc, vl);
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
