/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_kernel_neon.cpp

Abstract:

    This module implements the single precision convolution kernels for ARM NEON.

--*/


#include "mlasi.h"

static
void
MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1(
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
    const size_t W = Parameters->InputShape[1];
    const float beta = Parameters->Beta;

    if (W > 1) {

        const float w00 = Filter[0];
        const float w01 = Filter[1];
        const float w02 = Filter[2];
        const float w10 = Filter[3];
        const float w11 = Filter[4];
        const float w12 = Filter[5];
        const float w20 = Filter[6];
        const float w21 = Filter[7];
        const float w22 = Filter[8];

        const size_t H = Parameters->InputShape[0];
        const size_t pad_top = Parameters->Padding[0];
        const size_t pad_left = Parameters->Padding[1];
        const size_t stride_h = Parameters->StrideShape[0];
        const size_t stride_w = Parameters->StrideShape[1];

        // We treat pad_left, pad_top are hard require.
        // While pad_right and pad_bottom could be adjusted if they do not 100% match other parameters.
        const size_t pad_right = (((Parameters->OutputShape[1] - 1) * stride_w + 3) > (pad_left + W)) ? 1 : 0;

        const float* row0 = (pad_top > 0) ? Zeros : (Input - pad_left);
        // Need to handle effective pad_bottom is 2 when H == 1
        const float* row1 = (H + pad_top <= 1) ? Zeros : (Input + (1 - pad_top) * W) - pad_left;
        const float* row2 = (H + pad_top <= 2) ? Zeros : (row1 + W);

        for (size_t h = 0, out_row = Parameters->OutputShape[0]; out_row > 0; --out_row) {
            auto out_col = Parameters->OutputShape[1];

            if (pad_left == 1) {
                float dotsum = w01 * row0[1] + w02 * row0[2] + w11 * row1[1] + w12 * row1[2] +
                               w21 * row2[1] + w22 * row2[2] + (beta == 0.f ? 0.f : *Output * beta);
                *Output++ = dotsum;
                out_col--;
                row0 += stride_w;
                row1 += stride_w;
                row2 += stride_w;
            }

            for (; out_col > pad_right; out_col--) {
                float dotsum = w00 * row0[0] + w01 * row0[1] + w02 * row0[2] + w10 * row1[0] +
                               w11 * row1[1] + w12 * row1[2] + w20 * row2[0] + w21 * row2[1] +
                               w22 * row2[2] + (beta == 0.f ? 0.f : *Output * beta);
                *Output++ = dotsum;
                row0 += stride_w;
                row1 += stride_w;
                row2 += stride_w;
            }

            if (out_col == 1) { // pad_right == 1
                float dotsum = w00 * row0[0] + w01 * row0[1] + w10 * row1[0] + w11 * row1[1] +
                               w20 * row2[0] + w21 * row2[1] + (beta == 0.f ? 0.f : *Output * beta);
                *Output++ = dotsum;
            }

            h += stride_h;
            row0 = (Input + (h - pad_top) * W) - pad_left;
            row1 = row0 + W;
            row2 = (h + 2 >= H + pad_top) ? Zeros : (row1 + W);
        }

    } else { // W == 1

        const size_t H = Parameters->InputShape[0];
        const size_t pad_left = Parameters->Padding[1];
        const size_t pad_top = Parameters->Padding[0];
        const size_t stride_h = Parameters->StrideShape[0];
        size_t out_row = Parameters->OutputShape[0];

        // Make sure pad_bottom is consistent with other parameters.
        size_t pad_bottom = ((out_row - 1) * stride_h + 3) > (pad_top + H) ?
                                ((out_row - 1) * stride_h + 3) - (pad_top + H) : 0;

        const float w0 = Filter[pad_left ? 1 : 0];
        const float w1 = Filter[pad_left ? 4 : 3];
        const float w2 = Filter[pad_left ? 7 : 6];
        auto init_v = (beta == 0.f ? 0.f : *Output * beta);

        if (pad_top == 1) {
            *Output++ = w1 * Input[0] + w2 * ((H + pad_top <= 2) ? 0.0f : Input[1]) + init_v;
            out_row--;
        }

        for (const float* row = Input + pad_top * stride_h - pad_top; out_row > pad_bottom; --out_row) {
            // All pixels are in the input col
            auto init = (beta == 0.f ? 0.f : *Output * beta);
            *Output++ = w0 * row[0] + w1 * row[1] + w2 * row[2] + init;
            row += stride_h;
        }

        if (out_row > 0) {
            // last 1 or 2 rows are from the padding zero row.
            // out_row == 1 when arrive here
            if (pad_bottom == 1) {
                const float* row = Input + H - 2;
                *Output++ = w0 * row[0] + w1 * row[1] + init_v;
            } else { // pad_bottom == 2 and H == 1 and padding_top == 0
                *Output++ = w0 * Input[0] + init_v;
            }
        }
    }

}


void
MlasConvDepthwiseFloat_CHW(
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

#if defined(MLAS_USE_ARM_NEON_NCHWC)
// Everything below these are NCHWC related kernels

#include "sconv.h"

constexpr size_t BlockSize = MLAS_PLATFORM::MLAS_NEON_NCHWC_BLOCK_SIZE;

// Common implementation for NCHW and NCHWC convolution kernels
template <bool IsNchwcFormat>
void
    MLASCALL
    MlasConvFloatKernelNeonImpl(
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
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    MLAS_UNREFERENCED_PARAMETER(InputStride);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        bool is_main_region = (output_idx >= OutputCountLeftPad && output_idx < OutputCountLeftPad + OutputCount);

        for (size_t filterSetBlock = 0; filterSetBlock < FilterCount; filterSetBlock++) {
            const float* filter = Filter + filterSetBlock * FilterStrideElements;
            float* output = Output + filterSetBlock * OutputStrideElements;

            float32x4_t Accumulator0, Accumulator1, Accumulator2, Accumulator3;

            if (AccumulateOutput) {
                Accumulator0 = MlasLoadFloat32x4(&output[output_idx * BlockSize]);
                Accumulator1 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 4]);
                Accumulator2 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 8]);
                Accumulator3 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 12]);
            } else {
                Accumulator0 = MlasBroadcastFloat32x4(0.0f);
                Accumulator1 = MlasBroadcastFloat32x4(0.0f);
                Accumulator2 = MlasBroadcastFloat32x4(0.0f);
                Accumulator3 = MlasBroadcastFloat32x4(0.0f);
            }

            if (BiasAddition) {
                const float32x4_t BiasVector0 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize]);
                const float32x4_t BiasVector1 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize + 4]);
                const float32x4_t BiasVector2 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize + 8]);
                const float32x4_t BiasVector3 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize + 12]);

                Accumulator0 = MlasAddFloat32x4(Accumulator0, BiasVector0);
                Accumulator1 = MlasAddFloat32x4(Accumulator1, BiasVector1);
                Accumulator2 = MlasAddFloat32x4(Accumulator2, BiasVector2);
                Accumulator3 = MlasAddFloat32x4(Accumulator3, BiasVector3);
            }

            for (size_t kh = 0; kh < KernelHeight; kh++) {
                for (size_t kw = 0; kw < KernelWidth; kw++) {
                    const float* input_base = Input + output_idx * StrideWidthElements +
                                              kh * DilatedInputWidthElements + kw * DilationWidthElements;

                    if constexpr (IsNchwcFormat) {
                        for (size_t filterBlock = 0; filterBlock < BlockSize; filterBlock++) {
                            const float* input_element = input_base + filterBlock;
                            const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                            const float* input_row_end = input_row_start + InputWidthElements;

                            float input_value;
                            if (is_main_region || (input_element >= input_row_start && input_element < input_row_end)) {
                                input_value = *input_element;
                            } else {
                                input_value = 0.0f;
                            }

                            const float32x4_t InputVector = MlasBroadcastFloat32x4(input_value);

                            size_t kernel_base_pos = kh * (KernelWidth * BlockSize * BlockSize) +
                                                     kw * (BlockSize * BlockSize) +
                                                     filterBlock * BlockSize;

                            const float32x4_t FilterVector0 = MlasLoadFloat32x4(&filter[kernel_base_pos]);
                            const float32x4_t FilterVector1 = MlasLoadFloat32x4(&filter[kernel_base_pos + 4]);
                            const float32x4_t FilterVector2 = MlasLoadFloat32x4(&filter[kernel_base_pos + 8]);
                            const float32x4_t FilterVector3 = MlasLoadFloat32x4(&filter[kernel_base_pos + 12]);

                            Accumulator0 = MlasMultiplyAddFloat32x4(InputVector, FilterVector0, Accumulator0);
                            Accumulator1 = MlasMultiplyAddFloat32x4(InputVector, FilterVector1, Accumulator1);
                            Accumulator2 = MlasMultiplyAddFloat32x4(InputVector, FilterVector2, Accumulator2);
                            Accumulator3 = MlasMultiplyAddFloat32x4(InputVector, FilterVector3, Accumulator3);
                        }
                    } else {
                        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                        const float* input_row_end = input_row_start + InputWidthElements;

                        float input_value;
                        if (is_main_region || (input_base >= input_row_start && input_base < input_row_end)) {
                            input_value = *input_base;
                        } else {
                            input_value = 0.0f;
                        }

                        const float32x4_t InputVector = MlasBroadcastFloat32x4(input_value);

                        size_t kernel_base_pos = kh * KernelWidth + kw;

                        const float32x4_t FilterVector0 = MlasLoadFloat32x4(&filter[kernel_base_pos * BlockSize]);
                        const float32x4_t FilterVector1 = MlasLoadFloat32x4(&filter[kernel_base_pos * BlockSize + 4]);
                        const float32x4_t FilterVector2 = MlasLoadFloat32x4(&filter[kernel_base_pos * BlockSize + 8]);
                        const float32x4_t FilterVector3 = MlasLoadFloat32x4(&filter[kernel_base_pos * BlockSize + 12]);

                        Accumulator0 = MlasMultiplyAddFloat32x4(InputVector, FilterVector0, Accumulator0);
                        Accumulator1 = MlasMultiplyAddFloat32x4(InputVector, FilterVector1, Accumulator1);
                        Accumulator2 = MlasMultiplyAddFloat32x4(InputVector, FilterVector2, Accumulator2);
                        Accumulator3 = MlasMultiplyAddFloat32x4(InputVector, FilterVector3, Accumulator3);
                    }
                }
            }

            if (ReluActivation) {
                Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
                Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
                Accumulator2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
                Accumulator3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);
            }

            MlasStoreFloat32x4(&output[output_idx * BlockSize], Accumulator0);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], Accumulator1);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], Accumulator2);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], Accumulator3);
        }
    }
}

void
    MLASCALL
    MlasConvNchwFloatKernelNeon(
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
    MlasConvFloatKernelNeonImpl<false>(
        Input,
        Filter,
        Output,
        StrideWidth,
        DilationWidth,
        FilterCount,
        InputStride,
        FilterStride,
        OutputStride,
        KernelHeight,
        KernelWidth,
        InputBase,
        InputWidth,
        DilatedInputWidth,
        OutputCountLeftPad,
        OutputCount,
        OutputCountRightPad,
        Bias,
        KernelFlags
    );
}

//
// Implementation of MlasConvNchwcFloatKernelNeon
//

void
    MLASCALL
    MlasConvNchwcFloatKernelNeon(
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
    MlasConvFloatKernelNeonImpl<true>(
        Input,
        Filter,
        Output,
        StrideWidth,
        DilationWidth,
        FilterCount,
        InputStride,
        FilterStride,
        OutputStride,
        KernelHeight,
        KernelWidth,
        InputBase,
        InputWidth,
        DilatedInputWidth,
        OutputCountLeftPad,
        OutputCount,
        OutputCountRightPad,
        Bias,
        KernelFlags
    );
}

//
// Helper function to load input vector with bounds checking
//
static inline float32x4_t
LoadInputVectorWithBounds(
    const float* input_base,
    size_t offset,
    bool is_main_region,
    const float* InputBase,
    size_t kh,
    size_t DilatedInputWidthElements,
    size_t InputWidthElements
)
{
    if (is_main_region) {
        return MlasLoadFloat32x4(input_base + offset);
    } else {
        float input_values[4];
        for (size_t i = 0; i < 4; i++) {
            const float* input_element = input_base + offset + i;
            const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
            const float* input_row_end = input_row_start + InputWidthElements;

            if (input_element >= input_row_start && input_element < input_row_end) {
                input_values[i] = *input_element;
            } else {
                input_values[i] = 0.0f;
            }
        }
        return MlasLoadFloat32x4(input_values);
    }
}

//
// Implementation of MlasConvDepthwiseFloatKernelNeon
//
// This kernel performs depthwise separable convolution where each input channel
// is convolved with its own filter. This is more efficient than standard convolution
// for certain network architectures like MobileNets.
//

void
    MLASCALL
    MlasConvDepthwiseFloatKernelNeon(
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
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    MLAS_UNREFERENCED_PARAMETER(InputStrideElements);

    const size_t InputWidthElements = InputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        bool is_main_region = (output_idx >= OutputCountLeftPad && output_idx < OutputCountLeftPad + OutputCount);

        float32x4_t Accumulator0, Accumulator1, Accumulator2, Accumulator3;

        if (AccumulateOutput) {
            Accumulator0 = MlasLoadFloat32x4(&Output[output_idx * BlockSize]);
            Accumulator1 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 4]);
            Accumulator2 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 8]);
            Accumulator3 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 12]);
        } else {
            Accumulator0 = MlasBroadcastFloat32x4(0.0f);
            Accumulator1 = MlasBroadcastFloat32x4(0.0f);
            Accumulator2 = MlasBroadcastFloat32x4(0.0f);
            Accumulator3 = MlasBroadcastFloat32x4(0.0f);
        }

        if (BiasAddition) {
            const float32x4_t BiasVector0 = MlasLoadFloat32x4(Bias);
            const float32x4_t BiasVector1 = MlasLoadFloat32x4(Bias + 4);
            const float32x4_t BiasVector2 = MlasLoadFloat32x4(Bias + 8);
            const float32x4_t BiasVector3 = MlasLoadFloat32x4(Bias + 12);

            Accumulator0 = MlasAddFloat32x4(Accumulator0, BiasVector0);
            Accumulator1 = MlasAddFloat32x4(Accumulator1, BiasVector1);
            Accumulator2 = MlasAddFloat32x4(Accumulator2, BiasVector2);
            Accumulator3 = MlasAddFloat32x4(Accumulator3, BiasVector3);
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                size_t kernel_pos = kh * KernelWidth + kw;

                const float* input_base = Input + output_idx * StrideWidthElements +
                                          kh * DilatedInputWidthElements + kw * DilationWidthElements;

                float32x4_t InputVector0 = LoadInputVectorWithBounds(input_base, 0, is_main_region, InputBase, kh, DilatedInputWidthElements, InputWidthElements);
                float32x4_t InputVector1 = LoadInputVectorWithBounds(input_base, 4, is_main_region, InputBase, kh, DilatedInputWidthElements, InputWidthElements);
                float32x4_t InputVector2 = LoadInputVectorWithBounds(input_base, 8, is_main_region, InputBase, kh, DilatedInputWidthElements, InputWidthElements);
                float32x4_t InputVector3 = LoadInputVectorWithBounds(input_base, 12, is_main_region, InputBase, kh, DilatedInputWidthElements, InputWidthElements);

                const float32x4_t FilterVector0 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize]);
                const float32x4_t FilterVector1 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 4]);
                const float32x4_t FilterVector2 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 8]);
                const float32x4_t FilterVector3 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 12]);

                Accumulator0 = MlasMultiplyAddFloat32x4(InputVector0, FilterVector0, Accumulator0);
                Accumulator1 = MlasMultiplyAddFloat32x4(InputVector1, FilterVector1, Accumulator1);
                Accumulator2 = MlasMultiplyAddFloat32x4(InputVector2, FilterVector2, Accumulator2);
                Accumulator3 = MlasMultiplyAddFloat32x4(InputVector3, FilterVector3, Accumulator3);
            }
        }

        if (ReluActivation) {
            Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
            Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
            Accumulator2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
            Accumulator3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);
        }

        MlasStoreFloat32x4(&Output[output_idx * BlockSize], Accumulator0);
        MlasStoreFloat32x4(&Output[output_idx * BlockSize + 4], Accumulator1);
        MlasStoreFloat32x4(&Output[output_idx * BlockSize + 8], Accumulator2);
        MlasStoreFloat32x4(&Output[output_idx * BlockSize + 12], Accumulator3);
    }
}

//
// Implementation of MlasConvPointwiseFloatKernelNeon
//
// This kernel performs pointwise (1x1) convolution which is essentially
// a matrix multiplication across the channel dimension. It's optimized
// for cases where the kernel size is 1x1.
//

void
    MLASCALL
    MlasConvPointwiseFloatKernelNeon(
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
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

    for (size_t output_idx = 0; output_idx < OutputCount; output_idx++) {
        for (size_t f = 0; f < FilterCount; f++) {
            const float* filter = Filter + f * FilterStrideElements;
            float* output = Output + f * OutputStrideElements;

            float32x4_t Accumulator0, Accumulator1, Accumulator2, Accumulator3;

            if (AccumulateOutput) {
                Accumulator0 = MlasLoadFloat32x4(&output[output_idx * BlockSize]);
                Accumulator1 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 4]);
                Accumulator2 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 8]);
                Accumulator3 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 12]);
            } else {
                Accumulator0 = MlasBroadcastFloat32x4(0.0f);
                Accumulator1 = MlasBroadcastFloat32x4(0.0f);
                Accumulator2 = MlasBroadcastFloat32x4(0.0f);
                Accumulator3 = MlasBroadcastFloat32x4(0.0f);
            }

            if (BiasAddition) {
                const float32x4_t BiasVector0 = MlasLoadFloat32x4(&Bias[f * BlockSize]);
                const float32x4_t BiasVector1 = MlasLoadFloat32x4(&Bias[f * BlockSize + 4]);
                const float32x4_t BiasVector2 = MlasLoadFloat32x4(&Bias[f * BlockSize + 8]);
                const float32x4_t BiasVector3 = MlasLoadFloat32x4(&Bias[f * BlockSize + 12]);

                Accumulator0 = MlasAddFloat32x4(Accumulator0, BiasVector0);
                Accumulator1 = MlasAddFloat32x4(Accumulator1, BiasVector1);
                Accumulator2 = MlasAddFloat32x4(Accumulator2, BiasVector2);
                Accumulator3 = MlasAddFloat32x4(Accumulator3, BiasVector3);
            }

            for (size_t c = 0; c < InputChannels; c++) {
                const float* input_ptr = Input + c * InputStrideElements + output_idx * StrideWidthElements;

                for (size_t input_b = 0; input_b < BlockSize; input_b++) {
                    const float input_value = input_ptr[input_b];
                    const float32x4_t InputVector = MlasBroadcastFloat32x4(input_value);

                    const float* filter_ptr = filter + (c * BlockSize + input_b) * BlockSize;

                    const float32x4_t FilterVector0 = MlasLoadFloat32x4(filter_ptr);
                    const float32x4_t FilterVector1 = MlasLoadFloat32x4(filter_ptr + 4);
                    const float32x4_t FilterVector2 = MlasLoadFloat32x4(filter_ptr + 8);
                    const float32x4_t FilterVector3 = MlasLoadFloat32x4(filter_ptr + 12);

                    Accumulator0 = MlasMultiplyAddFloat32x4(InputVector, FilterVector0, Accumulator0);
                    Accumulator1 = MlasMultiplyAddFloat32x4(InputVector, FilterVector1, Accumulator1);
                    Accumulator2 = MlasMultiplyAddFloat32x4(InputVector, FilterVector2, Accumulator2);
                    Accumulator3 = MlasMultiplyAddFloat32x4(InputVector, FilterVector3, Accumulator3);
                }
            }

            if (ReluActivation) {
                Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
                Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
                Accumulator2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
                Accumulator3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);
            }

            MlasStoreFloat32x4(&output[output_idx * BlockSize], Accumulator0);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], Accumulator1);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], Accumulator2);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], Accumulator3);
        }
    }
}

#endif
