/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_kernel_neon.cpp

Abstract:

    This module implements the single precision convolution kernels for ARM NEON.

--*/

#if defined(MLAS_USE_ARM_NEON_NCHWC)

#include <vector>
#include "mlasi.h"
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
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);
    const float32x4_t AccumulateMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT)));
    const float32x4_t BiasMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION)));
    const float32x4_t ReluMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION)));

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    MLAS_UNREFERENCED_PARAMETER(InputStride);

    const size_t InputWidthElements = InputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {

        float32x4_t OldOutput0 = MlasLoadFloat32x4(&Output[output_idx * BlockSize]);
        float32x4_t OldOutput1 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 4]);
        float32x4_t OldOutput2 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 8]);
        float32x4_t OldOutput3 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 12]);

        float32x4_t Accumulator0 = MlasBroadcastFloat32x4(0.0f);
        float32x4_t Accumulator1 = MlasBroadcastFloat32x4(0.0f);
        float32x4_t Accumulator2 = MlasBroadcastFloat32x4(0.0f);
        float32x4_t Accumulator3 = MlasBroadcastFloat32x4(0.0f);

        const float32x4_t BiasVector0 = MlasLoadFloat32x4(Bias);
        const float32x4_t BiasVector1 = MlasLoadFloat32x4(Bias + 4);
        const float32x4_t BiasVector2 = MlasLoadFloat32x4(Bias + 8);
        const float32x4_t BiasVector3 = MlasLoadFloat32x4(Bias + 12);

        for (size_t kernel_pos = 0; kernel_pos < KernelHeight * KernelWidth; kernel_pos++) {
            size_t kh = kernel_pos / KernelWidth;
            size_t kw = kernel_pos % KernelWidth;

            const float* input_base = Input + output_idx * StrideWidthElements +
                                      kh * DilatedInputWidthElements + kw * DilationWidthElements;

            float32x4_t InputVector0 = LoadInputVectorWithBounds(input_base, 0, InputBase, kh, DilatedInputWidthElements, InputWidthElements);
            float32x4_t InputVector1 = LoadInputVectorWithBounds(input_base, 4, InputBase, kh, DilatedInputWidthElements, InputWidthElements);
            float32x4_t InputVector2 = LoadInputVectorWithBounds(input_base, 8, InputBase, kh, DilatedInputWidthElements, InputWidthElements);
            float32x4_t InputVector3 = LoadInputVectorWithBounds(input_base, 12, InputBase, kh, DilatedInputWidthElements, InputWidthElements);

            const float32x4_t FilterVector0 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize]);
            const float32x4_t FilterVector1 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 4]);
            const float32x4_t FilterVector2 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 8]);
            const float32x4_t FilterVector3 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 12]);

            Accumulator0 = MlasMultiplyAddFloat32x4(InputVector0, FilterVector0, Accumulator0);
            Accumulator1 = MlasMultiplyAddFloat32x4(InputVector1, FilterVector1, Accumulator1);
            Accumulator2 = MlasMultiplyAddFloat32x4(InputVector2, FilterVector2, Accumulator2);
            Accumulator3 = MlasMultiplyAddFloat32x4(InputVector3, FilterVector3, Accumulator3);
        }

        Accumulator0 = MlasAddFloat32x4(Accumulator0, MlasAndFloat32x4(OldOutput0, AccumulateMask));
        Accumulator1 = MlasAddFloat32x4(Accumulator1, MlasAndFloat32x4(OldOutput1, AccumulateMask));
        Accumulator2 = MlasAddFloat32x4(Accumulator2, MlasAndFloat32x4(OldOutput2, AccumulateMask));
        Accumulator3 = MlasAddFloat32x4(Accumulator3, MlasAndFloat32x4(OldOutput3, AccumulateMask));

        Accumulator0 = MlasAddFloat32x4(Accumulator0, MlasAndFloat32x4(BiasVector0, BiasMask));
        Accumulator1 = MlasAddFloat32x4(Accumulator1, MlasAndFloat32x4(BiasVector1, BiasMask));
        Accumulator2 = MlasAddFloat32x4(Accumulator2, MlasAndFloat32x4(BiasVector2, BiasMask));
        Accumulator3 = MlasAddFloat32x4(Accumulator3, MlasAndFloat32x4(BiasVector3, BiasMask));

        float32x4_t Relu0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
        float32x4_t Relu1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
        float32x4_t Relu2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
        float32x4_t Relu3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);

        Accumulator0 = MlasBlendFloat32x4(Accumulator0, Relu0, ReluMask);
        Accumulator1 = MlasBlendFloat32x4(Accumulator1, Relu1, ReluMask);
        Accumulator2 = MlasBlendFloat32x4(Accumulator2, Relu2, ReluMask);
        Accumulator3 = MlasBlendFloat32x4(Accumulator3, Relu3, ReluMask);

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
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    MLAS_UNREFERENCED_PARAMETER(InputStride);

    std::vector<MLAS_SGEMM_DATA_PARAMS> gemm_params(FilterCount);
    
    for (size_t f = 0; f < FilterCount; f++) {
        const float* filter = Filter + f * FilterStrideElements;
        float* output = Output + f * OutputStrideElements;

        if (BiasAddition) {
            for (size_t output_idx = 0; output_idx < OutputCount; output_idx++) {
                for (size_t i = 0; i < BlockSize; i++) {
                    output[output_idx * BlockSize + i] = Bias[f * BlockSize + i];
                }
            }
        }

        gemm_params[f].A = Input;
        gemm_params[f].B = filter;
        gemm_params[f].C = output;
        gemm_params[f].lda = StrideWidthElements;
        gemm_params[f].ldb = BlockSize;
        gemm_params[f].ldc = BlockSize;
        gemm_params[f].alpha = 1.0f;
        gemm_params[f].beta = BiasAddition ? 1.0f : 0.0f;
        gemm_params[f].BIsPacked = false;
    }

    MlasGemmBatch(CblasNoTrans, CblasNoTrans, OutputCount, BlockSize, InputChannels * BlockSize, 
                  gemm_params.data(), FilterCount, nullptr);

    if (ReluActivation) {
        const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);
        
        for (size_t f = 0; f < FilterCount; f++) {
            float* output = Output + f * OutputStrideElements;
            
            for (size_t output_idx = 0; output_idx < OutputCount; output_idx++) {
                float32x4_t Accumulator0 = MlasLoadFloat32x4(&output[output_idx * BlockSize]);
                float32x4_t Accumulator1 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 4]);
                float32x4_t Accumulator2 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 8]);
                float32x4_t Accumulator3 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 12]);

                Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
                Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
                Accumulator2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
                Accumulator3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);

                MlasStoreFloat32x4(&output[output_idx * BlockSize], Accumulator0);
                MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], Accumulator1);
                MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], Accumulator2);
                MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], Accumulator3);
            }
        }
    }
}

#endif
