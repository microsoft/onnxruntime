/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    ConvSymKernelCommon.h

Abstract:

    This module contains common kernel macros and structures for the symmetric
    quantized integer convolution operation.

--*/

//
// Define the convolution kernel flags.
//

#define MLAS_CONV_SYM_FLAG_INPUT_DIRECT         0x00000001
#define MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE    0x00000002

//
// Define the structure of the post process parameter block.
//

        .equ    .LConvSymPostProcessParams_Bias, 0
        .equ    .LConvSymPostProcessParams_Scale, 8
        .equ    .LConvSymPostProcessParams_MinimumValue, 16
        .equ    .LConvSymPostProcessParams_MaximumValue, 20
        .equ    .LConvSymPostProcessParams_OutputZeroPoint, 24

//
// Stack frame layout for the symmetric convolution kernels.
//

        .equ    .LConvSymKernelFrame_InputChannels, 0
        .equ    .LConvSymKernelFrame_OutputChannels, 8
        .equ    .LConvSymKernelFrame_Padding, 16
        .equ    .LConvSymKernelFrame_SavedR15, 24
        .equ    .LConvSymKernelFrame_SavedR14, 32
        .equ    .LConvSymKernelFrame_SavedR13, 40
        .equ    .LConvSymKernelFrame_SavedR12, 48
        .equ    .LConvSymKernelFrame_SavedRbx, 56
        .equ    .LConvSymKernelFrame_SavedRbp, 64
        .equ    .LConvSymKernelFrame_ReturnAddress, 72
        .equ    .LConvSymKernelFrame_ChannelCount, 80
        .equ    .LConvSymKernelFrame_OutputCount, 88
        .equ    .LConvSymKernelFrame_PostProcessParams, 96
        .equ    .LConvSymKernelFrame_KernelFlags, 104

        .equ    .LConvSymDepthwiseKernelFrame_Channels, 0
        .equ    .LConvSymDepthwiseKernelFrame_ChannelOffset, 8
        .equ    .LConvSymDepthwiseKernelFrame_Padding, 16
        .equ    .LConvSymDepthwiseKernelFrame_SavedR15, 24
        .equ    .LConvSymDepthwiseKernelFrame_SavedR14, 32
        .equ    .LConvSymDepthwiseKernelFrame_SavedR13, 40
        .equ    .LConvSymDepthwiseKernelFrame_SavedR12, 48
        .equ    .LConvSymDepthwiseKernelFrame_SavedRbx, 56
        .equ    .LConvSymDepthwiseKernelFrame_SavedRbp, 64
        .equ    .LConvSymDepthwiseKernelFrame_ReturnAddress, 72
        .equ    .LConvSymDepthwiseKernelFrame_ChannelCount, 80
        .equ    .LConvSymDepthwiseKernelFrame_OutputCount, 88
        .equ    .LConvSymDepthwiseKernelFrame_PostProcessParams, 96
        .equ    .LConvSymDepthwiseKernelFrame_KernelFlags, 104
