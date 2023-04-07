;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   ConvSymKernelCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the symmetric
;   quantized integer convolution operation.
;
;--

;
; Define the convolution kernel flags.
;

MLAS_CONV_SYM_FLAG_INPUT_DIRECT             EQU     00000001h
MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE        EQU     00000002h

;
; Define the structure of the post process parameter block.
;

ConvSymPostProcessParams STRUCT

        Bias QWORD ?
        Scale QWORD ?
        MinimumValue DWORD ?
        MaximumValue DWORD ?
        OutputZeroPoint DWORD ?

ConvSymPostProcessParams ENDS

;
; Stack frame layout for the symmetric convolution kernels.
;

ConvSymKernelFrame STRUCT

        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
        SavedXmm11 OWORD ?
        SavedXmm12 OWORD ?
        SavedXmm13 OWORD ?
        SavedXmm14 OWORD ?
        SavedXmm15 OWORD ?
        Padding QWORD ?
        SavedR15 QWORD ?
        SavedR14 QWORD ?
        SavedR13 QWORD ?
        SavedR12 QWORD ?
        SavedRdi QWORD ?
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        InputChannels QWORD ?
        OutputChannels QWORD ?
        ChannelCount QWORD ?
        OutputCount QWORD ?
        PostProcessParams QWORD ?
        KernelFlags QWORD ?

ConvSymKernelFrame ENDS

ConvSymDepthwiseKernelFrame STRUCT

        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
        SavedXmm11 OWORD ?
        SavedXmm12 OWORD ?
        SavedXmm13 OWORD ?
        SavedXmm14 OWORD ?
        SavedXmm15 OWORD ?
        Padding QWORD ?
        SavedR15 QWORD ?
        SavedR14 QWORD ?
        SavedR13 QWORD ?
        SavedR12 QWORD ?
        SavedRdi QWORD ?
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        Channels QWORD ?
        ChannelOffset QWORD ?
        ChannelCount QWORD ?
        OutputCount QWORD ?
        PostProcessParams QWORD ?
        KernelFlags QWORD ?

ConvSymDepthwiseKernelFrame ENDS
