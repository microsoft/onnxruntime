;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   TransKernelCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the
;   transcendental functions.
;
;--

;
; Structure layout for the exponential function constants block.
;

ExpConstants STRUCT

        LowerRange DWORD ?
        UpperRange DWORD ?
        LowerRangeSumExp DWORD ?
        UpperRangeSumExp DWORD ?
        RoundingBias DWORD ?
        Log2Reciprocal DWORD ?
        Log2High DWORD ?
        Log2Low DWORD ?
        poly_0 DWORD ?
        poly_1 DWORD ?
        poly_2 DWORD ?
        poly_3 DWORD ?
        poly_4 DWORD ?
        poly_56 DWORD ?
        MinimumExponent DWORD ?
        MaximumExponent DWORD ?

ExpConstants ENDS

;
; Stack frame layout for the transcedental functions.
;

TransKernelFrame STRUCT

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
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?

TransKernelFrame ENDS
