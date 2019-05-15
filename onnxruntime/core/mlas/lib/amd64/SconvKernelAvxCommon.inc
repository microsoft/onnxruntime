;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SconvKernelAvxCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the single
;   precision convolution operation for the AVX and FMA3 kernels.
;
;--

INCLUDE SconvKernelCommon.inc

;
; Macro Description:
;
;   This macro generates code to clear the block accumulators.
;
; Arguments:
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
; Implicit Arguments:
;
;   ymm0-ymm11 - Supplies the block accumulators.
;

ClearBlock MACRO FilterCount, OutputCount

        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vxorps xmm0,xmm0,xmm0>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vxorps xmm4,xmm4,xmm4>
        EmitIfCount2GE FilterCount, 1, OutputCount, 3, <vxorps xmm8,xmm8,xmm8>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vxorps xmm1,xmm1,xmm1>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vxorps xmm5,xmm5,xmm5>
        EmitIfCount2GE FilterCount, 2, OutputCount, 3, <vxorps xmm9,xmm9,xmm9>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vxorps xmm2,xmm2,xmm2>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vxorps xmm6,xmm6,xmm6>
        EmitIfCount2GE FilterCount, 3, OutputCount, 3, <vxorps xmm10,xmm10,xmm10>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vxorps xmm3,xmm3,xmm3>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vxorps xmm7,xmm7,xmm7>
        EmitIfCount2GE FilterCount, 4, OutputCount, 3, <vxorps xmm11,xmm11,xmm11>

        ENDM
