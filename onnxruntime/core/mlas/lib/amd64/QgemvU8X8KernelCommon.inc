;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemvU8X8KernelCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the quantized
;   integer matrix/matrix multiply operation (QGEMM) for the AVX2 kernels.
;
;--

GemvU8X8KernelFrame STRUCT

        SavedRdi QWORD ?
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        CountN QWORD ?
        ldb QWORD ?

GemvU8X8KernelFrame ENDS
