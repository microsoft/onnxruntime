;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the single
;   precision matrix/matrix multiply operation (SGEMM).
;
;--

;
; Stack frame layout for the SGEMM kernels.
;

SgemmKernelFrame STRUCT

        SavedEdi DWORD ?
        SavedEsi DWORD ?
        SavedEbx DWORD ?
        SavedEbp DWORD ?
        ReturnAddress DWORD ?
        MatrixA DWORD ?
        MatrixB DWORD ?
        MatrixC DWORD ?
        CountK DWORD ?
        CountM DWORD ?
        CountN DWORD ?
        lda DWORD ?
        ldc DWORD ?
        Alpha DWORD ?
        ZeroMode DWORD ?

SgemmKernelFrame ENDS

;
; Macro Description:
;
;   This macro implements the common prologue code for the SGEMM kernels.
;
; Arguments:
;
;   None.
;
; Return Registers:
;
;   ecx - Stores the address of the matrix A data from the stack frame.
;
;   edx - Stores the address of the matrix B data from the stack frame.
;
;   ebp - Stores the CountN argument from the stack frame.
;
;   ebx, esi, edi - Previous values stored on the stack and the registers are
;       available as temporaries.
;

SgemmKernelEntry MACRO

        push    ebp
        push    ebx
        push    esi
        push    edi
        mov     edx,SgemmKernelFrame.MatrixB[esp]
        mov     esi,SgemmKernelFrame.MatrixC[esp]
        mov     ebp,SgemmKernelFrame.CountN[esp]

cPublicFpo ((SgemmKernelFrame.ReturnAddress)/4),10

        ENDM

;
; Macro Description:
;
;   This macro implements the common epilogue code for the SGEMM kernels.
;
; Arguments:
;
;   None.
;

SgemmKernelExit MACRO

        pop     edi
        pop     esi
        pop     ebx
        pop     ebp

        ENDM
