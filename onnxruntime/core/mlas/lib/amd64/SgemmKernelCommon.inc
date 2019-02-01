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
        SavedR12 QWORD ?
        SavedR13 QWORD ?
        SavedR14 QWORD ?
        SavedRdi QWORD ?
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        CountM QWORD ?
        CountN QWORD ?
        lda QWORD ?
        ldc QWORD ?
        Alpha QWORD ?

SgemmKernelFrame ENDS

;
; Stack frame layout for the SGEMM M=1 kernels.
;

SgemmKernelM1Frame STRUCT

        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
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
        Beta QWORD ?

SgemmKernelM1Frame ENDS

;
; SgemmKernelAvxEntry
;
; Macro Description:
;
;   This macro implements the common prologue code for the AVX and FMA3
;   SGEMM kernels.
;
; Arguments:
;
;   SaveExtra - Supplies a non-blank value if registers r12-r14 should also be
;       saved to make available as temporaries.
;
; Return Registers:
;
;   rax - Stores the length in bytes of a row from matrix C.
;
;   rsi - Stores the address of the matrix A data.
;
;   rbp - Stores the CountN argument from the stack frame.
;
;   r10 - Stores the length in bytes of a row from matrix A.
;
;   r11 - Stores the CountM argument from the stack frame.
;
;   rbx, rsi, rdi - Previous values stored on the stack and the registers
;       are available as temporaries.
;

SgemmKernelAvxEntry MACRO SaveExtra

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        alloc_stack (SgemmKernelFrame.SavedRdi)
IFNB <SaveExtra>
        save_reg r12, SgemmKernelFrame.SavedR12
        save_reg r13, SgemmKernelFrame.SavedR13
        save_reg r14, SgemmKernelFrame.SavedR14
ENDIF
        save_xmm128_avx xmm6,SgemmKernelFrame.SavedXmm6
        save_xmm128_avx xmm7,SgemmKernelFrame.SavedXmm7
        save_xmm128_avx xmm8,SgemmKernelFrame.SavedXmm8
        save_xmm128_avx xmm9,SgemmKernelFrame.SavedXmm9
        save_xmm128_avx xmm10,SgemmKernelFrame.SavedXmm10
        save_xmm128_avx xmm11,SgemmKernelFrame.SavedXmm11
        save_xmm128_avx xmm12,SgemmKernelFrame.SavedXmm12
        save_xmm128_avx xmm13,SgemmKernelFrame.SavedXmm13
        save_xmm128_avx xmm14,SgemmKernelFrame.SavedXmm14
        save_xmm128_avx xmm15,SgemmKernelFrame.SavedXmm15

        END_PROLOGUE

        vzeroall

        mov     rsi,rcx
        mov     rbp,SgemmKernelFrame.CountN[rsp]
        mov     rax,SgemmKernelFrame.ldc[rsp]
        shl     rax,2
        mov     r10,SgemmKernelFrame.lda[rsp]
        shl     r10,2
        mov     r11,SgemmKernelFrame.CountM[rsp]

        ENDM

;
; SgemmKernelAvxExit
;
; Macro Description:
;
;   This macro implements the common epilogue code for the AVX and FMA3
;   SGEMM kernels.
;
; Arguments:
;
;   RestoreExtra - Supplies a non-blank value if registers r12-r14 should also
;       be restored.
;
; Implicit Arguments:
;
;   r11d - Stores the number of rows handled.
;

SgemmKernelAvxExit MACRO RestoreExtra

        mov     eax,r11d
        vmovaps xmm6,SgemmKernelFrame.SavedXmm6[rsp]
        vmovaps xmm7,SgemmKernelFrame.SavedXmm7[rsp]
        vmovaps xmm8,SgemmKernelFrame.SavedXmm8[rsp]
        vmovaps xmm9,SgemmKernelFrame.SavedXmm9[rsp]
        vmovaps xmm10,SgemmKernelFrame.SavedXmm10[rsp]
        vmovaps xmm11,SgemmKernelFrame.SavedXmm11[rsp]
        vmovaps xmm12,SgemmKernelFrame.SavedXmm12[rsp]
        vmovaps xmm13,SgemmKernelFrame.SavedXmm13[rsp]
        vmovaps xmm14,SgemmKernelFrame.SavedXmm14[rsp]
        vmovaps xmm15,SgemmKernelFrame.SavedXmm15[rsp]
IFNB <RestoreExtra>
        mov     r12,SgemmKernelFrame.SavedR12[rsp]
        mov     r13,SgemmKernelFrame.SavedR13[rsp]
        mov     r14,SgemmKernelFrame.SavedR14[rsp]
ENDIF
        add     rsp,(SgemmKernelFrame.SavedRdi)

        BEGIN_EPILOGUE

        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        ENDM

;
; ComputeBlockLoop
;
;   This macro generates code to execute the block compute macro multiple
;   times and advancing the matrix A and matrix B data pointers.
;
; Arguments:
;
;   ComputeBlock - Supplies the macro to compute a single block.
;
;   Count - Supplies the number of rows to access from matrix A.
;
;   AdvanceMatrixAPlusRows - Supplies a non-zero value if the data pointer
;       in rbx should also be advanced as part of the loop.
;
; Implicit Arguments:
;
;   rbx - Supplies the address into the matrix A data plus N rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r9 - Supplies the number of columns from matrix A and the number of rows
;       from matrix B to iterate over.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockLoop MACRO ComputeBlock, Count, AdvanceMatrixAPlusRows

        LOCAL   ComputeBlockBy4Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   OutputBlock

        mov     rdi,r9                      ; reload CountK
        sub     rdi,4
        jb      ProcessRemainingBlocks

ComputeBlockBy4Loop:
        ComputeBlock Count, 0, 0, 64*4
        ComputeBlock Count, 16*4, 4, 64*4
        sub     rdx,-32*4                   ; advance matrix B by 32 columns
        ComputeBlock Count, 0, 8, 64*4
        ComputeBlock Count, 16*4, 12, 64*4
        sub     rdx,-32*4                   ; advance matrix B by 32 columns
        add     rcx,4*4                     ; advance matrix A by 4 columns
IF AdvanceMatrixAPlusRows
        add     rbx,4*4                     ; advance matrix A plus rows by 4 columns
IF Count GE 12
        add     r13,4*4
        add     r14,4*4
ENDIF
ENDIF
        sub     rdi,4
        jae     ComputeBlockBy4Loop

ProcessRemainingBlocks:
        add     rdi,4                       ; correct for over-subtract above
        jz      OutputBlock

ComputeBlockBy1Loop:
        ComputeBlock Count, 0, 0
        add     rdx,16*4                    ; advance matrix B by 16 columns
        add     rcx,4                       ; advance matrix A by 1 column
IF AdvanceMatrixAPlusRows
        add     rbx,4                       ; advance matrix A plus rows by 1 column
IF Count GE 12
        add     r13,4
        add     r14,4
ENDIF
ENDIF
        dec     rdi
        jne     ComputeBlockBy1Loop

OutputBlock:

        ENDM
