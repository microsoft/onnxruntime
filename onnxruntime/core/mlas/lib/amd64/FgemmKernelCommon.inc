;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   FgemmKernelCommon.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the floating
;   point matrix/matrix multiply operation (SGEMM and DGEMM).
;
;--

;
; Stack frame layout for the floating point kernels.
;

FgemmKernelFrame STRUCT

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
        SavedR12 QWORD ?
        SavedR13 QWORD ?
        SavedR14 QWORD ?
        SavedR15 QWORD ?
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
        ZeroMode QWORD ?

FgemmKernelFrame ENDS

;
; Define the number of elements per vector register.
;

FgemmXmmElementCount    EQU     (16 / FgemmElementSize)
FgemmYmmElementCount    EQU     (32 / FgemmElementSize)
FgemmZmmElementCount    EQU     (64 / FgemmElementSize)

;
; Macro Description:
;
;   This macro implements the common prologue code for the SGEMM and DGEMM
;   kernels.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string.
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
;   r15 - Stores the ZeroMode argument from the stack frame.
;

FgemmKernelEntry MACRO Isa

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r15
        alloc_stack (FgemmKernelFrame.SavedR15)
IFIDNI <Isa>, <Avx512F>
        save_reg r12,FgemmKernelFrame.SavedR12
        save_reg r13,FgemmKernelFrame.SavedR13
        save_reg r14,FgemmKernelFrame.SavedR14
ENDIF
        save_xmm128 xmm6,FgemmKernelFrame.SavedXmm6
        save_xmm128 xmm7,FgemmKernelFrame.SavedXmm7
        save_xmm128 xmm8,FgemmKernelFrame.SavedXmm8
        save_xmm128 xmm9,FgemmKernelFrame.SavedXmm9
        save_xmm128 xmm10,FgemmKernelFrame.SavedXmm10
        save_xmm128 xmm11,FgemmKernelFrame.SavedXmm11
        save_xmm128 xmm12,FgemmKernelFrame.SavedXmm12
        save_xmm128 xmm13,FgemmKernelFrame.SavedXmm13
        save_xmm128 xmm14,FgemmKernelFrame.SavedXmm14
        save_xmm128 xmm15,FgemmKernelFrame.SavedXmm15

        END_PROLOGUE

IFDIFI <Isa>, <Sse>
        vzeroall
ENDIF
        mov     rsi,rcx
        mov     rbp,FgemmKernelFrame.CountN[rsp]
        mov     rax,FgemmKernelFrame.ldc[rsp]
        shl     rax,FgemmElementShift       ; convert ldc to bytes
        mov     r10,FgemmKernelFrame.lda[rsp]
        shl     r10,FgemmElementShift       ; convert lda to bytes
        mov     r11,FgemmKernelFrame.CountM[rsp]
        movzx   r15,BYTE PTR FgemmKernelFrame.ZeroMode[rsp]

        ENDM

;
; Macro Description:
;
;   This macro implements the common epilogue code for the SGEMM and DGEMM
;   kernels.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string.
;
; Implicit Arguments:
;
;   r11d - Stores the number of rows handled.
;

FgemmKernelExit MACRO Isa

        mov     eax,r11d
        movaps  xmm6,FgemmKernelFrame.SavedXmm6[rsp]
        movaps  xmm7,FgemmKernelFrame.SavedXmm7[rsp]
        movaps  xmm8,FgemmKernelFrame.SavedXmm8[rsp]
        movaps  xmm9,FgemmKernelFrame.SavedXmm9[rsp]
        movaps  xmm10,FgemmKernelFrame.SavedXmm10[rsp]
        movaps  xmm11,FgemmKernelFrame.SavedXmm11[rsp]
        movaps  xmm12,FgemmKernelFrame.SavedXmm12[rsp]
        movaps  xmm13,FgemmKernelFrame.SavedXmm13[rsp]
        movaps  xmm14,FgemmKernelFrame.SavedXmm14[rsp]
        movaps  xmm15,FgemmKernelFrame.SavedXmm15[rsp]
IFIDNI <Isa>, <Avx512F>
        mov     r12,FgemmKernelFrame.SavedR12[rsp]
        mov     r13,FgemmKernelFrame.SavedR13[rsp]
        mov     r14,FgemmKernelFrame.SavedR14[rsp]
ENDIF
        add     rsp,(FgemmKernelFrame.SavedR15)

        BEGIN_EPILOGUE

        pop     r15
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        ENDM

;
; Macro Description:
;
;   This macro generates code to execute the block compute macro multiple
;   times and advancing the matrix A and matrix B data pointers.
;
; Arguments:
;
;   ComputeBlock - Supplies the macro to compute a single block.
;
;   RowCount - Supplies the number of rows to access from matrix A.
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

ComputeBlockLoop MACRO ComputeBlock, RowCount, AdvanceMatrixAPlusRows

        LOCAL   ComputeBlockBy4Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   OutputBlock

        mov     rdi,r9                      ; reload CountK
        sub     rdi,4
        jb      ProcessRemainingBlocks

ComputeBlockBy4Loop:
        ComputeBlock RowCount, 0, FgemmElementSize*0, 64*4
        ComputeBlock RowCount, 2*32, FgemmElementSize*1, 64*4
        add_immed rdx,2*2*32                ; advance matrix B by 128 bytes
        ComputeBlock RowCount, 0, FgemmElementSize*2, 64*4
        ComputeBlock RowCount, 2*32, FgemmElementSize*3, 64*4
        add_immed rdx,2*2*32                ; advance matrix B by 128 bytes
        add     rcx,4*FgemmElementSize      ; advance matrix A by 4 elements
IF AdvanceMatrixAPlusRows
        add     rbx,4*FgemmElementSize      ; advance matrix A plus rows by 4 elements
IF RowCount GE 12
        add     r13,4*FgemmElementSize
        add     r14,4*FgemmElementSize
ENDIF
ENDIF
        sub     rdi,4
        jae     ComputeBlockBy4Loop

ProcessRemainingBlocks:
        add     rdi,4                       ; correct for over-subtract above
        jz      OutputBlock

ComputeBlockBy1Loop:
        ComputeBlock RowCount, 0, 0
        add     rdx,2*32                    ; advance matrix B by 64 bytes
        add     rcx,FgemmElementSize        ; advance matrix A by 1 element
IF AdvanceMatrixAPlusRows
        add     rbx,FgemmElementSize        ; advance matrix A plus rows by 1 element
IF RowCount GE 12
        add     r13,FgemmElementSize
        add     r14,FgemmElementSize
ENDIF
ENDIF
        dec     rdi
        jne     ComputeBlockBy1Loop

OutputBlock:

        ENDM
