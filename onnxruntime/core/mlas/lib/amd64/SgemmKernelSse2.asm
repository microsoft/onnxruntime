;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelSse2.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;   This implementation uses SSE2 instructions.
;
;--

        .xlist
INCLUDE macamd64.inc
INCLUDE SgemmKernelCommon.inc
        .list

;
; ComputeBlocksSseBy16
;
; Macro Description:
;
;   This macro multiplies and accumulates for a 16xN block (where N is 1,2)
;   of the output matrix.
;
; Arguments:
;
;   Count - Supplies the number of rows to access from matrix A.
;
;   VectorOffset - Supplies the byte offset from matrix B to fetch elements.
;
;   Shuffle - Supplies the shuffle mask to extract the element from matrix A.
;
; Implicit Arguments:
;
;   rdx - Supplies the address into the matrix B data.
;
;   xmm0-xmm1 - Supplies up to four elements loaded from matrix A and matrix A
;       plus one row.
;
;   xmm8-xmm15 - Supplies the block accumulators.
;

ComputeBlockSseBy16 MACRO Count, VectorOffset, Shuffle

        movaps  xmm4,XMMWORD PTR [rdx+VectorOffset]
        movaps  xmm5,XMMWORD PTR [rdx+VectorOffset+16]
        pshufd  xmm2,xmm0,Shuffle
IF Count EQ 2
        pshufd  xmm3,xmm1,Shuffle
        movaps  xmm6,xmm4
        movaps  xmm7,xmm5
ENDIF
        mulps   xmm4,xmm2
        mulps   xmm5,xmm2
        addps   xmm8,xmm4
        addps   xmm9,xmm5
IF Count EQ 2
        mulps   xmm6,xmm3
        mulps   xmm7,xmm3
        addps   xmm12,xmm6
        addps   xmm13,xmm7
ENDIF
        movaps  xmm4,XMMWORD PTR [rdx+VectorOffset+32]
        movaps  xmm5,XMMWORD PTR [rdx+VectorOffset+48]
IF Count EQ 2
        movaps  xmm6,xmm4
        movaps  xmm7,xmm5
ENDIF
        mulps   xmm4,xmm2
        mulps   xmm5,xmm2
        addps   xmm10,xmm4
        addps   xmm11,xmm5
IF Count EQ 2
        mulps   xmm6,xmm3
        mulps   xmm7,xmm3
        addps   xmm14,xmm6
        addps   xmm15,xmm7
ENDIF

        ENDM

;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows.
;
; Arguments:
;
;   A (rcx) - Supplies the address of matrix A.
;
;   B (rdx) - Supplies the address of matrix B. The matrix data has been packed
;       using MlasSgemmCopyPackB or MlasSgemmTransposePackB.
;
;   C (r8) - Supplies the address of matrix C.
;
;   CountK (r9d) - Supplies the number of columns from matrix A and the number
;       of rows from matrix B to iterate over.
;
;   CountM - Supplies the maximum number of rows that can be processed for
;       matrix A and matrix C. The actual number of rows handled for this
;       invocation depends on the kernel implementation.
;
;   CountN - Supplies the number of columns from matrix B and matrix C to iterate
;       over.
;
;   lda - Supplies the first dimension of matrix A.
;
;   ldc - Supplies the first dimension of matrix C.
;
;   Alpha - Supplies the scaler multiplier (see SGEMM definition).
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

SgemmKernelSseFunction MACRO Mode

        NESTED_ENTRY MlasSgemmKernel&Mode&Sse, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        alloc_stack (SgemmKernelFrame.SavedRdi)
        save_xmm128 xmm6,SgemmKernelFrame.SavedXmm6
        save_xmm128 xmm7,SgemmKernelFrame.SavedXmm7
        save_xmm128 xmm8,SgemmKernelFrame.SavedXmm8
        save_xmm128 xmm9,SgemmKernelFrame.SavedXmm9
        save_xmm128 xmm10,SgemmKernelFrame.SavedXmm10
        save_xmm128 xmm11,SgemmKernelFrame.SavedXmm11
        save_xmm128 xmm12,SgemmKernelFrame.SavedXmm12
        save_xmm128 xmm13,SgemmKernelFrame.SavedXmm13
        save_xmm128 xmm14,SgemmKernelFrame.SavedXmm14
        save_xmm128 xmm15,SgemmKernelFrame.SavedXmm15

        END_PROLOGUE

        mov     rsi,rcx
        mov     rbp,SgemmKernelFrame.CountN[rsp]
        mov     rax,SgemmKernelFrame.ldc[rsp]
        shl     rax,2

;
; Process 2 rows of the matrices.
;

        cmp     QWORD PTR SgemmKernelFrame.CountM[rsp],2
        jb      ProcessCountMLessThan2
        mov     r10,SgemmKernelFrame.lda[rsp]
        shl     r10,2
        mov     r11d,2                      ; return 2 rows handled

ProcessNextColumnLoop16x2:
        xorps   xmm8,xmm8                   ; clear block accumulators
        xorps   xmm9,xmm9
        xorps   xmm10,xmm10
        xorps   xmm11,xmm11
        xorps   xmm12,xmm12
        xorps   xmm13,xmm13
        xorps   xmm14,xmm14
        xorps   xmm15,xmm15
        mov     rdi,r9                      ; reload CountK
        sub     rdi,4
        jb      ProcessRemaining16x2Blocks

Compute16x2BlockBy4Loop:
        movups  xmm0,XMMWORD PTR [rcx]
        movups  xmm1,XMMWORD PTR [rcx+r10]
        ComputeBlockSseBy16 2, 0, 000h
        ComputeBlockSseBy16 2, 16*4, 055h
        sub     rdx,-32*4                   ; advance matrix B by 32 columns
        ComputeBlockSseBy16 2, 0, 0AAh
        ComputeBlockSseBy16 2, 16*4, 0FFh
        sub     rdx,-32*4                   ; advance matrix B by 32 columns
        add     rcx,4*4                     ; advance matrix A by 4 columns
        sub     rdi,4
        jae     Compute16x2BlockBy4Loop

ProcessRemaining16x2Blocks:
        add     rdi,4                       ; correct for over-subtract above
        jz      Output16x2Block

Compute16x2BlockBy1Loop:
        movss   xmm0,DWORD PTR [rcx]
        movss   xmm1,DWORD PTR [rcx+r10]
        ComputeBlockSseBy16 2, 0, 000h
        add     rdx,16*4                    ; advance matrix B by 16 columns
        add     rcx,4                       ; advance matrix A by 1 column
        dec     rdi
        jne     Compute16x2BlockBy1Loop

Output16x2Block:
        movss   xmm2,DWORD PTR SgemmKernelFrame.Alpha[rsp]
        lea     rcx,[r8+rax]                ; compute matrix C plus 1 row
        shufps  xmm2,xmm2,0
        mulps   xmm8,xmm2                   ; multiply by alpha
        mulps   xmm9,xmm2
        mulps   xmm10,xmm2
        mulps   xmm11,xmm2
        mulps   xmm12,xmm2
        mulps   xmm13,xmm2
        mulps   xmm14,xmm2
        mulps   xmm15,xmm2
        sub     rbp,16
        jb      OutputPartial16x2Block
IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [r8]
        movups  xmm1,XMMWORD PTR [r8+16]
        movups  xmm2,XMMWORD PTR [r8+32]
        movups  xmm3,XMMWORD PTR [r8+48]
        movups  xmm4,XMMWORD PTR [rcx]
        movups  xmm5,XMMWORD PTR [rcx+16]
        movups  xmm6,XMMWORD PTR [rcx+32]
        movups  xmm7,XMMWORD PTR [rcx+48]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
        addps   xmm10,xmm2
        addps   xmm11,xmm3
        addps   xmm12,xmm4
        addps   xmm13,xmm5
        addps   xmm14,xmm6
        addps   xmm15,xmm7
ENDIF
        movups  XMMWORD PTR [r8],xmm8
        movups  XMMWORD PTR [r8+16],xmm9
        movups  XMMWORD PTR [r8+32],xmm10
        movups  XMMWORD PTR [r8+48],xmm11
        movups  XMMWORD PTR [rcx],xmm12
        movups  XMMWORD PTR [rcx+16],xmm13
        movups  XMMWORD PTR [rcx+32],xmm14
        movups  XMMWORD PTR [rcx+48],xmm15
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        test    rbp,rbp
        jnz     ProcessNextColumnLoop16x2

;
; Restore non-volatile registers and return.
;

ExitKernel:
        mov     eax,r11d
        movaps  xmm6,SgemmKernelFrame.SavedXmm6[rsp]
        movaps  xmm7,SgemmKernelFrame.SavedXmm7[rsp]
        movaps  xmm8,SgemmKernelFrame.SavedXmm8[rsp]
        movaps  xmm9,SgemmKernelFrame.SavedXmm9[rsp]
        movaps  xmm10,SgemmKernelFrame.SavedXmm10[rsp]
        movaps  xmm11,SgemmKernelFrame.SavedXmm11[rsp]
        movaps  xmm12,SgemmKernelFrame.SavedXmm12[rsp]
        movaps  xmm13,SgemmKernelFrame.SavedXmm13[rsp]
        movaps  xmm14,SgemmKernelFrame.SavedXmm14[rsp]
        movaps  xmm15,SgemmKernelFrame.SavedXmm15[rsp]
        add     rsp,(SgemmKernelFrame.SavedRdi)

        BEGIN_EPILOGUE

        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

;
; Output a partial 16x2 block to the matrix.
;

OutputPartial16x2Block:
        add     rbp,16                      ; correct for over-subtract above
        cmp     rbp,4
        jb      OutputPartialLessThan4x2Block
        cmp     rbp,8
        jb      OutputPartialLessThan8x2Block
        cmp     rbp,12
        jb      OutputPartialLessThan12x2Block

IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [r8]
        movups  xmm1,XMMWORD PTR [r8+16]
        movups  xmm2,XMMWORD PTR [r8+32]
        movups  xmm3,XMMWORD PTR [rcx]
        movups  xmm4,XMMWORD PTR [rcx+16]
        movups  xmm5,XMMWORD PTR [rcx+32]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
        addps   xmm10,xmm2
        addps   xmm12,xmm3
        addps   xmm13,xmm4
        addps   xmm14,xmm5
ENDIF
        movups  XMMWORD PTR [r8],xmm8
        movups  XMMWORD PTR [r8+16],xmm9
        movups  XMMWORD PTR [r8+32],xmm10
        movups  XMMWORD PTR [rcx],xmm12
        movups  XMMWORD PTR [rcx+16],xmm13
        movups  XMMWORD PTR [rcx+32],xmm14
        and     ebp,3                       ; check if remaining count is small
        jz      ExitKernel
        movaps  xmm8,xmm11                  ; shift remaining elements down
        movaps  xmm12,xmm15
        add     r8,12*4                     ; advance matrix C by 12 columns
        add     rcx,12*4                    ; advance matrix C plus 1 row by 12 columns
        jmp     OutputPartialLessThan4x2Block

OutputPartialLessThan12x2Block:
IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [r8]
        movups  xmm1,XMMWORD PTR [r8+16]
        movups  xmm2,XMMWORD PTR [rcx]
        movups  xmm3,XMMWORD PTR [rcx+16]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
        addps   xmm12,xmm2
        addps   xmm13,xmm3
ENDIF
        movups  XMMWORD PTR [r8],xmm8
        movups  XMMWORD PTR [r8+16],xmm9
        movups  XMMWORD PTR [rcx],xmm12
        movups  XMMWORD PTR [rcx+16],xmm13
        and     ebp,3                       ; check if remaining count is small
        jz      ExitKernel
        movaps  xmm8,xmm10                  ; shift remaining elements down
        movaps  xmm12,xmm14
        add     r8,8*4                      ; advance matrix C by 8 columns
        add     rcx,8*4                     ; advance matrix C plus 1 row by 8 columns
        jmp     OutputPartialLessThan4x2Block

OutputPartialLessThan8x2Block:
IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [r8]
        movups  xmm1,XMMWORD PTR [rcx]
        addps   xmm8,xmm0
        addps   xmm12,xmm1
ENDIF
        movups  XMMWORD PTR [r8],xmm8
        movups  XMMWORD PTR [rcx],xmm12
        and     ebp,3                       ; check if remaining count is small
        jz      ExitKernel
        movaps  xmm8,xmm9                   ; shift remaining elements down
        movaps  xmm12,xmm13
        add     r8,4*4                      ; advance matrix C by 4 columns
        add     rcx,4*4                     ; advance matrix C plus 1 row by 4 columns

OutputPartialLessThan4x2Block:
        cmp     rbp,2
        jb      OutputPartial1x2Block
IFIDNI <Mode>, <Add>
        movsd   xmm0,QWORD PTR [r8]
        movsd   xmm1,QWORD PTR [rcx]
        addps   xmm8,xmm0
        addps   xmm12,xmm1
ENDIF
        movsd   QWORD PTR [r8],xmm8
        movsd   QWORD PTR [rcx],xmm12
        and     ebp,1                       ; check if remaining count is odd
        jz      ExitKernel
        movhlps xmm8,xmm8                   ; shift third element down
        movhlps xmm12,xmm12
        add     r8,2*4                      ; advance matrix C by 2 columns
        add     rcx,2*4                     ; advance matrix C plus 1 row by 2 columns

OutputPartial1x2Block:
IFIDNI <Mode>, <Add>
        addss   xmm8,DWORD PTR [r8]
        addss   xmm12,DWORD PTR [rcx]
ENDIF
        movss   DWORD PTR [r8],xmm8
        movss   DWORD PTR [rcx],xmm12
        jmp     ExitKernel

;
; Process 1 row of the matrices.
;

ProcessCountMLessThan2:
        mov     r11d,1                      ; return 1 row handled

ProcessNextColumnLoop16x1:
        xorps   xmm8,xmm8                   ; clear block accumulators
        xorps   xmm9,xmm9
        xorps   xmm10,xmm10
        xorps   xmm11,xmm11
        mov     rdi,r9                      ; reload CountK
        sub     rdi,4
        jb      ProcessRemaining16x1Blocks

Compute16x1BlockBy4Loop:
        movups  xmm0,XMMWORD PTR [rcx]
        ComputeBlockSseBy16 1, 0, 000h
        ComputeBlockSseBy16 1, 16*4, 055h
        sub     rdx,-32*4                   ; advance matrix B by 32 columns
        ComputeBlockSseBy16 1, 0, 0AAh
        ComputeBlockSseBy16 1, 16*4, 0FFh
        sub     rdx,-32*4                   ; advance matrix B by 32 columns
        add     rcx,4*4                     ; advance matrix A by 4 columns
        sub     rdi,4
        jae     Compute16x1BlockBy4Loop

ProcessRemaining16x1Blocks:
        add     rdi,4                       ; correct for over-subtract above
        jz      Output16x1Block

Compute16x1BlockBy1Loop:
        movss   xmm0,DWORD PTR [rcx]
        ComputeBlockSseBy16 1, 0, 000h
        add     rdx,16*4                    ; advance matrix B by 16 columns
        add     rcx,4                       ; advance matrix A by 1 column
        dec     rdi
        jne     Compute16x1BlockBy1Loop

Output16x1Block:
        movss   xmm2,DWORD PTR SgemmKernelFrame.Alpha[rsp]
        shufps  xmm2,xmm2,0
        mulps   xmm8,xmm2                   ; multiply by alpha
        mulps   xmm9,xmm2
        mulps   xmm10,xmm2
        mulps   xmm11,xmm2
        sub     rbp,16
        jb      OutputPartial16x1Block
IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [r8]
        movups  xmm1,XMMWORD PTR [r8+16]
        movups  xmm2,XMMWORD PTR [r8+32]
        movups  xmm3,XMMWORD PTR [r8+48]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
        addps   xmm10,xmm2
        addps   xmm11,xmm3
ENDIF
        movups  XMMWORD PTR [r8],xmm8
        movups  XMMWORD PTR [r8+16],xmm9
        movups  XMMWORD PTR [r8+32],xmm10
        movups  XMMWORD PTR [r8+48],xmm11
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        test    rbp,rbp
        jnz     ProcessNextColumnLoop16x1
        jmp     ExitKernel

;
; Output a partial 16x1 block to the matrix.
;

OutputPartial16x1Block:
        add     rbp,16                      ; correct for over-subtract above
        cmp     rbp,4
        jb      OutputPartialLessThan4x1Block
        cmp     rbp,8
        jb      OutputPartialLessThan8x1Block
        cmp     rbp,12
        jb      OutputPartialLessThan12x1Block

IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [r8]
        movups  xmm1,XMMWORD PTR [r8+16]
        movups  xmm2,XMMWORD PTR [r8+32]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
        addps   xmm10,xmm2
ENDIF
        movups  XMMWORD PTR [r8],xmm8
        movups  XMMWORD PTR [r8+16],xmm9
        movups  XMMWORD PTR [r8+32],xmm10
        and     ebp,3                       ; check if remaining count is small
        jz      ExitKernel
        movaps  xmm8,xmm11                  ; shift remaining elements down
        add     r8,12*4                     ; advance matrix C by 12 columns
        jmp     OutputPartialLessThan4x1Block

OutputPartialLessThan12x1Block:
IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [r8]
        movups  xmm1,XMMWORD PTR [r8+16]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
ENDIF
        movups  XMMWORD PTR [r8],xmm8
        movups  XMMWORD PTR [r8+16],xmm9
        and     ebp,3                       ; check if remaining count is small
        jz      ExitKernel
        movaps  xmm8,xmm10                  ; shift remaining elements down
        add     r8,8*4                      ; advance matrix C by 8 columns
        jmp     OutputPartialLessThan4x1Block

OutputPartialLessThan8x1Block:
IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [r8]
        addps   xmm8,xmm0
ENDIF
        movups  XMMWORD PTR [r8],xmm8
        and     ebp,3                       ; check if remaining count is small
        jz      ExitKernel
        movaps  xmm8,xmm9                   ; shift remaining elements down
        add     r8,4*4                      ; advance matrix C by 4 columns

OutputPartialLessThan4x1Block:
        cmp     rbp,2
        jb      OutputPartial1x1Block
IFIDNI <Mode>, <Add>
        movsd   xmm0,QWORD PTR [r8]
        addps   xmm8,xmm0
ENDIF
        movsd   QWORD PTR [r8],xmm8
        and     ebp,1                       ; check if remaining count is odd
        jz      ExitKernel
        movhlps xmm8,xmm8                   ; shift third element down
        add     r8,2*4                      ; advance matrix C by 2 columns

OutputPartial1x1Block:
IFIDNI <Mode>, <Add>
        addss   xmm8,DWORD PTR [r8]
ENDIF
        movss   DWORD PTR [r8],xmm8
        jmp     ExitKernel

        NESTED_END MlasSgemmKernel&Mode&Sse, _TEXT

        ENDM

SgemmKernelSseFunction Zero
SgemmKernelSseFunction Add

        END
