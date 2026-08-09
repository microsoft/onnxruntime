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
INCLUDE mlasi.inc
INCLUDE SgemmKernelCommon.inc
INCLUDE FgemmKernelSse2Common.inc
        .list

;
; Macro Description:
;
;   This macro multiplies and accumulates for a 16xN block of the output matrix.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
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

ComputeBlockSseBy16 MACRO RowCount, VectorOffset, Shuffle

        movaps  xmm4,XMMWORD PTR [rdx+VectorOffset]
        movaps  xmm5,XMMWORD PTR [rdx+VectorOffset+16]
        pshufd  xmm2,xmm0,Shuffle
IF RowCount EQ 2
        pshufd  xmm3,xmm1,Shuffle
        movaps  xmm6,xmm4
        movaps  xmm7,xmm5
ENDIF
        mulps   xmm4,xmm2
        mulps   xmm5,xmm2
        addps   xmm8,xmm4
        addps   xmm9,xmm5
IF RowCount EQ 2
        mulps   xmm6,xmm3
        mulps   xmm7,xmm3
        addps   xmm12,xmm6
        addps   xmm13,xmm7
ENDIF
        movaps  xmm4,XMMWORD PTR [rdx+VectorOffset+32]
        movaps  xmm5,XMMWORD PTR [rdx+VectorOffset+48]
IF RowCount EQ 2
        movaps  xmm6,xmm4
        movaps  xmm7,xmm5
ENDIF
        mulps   xmm4,xmm2
        mulps   xmm5,xmm2
        addps   xmm10,xmm4
        addps   xmm11,xmm5
IF RowCount EQ 2
        mulps   xmm6,xmm3
        mulps   xmm7,xmm3
        addps   xmm14,xmm6
        addps   xmm15,xmm7
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to compute matrix multiplication for a fixed set
;   of rows.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
;
;   Fallthrough - Supplies a non-blank value if the macro may fall through to
;       the ExitKernel label.
;
; Implicit Arguments:
;
;   rax - Supplies the length in bytes of a row from matrix C.
;
;   rcx - Supplies the address of matrix A.
;
;   rdx - Supplies the address of matrix B.
;
;   rsi - Supplies the address of matrix A.
;
;   rbp - Supplies the number of columns from matrix B and matrix C to iterate
;       over.
;
;   r8 - Supplies the address of matrix C.
;
;   r9 - Supplies the number of columns from matrix A and the number of rows
;       from matrix B to iterate over.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   r15 - Stores the ZeroMode argument from the stack frame.
;

ProcessCountM MACRO RowCount, Fallthrough

        LOCAL   ProcessNextColumnLoop16xN
        LOCAL   Compute16xNBlockBy4Loop
        LOCAL   ProcessRemaining16xNBlocks
        LOCAL   Compute16xNBlockBy1Loop
        LOCAL   Output16xNBlock
        LOCAL   OutputPartial16xNBlock
        LOCAL   OutputPartialLessThan12xNBlock
        LOCAL   OutputPartialLessThan8xNBlock
        LOCAL   OutputPartialLessThan4xNBlock
        LOCAL   SkipAccumulateOutput2xN
        LOCAL   OutputPartial1xNBlock
        LOCAL   SkipAccumulateOutput1xN

ProcessNextColumnLoop16xN:
        EmitIfCountGE RowCount, 1, <xorps xmm8,xmm8>
        EmitIfCountGE RowCount, 1, <xorps xmm9,xmm9>
        EmitIfCountGE RowCount, 1, <xorps xmm10,xmm10>
        EmitIfCountGE RowCount, 1, <xorps xmm11,xmm11>
        EmitIfCountGE RowCount, 2, <xorps xmm12,xmm12>
        EmitIfCountGE RowCount, 2, <xorps xmm13,xmm13>
        EmitIfCountGE RowCount, 2, <xorps xmm14,xmm14>
        EmitIfCountGE RowCount, 2, <xorps xmm15,xmm15>
        mov     rdi,r9                      ; reload CountK
        sub     rdi,4
        jb      ProcessRemaining16xNBlocks

Compute16xNBlockBy4Loop:
        EmitIfCountGE RowCount, 1, <movups xmm0,XMMWORD PTR [rcx]>
        EmitIfCountGE RowCount, 2, <movups xmm1,XMMWORD PTR [rcx+r10]>
        ComputeBlockSseBy16 RowCount, 0, 000h
        ComputeBlockSseBy16 RowCount, 16*4, 055h
        sub     rdx,-32*4                   ; advance matrix B by 32 columns
        ComputeBlockSseBy16 RowCount, 0, 0AAh
        ComputeBlockSseBy16 RowCount, 16*4, 0FFh
        sub     rdx,-32*4                   ; advance matrix B by 32 columns
        add     rcx,4*4                     ; advance matrix A by 4 columns
        sub     rdi,4
        jae     Compute16xNBlockBy4Loop

ProcessRemaining16xNBlocks:
        add     rdi,4                       ; correct for over-subtract above
        jz      Output16xNBlock

Compute16xNBlockBy1Loop:
        EmitIfCountGE RowCount, 1, <movss xmm0,DWORD PTR [rcx]>
        EmitIfCountGE RowCount, 2, <movss xmm1,DWORD PTR [rcx+r10]>
        ComputeBlockSseBy16 RowCount, 0, 000h
        add     rdx,16*4                    ; advance matrix B by 16 columns
        add     rcx,4                       ; advance matrix A by 1 column
        dec     rdi
        jne     Compute16xNBlockBy1Loop

Output16xNBlock:
        movss   xmm2,DWORD PTR FgemmKernelFrame.Alpha[rsp]
        shufps  xmm2,xmm2,0
        EmitIfCountGE RowCount, 1, <mulps xmm8,xmm2>
                                            ; multiply by alpha
        EmitIfCountGE RowCount, 1, <mulps xmm9,xmm2>
        EmitIfCountGE RowCount, 1, <mulps xmm10,xmm2>
        EmitIfCountGE RowCount, 1, <mulps xmm11,xmm2>
        EmitIfCountGE RowCount, 2, <mulps xmm12,xmm2>
        EmitIfCountGE RowCount, 2, <mulps xmm13,xmm2>
        EmitIfCountGE RowCount, 2, <mulps xmm14,xmm2>
        EmitIfCountGE RowCount, 2, <mulps xmm15,xmm2>
        sub     rbp,16
        jb      OutputPartial16xNBlock
        AccumulateAndStoreBlock RowCount, 4
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        test    rbp,rbp
        jnz     ProcessNextColumnLoop16xN
        jmp     ExitKernel

;
; Output a partial 16xN block to the matrix.
;

OutputPartial16xNBlock:
        add     rbp,16                      ; correct for over-subtract above
        cmp     ebp,4
        jb      OutputPartialLessThan4xNBlock
        cmp     ebp,8
        jb      OutputPartialLessThan8xNBlock
        cmp     ebp,12
        jb      OutputPartialLessThan12xNBlock
        AccumulateAndStoreBlock RowCount, 3
        and     ebp,3                       ; check if remaining count is small
        jz      ExitKernel
        EmitIfCountGE RowCount, 1, <movaps xmm8,xmm11>
                                            ; shift remaining elements down
        EmitIfCountGE RowCount, 2, <movaps xmm12,xmm15>
        add     r8,12*4                     ; advance matrix C by 12 columns
        jmp     OutputPartialLessThan4xNBlock

OutputPartialLessThan12xNBlock:
        AccumulateAndStoreBlock RowCount, 2
        and     ebp,3                       ; check if remaining count is small
        jz      ExitKernel
        EmitIfCountGE RowCount, 1, <movaps xmm8,xmm10>
                                            ; shift remaining elements down
        EmitIfCountGE RowCount, 2, <movaps xmm12,xmm14>
        add     r8,8*4                      ; advance matrix C by 8 columns
        jmp     OutputPartialLessThan4xNBlock

OutputPartialLessThan8xNBlock:
        AccumulateAndStoreBlock RowCount, 1
        and     ebp,3                       ; check if remaining count is small
        jz      ExitKernel
        EmitIfCountGE RowCount, 1, <movaps xmm8,xmm9>
                                            ; shift remaining elements down
        EmitIfCountGE RowCount, 2, <movaps xmm12,xmm13>
        add     r8,4*4                      ; advance matrix C by 4 columns

OutputPartialLessThan4xNBlock:
        test    ebp,2
        jz      OutputPartial1xNBlock
        test    r15b,r15b                   ; ZeroMode?
        jnz     SkipAccumulateOutput2xN
        EmitIfCountGE RowCount, 1, <movsd xmm0,QWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <movsd xmm1,QWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 1, <addps xmm8,xmm0>
        EmitIfCountGE RowCount, 2, <addps xmm12,xmm1>

SkipAccumulateOutput2xN:
        EmitIfCountGE RowCount, 1, <movsd QWORD PTR [r8],xmm8>
        EmitIfCountGE RowCount, 2, <movsd QWORD PTR [r8+rax],xmm12>
        test    ebp,1                       ; check if remaining count is odd
        jz      ExitKernel
        EmitIfCountGE RowCount, 1, <movhlps xmm8,xmm8>
                                            ; shift third element down
        EmitIfCountGE RowCount, 2, <movhlps xmm12,xmm12>
        add     r8,2*4                      ; advance matrix C by 2 columns

OutputPartial1xNBlock:
        test    r15b,r15b                   ; ZeroMode?
        jnz     SkipAccumulateOutput1xN
        EmitIfCountGE RowCount, 1, <addss xmm8,DWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <addss xmm12,DWORD PTR [r8+rax]>

SkipAccumulateOutput1xN:
        EmitIfCountGE RowCount, 1, <movss DWORD PTR [r8],xmm8>
        EmitIfCountGE RowCount, 2, <movss DWORD PTR [r8+rax],xmm12>
IFB <Fallthrough>
        jmp     ExitKernel
ENDIF

        ENDM

;
; Generate the GEMM kernel.
;

FgemmKernelSse2Function Float

        END
