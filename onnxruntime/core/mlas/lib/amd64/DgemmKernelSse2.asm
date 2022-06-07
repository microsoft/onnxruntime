;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   DgemmKernelSse2.asm
;
; Abstract:
;
;   This module implements the kernels for the double precision matrix/matrix
;   multiply operation (SGEMM).
;
;   This implementation uses SSE2 instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE DgemmKernelCommon.inc
INCLUDE FgemmKernelSse2Common.inc
        .list

;
; Macro Description:
;
;   This macro multiplies and accumulates for a 8xN block of the output matrix.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
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

ComputeBlockSseBy8 MACRO RowCount

        movapd  xmm4,XMMWORD PTR [rdx]
        movapd  xmm5,XMMWORD PTR [rdx+16]
IF RowCount EQ 2
        movapd  xmm6,xmm4
        movapd  xmm7,xmm5
ENDIF
        mulpd   xmm4,xmm0
        mulpd   xmm5,xmm0
        addpd   xmm8,xmm4
        addpd   xmm9,xmm5
IF RowCount EQ 2
        mulpd   xmm6,xmm1
        mulpd   xmm7,xmm1
        addpd   xmm12,xmm6
        addpd   xmm13,xmm7
ENDIF
        movapd  xmm4,XMMWORD PTR [rdx+32]
        movapd  xmm5,XMMWORD PTR [rdx+48]
IF RowCount EQ 2
        movapd  xmm6,xmm4
        movapd  xmm7,xmm5
ENDIF
        mulpd   xmm4,xmm0
        mulpd   xmm5,xmm0
        addpd   xmm10,xmm4
        addpd   xmm11,xmm5
IF RowCount EQ 2
        mulpd   xmm6,xmm1
        mulpd   xmm7,xmm1
        addpd   xmm14,xmm6
        addpd   xmm15,xmm7
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

        LOCAL   ProcessNextColumnLoop8xN
        LOCAL   Compute8xNBlockBy1Loop
        LOCAL   Output8xNBlock
        LOCAL   OutputPartial8xNBlock
        LOCAL   OutputPartialLessThan6xNBlock
        LOCAL   OutputPartialLessThan4xNBlock
        LOCAL   OutputPartial1xNBlock
        LOCAL   SkipAccumulateOutput1xN

ProcessNextColumnLoop8xN:
        EmitIfCountGE RowCount, 1, <xorpd xmm8,xmm8>
        EmitIfCountGE RowCount, 1, <xorpd xmm9,xmm9>
        EmitIfCountGE RowCount, 1, <xorpd xmm10,xmm10>
        EmitIfCountGE RowCount, 1, <xorpd xmm11,xmm11>
        EmitIfCountGE RowCount, 2, <xorpd xmm12,xmm12>
        EmitIfCountGE RowCount, 2, <xorpd xmm13,xmm13>
        EmitIfCountGE RowCount, 2, <xorpd xmm14,xmm14>
        EmitIfCountGE RowCount, 2, <xorpd xmm15,xmm15>
        mov     rdi,r9                      ; reload CountK

Compute8xNBlockBy1Loop:
        EmitIfCountGE RowCount, 1, <movsd xmm0,QWORD PTR [rcx]>
        EmitIfCountGE RowCount, 1, <movlhps xmm0,xmm0>
        EmitIfCountGE RowCount, 2, <movsd xmm1,QWORD PTR [rcx+r10]>
        EmitIfCountGE RowCount, 1, <movlhps xmm1,xmm1>
        ComputeBlockSseBy8 RowCount
        add     rdx,8*8                     ; advance matrix B by 8 columns
        add     rcx,8                       ; advance matrix A by 1 column
        dec     rdi
        jne     Compute8xNBlockBy1Loop

Output8xNBlock:
        movsd   xmm2,QWORD PTR FgemmKernelFrame.Alpha[rsp]
        movlhps xmm2,xmm2
        EmitIfCountGE RowCount, 1, <mulpd xmm8,xmm2>
                                            ; multiply by alpha
        EmitIfCountGE RowCount, 1, <mulpd xmm9,xmm2>
        EmitIfCountGE RowCount, 1, <mulpd xmm10,xmm2>
        EmitIfCountGE RowCount, 1, <mulpd xmm11,xmm2>
        EmitIfCountGE RowCount, 2, <mulpd xmm12,xmm2>
        EmitIfCountGE RowCount, 2, <mulpd xmm13,xmm2>
        EmitIfCountGE RowCount, 2, <mulpd xmm14,xmm2>
        EmitIfCountGE RowCount, 2, <mulpd xmm15,xmm2>
        sub     rbp,8
        jb      OutputPartial8xNBlock
        AccumulateAndStoreBlock RowCount, 4
        add     r8,8*8                      ; advance matrix C by 8 columns
        mov     rcx,rsi                     ; reload matrix A
        test    rbp,rbp
        jnz     ProcessNextColumnLoop8xN
        jmp     ExitKernel

;
; Output a partial 8xN block to the matrix.
;

OutputPartial8xNBlock:
        add     rbp,8                       ; correct for over-subtract above
        cmp     ebp,2
        jb      OutputPartial1xNBlock
        cmp     ebp,4
        jb      OutputPartialLessThan4xNBlock
        cmp     ebp,6
        jb      OutputPartialLessThan6xNBlock
        AccumulateAndStoreBlock RowCount, 3
        test    ebp,1                       ; check if remaining count is small
        jz      ExitKernel
        EmitIfCountGE RowCount, 1, <movapd xmm8,xmm11>
                                            ; shift remaining elements down
        EmitIfCountGE RowCount, 2, <movapd xmm12,xmm15>
        add     r8,6*8                      ; advance matrix C by 6 columns
        jmp     OutputPartial1xNBlock

OutputPartialLessThan6xNBlock:
        AccumulateAndStoreBlock RowCount, 2
        test    ebp,1                       ; check if remaining count is small
        jz      ExitKernel
        EmitIfCountGE RowCount, 1, <movapd xmm8,xmm10>
                                            ; shift remaining elements down
        EmitIfCountGE RowCount, 2, <movapd xmm12,xmm14>
        add     r8,4*8                      ; advance matrix C by 4 columns
        jmp     OutputPartial1xNBlock

OutputPartialLessThan4xNBlock:
        AccumulateAndStoreBlock RowCount, 1
        test    ebp,1                       ; check if remaining count is small
        jz      ExitKernel
        EmitIfCountGE RowCount, 1, <movapd xmm8,xmm9>
                                            ; shift remaining elements down
        EmitIfCountGE RowCount, 2, <movapd xmm12,xmm13>
        add     r8,2*8                      ; advance matrix C by 2 columns

OutputPartial1xNBlock:
        test    r15b,r15b                   ; ZeroMode?
        jnz     SkipAccumulateOutput1xN
        EmitIfCountGE RowCount, 1, <addsd xmm8,QWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <addsd xmm12,QWORD PTR [r8+rax]>

SkipAccumulateOutput1xN:
        EmitIfCountGE RowCount, 1, <movsd QWORD PTR [r8],xmm8>
        EmitIfCountGE RowCount, 2, <movsd QWORD PTR [r8+rax],xmm12>
IFB <Fallthrough>
        jmp     ExitKernel
ENDIF

        ENDM

;
; Generate the GEMM kernel.
;

FgemmKernelSse2Function Double

        END
