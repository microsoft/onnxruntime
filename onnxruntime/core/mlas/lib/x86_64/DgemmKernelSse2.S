/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelSse2.s

Abstract:

    This module implements the kernels for the double precision matrix/matrix
    multiply operation (DGEMM).

    This implementation uses SSE2 instructions.

--*/

#include "asmmacro.h"
#include "DgemmKernelCommon.h"
#include "FgemmKernelSse2Common.h"

        .intel_syntax noprefix

        .text

/*++

Macro Description:

    This macro multiplies and accumulates for a 8xN block of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

Implicit Arguments:

    rsi - Supplies the address into the matrix B data.

    xmm0-xmm1 - Supplies up to two elements loaded from matrix A and matrix A
        plus one row.

    xmm8-xmm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockSseBy8 RowCount

        movapd  xmm4,XMMWORD PTR [rsi]
        movapd  xmm5,XMMWORD PTR [rsi+16]
.if \RowCount\() == 2
        movapd  xmm6,xmm4
        movapd  xmm7,xmm5
.endif
        mulpd   xmm4,xmm0
        mulpd   xmm5,xmm0
        addpd   xmm8,xmm4
        addpd   xmm9,xmm5
.if \RowCount\() == 2
        mulpd   xmm6,xmm1
        mulpd   xmm7,xmm1
        addpd   xmm12,xmm6
        addpd   xmm13,xmm7
.endif
        movapd  xmm4,XMMWORD PTR [rsi+32]
        movapd  xmm5,XMMWORD PTR [rsi+48]
.if \RowCount\() == 2
        movapd  xmm6,xmm4
        movapd  xmm7,xmm5
.endif
        mulpd   xmm4,xmm0
        mulpd   xmm5,xmm0
        addpd   xmm10,xmm4
        addpd   xmm11,xmm5
.if \RowCount\() == 2
        mulpd   xmm6,xmm1
        mulpd   xmm7,xmm1
        addpd   xmm14,xmm6
        addpd   xmm15,xmm7
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute matrix multiplication for a fixed set
    of rows.

Arguments:

    RowCount - Supplies the number of rows to process.

    Fallthrough - Supplies a non-blank value if the macro may fall through to
        the ExitKernel label.

Implicit Arguments:

    rdi - Supplies the address of matrix A.

    rsi - Supplies the address of matrix B.

    r11 - Supplies the address of matrix A.

    r9 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    rdx - Supplies the address of matrix C.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    rax - Supplies the length in bytes of a row from matrix C.

    r15 - Stores the ZeroMode argument from the stack frame.

--*/

        .macro ProcessCountM RowCount, Fallthrough

.LProcessNextColumnLoop8xN\@:
        EmitIfCountGE \RowCount\(), 1, "xorpd xmm8,xmm8"
        EmitIfCountGE \RowCount\(), 1, "xorpd xmm9,xmm9"
        EmitIfCountGE \RowCount\(), 1, "xorpd xmm10,xmm10"
        EmitIfCountGE \RowCount\(), 1, "xorpd xmm11,xmm11"
        EmitIfCountGE \RowCount\(), 2, "xorpd xmm12,xmm12"
        EmitIfCountGE \RowCount\(), 2, "xorpd xmm13,xmm13"
        EmitIfCountGE \RowCount\(), 2, "xorpd xmm14,xmm14"
        EmitIfCountGE \RowCount\(), 2, "xorpd xmm15,xmm15"
        mov     rbp,rcx                     # reload CountK

.LCompute8xNBlockBy1Loop\@:
        EmitIfCountGE \RowCount\(), 1, "movsd xmm0,[rdi]"
        EmitIfCountGE \RowCount\(), 1, "movlhps xmm0,xmm0"
        EmitIfCountGE \RowCount\(), 2, "movsd xmm1,[rdi+r10]"
        EmitIfCountGE \RowCount\(), 2, "movlhps xmm1,xmm1"
        ComputeBlockSseBy8 \RowCount\()
        add     rsi,8*8                     # advance matrix B by 8 columns
        add     rdi,8                       # advance matrix A by 1 column
        dec     rbp
        jne     .LCompute8xNBlockBy1Loop\@

.LOutput8xNBlock\@:
        movsd   xmm2,.LFgemmKernelFrame_alpha[rsp]
        movlhps xmm2,xmm2
        EmitIfCountGE \RowCount\(), 1, "mulpd xmm8,xmm2"
                                            # multiply by alpha
        EmitIfCountGE \RowCount\(), 1, "mulpd xmm9,xmm2"
        EmitIfCountGE \RowCount\(), 1, "mulpd xmm10,xmm2"
        EmitIfCountGE \RowCount\(), 1, "mulpd xmm11,xmm2"
        EmitIfCountGE \RowCount\(), 2, "mulpd xmm12,xmm2"
        EmitIfCountGE \RowCount\(), 2, "mulpd xmm13,xmm2"
        EmitIfCountGE \RowCount\(), 2, "mulpd xmm14,xmm2"
        EmitIfCountGE \RowCount\(), 2, "mulpd xmm15,xmm2"
        sub     r9,8
        jb      .LOutputPartial8xNBlock\@
        AccumulateAndStoreBlock \RowCount\(), 4
        add     rdx,8*8                     # advance matrix C by 8 columns
        mov     rdi,r11                     # reload matrix A
        test    r9,r9
        jnz     .LProcessNextColumnLoop8xN\@
        jmp     .LExitKernel

//
// Output a partial 8xN block to the matrix.
//

.LOutputPartial8xNBlock\@:
        add     r9,8                        # correct for over-subtract above
        cmp     r9,2
        jb      .LOutputPartial1xNBlock\@
        cmp     r9,4
        jb      .LOutputPartialLessThan4xNBlock\@
        cmp     r9,6
        jb      .LOutputPartialLessThan6xNBlock\@
        AccumulateAndStoreBlock \RowCount\(), 3
        test    r9d,1                       # check if remaining count is small
        jz      .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "movapd xmm8,xmm11"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "movapd xmm12,xmm15"
        add     rdx,6*8                     # advance matrix C by 6 columns
        jmp     .LOutputPartial1xNBlock\@

.LOutputPartialLessThan6xNBlock\@:
        AccumulateAndStoreBlock \RowCount\(), 2
        test    r9d,1                       # check if remaining count is small
        jz      .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "movapd xmm8,xmm10"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "movapd xmm12,xmm14"
        add     rdx,4*8                     # advance matrix C by 4 columns
        jmp     .LOutputPartial1xNBlock\@

.LOutputPartialLessThan4xNBlock\@:
        AccumulateAndStoreBlock \RowCount\(), 1
        test    r9d,1                       # check if remaining count is small
        jz      .LExitKernel
        EmitIfCountGE \RowCount\(), 1, "movapd xmm8,xmm9"
                                            # shift remaining elements down
        EmitIfCountGE \RowCount\(), 2, "movapd xmm12,xmm13"
        add     rdx,2*8                     # advance matrix C by 2 columns

.LOutputPartial1xNBlock\@:
        test    r15b,r15b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput1xN\@
        EmitIfCountGE \RowCount\(), 1, "addsd xmm8,[rdx]"
        EmitIfCountGE \RowCount\(), 2, "addsd xmm12,[rdx+rax]"

.LSkipAccumulateOutput1xN\@:
        EmitIfCountGE \RowCount\(), 1, "movsd [rdx],xmm8"
        EmitIfCountGE \RowCount\(), 2, "movsd [rdx+rax],xmm12"
.ifb \Fallthrough\()
        jmp     .LExitKernel
.endif

        .endm

//
// Generate the GEMM kernel.
//

FgemmKernelSse2Function MlasGemmDoubleKernelSse

        .end
