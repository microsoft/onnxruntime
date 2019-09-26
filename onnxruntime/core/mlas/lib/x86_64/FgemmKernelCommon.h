/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

   SgemmKernelCommon.h

Abstract:

   This module contains common kernel macros and structures for the single
   precision matrix/matrix multiply operation (SGEMM).

--*/

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple
    times and advancing the matrix A and matrix B data pointers.

Arguments:

    ComputeBlock - Supplies the macro to compute a single block.

    RowCount - Supplies the number of rows to process.

    AdvanceMatrixAPlusRows - Supplies a non-zero value if the data pointer
        in rbx should also be advanced as part of the loop.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLoop ComputeBlock, RowCount, AdvanceMatrixAPlusRows

        mov     rbp,rcx                     # reload CountK
        sub     rbp,4
        jb      .LProcessRemainingBlocks\@

.LComputeBlockBy4Loop\@:
        \ComputeBlock\() \RowCount\(), 0, 0, 64*4
        \ComputeBlock\() \RowCount\(), 16*4, 4, 64*4
        sub     rsi,-32*4                   # advance matrix B by 32 columns
        \ComputeBlock\() \RowCount\(), 0, 8, 64*4
        \ComputeBlock\() \RowCount\(), 16*4, 12, 64*4
        sub     rsi,-32*4                   # advance matrix B by 32 columns
        add     rdi,4*4                     # advance matrix A by 4 columns
.if \RowCount\() > 3
        add     rbx,4*4                     # advance matrix A plus rows by 4 columns
.if \RowCount\() == 12
        add     r13,4*4
        add     r14,4*4
.endif
.endif
        sub     rbp,4
        jae     .LComputeBlockBy4Loop\@

.LProcessRemainingBlocks\@:
        add     rbp,4                       # correct for over-subtract above
        jz      .LOutputBlock\@

.LComputeBlockBy1Loop\@:
        \ComputeBlock\() \RowCount\(), 0, 0
        add     rsi,16*4                    # advance matrix B by 16 columns
        add     rdi,4                       # advance matrix A by 1 column
.if \RowCount\() > 3
        add     rbx,4                       # advance matrix A plus rows by 1 column
.if \RowCount\() == 12
        add     r13,4
        add     r14,4
.endif
.endif
        dec     rbp
        jne     .LComputeBlockBy1Loop\@

.LOutputBlock\@:

        .endm
