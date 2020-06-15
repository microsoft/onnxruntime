/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8S8KernelAvx512Common.h

Abstract:

    This module contains common kernel macros and structures for the quantized
    integer matrix/matrix multiply operation (QGEMM) for the AVX512 core and
    AVX512VNNI kernels.

--*/

#include "QgemmU8X8KernelAvx512Common.h"

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple
    times and advancing the matrix A and matrix B data pointers.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    RowCount - Supplies the number of rows to produce.

Implicit Arguments:

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rdi - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the length in bytes of a row from matrix A.

    r14 - Supplies the stride in bytes of between packed blocks of matrix B.

    zmm14-zmm31 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLoop ColumnCount, RowCount

        mov     rbp,rcx                     # reload row length remaining

.if ((\RowCount\() & 1) == 0)
        sub     rbp,4*4
        jb      .LProcessRemainingBlocks\@

.LComputeBlockBy4Loop\@:
        ComputeBlock \ColumnCount\(), \RowCount\(), 0*64, 0
        ComputeBlock \ColumnCount\(), \RowCount\(), 1*64, 4
        ComputeBlock \ColumnCount\(), \RowCount\(), 2*64, 8
        ComputeBlock \ColumnCount\(), \RowCount\(), 3*64, 12
        add     rdi,4*4                     # advance matrix A by 1 quad
.if \RowCount\() > 3
        add     rbx,4*4                     # advance matrix A plus 3 rows by 1 quad
.endif
        add     rsi,4*64                    # advance matrix B
        sub     rbp,4*4                     # decrement quads remaining
        jae     .LComputeBlockBy4Loop\@

.LProcessRemainingBlocks\@:
        add     rbp,4*4                     # correct for over-subtract above
        jz      .LComputeBlockLoopExit\@
.endif

.LComputeBlockBy1Loop\@:
        ComputeBlock \ColumnCount\(), \RowCount\(), 0, 0
        add     rdi,4                       # advance matrix A by 1 quad
.if \RowCount\() > 3
        add     rbx,4                       # advance matrix A plus 3 rows by 1 quad
.endif
        add     rsi,64                      # advance matrix B
        sub     rbp,4                       # decrement quads remaining
        jnz     .LComputeBlockBy1Loop\@

.LComputeBlockLoopExit\@:

        .endm
