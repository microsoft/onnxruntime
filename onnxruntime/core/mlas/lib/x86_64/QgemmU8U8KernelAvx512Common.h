/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8U8KernelAvx512Common.h

Abstract:

    This module contains common kernel macros and structures for the quantized
    integer matrix/matrix multiply operation (QGEMM) for the AVX512BW and
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

.LComputeBlockBy1Loop\@:
        ComputeBlock \ColumnCount\(), \RowCount\(), 0, 0
        add     rdi,4                       # advance matrix A by 1 pair
.if \RowCount\() > 3
        add     rbx,4                       # advance matrix A plus 3 rows by 1 pair
.endif
        add     rsi,32                      # advance matrix B
        sub     rbp,4
        jnz     .LComputeBlockBy1Loop\@

        .endm
