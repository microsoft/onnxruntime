/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8U8KernelAvx512BW.S

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

    This implementation uses AVX512BW instructions.

--*/

#include "asmmacro.h"
#include "QgemmU8U8KernelAvx512Common.h"

        .intel_syntax noprefix

        .text

/*++

Macro Description:

    This macro generates code to multiply and accumulator a single row of the
    output block.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    Vec1Reg - Supplies the high block accumulator register (when ColumnCount
        is 32).

    Vec2Reg - Supplies the low block accumulator register.

Implicit Arguments:

    zmm28 - Supplies the first vector loaded from matrix B.

    zmm29 - Supplies the second vector loaded from matrix B (when ColumnCount
        is 32).

    zmm30 - Supplies the broadcast value loaded from matrix A.

--*/

        .macro MultiplyAccumulateRow ColumnCount, Vec1Reg, Vec2Reg

.if \ColumnCount\() == 32
        vpmaddwd zmm31,zmm30,zmm28
        vpaddd  \Vec1Reg\(),\Vec1Reg\(),zmm31
        vpmaddwd zmm30,zmm30,zmm29
        vpaddd  \Vec2Reg\(),\Vec2Reg\(),zmm30
.else
        vpmaddwd zmm31,zmm30,zmm28
        vpaddd  \Vec2Reg\(),\Vec2Reg\(),zmm31
.endif

        .endm

/*++

Macro Description:

    This macro generates code to multiply and accumulate each row of the output
    block.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    RowCount - Supplies the number of rows to produce.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    zmm16-zmm27 - Supplies the block accumulators.

--*/

        .macro ComputeBlock ColumnCount, RowCount

        vpmovzxbw zmm28,YMMWORD PTR [rsi]
        EmitIfCountGE \ColumnCount\(), 32, "vpmovzxbw zmm29,YMMWORD PTR [rsi+r10*8]"
        EmitIfCountGE \RowCount\(), 1, "vpbroadcastd zmm30,DWORD PTR [rdi]"
        EmitIfCountGE \RowCount\(), 1, "MultiplyAccumulateRow \ColumnCount\(), zmm16, zmm17"
        EmitIfCountGE \RowCount\(), 2, "vpbroadcastd zmm30,DWORD PTR [rdi+r10]"
        EmitIfCountGE \RowCount\(), 2, "MultiplyAccumulateRow \ColumnCount\(), zmm18, zmm19"
        EmitIfCountGE \RowCount\(), 3, "vpbroadcastd zmm30,DWORD PTR [rdi+r10*2]"
        EmitIfCountGE \RowCount\(), 3, "MultiplyAccumulateRow \ColumnCount\(), zmm20, zmm21"
        EmitIfCountGE \RowCount\(), 4, "vpbroadcastd zmm30,DWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "MultiplyAccumulateRow \ColumnCount\(), zmm22, zmm23"
        EmitIfCountGE \RowCount\(), 5, "vpbroadcastd zmm30,DWORD PTR [rbx+r10]"
        EmitIfCountGE \RowCount\(), 5, "MultiplyAccumulateRow \ColumnCount\(), zmm24, zmm25"
        EmitIfCountGE \RowCount\(), 6, "vpbroadcastd zmm30,DWORD PTR [rbx+r10*2]"
        EmitIfCountGE \RowCount\(), 6, "MultiplyAccumulateRow \ColumnCount\(), zmm26, zmm27"

        .endm

//
// Generate the GEMM kernel.
//

GemmU8U8KernelAvx512Function Avx512BW

        .end
