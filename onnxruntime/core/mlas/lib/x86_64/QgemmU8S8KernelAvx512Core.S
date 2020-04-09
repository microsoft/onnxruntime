/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8S8KernelAvx512Core.s

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

    This implementation uses AVX512 core instructions (BW/DQ/VL).

--*/

#include "asmmacro.h"
#include "QgemmU8S8KernelAvx512Common.h"

        .intel_syntax noprefix

        .text

/*++

Macro Description:

    This macro generates code to multiply and accumulator a single cell of the
    output block.

Arguments:

    AccumReg - Supplies the register to accumulate into.

    Mult1Reg - Supplies the first multiplication operand register.

    Mult2Reg - Supplies the second multiplication operand register.

Implicit Arguments:

    zmm4 - Supplies a scratch register for intermediate results.

    zmm5 - Supplies a 512-bit with the broadcasted word value 0x0001.

--*/

        .macro MultiplyAccumulateCell AccumReg, Mult1Reg, Mult2Reg

        vpmaddubsw zmm4,\Mult1Reg\(),\Mult2Reg\()
        vpmaddwd zmm4,zmm4,zmm5
        vpaddd  \AccumReg\(),\AccumReg\(),zmm4

        .endm

/*++

Macro Description:

    This macro generates code to multiply and accumulate each row of the output
    block.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    RowCount - Supplies the number of rows to produce.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rdi - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the length in bytes of a row from matrix A.

    r14 - Supplies the stride in bytes of between packed blocks of matrix B.

    zmm14-zmm31 - Supplies the block accumulators.

--*/

        .macro ComputeBlock ColumnCount, RowCount, VectorOffset, BroadcastOffset

.if \ColumnCount\() >= 48
        vmovdqu32 zmm0,ZMMWORD PTR [rsi+\VectorOffset\()]
        vmovdqu32 zmm1,ZMMWORD PTR [rsi+r14+\VectorOffset\()]
        vmovdqu32 zmm2,ZMMWORD PTR [rsi+r14*2+\VectorOffset\()]
.elseif \ColumnCount\() >= 32
        vmovdqu32 zmm1,ZMMWORD PTR [rsi+\VectorOffset\()]
        vmovdqu32 zmm2,ZMMWORD PTR [rsi+r14+\VectorOffset\()]
.else
        vmovdqu32 zmm2,ZMMWORD PTR [rsi+\VectorOffset\()]
.endif
        EmitIfCountGE \RowCount\(), 1, "vpbroadcastd zmm3,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 48, "MultiplyAccumulateCell zmm26,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 32, "MultiplyAccumulateCell zmm20,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 16, "MultiplyAccumulateCell zmm14,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 2, "vpbroadcastd zmm3,DWORD PTR [rdi+rcx+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 48, "MultiplyAccumulateCell zmm27,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 32, "MultiplyAccumulateCell zmm21,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 16, "MultiplyAccumulateCell zmm15,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 3, "vpbroadcastd zmm3,DWORD PTR [rdi+rcx*2+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 48, "MultiplyAccumulateCell zmm28,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 32, "MultiplyAccumulateCell zmm22,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 16, "MultiplyAccumulateCell zmm16,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 4, "vpbroadcastd zmm3,DWORD PTR [rbx+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 48, "MultiplyAccumulateCell zmm29,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 32, "MultiplyAccumulateCell zmm23,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 16, "MultiplyAccumulateCell zmm17,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 5, "vpbroadcastd zmm3,DWORD PTR [rbx+rcx+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 48, "MultiplyAccumulateCell zmm30,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 32, "MultiplyAccumulateCell zmm24,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 16, "MultiplyAccumulateCell zmm18,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 6, "vpbroadcastd zmm3,DWORD PTR [rbx+rcx*2+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 48, "MultiplyAccumulateCell zmm31,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 32, "MultiplyAccumulateCell zmm25,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 16, "MultiplyAccumulateCell zmm19,zmm3,zmm2"

        .endm

//
// Generate the GEMM kernel.
//

GemmU8X8KernelAvx512Function U8S8, Avx512Core

        .end
