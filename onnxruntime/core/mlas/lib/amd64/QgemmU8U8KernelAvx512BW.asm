;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8U8KernelAvx512BW.asm
;
; Abstract:
;
;   This module implements the kernels for the quantized integer matrix/matrix
;   multiply operation (QGEMM).
;
;   This implementation uses AVX512BW instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE QgemmU8U8KernelAvx512Common.inc
        .list

;
; Macro Description:
;
;   This macro generates code to multiply and accumulator a single row of the
;   output block.
;
; Arguments:
;
;   ColumnCount - Supplies the number of columns to produce.
;
;   Vec1Reg - Supplies the high block accumulator register (when ColumnCount
;       is 32).
;
;   Vec2Reg - Supplies the low block accumulator register.
;
; Implicit Arguments:
;
;   zmm28 - Supplies the first vector loaded from matrix B.
;
;   zmm29 - Supplies the second vector loaded from matrix B (when ColumnCount
;       is 32).
;
;   zmm30 - Supplies the broadcast value loaded from matrix A.
;

MultiplyAccumulateRow MACRO ColumnCount, Vec1Reg, Vec2Reg

IF ColumnCount EQ 32
        vpmaddwd zmm31,zmm30,zmm28
        vpaddd  Vec1Reg,Vec1Reg,zmm31
        vpmaddwd zmm30,zmm30,zmm29
        vpaddd  Vec2Reg,Vec2Reg,zmm30
ELSE
        vpmaddwd zmm31,zmm30,zmm28
        vpaddd  Vec2Reg,Vec2Reg,zmm31
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to multiply and accumulate each row of the output
;   block.
;
; Arguments:
;
;   ColumnCount - Supplies the number of columns to produce.
;
;   RowCount - Supplies the number of rows to produce.
;
; Implicit Arguments:
;
;   rbx - Supplies the address into the matrix A data plus 3 rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   zmm16-zmm27 - Supplies the block accumulators.
;

ComputeBlock MACRO ColumnCount, RowCount

        vpmovzxbw zmm28,YMMWORD PTR [rdx]
        EmitIfCountGE ColumnCount, 32, <vpmovzxbw zmm29,YMMWORD PTR [rdx+r10*8]>
        EmitIfCountGE RowCount, 1, <vpbroadcastd zmm30,DWORD PTR [rcx]>
        EmitIfCountGE RowCount, 1, <MultiplyAccumulateRow ColumnCount, zmm16, zmm17>
        EmitIfCountGE RowCount, 2, <vpbroadcastd zmm30,DWORD PTR [rcx+r10]>
        EmitIfCountGE RowCount, 2, <MultiplyAccumulateRow ColumnCount, zmm18, zmm19>
        EmitIfCountGE RowCount, 3, <vpbroadcastd zmm30,DWORD PTR [rcx+r10*2]>
        EmitIfCountGE RowCount, 3, <MultiplyAccumulateRow ColumnCount, zmm20, zmm21>
        EmitIfCountGE RowCount, 4, <vpbroadcastd zmm30,DWORD PTR [rbx]>
        EmitIfCountGE RowCount, 4, <MultiplyAccumulateRow ColumnCount, zmm22, zmm23>
        EmitIfCountGE RowCount, 5, <vpbroadcastd zmm30,DWORD PTR [rbx+r10]>
        EmitIfCountGE RowCount, 5, <MultiplyAccumulateRow ColumnCount, zmm24, zmm25>
        EmitIfCountGE RowCount, 6, <vpbroadcastd zmm30,DWORD PTR [rbx+r10*2]>
        EmitIfCountGE RowCount, 6, <MultiplyAccumulateRow ColumnCount, zmm26, zmm27>

        ENDM

;
; Generate the GEMM kernel.
;

GemmU8U8KernelAvx512Function Avx512BW

        END
