;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8S8KernelAvx512Vnni.asm
;
; Abstract:
;
;   This module implements the kernels for the quantized integer matrix/matrix
;   multiply operation (QGEMM).
;
;   This implementation uses AVX512VNNI instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE QgemmU8S8KernelAvx512Common.inc
INCLUDE AssembleAvx512Vnni.inc
        .list

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
;   VectorOffset - Supplies the byte offset from matrix B to fetch elements.
;
;   BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.
;
; Implicit Arguments:
;
;   rbx - Supplies the address into the matrix A data plus 3 rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r9 - Supplies the length in bytes of a row from matrix A.
;
;   r14 - Supplies the stride in bytes of between packed blocks of matrix B.
;
;   zmm14-zmm31 - Supplies the block accumulators.
;

ComputeBlock MACRO ColumnCount, RowCount, VectorOffset, BroadcastOffset

IF ColumnCount GE 48
        vmovdqu32 zmm0,ZMMWORD PTR [rdx+VectorOffset]
        vmovdqu32 zmm1,ZMMWORD PTR [rdx+r14+VectorOffset]
        vmovdqu32 zmm2,ZMMWORD PTR [rdx+r14*2+VectorOffset]
ELSEIF ColumnCount GE 32
        vmovdqu32 zmm1,ZMMWORD PTR [rdx+VectorOffset]
        vmovdqu32 zmm2,ZMMWORD PTR [rdx+r14+VectorOffset]
ELSE
        vmovdqu32 zmm2,ZMMWORD PTR [rdx+VectorOffset]
ENDIF
        EmitIfCountGE RowCount, 1, <vpbroadcastd zmm3,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 48, <VpdpbusdsZmmZmmZmm zmm26,zmm3,zmm0>
        EmitIfCount2GE RowCount, 1, ColumnCount, 32, <VpdpbusdsZmmZmmZmm zmm20,zmm3,zmm1>
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <VpdpbusdsZmmZmmZmm zmm14,zmm3,zmm2>
        EmitIfCountGE RowCount, 2, <vpbroadcastd zmm3,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 48, <VpdpbusdsZmmZmmZmm zmm27,zmm3,zmm0>
        EmitIfCount2GE RowCount, 2, ColumnCount, 32, <VpdpbusdsZmmZmmZmm zmm21,zmm3,zmm1>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <VpdpbusdsZmmZmmZmm zmm15,zmm3,zmm2>
        EmitIfCountGE RowCount, 3, <vpbroadcastd zmm3,DWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 48, <VpdpbusdsZmmZmmZmm zmm28,zmm3,zmm0>
        EmitIfCount2GE RowCount, 3, ColumnCount, 32, <VpdpbusdsZmmZmmZmm zmm22,zmm3,zmm1>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <VpdpbusdsZmmZmmZmm zmm16,zmm3,zmm2>
        EmitIfCountGE RowCount, 4, <vpbroadcastd zmm3,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 48, <VpdpbusdsZmmZmmZmm zmm29,zmm3,zmm0>
        EmitIfCount2GE RowCount, 4, ColumnCount, 32, <VpdpbusdsZmmZmmZmm zmm23,zmm3,zmm1>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <VpdpbusdsZmmZmmZmm zmm17,zmm3,zmm2>
        EmitIfCountGE RowCount, 5, <vpbroadcastd zmm3,DWORD PTR [rbx+r9+BroadcastOffset]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 48, <VpdpbusdsZmmZmmZmm zmm30,zmm3,zmm0>
        EmitIfCount2GE RowCount, 5, ColumnCount, 32, <VpdpbusdsZmmZmmZmm zmm24,zmm3,zmm1>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <VpdpbusdsZmmZmmZmm zmm18,zmm3,zmm2>
        EmitIfCountGE RowCount, 6, <vpbroadcastd zmm3,DWORD PTR [rbx+r9*2+BroadcastOffset]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 48, <VpdpbusdsZmmZmmZmm zmm31,zmm3,zmm0>
        EmitIfCount2GE RowCount, 6, ColumnCount, 32, <VpdpbusdsZmmZmmZmm zmm25,zmm3,zmm1>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <VpdpbusdsZmmZmmZmm zmm19,zmm3,zmm2>

        ENDM

;
; Generate the GEMM kernel.
;

GemmU8X8KernelAvx512Function U8S8, Avx512Vnni

        END
