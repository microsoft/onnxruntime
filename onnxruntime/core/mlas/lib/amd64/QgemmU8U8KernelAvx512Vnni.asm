;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8U8KernelAvx512Vnni.asm
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
INCLUDE QgemmU8U8KernelAvx512Common.inc
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
IF ColumnCount EQ 32
        vpmovzxbw zmm29,YMMWORD PTR [rdx+r10*8]
        EmitIfCountGE RowCount, 1, <vpbroadcastd zmm30,DWORD PTR [rcx]>
        EmitIfCountGE RowCount, 1, <VpdpwssdZmmZmmZmm zmm16,zmm28,zmm30>
        EmitIfCountGE RowCount, 1, <VpdpwssdZmmZmmZmm zmm17,zmm29,zmm30>
        EmitIfCountGE RowCount, 2, <vpbroadcastd zmm30,DWORD PTR [rcx+r10]>
        EmitIfCountGE RowCount, 2, <VpdpwssdZmmZmmZmm zmm18,zmm28,zmm30>
        EmitIfCountGE RowCount, 2, <VpdpwssdZmmZmmZmm zmm19,zmm29,zmm30>
        EmitIfCountGE RowCount, 3, <vpbroadcastd zmm30,DWORD PTR [rcx+r10*2]>
        EmitIfCountGE RowCount, 3, <VpdpwssdZmmZmmZmm zmm20,zmm28,zmm30>
        EmitIfCountGE RowCount, 3, <VpdpwssdZmmZmmZmm zmm21,zmm29,zmm30>
        EmitIfCountGE RowCount, 4, <vpbroadcastd zmm30,DWORD PTR [rbx]>
        EmitIfCountGE RowCount, 4, <VpdpwssdZmmZmmZmm zmm22,zmm28,zmm30>
        EmitIfCountGE RowCount, 4, <VpdpwssdZmmZmmZmm zmm23,zmm29,zmm30>
        EmitIfCountGE RowCount, 5, <vpbroadcastd zmm30,DWORD PTR [rbx+r10]>
        EmitIfCountGE RowCount, 5, <VpdpwssdZmmZmmZmm zmm24,zmm28,zmm30>
        EmitIfCountGE RowCount, 5, <VpdpwssdZmmZmmZmm zmm25,zmm29,zmm30>
        EmitIfCountGE RowCount, 6, <vpbroadcastd zmm30,DWORD PTR [rbx+r10*2]>
        EmitIfCountGE RowCount, 6, <VpdpwssdZmmZmmZmm zmm26,zmm28,zmm30>
        EmitIfCountGE RowCount, 6, <VpdpwssdZmmZmmZmm zmm27,zmm29,zmm30>
ELSE
        EmitIfCountGE RowCount, 1, <VpdpwssdZmmZmmBroadcast zmm17,zmm28,rcx>
        EmitIfCountGE RowCount, 2, <VpdpwssdZmmZmmBroadcast zmm19,zmm28,rcx,r10,1>
        EmitIfCountGE RowCount, 3, <VpdpwssdZmmZmmBroadcast zmm21,zmm28,rcx,r10,2>
        EmitIfCountGE RowCount, 4, <VpdpwssdZmmZmmBroadcast zmm23,zmm28,rbx>
        EmitIfCountGE RowCount, 5, <VpdpwssdZmmZmmBroadcast zmm25,zmm28,rbx,r10,1>
        EmitIfCountGE RowCount, 6, <VpdpwssdZmmZmmBroadcast zmm27,zmm28,rbx,r10,2>
ENDIF

        ENDM

;
; Generate the GEMM kernel.
;

GemmU8U8KernelAvx512Function Avx512Vnni

        END
