;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8S8KernelAvx512Common.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the quantized
;   integer matrix/matrix multiply operation (QGEMM) for the AVX512 core and
;   AVX512VNNI kernels.
;
;--

INCLUDE QgemmU8X8KernelAvx512Common.inc

;
; Macro Description:
;
;   This macro generates code to execute the block compute macro multiple
;   times and advancing the matrix A and matrix B data pointers.
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
;   r9 - Supplies the length in bytes of a row from matrix A.
;
;   r14 - Supplies the stride in bytes of between packed blocks of matrix B.
;
;   zmm14-zmm31 - Supplies the block accumulators.
;

ComputeBlockLoop MACRO ColumnCount, RowCount

        LOCAL   ComputeBlockBy4Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   ComputeBlockLoopExit

        mov     rsi,r9                      ; reload row length remaining

IF ((RowCount AND 1) EQ 0)
        sub     rsi,4*4
        jb      ProcessRemainingBlocks

ComputeBlockBy4Loop:
        ComputeBlock ColumnCount, RowCount, 0*64, 0
        ComputeBlock ColumnCount, RowCount, 1*64, 4
        ComputeBlock ColumnCount, RowCount, 2*64, 8
        ComputeBlock ColumnCount, RowCount, 3*64, 12
        add     rcx,4*4                     ; advance matrix A by 1 quad
IF RowCount GT 3
        add     rbx,4*4                     ; advance matrix A plus 3 rows by 1 quad
ENDIF
        add     rdx,4*64                    ; advance matrix B
        sub     rsi,4*4                     ; decrement quads remaining
        jae     ComputeBlockBy4Loop

ProcessRemainingBlocks:
        add     rsi,4*4                     ; correct for over-subtract above
        jz      ComputeBlockLoopExit
ENDIF

ComputeBlockBy1Loop:
        ComputeBlock ColumnCount, RowCount, 0, 0
        add     rcx,4                       ; advance matrix A by 1 quad
IF RowCount GT 3
        add     rbx,4                       ; advance matrix A plus 3 rows by 1 quad
ENDIF
        add     rdx,64                      ; advance matrix B
        sub     rsi,4                       ; decrement quads remaining
        jnz     ComputeBlockBy1Loop

ComputeBlockLoopExit:

        ENDM
