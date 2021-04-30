;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8X8KernelAvx2.asm
;
; Abstract:
;
;   This module implements the kernels for the quantized integer matrix/matrix
;   multiply operation (QGEMM).
;
;   This implementation uses AVX2 and AVX VNNI instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE AssembleAvxVnni.inc
        .list

        EXTERN  MlasMaskMoveTableAvx:NEAR

;
; Stack frame layout for the U8X8 kernel.
;

GemmU8X8KernelFrame STRUCT

        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
        SavedXmm11 OWORD ?
        SavedXmm12 OWORD ?
        SavedXmm13 OWORD ?
        SavedXmm14 OWORD ?
        SavedXmm15 OWORD ?
        Padding QWORD ?
        SavedR13 QWORD ?
        SavedR12 QWORD ?
        SavedRdi QWORD ?
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        CountM QWORD ?
        CountN QWORD ?
        ldc QWORD ?
        RowSumBuffer QWORD ?
        ColumnSumBuffer QWORD ?
        ZeroPointB QWORD ?
        ZeroMode QWORD ?

GemmU8X8KernelFrame ENDS

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
;       is 16).
;
;   Vec2Reg - Supplies the low block accumulator register.
;
; Implicit Arguments:
;
;   ymm0 - Supplies the first vector loaded from matrix B.
;
;   ymm1 - Supplies the second vector loaded from matrix B (when ColumnCount
;       is 16).
;
;   ymm2 - Supplies the broadcast value loaded from matrix A.
;
;   ymm12 - Supplies a 256-bit with the broadcasted word value 0x0001.
;

MultiplyAccumulateRowU8S8Avx2 MACRO ColumnCount, Vec1Reg, Vec2Reg

        vpmaddubsw ymm3,ymm2,ymm0
        vpmaddwd ymm3,ymm3,ymm12
IF ColumnCount EQ 16
        vpaddd  Vec1Reg,Vec1Reg,ymm3
        vpmaddubsw ymm2,ymm2,ymm1
        vpmaddwd ymm2,ymm2,ymm12
        vpaddd  Vec2Reg,Vec2Reg,ymm2
ELSE
        vpaddd  Vec2Reg,Vec2Reg,ymm3
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
;   ymm4-ymm11 - Supplies the block accumulators.
;
;   ymm12 - Supplies a 256-bit with the broadcasted word value 0x0001.
;

ComputeBlockU8S8Avx2 MACRO ColumnCount, RowCount, VectorOffset, BroadcastOffset

IF RowCount EQ 1
        vpbroadcastd ymm2,DWORD PTR [rcx+BroadcastOffset]
        vpmaddubsw ymm3,ymm2,YMMWORD PTR [rdx+VectorOffset]
        vpmaddwd ymm3,ymm3,ymm12
IF ColumnCount EQ 16
        vpaddd  ymm4,ymm4,ymm3
        vpmaddubsw ymm2,ymm2,YMMWORD PTR [rdx+VectorOffset+32]
        vpmaddwd ymm2,ymm2,ymm12
        vpaddd  ymm5,ymm5,ymm2
ELSE
        vpaddd  ymm5,ymm5,ymm3
ENDIF
ELSE
        vmovdqu ymm0,YMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE ColumnCount, 16, <vmovdqu ymm1,YMMWORD PTR [rdx+VectorOffset+32]>
        EmitIfCountGE RowCount, 1, <vpbroadcastd ymm2,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 1, <MultiplyAccumulateRowU8S8Avx2 ColumnCount, ymm4, ymm5>
        EmitIfCountGE RowCount, 2, <vpbroadcastd ymm2,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <MultiplyAccumulateRowU8S8Avx2 ColumnCount, ymm6, ymm7>
        EmitIfCountGE RowCount, 3, <vpbroadcastd ymm2,DWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <MultiplyAccumulateRowU8S8Avx2 ColumnCount, ymm8, ymm9>
        EmitIfCountGE RowCount, 4, <vpbroadcastd ymm2,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <MultiplyAccumulateRowU8S8Avx2 ColumnCount, ymm10, ymm11>
ENDIF

        ENDM

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
;       is 16).
;
;   Vec2Reg - Supplies the low block accumulator register.
;
; Implicit Arguments:
;
;   ymm0 - Supplies the first vector loaded from matrix B.
;
;   ymm1 - Supplies the second vector loaded from matrix B (when ColumnCount
;       is 16).
;
;   ymm2 - Supplies the broadcast value loaded from matrix A.
;

MultiplyAccumulateRowU8S8AvxVnni MACRO ColumnCount, Vec1Reg, Vec2Reg

IF ColumnCount EQ 16
        VpdpbusdsYmmYmmYmm Vec1Reg,ymm2,ymm0
        VpdpbusdsYmmYmmYmm Vec2Reg,ymm2,ymm1
ELSE
        VpdpbusdsYmmYmmYmm Vec2Reg,ymm2,ymm0
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
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockU8S8AvxVnni MACRO ColumnCount, RowCount, VectorOffset, BroadcastOffset

        vmovdqu ymm0,YMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE ColumnCount, 16, <vmovdqu ymm1,YMMWORD PTR [rdx+VectorOffset+32]>
        EmitIfCountGE RowCount, 1, <vpbroadcastd ymm2,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 1, <MultiplyAccumulateRowU8S8AvxVnni ColumnCount, ymm4, ymm5>
        EmitIfCountGE RowCount, 2, <vpbroadcastd ymm2,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <MultiplyAccumulateRowU8S8AvxVnni ColumnCount, ymm6, ymm7>
        EmitIfCountGE RowCount, 3, <vpbroadcastd ymm2,DWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <MultiplyAccumulateRowU8S8AvxVnni ColumnCount, ymm8, ymm9>
        EmitIfCountGE RowCount, 4, <vpbroadcastd ymm2,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <MultiplyAccumulateRowU8S8AvxVnni ColumnCount, ymm10, ymm11>
        EmitIfCountGE RowCount, 5, <vpbroadcastd ymm2,DWORD PTR [rbx+r9+BroadcastOffset]>
        EmitIfCountGE RowCount, 5, <MultiplyAccumulateRowU8S8AvxVnni ColumnCount, ymm12, ymm13>
        EmitIfCountGE RowCount, 6, <vpbroadcastd ymm2,DWORD PTR [rbx+r9*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 6, <MultiplyAccumulateRowU8S8AvxVnni ColumnCount, ymm14, ymm15>

        ENDM

;
; Macro Description:
;
;   This macro generates code to execute the block compute macro multiple times
;   and advancing the matrix A and matrix B data pointers.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string.
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
;   ymm4-ymm11 - Supplies the block accumulators.
;

ComputeBlockLoopU8S8 MACRO Isa, ColumnCount, RowCount

        LOCAL   ComputeBlockBy1Loop

        mov     rsi,r9                      ; reload row length remaining

ComputeBlockBy1Loop:
        ComputeBlockU8S8&Isa& ColumnCount, RowCount, 0, 0
        add     rcx,4                       ; advance matrix A by 1 quad
IF RowCount GT 3
        add     rbx,4                       ; advance matrix A plus 3 rows by 1 quad
ENDIF
        add     rdx,64                      ; advance matrix B
        sub     rsi,4
        jnz     ComputeBlockBy1Loop

        ENDM

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
;       is 16).
;
;   Vec2Reg - Supplies the low block accumulator register.
;
; Implicit Arguments:
;
;   ymm0 - Supplies the first vector loaded from matrix B.
;
;   ymm1 - Supplies the second vector loaded from matrix B (when ColumnCount
;       is 16).
;
;   ymm2 - Supplies the broadcast value loaded from matrix A.
;

MultiplyAccumulateRowU8U8Avx2 MACRO ColumnCount, Vec1Reg, Vec2Reg

        vpmaddwd ymm3,ymm2,ymm0
IF ColumnCount EQ 16
        vpaddd  Vec1Reg,Vec1Reg,ymm3
        vpmaddwd ymm2,ymm2,ymm1
        vpaddd  Vec2Reg,Vec2Reg,ymm2
ELSE
        vpaddd  Vec2Reg,Vec2Reg,ymm3
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
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockU8U8Avx2 MACRO ColumnCount, RowCount, VectorOffset, BroadcastOffset

        vpmovzxbw ymm0,XMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE ColumnCount, 16, <vpmovzxbw ymm1,XMMWORD PTR [rdx+VectorOffset+16]>
        EmitIfCountGE RowCount, 1, <vpbroadcastd ymm2,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 1, <MultiplyAccumulateRowU8U8Avx2 ColumnCount, ymm4, ymm5>
        EmitIfCountGE RowCount, 2, <vpbroadcastd ymm2,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <MultiplyAccumulateRowU8U8Avx2 ColumnCount, ymm6, ymm7>
        EmitIfCountGE RowCount, 3, <vpbroadcastd ymm2,DWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <MultiplyAccumulateRowU8U8Avx2 ColumnCount, ymm8, ymm9>
        EmitIfCountGE RowCount, 4, <vpbroadcastd ymm2,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <MultiplyAccumulateRowU8U8Avx2 ColumnCount, ymm10, ymm11>
        EmitIfCountGE RowCount, 5, <vpbroadcastd ymm2,DWORD PTR [rbx+r9+BroadcastOffset]>
        EmitIfCountGE RowCount, 5, <MultiplyAccumulateRowU8U8Avx2 ColumnCount, ymm12, ymm13>
        EmitIfCountGE RowCount, 6, <vpbroadcastd ymm2,DWORD PTR [rbx+r9*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 6, <MultiplyAccumulateRowU8U8Avx2 ColumnCount, ymm14, ymm15>

        ENDM

;
; Macro Description:
;
;   This macro generates code to execute the block compute macro multiple times
;   and advancing the matrix A and matrix B data pointers.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string.
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
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockLoopU8U8 MACRO Isa, ColumnCount, RowCount

        LOCAL   ComputeBlockBy2Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   ExitComputeBlockLoop

        mov     rsi,r9                      ; reload row length remaining

IF (ColumnCount EQ 16) AND ((RowCount AND 1) EQ 0)
        sub     rsi,2*4
        jb      ProcessRemainingBlocks

ComputeBlockBy2Loop:
        ComputeBlockU8U8&Isa& ColumnCount, RowCount, 0, 0
        ComputeBlockU8U8&Isa& ColumnCount, RowCount, 32, 4
        add     rcx,2*4                     ; advance matrix A by 2 pairs
IF RowCount GT 3
        add     rbx,2*4                     ; advance matrix A plus 3 rows by 2 pairs
ENDIF
        add     rdx,2*32                    ; advance matrix B
        sub     rsi,2*4
        jae     ComputeBlockBy2Loop

ProcessRemainingBlocks:
        add     rsi,2*4                     ; correct for over-subtract above
        jz      ExitComputeBlockLoop
        ComputeBlockU8U8&Isa& ColumnCount, RowCount, 0, 0
        add     rdx,32                      ; advance matrix B
ELSE
ComputeBlockBy1Loop:
        ComputeBlockU8U8&Isa& ColumnCount, RowCount, 0, 0
        add     rcx,4                       ; advance matrix A by 1 pair
IF RowCount GT 3
        add     rbx,4                       ; advance matrix A plus 3 rows by 1 pair
ENDIF
        add     rdx,32                      ; advance matrix B
        sub     rsi,4
        jnz     ComputeBlockBy1Loop
ENDIF

ExitComputeBlockLoop:

        ENDM

;
; Macro Description:
;
;   This macro generates code to produce an output block for a set of columns
;   and rows.
;
; Arguments:
;
;   ColumnCount - Supplies the number of columns to produce.
;
;   RowCount - Supplies the number of rows to produce.
;
; Implicit Arguments:
;
;   rax - Supplies the length in bytes of a row from matrix C.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r9 - Supplies the length in bytes of a row from matrix A.
;
;   r11 - Supplies the address of the row sum buffer.
;
;   r12 - Supplies the address of the column sum buffer.
;
;   r13 - Optionally supplies the address of the matrix B zero point buffer.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ProduceOutputBlock MACRO ColumnCount, RowCount

        LOCAL   SkipScaleByZeroPointB
        LOCAL   AccumulatorsInitialized
        LOCAL   ProduceWithU8S8AvxVnni
        LOCAL   ProduceWithU8U8Avx2
        LOCAL   ExitProduceOutputBlock

;
; Initialize the accumulators with the row and column sums.
;

        EmitIfCountGE RowCount, 1, <vpbroadcastd ymm5,DWORD PTR [r11]>
        EmitIfCountGE RowCount, 2, <vpbroadcastd ymm7,DWORD PTR [r11+4]>
        EmitIfCountGE RowCount, 3, <vpbroadcastd ymm9,DWORD PTR [r11+8]>
        EmitIfCountGE RowCount, 4, <vpbroadcastd ymm11,DWORD PTR [r11+12]>
        EmitIfCountGE RowCount, 5, <vpbroadcastd ymm13,DWORD PTR [r11+16]>
        EmitIfCountGE RowCount, 6, <vpbroadcastd ymm15,DWORD PTR [r11+20]>
IF ColumnCount EQ 16
        vmovdqu ymm0,YMMWORD PTR [r12]
        vmovdqu ymm1,YMMWORD PTR [r12+32]
        add     r12,16*4                    ; advance ColumnSumBuffer by 16 columns
ELSE
        vmovdqu ymm1,YMMWORD PTR [r12]
ENDIF
        test    r13,r13                     ; per column zero points?
        jz      SkipScaleByZeroPointB
IF ColumnCount EQ 16
        vmovdqu ymm2,YMMWORD PTR [r13]
        vmovdqu ymm3,YMMWORD PTR [r13+32]
        add     r13,16*4                    ; advance ZeroPointB by 16 columns
ELSE
        vmovdqu ymm3,YMMWORD PTR [r13]
ENDIF
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <vpmulld ymm4,ymm5,ymm2>
        EmitIfCountGE RowCount, 1, <vpmulld ymm5,ymm5,ymm3>
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <vpaddd ymm4,ymm0,ymm4>
        EmitIfCountGE RowCount, 1, <vpaddd ymm5,ymm1,ymm5>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <vpmulld ymm6,ymm7,ymm2>
        EmitIfCountGE RowCount, 2, <vpmulld ymm7,ymm7,ymm3>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <vpaddd ymm6,ymm0,ymm6>
        EmitIfCountGE RowCount, 2, <vpaddd ymm7,ymm1,ymm7>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <vpmulld ymm8,ymm9,ymm2>
        EmitIfCountGE RowCount, 3, <vpmulld ymm9,ymm9,ymm3>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <vpaddd ymm8,ymm0,ymm8>
        EmitIfCountGE RowCount, 3, <vpaddd ymm9,ymm1,ymm9>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <vpmulld ymm10,ymm11,ymm2>
        EmitIfCountGE RowCount, 4, <vpmulld ymm11,ymm11,ymm3>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <vpaddd ymm10,ymm0,ymm10>
        EmitIfCountGE RowCount, 4, <vpaddd ymm11,ymm1,ymm11>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <vpmulld ymm12,ymm13,ymm2>
        EmitIfCountGE RowCount, 5, <vpmulld ymm13,ymm13,ymm3>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <vpaddd ymm12,ymm0,ymm12>
        EmitIfCountGE RowCount, 5, <vpaddd ymm13,ymm1,ymm13>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <vpmulld ymm14,ymm15,ymm2>
        EmitIfCountGE RowCount, 6, <vpmulld ymm15,ymm15,ymm3>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <vpaddd ymm14,ymm0,ymm14>
        EmitIfCountGE RowCount, 6, <vpaddd ymm15,ymm1,ymm15>
        jmp     AccumulatorsInitialized

SkipScaleByZeroPointB:
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <vpaddd ymm4,ymm0,ymm5>
        EmitIfCountGE RowCount, 1, <vpaddd ymm5,ymm1,ymm5>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <vpaddd ymm6,ymm0,ymm7>
        EmitIfCountGE RowCount, 2, <vpaddd ymm7,ymm1,ymm7>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <vpaddd ymm8,ymm0,ymm9>
        EmitIfCountGE RowCount, 3, <vpaddd ymm9,ymm1,ymm9>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <vpaddd ymm10,ymm0,ymm11>
        EmitIfCountGE RowCount, 4, <vpaddd ymm11,ymm1,ymm11>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <vpaddd ymm12,ymm0,ymm13>
        EmitIfCountGE RowCount, 5, <vpaddd ymm13,ymm1,ymm13>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <vpaddd ymm14,ymm0,ymm15>
        EmitIfCountGE RowCount, 6, <vpaddd ymm15,ymm1,ymm15>

AccumulatorsInitialized:

;
; Iterate over the length of a matrix A row to produce the output accumulators.
;

IF RowCount GT 3
        lea     rbx,[r9*2+r9]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF
        cmp     DWORD PTR GemmU8X8KernelFrame.PreviousP1Home[rsp],0
        jg      ProduceWithU8U8Avx2
IF RowCount LE 4
        jl      ProduceWithU8S8AvxVnni
        ComputeBlockLoopU8S8 Avx2, ColumnCount, RowCount
        jmp     ExitProduceOutputBlock
ENDIF

ProduceWithU8S8AvxVnni:
        ComputeBlockLoopU8S8 AvxVnni, ColumnCount, RowCount
        jmp     ExitProduceOutputBlock

ProduceWithU8U8Avx2:
        ComputeBlockLoopU8U8 Avx2, ColumnCount, RowCount

ExitProduceOutputBlock:
IF RowCount GT 3
        lea     rbx,[rax*2+rax]
        add     rbx,r8                      ; compute matrix C plus 3 rows
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to compute matrix multiplication for a fixed set
;   of rows.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
;
; Implicit Arguments:
;
;   rax - Supplies the length in bytes of a row from matrix C.
;
;   rcx - Supplies the address of matrix A.
;
;   rdx - Supplies the address of matrix B.
;
;   r8 - Supplies the address of matrix C.
;
;   rdi - Supplies the address of matrix A.
;
;   rbp - Supplies the number of columns from matrix B and matrix C to iterate
;       over.
;
;   r9 - Supplies the length in bytes of a row from matrix A.
;
;   r10b - Supplies the zero mode flag.
;
;   r11 - Supplies the address of the row sum buffer.
;
;   r12 - Supplies the address of the column sum buffer.
;
;   r13 - Optionally supplies the address of the matrix B zero point buffer.
;

ProcessCountM MACRO RowCount, Fallthrough

        LOCAL   ProcessNextColumnLoop16xN
        LOCAL   SkipAccumulateOutput16xNBlock
        LOCAL   OutputMasked16xNBlock
        LOCAL   ExitProcessCountM
        LOCAL   ProcessRemainingCountN
        LOCAL   SkipAccumulateOutput8xNBlock
        LOCAL   SkipAccumulateOutputMasked16xNBlock
        LOCAL   OutputMasked8xNBlock
        LOCAL   SkipAccumulateOutputMasked8xNBlock

        cmp     rbp,8
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop16xN:
        ProduceOutputBlock 16, RowCount
        sub     rbp,16
        jb      OutputMasked16xNBlock
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput16xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd ymm4,ymm4,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 1, <vpaddd ymm5,ymm5,YMMWORD PTR [r8+32]>
        EmitIfCountGE RowCount, 2, <vpaddd ymm6,ymm6,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 2, <vpaddd ymm7,ymm7,YMMWORD PTR [r8+rax+32]>
        EmitIfCountGE RowCount, 3, <vpaddd ymm8,ymm8,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 3, <vpaddd ymm9,ymm9,YMMWORD PTR [r8+rax*2+32]>
        EmitIfCountGE RowCount, 4, <vpaddd ymm10,ymm10,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 4, <vpaddd ymm11,ymm11,YMMWORD PTR [rbx+32]>
        EmitIfCountGE RowCount, 5, <vpaddd ymm12,ymm12,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 5, <vpaddd ymm13,ymm13,YMMWORD PTR [rbx+rax+32]>
        EmitIfCountGE RowCount, 6, <vpaddd ymm14,ymm14,YMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE RowCount, 6, <vpaddd ymm15,ymm15,YMMWORD PTR [rbx+rax*2+32]>

SkipAccumulateOutput16xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu YMMWORD PTR [r8],ymm4>
        EmitIfCountGE RowCount, 1, <vmovdqu YMMWORD PTR [r8+32],ymm5>
        EmitIfCountGE RowCount, 2, <vmovdqu YMMWORD PTR [r8+rax],ymm6>
        EmitIfCountGE RowCount, 2, <vmovdqu YMMWORD PTR [r8+rax+32],ymm7>
        EmitIfCountGE RowCount, 3, <vmovdqu YMMWORD PTR [r8+rax*2],ymm8>
        EmitIfCountGE RowCount, 3, <vmovdqu YMMWORD PTR [r8+rax*2+32],ymm9>
        EmitIfCountGE RowCount, 4, <vmovdqu YMMWORD PTR [rbx],ymm10>
        EmitIfCountGE RowCount, 4, <vmovdqu YMMWORD PTR [rbx+32],ymm11>
        EmitIfCountGE RowCount, 5, <vmovdqu YMMWORD PTR [rbx+rax],ymm12>
        EmitIfCountGE RowCount, 5, <vmovdqu YMMWORD PTR [rbx+rax+32],ymm13>
        EmitIfCountGE RowCount, 6, <vmovdqu YMMWORD PTR [rbx+rax*2],ymm14>
        EmitIfCountGE RowCount, 6, <vmovdqu YMMWORD PTR [rbx+rax*2+32],ymm15>
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rdi                     ; reload matrix A
        cmp     rbp,8
        ja      ProcessNextColumnLoop16xN
        test    rbp,rbp
        jnz     ProcessRemainingCountN

ExitProcessCountM:
        mov     eax,RowCount
        jmp     ExitKernel

ProcessRemainingCountN:
        ProduceOutputBlock 8, RowCount
        cmp     rbp,8
        jb      OutputMasked8xNBlock
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput8xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd ymm5,ymm5,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd ymm7,ymm7,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd ymm9,ymm9,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd ymm11,ymm11,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd ymm13,ymm13,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd ymm15,ymm15,YMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput8xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu YMMWORD PTR [r8],ymm5>
        EmitIfCountGE RowCount, 2, <vmovdqu YMMWORD PTR [r8+rax],ymm7>
        EmitIfCountGE RowCount, 3, <vmovdqu YMMWORD PTR [r8+rax*2],ymm9>
        EmitIfCountGE RowCount, 4, <vmovdqu YMMWORD PTR [rbx],ymm11>
        EmitIfCountGE RowCount, 5, <vmovdqu YMMWORD PTR [rbx+rax],ymm13>
        EmitIfCountGE RowCount, 6, <vmovdqu YMMWORD PTR [rbx+rax*2],ymm15>
        jmp     ExitProcessCountM

OutputMasked16xNBlock:
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutputMasked16xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd ymm4,ymm4,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd ymm6,ymm6,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd ymm8,ymm8,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd ymm10,ymm10,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd ymm12,ymm12,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd ymm14,ymm14,YMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutputMasked16xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu YMMWORD PTR [r8],ymm4>
        EmitIfCountGE RowCount, 2, <vmovdqu YMMWORD PTR [r8+rax],ymm6>
        EmitIfCountGE RowCount, 3, <vmovdqu YMMWORD PTR [r8+rax*2],ymm8>
        EmitIfCountGE RowCount, 4, <vmovdqu YMMWORD PTR [rbx],ymm10>
        EmitIfCountGE RowCount, 5, <vmovdqu YMMWORD PTR [rbx+rax],ymm12>
        EmitIfCountGE RowCount, 6, <vmovdqu YMMWORD PTR [rbx+rax*2],ymm14>
        add     r8,8*4                      ; advance matrix C by 8 columns
IF RowCount GT 3
        add     rbx,8*4                     ; advance matrix C plus 3 rows by 8 columns
ENDIF
        add     rbp,8                       ; correct for over-subtract above

OutputMasked8xNBlock:
        neg     rbp
        lea     rcx,MlasMaskMoveTableAvx+8*4
        vmovdqu ymm0,YMMWORD PTR [rcx+rbp*4]
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutputMasked8xNBlock
        EmitIfCountGE RowCount, 1, <vpmaskmovd ymm4,ymm0,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpmaskmovd ymm6,ymm0,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpmaskmovd ymm8,ymm0,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpmaskmovd ymm10,ymm0,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpmaskmovd ymm12,ymm0,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpmaskmovd ymm14,ymm0,YMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE RowCount, 1, <vpaddd ymm5,ymm5,ymm4>
        EmitIfCountGE RowCount, 2, <vpaddd ymm7,ymm7,ymm6>
        EmitIfCountGE RowCount, 3, <vpaddd ymm9,ymm9,ymm8>
        EmitIfCountGE RowCount, 4, <vpaddd ymm11,ymm11,ymm10>
        EmitIfCountGE RowCount, 5, <vpaddd ymm13,ymm13,ymm12>
        EmitIfCountGE RowCount, 6, <vpaddd ymm15,ymm15,ymm14>

SkipAccumulateOutputMasked8xNBlock:
        EmitIfCountGE RowCount, 1, <vpmaskmovd YMMWORD PTR [r8],ymm0,ymm5>
        EmitIfCountGE RowCount, 2, <vpmaskmovd YMMWORD PTR [r8+rax],ymm0,ymm7>
        EmitIfCountGE RowCount, 3, <vpmaskmovd YMMWORD PTR [r8+rax*2],ymm0,ymm9>
        EmitIfCountGE RowCount, 4, <vpmaskmovd YMMWORD PTR [rbx],ymm0,ymm11>
        EmitIfCountGE RowCount, 5, <vpmaskmovd YMMWORD PTR [rbx+rax],ymm0,ymm13>
        EmitIfCountGE RowCount, 6, <vpmaskmovd YMMWORD PTR [rbx+rax*2],ymm0,ymm15>
        jmp     ExitProcessCountM

        ENDM

;
; Reduce code size for the various types of kernels by sharing the outer logic
; and switching on the selector codes (using sign bit to discriminate).
;

        LEAF_ENTRY MlasGemmU8S8KernelAvxVnni, _TEXT

        mov     eax,-1
        jmp     MlasGemmU8X8KernelAvx2

        LEAF_END MlasGemmU8S8KernelAvxVnni, _TEXT

        LEAF_ENTRY MlasGemmU8U8KernelAvx2, _TEXT

        mov     eax,1
        jmp     MlasGemmU8X8KernelAvx2

        LEAF_END MlasGemmU8U8KernelAvx2, _TEXT

        LEAF_ENTRY MlasGemmU8S8KernelAvx2, _TEXT

        xor     eax,eax
        jmp     MlasGemmU8X8KernelAvx2

        LEAF_END MlasGemmU8S8KernelAvx2, _TEXT

;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows.
;
; Arguments:
;
;   A (rcx) - Supplies the address of matrix A. The matrix data has been packed
;       using MlasGemmU8X8CopyPackAAvx2.
;
;   B (rdx) - Supplies the address of matrix B. The matrix data has been packed
;       using MlasGemmU8X8CopyPackBAvx2.
;
;   C (r8) - Supplies the address of matrix C.
;
;   PackedCountK (r9) - Supplies the number of packed columns from matrix A and
;       the number of packed rows from matrix B to iterate over.
;
;   CountM - Supplies the maximum number of rows that can be processed for
;       matrix A and matrix C. The actual number of rows handled for this
;       invocation depends on the kernel implementation.
;
;   CountN - Supplies the number of columns from matrix B and matrix C to iterate
;       over.
;
;   ldc - Supplies the first dimension of matrix C.
;
;   RowSumBuffer - Supplies the sum of each row from matrix A. These values have
;       been pre-scaled by the zero point offset of matrix B if the offset is
;       per-tensor (ZeroPointB is nullptr). Otherwise, these values must be
;       scaled by the per-column zero point offsets of matrix B. These values are
;       accumulated into every row of matrix C.
;
;   ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
;       by the zero point offset of matrix A. These values are accumulated into
;       every column of matrix C.
;
;   ZeroPointB - Optionally supplies the per-column zero point offsets of matrix
;       B, else nullptr if the matrix B is using per-tensor quantization.
;
;   ZeroMode - Supplies true if the output matrix must be zero initialized,
;       else false if the output matrix is accumulated into.
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

        NESTED_ENTRY MlasGemmU8X8KernelAvx2, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        alloc_stack (GemmU8X8KernelFrame.SavedR13)
        save_xmm128 xmm6,GemmU8X8KernelFrame.SavedXmm6
        save_xmm128 xmm7,GemmU8X8KernelFrame.SavedXmm7
        save_xmm128 xmm8,GemmU8X8KernelFrame.SavedXmm8
        save_xmm128 xmm9,GemmU8X8KernelFrame.SavedXmm9
        save_xmm128 xmm10,GemmU8X8KernelFrame.SavedXmm10
        save_xmm128 xmm11,GemmU8X8KernelFrame.SavedXmm11
        save_xmm128 xmm12,GemmU8X8KernelFrame.SavedXmm12
        save_xmm128 xmm13,GemmU8X8KernelFrame.SavedXmm13
        save_xmm128 xmm14,GemmU8X8KernelFrame.SavedXmm14
        save_xmm128 xmm15,GemmU8X8KernelFrame.SavedXmm15

        END_PROLOGUE

        mov     DWORD PTR GemmU8X8KernelFrame.PreviousP1Home[rsp],eax
        mov     rdi,rcx
        mov     rbx,GemmU8X8KernelFrame.CountM[rsp]
        mov     rbp,GemmU8X8KernelFrame.CountN[rsp]
        mov     rax,GemmU8X8KernelFrame.ldc[rsp]
        shl     rax,2                       ; convert ldc to bytes
        shl     r9,2                        ; convert to row length
        movzx   r10,BYTE PTR GemmU8X8KernelFrame.ZeroMode[rsp]
        mov     r11,GemmU8X8KernelFrame.RowSumBuffer[rsp]
        mov     r12,GemmU8X8KernelFrame.ColumnSumBuffer[rsp]
        mov     r13,GemmU8X8KernelFrame.ZeroPointB[rsp]
        vpcmpeqw ymm12,ymm12,ymm12          ; generate 256-bit word vector [0xFFFF]
        vpsrlw  ymm12,ymm12,15              ; generate 256-bit word vector [0x0001]
        cmp     DWORD PTR GemmU8X8KernelFrame.PreviousP1Home[rsp],0
        je      CheckCountM4OrMore          ; U8S8 AVX2 kernel requires extra registers

;
; Process CountM rows of the matrices.
;

CheckCountM6OrMore:
        cmp     rbx,5
        ja      ProcessCountM6
        je      ProcessCountM5

CheckCountM4OrMore:
        cmp     rbx,3
        ja      ProcessCountM4
        je      ProcessCountM3
        cmp     rbx,1
        je      ProcessCountM1

ProcessCountM2:
        ProcessCountM 2

ProcessCountM4:
        ProcessCountM 4

ProcessCountM6:
        ProcessCountM 6

;
; Restore non-volatile registers and return.
;

ExitKernel:
        vzeroupper
        movaps  xmm6,GemmU8X8KernelFrame.SavedXmm6[rsp]
        movaps  xmm7,GemmU8X8KernelFrame.SavedXmm7[rsp]
        movaps  xmm8,GemmU8X8KernelFrame.SavedXmm8[rsp]
        movaps  xmm9,GemmU8X8KernelFrame.SavedXmm9[rsp]
        movaps  xmm10,GemmU8X8KernelFrame.SavedXmm10[rsp]
        movaps  xmm11,GemmU8X8KernelFrame.SavedXmm11[rsp]
        movaps  xmm12,GemmU8X8KernelFrame.SavedXmm12[rsp]
        movaps  xmm13,GemmU8X8KernelFrame.SavedXmm13[rsp]
        movaps  xmm14,GemmU8X8KernelFrame.SavedXmm14[rsp]
        movaps  xmm15,GemmU8X8KernelFrame.SavedXmm15[rsp]
        add     rsp,(GemmU8X8KernelFrame.SavedR13)

        BEGIN_EPILOGUE

        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

ProcessCountM1:
        ProcessCountM 1

ProcessCountM3:
        ProcessCountM 3

ProcessCountM5:
        ProcessCountM 5

        NESTED_END MlasGemmU8X8KernelAvx2, _TEXT

        END
