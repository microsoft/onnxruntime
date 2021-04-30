;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8X8KernelAvx512Core.asm
;
; Abstract:
;
;   This module implements the kernels for the quantized integer matrix/matrix
;   multiply operation (QGEMM).
;
;   This implementation uses AVX512 core (BW/DQ/VL) and AVX512 VNNI instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE AssembleAvx512Vnni.inc
        .list

;
; Stack frame layout for the U8X8 kernel.
;

GemmU8X8KernelFrame STRUCT

        SavedXmm13 OWORD ?
        SavedXmm14 OWORD ?
        SavedXmm15 OWORD ?
        SavedR14 QWORD ?
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
;   This macro generates code to load packed data from matrix B.
;
; Arguments:
;
;   VecReg - Supplies the register to load the data into.
;
;   AddressOperand - Supplies the address operand.
;

LoadPackedMatrixBU8S8 MACRO VecReg, AddressOperand

        vmovdqu32 VecReg,ZMMWORD PTR AddressOperand

        ENDM

LoadPackedMatrixBU8U8 MACRO VecReg, AddressOperand

        vpmovzxbw VecReg,YMMWORD PTR AddressOperand

        ENDM

;
; Macro Description:
;
;   This macro generates code to multiply and accumulator a single cell of the
;   output block.
;
; Arguments:
;
;   AccumReg - Supplies the register to accumulate into.
;
;   Mult1Reg - Supplies the first multiplication operand register.
;
;   Mult2Reg - Supplies the second multiplication operand register.
;
; Implicit Arguments:
;
;   zmm4 - Supplies a scratch register for intermediate results.
;
;   zmm13 - Supplies a 512-bit with the broadcasted word value 0x0001.
;

MultiplyAccumulateCellU8S8Avx512Core MACRO AccumReg, Mult1Reg, Mult2Reg

        vpmaddubsw zmm4,Mult1Reg,Mult2Reg
        vpmaddwd zmm4,zmm4,zmm13
        vpaddd  AccumReg,AccumReg,zmm4

        ENDM

MultiplyAccumulateCellU8S8Avx512Vnni MACRO AccumReg, Mult1Reg, Mult2Reg

        VpdpbusdsZmmZmmZmm AccumReg,Mult1Reg,Mult2Reg

        ENDM

MultiplyAccumulateCellU8U8Avx512Core MACRO AccumReg, Mult1Reg, Mult2Reg

        vpmaddwd zmm4,Mult1Reg,Mult2Reg
        vpaddd  AccumReg,AccumReg,zmm4

        ENDM

;
; Macro Description:
;
;   This macro generates code to multiply and accumulate each row of the output
;   block.
;
; Arguments:
;
;   Type - Supplies the type of kernel to generate (U8S8 or U8U8).
;
;   Isa - Supplies the instruction set architecture string.
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
;   zmm13 - Supplies a 512-bit with the broadcasted word value 0x0001.
;
;   zmm14-zmm31 - Supplies the block accumulators.
;

ComputeBlock MACRO Type, Isa, ColumnCount, RowCount, VectorOffset, BroadcastOffset

IF ColumnCount GE 48
        LoadPackedMatrixB&Type& zmm0,[rdx+VectorOffset]
        LoadPackedMatrixB&Type& zmm1,[rdx+r14+VectorOffset]
        LoadPackedMatrixB&Type& zmm2,[rdx+r14*2+VectorOffset]
ELSEIF ColumnCount GE 32
        LoadPackedMatrixB&Type& zmm1,[rdx+VectorOffset]
        LoadPackedMatrixB&Type& zmm2,[rdx+r14+VectorOffset]
ELSE
        LoadPackedMatrixB&Type& zmm2,[rdx+VectorOffset]
ENDIF
        EmitIfCountGE RowCount, 1, <vpbroadcastd zmm3,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 48, <MultiplyAccumulateCell&Type&&Isa& zmm26,zmm3,zmm0>
        EmitIfCount2GE RowCount, 1, ColumnCount, 32, <MultiplyAccumulateCell&Type&&Isa& zmm20,zmm3,zmm1>
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <MultiplyAccumulateCell&Type&&Isa& zmm14,zmm3,zmm2>
        EmitIfCountGE RowCount, 2, <vpbroadcastd zmm3,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 48, <MultiplyAccumulateCell&Type&&Isa& zmm27,zmm3,zmm0>
        EmitIfCount2GE RowCount, 2, ColumnCount, 32, <MultiplyAccumulateCell&Type&&Isa& zmm21,zmm3,zmm1>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <MultiplyAccumulateCell&Type&&Isa& zmm15,zmm3,zmm2>
        EmitIfCountGE RowCount, 3, <vpbroadcastd zmm3,DWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 48, <MultiplyAccumulateCell&Type&&Isa& zmm28,zmm3,zmm0>
        EmitIfCount2GE RowCount, 3, ColumnCount, 32, <MultiplyAccumulateCell&Type&&Isa& zmm22,zmm3,zmm1>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <MultiplyAccumulateCell&Type&&Isa& zmm16,zmm3,zmm2>
        EmitIfCountGE RowCount, 4, <vpbroadcastd zmm3,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 48, <MultiplyAccumulateCell&Type&&Isa& zmm29,zmm3,zmm0>
        EmitIfCount2GE RowCount, 4, ColumnCount, 32, <MultiplyAccumulateCell&Type&&Isa& zmm23,zmm3,zmm1>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <MultiplyAccumulateCell&Type&&Isa& zmm17,zmm3,zmm2>
        EmitIfCountGE RowCount, 5, <vpbroadcastd zmm3,DWORD PTR [rbx+r9+BroadcastOffset]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 48, <MultiplyAccumulateCell&Type&&Isa& zmm30,zmm3,zmm0>
        EmitIfCount2GE RowCount, 5, ColumnCount, 32, <MultiplyAccumulateCell&Type&&Isa& zmm24,zmm3,zmm1>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <MultiplyAccumulateCell&Type&&Isa& zmm18,zmm3,zmm2>
        EmitIfCountGE RowCount, 6, <vpbroadcastd zmm3,DWORD PTR [rbx+r9*2+BroadcastOffset]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 48, <MultiplyAccumulateCell&Type&&Isa& zmm31,zmm3,zmm0>
        EmitIfCount2GE RowCount, 6, ColumnCount, 32, <MultiplyAccumulateCell&Type&&Isa& zmm25,zmm3,zmm1>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <MultiplyAccumulateCell&Type&&Isa& zmm19,zmm3,zmm2>

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
;   r14 - Supplies the stride in bytes of between packed blocks of matrix B.
;
;   zmm14-zmm31 - Supplies the block accumulators.
;

ComputeBlockLoopU8S8 MACRO Isa, ColumnCount, RowCount

        LOCAL   ComputeBlockBy4Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   ComputeBlockLoopExit

        mov     rsi,r9                      ; reload row length remaining

IF ((RowCount AND 1) EQ 0)
        sub     rsi,4*4
        jb      ProcessRemainingBlocks

ComputeBlockBy4Loop:
        ComputeBlock U8S8, Isa, ColumnCount, RowCount, 0*64, 0
        ComputeBlock U8S8, Isa, ColumnCount, RowCount, 1*64, 4
        ComputeBlock U8S8, Isa, ColumnCount, RowCount, 2*64, 8
        ComputeBlock U8S8, Isa, ColumnCount, RowCount, 3*64, 12
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
        ComputeBlock U8S8, Isa, ColumnCount, RowCount, 0, 0
        add     rcx,4                       ; advance matrix A by 1 quad
IF RowCount GT 3
        add     rbx,4                       ; advance matrix A plus 3 rows by 1 quad
ENDIF
        add     rdx,64                      ; advance matrix B
        sub     rsi,4                       ; decrement quads remaining
        jnz     ComputeBlockBy1Loop

ComputeBlockLoopExit:

        ENDM

ComputeBlockLoopU8U8 MACRO Isa, ColumnCount, RowCount

        LOCAL   ComputeBlockBy1Loop

        mov     rsi,r9                      ; reload row length remaining

ComputeBlockBy1Loop:
        ComputeBlock U8U8, Isa, ColumnCount, RowCount, 0, 0
        add     rcx,4                       ; advance matrix A by 1 pair
IF RowCount GT 3
        add     rbx,4                       ; advance matrix A plus 3 rows by 1 pair
ENDIF
        add     rdx,32                      ; advance matrix B
        sub     rsi,4
        jnz     ComputeBlockBy1Loop

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

ProduceOutputBlock MACRO ColumnCount, RowCount

        LOCAL   SkipScaleByZeroPointB
        LOCAL   AccumulatorsInitialized
        LOCAL   ProduceWithU8S8Avx512Core
        LOCAL   ProduceWithU8U8Avx512Core
        LOCAL   ExitProduceOutputBlock

;
; Initialize the accumulators with the row and column sums.
;

IF ColumnCount GE 32
IF ColumnCount GE 48
        vmovdqu32 zmm2,ZMMWORD PTR [r12]
        vmovdqu32 zmm1,ZMMWORD PTR [r12+64]
        vmovdqu32 zmm0,ZMMWORD PTR [r12+128]
ELSE
        vmovdqu32 zmm1,ZMMWORD PTR [r12]
        vmovdqu32 zmm0,ZMMWORD PTR [r12+64]
ENDIF
        add_immed r12,ColumnCount*4         ; advance ColumnSumBuffer by N columns
ELSE
        vmovdqu32 zmm0,ZMMWORD PTR [r12]
ENDIF
        test    r13,r13                     ; per column zero points?
        jz      SkipScaleByZeroPointB
IF ColumnCount GE 32
IF ColumnCount GE 48
        vmovdqu32 zmm5,ZMMWORD PTR [r13]
        vmovdqu32 zmm4,ZMMWORD PTR [r13+64]
        vmovdqu32 zmm3,ZMMWORD PTR [r13+128]
ELSE
        vmovdqu32 zmm4,ZMMWORD PTR [r13]
        vmovdqu32 zmm3,ZMMWORD PTR [r13+64]
ENDIF
        add_immed r13,ColumnCount*4         ; advance ZeroPointB by N columns
ELSE
        vmovdqu32 zmm3,ZMMWORD PTR [r13]
ENDIF
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <vpmulld zmm14,zmm3,DWORD BCST [r11]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 32, <vpmulld zmm20,zmm4,DWORD BCST [r11]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 48, <vpmulld zmm26,zmm5,DWORD BCST [r11]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <vpaddd zmm14,zmm0,zmm14>
        EmitIfCount2GE RowCount, 1, ColumnCount, 32, <vpaddd zmm20,zmm1,zmm20>
        EmitIfCount2GE RowCount, 1, ColumnCount, 48, <vpaddd zmm26,zmm2,zmm26>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <vpmulld zmm15,zmm3,DWORD BCST [r11+4]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 32, <vpmulld zmm21,zmm4,DWORD BCST [r11+4]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 48, <vpmulld zmm27,zmm5,DWORD BCST [r11+4]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <vpaddd zmm15,zmm0,zmm15>
        EmitIfCount2GE RowCount, 2, ColumnCount, 32, <vpaddd zmm21,zmm1,zmm21>
        EmitIfCount2GE RowCount, 2, ColumnCount, 48, <vpaddd zmm27,zmm2,zmm27>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <vpmulld zmm16,zmm3,DWORD BCST [r11+8]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 32, <vpmulld zmm22,zmm4,DWORD BCST [r11+8]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 48, <vpmulld zmm28,zmm5,DWORD BCST [r11+8]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <vpaddd zmm16,zmm0,zmm16>
        EmitIfCount2GE RowCount, 3, ColumnCount, 32, <vpaddd zmm22,zmm1,zmm22>
        EmitIfCount2GE RowCount, 3, ColumnCount, 48, <vpaddd zmm28,zmm2,zmm28>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <vpmulld zmm17,zmm3,DWORD BCST [r11+12]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 32, <vpmulld zmm23,zmm4,DWORD BCST [r11+12]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 48, <vpmulld zmm29,zmm5,DWORD BCST [r11+12]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <vpaddd zmm17,zmm0,zmm17>
        EmitIfCount2GE RowCount, 4, ColumnCount, 32, <vpaddd zmm23,zmm1,zmm23>
        EmitIfCount2GE RowCount, 4, ColumnCount, 48, <vpaddd zmm29,zmm2,zmm29>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <vpmulld zmm18,zmm3,DWORD BCST [r11+16]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 32, <vpmulld zmm24,zmm4,DWORD BCST [r11+16]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 48, <vpmulld zmm30,zmm5,DWORD BCST [r11+16]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <vpaddd zmm18,zmm0,zmm18>
        EmitIfCount2GE RowCount, 5, ColumnCount, 32, <vpaddd zmm24,zmm1,zmm24>
        EmitIfCount2GE RowCount, 5, ColumnCount, 48, <vpaddd zmm30,zmm2,zmm30>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <vpmulld zmm19,zmm3,DWORD BCST [r11+20]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 32, <vpmulld zmm25,zmm4,DWORD BCST [r11+20]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 48, <vpmulld zmm31,zmm5,DWORD BCST [r11+20]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <vpaddd zmm19,zmm0,zmm19>
        EmitIfCount2GE RowCount, 6, ColumnCount, 32, <vpaddd zmm25,zmm1,zmm25>
        EmitIfCount2GE RowCount, 6, ColumnCount, 48, <vpaddd zmm31,zmm2,zmm31>
        jmp     AccumulatorsInitialized

SkipScaleByZeroPointB:
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <vpaddd zmm14,zmm0,DWORD BCST [r11]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 32, <vpaddd zmm20,zmm1,DWORD BCST [r11]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 48, <vpaddd zmm26,zmm2,DWORD BCST [r11]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <vpaddd zmm15,zmm0,DWORD BCST [r11+4]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 32, <vpaddd zmm21,zmm1,DWORD BCST [r11+4]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 48, <vpaddd zmm27,zmm2,DWORD BCST [r11+4]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <vpaddd zmm16,zmm0,DWORD BCST [r11+8]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 32, <vpaddd zmm22,zmm1,DWORD BCST [r11+8]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 48, <vpaddd zmm28,zmm2,DWORD BCST [r11+8]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <vpaddd zmm17,zmm0,DWORD BCST [r11+12]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 32, <vpaddd zmm23,zmm1,DWORD BCST [r11+12]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 48, <vpaddd zmm29,zmm2,DWORD BCST [r11+12]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <vpaddd zmm18,zmm0,DWORD BCST [r11+16]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 32, <vpaddd zmm24,zmm1,DWORD BCST [r11+16]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 48, <vpaddd zmm30,zmm2,DWORD BCST [r11+16]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <vpaddd zmm19,zmm0,DWORD BCST [r11+20]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 32, <vpaddd zmm25,zmm1,DWORD BCST [r11+20]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 48, <vpaddd zmm31,zmm2,DWORD BCST [r11+20]>

AccumulatorsInitialized:

;
; Iterate over the length of a matrix A row to produce the output accumulators.
;

IF RowCount GT 3
        lea     rbx,[r9*2+r9]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF
        cmp     DWORD PTR GemmU8X8KernelFrame.PreviousP1Home[rsp],0
        je      ProduceWithU8S8Avx512Core
        jg      ProduceWithU8U8Avx512Core
        ComputeBlockLoopU8S8 Avx512Vnni, ColumnCount, RowCount
        jmp     ExitProduceOutputBlock

ProduceWithU8U8Avx512Core:
        ComputeBlockLoopU8U8 Avx512Core, ColumnCount, RowCount
        jmp     ExitProduceOutputBlock

ProduceWithU8S8Avx512Core:
        ComputeBlockLoopU8S8 Avx512Core, ColumnCount, RowCount

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
;   r14 - Supplies the stride in bytes of between packed blocks of matrix B.
;

ProcessCountM MACRO RowCount

        LOCAL   ProcessNextColumnLoop32xN
        LOCAL   Output32xNBlock
        LOCAL   SkipAccumulateOutput32xNBlock
        LOCAL   Output16xNBlock
        LOCAL   Output16xNBlockWithMask
        LOCAL   SkipAccumulateOutput16xNBlockWithMask
        LOCAL   ProcessRemainingCountN
        LOCAL   ProcessNextColumnLoop48xN
        LOCAL   SkipAccumulateOutput48xNBlock

        cmp     rbp,32
        ja      ProcessNextColumnLoop48xN
        cmp     rbp,16
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop32xN:
        ProduceOutputBlock 32, RowCount
        add     rdx,r14                     ; advance matrix B by packed block stride

Output32xNBlock:
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput32xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd zmm20,zmm20,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm21,zmm21,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm22,zmm22,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm23,zmm23,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm24,zmm24,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm25,zmm25,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput32xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8],zmm20>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax],zmm21>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2],zmm22>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx],zmm23>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax],zmm24>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2],zmm25>
        add     r8,16*4                     ; advance matrix C by 16 columns
IF RowCount GT 3
        add     rbx,16*4                    ; advance matrix C plus 3 rows by 16 columns
ENDIF
        sub     rbp,16

Output16xNBlock:
        sub     rbp,16
        jae     Output16xNBlockWithMask
        lea     ecx,[ebp+16]                ; correct for over-subtract above
        mov     esi,1
        shl     esi,cl
        dec     esi
        kmovw   k1,esi                      ; update mask for remaining columns
        xor     ebp,ebp                     ; no more columns remaining

Output16xNBlockWithMask:
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput16xNBlockWithMask
        EmitIfCountGE RowCount, 1, <vpaddd zmm14{k1},zmm14,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm15{k1},zmm15,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm16{k1},zmm16,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm17{k1},zmm17,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm18{k1},zmm18,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm19{k1},zmm19,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput16xNBlockWithMask:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8]{k1},zmm14>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax]{k1},zmm15>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2]{k1},zmm16>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx]{k1},zmm17>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax]{k1},zmm18>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2]{k1},zmm19>
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rdi                     ; reload matrix A
        cmp     rbp,32
        ja      ProcessNextColumnLoop48xN
        cmp     rbp,16
        ja      ProcessNextColumnLoop32xN
        test    rbp,rbp
        jnz     ProcessRemainingCountN
        mov     eax,RowCount
        jmp     ExitKernel

ProcessRemainingCountN:
        ProduceOutputBlock 16, RowCount
        jmp     Output16xNBlock

ProcessNextColumnLoop48xN:
        ProduceOutputBlock 48, RowCount
        lea     rdx,[rdx+r14*2]             ; advance matrix B by packed block stride
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput48xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd zmm26,zmm26,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm27,zmm27,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm28,zmm28,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm29,zmm29,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm30,zmm30,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm31,zmm31,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput48xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8],zmm26>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax],zmm27>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2],zmm28>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx],zmm29>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax],zmm30>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2],zmm31>
        add     r8,16*4                     ; advance matrix C by 16 columns
IF RowCount GT 3
        add     rbx,16*4                    ; advance matrix C plus 3 rows by 16 columns
ENDIF
        sub     rbp,16
        jmp     Output32xNBlock

        ENDM

;
; Reduce code size for the various types of kernels by sharing the outer logic
; and switching on the selector codes (using sign bit to discriminate).
;

        LEAF_ENTRY MlasGemmU8S8KernelAvx512Vnni, _TEXT

        mov     eax,-1
        jmp     MlasGemmU8X8KernelAvx512Core

        LEAF_END MlasGemmU8S8KernelAvx512Vnni, _TEXT

        LEAF_ENTRY MlasGemmU8U8KernelAvx512Core, _TEXT

        mov     eax,1
        jmp     MlasGemmU8X8KernelAvx512Core

        LEAF_END MlasGemmU8U8KernelAvx512Core, _TEXT

        LEAF_ENTRY MlasGemmU8S8KernelAvx512Core, _TEXT

        xor     eax,eax
        jmp     MlasGemmU8X8KernelAvx512Core

        LEAF_END MlasGemmU8S8KernelAvx512Core, _TEXT

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

        NESTED_ENTRY MlasGemmU8X8KernelAvx512Core, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        push_reg r14
        alloc_stack (GemmU8X8KernelFrame.SavedR14)
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
        mov     esi,-1
        kmovw   k1,esi                      ; update mask to write all columns
        neg     esi
        vpbroadcastw zmm13,esi              ; generate 512-bit word vector [0x0001]
        lea     rsi,[r9*8]                  ; compute matrix B packed stride (U8U8)
        lea     r14,[rsi*2]                 ; compute matrix B packed stride (U8S8)
        cmp     DWORD PTR GemmU8X8KernelFrame.PreviousP1Home[rsp],0
        cmovg   r14,rsi                     ; select matrix B packed stride

;
; Process CountM rows of the matrices.
;

        cmp     rbx,5
        ja      ProcessCountM6
        je      ProcessCountM5
        cmp     rbx,3
        ja      ProcessCountM4
        je      ProcessCountM3
        cmp     rbx,1
        ja      ProcessCountM2

ProcessCountM1:
        ProcessCountM 1

ProcessCountM2:
        ProcessCountM 2

ProcessCountM3:
        ProcessCountM 3

ProcessCountM4:
        ProcessCountM 4

ProcessCountM5:
        ProcessCountM 5

ProcessCountM6:
        ProcessCountM 6

;
; Restore non-volatile registers and return.
;

ExitKernel:
        vzeroupper
        movaps  xmm13,GemmU8X8KernelFrame.SavedXmm13[rsp]
        movaps  xmm14,GemmU8X8KernelFrame.SavedXmm14[rsp]
        movaps  xmm15,GemmU8X8KernelFrame.SavedXmm15[rsp]
        add     rsp,(GemmU8X8KernelFrame.SavedR14)

        BEGIN_EPILOGUE

        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasGemmU8X8KernelAvx512Core, _TEXT

        END
