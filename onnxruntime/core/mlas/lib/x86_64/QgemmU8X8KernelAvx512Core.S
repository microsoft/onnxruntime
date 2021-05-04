/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8X8KernelAvx512Core.s

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

    This implementation uses AVX512 core (BW/DQ/VL) and AVX512 VNNI instructions.

--*/

#include "asmmacro.h"
#include "AssembleAvx512Vnni.h"

        .intel_syntax noprefix

//
// Stack frame layout for the U8X8 kernel.
//

        .equ    .LGemmU8X8KernelFrame_type, -8
        .equ    .LGemmU8X8KernelFrame_SavedR14, 0
        .equ    .LGemmU8X8KernelFrame_SavedR13, 8
        .equ    .LGemmU8X8KernelFrame_SavedR12, 16
        .equ    .LGemmU8X8KernelFrame_SavedRbx, 24
        .equ    .LGemmU8X8KernelFrame_SavedRbp, 32
        .equ    .LGemmU8X8KernelFrame_ReturnAddress, 40
        .equ    .LGemmU8X8KernelFrame_ldc, 48
        .equ    .LGemmU8X8KernelFrame_RowSumBuffer, 56
        .equ    .LGemmU8X8KernelFrame_ColumnSumBuffer, 64
        .equ    .LGemmU8X8KernelFrame_ZeroPointB, 72
        .equ    .LGemmU8X8KernelFrame_ZeroMode, 80

        .text

/*++

Macro Description:

    This macro generates code to load packed data from matrix B.

Arguments:

    VecReg - Supplies the register to load the data into.

    AddressOperand - Supplies the address operand.

--*/

        .macro LoadPackedMatrixBU8S8 VecReg, AddressOperand

        vmovdqu32 \VecReg\(),ZMMWORD PTR \AddressOperand\()

        .endm

        .macro LoadPackedMatrixBU8U8 VecReg, AddressOperand

        vpmovzxbw \VecReg\(),YMMWORD PTR \AddressOperand\()

        .endm

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

    zmm13 - Supplies a 512-bit with the broadcasted word value 0x0001.

--*/

        .macro MultiplyAccumulateCellU8S8Avx512Core AccumReg, Mult1Reg, Mult2Reg

        vpmaddubsw zmm4,\Mult1Reg\(),\Mult2Reg\()
        vpmaddwd zmm4,zmm4,zmm13
        vpaddd  \AccumReg\(),\AccumReg\(),zmm4

        .endm

        .macro MultiplyAccumulateCellU8S8Avx512Vnni AccumReg, Mult1Reg, Mult2Reg

        VpdpbusdsZmmZmmZmm \AccumReg\(),\Mult1Reg\(),\Mult2Reg\()

        .endm

        .macro MultiplyAccumulateCellU8U8Avx512Core AccumReg, Mult1Reg, Mult2Reg

        vpmaddwd zmm4,\Mult1Reg\(),\Mult2Reg\()
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

    rdi - Supplies the address into the matrix A data.

    r8 - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the length in bytes of a row from matrix A.

    r14 - Supplies the stride in bytes of between packed blocks of matrix B.

    zmm14-zmm31 - Supplies the block accumulators.

--*/

        .macro ComputeBlock Type, Isa, ColumnCount, RowCount, VectorOffset, BroadcastOffset

.if \ColumnCount\() >= 48
        LoadPackedMatrixB\Type\() zmm0,"[rsi+\VectorOffset\()]"
        LoadPackedMatrixB\Type\() zmm1,"[rsi+r14+\VectorOffset\()]"
        LoadPackedMatrixB\Type\() zmm2,"[rsi+r14*2+\VectorOffset\()]"
.elseif \ColumnCount\() >= 32
        LoadPackedMatrixB\Type\() zmm1,"[rsi+\VectorOffset\()]"
        LoadPackedMatrixB\Type\() zmm2,"[rsi+r14+\VectorOffset\()]"
.else
        LoadPackedMatrixB\Type\() zmm2,"[rsi+\VectorOffset\()]"
.endif
        EmitIfCountGE \RowCount\(), 1, "vpbroadcastd zmm3,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm26,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm20,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm14,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 2, "vpbroadcastd zmm3,DWORD PTR [rdi+rcx+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm27,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm21,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm15,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 3, "vpbroadcastd zmm3,DWORD PTR [rdi+rcx*2+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm28,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm22,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm16,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 4, "vpbroadcastd zmm3,DWORD PTR [r8+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm29,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm23,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm17,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 5, "vpbroadcastd zmm3,DWORD PTR [r8+rcx+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm30,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm24,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm18,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 6, "vpbroadcastd zmm3,DWORD PTR [r8+rcx*2+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm31,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm25,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm19,zmm3,zmm2"

        .endm

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple times
    and advancing the matrix A and matrix B data pointers.

Arguments:

    Isa - Supplies the instruction set architecture string.

    ColumnCount - Supplies the number of columns to produce.

    RowCount - Supplies the number of rows to produce.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    r8 - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the length in bytes of a row from matrix A.

    r14 - Supplies the stride in bytes of between packed blocks of matrix B.

    zmm14-zmm31 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLoopU8S8 Isa, ColumnCount, RowCount

        mov     rbp,rcx                     # reload row length remaining

.if ((\RowCount\() & 1) == 0)
        sub     rbp,4*4
        jb      .LProcessRemainingBlocks\@

.LComputeBlockBy4Loop\@:
        ComputeBlock U8S8, \Isa\(), \ColumnCount\(), \RowCount\(), 0*64, 0
        ComputeBlock U8S8, \Isa\(), \ColumnCount\(), \RowCount\(), 1*64, 4
        ComputeBlock U8S8, \Isa\(), \ColumnCount\(), \RowCount\(), 2*64, 8
        ComputeBlock U8S8, \Isa\(), \ColumnCount\(), \RowCount\(), 3*64, 12
        add     rdi,4*4                     # advance matrix A by 1 quad
.if \RowCount\() > 3
        add     r8,4*4                      # advance matrix A plus 3 rows by 1 quad
.endif
        add     rsi,4*64                    # advance matrix B
        sub     rbp,4*4                     # decrement quads remaining
        jae     .LComputeBlockBy4Loop\@

.LProcessRemainingBlocks\@:
        add     rbp,4*4                     # correct for over-subtract above
        jz      .LComputeBlockLoopExit\@
.endif

.LComputeBlockBy1Loop\@:
        ComputeBlock U8S8, \Isa\(), \ColumnCount\(), \RowCount\(), 0, 0
        add     rdi,4                       # advance matrix A by 1 quad
.if \RowCount\() > 3
        add     r8,4                        # advance matrix A plus 3 rows by 1 quad
.endif
        add     rsi,64                      # advance matrix B
        sub     rbp,4                       # decrement quads remaining
        jnz     .LComputeBlockBy1Loop\@

.LComputeBlockLoopExit\@:

        .endm

        .macro ComputeBlockLoopU8U8 Isa, ColumnCount, RowCount

        mov     rbp,rcx                     # reload row length remaining

.LComputeBlockBy1Loop\@:
        ComputeBlock U8U8, \Isa\(), \ColumnCount\(), \RowCount\(), 0, 0
        add     rdi,4                       # advance matrix A by 1 pair
.if \RowCount\() > 3
        add     r8,4                        # advance matrix A plus 3 rows by 1 pair
.endif
        add     rsi,32                      # advance matrix B
        sub     rbp,4
        jnz     .LComputeBlockBy1Loop\@

        .endm

/*++

Macro Description:

    This macro generates code to produce an output block for a set of columns
    and rows.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    RowCount - Supplies the number of rows to produce.

Implicit Arguments:

    rax - Supplies the length in bytes of a row from matrix C.

    rdi - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the length in bytes of a row from matrix A.

    r11 - Supplies the address of the row sum buffer.

    r12 - Supplies the address of the column sum buffer.

--*/

        .macro ProduceOutputBlock ColumnCount, RowCount

//
// Initialize the accumulators with the row and column sums.
//

.if \ColumnCount\() >= 32
.if \ColumnCount\() >= 48
        vmovdqu32 zmm2,ZMMWORD PTR [r12]
        vmovdqu32 zmm1,ZMMWORD PTR [r12+64]
        vmovdqu32 zmm0,ZMMWORD PTR [r12+128]
.else
        vmovdqu32 zmm1,ZMMWORD PTR [r12]
        vmovdqu32 zmm0,ZMMWORD PTR [r12+64]
.endif
        add_immed r12,\ColumnCount\()*4     # advance ColumnSumBuffer by N columns
.else
        vmovdqu32 zmm0,ZMMWORD PTR [r12]
.endif
        test    r13,r13                     # per column zero points?
        jz      .LSkipScaleByZeroPointB\@
.if \ColumnCount\() >= 32
.if \ColumnCount\() >= 48
        vmovdqu32 zmm5,ZMMWORD PTR [r13]
        vmovdqu32 zmm4,ZMMWORD PTR [r13+64]
        vmovdqu32 zmm3,ZMMWORD PTR [r13+128]
.else
        vmovdqu32 zmm4,ZMMWORD PTR [r13]
        vmovdqu32 zmm3,ZMMWORD PTR [r13+64]
.endif
        add_immed r13,\ColumnCount\()*4     # advance ZeroPointB by N columns
.else
        vmovdqu32 zmm3,ZMMWORD PTR [r13]
.endif
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 16, "vpmulld zmm14,zmm3,DWORD PTR [r11]{1to16}"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 32, "vpmulld zmm20,zmm4,DWORD PTR [r11]{1to16}"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 48, "vpmulld zmm26,zmm5,DWORD PTR [r11]{1to16}"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 16, "vpaddd zmm14,zmm0,zmm14"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 32, "vpaddd zmm20,zmm1,zmm20"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 48, "vpaddd zmm26,zmm2,zmm26"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 16, "vpmulld zmm15,zmm3,DWORD PTR [r11+4]{1to16}"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 32, "vpmulld zmm21,zmm4,DWORD PTR [r11+4]{1to16}"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 48, "vpmulld zmm27,zmm5,DWORD PTR [r11+4]{1to16}"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 16, "vpaddd zmm15,zmm0,zmm15"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 32, "vpaddd zmm21,zmm1,zmm21"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 48, "vpaddd zmm27,zmm2,zmm27"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 16, "vpmulld zmm16,zmm3,DWORD PTR [r11+8]{1to16}"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 32, "vpmulld zmm22,zmm4,DWORD PTR [r11+8]{1to16}"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 48, "vpmulld zmm28,zmm5,DWORD PTR [r11+8]{1to16}"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 16, "vpaddd zmm16,zmm0,zmm16"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 32, "vpaddd zmm22,zmm1,zmm22"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 48, "vpaddd zmm28,zmm2,zmm28"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 16, "vpmulld zmm17,zmm3,DWORD PTR [r11+12]{1to16}"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 32, "vpmulld zmm23,zmm4,DWORD PTR [r11+12]{1to16}"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 48, "vpmulld zmm29,zmm5,DWORD PTR [r11+12]{1to16}"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 16, "vpaddd zmm17,zmm0,zmm17"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 32, "vpaddd zmm23,zmm1,zmm23"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 48, "vpaddd zmm29,zmm2,zmm29"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 16, "vpmulld zmm18,zmm3,DWORD PTR [r11+16]{1to16}"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 32, "vpmulld zmm24,zmm4,DWORD PTR [r11+16]{1to16}"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 48, "vpmulld zmm30,zmm5,DWORD PTR [r11+16]{1to16}"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 16, "vpaddd zmm18,zmm0,zmm18"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 32, "vpaddd zmm24,zmm1,zmm24"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 48, "vpaddd zmm30,zmm2,zmm30"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 16, "vpmulld zmm19,zmm3,DWORD PTR [r11+20]{1to16}"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 32, "vpmulld zmm25,zmm4,DWORD PTR [r11+20]{1to16}"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 48, "vpmulld zmm31,zmm5,DWORD PTR [r11+20]{1to16}"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 16, "vpaddd zmm19,zmm0,zmm19"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 32, "vpaddd zmm25,zmm1,zmm25"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 48, "vpaddd zmm31,zmm2,zmm31"
        jmp     .LAccumulatorsInitialized\@

.LSkipScaleByZeroPointB\@:
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 16, "vpaddd zmm14,zmm0,DWORD PTR [r11]{1to16}"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 32, "vpaddd zmm20,zmm1,DWORD PTR [r11]{1to16}"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 48, "vpaddd zmm26,zmm2,DWORD PTR [r11]{1to16}"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 16, "vpaddd zmm15,zmm0,DWORD PTR [r11+4]{1to16}"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 32, "vpaddd zmm21,zmm1,DWORD PTR [r11+4]{1to16}"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 48, "vpaddd zmm27,zmm2,DWORD PTR [r11+4]{1to16}"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 16, "vpaddd zmm16,zmm0,DWORD PTR [r11+8]{1to16}"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 32, "vpaddd zmm22,zmm1,DWORD PTR [r11+8]{1to16}"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 48, "vpaddd zmm28,zmm2,DWORD PTR [r11+8]{1to16}"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 16, "vpaddd zmm17,zmm0,DWORD PTR [r11+12]{1to16}"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 32, "vpaddd zmm23,zmm1,DWORD PTR [r11+12]{1to16}"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 48, "vpaddd zmm29,zmm2,DWORD PTR [r11+12]{1to16}"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 16, "vpaddd zmm18,zmm0,DWORD PTR [r11+16]{1to16}"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 32, "vpaddd zmm24,zmm1,DWORD PTR [r11+16]{1to16}"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 48, "vpaddd zmm30,zmm2,DWORD PTR [r11+16]{1to16}"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 16, "vpaddd zmm19,zmm0,DWORD PTR [r11+20]{1to16}"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 32, "vpaddd zmm25,zmm1,DWORD PTR [r11+20]{1to16}"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 48, "vpaddd zmm31,zmm2,DWORD PTR [r11+20]{1to16}"

.LAccumulatorsInitialized\@:

//
// Iterate over the length of a matrix A row to produce the output accumulators.
//

.if \RowCount\() > 3
        lea     r8,[rcx*2+rcx]
        add     r8,rdi                      # compute matrix A plus 3 rows
.endif
        cmp     DWORD PTR .LGemmU8X8KernelFrame_type[rsp],0
        je      .LProduceWithU8S8Avx512Core\@
        jg      .LProduceWithU8U8Avx512Core\@
        ComputeBlockLoopU8S8 Avx512Vnni, \ColumnCount\(), \RowCount\()
        jmp     .LExitProduceOutputBlock\@

.LProduceWithU8U8Avx512Core\@:
        ComputeBlockLoopU8U8 Avx512Core, \ColumnCount\(), \RowCount\()
        jmp     .LExitProduceOutputBlock\@

.LProduceWithU8S8Avx512Core\@:
        ComputeBlockLoopU8S8 Avx512Core, \ColumnCount\(), \RowCount\()

.LExitProduceOutputBlock\@:
.if \RowCount\() > 3
        lea     r8,[rax*2+rax]
        add     r8,rdx                      # compute matrix C plus 3 rows
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute matrix multiplication for a fixed set
    of rows.

Arguments:

    RowCount - Supplies the number of rows to process.

Implicit Arguments:

    rax - Supplies the length in bytes of a row from matrix C.

    rdi - Supplies the address of matrix A.

    rsi - Supplies the address of matrix B.

    rdx - Supplies the address of matrix C.

    rbx - Supplies the address of matrix A.

    r9 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    rcx - Supplies the length in bytes of a row from matrix A.

    r10b - Supplies the zero mode flag.

    r11 - Supplies the address of the row sum buffer.

    r12 - Supplies the address of the column sum buffer.

    r14 - Supplies the stride in bytes of between packed blocks of matrix B.

--*/

        .macro ProcessCountM RowCount

        cmp     r9,32
        ja      .LProcessNextColumnLoop48xN\@
        cmp     r9,16
        jbe     .LProcessRemainingCountN\@

.LProcessNextColumnLoop32xN\@:
        ProduceOutputBlock 32, \RowCount\()
        add     rsi,r14                     # advance matrix B by packed block stride

.LOutput32xNBlock\@:
        test    r10b,r10b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput32xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vpaddd zmm20,zmm20,ZMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd zmm21,zmm21,ZMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd zmm22,zmm22,ZMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd zmm23,zmm23,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd zmm24,zmm24,ZMMWORD PTR [r8+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd zmm25,zmm25,ZMMWORD PTR [r8+rax*2]"

.LSkipAccumulateOutput32xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovdqu32 ZMMWORD PTR [rdx],zmm20"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu32 ZMMWORD PTR [rdx+rax],zmm21"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu32 ZMMWORD PTR [rdx+rax*2],zmm22"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu32 ZMMWORD PTR [r8],zmm23"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu32 ZMMWORD PTR [r8+rax],zmm24"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu32 ZMMWORD PTR [r8+rax*2],zmm25"
        add     rdx,16*4                    # advance matrix C by 16 columns
.if \RowCount\() > 3
        add     r8,16*4                     # advance matrix C plus 3 rows by 16 columns
.endif
        sub     r9,16

.LOutput16xNBlock\@:
        sub     r9,16
        jae     .LOutput16xNBlockWithMask\@
        lea     rcx,[r9+16]                 # correct for over-subtract above
        mov     ebp,1
        shl     ebp,cl
        dec     ebp
        kmovw   k1,ebp                      # update mask for remaining columns
        xor     r9,r9                       # no more columns remaining

.LOutput16xNBlockWithMask\@:
        test    r10b,r10b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput16xNBlockWithMask\@
        EmitIfCountGE \RowCount\(), 1, "vpaddd zmm14{k1},zmm14,ZMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd zmm15{k1},zmm15,ZMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd zmm16{k1},zmm16,ZMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd zmm17{k1},zmm17,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd zmm18{k1},zmm18,ZMMWORD PTR [r8+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd zmm19{k1},zmm19,ZMMWORD PTR [r8+rax*2]"

.LSkipAccumulateOutput16xNBlockWithMask\@:
        EmitIfCountGE \RowCount\(), 1, "vmovdqu32 ZMMWORD PTR [rdx]{k1},zmm14"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu32 ZMMWORD PTR [rdx+rax]{k1},zmm15"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu32 ZMMWORD PTR [rdx+rax*2]{k1},zmm16"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu32 ZMMWORD PTR [r8]{k1},zmm17"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu32 ZMMWORD PTR [r8+rax]{k1},zmm18"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu32 ZMMWORD PTR [r8+rax*2]{k1},zmm19"
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,rbx                     # reload matrix A
        cmp     r9,32
        ja      .LProcessNextColumnLoop48xN\@
        cmp     r9,16
        ja      .LProcessNextColumnLoop32xN\@
        test    r9,r9
        jnz     .LProcessRemainingCountN\@
        mov     eax,\RowCount\()
        jmp     .LExitKernel

.LProcessRemainingCountN\@:
        ProduceOutputBlock 16, \RowCount\()
        jmp     .LOutput16xNBlock\@

.LProcessNextColumnLoop48xN\@:
        ProduceOutputBlock 48, \RowCount\()
        lea     rsi,[rsi+r14*2]             # advance matrix B by packed block stride
        test    r10b,r10b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput48xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vpaddd zmm26,zmm26,ZMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd zmm27,zmm27,ZMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd zmm28,zmm28,ZMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd zmm29,zmm29,ZMMWORD PTR [r8]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd zmm30,zmm30,ZMMWORD PTR [r8+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd zmm31,zmm31,ZMMWORD PTR [r8+rax*2]"

.LSkipAccumulateOutput48xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovdqu32 ZMMWORD PTR [rdx],zmm26"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu32 ZMMWORD PTR [rdx+rax],zmm27"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu32 ZMMWORD PTR [rdx+rax*2],zmm28"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu32 ZMMWORD PTR [r8],zmm29"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu32 ZMMWORD PTR [r8+rax],zmm30"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu32 ZMMWORD PTR [r8+rax*2],zmm31"
        add     rdx,16*4                    # advance matrix C by 16 columns
.if \RowCount\() > 3
        add     r8,16*4                    # advance matrix C plus 3 rows by 16 columns
.endif
        sub     r9,16
        jmp     .LOutput32xNBlock\@

        .endm

//
// Reduce code size for the various types of kernels by sharing the outer logic
// and switching on the selector codes (using sign bit to discriminate).
//

        FUNCTION_ENTRY MlasGemmU8U8KernelAvx512Core

        mov     eax,1
        jmp     C_UNDERSCORE(MlasGemmU8X8KernelAvx512Core)

        FUNCTION_ENTRY MlasGemmU8S8KernelAvx512Core

        xor     eax,eax
        jmp     C_UNDERSCORE(MlasGemmU8X8KernelAvx512Core)

        FUNCTION_ENTRY MlasGemmU8S8KernelAvx512Vnni

        mov     eax,-1
        jmp     C_UNDERSCORE(MlasGemmU8X8KernelAvx512Core)

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (rdi) - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8X8CopyPackAAvx2.

    B (rsi) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8X8CopyPackBAvx2.

    C (rdx) - Supplies the address of matrix C.

    PackedCountK (rcx) - Supplies the number of packed columns from matrix A and
        the number of packed rows from matrix B to iterate over.

    CountM (r8) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (r9) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldc - Supplies the first dimension of matrix C.

    RowSumBuffer - Supplies the sum of each row from matrix A. These values have
        been pre-scaled by the zero point offset of matrix B if the offset is
        per-tensor (ZeroPointB is nullptr). Otherwise, these values must be
        scaled by the per-column zero point offsets of matrix B. These values are
        accumulated into every row of matrix C.

    ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
        by the zero point offset of matrix A. These values are accumulated into
        every column of matrix C.

    ZeroPointB - Optionally supplies the per-column zero point offsets of matrix
        B, else nullptr if the matrix B is using per-tensor quantization.

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        FUNCTION_ENTRY MlasGemmU8X8KernelAvx512Core

        push    rbp
        push    rbx
        push    r12
        push    r13
        push    r14

        mov     DWORD PTR .LGemmU8X8KernelFrame_type[rsp],eax
        mov     rbx,rdi
        mov     rax,.LGemmU8X8KernelFrame_ldc[rsp]
        shl     rax,2                       # convert ldc to bytes
        shl     rcx,2                       # convert to row length
        movzx   r10,BYTE PTR .LGemmU8X8KernelFrame_ZeroMode[rsp]
        mov     r11,.LGemmU8X8KernelFrame_RowSumBuffer[rsp]
        mov     r12,.LGemmU8X8KernelFrame_ColumnSumBuffer[rsp]
        mov     r13,.LGemmU8X8KernelFrame_ZeroPointB[rsp]
        mov     ebp,-1
        kmovw   k1,ebp                      # update mask to write all columns
        neg     ebp
        vpbroadcastw zmm13,ebp              # generate 512-bit word vector [0x0001]
        lea     rbp,[rcx*8]
        lea     r14,[rbp*2]
        cmp     DWORD PTR .LGemmU8X8KernelFrame_type[rsp],0
        cmovg   r14,rbp                     # select matrix B packed stride

//
// Process CountM rows of the matrices.
//

        cmp     r8,5
        ja      .LProcessCountM6
        je      .LProcessCountM5
        cmp     r8,3
        ja      .LProcessCountM4
        je      .LProcessCountM3
        cmp     r8,1
        je      .LProcessCountM1

.LProcessCountM2:
        ProcessCountM 2

.LProcessCountM4:
        ProcessCountM 4

.LProcessCountM6:
        ProcessCountM 6

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        vzeroupper

        pop     r14
        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret

.LProcessCountM1:
        ProcessCountM 1

.LProcessCountM3:
        ProcessCountM 3

.LProcessCountM5:
        ProcessCountM 5

        .end
