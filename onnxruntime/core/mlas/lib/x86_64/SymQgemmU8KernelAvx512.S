/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SymQgemmU8S8KernelAvx512.s

Abstract:

    This module implements the kernels for the symmetrically quantized integer matrix/matrix
    multiply operation (QGEMM), where the right hand side matrix is quantized with
    zero point being 0

    This implementation uses AVX512 core (BW/DQ/VL) and AVX512 VNNI instructions.

--*/

#include "asmmacro.h"
#include "AssembleAvx512Vnni.h"

        .intel_syntax noprefix

//
// Stack frame layout for the U8X8 kernel.
//

        .equ    .LSQgemmU8KernelFrame_type, -8
        .equ    .LSQgemmU8KernelFrame_SavedR12, 0
        .equ    .LSQgemmU8KernelFrame_SavedRbx, 8
        .equ    .LSQgemmU8KernelFrame_SavedRbp, 16
        .equ    .LSQgemmU8KernelFrame_ReturnAddress, 24
        .equ    .LSQgemmU8KernelFrame_ldc, 32
        .equ    .LSQgemmU8KernelFrame_lda, 40
        .equ    .LSQgemmU8KernelFrame_ColumnSumBuffer, 48

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

    rcx - Aligned K.

    r10 - Supplies the length in bytes of a row from matrix A (lda).

    r11 - Supplies the stride in bytes of between packed blocks of matrix B.

    zmm14-zmm31 - Supplies the block accumulators.

--*/

        .macro ComputeBlock Type, Isa, ColumnCount, RowCount, VectorOffset, BroadcastOffset

.if \ColumnCount\() >= 48
        vmovdqu32 zmm0,ZMMWORD PTR [rsi+\VectorOffset\()]
        vmovdqu32 zmm1,ZMMWORD PTR [rsi+r11+\VectorOffset\()]
        vmovdqu32 zmm2,ZMMWORD PTR [rsi+r11*2+\VectorOffset\()]
.elseif \ColumnCount\() >= 32
        vmovdqu32 zmm1,ZMMWORD PTR [rsi+\VectorOffset\()]
        vmovdqu32 zmm2,ZMMWORD PTR [rsi+r11+\VectorOffset\()]
.else
        vmovdqu32 zmm2,ZMMWORD PTR [rsi+\VectorOffset\()]
.endif
        EmitIfCountGE \RowCount\(), 1, "vpbroadcastd zmm3,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm26,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm20,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm14,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 2, "vpbroadcastd zmm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm27,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm21,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm15,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 3, "vpbroadcastd zmm3,DWORD PTR [rdi+r10*2+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm28,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm22,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm16,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 4, "vpbroadcastd zmm3,DWORD PTR [r8+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm29,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm23,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm17,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 5, "vpbroadcastd zmm3,DWORD PTR [r8+r10+\BroadcastOffset\()]"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 48, "MultiplyAccumulateCell\Type\()\Isa\() zmm30,zmm3,zmm0"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 32, "MultiplyAccumulateCell\Type\()\Isa\() zmm24,zmm3,zmm1"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 16, "MultiplyAccumulateCell\Type\()\Isa\() zmm18,zmm3,zmm2"
        EmitIfCountGE \RowCount\(), 6, "vpbroadcastd zmm3,DWORD PTR [r8+r10*2+\BroadcastOffset\()]"
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

    rcx - Aligned K.

    r10 - Supplies the length in bytes of a row from matrix A (lda).

    r11 - Supplies the stride in bytes of between packed blocks of matrix B.

    zmm14-zmm31 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLoopU8S8 Isa, ColumnCount, RowCount

        mov     rbp,rcx                     # reload K remaining

.if  (\RowCount\() == 1) || ((\RowCount\() & 1) == 0)
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

    rcx - Aligned K.

    r10 - Supplies the length in bytes of a row from matrix A (lda).

    r12 - Supplies the address of the column sum buffer.

--*/

        .macro ProduceOutputBlock ColumnCount, RowCount

//
// Initialize the accumulators with the row and column sums.
//

.if \ColumnCount\() >= 32
.if \ColumnCount\() >= 48
        vmovdqu32 zmm26,ZMMWORD PTR [r12]
        vmovdqu32 zmm20,ZMMWORD PTR [r12+64]
        vmovdqu32 zmm14,ZMMWORD PTR [r12+128]
.else
        vmovdqu32 zmm20,ZMMWORD PTR [r12]
        vmovdqu32 zmm14,ZMMWORD PTR [r12+64]
.endif
        add_immed r12,\ColumnCount\()*4     # advance ColumnSumBuffer by N columns
.else
        vmovdqu32 zmm14,ZMMWORD PTR [r12]
.endif
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 16, "vmovdqu32 zmm15,zmm14"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 32, "vmovdqu32 zmm21,zmm20"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 48, "vmovdqu32 zmm27,zmm26"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 16, "vmovdqu32 zmm16,zmm14"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 32, "vmovdqu32 zmm22,zmm20"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 48, "vmovdqu32 zmm28,zmm26"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 16, "vmovdqu32 zmm17,zmm14"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 32, "vmovdqu32 zmm23,zmm20"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 48, "vmovdqu32 zmm29,zmm26"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 16, "vmovdqu32 zmm18,zmm14"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 32, "vmovdqu32 zmm24,zmm20"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 48, "vmovdqu32 zmm30,zmm26"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 16, "vmovdqu32 zmm19,zmm14"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 32, "vmovdqu32 zmm25,zmm20"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 48, "vmovdqu32 zmm31,zmm26"

.LAccumulatorsInitialized\@:

//
// Iterate over the length of a matrix A row to produce the output accumulators.
//

.if \RowCount\() > 3
        lea     r8,[r10*2+r10]
        add     r8,rdi                      # compute matrix A plus 3 rows
.endif
        cmp     DWORD PTR .LSQgemmU8KernelFrame_type[rsp],0
        je      .LProduceWithU8S8Avx512Core\@
        ComputeBlockLoopU8S8 Avx512Vnni, \ColumnCount\(), \RowCount\()
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

    rax - Supplies the length in bytes of a row from matrix C (ldc).

    rdi,rbx - Supplies the address of matrix A.

    rsi - Supplies the address of matrix B.

    rdx - Supplies the address of matrix C.

    r9 - Supplies the number of columns from matrix B and matrix C to iterate
        over (CountN).

    rcx - Aligned K.

    r10 - Supplies the length in bytes of a row from matrix A (lda).

    r12 - Supplies the address of the column sum buffer.

    r11 - Supplies the stride in bytes of between packed blocks of matrix B.

--*/

        .macro ProcessCountM RowCount

        cmp     r9,32
        ja      .LProcessNextColumnLoop48xN\@
        cmp     r9,16
        jbe     .LProcessRemainingCountN\@

.LProcessNextColumnLoop32xN\@:
        ProduceOutputBlock 32, \RowCount\()
        add     rsi,r11                     # advance matrix B by packed block stride

.LOutput32xNBlock\@:
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
        lea     rsi,[rsi+r11*2]             # advance matrix B by packed block stride
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

        FUNCTION_ENTRY MlasSymQgemmU8KernelAvx512Core

        xor     eax,eax
        jmp     C_UNDERSCORE(MlasSymQgemmU8KernelAvx512)

        FUNCTION_ENTRY MlasSymQgemmU8KernelAvx512Vnni

        mov     eax,-1
        jmp     C_UNDERSCORE(MlasSymQgemmU8KernelAvx512)

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

    lda - Supplies the first dimension of matrix A

    ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
        by the zero point offset of matrix A. These values are accumulated into
        every column of matrix C.

Return Value:

    Returns the number of rows handled.

--*/

        FUNCTION_ENTRY MlasSymQgemmU8KernelAvx512

        push    rbp
        push    rbx
        push    r12

        mov     DWORD PTR .LSQgemmU8KernelFrame_type[rsp],eax
        mov     rbx,rdi
        mov     rax,.LSQgemmU8KernelFrame_ldc[rsp]
        shl     rax,2                       # convert ldc to bytes
        shl     rcx,2                       # convert to row length
        mov     r10,.LSQgemmU8KernelFrame_lda[rsp]
        mov     r12,.LSQgemmU8KernelFrame_ColumnSumBuffer[rsp]
        mov     ebp,-1
        kmovw   k1,ebp                      # update mask to write all columns
        neg     ebp
        vpbroadcastw zmm13,ebp              # generate 512-bit word vector [0x0001]
        mov     r11,rcx
        shl     r11,4

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
