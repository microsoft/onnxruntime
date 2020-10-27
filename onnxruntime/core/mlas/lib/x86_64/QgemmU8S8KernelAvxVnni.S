/*++

Copyright (c) 2020 Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8S8KernelAvxVnni.s

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

    This implementation uses AVXVNNI instructions.

--*/

#include "asmmacro.h"
#include "QgemmU8X8KernelAvx2Common.h"
#include "AssembleAvxVnni.h"

        .intel_syntax noprefix

/*++
Macro Description:

    This macro generates code to multiply and accumulator a single row of the
    output block.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    Vec1Reg - Supplies the high block accumulator register (when ColumnCount
        is 16).

    Vec2Reg - Supplies the low block accumulator register.

Implicit Arguments:

    ymm0 - Supplies the first vector loaded from matrix B.

    ymm1 - Supplies the second vector loaded from matrix B (when ColumnCount
        is 16).

    ymm2 - Supplies the broadcast value loaded from matrix A.

--*/

        .macro MultiplyAccumulateRow ColumnCount, Vec1Reg, Vec2Reg

.if \ColumnCount\() == 16
        VpdpbusdsYmmYmmYmm \Vec1Reg\(),ymm2,ymm0
        VpdpbusdsYmmYmmYmm \Vec2Reg\(),ymm2,ymm1
.else
        VpdpbusdsYmmYmmYmm \Vec2Reg\(),ymm2,ymm0
.endif

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

    rcx - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlock ColumnCount, RowCount, VectorOffset, BroadcastOffset

        vmovdqu ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        EmitIfCountGE \ColumnCount\(), 16, "vmovdqu ymm1,YMMWORD PTR [rsi+\VectorOffset\()+32]"
        EmitIfCountGE \RowCount\(), 1, "vpbroadcastd ymm2,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "MultiplyAccumulateRow \ColumnCount\(), ymm4, ymm5"
        EmitIfCountGE \RowCount\(), 2, "vpbroadcastd ymm2,DWORD PTR [rdi+rcx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "MultiplyAccumulateRow \ColumnCount\(), ymm6, ymm7"
        EmitIfCountGE \RowCount\(), 3, "vpbroadcastd ymm2,DWORD PTR [rdi+rcx*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "MultiplyAccumulateRow \ColumnCount\(), ymm8, ymm9"
        EmitIfCountGE \RowCount\(), 4, "vpbroadcastd ymm2,DWORD PTR [rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "MultiplyAccumulateRow \ColumnCount\(), ymm10, ymm11"
        EmitIfCountGE \RowCount\(), 5, "vpbroadcastd ymm2,DWORD PTR [rbx+r9+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 5, "MultiplyAccumulateRow \ColumnCount\(), ymm12, ymm13"
        EmitIfCountGE \RowCount\(), 6, "vpbroadcastd ymm2,DWORD PTR [rbx+r9*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 6, "MultiplyAccumulateRow \ColumnCount\(), ymm14, ymm15"

        .endm

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple
    times and advancing the matrix A and matrix B data pointers.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    RowCount - Supplies the number of rows to produce.

Implicit Arguments:

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rdi - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLoop ColumnCount, RowCount

        mov     rbp,rcx                     # reload row length remaining

.LComputeBlockBy1Loop\@:
        ComputeBlock \ColumnCount\(), \RowCount\(), 0, 0
        add     rdi,4                       # advance matrix A by 1 quad
.if \RowCount\() > 3
        add     rbx,4                       # advance matrix A plus 3 rows by 1 quad
.endif
        add     rsi,64                      # advance matrix B
        sub     rbp,4
        jnz     .LComputeBlockBy1Loop\@

        .endm

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (rdi) - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8S8CopyPackAAvx2.

    B (rsi) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8S8CopyPackBAvx2.

    C (rdx) - Supplies the address of matrix C.

    PackedCountK (rcx) - Supplies the number of packed columns from matrix A
        and the number of packed rows from matrix B to iterate over.

    CountM (r8) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (r9) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldc - Supplies the first dimension of matrix C.

    RowSumBuffer - Supplies the sum of each row from matrix A multiplied by the
        zero point offset of matrix B. These values are accumulated into every
        row of matrix C.

    ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
        by the zero point offset of matrix A. These values are accumulated into
        every column of matrix C.

    DepthValue - Supplies the value CountK multiplied by the zero point offset
        of matrix A multplied by the zero point offset of matrix B. This value
        is accumulated into every element of matrix C.

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        .globl  C_UNDERSCORE(MlasGemmU8S8KernelAvxVnni)
C_UNDERSCORE(MlasGemmU8S8KernelAvxVnni):

        push    rbp
        push    rbx
        push    r12
        push    r13

        mov     rax,.LGemmU8X8KernelFrame_ldc[rsp]
        shl     rax,2                       # convert ldc to bytes
        shl     rcx,2                       # convert to row length
        movzx   r10,BYTE PTR .LGemmU8X8KernelFrame_ZeroMode[rsp]
        mov     r11,rdi
        mov     r12,.LGemmU8X8KernelFrame_RowSumBuffer[rsp]
        mov     r13,.LGemmU8X8KernelFrame_ColumnSumBuffer[rsp]

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
        mov     r8d,6                      # return 6 rows handled
        ProcessCountM 6, Fallthrough

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        mov     eax,r8d
        vzeroupper

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
