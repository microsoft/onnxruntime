/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8X8KernelAvx2Common.h

Abstract:

    This module contains common kernel macros and structures for the quantized
    integer matrix/matrix multiply operation (QGEMM) for the AVX2 kernels.

--*/

//
// Stack frame layout for the U8S8 and U8U8 kernels.
//

        .equ    .LGemmU8X8KernelFrame_mask, -8
        .equ    .LGemmU8X8KernelFrame_SavedR13, 0
        .equ    .LGemmU8X8KernelFrame_SavedR12, 8
        .equ    .LGemmU8X8KernelFrame_SavedRbx, 16
        .equ    .LGemmU8X8KernelFrame_SavedRbp, 24
        .equ    .LGemmU8X8KernelFrame_ReturnAddress, 32
        .equ    .LGemmU8X8KernelFrame_ldc, 40
        .equ    .LGemmU8X8KernelFrame_RowSumBuffer, 48
        .equ    .LGemmU8X8KernelFrame_ColumnSumBuffer, 56
        .equ    .LGemmU8X8KernelFrame_DepthValue, 64
        .equ    .LGemmU8X8KernelFrame_ZeroMode, 72

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

    r12 - Supplies the address of the row sum buffer.

    r13 - Supplies the address of the column sum buffer.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ProduceOutputBlock ColumnCount, RowCount

//
// Initialize the accumulators with the sum of the global depth value constant,
// the column sums, and the row sums.
//

        vpbroadcastd ymm1,DWORD PTR .LGemmU8X8KernelFrame_DepthValue[rsp]
.if \ColumnCount\() == 16
        vpaddd  ymm0,ymm1,YMMWORD PTR [r13]
        vpaddd  ymm1,ymm1,YMMWORD PTR [r13+32]
        add     r13,16*4                    # advance ColumnSumBuffer by 16 columns
.else
        vpaddd  ymm1,ymm1,YMMWORD PTR [r13]
.endif
        EmitIfCountGE \RowCount\(), 1, "vpbroadcastd ymm5,DWORD PTR [r12]"
        EmitIfCountGE \RowCount\(), 2, "vpbroadcastd ymm7,DWORD PTR [r12+4]"
        EmitIfCountGE \RowCount\(), 3, "vpbroadcastd ymm9,DWORD PTR [r12+8]"
        EmitIfCountGE \RowCount\(), 4, "vpbroadcastd ymm11,DWORD PTR [r12+12]"
        EmitIfCountGE \RowCount\(), 5, "vpbroadcastd ymm13,DWORD PTR [r12+16]"
        EmitIfCountGE \RowCount\(), 6, "vpbroadcastd ymm15,DWORD PTR [r12+20]"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 16, "vpaddd ymm4,ymm5,ymm0"
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm5,ymm5,ymm1"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 16, "vpaddd ymm6,ymm7,ymm0"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm7,ymm7,ymm1"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 16, "vpaddd ymm8,ymm9,ymm0"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm9,ymm9,ymm1"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 16, "vpaddd ymm10,ymm11,ymm0"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm11,ymm11,ymm1"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 16, "vpaddd ymm12,ymm13,ymm0"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm13,ymm13,ymm1"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 16, "vpaddd ymm14,ymm15,ymm0"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm15,ymm15,ymm1"

//
// Iterate over the length of a matrix A row to produce the output accumulators.
//

.if \RowCount\() > 3
        lea     rbx,[rcx*2+rcx]
        add     rbx,rdi                     # compute matrix A plus 3 rows
.endif
        ComputeBlockLoop \ColumnCount\(), \RowCount\()
.if \RowCount\() > 3
        lea     rbx,[rax*2+rax]
        add     rbx,rdx                     # compute matrix C plus 3 rows
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute matrix multiplication for a fixed set
    of rows.

Arguments:

    RowCount - Supplies the number of rows to process.

    Fallthrough - Supplies a non-blank value if the macro may fall through to
        the ExitKernel label.

Implicit Arguments:

    rax - Supplies the length in bytes of a row from matrix C.

    rdi - Supplies the address of matrix A.

    rsi - Supplies the address of matrix B.

    rdx - Supplies the address of matrix C.

    r11 - Supplies the address of matrix A.

    r9 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    rcx - Supplies the length in bytes of a row from matrix A.

    r10b - Supplies the zero mode flag.

    r12 - Supplies the address of the row sum buffer.

    r13 - Supplies the address of the column sum buffer.

--*/

        .macro ProcessCountM RowCount, Fallthrough

        cmp     r9,8
        jbe     .LProcessRemainingCountN\@

.LProcessNextColumnLoop16xN\@:
        ProduceOutputBlock 16, \RowCount\()
        sub     r9,16
        jb      .LOutputMasked16xNBlock\@
        test    r10b,r10b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput16xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm4,ymm4,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm5,ymm5,YMMWORD PTR [rdx+32]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm6,ymm6,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm7,ymm7,YMMWORD PTR [rdx+rax+32]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm8,ymm8,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm9,ymm9,YMMWORD PTR [rdx+rax*2+32]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm10,ymm10,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm11,ymm11,YMMWORD PTR [rbx+32]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm12,ymm12,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm13,ymm13,YMMWORD PTR [rbx+rax+32]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm14,ymm14,YMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm15,ymm15,YMMWORD PTR [rbx+rax*2+32]"

.LSkipAccumulateOutput16xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovdqu YMMWORD PTR [rdx],ymm4"
        EmitIfCountGE \RowCount\(), 1, "vmovdqu YMMWORD PTR [rdx+32],ymm5"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu YMMWORD PTR [rdx+rax],ymm6"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu YMMWORD PTR [rdx+rax+32],ymm7"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu YMMWORD PTR [rdx+rax*2],ymm8"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu YMMWORD PTR [rdx+rax*2+32],ymm9"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu YMMWORD PTR [rbx],ymm10"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu YMMWORD PTR [rbx+32],ymm11"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu YMMWORD PTR [rbx+rax],ymm12"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu YMMWORD PTR [rbx+rax+32],ymm13"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu YMMWORD PTR [rbx+rax*2],ymm14"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu YMMWORD PTR [rbx+rax*2+32],ymm15"
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        cmp     r9,8
        ja      .LProcessNextColumnLoop16xN\@
        test    r9,r9
        jz      .LExitKernel

.LProcessRemainingCountN\@:
        ProduceOutputBlock 8, \RowCount\()
        cmp     r9,8
        jb      .LOutputMasked8xNBlock\@
        test    r10b,r10b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput8xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm5,ymm5,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm7,ymm7,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm9,ymm9,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm11,ymm11,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm13,ymm13,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm15,ymm15,YMMWORD PTR [rbx+rax*2]"

.LSkipAccumulateOutput8xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovdqu YMMWORD PTR [rdx],ymm5"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu YMMWORD PTR [rdx+rax],ymm7"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu YMMWORD PTR [rdx+rax*2],ymm9"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu YMMWORD PTR [rbx],ymm11"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu YMMWORD PTR [rbx+rax],ymm13"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu YMMWORD PTR [rbx+rax*2],ymm15"
        jmp     .LExitKernel

.LOutputMasked16xNBlock\@:
        test    r10b,r10b                   # ZeroMode?
        jnz     .LSkipAccumulateOutputMasked16xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm4,ymm4,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm6,ymm6,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm8,ymm8,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm10,ymm10,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm12,ymm12,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm14,ymm14,YMMWORD PTR [rbx+rax*2]"

.LSkipAccumulateOutputMasked16xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovdqu YMMWORD PTR [rdx],ymm4"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu YMMWORD PTR [rdx+rax],ymm6"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu YMMWORD PTR [rdx+rax*2],ymm8"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu YMMWORD PTR [rbx],ymm10"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu YMMWORD PTR [rbx+rax],ymm12"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu YMMWORD PTR [rbx+rax*2],ymm14"
        add     rdx,8*4                     # advance matrix C by 8 columns
.if \RowCount\() > 3
        add     rbx,8*4                     # advance matrix C plus 3 rows by 8 columns
.endif
        add     r9,8                        # correct for over-subtract above

.LOutputMasked8xNBlock\@:
        mov     DWORD PTR .LGemmU8X8KernelFrame_mask[rsp],r9d
        vpbroadcastd ymm0,DWORD PTR .LGemmU8X8KernelFrame_mask[rsp]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        test    r10b,r10b                   # ZeroMode?
        jnz     .LSkipAccumulateOutputMasked8xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vpmaskmovd ymm4,ymm0,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpmaskmovd ymm6,ymm0,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpmaskmovd ymm8,ymm0,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpmaskmovd ymm10,ymm0,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpmaskmovd ymm12,ymm0,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpmaskmovd ymm14,ymm0,YMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm5,ymm5,ymm4"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm7,ymm7,ymm6"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm9,ymm9,ymm8"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm11,ymm11,ymm10"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm13,ymm13,ymm12"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm15,ymm15,ymm14"

.LSkipAccumulateOutputMasked8xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vpmaskmovd YMMWORD PTR [rdx],ymm0,ymm5"
        EmitIfCountGE \RowCount\(), 2, "vpmaskmovd YMMWORD PTR [rdx+rax],ymm0,ymm7"
        EmitIfCountGE \RowCount\(), 3, "vpmaskmovd YMMWORD PTR [rdx+rax*2],ymm0,ymm9"
        EmitIfCountGE \RowCount\(), 4, "vpmaskmovd YMMWORD PTR [rbx],ymm0,ymm11"
        EmitIfCountGE \RowCount\(), 5, "vpmaskmovd YMMWORD PTR [rbx+rax],ymm0,ymm13"
        EmitIfCountGE \RowCount\(), 6, "vpmaskmovd YMMWORD PTR [rbx+rax*2],ymm0,ymm15"
.ifb \Fallthrough\()
        jmp     .LExitKernel
.endif

        .endm
