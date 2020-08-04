/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8X8KernelAvx512Common.h

Abstract:

    This module contains common kernel macros and structures for the quantized
    integer matrix/matrix multiply operation (QGEMM) for the AVX512 core and
    AVX512VNNI kernels.

--*/

//
// Stack frame layout for the U8S8 and U8U8 kernels.
//

        .equ    .LGemmU8X8KernelFrame_SavedR14, 0
        .equ    .LGemmU8X8KernelFrame_SavedR13, 8
        .equ    .LGemmU8X8KernelFrame_SavedR12, 16
        .equ    .LGemmU8X8KernelFrame_SavedRbx, 24
        .equ    .LGemmU8X8KernelFrame_SavedRbp, 32
        .equ    .LGemmU8X8KernelFrame_ReturnAddress, 40
        .equ    .LGemmU8X8KernelFrame_ldc, 48
        .equ    .LGemmU8X8KernelFrame_RowSumBuffer, 56
        .equ    .LGemmU8X8KernelFrame_ColumnSumBuffer, 64
        .equ    .LGemmU8X8KernelFrame_DepthValue, 72
        .equ    .LGemmU8X8KernelFrame_ZeroMode, 80

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

--*/

        .macro ProduceOutputBlock ColumnCount, RowCount

//
// Initialize the accumulators with the sum of the global depth value constant,
// the column sums, and the row sums.
//

        vpbroadcastd zmm3,DWORD PTR .LGemmU8X8KernelFrame_DepthValue[rsp]
.if \ColumnCount\() >= 32
.if \ColumnCount\() >= 48
        vpaddd  zmm2,zmm3,ZMMWORD PTR [r13]
        vpaddd  zmm1,zmm3,ZMMWORD PTR [r13+64]
        vpaddd  zmm0,zmm3,ZMMWORD PTR [r13+128]
.else
        vpaddd  zmm1,zmm3,ZMMWORD PTR [r13]
        vpaddd  zmm0,zmm3,ZMMWORD PTR [r13+64]
.endif
        add_immed r13,\ColumnCount\()*4     # advance ColumnSumBuffer by N columns
.else
        vpaddd zmm0,zmm3,ZMMWORD PTR [r13]
.endif
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 16, "vpaddd zmm14,zmm0,DWORD PTR [r12]{1to16}"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 32, "vpaddd zmm20,zmm1,DWORD PTR [r12]{1to16}"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 48, "vpaddd zmm26,zmm2,DWORD PTR [r12]{1to16}"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 16, "vpaddd zmm15,zmm0,DWORD PTR [r12+4]{1to16}"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 32, "vpaddd zmm21,zmm1,DWORD PTR [r12+4]{1to16}"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 48, "vpaddd zmm27,zmm2,DWORD PTR [r12+4]{1to16}"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 16, "vpaddd zmm16,zmm0,DWORD PTR [r12+8]{1to16}"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 32, "vpaddd zmm22,zmm1,DWORD PTR [r12+8]{1to16}"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 48, "vpaddd zmm28,zmm2,DWORD PTR [r12+8]{1to16}"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 16, "vpaddd zmm17,zmm0,DWORD PTR [r12+12]{1to16}"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 32, "vpaddd zmm23,zmm1,DWORD PTR [r12+12]{1to16}"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 48, "vpaddd zmm29,zmm2,DWORD PTR [r12+12]{1to16}"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 16, "vpaddd zmm18,zmm0,DWORD PTR [r12+16]{1to16}"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 32, "vpaddd zmm24,zmm1,DWORD PTR [r12+16]{1to16}"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 48, "vpaddd zmm30,zmm2,DWORD PTR [r12+16]{1to16}"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 16, "vpaddd zmm19,zmm0,DWORD PTR [r12+20]{1to16}"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 32, "vpaddd zmm25,zmm1,DWORD PTR [r12+20]{1to16}"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 48, "vpaddd zmm31,zmm2,DWORD PTR [r12+20]{1to16}"

//
// Iterate over the length of a matrix A row to produce the output accumulators.
//

.if \RowCount\() > 3
        lea     rbx,[rcx*2+rcx]
        add     rbx,rdi                     # compute matrix A plus 3 rows
.endif
        ComputeBlockLoop \ColumnCount\(), \RowCount\()
.if \RowCount\() > 3
        lea     rbx,[rdx+rax*2]             # compute matrix C plus 3 rows
        add     rbx,rax
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

    r11 - Supplies the address of matrix A.

    r9 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    rcx - Supplies the length in bytes of a row from matrix A.

    r10b - Supplies the zero mode flag.

    r12 - Supplies the address of the row sum buffer.

    r13 - Supplies the address of the column sum buffer.

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
        EmitIfCountGE \RowCount\(), 4, "vpaddd zmm23,zmm23,ZMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd zmm24,zmm24,ZMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd zmm25,zmm25,ZMMWORD PTR [rbx+rax*2]"

.LSkipAccumulateOutput32xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovdqu32 ZMMWORD PTR [rdx],zmm20"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu32 ZMMWORD PTR [rdx+rax],zmm21"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu32 ZMMWORD PTR [rdx+rax*2],zmm22"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu32 ZMMWORD PTR [rbx],zmm23"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu32 ZMMWORD PTR [rbx+rax],zmm24"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu32 ZMMWORD PTR [rbx+rax*2],zmm25"
        add     rdx,16*4                    # advance matrix C by 16 columns
.if \RowCount\() > 3
        add     rbx,16*4                    # advance matrix C plus 3 rows by 16 columns
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
        EmitIfCountGE \RowCount\(), 4, "vpaddd zmm17{k1},zmm17,ZMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd zmm18{k1},zmm18,ZMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd zmm19{k1},zmm19,ZMMWORD PTR [rbx+rax*2]"

.LSkipAccumulateOutput16xNBlockWithMask\@:
        EmitIfCountGE \RowCount\(), 1, "vmovdqu32 ZMMWORD PTR [rdx]{k1},zmm14"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu32 ZMMWORD PTR [rdx+rax]{k1},zmm15"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu32 ZMMWORD PTR [rdx+rax*2]{k1},zmm16"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu32 ZMMWORD PTR [rbx]{k1},zmm17"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu32 ZMMWORD PTR [rbx+rax]{k1},zmm18"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu32 ZMMWORD PTR [rbx+rax*2]{k1},zmm19"
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        cmp     r9,32
        ja      .LProcessNextColumnLoop48xN\@
        cmp     r9,16
        ja      .LProcessNextColumnLoop32xN\@
        test    r9,r9
        jz      .LExitKernel

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
        EmitIfCountGE \RowCount\(), 4, "vpaddd zmm29,zmm29,ZMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd zmm30,zmm30,ZMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd zmm31,zmm31,ZMMWORD PTR [rbx+rax*2]"

.LSkipAccumulateOutput48xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovdqu32 ZMMWORD PTR [rdx],zmm26"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu32 ZMMWORD PTR [rdx+rax],zmm27"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu32 ZMMWORD PTR [rdx+rax*2],zmm28"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu32 ZMMWORD PTR [rbx],zmm29"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu32 ZMMWORD PTR [rbx+rax],zmm30"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu32 ZMMWORD PTR [rbx+rax*2],zmm31"
        add     rdx,16*4                    # advance matrix C by 16 columns
.if \RowCount\() > 3
        add     rbx,16*4                    # advance matrix C plus 3 rows by 16 columns
.endif
        sub     r9,16
        jmp     .LOutput32xNBlock\@

        .endm

/*++

Macro Description:

    This macro generates the common AVX512 code for the inner kernel to compute
    matrix multiplication.

Arguments:

    Type - Supplies the kernel type string for function tags.

    Isa - Supplies the instruction set architecture string for function tags.

--*/

        .macro GemmU8X8KernelAvx512Function Type, Isa

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

    RowSumBuffer - Supplies the sum of each row from matrix A multiplied by the
        zero point offset of matrix B. These values are accumulated into every
        row of matrix C.

    ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
        by the zero point offset of matrix A. These values are accumulated into
        every column of matrix C.

    DepthValue - Supplies the value CountK multiplied by the zero point offset
        of matrixA multplied by the zero point offset of matrix B. This value is
        accumulated into every element of matrix C.

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        .globl  C_UNDERSCORE(MlasGemm\Type\()Kernel\Isa\())
C_UNDERSCORE(MlasGemm\Type\()Kernel\Isa\()):

        push    rbp
        push    rbx
        push    r12
        push    r13
        push    r14

        mov     rax,.LGemmU8X8KernelFrame_ldc[rsp]
        shl     rax,2                       # convert ldc to bytes
        shl     rcx,2                       # convert to row length
        movzx   r10,BYTE PTR .LGemmU8X8KernelFrame_ZeroMode[rsp]
        mov     r11,rdi
        mov     r12,.LGemmU8X8KernelFrame_RowSumBuffer[rsp]
        mov     r13,.LGemmU8X8KernelFrame_ColumnSumBuffer[rsp]
        mov     ebp,-1
        kmovw   k1,ebp                      # update mask to write all columns
.ifeqs "\Type\()", "U8S8"
.ifeqs "\Isa\()", "Avx512Core"
        neg     ebp
        vpbroadcastw zmm5,ebp               # generate 512-bit word vector [0x0001]
.endif
        mov     r14,rcx
        shl     r14,4                       # compute matrix B packed stride
.else
        lea     r14,[rcx*8]                 # compute matrix B packed stride
.endif

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
        ProcessCountM 6

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        mov     eax,r8d
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

        .endm
