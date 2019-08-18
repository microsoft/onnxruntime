/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8U8KernelAvx512Common.inc

Abstract:

    This module contains common kernel macros and structures for the quantized
    integer matrix/matrix multiply operation (QGEMM) for the AVX512BW and
    AVX512VNNI kernels.

--*/

//
// Stack frame layout for the U8U8 kernel.
//

        .equ    .LGemmU8U8KernelFrame_SavedR14, 0
        .equ    .LGemmU8U8KernelFrame_SavedR13, 8
        .equ    .LGemmU8U8KernelFrame_SavedR12, 16
        .equ    .LGemmU8U8KernelFrame_SavedRbx, 24
        .equ    .LGemmU8U8KernelFrame_SavedRbp, 32
        .equ    .LGemmU8U8KernelFrame_ReturnAddress, 40
        .equ    .LGemmU8U8KernelFrame_ldc, 48
        .equ    .LGemmU8U8KernelFrame_RowSumVector, 56
        .equ    .LGemmU8U8KernelFrame_ColumnSumVector, 64
        .equ    .LGemmU8U8KernelFrame_DepthValue, 72
        .equ    .LGemmU8U8KernelFrame_ZeroMode, 80

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

    rcx - Supplies the number of paired columns from matrix A and the number of
        paired rows from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    r12 - Supplies the address of the row sum vector.

    r13 - Supplies the address of the column sum vector.

--*/

        .macro ProduceOutputBlock ColumnCount, RowCount

//
// Initialize the accumulators with the sum of the global depth value constant,
// the column sums, and the row sums.
//

        vpbroadcastd zmm31,DWORD PTR .LGemmU8U8KernelFrame_DepthValue[rsp]
.if \ColumnCount\() == 32
        vpaddd  zmm30,zmm31,ZMMWORD PTR [r13]
        vpaddd  zmm31,zmm31,ZMMWORD PTR [r13+64]
        add     r13,32*4                    # advance ColumnSumVector by 32 columns
.else
        vpaddd zmm31,zmm31,ZMMWORD PTR [r13]
.endif
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 32, "vpaddd zmm16,zmm30,DWORD PTR [r12]{1to16}"
        EmitIfCountGE \RowCount\(), 1, "vpaddd zmm17,zmm31,DWORD PTR [r12]{1to16}"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 32, "vpaddd zmm18,zmm30,DWORD PTR [r12+4]{1to16}"
        EmitIfCountGE \RowCount\(), 2, "vpaddd zmm19,zmm31,DWORD PTR [r12+4]{1to16}"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 32, "vpaddd zmm20,zmm30,DWORD PTR [r12+8]{1to16}"
        EmitIfCountGE \RowCount\(), 3, "vpaddd zmm21,zmm31,DWORD PTR [r12+8]{1to16}"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 32, "vpaddd zmm22,zmm30,DWORD PTR [r12+12]{1to16}"
        EmitIfCountGE \RowCount\(), 4, "vpaddd zmm23,zmm31,DWORD PTR [r12+12]{1to16}"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 32, "vpaddd zmm24,zmm30,DWORD PTR [r12+16]{1to16}"
        EmitIfCountGE \RowCount\(), 5, "vpaddd zmm25,zmm31,DWORD PTR [r12+16]{1to16}"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 32, "vpaddd zmm26,zmm30,DWORD PTR [r12+20]{1to16}"
        EmitIfCountGE \RowCount\(), 6, "vpaddd zmm27,zmm31,DWORD PTR [r12+20]{1to16}"

//
// Iterate over PairedCountK elements from matrix A and matrix B.
//

        mov     rbp,rcx                     # reload PairedCountK
.if \RowCount\() > 3
        lea     rbx,[r10*2+r10]
        add     rbx,rdi                     # compute matrix A plus 3 rows
.endif

.LComputeBlockLoop.\ColumnCount\().\RowCount\():
        ComputeBlock \ColumnCount\(), \RowCount\()
        add     rdi,4                       # advance matrix A by 1 pair
.if \RowCount\() > 3
        add     rbx,4                       # advance matrix A plus 3 rows by 1 pair
.endif
        add     rsi,32
        dec     rbp                         # decrement pairs remaining
        jnz     .LComputeBlockLoop.\ColumnCount\().\RowCount\()

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

    rcx - Supplies the number of paired columns from matrix A and the number of
        paired rows from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    r12 - Supplies the address of the row sum vector.

    r13 - Supplies the address of the column sum vector.

    r14b - Supplies the zero mode flag.

--*/

        .macro ProcessCountM RowCount

        cmp     r9,16
        jbe     .LProcessRemainingCountN.\RowCount\()

.LProcessNextColumnLoop32xN.\RowCount\():
        ProduceOutputBlock 32, \RowCount\()
        lea     rsi,[rsi+r10*8]             # advance matrix B by 8*PairedCountK
        test    r14b,r14b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput32xNBlock.\RowCount\()
        EmitIfCountGE \RowCount\(), 1, "vpaddd zmm16,zmm16,ZMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd zmm18,zmm18,ZMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd zmm20,zmm20,ZMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd zmm22,zmm22,ZMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd zmm24,zmm24,ZMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd zmm26,zmm26,ZMMWORD PTR [rbx+rax*2]"

.LSkipAccumulateOutput32xNBlock.\RowCount\():
        EmitIfCountGE \RowCount\(), 1, "vmovdqu32 ZMMWORD PTR [rdx],zmm16"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu32 ZMMWORD PTR [rdx+rax],zmm18"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu32 ZMMWORD PTR [rdx+rax*2],zmm20"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu32 ZMMWORD PTR [rbx],zmm22"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu32 ZMMWORD PTR [rbx+rax],zmm24"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu32 ZMMWORD PTR [rbx+rax*2],zmm26"
        add     rdx,16*4                    # advance matrix C by 16 columns
.if \RowCount\() > 3
        add     rbx,16*4                    # advance matrix C plus 3 rows by 16 columns
.endif
        sub     r9,16

.LOutput16xNBlock.\RowCount\():
        sub     r9,16
        jae     .LOutput16xNBlockWithMask.\RowCount\()
        lea     rcx,[r9+16]                 # correct for over-subtract above
        mov     ebp,1
        shl     ebp,cl
        dec     ebp
        kmovw   k1,ebp                      # update mask for remaining columns
        xor     r9,r9                       # no more columns remaining

.LOutput16xNBlockWithMask.\RowCount\():
        test    r14b,r14b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput16xNBlockWithMask.\RowCount\()
        EmitIfCountGE \RowCount\(), 1, "vpaddd zmm17{k1},zmm17,ZMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd zmm19{k1},zmm19,ZMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd zmm21{k1},zmm21,ZMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd zmm23{k1},zmm23,ZMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd zmm25{k1},zmm25,ZMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd zmm27{k1},zmm27,ZMMWORD PTR [rbx+rax*2]"

.LSkipAccumulateOutput16xNBlockWithMask.\RowCount\():
        EmitIfCountGE \RowCount\(), 1, "vmovdqu32 ZMMWORD PTR [rdx]{k1},zmm17"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu32 ZMMWORD PTR [rdx+rax]{k1},zmm19"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu32 ZMMWORD PTR [rdx+rax*2]{k1},zmm21"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu32 ZMMWORD PTR [rbx]{k1},zmm23"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu32 ZMMWORD PTR [rbx+rax]{k1},zmm25"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu32 ZMMWORD PTR [rbx+rax*2]{k1},zmm27"
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        cmp     r9,16
        ja      .LProcessNextColumnLoop32xN.\RowCount\()
        test    r9,r9
        jz      .LExitKernel

.LProcessRemainingCountN.\RowCount\():
        ProduceOutputBlock 16, \RowCount\()
        jmp     .LOutput16xNBlock.\RowCount\()

        .endm

/*++

Macro Description:

    This macro generates the common AVX512 code for the inner kernel to compute
    matrix multiplication.

Arguments:

    Isa - Supplies the instruction set architecture string for function tags.

--*/

        .macro GemmU8U8KernelAvx512Function Isa

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (rdi) - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8U8CopyPackAAvx2.

    B (rsi) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8U8CopyPackBAvx2.

    C (rdx) - Supplies the address of matrix C.

    PairedCountK (rcx) - Supplies the number of paired columns from matrix A and
        the number of paired rows from matrix B to iterate over.

    CountM (r8) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (r9) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldc - Supplies the first dimension of matrix C.

    RowSumVector - Supplies the sum of each row from matrix A multiplied by the
        zero point offset of matrix B. These values are accumulated into every
        row of matrix C.

    ColumnSumVector - Supplies the sum of each column from matrix B multiplied
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

        .globl  C_UNDERSCORE(MlasGemmU8U8Kernel\Isa\())
C_UNDERSCORE(MlasGemmU8U8Kernel\Isa\()):

        push    rbp
        push    rbx
        push    r12
        push    r13
        push    r14

        mov     rax,.LGemmU8U8KernelFrame_ldc[rsp]
        shl     rax,2                       # convert ldc to bytes
        lea     r10,[rcx*4]
        mov     r11,rdi
        mov     r12,.LGemmU8U8KernelFrame_RowSumVector[rsp]
        mov     r13,.LGemmU8U8KernelFrame_ColumnSumVector[rsp]
        movzx   r14,BYTE PTR .LGemmU8U8KernelFrame_ZeroMode[rsp]
        mov     ebp,-1
        kmovw   k1,ebp                      # update mask to write all columns

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
        mov     r8d,6                       # return 6 rows handled
        ProcessCountM 6

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        mov     eax,r8d

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
