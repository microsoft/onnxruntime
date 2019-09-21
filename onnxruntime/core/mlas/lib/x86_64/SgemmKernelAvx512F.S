/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelAvx512F.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses AVX512F instructions.

--*/

#include "asmmacro.h"
#include "SgemmKernelCommon.h"

        .intel_syntax noprefix

        .equ    .LSgemmKernelFrame_SavedR15, 0
        .equ    .LSgemmKernelFrame_SavedR14, 8
        .equ    .LSgemmKernelFrame_SavedR13, 16
        .equ    .LSgemmKernelFrame_SavedR12, 24
        .equ    .LSgemmKernelFrame_SavedRbx, 32
        .equ    .LSgemmKernelFrame_SavedRbp, 40
        .equ    .LSgemmKernelFrame_ReturnAddress, 48
        .equ    .LSgemmKernelFrame_lda, 56
        .equ    .LSgemmKernelFrame_ldc, 64
        .equ    .LSgemmKernelFrame_ZeroMode, 72

        .text

/*++

Macro Description:

    This macro multiplies and accumulates for a 32xN block of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

    PrefetchOffset - Optionally supplies the byte offset from matrix B to
        prefetch elements.

Implicit Arguments:

    rcx - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    r13 - Supplies the address into the matrix A data plus 6 rows.

    r14 - Supplies the address into the matrix A data plus 9 rows.

    rdx - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    zmm4-zmm27 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvx512FBy32 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.ifnb \PrefetchOffset\()
        prefetcht0 [rsi+\VectorOffset\()+\PrefetchOffset\()]
        prefetcht0 [rsi+r12+\VectorOffset\()+\PrefetchOffset\()]
.endif
.if \RowCount\() == 1
        vbroadcastss zmm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vfmadd231ps zmm4,zmm3,ZMMWORD PTR [rsi+\VectorOffset\()]
        vfmadd231ps zmm5,zmm3,ZMMWORD PTR [rsi+r12+\VectorOffset\()]
.else
        vmovaps zmm0,ZMMWORD PTR [rsi+\VectorOffset\()]
        vmovaps zmm1,ZMMWORD PTR [rsi+r12+\VectorOffset\()]
        EmitIfCountGE \RowCount\(), 1, "vbroadcastss zmm3,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "vfmadd231ps zmm4,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 1, "vfmadd231ps zmm5,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 2, "vbroadcastss zmm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231ps zmm6,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231ps zmm7,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 3, "vbroadcastss zmm3,DWORD PTR [rdi+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231ps zmm8,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231ps zmm9,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 4, "vbroadcastss zmm3,DWORD PTR [rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231ps zmm10,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231ps zmm11,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 5, "vbroadcastss zmm3,DWORD PTR [rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231ps zmm12,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231ps zmm13,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 6, "vbroadcastss zmm3,DWORD PTR [rbx+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231ps zmm14,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231ps zmm15,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastss zmm3,DWORD PTR [r13+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm16,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm17,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastss zmm3,DWORD PTR [r13+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm18,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm19,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastss zmm3,DWORD PTR [r13+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm20,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm21,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastss zmm3,DWORD PTR [r14+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm22,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm23,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastss zmm3,DWORD PTR [r14+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm24,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm25,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastss zmm3,DWORD PTR [r14+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm26,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm27,zmm3,zmm1"
.endif

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for a 16xN block of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

    PrefetchOffset - Optionally supplies the byte offset from matrix B to
        prefetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    r13 - Supplies the address into the matrix A data plus 6 rows.

    r14 - Supplies the address into the matrix A data plus 9 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    zmm4-zmm27 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvx512FBy16 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.ifnb \PrefetchOffset\()
        prefetcht0 [rsi+\VectorOffset\()+\PrefetchOffset\()]
.endif
        vmovaps zmm0,ZMMWORD PTR [rsi+\VectorOffset\()]
        EmitIfCountGE \RowCount\(), 1, "vfmadd231ps zmm5,zmm0,DWORD PTR [rdi+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231ps zmm7,zmm0,DWORD PTR [rdi+r10+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231ps zmm9,zmm0,DWORD PTR [rdi+r10*2+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231ps zmm11,zmm0,DWORD PTR [rbx+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231ps zmm13,zmm0,DWORD PTR [rbx+r10+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231ps zmm15,zmm0,DWORD PTR [rbx+r10*2+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm17,zmm0,DWORD PTR [r13+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm19,zmm0,DWORD PTR [r13+r10+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm21,zmm0,DWORD PTR [r13+r10*2+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm23,zmm0,DWORD PTR [r14+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm25,zmm0,DWORD PTR [r14+r10+\BroadcastOffset\()]{1to16}"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231ps zmm27,zmm0,DWORD PTR [r14+r10*2+\BroadcastOffset\()]{1to16}"

        .endm

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple
    times and advancing the matrix A and matrix B data pointers.

Arguments:

    ComputeBlock - Supplies the macro to compute a single block.

    RowCount - Supplies the number of rows to process.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    r13 - Supplies the address into the matrix A data plus 6 rows.

    r14 - Supplies the address into the matrix A data plus 9 rows.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    zmm4-zmm27 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvx512FLoop ComputeBlock, RowCount

.if \RowCount\() > 3
        lea     rbx,[r10*2+r10]
.if \RowCount\() == 12
        lea     r13,[rdi+rbx*2]             # compute matrix A plus 6 rows
        lea     r14,[r13+rbx]               # compute matrix A plus 9 rows
.endif
        add     rbx,rdi                     # compute matrix A plus 3 rows
.endif
        ComputeBlockLoop \ComputeBlock\(), \RowCount\(), \RowCount\() > 3
.if \RowCount\() > 3
        lea     rbx,[rax*2+rax]
.if \RowCount\() == 12
        lea     r13,[rdx+rbx*2]             # compute matrix C plus 6 rows
        lea     r14,[r13+rbx]               # compute matrix C plus 9 rows
.endif
        add     rbx,rdx                     # compute matrix C plus 3 rows
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute matrix multiplication for a fixed set
    of rows.

Arguments:

    RowCount - Supplies the number of rows to process.

Implicit Arguments:

    rdi - Supplies the address of matrix A.

    rsi - Supplies the address of matrix B.

    r11 - Supplies the address of matrix A.

    r9 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    rdx - Supplies the address of matrix C.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    rax - Supplies the length in bytes of a row from matrix C.

    r15 - Stores the ZeroMode argument from the stack frame.

--*/

        .macro ProcessCountM RowCount

        cmp     r9,16
        jbe     .LProcessRemainingCountN\@

.LProcessNextColumnLoop32xN\@:
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm16,zmm4"
                                            # clear upper block accumulators
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm17,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm18,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm19,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm20,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm21,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm22,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm23,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm24,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm25,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm26,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm27,zmm5"
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy32, \RowCount\()
        add     rsi,r12                     # advance matrix B by 16*CountK floats
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlpha32xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213ps zmm4,zmm31,ZMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213ps zmm6,zmm31,ZMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213ps zmm8,zmm31,ZMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213ps zmm10,zmm31,ZMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213ps zmm12,zmm31,ZMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213ps zmm14,zmm31,ZMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm16,zmm31,ZMMWORD PTR [r13]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm18,zmm31,ZMMWORD PTR [r13+rax]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm20,zmm31,ZMMWORD PTR [r13+rax*2]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm22,zmm31,ZMMWORD PTR [r14]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm24,zmm31,ZMMWORD PTR [r14+rax]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm26,zmm31,ZMMWORD PTR [r14+rax*2]"
        jmp     .LStore32xNBlock\@

.LMultiplyAlpha32xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmulps zmm4,zmm4,zmm31"
        EmitIfCountGE \RowCount\(), 2, "vmulps zmm6,zmm6,zmm31"
        EmitIfCountGE \RowCount\(), 3, "vmulps zmm8,zmm8,zmm31"
        EmitIfCountGE \RowCount\(), 4, "vmulps zmm10,zmm10,zmm31"
        EmitIfCountGE \RowCount\(), 5, "vmulps zmm12,zmm12,zmm31"
        EmitIfCountGE \RowCount\(), 6, "vmulps zmm14,zmm14,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm16,zmm16,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm18,zmm18,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm20,zmm20,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm22,zmm22,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm24,zmm24,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm26,zmm26,zmm31"

.LStore32xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovups ZMMWORD PTR [rdx],zmm4"
        EmitIfCountGE \RowCount\(), 2, "vmovups ZMMWORD PTR [rdx+rax],zmm6"
        EmitIfCountGE \RowCount\(), 3, "vmovups ZMMWORD PTR [rdx+rax*2],zmm8"
        EmitIfCountGE \RowCount\(), 4, "vmovups ZMMWORD PTR [rbx],zmm10"
        EmitIfCountGE \RowCount\(), 5, "vmovups ZMMWORD PTR [rbx+rax],zmm12"
        EmitIfCountGE \RowCount\(), 6, "vmovups ZMMWORD PTR [rbx+rax*2],zmm14"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r13],zmm16"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r13+rax],zmm18"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r13+rax*2],zmm20"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r14],zmm22"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r14+rax],zmm24"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r14+rax*2],zmm26"
        add     rdx,16*4                    # advance matrix C by 16 columns
.if \RowCount\() > 3
        add     rbx,16*4                    # advance matrix C plus 3 rows by 16 columns
.if \RowCount\() == 12
        add     r13,16*4                    # advance matrix C plus 6 rows by 16 columns
        add     r14,16*4                    # advance matrix C plus 9 rows by 16 columns
.endif
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
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlpha16xNBlockWithMask\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213ps zmm5{k1},zmm31,ZMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213ps zmm7{k1},zmm31,ZMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213ps zmm9{k1},zmm31,ZMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213ps zmm11{k1},zmm31,ZMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213ps zmm13{k1},zmm31,ZMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213ps zmm15{k1},zmm31,ZMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm17{k1},zmm31,ZMMWORD PTR [r13]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm19{k1},zmm31,ZMMWORD PTR [r13+rax]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm21{k1},zmm31,ZMMWORD PTR [r13+rax*2]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm23{k1},zmm31,ZMMWORD PTR [r14]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm25{k1},zmm31,ZMMWORD PTR [r14+rax]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213ps zmm27{k1},zmm31,ZMMWORD PTR [r14+rax*2]"
        jmp     .LStore16xNBlockWithMask\@

.LMultiplyAlpha16xNBlockWithMask\@:
        EmitIfCountGE \RowCount\(), 1, "vmulps zmm5,zmm5,zmm31"
        EmitIfCountGE \RowCount\(), 2, "vmulps zmm7,zmm7,zmm31"
        EmitIfCountGE \RowCount\(), 3, "vmulps zmm9,zmm9,zmm31"
        EmitIfCountGE \RowCount\(), 4, "vmulps zmm11,zmm11,zmm31"
        EmitIfCountGE \RowCount\(), 5, "vmulps zmm13,zmm13,zmm31"
        EmitIfCountGE \RowCount\(), 6, "vmulps zmm15,zmm15,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm17,zmm17,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm19,zmm19,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm21,zmm21,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm23,zmm23,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm25,zmm25,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulps zmm27,zmm27,zmm31"

.LStore16xNBlockWithMask\@:
        EmitIfCountGE \RowCount\(), 1, "vmovups ZMMWORD PTR [rdx]{k1},zmm5"
        EmitIfCountGE \RowCount\(), 2, "vmovups ZMMWORD PTR [rdx+rax]{k1},zmm7"
        EmitIfCountGE \RowCount\(), 3, "vmovups ZMMWORD PTR [rdx+rax*2]{k1},zmm9"
        EmitIfCountGE \RowCount\(), 4, "vmovups ZMMWORD PTR [rbx]{k1},zmm11"
        EmitIfCountGE \RowCount\(), 5, "vmovups ZMMWORD PTR [rbx+rax]{k1},zmm13"
        EmitIfCountGE \RowCount\(), 6, "vmovups ZMMWORD PTR [rbx+rax*2]{k1},zmm15"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r13]{k1},zmm17"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r13+rax]{k1},zmm19"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r13+rax*2]{k1},zmm21"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r14]{k1},zmm23"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r14+rax]{k1},zmm25"
        EmitIfCountGE \RowCount\(), 12, "vmovups ZMMWORD PTR [r14+rax*2]{k1},zmm27"
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,16
        ja      .LProcessNextColumnLoop32xN\@
        test    r9,r9
        jz      .LExitKernel

.LProcessRemainingCountN\@:
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm17,zmm5"
                                            # clear upper block accumulators
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm19,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm21,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm23,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm25,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovaps zmm27,zmm5"
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy16, \RowCount\()
        jmp     .LOutput16xNBlock\@

        .endm

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (rdi) - Supplies the address of matrix A.

    B (rsi) - Supplies the address of matrix B. The matrix data has been packed
        using MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C (rdx) - Supplies the address of matrix C.

    CountK (rcx) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountM (r8) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (r9) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    Alpha (xmm0) - Supplies the scalar alpha multiplier (see SGEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        .globl  C_UNDERSCORE(MlasGemmFloatKernelAvx512F)
C_UNDERSCORE(MlasGemmFloatKernelAvx512F):

        push    rbp
        push    rbx
        push    r12
        push    r13
        push    r14
        push    r15
        mov     r11,rdi
        mov     r10,.LSgemmKernelFrame_lda[rsp]
        shl     r10,2                       # convert lda to bytes
        mov     rax,.LSgemmKernelFrame_ldc[rsp]
        shl     rax,2                       # convert ldc to bytes
        mov     r12,rcx
        shl     r12,6                       # compute 16*CountK*sizeof(float)
        mov     ebp,-1
        kmovw   k1,ebp                      # update mask to write all columns
        movzx   r15,BYTE PTR .LSgemmKernelFrame_ZeroMode[rsp]
        vbroadcastss zmm31,xmm0
        vzeroall

//
// Process CountM rows of the matrices.
//

        cmp     r8,12
        jb      .LProcessCountMLessThan12
        mov     r8d,12                      # return 12 rows handled
        ProcessCountM 12

.LProcessCountMLessThan12:
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
        pop     r15
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
