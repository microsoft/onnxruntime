/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelFma3.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses AVX fused multiply/add instructions.

--*/

#include "asmmacro.h"
#include "SgemmKernelCommon.h"

        .intel_syntax noprefix

        .equ    .LSgemmKernelFrame_alpha, -8
        .equ    .LSgemmKernelFrame_mask, -4
        .equ    .LSgemmKernelFrame_SavedR15, 0
        .equ    .LSgemmKernelFrame_SavedRbx, 8
        .equ    .LSgemmKernelFrame_SavedRbp, 16
        .equ    .LSgemmKernelFrame_ReturnAddress, 24
        .equ    .LSgemmKernelFrame_lda, 32
        .equ    .LSgemmKernelFrame_ldc, 40
        .equ    .LSgemmKernelFrame_ZeroMode, 48

        .text

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

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockFma3By16 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.ifnb \PrefetchOffset\()
        prefetcht0 [rsi+\VectorOffset\()+\PrefetchOffset\()]
.endif
.if \RowCount\() == 1
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vfmadd231ps ymm4,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
        vfmadd231ps ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()+32]
.else
        vmovaps ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        vmovaps ymm1,YMMWORD PTR [rsi+\VectorOffset\()+32]
        EmitIfCountGE \RowCount\(), 1, "vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "vfmadd231ps ymm4,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 1, "vfmadd231ps ymm5,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 2, "vbroadcastss ymm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231ps ymm6,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231ps ymm7,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 3, "vbroadcastss ymm3,DWORD PTR [rdi+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231ps ymm8,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231ps ymm9,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 4, "vbroadcastss ymm3,DWORD PTR [rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231ps ymm10,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231ps ymm11,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 5, "vbroadcastss ymm3,DWORD PTR [rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231ps ymm12,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231ps ymm13,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 6, "vbroadcastss ymm3,DWORD PTR [rbx+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231ps ymm14,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231ps ymm15,ymm3,ymm1"
.endif

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for a 8xN block of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

    PrefetchOffset - Optionally supplies the byte offset from matrix B to
        prefetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockFma3By8 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.ifnb \PrefetchOffset\()
        prefetcht0 [rsi+\VectorOffset\()+\PrefetchOffset\()]
.endif
.if \RowCount\() == 1
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vfmadd231ps ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
.else
        vmovaps ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        EmitIfCountGE \RowCount\(), 1, "vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "vfmadd231ps ymm5,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 2, "vbroadcastss ymm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231ps ymm7,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 3, "vbroadcastss ymm3,DWORD PTR [rdi+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231ps ymm9,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 4, "vbroadcastss ymm3,DWORD PTR [rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231ps ymm11,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 5, "vbroadcastss ymm3,DWORD PTR [rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231ps ymm13,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 6, "vbroadcastss ymm3,DWORD PTR [rbx+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231ps ymm15,ymm3,ymm0"
.endif

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

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockFma3Loop ComputeBlock, RowCount

.if \RowCount\() > 3
        lea     rbx,[r10*2+r10]
        add     rbx,rdi                     # compute matrix A plus 3 rows
.endif
        ComputeBlockLoop \ComputeBlock\(), \RowCount\(), \RowCount\() > 3
        vbroadcastss ymm2,DWORD PTR [rsp+.LSgemmKernelFrame_alpha]
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
        the ExitKernelAndZeroUpper label.

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

        .macro ProcessCountM RowCount, Fallthrough

        cmp     r9,8
        jbe     .LProcessRemainingCountN\@

.LProcessNextColumnLoop16xN\@:
        ComputeBlockFma3Loop ComputeBlockFma3By16, \RowCount\()
        EmitIfCountGE \RowCount\(), 1, "prefetcht0 [rdx+64]"
        EmitIfCountGE \RowCount\(), 2, "prefetcht0 [rdx+rax+64]"
        EmitIfCountGE \RowCount\(), 3, "prefetcht0 [rdx+rax*2+64]"
        EmitIfCountGE \RowCount\(), 4, "prefetcht0 [rbx+64]"
        EmitIfCountGE \RowCount\(), 5, "prefetcht0 [rbx+rax+64]"
        EmitIfCountGE \RowCount\(), 6, "prefetcht0 [rbx+rax*2+64]"
        sub     r9,16
        jb      .LOutputMasked16xNBlock\@
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlpha16xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213ps ymm4,ymm2,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 1, "vfmadd213ps ymm5,ymm2,YMMWORD PTR [rdx+32]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213ps ymm6,ymm2,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213ps ymm7,ymm2,YMMWORD PTR [rdx+rax+32]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213ps ymm8,ymm2,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213ps ymm9,ymm2,YMMWORD PTR [rdx+rax*2+32]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213ps ymm10,ymm2,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213ps ymm11,ymm2,YMMWORD PTR [rbx+32]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213ps ymm12,ymm2,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213ps ymm13,ymm2,YMMWORD PTR [rbx+rax+32]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213ps ymm14,ymm2,YMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213ps ymm15,ymm2,YMMWORD PTR [rbx+rax*2+32]"
        jmp     .LStore16xNBlock\@

.LMultiplyAlpha16xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm4,ymm4,ymm2"
                                            # multiply by alpha
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm5,ymm5,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm6,ymm6,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm7,ymm7,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm8,ymm8,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm9,ymm9,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm10,ymm10,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm11,ymm11,ymm2"
        EmitIfCountGE \RowCount\(), 5, "vmulps ymm12,ymm12,ymm2"
        EmitIfCountGE \RowCount\(), 5, "vmulps ymm13,ymm13,ymm2"
        EmitIfCountGE \RowCount\(), 6, "vmulps ymm14,ymm14,ymm2"
        EmitIfCountGE \RowCount\(), 6, "vmulps ymm15,ymm15,ymm2"

.LStore16xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovups YMMWORD PTR [rdx],ymm4"
        EmitIfCountGE \RowCount\(), 1, "vmovups YMMWORD PTR [rdx+32],ymm5"
        EmitIfCountGE \RowCount\(), 2, "vmovups YMMWORD PTR [rdx+rax],ymm6"
        EmitIfCountGE \RowCount\(), 2, "vmovups YMMWORD PTR [rdx+rax+32],ymm7"
        EmitIfCountGE \RowCount\(), 3, "vmovups YMMWORD PTR [rdx+rax*2],ymm8"
        EmitIfCountGE \RowCount\(), 3, "vmovups YMMWORD PTR [rdx+rax*2+32],ymm9"
        EmitIfCountGE \RowCount\(), 4, "vmovups YMMWORD PTR [rbx],ymm10"
        EmitIfCountGE \RowCount\(), 4, "vmovups YMMWORD PTR [rbx+32],ymm11"
        EmitIfCountGE \RowCount\(), 5, "vmovups YMMWORD PTR [rbx+rax],ymm12"
        EmitIfCountGE \RowCount\(), 5, "vmovups YMMWORD PTR [rbx+rax+32],ymm13"
        EmitIfCountGE \RowCount\(), 6, "vmovups YMMWORD PTR [rbx+rax*2],ymm14"
        EmitIfCountGE \RowCount\(), 6, "vmovups YMMWORD PTR [rbx+rax*2+32],ymm15"
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,8
        ja      .LProcessNextColumnLoop16xN\@
        test    r9,r9
        jz      .LExitKernel

.LProcessRemainingCountN\@:
        ComputeBlockFma3Loop ComputeBlockFma3By8, \RowCount\()
        cmp     r9,8
        jb      .LOutputMasked8xNBlock\@
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlpha8xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213ps ymm5,ymm2,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213ps ymm7,ymm2,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213ps ymm9,ymm2,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213ps ymm11,ymm2,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213ps ymm13,ymm2,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213ps ymm15,ymm2,YMMWORD PTR [rbx+rax*2]"
        jmp     .LStore8xNBlock\@

.LMultiplyAlpha8xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm5,ymm5,ymm2"
                                            # multiply by alpha
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm7,ymm7,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm9,ymm9,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm11,ymm11,ymm2"
        EmitIfCountGE \RowCount\(), 5, "vmulps ymm13,ymm13,ymm2"
        EmitIfCountGE \RowCount\(), 6, "vmulps ymm15,ymm15,ymm2"

.LStore8xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovups YMMWORD PTR [rdx],ymm5"
        EmitIfCountGE \RowCount\(), 2, "vmovups YMMWORD PTR [rdx+rax],ymm7"
        EmitIfCountGE \RowCount\(), 3, "vmovups YMMWORD PTR [rdx+rax*2],ymm9"
        EmitIfCountGE \RowCount\(), 4, "vmovups YMMWORD PTR [rbx],ymm11"
        EmitIfCountGE \RowCount\(), 5, "vmovups YMMWORD PTR [rbx+rax],ymm13"
        EmitIfCountGE \RowCount\(), 6, "vmovups YMMWORD PTR [rbx+rax*2],ymm15"
        jmp     .LExitKernelAndZeroUpper

.LOutputMasked16xNBlock\@:
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlphaMasked16xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213ps ymm4,ymm2,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213ps ymm6,ymm2,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213ps ymm8,ymm2,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213ps ymm10,ymm2,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213ps ymm12,ymm2,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213ps ymm14,ymm2,YMMWORD PTR [rbx+rax*2]"
        jmp     .LStoreMasked16xNBlock\@

.LMultiplyAlphaMasked16xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm4,ymm4,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm6,ymm6,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm8,ymm8,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm10,ymm10,ymm2"
        EmitIfCountGE \RowCount\(), 5, "vmulps ymm12,ymm12,ymm2"
        EmitIfCountGE \RowCount\(), 6, "vmulps ymm14,ymm14,ymm2"

.LStoreMasked16xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovups YMMWORD PTR [rdx],ymm4"
        EmitIfCountGE \RowCount\(), 2, "vmovups YMMWORD PTR [rdx+rax],ymm6"
        EmitIfCountGE \RowCount\(), 3, "vmovups YMMWORD PTR [rdx+rax*2],ymm8"
        EmitIfCountGE \RowCount\(), 4, "vmovups YMMWORD PTR [rbx],ymm10"
        EmitIfCountGE \RowCount\(), 5, "vmovups YMMWORD PTR [rbx+rax],ymm12"
        EmitIfCountGE \RowCount\(), 6, "vmovups YMMWORD PTR [rbx+rax*2],ymm14"
        add     rdx,8*4                     # advance matrix C by 8 columns
.if \RowCount\() > 3
        add     rbx,8*4                     # advance matrix C plus 3 rows by 8 columns
.endif
        add     r9,8                        # correct for over-subtract above

.LOutputMasked8xNBlock\@:
        mov     DWORD PTR [rsp+.LSgemmKernelFrame_mask],r9d
        vbroadcastss ymm0,DWORD PTR [rsp+.LSgemmKernelFrame_mask]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlphaMasked8xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vmaskmovps ymm4,ymm0,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vmaskmovps ymm6,ymm0,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vmaskmovps ymm8,ymm0,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vmaskmovps ymm10,ymm0,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vmaskmovps ymm12,ymm0,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vmaskmovps ymm14,ymm0,YMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 1, "vfmadd213ps ymm5,ymm2,ymm4"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213ps ymm7,ymm2,ymm6"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213ps ymm9,ymm2,ymm8"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213ps ymm11,ymm2,ymm10"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213ps ymm13,ymm2,ymm12"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213ps ymm15,ymm2,ymm14"
        jmp     .LStoreMasked8xNBlock\@

.LMultiplyAlphaMasked8xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm5,ymm5,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm7,ymm7,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm9,ymm9,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm11,ymm11,ymm2"
        EmitIfCountGE \RowCount\(), 5, "vmulps ymm13,ymm13,ymm2"
        EmitIfCountGE \RowCount\(), 6, "vmulps ymm15,ymm15,ymm2"

.LStoreMasked8xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmaskmovps YMMWORD PTR [rdx],ymm0,ymm5"
        EmitIfCountGE \RowCount\(), 2, "vmaskmovps YMMWORD PTR [rdx+rax],ymm0,ymm7"
        EmitIfCountGE \RowCount\(), 3, "vmaskmovps YMMWORD PTR [rdx+rax*2],ymm0,ymm9"
        EmitIfCountGE \RowCount\(), 4, "vmaskmovps YMMWORD PTR [rbx],ymm0,ymm11"
        EmitIfCountGE \RowCount\(), 5, "vmaskmovps YMMWORD PTR [rbx+rax],ymm0,ymm13"
        EmitIfCountGE \RowCount\(), 6, "vmaskmovps YMMWORD PTR [rbx+rax*2],ymm0,ymm15"
.ifb \Fallthrough\()
        jmp     .LExitKernelAndZeroUpper
.endif

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

        .globl  C_UNDERSCORE(MlasGemmFloatKernelFma3)
C_UNDERSCORE(MlasGemmFloatKernelFma3):

        push    rbp
        push    rbx
        push    r15
        mov     r11,rdi
        mov     r10,.LSgemmKernelFrame_lda[rsp]
        shl     r10,2                       # convert lda to bytes
        mov     rax,.LSgemmKernelFrame_ldc[rsp]
        shl     rax,2                       # convert ldc to bytes
        movzx   r15,BYTE PTR .LSgemmKernelFrame_ZeroMode[rsp]
        vmovss  DWORD PTR .LSgemmKernelFrame_alpha[rsp],xmm0
        vzeroall

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
        ProcessCountM 6, Fallthrough

//
// Restore non-volatile registers and return.
//

.LExitKernelAndZeroUpper:
        vzeroupper

.LExitKernel:
        mov     eax,r8d
        pop     r15
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
