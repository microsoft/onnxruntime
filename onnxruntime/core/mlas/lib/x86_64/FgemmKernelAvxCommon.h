/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelAvx.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses AVX instructions.

--*/

#include "asmmacro.h"
#include "SgemmKernelCommon.h"

        .intel_syntax noprefix

        .equ    .LSgemmKernelFrame_alpha, -8
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

    rbx - Supplies the address into the matrix A data plus 2 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm8-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxBy16 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.if \RowCount\() == 1
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vmulps  ymm4,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
        vaddps  ymm8,ymm8,ymm4
        vmulps  ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()+32]
        vaddps  ymm9,ymm9,ymm5
.else
        vmovaps ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        vmovaps ymm1,YMMWORD PTR [rsi+\VectorOffset\()+32]
        EmitIfCountGE \RowCount\(), 1, "vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm4,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 1, "vaddps ymm8,ymm8,ymm4"
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm5,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 1, "vaddps ymm9,ymm9,ymm5"
        EmitIfCountGE \RowCount\(), 2, "vbroadcastss ymm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm6,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 2, "vaddps ymm10,ymm10,ymm6"
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm7,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 2, "vaddps ymm11,ymm11,ymm7"
        EmitIfCountGE \RowCount\(), 3, "vbroadcastss ymm3,DWORD PTR [rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm4,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 3, "vaddps ymm12,ymm12,ymm4"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm5,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 3, "vaddps ymm13,ymm13,ymm5"
        EmitIfCountGE \RowCount\(), 4, "vbroadcastss ymm3,DWORD PTR [rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm6,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 4, "vaddps ymm14,ymm14,ymm6"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm7,ymm3,ymm1"
        EmitIfCountGE \RowCount\(), 4, "vaddps ymm15,ymm15,ymm7"
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

    rbx - Supplies the address into the matrix A data plus 2 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm8-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxBy8 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.if \RowCount\() == 1
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vmulps  ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
        vaddps  ymm9,ymm9,ymm5
.else
        vmovaps ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        EmitIfCountGE \RowCount\(), 1, "vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm5,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 1, "vaddps ymm9,ymm9,ymm5"
        EmitIfCountGE \RowCount\(), 2, "vbroadcastss ymm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm7,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 2, "vaddps ymm11,ymm11,ymm7"
        EmitIfCountGE \RowCount\(), 3, "vbroadcastss ymm3,DWORD PTR [rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm5,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 3, "vaddps ymm13,ymm13,ymm5"
        EmitIfCountGE \RowCount\(), 4, "vbroadcastss ymm3,DWORD PTR [rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm7,ymm3,ymm0"
        EmitIfCountGE \RowCount\(), 4, "vaddps ymm15,ymm15,ymm7"
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

    rbx - Supplies the address into the matrix A data plus 2 rows.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxLoop ComputeBlock, RowCount

.if \RowCount\() > 2
        lea     rbx,[rdi+r10*2]             # compute matrix A plus 2 rows
.endif
        ComputeBlockLoop \ComputeBlock\(), \RowCount\(), \RowCount\() > 2
.if \RowCount\() > 2
        lea     rbx,[rdx+rax*2]             # compute matrix C plus 2 rows
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
        EmitIfCountGE \RowCount\(), 1, "vxorps xmm8,xmm8,xmm8"
        EmitIfCountGE \RowCount\(), 1, "vxorps xmm9,xmm9,xmm9"
        EmitIfCountGE \RowCount\(), 2, "vxorps xmm10,xmm10,xmm10"
        EmitIfCountGE \RowCount\(), 2, "vxorps xmm11,xmm11,xmm11"
        EmitIfCountGE \RowCount\(), 3, "vxorps xmm12,xmm12,xmm12"
        EmitIfCountGE \RowCount\(), 3, "vxorps xmm13,xmm13,xmm13"
        EmitIfCountGE \RowCount\(), 4, "vxorps xmm14,xmm14,xmm14"
        EmitIfCountGE \RowCount\(), 4, "vxorps xmm15,xmm15,xmm15"
        ComputeBlockAvxLoop ComputeBlockAvxBy16, \RowCount\()
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm8,ymm8,ymm2"
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm9,ymm9,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm10,ymm10,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm11,ymm11,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm12,ymm12,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm13,ymm13,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm14,ymm14,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm15,ymm15,ymm2"
        sub     r9,16
        jb      .LOutputMasked16xNBlock\@
        test    r15b,r15b                   # ZeroMode?
        jnz     .LStore16xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vaddps ymm8,ymm8,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 1, "vaddps ymm9,ymm9,YMMWORD PTR [rdx+32]"
        EmitIfCountGE \RowCount\(), 2, "vaddps ymm10,ymm10,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 2, "vaddps ymm11,ymm11,YMMWORD PTR [rdx+rax+32]"
        EmitIfCountGE \RowCount\(), 3, "vaddps ymm12,ymm12,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 3, "vaddps ymm13,ymm13,YMMWORD PTR [rbx+32]"
        EmitIfCountGE \RowCount\(), 4, "vaddps ymm14,ymm14,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 4, "vaddps ymm15,ymm15,YMMWORD PTR [rbx+rax+32]"

.LStore16xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovups YMMWORD PTR [rdx],ymm8"
        EmitIfCountGE \RowCount\(), 1, "vmovups YMMWORD PTR [rdx+32],ymm9"
        EmitIfCountGE \RowCount\(), 2, "vmovups YMMWORD PTR [rdx+rax],ymm10"
        EmitIfCountGE \RowCount\(), 2, "vmovups YMMWORD PTR [rdx+rax+32],ymm11"
        EmitIfCountGE \RowCount\(), 3, "vmovups YMMWORD PTR [rbx],ymm12"
        EmitIfCountGE \RowCount\(), 3, "vmovups YMMWORD PTR [rbx+32],ymm13"
        EmitIfCountGE \RowCount\(), 4, "vmovups YMMWORD PTR [rbx+rax],ymm14"
        EmitIfCountGE \RowCount\(), 4, "vmovups YMMWORD PTR [rbx+rax+32],ymm15"
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        cmp     r9,8
        ja      .LProcessNextColumnLoop16xN\@
        test    r9,r9
        jz      .LExitKernel

.LProcessRemainingCountN\@:
        EmitIfCountGE \RowCount\(), 1, "vxorps xmm9,xmm9,xmm9"
        EmitIfCountGE \RowCount\(), 2, "vxorps xmm11,xmm11,xmm11"
        EmitIfCountGE \RowCount\(), 3, "vxorps xmm13,xmm13,xmm13"
        EmitIfCountGE \RowCount\(), 4, "vxorps xmm15,xmm15,xmm15"
        ComputeBlockAvxLoop ComputeBlockAvxBy8, \RowCount\()
        EmitIfCountGE \RowCount\(), 1, "vmulps ymm9,ymm9,ymm2"
        EmitIfCountGE \RowCount\(), 2, "vmulps ymm11,ymm11,ymm2"
        EmitIfCountGE \RowCount\(), 3, "vmulps ymm13,ymm13,ymm2"
        EmitIfCountGE \RowCount\(), 4, "vmulps ymm15,ymm15,ymm2"
        cmp     r9,8
        jb      .LOutputMasked8xNBlock\@
        test    r15b,r15b                   # ZeroMode?
        jnz     .LStore8xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vaddps ymm9,ymm9,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vaddps ymm11,ymm11,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vaddps ymm13,ymm13,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "vaddps ymm15,ymm15,YMMWORD PTR [rbx+rax]"

.LStore8xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovups YMMWORD PTR [rdx],ymm9"
        EmitIfCountGE \RowCount\(), 2, "vmovups YMMWORD PTR [rdx+rax],ymm11"
        EmitIfCountGE \RowCount\(), 3, "vmovups YMMWORD PTR [rbx],ymm13"
        EmitIfCountGE \RowCount\(), 4, "vmovups YMMWORD PTR [rbx+rax],ymm15"
        jmp     .LExitKernel

.LOutputMasked16xNBlock\@:
        test    r15b,r15b                   # ZeroMode?
        jnz     .LStoreMasked16xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vaddps ymm8,ymm8,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vaddps ymm10,ymm10,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vaddps ymm12,ymm12,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "vaddps ymm14,ymm14,YMMWORD PTR [rbx+rax]"

.LStoreMasked16xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovups YMMWORD PTR [rdx],ymm8"
        EmitIfCountGE \RowCount\(), 2, "vmovups YMMWORD PTR [rdx+rax],ymm10"
        EmitIfCountGE \RowCount\(), 3, "vmovups YMMWORD PTR [rbx],ymm12"
        EmitIfCountGE \RowCount\(), 4, "vmovups YMMWORD PTR [rbx+rax],ymm14"
        add     rdx,8*4                     # advance matrix C by 8 columns
.if \RowCount\() > 2
        add     rbx,8*4                     # advance matrix C plus 2 rows by 8 columns
.endif
        add     r9,8                        # correct for over-subtract above

.LOutputMasked8xNBlock\@:
        vmovd   xmm0,r9d
        vshufps xmm0,xmm0,xmm0,0
        vpcmpgtd xmm1,xmm0,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip+16]
        vpcmpgtd xmm0,xmm0,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        vinsertf128 ymm0,ymm0,xmm1,1
        test    r15b,r15b                   # ZeroMode?
        jnz     .LStoreMasked8xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vmaskmovps ymm8,ymm0,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vmaskmovps ymm10,ymm0,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vmaskmovps ymm12,ymm0,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "vmaskmovps ymm14,ymm0,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 1, "vaddps ymm9,ymm9,ymm8"
        EmitIfCountGE \RowCount\(), 2, "vaddps ymm11,ymm11,ymm10"
        EmitIfCountGE \RowCount\(), 3, "vaddps ymm13,ymm13,ymm12"
        EmitIfCountGE \RowCount\(), 4, "vaddps ymm15,ymm15,ymm14"

.LStoreMasked8xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmaskmovps YMMWORD PTR [rdx],ymm0,ymm9"
        EmitIfCountGE \RowCount\(), 2, "vmaskmovps YMMWORD PTR [rdx+rax],ymm0,ymm11"
        EmitIfCountGE \RowCount\(), 3, "vmaskmovps YMMWORD PTR [rbx],ymm0,ymm13"
        EmitIfCountGE \RowCount\(), 4, "vmaskmovps YMMWORD PTR [rbx+rax],ymm0,ymm15"
.ifb \Fallthrough\()
        jmp     .LExitKernel
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

        .globl  C_UNDERSCORE(MlasGemmFloatKernelAvx)
C_UNDERSCORE(MlasGemmFloatKernelAvx):

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
        vbroadcastss ymm2,DWORD PTR .LSgemmKernelFrame_alpha[rsp]

//
// Process 4 rows of the matrices.
//

        cmp     r8,4
        jb      .LProcessCountMLessThan4
        mov     r8d,4                      # return 4 rows handled
        ProcessCountM 4, Fallthrough

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        vzeroupper
        mov     eax,r8d
        pop     r15
        pop     rbx
        pop     rbp
        ret

//
// Process 2 rows of the matrices.
//

.LProcessCountMLessThan4:
        cmp     r8,2
        jb      .LProcessCountMLessThan2
        mov     r8d,2                       # return 2 rows handled
        ProcessCountM 2

//
// Process 1 row of the matrices.
//

.LProcessCountMLessThan2:
        ProcessCountM 1

        .end
