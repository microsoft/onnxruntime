/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelSse2.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses SSE2 instructions.

--*/

        .intel_syntax noprefix

        .equ    SgemmKernelFrame_alpha, -8
        .equ    SgemmKernelFrame_SavedRbx, 0
        .equ    SgemmKernelFrame_SavedRbp, 8
        .equ    SgemmKernelFrame_ReturnAddress, 16
        .equ    SgemmKernelFrame_lda, 24
        .equ    SgemmKernelFrame_ldc, 32

        .text

/*++

Macro Description:

    This macro multiplies and accumulates for a 16xN block (where N is 1,2)
    of the output matrix.

Arguments:

    Count - Supplies the number of rows to access from matrix A.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    Shuffle - Supplies the shuffle mask to extract the element from matrix A.

Implicit Arguments:

    rsi - Supplies the address into the matrix B data.

    xmm0-xmm1 - Supplies up to four elements loaded from matrix A and matrix A
        plus one row.

    xmm8-xmm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockSseBy16 Count, VectorOffset, Shuffle

        movaps  xmm4,XMMWORD PTR [rsi+\VectorOffset\()]
        movaps  xmm5,XMMWORD PTR [rsi+\VectorOffset\()+16]
        pshufd  xmm2,xmm0,\Shuffle\()
.if \Count\() == 2
        pshufd  xmm3,xmm1,\Shuffle\()
        movaps  xmm6,xmm4
        movaps  xmm7,xmm5
.endif
        mulps   xmm4,xmm2
        mulps   xmm5,xmm2
        addps   xmm8,xmm4
        addps   xmm9,xmm5
.if \Count\() == 2
        mulps   xmm6,xmm3
        mulps   xmm7,xmm3
        addps   xmm12,xmm6
        addps   xmm13,xmm7
.endif
        movaps  xmm4,XMMWORD PTR [rsi+\VectorOffset\()+32]
        movaps  xmm5,XMMWORD PTR [rsi+\VectorOffset\()+48]
.if \Count\() == 2
        movaps  xmm6,xmm4
        movaps  xmm7,xmm5
.endif
        mulps   xmm4,xmm2
        mulps   xmm5,xmm2
        addps   xmm10,xmm4
        addps   xmm11,xmm5
.if \Count\() == 2
        mulps   xmm6,xmm3
        mulps   xmm7,xmm3
        addps   xmm14,xmm6
        addps   xmm15,xmm7
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

    Alpha (xmm0) - Supplies the scaler multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/

        .macro  SgemmKernelSseFunction Mode

        .globl  MlasSgemmKernel\Mode\()Sse
MlasSgemmKernel\Mode\()Sse:

        push    rbp
        push    rbx
        mov     r11,rdi
        mov     r10,[rsp+SgemmKernelFrame_lda]
        shl     r10,2                       # convert lda to bytes
        mov     rax,[rsp+SgemmKernelFrame_ldc]
        shl     rax,2                       # convert ldc to bytes
        movss   DWORD PTR [rsp+SgemmKernelFrame_alpha],xmm0

//
// Process 2 rows of the matrices.
//

        cmp     r8,2
        jb      .L\Mode\().ProcessCountMLessThan2
        mov     r8d,2                       # return 2 rows handled

.L\Mode\().ProcessNextColumnLoop16x2:
        xorps   xmm8,xmm8                   # clear block accumulators
        xorps   xmm9,xmm9
        xorps   xmm10,xmm10
        xorps   xmm11,xmm11
        xorps   xmm12,xmm12
        xorps   xmm13,xmm13
        xorps   xmm14,xmm14
        xorps   xmm15,xmm15
        mov     rbp,rcx                     # reload CountK
        sub     rbp,4
        jb      .L\Mode\().ProcessRemaining16x2Blocks

.L\Mode\().Compute16x2BlockBy4Loop:
        movups  xmm0,XMMWORD PTR [rdi]
        movups  xmm1,XMMWORD PTR [rdi+r10]
        ComputeBlockSseBy16 2, 0, 0x00
        ComputeBlockSseBy16 2, 16*4, 0x55
        sub     rsi,-32*4                   # advance matrix B by 32 columns
        ComputeBlockSseBy16 2, 0, 0xAA
        ComputeBlockSseBy16 2, 16*4, 0xFF
        sub     rsi,-32*4                   # advance matrix B by 32 columns
        add     rdi,4*4                     # advance matrix A by 4 columns
        sub     rbp,4
        jae     .L\Mode\().Compute16x2BlockBy4Loop

.L\Mode\().ProcessRemaining16x2Blocks:
        add     rbp,4                       # correct for over-subtract above
        jz      .L\Mode\().Output16x2Block

.L\Mode\().Compute16x2BlockBy1Loop:
        movss   xmm0,DWORD PTR [rdi]
        movss   xmm1,DWORD PTR [rdi+r10]
        ComputeBlockSseBy16 2, 0, 0x00
        add     rsi,16*4                    # advance matrix B by 16 columns
        add     rdi,4                       # advance matrix A by 1 column
        dec     rbp
        jne     .L\Mode\().Compute16x2BlockBy1Loop

.L\Mode\().Output16x2Block:
        movss   xmm2,DWORD PTR [rsp+SgemmKernelFrame_alpha]
        lea     rdi,[rdx+rax]               # compute matrix C plus 1 row
        shufps  xmm2,xmm2,0
        mulps   xmm8,xmm2                   # multiply by alpha
        mulps   xmm9,xmm2
        mulps   xmm10,xmm2
        mulps   xmm11,xmm2
        mulps   xmm12,xmm2
        mulps   xmm13,xmm2
        mulps   xmm14,xmm2
        mulps   xmm15,xmm2
        sub     r9,16
        jb      .L\Mode\().OutputPartial16x2Block
.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [rdx]
        movups  xmm1,XMMWORD PTR [rdx+16]
        movups  xmm2,XMMWORD PTR [rdx+32]
        movups  xmm3,XMMWORD PTR [rdx+48]
        movups  xmm4,XMMWORD PTR [rdi]
        movups  xmm5,XMMWORD PTR [rdi+16]
        movups  xmm6,XMMWORD PTR [rdi+32]
        movups  xmm7,XMMWORD PTR [rdi+48]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
        addps   xmm10,xmm2
        addps   xmm11,xmm3
        addps   xmm12,xmm4
        addps   xmm13,xmm5
        addps   xmm14,xmm6
        addps   xmm15,xmm7
.endif
        movups  XMMWORD PTR [rdx],xmm8
        movups  XMMWORD PTR [rdx+16],xmm9
        movups  XMMWORD PTR [rdx+32],xmm10
        movups  XMMWORD PTR [rdx+48],xmm11
        movups  XMMWORD PTR [rdi],xmm12
        movups  XMMWORD PTR [rdi+16],xmm13
        movups  XMMWORD PTR [rdi+32],xmm14
        movups  XMMWORD PTR [rdi+48],xmm15
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        test    r9,r9
        jnz     .L\Mode\().ProcessNextColumnLoop16x2

//
// Restore non-volatile registers and return.
//

.L\Mode\().ExitKernel:
        mov     eax,r8d
        pop     rbx
        pop     rbp
        ret

//
// Output a partial 16x2 block to the matrix.
//

.L\Mode\().OutputPartial16x2Block:
        add     r9,16                       # correct for over-subtract above
        cmp     r9,4
        jb      .L\Mode\().OutputPartialLessThan4x2Block
        cmp     r9,8
        jb      .L\Mode\().OutputPartialLessThan8x2Block
        cmp     r9,12
        jb      .L\Mode\().OutputPartialLessThan12x2Block

.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [rdx]
        movups  xmm1,XMMWORD PTR [rdx+16]
        movups  xmm2,XMMWORD PTR [rdx+32]
        movups  xmm3,XMMWORD PTR [rdi]
        movups  xmm4,XMMWORD PTR [rdi+16]
        movups  xmm5,XMMWORD PTR [rdi+32]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
        addps   xmm10,xmm2
        addps   xmm12,xmm3
        addps   xmm13,xmm4
        addps   xmm14,xmm5
.endif
        movups  XMMWORD PTR [rdx],xmm8
        movups  XMMWORD PTR [rdx+16],xmm9
        movups  XMMWORD PTR [rdx+32],xmm10
        movups  XMMWORD PTR [rdi],xmm12
        movups  XMMWORD PTR [rdi+16],xmm13
        movups  XMMWORD PTR [rdi+32],xmm14
        and     r9d,3                       # check if remaining count is small
        jz      .L\Mode\().ExitKernel
        movaps  xmm8,xmm11                  # shift remaining elements down
        movaps  xmm12,xmm15
        add     rdx,12*4                    # advance matrix C by 12 columns
        add     rdi,12*4                    # advance matrix C plus 1 row by 12 columns
        jmp     .L\Mode\().OutputPartialLessThan4x2Block

.L\Mode\().OutputPartialLessThan12x2Block:
.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [rdx]
        movups  xmm1,XMMWORD PTR [rdx+16]
        movups  xmm2,XMMWORD PTR [rdi]
        movups  xmm3,XMMWORD PTR [rdi+16]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
        addps   xmm12,xmm2
        addps   xmm13,xmm3
.endif
        movups  XMMWORD PTR [rdx],xmm8
        movups  XMMWORD PTR [rdx+16],xmm9
        movups  XMMWORD PTR [rdi],xmm12
        movups  XMMWORD PTR [rdi+16],xmm13
        and     r9d,3                       # check if remaining count is small
        jz      .L\Mode\().ExitKernel
        movaps  xmm8,xmm10                  # shift remaining elements down
        movaps  xmm12,xmm14
        add     rdx,8*4                     # advance matrix C by 8 columns
        add     rdi,8*4                     # advance matrix C plus 1 row by 8 columns
        jmp     .L\Mode\().OutputPartialLessThan4x2Block

.L\Mode\().OutputPartialLessThan8x2Block:
.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [rdx]
        movups  xmm1,XMMWORD PTR [rdi]
        addps   xmm8,xmm0
        addps   xmm12,xmm1
.endif
        movups  XMMWORD PTR [rdx],xmm8
        movups  XMMWORD PTR [rdi],xmm12
        and     r9d,3                       # check if remaining count is small
        jz      .L\Mode\().ExitKernel
        movaps  xmm8,xmm9                   # shift remaining elements down
        movaps  xmm12,xmm13
        add     rdx,4*4                     # advance matrix C by 4 columns
        add     rdi,4*4                     # advance matrix C plus 1 row by 4 columns

.L\Mode\().OutputPartialLessThan4x2Block:
        cmp     r9,2
        jb      .L\Mode\().OutputPartial1x2Block
.ifeqs "\Mode\()","Add"
        movsd   xmm0,QWORD PTR [rdx]
        movsd   xmm1,QWORD PTR [rdi]
        addps   xmm8,xmm0
        addps   xmm12,xmm1
.endif
        movsd   QWORD PTR [rdx],xmm8
        movsd   QWORD PTR [rdi],xmm12
        and     r9d,1                       # check if remaining count is odd
        jz      .L\Mode\().ExitKernel
        movhlps xmm8,xmm8                   # shift third element down
        movhlps xmm12,xmm12
        add     rdx,2*4                     # advance matrix C by 2 columns
        add     rdi,2*4                     # advance matrix C plus 1 row by 2 columns

.L\Mode\().OutputPartial1x2Block:
.ifeqs "\Mode\()","Add"
        addss   xmm8,DWORD PTR [rdx]
        addss   xmm12,DWORD PTR [rdi]
.endif
        movss   DWORD PTR [rdx],xmm8
        movss   DWORD PTR [rdi],xmm12
        jmp     .L\Mode\().ExitKernel

//
// Process 1 row of the matrices.
//

.L\Mode\().ProcessCountMLessThan2:
        mov     r8d,1                       # return 1 row handled

.L\Mode\().ProcessNextColumnLoop16x1:
        xorps   xmm8,xmm8                   # clear block accumulators
        xorps   xmm9,xmm9
        xorps   xmm10,xmm10
        xorps   xmm11,xmm11
        mov     rbp,rcx                     # reload CountK
        sub     rbp,4
        jb      .L\Mode\().ProcessRemaining16x1Blocks

.L\Mode\().Compute16x1BlockBy4Loop:
        movups  xmm0,XMMWORD PTR [rdi]
        ComputeBlockSseBy16 1, 0, 0x00
        ComputeBlockSseBy16 1, 16*4, 0x55
        sub     rsi,-32*4                   # advance matrix B by 32 columns
        ComputeBlockSseBy16 1, 0, 0xAA
        ComputeBlockSseBy16 1, 16*4, 0xFF
        sub     rsi,-32*4                   # advance matrix B by 32 columns
        add     rdi,4*4                     # advance matrix A by 4 columns
        sub     rbp,4
        jae     .L\Mode\().Compute16x1BlockBy4Loop

.L\Mode\().ProcessRemaining16x1Blocks:
        add     rbp,4                       # correct for over-subtract above
        jz      .L\Mode\().Output16x1Block

.L\Mode\().Compute16x1BlockBy1Loop:
        movss   xmm0,DWORD PTR [rdi]
        ComputeBlockSseBy16 1, 0, 0x00
        add     rsi,16*4                    # advance matrix B by 16 columns
        add     rdi,4                       # advance matrix A by 1 column
        dec     rbp
        jne     .L\Mode\().Compute16x1BlockBy1Loop

.L\Mode\().Output16x1Block:
        movss   xmm2,DWORD PTR [rsp+SgemmKernelFrame_alpha]
        shufps  xmm2,xmm2,0
        mulps   xmm8,xmm2                   # multiply by alpha
        mulps   xmm9,xmm2
        mulps   xmm10,xmm2
        mulps   xmm11,xmm2
        sub     r9,16
        jb      .L\Mode\().OutputPartial16x1Block
.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [rdx]
        movups  xmm1,XMMWORD PTR [rdx+16]
        movups  xmm2,XMMWORD PTR [rdx+32]
        movups  xmm3,XMMWORD PTR [rdx+48]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
        addps   xmm10,xmm2
        addps   xmm11,xmm3
.endif
        movups  XMMWORD PTR [rdx],xmm8
        movups  XMMWORD PTR [rdx+16],xmm9
        movups  XMMWORD PTR [rdx+32],xmm10
        movups  XMMWORD PTR [rdx+48],xmm11
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        test    r9,r9
        jnz     .L\Mode\().ProcessNextColumnLoop16x1
        jmp     .L\Mode\().ExitKernel

//
// Output a partial 16x1 block to the matrix.
//

.L\Mode\().OutputPartial16x1Block:
        add     r9,16                      # correct for over-subtract above
        cmp     r9,4
        jb      .L\Mode\().OutputPartialLessThan4x1Block
        cmp     r9,8
        jb      .L\Mode\().OutputPartialLessThan8x1Block
        cmp     r9,12
        jb      .L\Mode\().OutputPartialLessThan12x1Block

.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [rdx]
        movups  xmm1,XMMWORD PTR [rdx+16]
        movups  xmm2,XMMWORD PTR [rdx+32]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
        addps   xmm10,xmm2
.endif
        movups  XMMWORD PTR [rdx],xmm8
        movups  XMMWORD PTR [rdx+16],xmm9
        movups  XMMWORD PTR [rdx+32],xmm10
        and     r9d,3                       # check if remaining count is small
        jz      .L\Mode\().ExitKernel
        movaps  xmm8,xmm11                  # shift remaining elements down
        add     rdx,12*4                    # advance matrix C by 12 columns
        jmp     .L\Mode\().OutputPartialLessThan4x1Block

.L\Mode\().OutputPartialLessThan12x1Block:
.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [rdx]
        movups  xmm1,XMMWORD PTR [rdx+16]
        addps   xmm8,xmm0
        addps   xmm9,xmm1
.endif
        movups  XMMWORD PTR [rdx],xmm8
        movups  XMMWORD PTR [rdx+16],xmm9
        and     r9d,3                       # check if remaining count is small
        jz      .L\Mode\().ExitKernel
        movaps  xmm8,xmm10                  # shift remaining elements down
        add     rdx,8*4                     # advance matrix C by 8 columns
        jmp     .L\Mode\().OutputPartialLessThan4x1Block

.L\Mode\().OutputPartialLessThan8x1Block:
.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [rdx]
        addps   xmm8,xmm0
.endif
        movups  XMMWORD PTR [rdx],xmm8
        and     r9d,3                       # check if remaining count is small
        jz      .L\Mode\().ExitKernel
        movaps  xmm8,xmm9                   # shift remaining elements down
        add     rdx,4*4                     # advance matrix C by 4 columns

.L\Mode\().OutputPartialLessThan4x1Block:
        cmp     r9,2
        jb      .L\Mode\().OutputPartial1x1Block
.ifeqs "\Mode\()","Add"
        movsd   xmm0,QWORD PTR [rdx]
        addps   xmm8,xmm0
.endif
        movsd   QWORD PTR [rdx],xmm8
        and     r9d,1                       # check if remaining count is odd
        jz      .L\Mode\().ExitKernel
        movhlps xmm8,xmm8                   # shift third element down
        add     rdx,2*4                     # advance matrix C by 2 columns

.L\Mode\().OutputPartial1x1Block:
.ifeqs "\Mode\()","Add"
        addss   xmm8,DWORD PTR [rdx]
.endif
        movss   DWORD PTR [rdx],xmm8
        jmp     .L\Mode\().ExitKernel

        .endm

        SgemmKernelSseFunction Zero
        SgemmKernelSseFunction Add

        .end
