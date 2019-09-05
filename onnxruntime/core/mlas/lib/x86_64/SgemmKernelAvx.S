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

    This macro multiplies and accumulates for a 16xN block (where N is 1,2,4)
    of the output matrix.

Arguments:

    Count - Supplies the number of rows to access from matrix A.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 2 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm8-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxBy16 Count, VectorOffset, BroadcastOffset

.if \Count\() == 1
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vmulps  ymm4,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
        vaddps  ymm8,ymm8,ymm4
        vmulps  ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()+32]
        vaddps  ymm9,ymm9,ymm5
.else
        vmovaps ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        vmovaps ymm1,YMMWORD PTR [rsi+\VectorOffset\()+32]
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vmulps  ymm4,ymm3,ymm0
        vaddps  ymm8,ymm8,ymm4
        vmulps  ymm5,ymm3,ymm1
        vaddps  ymm9,ymm9,ymm5
.if \Count\() >= 2
        vbroadcastss ymm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]
        vmulps  ymm6,ymm3,ymm0
        vaddps  ymm10,ymm10,ymm6
        vmulps  ymm7,ymm3,ymm1
        vaddps  ymm11,ymm11,ymm7
.endif
.if \Count\() >= 4
        vbroadcastss ymm3,DWORD PTR [rbx+\BroadcastOffset\()]
        vmulps  ymm4,ymm3,ymm0
        vaddps  ymm12,ymm12,ymm4
        vmulps  ymm5,ymm3,ymm1
        vaddps  ymm13,ymm13,ymm5
        vbroadcastss ymm3,DWORD PTR [rbx+r10+\BroadcastOffset\()]
        vmulps  ymm6,ymm3,ymm0
        vaddps  ymm14,ymm14,ymm6
        vmulps  ymm7,ymm3,ymm1
        vaddps  ymm15,ymm15,ymm7
.endif
.endif

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for a 8xN block (where N is 1,2,4)
    of the output matrix.

Arguments:

    Count - Supplies the number of rows to access from matrix A.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 2 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm8-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxBy8 Count, VectorOffset, BroadcastOffset

.if \Count\() == 1
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vmulps  ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
        vaddps  ymm9,ymm9,ymm5
.else
        vmovaps ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vmulps  ymm5,ymm3,ymm0
        vaddps  ymm9,ymm9,ymm5
.if \Count\() >= 2
        vbroadcastss ymm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]
        vmulps  ymm7,ymm3,ymm0
        vaddps  ymm11,ymm11,ymm7
.endif
.if \Count\() >= 4
        vbroadcastss ymm3,DWORD PTR [rbx+\BroadcastOffset\()]
        vmulps  ymm5,ymm3,ymm0
        vaddps  ymm13,ymm13,ymm5
        vbroadcastss ymm3,DWORD PTR [rbx+r10+\BroadcastOffset\()]
        vmulps  ymm7,ymm3,ymm0
        vaddps  ymm15,ymm15,ymm7
.endif
.endif

        .endm

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple
    times and advancing the matrix A and matrix B data pointers.

Arguments:

    ComputeBlock - Supplies the macro to compute a single block.

    Count - Supplies the number of rows to access from matrix A.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 2 rows.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxLoop Mode, ComputeBlock, Count

        mov     rbp,rcx                     # reload CountK
        sub     rbp,4
        jb      .L\Mode\().\ComputeBlock\().\Count\().ProcessRemainingBlocks

.L\Mode\().\ComputeBlock\().\Count\().ComputeBlockBy4Loop:
        \ComputeBlock\() \Count\(), 0, 0
        \ComputeBlock\() \Count\(), 16*4, 4
        sub     rsi,-32*4                   # advance matrix B by 32 columns
        \ComputeBlock\() \Count\(), 0, 8
        \ComputeBlock\() \Count\(), 16*4, 12
        sub     rsi,-32*4                   # advance matrix B by 32 columns
        add     rdi,4*4                     # advance matrix A by 4 columns
.if \Count\() > 2
        add     rbx,4*4                     # advance matrix A plus rows by 4 columns
.endif
        sub     rbp,4
        jae     .L\Mode\().\ComputeBlock\().\Count\().ComputeBlockBy4Loop

.L\Mode\().\ComputeBlock\().\Count\().ProcessRemainingBlocks:
        add     rbp,4                       # correct for over-subtract above
        jz      .L\Mode\().\ComputeBlock\().\Count\().OutputBlock

.L\Mode\().\ComputeBlock\().\Count\().ComputeBlockBy1Loop:
        \ComputeBlock\() \Count\(), 0, 0
        add     rsi,16*4                    # advance matrix B by 16 columns
        add     rdi,4                       # advance matrix A by 1 column
.if \Count\() > 2
        add     rbx,4                       # advance matrix A plus rows by 1 column
.endif
        dec     rbp
        jne     .L\Mode\().\ComputeBlock\().\Count\().ComputeBlockBy1Loop

.L\Mode\().\ComputeBlock\().\Count\().OutputBlock:

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

        .macro  SgemmKernelAvxFunction Mode

        .globl  C_UNDERSCORE(MlasSgemmKernel\Mode\()Avx)
C_UNDERSCORE(MlasSgemmKernel\Mode\()Avx):

        push    rbp
        push    rbx
        mov     r11,rdi
        mov     r10,[rsp+SgemmKernelFrame_lda]
        shl     r10,2                       # convert lda to bytes
        mov     rax,[rsp+SgemmKernelFrame_ldc]
        shl     rax,2                       # convert ldc to bytes
        vmovss  DWORD PTR [rsp+SgemmKernelFrame_alpha],xmm0
        vbroadcastss ymm2,DWORD PTR [rsp+SgemmKernelFrame_alpha]

//
// Process 4 rows of the matrices.
//

        cmp     r8,4
        jb      .L\Mode\().ProcessCountMLessThan4
        mov     r8d,4                      # return 4 rows handled
        cmp     r9,8
        jbe     .L\Mode\().ProcessRemainingCountN4

.L\Mode\().ProcessNextColumnLoop16x4:
        vxorps  xmm8,xmm8,xmm8              # clear block accumulators
        vxorps  xmm9,xmm9,xmm9
        vxorps  xmm10,xmm10,xmm10
        vxorps  xmm11,xmm11,xmm11
        vxorps  xmm12,xmm12,xmm12
        vxorps  xmm13,xmm13,xmm13
        vxorps  xmm14,xmm14,xmm14
        vxorps  xmm15,xmm15,xmm15
        lea     rbx,[rdi+r10*2]             # compute matrix A plus 2 rows
        ComputeBlockAvxLoop \Mode\(), ComputeBlockAvxBy16, 4
        vmulps  ymm8,ymm8,ymm2              # multiply by alpha
        vmulps  ymm9,ymm9,ymm2
        vmulps  ymm10,ymm10,ymm2
        vmulps  ymm11,ymm11,ymm2
        vmulps  ymm12,ymm12,ymm2
        vmulps  ymm13,ymm13,ymm2
        vmulps  ymm14,ymm14,ymm2
        vmulps  ymm15,ymm15,ymm2
        lea     rdi,[rdx+rax*2]             # compute matrix C plus 2 rows
        sub     r9,16
        jb      .L\Mode\().OutputMasked16x4Block
.ifeqs "\Mode\()","Add"
        vaddps  ymm8,ymm8,YMMWORD PTR [rdx]
        vaddps  ymm9,ymm9,YMMWORD PTR [rdx+32]
        vaddps  ymm10,ymm10,YMMWORD PTR [rdx+rax]
        vaddps  ymm11,ymm11,YMMWORD PTR [rdx+rax+32]
        vaddps  ymm12,ymm12,YMMWORD PTR [rdi]
        vaddps  ymm13,ymm13,YMMWORD PTR [rdi+32]
        vaddps  ymm14,ymm14,YMMWORD PTR [rdi+rax]
        vaddps  ymm15,ymm15,YMMWORD PTR [rdi+rax+32]
.endif
        vmovups YMMWORD PTR [rdx],ymm8
        vmovups YMMWORD PTR [rdx+32],ymm9
        vmovups YMMWORD PTR [rdx+rax],ymm10
        vmovups YMMWORD PTR [rdx+rax+32],ymm11
        vmovups YMMWORD PTR [rdi],ymm12
        vmovups YMMWORD PTR [rdi+32],ymm13
        vmovups YMMWORD PTR [rdi+rax],ymm14
        vmovups YMMWORD PTR [rdi+rax+32],ymm15
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        cmp     r9,8
        ja      .L\Mode\().ProcessNextColumnLoop16x4
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN4:
        vxorps  xmm9,xmm9,xmm9              # clear block accumulators
        vxorps  xmm11,xmm11,xmm11
        vxorps  xmm13,xmm13,xmm13
        vxorps  xmm15,xmm15,xmm15
        lea     rbx,[rdi+r10*2]             # compute matrix A plus 2 rows
        ComputeBlockAvxLoop \Mode\(), ComputeBlockAvxBy8, 4
        vmulps  ymm9,ymm9,ymm2              # multiply by alpha
        vmulps  ymm11,ymm11,ymm2
        vmulps  ymm13,ymm13,ymm2
        vmulps  ymm15,ymm15,ymm2
        lea     rdi,[rdx+rax*2]             # compute matrix C plus 2 rows
        cmp     r9,8
        jb      .L\Mode\().OutputMasked8x4Block
.ifeqs "\Mode\()","Add"
        vaddps  ymm9,ymm9,YMMWORD PTR [rdx]
        vaddps  ymm11,ymm11,YMMWORD PTR [rdx+rax]
        vaddps  ymm13,ymm13,YMMWORD PTR [rdi]
        vaddps  ymm15,ymm15,YMMWORD PTR [rdi+rax]
.endif
        vmovups YMMWORD PTR [rdx],ymm9
        vmovups YMMWORD PTR [rdx+rax],ymm11
        vmovups YMMWORD PTR [rdi],ymm13
        vmovups YMMWORD PTR [rdi+rax],ymm15
        jmp     .L\Mode\().ExitKernel

.L\Mode\().OutputMasked16x4Block:
.ifeqs "\Mode\()","Add"
        vaddps  ymm8,ymm8,YMMWORD PTR [rdx]
        vaddps  ymm10,ymm10,YMMWORD PTR [rdx+rax]
        vaddps  ymm12,ymm12,YMMWORD PTR [rdi]
        vaddps  ymm14,ymm14,YMMWORD PTR [rdi+rax]
.endif
        vmovups YMMWORD PTR [rdx],ymm8
        vmovups YMMWORD PTR [rdx+rax],ymm10
        vmovups YMMWORD PTR [rdi],ymm12
        vmovups YMMWORD PTR [rdi+rax],ymm14
        add     rdx,8*4                     # advance matrix C by 8 columns
        add     rdi,8*4                     # advance matrix C plus 2 rows by 8 columns
        add     r9,8                        # correct for over-subtract above

.L\Mode\().OutputMasked8x4Block:
        vmovd   xmm0,r9d
        vshufps xmm0,xmm0,xmm0,0
        vpcmpgtd xmm1,xmm0,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip+16]
        vpcmpgtd xmm0,xmm0,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        vinsertf128 ymm0,ymm0,xmm1,1
.ifeqs "\Mode\()","Add"
        vmaskmovps ymm8,ymm0,YMMWORD PTR [rdx]
        vmaskmovps ymm10,ymm0,YMMWORD PTR [rdx+rax]
        vmaskmovps ymm12,ymm0,YMMWORD PTR [rdi]
        vmaskmovps ymm14,ymm0,YMMWORD PTR [rdi+rax]
        vaddps  ymm9,ymm9,ymm8
        vaddps  ymm11,ymm11,ymm10
        vaddps  ymm13,ymm13,ymm12
        vaddps  ymm15,ymm15,ymm14
.endif
        vmaskmovps YMMWORD PTR [rdx],ymm0,ymm9
        vmaskmovps YMMWORD PTR [rdx+rax],ymm0,ymm11
        vmaskmovps YMMWORD PTR [rdi],ymm0,ymm13
        vmaskmovps YMMWORD PTR [rdi+rax],ymm0,ymm15

//
// Restore non-volatile registers and return.
//

.L\Mode\().ExitKernel:
        vzeroupper
        mov     eax,r8d
        pop     rbx
        pop     rbp
        ret

//
// Process 2 rows of the matrices.
//

.L\Mode\().ProcessCountMLessThan4:
        cmp     r8,2
        jb      .L\Mode\().ProcessCountMLessThan2
        mov     r8d,2                       # return 2 rows handled
        cmp     r9,8
        jbe     .L\Mode\().ProcessRemainingCountN2

.L\Mode\().ProcessNextColumnLoop16x2:
        vxorps  xmm8,xmm8,xmm8              # clear block accumulators
        vxorps  xmm9,xmm9,xmm9
        vxorps  xmm10,xmm10,xmm10
        vxorps  xmm11,xmm11,xmm11
        ComputeBlockAvxLoop \Mode\(), ComputeBlockAvxBy16, 2
        vmulps  ymm8,ymm8,ymm2              # multiply by alpha
        vmulps  ymm9,ymm9,ymm2
        vmulps  ymm10,ymm10,ymm2
        vmulps  ymm11,ymm11,ymm2
        sub     r9,16
        jb      .L\Mode\().OutputMasked16x2Block
.ifeqs "\Mode\()","Add"
        vaddps  ymm8,ymm8,YMMWORD PTR [rdx]
        vaddps  ymm9,ymm9,YMMWORD PTR [rdx+32]
        vaddps  ymm10,ymm10,YMMWORD PTR [rdx+rax]
        vaddps  ymm11,ymm11,YMMWORD PTR [rdx+rax+32]
.endif
        vmovups YMMWORD PTR [rdx],ymm8
        vmovups YMMWORD PTR [rdx+32],ymm9
        vmovups YMMWORD PTR [rdx+rax],ymm10
        vmovups YMMWORD PTR [rdx+rax+32],ymm11
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        cmp     r9,8
        ja      .L\Mode\().ProcessNextColumnLoop16x2
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN2:
        vxorps  xmm9,xmm9,xmm9              # clear block accumulators
        vxorps  xmm11,xmm11,xmm11
        ComputeBlockAvxLoop \Mode\(), ComputeBlockAvxBy8, 2
        vmulps  ymm9,ymm9,ymm2              # multiply by alpha
        vmulps  ymm11,ymm11,ymm2
        cmp     r9,8
        jb      .L\Mode\().OutputMasked8x2Block
.ifeqs "\Mode\()","Add"
        vaddps  ymm9,ymm9,YMMWORD PTR [rdx]
        vaddps  ymm11,ymm11,YMMWORD PTR [rdx+rax]
.endif
        vmovups YMMWORD PTR [rdx],ymm9
        vmovups YMMWORD PTR [rdx+rax],ymm11
        jmp     .L\Mode\().ExitKernel

.L\Mode\().OutputMasked16x2Block:
.ifeqs "\Mode\()","Add"
        vaddps  ymm8,ymm8,YMMWORD PTR [rdx]
        vaddps  ymm10,ymm10,YMMWORD PTR [rdx+rax]
.endif
        vmovups YMMWORD PTR [rdx],ymm8
        vmovups YMMWORD PTR [rdx+rax],ymm10
        add     rdx,8*4                     # advance matrix C by 8 columns
        add     r9,8                        # correct for over-subtract above

.L\Mode\().OutputMasked8x2Block:
        vmovd   xmm0,r9d
        vshufps xmm0,xmm0,xmm0,0
        vpcmpgtd xmm1,xmm0,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip+16]
        vpcmpgtd xmm0,xmm0,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        vinsertf128 ymm0,ymm0,xmm1,1
.ifeqs "\Mode\()","Add"
        vmaskmovps ymm8,ymm0,YMMWORD PTR [rdx]
        vmaskmovps ymm10,ymm0,YMMWORD PTR [rdx+rax]
        vaddps  ymm9,ymm9,ymm8
        vaddps  ymm11,ymm11,ymm10
.endif
        vmaskmovps YMMWORD PTR [rdx],ymm0,ymm9
        vmaskmovps YMMWORD PTR [rdx+rax],ymm0,ymm11
        jmp     .L\Mode\().ExitKernel

//
// Process 1 row of the matrices.
//

.L\Mode\().ProcessCountMLessThan2:
        mov     r8d,1                       # return 1 row handled
        cmp     r9,8
        jbe     .L\Mode\().ProcessRemainingCountN1

.L\Mode\().ProcessNextColumnLoop16x1:
        vxorps  xmm8,xmm8,xmm8              # clear block accumulators
        vxorps  xmm9,xmm9,xmm9
        ComputeBlockAvxLoop \Mode\(), ComputeBlockAvxBy16, 1
        vmulps  ymm8,ymm8,ymm2              # multiply by alpha
        vmulps  ymm9,ymm9,ymm2
        sub     r9,16
        jb      .L\Mode\().OutputMasked16x1Block
.ifeqs "\Mode\()","Add"
        vaddps  ymm8,ymm8,YMMWORD PTR [rdx]
        vaddps  ymm9,ymm9,YMMWORD PTR [rdx+32]
.endif
        vmovups YMMWORD PTR [rdx],ymm8
        vmovups YMMWORD PTR [rdx+32],ymm9
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        cmp     r9,8
        ja      .L\Mode\().ProcessNextColumnLoop16x1
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN1:
        vxorps  xmm9,xmm9,xmm9              # clear block accumulators
        ComputeBlockAvxLoop \Mode\(), ComputeBlockAvxBy8, 1
        vmulps  ymm9,ymm9,ymm2              # multiply by alpha
        cmp     r9,8
        jb      .L\Mode\().OutputMasked8x1Block
.ifeqs "\Mode\()","Add"
        vaddps  ymm9,ymm9,YMMWORD PTR [rdx]
.endif
        vmovups YMMWORD PTR [rdx],ymm9
        jmp     .L\Mode\().ExitKernel

.L\Mode\().OutputMasked16x1Block:
.ifeqs "\Mode\()","Add"
        vaddps  ymm8,ymm8,YMMWORD PTR [rdx]
.endif
        vmovups YMMWORD PTR [rdx],ymm8
        add     rdx,8*4                     # advance matrix C by 8 columns
        add     r9,8                        # correct for over-subtract above

.L\Mode\().OutputMasked8x1Block:
        vmovd   xmm0,r9d
        vshufps xmm0,xmm0,xmm0,0
        vpcmpgtd xmm1,xmm0,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip+16]
        vpcmpgtd xmm0,xmm0,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        vinsertf128 ymm0,ymm0,xmm1,1
.ifeqs "\Mode\()","Add"
        vmaskmovps ymm8,ymm0,YMMWORD PTR [rdx]
        vaddps  ymm9,ymm9,ymm8
.endif
        vmaskmovps YMMWORD PTR [rdx],ymm0,ymm9
        jmp     .L\Mode\().ExitKernel

        .endm

        SgemmKernelAvxFunction Zero
        SgemmKernelAvxFunction Add

        .end
