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

        .intel_syntax noprefix

        .equ    SgemmKernelFrame_alpha, -8
        .equ    SgemmKernelFrame_mask, -4
        .equ    SgemmKernelFrame_SavedRbx, 0
        .equ    SgemmKernelFrame_SavedRbp, 8
        .equ    SgemmKernelFrame_ReturnAddress, 16
        .equ    SgemmKernelFrame_lda, 24
        .equ    SgemmKernelFrame_ldc, 32

        .text

/*++

Macro Description:

    This macro multiplies and accumulates for a 32xN block (where N is 1,3)
    of the output matrix.

Arguments:

    Count - Supplies the number of rows to access from matrix A.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockFma3By32 Count, VectorOffset, BroadcastOffset

.if \Count\() == 1
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vfmadd231ps ymm4,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
        vfmadd231ps ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()+32]
        vfmadd231ps ymm6,ymm3,YMMWORD PTR [rsi+rbx+\VectorOffset\()]
        vfmadd231ps ymm7,ymm3,YMMWORD PTR [rsi+rbx+\VectorOffset\()+32]
.endif

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for a 16xN block (where N is 1,3,6)
    of the output matrix.

Arguments:

    Count - Supplies the number of rows to access from matrix A.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockFma3By16 Count, VectorOffset, BroadcastOffset

.if \Count\() == 1
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vfmadd231ps ymm4,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
        vfmadd231ps ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()+32]
.else
        vmovaps ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        vmovaps ymm1,YMMWORD PTR [rsi+\VectorOffset\()+32]
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vfmadd231ps ymm4,ymm3,ymm0
        vfmadd231ps ymm5,ymm3,ymm1
.if \Count\() >= 3
        vbroadcastss ymm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]
        vfmadd231ps ymm6,ymm3,ymm0
        vfmadd231ps ymm7,ymm3,ymm1
        vbroadcastss ymm3,DWORD PTR [rdi+r10*2+\BroadcastOffset\()]
        vfmadd231ps ymm8,ymm3,ymm0
        vfmadd231ps ymm9,ymm3,ymm1
.endif
.if \Count\() >= 6
        vbroadcastss ymm3,DWORD PTR [rbx+\BroadcastOffset\()]
        vfmadd231ps ymm10,ymm3,ymm0
        vfmadd231ps ymm11,ymm3,ymm1
        vbroadcastss ymm3,DWORD PTR [rbx+r10+\BroadcastOffset\()]
        vfmadd231ps ymm12,ymm3,ymm0
        vfmadd231ps ymm13,ymm3,ymm1
        vbroadcastss ymm3,DWORD PTR [rbx+r10*2+\BroadcastOffset\()]
        vfmadd231ps ymm14,ymm3,ymm0
        vfmadd231ps ymm15,ymm3,ymm1
.endif
.endif

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for a 8xN block (where N is 1,3,6)
    of the output matrix.

Arguments:

    Count - Supplies the number of rows to access from matrix A.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockFma3By8 Count, VectorOffset, BroadcastOffset

.if \Count\() == 1
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vfmadd231ps ymm5,ymm3,YMMWORD PTR [rsi+\VectorOffset\()]
.else
        vmovaps ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        vbroadcastss ymm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vfmadd231ps ymm5,ymm3,ymm0
.if \Count\() >= 3
        vbroadcastss ymm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]
        vfmadd231ps ymm7,ymm3,ymm0
        vbroadcastss ymm3,DWORD PTR [rdi+r10*2+\BroadcastOffset\()]
        vfmadd231ps ymm9,ymm3,ymm0
.endif
.if \Count\() >= 6
        vbroadcastss ymm3,DWORD PTR [rbx+\BroadcastOffset\()]
        vfmadd231ps ymm11,ymm3,ymm0
        vbroadcastss ymm3,DWORD PTR [rbx+r10+\BroadcastOffset\()]
        vfmadd231ps ymm13,ymm3,ymm0
        vbroadcastss ymm3,DWORD PTR [rbx+r10*2+\BroadcastOffset\()]
        vfmadd231ps ymm15,ymm3,ymm0
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

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlockFma3Loop Mode, ComputeBlock, Count

//
// Reload the alpha value which is lost after each vzeroall instruction.
//

        vbroadcastss ymm2,DWORD PTR [rsp+SgemmKernelFrame_alpha]

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
.if \Count\() > 3
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
.if \Count\() > 3
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

        .macro  SgemmKernelFma3Function Mode

        .globl  C_UNDERSCORE(MlasSgemmKernel\Mode\()Fma3)
C_UNDERSCORE(MlasSgemmKernel\Mode\()Fma3):

        push    rbp
        push    rbx
        mov     r11,rdi
        mov     r10,[rsp+SgemmKernelFrame_lda]
        shl     r10,2                       # convert lda to bytes
        mov     rax,[rsp+SgemmKernelFrame_ldc]
        shl     rax,2                       # convert ldc to bytes
        vmovss  DWORD PTR [rsp+SgemmKernelFrame_alpha],xmm0
        vzeroall

//
// Process 6 rows of the matrices.
//

        cmp     r8,6
        jb      .L\Mode\().ProcessCountMLessThan6
        mov     r8d,6                       # return 6 rows handled
        cmp     r9,8
        jbe     .L\Mode\().ProcessRemainingCountN6

.L\Mode\().ProcessNextColumnLoop16x6:
        lea     rbx,[r10*2+r10]
        add     rbx,rdi                     # compute matrix A plus 3 rows
        ComputeBlockFma3Loop \Mode\(), ComputeBlockFma3By16, 6
        lea     rdi,[rdx+rax*2]             # compute matrix C plus 2 rows
        lea     rbx,[rdx+rax*4]             # compute matrix C plus 4 rows
.ifnes "\Mode\()","Add"
        vmulps  ymm4,ymm4,ymm2              # multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        vmulps  ymm6,ymm6,ymm2
        vmulps  ymm7,ymm7,ymm2
        vmulps  ymm8,ymm8,ymm2
        vmulps  ymm9,ymm9,ymm2
        vmulps  ymm10,ymm10,ymm2
        vmulps  ymm11,ymm11,ymm2
        vmulps  ymm12,ymm12,ymm2
        vmulps  ymm13,ymm13,ymm2
        vmulps  ymm14,ymm14,ymm2
        vmulps  ymm15,ymm15,ymm2
.endif
        sub     r9,16
        jb      .L\Mode\().OutputMasked16x6Block
.ifeqs "\Mode\()","Add"
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [rdx]
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [rdx+32]
        vfmadd213ps ymm6,ymm2,YMMWORD PTR [rdx+rax]
        vfmadd213ps ymm7,ymm2,YMMWORD PTR [rdx+rax+32]
        vfmadd213ps ymm8,ymm2,YMMWORD PTR [rdi]
        vfmadd213ps ymm9,ymm2,YMMWORD PTR [rdi+32]
        vfmadd213ps ymm10,ymm2,YMMWORD PTR [rdi+rax]
        vfmadd213ps ymm11,ymm2,YMMWORD PTR [rdi+rax+32]
        vfmadd213ps ymm12,ymm2,YMMWORD PTR [rbx]
        vfmadd213ps ymm13,ymm2,YMMWORD PTR [rbx+32]
        vfmadd213ps ymm14,ymm2,YMMWORD PTR [rbx+rax]
        vfmadd213ps ymm15,ymm2,YMMWORD PTR [rbx+rax+32]
.endif
        vmovups YMMWORD PTR [rdx],ymm4
        vmovups YMMWORD PTR [rdx+32],ymm5
        vmovups YMMWORD PTR [rdx+rax],ymm6
        vmovups YMMWORD PTR [rdx+rax+32],ymm7
        vmovups YMMWORD PTR [rdi],ymm8
        vmovups YMMWORD PTR [rdi+32],ymm9
        vmovups YMMWORD PTR [rdi+rax],ymm10
        vmovups YMMWORD PTR [rdi+rax+32],ymm11
        vmovups YMMWORD PTR [rbx],ymm12
        vmovups YMMWORD PTR [rbx+32],ymm13
        vmovups YMMWORD PTR [rbx+rax],ymm14
        vmovups YMMWORD PTR [rbx+rax+32],ymm15
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,8
        ja      .L\Mode\().ProcessNextColumnLoop16x6
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN6:
        lea     rbx,[r10*2+r10]
        add     rbx,rdi                     # compute matrix A plus 3 rows
        ComputeBlockFma3Loop \Mode\(), ComputeBlockFma3By8, 6
        lea     rdi,[rdx+rax*2]             # compute matrix C plus 2 rows
        lea     rbx,[rdx+rax*4]             # compute matrix C plus 4 rows
.ifnes "\Mode\()","Add"
        vmulps  ymm5,ymm5,ymm2              # multiply by alpha
        vmulps  ymm7,ymm7,ymm2
        vmulps  ymm9,ymm9,ymm2
        vmulps  ymm11,ymm11,ymm2
        vmulps  ymm13,ymm13,ymm2
        vmulps  ymm15,ymm15,ymm2
.endif
        cmp     r9,8
        jb      .L\Mode\().OutputMasked8x6Block
.ifeqs "\Mode\()","Add"
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [rdx]
        vfmadd213ps ymm7,ymm2,YMMWORD PTR [rdx+rax]
        vfmadd213ps ymm9,ymm2,YMMWORD PTR [rdi]
        vfmadd213ps ymm11,ymm2,YMMWORD PTR [rdi+rax]
        vfmadd213ps ymm13,ymm2,YMMWORD PTR [rbx]
        vfmadd213ps ymm15,ymm2,YMMWORD PTR [rbx+rax]
.endif
        vmovups YMMWORD PTR [rdx],ymm5
        vmovups YMMWORD PTR [rdx+rax],ymm7
        vmovups YMMWORD PTR [rdi],ymm9
        vmovups YMMWORD PTR [rdi+rax],ymm11
        vmovups YMMWORD PTR [rbx],ymm13
        vmovups YMMWORD PTR [rbx+rax],ymm15
        jmp     .L\Mode\().ExitKernelAndZeroUpper

.L\Mode\().OutputMasked16x6Block:
.ifeqs "\Mode\()","Add"
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [rdx]
        vfmadd213ps ymm6,ymm2,YMMWORD PTR [rdx+rax]
        vfmadd213ps ymm8,ymm2,YMMWORD PTR [rdi]
        vfmadd213ps ymm10,ymm2,YMMWORD PTR [rdi+rax]
        vfmadd213ps ymm12,ymm2,YMMWORD PTR [rbx]
        vfmadd213ps ymm14,ymm2,YMMWORD PTR [rbx+rax]
.endif
        vmovups YMMWORD PTR [rdx],ymm4
        vmovups YMMWORD PTR [rdx+rax],ymm6
        vmovups YMMWORD PTR [rdi],ymm8
        vmovups YMMWORD PTR [rdi+rax],ymm10
        vmovups YMMWORD PTR [rbx],ymm12
        vmovups YMMWORD PTR [rbx+rax],ymm14
        add     rdx,8*4                     # advance matrix C by 8 columns
        add     rdi,8*4                     # advance matrix C plus 2 rows by 8 columns
        add     rbx,8*4                     # advance matrix C plus 4 rows by 8 columns
        add     r9,8                        # correct for over-subtract above

.L\Mode\().OutputMasked8x6Block:
        mov     DWORD PTR [rsp+SgemmKernelFrame_mask],r9d
        vbroadcastss ymm0,DWORD PTR [rsp+SgemmKernelFrame_mask]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
.ifeqs "\Mode\()","Add"
        vmaskmovps ymm4,ymm0,YMMWORD PTR [rdx]
        vmaskmovps ymm6,ymm0,YMMWORD PTR [rdx+rax]
        vmaskmovps ymm8,ymm0,YMMWORD PTR [rdi]
        vmaskmovps ymm10,ymm0,YMMWORD PTR [rdi+rax]
        vmaskmovps ymm12,ymm0,YMMWORD PTR [rbx]
        vmaskmovps ymm14,ymm0,YMMWORD PTR [rbx+rax]
        vfmadd213ps ymm5,ymm2,ymm4
        vfmadd213ps ymm7,ymm2,ymm6
        vfmadd213ps ymm9,ymm2,ymm8
        vfmadd213ps ymm11,ymm2,ymm10
        vfmadd213ps ymm13,ymm2,ymm12
        vfmadd213ps ymm15,ymm2,ymm14
.endif
        vmaskmovps YMMWORD PTR [rdx],ymm0,ymm5
        vmaskmovps YMMWORD PTR [rdx+rax],ymm0,ymm7
        vmaskmovps YMMWORD PTR [rdi],ymm0,ymm9
        vmaskmovps YMMWORD PTR [rdi+rax],ymm0,ymm11
        vmaskmovps YMMWORD PTR [rbx],ymm0,ymm13
        vmaskmovps YMMWORD PTR [rbx+rax],ymm0,ymm15

//
// Restore non-volatile registers and return.
//

.L\Mode\().ExitKernelAndZeroUpper:
        vzeroupper

.L\Mode\().ExitKernel:
        mov     eax,r8d
        pop     rbx
        pop     rbp
        ret

//
// Process 3 rows of the matrices.
//

.L\Mode\().ProcessCountMLessThan6:
        cmp     r8,3                        # try to process 3 rows of matrix A
        jb      .L\Mode\().ProcessCountMLessThan3
        mov     r8d,3                       # return 3 rows handled
        cmp     r9,8
        jbe     .L\Mode\().ProcessRemainingCountN3

.L\Mode\().ProcessNextColumnLoop16x3:
        ComputeBlockFma3Loop \Mode\(), ComputeBlockFma3By16, 3
.ifnes "\Mode\()","Add"
        vmulps  ymm4,ymm4,ymm2              # multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        vmulps  ymm6,ymm6,ymm2
        vmulps  ymm7,ymm7,ymm2
        vmulps  ymm8,ymm8,ymm2
        vmulps  ymm9,ymm9,ymm2
.endif
        sub     r9,16
        jb      .L\Mode\().OutputMasked16x3Block
.ifeqs "\Mode\()","Add"
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [rdx]
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [rdx+32]
        vfmadd213ps ymm6,ymm2,YMMWORD PTR [rdx+rax]
        vfmadd213ps ymm7,ymm2,YMMWORD PTR [rdx+rax+32]
        vfmadd213ps ymm8,ymm2,YMMWORD PTR [rdx+rax*2]
        vfmadd213ps ymm9,ymm2,YMMWORD PTR [rdx+rax*2+32]
.endif
        vmovups YMMWORD PTR [rdx],ymm4
        vmovups YMMWORD PTR [rdx+32],ymm5
        vmovups YMMWORD PTR [rdx+rax],ymm6
        vmovups YMMWORD PTR [rdx+rax+32],ymm7
        vmovups YMMWORD PTR [rdx+rax*2],ymm8
        vmovups YMMWORD PTR [rdx+rax*2+32],ymm9
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,8
        ja      .L\Mode\().ProcessNextColumnLoop16x3
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN3:
        ComputeBlockFma3Loop \Mode\(), ComputeBlockFma3By8, 3
.ifnes "\Mode\()","Add"
        vmulps  ymm5,ymm5,ymm2              # multiply by alpha
        vmulps  ymm7,ymm7,ymm2
        vmulps  ymm9,ymm9,ymm2
.endif
        cmp     r9,8
        jb      .L\Mode\().OutputMasked8x3Block
.ifeqs "\Mode\()","Add"
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [rdx]
        vfmadd213ps ymm7,ymm2,YMMWORD PTR [rdx+rax]
        vfmadd213ps ymm9,ymm2,YMMWORD PTR [rdx+rax*2]
.endif
        vmovups YMMWORD PTR [rdx],ymm5
        vmovups YMMWORD PTR [rdx+rax],ymm7
        vmovups YMMWORD PTR [rdx+rax*2],ymm9
        jmp     .L\Mode\().ExitKernelAndZeroUpper

.L\Mode\().OutputMasked16x3Block:
.ifeqs "\Mode\()","Add"
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [rdx]
        vfmadd213ps ymm6,ymm2,YMMWORD PTR [rdx+rax]
        vfmadd213ps ymm8,ymm2,YMMWORD PTR [rdx+rax*2]
.endif
        vmovups YMMWORD PTR [rdx],ymm4
        vmovups YMMWORD PTR [rdx+rax],ymm6
        vmovups YMMWORD PTR [rdx+rax*2],ymm8
        add     rdx,8*4                     # advance matrix C by 8 columns
        add     r9,8                        # correct for over-subtract above

.L\Mode\().OutputMasked8x3Block:
        mov     DWORD PTR [rsp+SgemmKernelFrame_mask],r9d
        vbroadcastss ymm0,DWORD PTR [rsp+SgemmKernelFrame_mask]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
.ifeqs "\Mode\()","Add"
        vmaskmovps ymm4,ymm0,YMMWORD PTR [rdx]
        vmaskmovps ymm6,ymm0,YMMWORD PTR [rdx+rax]
        vmaskmovps ymm8,ymm0,YMMWORD PTR [rdx+rax*2]
        vfmadd213ps ymm5,ymm2,ymm4
        vfmadd213ps ymm7,ymm2,ymm6
        vfmadd213ps ymm9,ymm2,ymm8
.endif
        vmaskmovps YMMWORD PTR [rdx],ymm0,ymm5
        vmaskmovps YMMWORD PTR [rdx+rax],ymm0,ymm7
        vmaskmovps YMMWORD PTR [rdx+rax*2],ymm0,ymm9
        jmp     .L\Mode\().ExitKernelAndZeroUpper

//
// Process 1 row of the matrices.
//

.L\Mode\().ProcessCountMLessThan3:
        mov     r8d,1                       # return 1 row handled
        cmp     r9,32
        jb      .L\Mode\().ProcessRemainingCountN1LessThan32
        mov     rbx,rcx
        shl     rbx,6                       # compute 16*CountK*sizeof(float)

.L\Mode\().ProcessNextColumnLoop32x1:
        ComputeBlockFma3Loop \Mode\(), ComputeBlockFma3By32, 1
        add     rsi,rbx                     # advance matrix B by 16*CountK floats
.ifnes "\Mode\()","Add"
        vmulps  ymm4,ymm4,ymm2              # multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        vmulps  ymm6,ymm6,ymm2
        vmulps  ymm7,ymm7,ymm2
.else
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [rdx]
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [rdx+32]
        vfmadd213ps ymm6,ymm2,YMMWORD PTR [rdx+64]
        vfmadd213ps ymm7,ymm2,YMMWORD PTR [rdx+96]
.endif
        sub     r9,32
        vmovups YMMWORD PTR [rdx],ymm4
        vmovups YMMWORD PTR [rdx+32],ymm5
        vmovups YMMWORD PTR [rdx+64],ymm6
        vmovups YMMWORD PTR [rdx+96],ymm7
        add     rdx,32*4                    # advance matrix C by 32 columns
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,32
        jae     .L\Mode\().ProcessNextColumnLoop32x1
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN1LessThan32:
        cmp     r9,8
        jbe     .L\Mode\().ProcessRemainingCountN1

.L\Mode\().ProcessNextColumnLoop16x1:
        ComputeBlockFma3Loop \Mode\(), ComputeBlockFma3By16, 1
.ifnes "\Mode\()","Add"
        vmulps  ymm4,ymm4,ymm2              # multiply by alpha
        vmulps  ymm5,ymm5,ymm2
.endif
        sub     r9,16
        jb      .L\Mode\().OutputMasked16x1Block
.ifeqs "\Mode\()","Add"
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [rdx]
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [rdx+32]
.endif
        vmovups YMMWORD PTR [rdx],ymm4
        vmovups YMMWORD PTR [rdx+32],ymm5
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,8
        ja      .L\Mode\().ProcessNextColumnLoop16x1
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN1:
        ComputeBlockFma3Loop \Mode\(), ComputeBlockFma3By8, 1
.ifnes "\Mode\()","Add"
        vmulps  ymm5,ymm5,ymm2              # multiply by alpha
.endif
        cmp     r9,8
        jb      .L\Mode\().OutputMasked8x1Block
.ifeqs "\Mode\()","Add"
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [rdx]
.endif
        vmovups YMMWORD PTR [rdx],ymm5
        jmp     .L\Mode\().ExitKernelAndZeroUpper

.L\Mode\().OutputMasked16x1Block:
.ifeqs "\Mode\()","Add"
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [rdx]
.endif
        vmovups YMMWORD PTR [rdx],ymm4
        add     rdx,8*4                     # advance matrix C by 8 columns
        add     r9,8                        # correct for over-subtract above

.L\Mode\().OutputMasked8x1Block:
        mov     DWORD PTR [rsp+SgemmKernelFrame_mask],r9d
        vbroadcastss ymm0,DWORD PTR [rsp+SgemmKernelFrame_mask]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
.ifeqs "\Mode\()","Add"
        vmaskmovps ymm4,ymm0,YMMWORD PTR [rdx]
        vfmadd213ps ymm5,ymm2,ymm4
.endif
        vmaskmovps YMMWORD PTR [rdx],ymm0,ymm5
        jmp     .L\Mode\().ExitKernelAndZeroUpper

        .endm

        SgemmKernelFma3Function Zero
        SgemmKernelFma3Function Add

        .end
