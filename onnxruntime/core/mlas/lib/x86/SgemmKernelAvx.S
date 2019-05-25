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

//
// Stack frame layout for the SGEMM kernel.
//

#define SgemmKernelFrame 0
#define SgemmKernelFrame_SavedEsi 4
#define SgemmKernelFrame_SavedEbx 8
#define SgemmKernelFrame_SavedEbp 12
#define SgemmKernelFrame_ReturnAddress 16
#define SgemmKernelFrame_MatrixA 20
#define SgemmKernelFrame_MatrixB 24
#define SgemmKernelFrame_MatrixC 28
#define SgemmKernelFrame_CountK 32
#define SgemmKernelFrame_CountM 36
#define SgemmKernelFrame_CountN 40
#define SgemmKernelFrame_lda 44
#define SgemmKernelFrame_ldc 48
#define SgemmKernelFrame_alpha 52

        .text

/*++

Macro Description:

    This macro multiplies and accumulates for a 16xN block (where N is 1,2)
    of the output matrix.

Arguments:

    Count - Supplies the number of rows to access from matrix A.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    ebx - Supplies the length in bytes of a row from matrix A.

    ecx - Supplies the address into the matrix A data.

    edx - Supplies the address into the matrix B data.

    ymm4-ymm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxBy16 Count, VectorOffset, BroadcastOffset

.if \Count\() == 1
        vbroadcastss ymm3,DWORD PTR [ecx+\BroadcastOffset\()]
        vmulps  ymm1,ymm3,YMMWORD PTR [edx+\VectorOffset\()]
        vaddps  ymm4,ymm1,ymm4
        vmulps  ymm3,ymm3,YMMWORD PTR [edx+\VectorOffset\()+32]
        vaddps  ymm5,ymm3,ymm5
.else
        vmovaps ymm0,YMMWORD PTR [edx+\VectorOffset\()]
        vmovaps ymm1,YMMWORD PTR [edx+\VectorOffset\()+32]
        vbroadcastss ymm3,DWORD PTR [ecx+\BroadcastOffset\()]
        vmulps  ymm2,ymm3,ymm0
        vaddps  ymm4,ymm2,ymm4
        vmulps  ymm2,ymm3,ymm1
        vaddps  ymm5,ymm2,ymm5
        vbroadcastss ymm3,DWORD PTR [ecx+ebx+\BroadcastOffset\()]
        vmulps  ymm2,ymm3,ymm0
        vaddps  ymm6,ymm2,ymm6
        vmulps  ymm2,ymm3,ymm1
        vaddps  ymm7,ymm2,ymm7
.endif

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for a 8xN block (where N is 1,2)
    of the output matrix.

Arguments:

    Count - Supplies the number of rows to access from matrix A.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    ebx - Supplies the length in bytes of a row from matrix A.

    ecx - Supplies the address into the matrix A data.

    edx - Supplies the address into the matrix B data.

    ymm4-ymm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxBy8 Count, VectorOffset, BroadcastOffset

.if \Count\() == 1
        vbroadcastss ymm3,DWORD PTR [ecx+\BroadcastOffset\()]
        vmulps  ymm3,ymm3,YMMWORD PTR [edx+\VectorOffset\()]
        vaddps  ymm5,ymm3,ymm5
.else
        vmovaps ymm0,YMMWORD PTR [edx+\VectorOffset\()]
        vbroadcastss ymm3,DWORD PTR [ecx+\BroadcastOffset\()]
        vmulps  ymm3,ymm3,ymm0
        vaddps  ymm5,ymm3,ymm5
        vbroadcastss ymm3,DWORD PTR [ecx+ebx+\BroadcastOffset\()]
        vmulps  ymm3,ymm3,ymm0
        vaddps  ymm7,ymm3,ymm7
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

    ebx - Supplies the number of bytes to the next row of matrix A.

    ecx - Supplies the address into the matrix A data.

    edx - Supplies the address into the matrix B data.

    edi - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    ymm4-ymm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxLoop Mode, ComputeBlock, Count

        sub     edi,4
        jb      .L\Mode\().\ComputeBlock\().\Count\().ProcessRemainingBlocks

.L\Mode\().\ComputeBlock\().\Count\().ComputeBlockBy4Loop:
        \ComputeBlock\() \Count\(), 0, 0
        \ComputeBlock\() \Count\(), 16*4, 4
        sub     edx,-32*4                   # advance matrix B by 32 columns
        \ComputeBlock\() \Count\(), 0, 8
        \ComputeBlock\() \Count\(), 16*4, 12
        sub     edx,-32*4                   # advance matrix B by 32 columns
        add     ecx,4*4                     # advance matrix A by 4 columns
        sub     edi,4
        jae     .L\Mode\().\ComputeBlock\().\Count\().ComputeBlockBy4Loop

.L\Mode\().\ComputeBlock\().\Count\().ProcessRemainingBlocks:
        add     edi,4                       # correct for over-subtract above
        jz      .L\Mode\().\ComputeBlock\().\Count\().OutputBlock

.L\Mode\().\ComputeBlock\().\Count\().ComputeBlockBy1Loop:
        \ComputeBlock\() \Count\(), 0, 0
        add     edx,16*4                    # advance matrix B by 16 columns
        add     ecx,4                       # advance matrix A by 1 column
        dec     edi
        jne     .L\Mode\().\ComputeBlock\().\Count\().ComputeBlockBy1Loop

.L\Mode\().\ComputeBlock\().\Count\().OutputBlock:

        .endm

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of
        rows from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    Alpha - Supplies the scaler multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/

        .macro  SgemmKernelAvxFunction Mode

        .globl  C_UNDERSCORE(MlasSgemmKernel\Mode\()Avx)
C_UNDERSCORE(MlasSgemmKernel\Mode\()Avx):

        push    ebp
        push    ebx
        push    esi
        push    edi
        mov     edx,SgemmKernelFrame_MatrixB[esp]
        mov     esi,SgemmKernelFrame_MatrixC[esp]
        mov     ebp,SgemmKernelFrame_CountN[esp]

//
// Process 2 rows of the matrices.
//

        cmp     DWORD PTR SgemmKernelFrame_CountM[esp],2
        jb      .L\Mode\().ProcessCountMLessThan2
        mov     BYTE PTR SgemmKernelFrame_CountM[esp],2
        mov     eax,SgemmKernelFrame_ldc[esp]
        mov     ebx,SgemmKernelFrame_lda[esp]
        shl     eax,2                       # convert ldc to bytes
        shl     ebx,2                       # convert lda to bytes
        cmp     ebp,8
        jbe     .L\Mode\().ProcessRemainingCountN2

.L\Mode\().ProcessNextColumnLoop16x2:
        mov     edi,SgemmKernelFrame_CountK[esp]
        mov     ecx,SgemmKernelFrame_MatrixA[esp]
        vxorps  xmm4,xmm4,xmm4              # clear block accumulators
        vxorps  xmm5,xmm5,xmm5
        vxorps  xmm6,xmm6,xmm6
        vxorps  xmm7,xmm7,xmm7
        ComputeBlockAvxLoop \Mode\(), ComputeBlockAvxBy16, 2
        vbroadcastss ymm2,DWORD PTR SgemmKernelFrame_alpha[esp]
        vmulps  ymm4,ymm4,ymm2              # multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        vmulps  ymm6,ymm6,ymm2
        vmulps  ymm7,ymm7,ymm2
        sub     ebp,16
        jb      .L\Mode\().OutputMasked16x2Block
.ifeqs "\Mode\()","Add"
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,YMMWORD PTR [esi+32]
        vaddps  ymm6,ymm6,YMMWORD PTR [esi+eax]
        vaddps  ymm7,ymm7,YMMWORD PTR [esi+eax+32]
.endif
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+32],ymm5
        vmovups YMMWORD PTR [esi+eax],ymm6
        vmovups YMMWORD PTR [esi+eax+32],ymm7
        add     esi,16*4                    # advance matrix C by 16 columns
        cmp     ebp,8
        ja      .L\Mode\().ProcessNextColumnLoop16x2
        test    ebp,ebp
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN2:
        mov     edi,SgemmKernelFrame_CountK[esp]
        mov     ecx,SgemmKernelFrame_MatrixA[esp]
        vxorps  xmm5,xmm5,xmm5              # clear block accumulators
        vxorps  xmm7,xmm7,xmm7
        ComputeBlockAvxLoop \Mode\(), ComputeBlockAvxBy8, 2
        vbroadcastss ymm2,DWORD PTR SgemmKernelFrame_alpha[esp]
        vmulps  ymm5,ymm5,ymm2              # multiply by alpha
        vmulps  ymm7,ymm7,ymm2
        cmp     ebp,8
        jb      .L\Mode\().OutputMasked8x2Block
.ifeqs "\Mode\()","Add"
        vaddps  ymm5,ymm5,YMMWORD PTR [esi]
        vaddps  ymm7,ymm7,YMMWORD PTR [esi+eax]
.endif
        vmovups YMMWORD PTR [esi],ymm5
        vmovups YMMWORD PTR [esi+eax],ymm7

//
// Restore non-volatile registers and return.
//

.L\Mode\().ExitKernel:
        movzx   eax,BYTE PTR SgemmKernelFrame_CountM[esp]
        vzeroupper
        pop     edi
        pop     esi
        pop     ebx
        pop     ebp
        ret

.L\Mode\().OutputMasked16x2Block:
.ifeqs "\Mode\()","Add"
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm6,ymm6,YMMWORD PTR [esi+eax]
.endif
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+eax],ymm6
        add     esi,8*4                     # advance matrix C by 8 columns
        add     ebp,8                       # correct for over-subtract above

.L\Mode\().OutputMasked8x2Block:
        call    __x86.get_pc_thunk.bx
        add     ebx,OFFSET _GLOBAL_OFFSET_TABLE_
        mov     ebx,DWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)@GOT[ebx]
        mov     SgemmKernelFrame_CountN[esp],ebp
        vbroadcastss xmm0,SgemmKernelFrame_CountN[esp]
        vpcmpgtd xmm1,xmm0,XMMWORD PTR [ebx+16]
        vpcmpgtd xmm0,xmm0,XMMWORD PTR [ebx]
        vinsertf128 ymm0,ymm0,xmm1,1
.ifeqs "\Mode\()","Add"
        vmaskmovps ymm4,ymm0,YMMWORD PTR [esi]
        vmaskmovps ymm6,ymm0,YMMWORD PTR [esi+eax]
        vaddps  ymm5,ymm5,ymm4
        vaddps  ymm7,ymm7,ymm6
.endif
        vmaskmovps YMMWORD PTR [esi],ymm0,ymm5
        vmaskmovps YMMWORD PTR [esi+eax],ymm0,ymm7
        jmp     .L\Mode\().ExitKernel

//
// Process 1 row of the matrices.
//

.L\Mode\().ProcessCountMLessThan2:
        mov     BYTE PTR SgemmKernelFrame_CountM[esp],1
        mov     ebx,SgemmKernelFrame_MatrixA[esp]
        vbroadcastss ymm2,DWORD PTR SgemmKernelFrame_alpha[esp]
        cmp     ebp,8
        jbe     .L\Mode\().ProcessRemainingCountN1

.L\Mode\().ProcessNextColumnLoop16x1:
        mov     edi,SgemmKernelFrame_CountK[esp]
        mov     ecx,ebx                     # reload matrix A
        vxorps  xmm4,xmm4,xmm4              # clear block accumulators
        vxorps  xmm5,xmm5,xmm5
        ComputeBlockAvxLoop \Mode\(), ComputeBlockAvxBy16, 1
        vmulps  ymm4,ymm4,ymm2              # multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        sub     ebp,16
        jb      .L\Mode\().OutputMasked16x1Block
.ifeqs "\Mode\()","Add"
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,YMMWORD PTR [esi+32]
.endif
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+32],ymm5
        add     esi,16*4                    # advance matrix C by 16 columns
        cmp     ebp,8
        ja      .L\Mode\().ProcessNextColumnLoop16x1
        test    ebp,ebp
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN1:
        mov     edi,SgemmKernelFrame_CountK[esp]
        mov     ecx,ebx                     # reload matrix A
        vxorps  xmm5,xmm5,xmm5              # clear block accumulators
        ComputeBlockAvxLoop \Mode\(), ComputeBlockAvxBy8, 1
        vmulps  ymm5,ymm5,ymm2              # multiply by alpha
        cmp     ebp,8
        jb      .L\Mode\().OutputMasked8x1Block
.ifeqs "\Mode\()","Add"
        vaddps  ymm5,ymm5,YMMWORD PTR [esi]
.endif
        vmovups YMMWORD PTR [esi],ymm5
        jmp     .L\Mode\().ExitKernel

.L\Mode\().OutputMasked16x1Block:
.ifeqs "\Mode\()","Add"
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
.endif
        vmovups YMMWORD PTR [esi],ymm4
        add     esi,8*4                     # advance matrix C by 8 columns
        add     ebp,8                       # correct for over-subtract above

.L\Mode\().OutputMasked8x1Block:
        call    __x86.get_pc_thunk.bx
        add     ebx,OFFSET _GLOBAL_OFFSET_TABLE_
        mov     ebx,DWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)@GOT[ebx]
        mov     SgemmKernelFrame_CountN[esp],ebp
        vbroadcastss xmm0,SgemmKernelFrame_CountN[esp]
        vpcmpgtd xmm1,xmm0,XMMWORD PTR [ebx+16]
        vpcmpgtd xmm0,xmm0,XMMWORD PTR [ebx]
        vinsertf128 ymm0,ymm0,xmm1,1
.ifeqs "\Mode\()","Add"
        vmaskmovps ymm4,ymm0,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,ymm4
.endif
        vmaskmovps YMMWORD PTR [esi],ymm0,ymm5
        jmp     .L\Mode\().ExitKernel

        .endm

        SgemmKernelAvxFunction Zero
        SgemmKernelAvxFunction Add

        .end
