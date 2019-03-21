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

    This macro multiplies and accumulates for a Nx1 block (where N is 1,2,3,4)
    of the output matrix.

Arguments:

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    Shuffle - Supplies the shuffle mask to extract the element from matrix A.

Implicit Arguments:

    ebx - Supplies the length in bytes of a row from matrix A.

    ecx - Supplies the address into the matrix A data.

    edx - Supplies the address into the matrix B data.

    xmm2 - Supplies up to four elements loaded from matrix A.

    xmm4-xmm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlockSseBy4 VectorOffset, Shuffle

        pshufd  xmm3,xmm1,\Shuffle\()
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()]
        mulps   xmm0,xmm3
        addps   xmm4,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+16]
        mulps   xmm0,xmm3
        addps   xmm5,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+32]
        mulps   xmm0,xmm3
        addps   xmm6,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+48]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

        .endm

        .macro ComputeBlockSseBy3 VectorOffset, Shuffle

        pshufd  xmm3,xmm1,\Shuffle\()
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()]
        mulps   xmm0,xmm3
        addps   xmm5,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+16]
        mulps   xmm0,xmm3
        addps   xmm6,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+32]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

        .endm

        .macro ComputeBlockSseBy2 VectorOffset, Shuffle

        pshufd  xmm3,xmm1,\Shuffle\()
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()]
        mulps   xmm0,xmm3
        addps   xmm6,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+16]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

        .endm

        .macro ComputeBlockSseBy1 VectorOffset, Shuffle

        pshufd  xmm3,xmm1,\Shuffle\()
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

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

    xmm4-xmm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlockSseLoop Mode, Count

        sub     edi,4
        jb      .L\Mode\().\Count\().ProcessRemainingBlocks

.L\Mode\().\Count\().ComputeBlockBy4Loop:
        movups  xmm1,XMMWORD PTR [ecx]
        ComputeBlockSseBy\Count\() 0, 0x00
        ComputeBlockSseBy\Count\() 16*4, 0x55
        sub     edx,-32*4                   # advance matrix B by 32 columns
        ComputeBlockSseBy\Count\() 0, 0xAA
        ComputeBlockSseBy\Count\() 16*4, 0xFF
        sub     edx,-32*4                   # advance matrix B by 32 columns
        add     ecx,4*4                     # advance matrix A by 4 columns
        sub     edi,4
        jae     .L\Mode\().\Count\().ComputeBlockBy4Loop

.L\Mode\().\Count\().ProcessRemainingBlocks:
        add     edi,4                       # correct for over-subtract above
        jz      .L\Mode\().\Count\().OutputBlock

.L\Mode\().\Count\().ComputeBlockBy1Loop:
        movss   xmm1,DWORD PTR [ecx]
        ComputeBlockSseBy\Count\() 0, 0x00
        add     edx,16*4                    # advance matrix B by 16 columns
        add     ecx,4                       # advance matrix A by 1 column
        dec     edi
        jne     .L\Mode\().\Count\().ComputeBlockBy1Loop

.L\Mode\().\Count\().OutputBlock:

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

        .macro  SgemmKernelSseFunction Mode

        .globl  C_UNDERSCORE(MlasSgemmKernel\Mode\()Sse)
C_UNDERSCORE(MlasSgemmKernel\Mode\()Sse):

        push    ebp
        push    ebx
        push    esi
        push    edi
        mov     edx,SgemmKernelFrame_MatrixB[esp]
        mov     esi,SgemmKernelFrame_MatrixC[esp]
        mov     ebp,SgemmKernelFrame_CountN[esp]

//
// Process 1 row of the matrices.
//

        mov     eax,SgemmKernelFrame_CountK[esp]
        mov     ebx,SgemmKernelFrame_MatrixA[esp]
        cmp     ebp,12
        jbe     .L\Mode\().ProcessRemainingCountN

.L\Mode\().ProcessNextColumnLoop16x1:
        mov     edi,eax                     # reload CountK
        mov     ecx,ebx                     # reload matrix A
        xorps   xmm4,xmm4                   # clear block accumulators
        xorps   xmm5,xmm5
        xorps   xmm6,xmm6
        xorps   xmm7,xmm7
        ComputeBlockSseLoop \Mode\(), 4
        movss   xmm2,DWORD PTR SgemmKernelFrame_alpha[esp]
        shufps  xmm2,xmm2,0
        mulps   xmm4,xmm2                   # multiply by alpha
        mulps   xmm5,xmm2
        mulps   xmm6,xmm2
        mulps   xmm7,xmm2
.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [esi]
        movups  xmm1,XMMWORD PTR [esi+16]
        movups  xmm2,XMMWORD PTR [esi+32]
        addps   xmm4,xmm0
        addps   xmm5,xmm1
        addps   xmm6,xmm2
.endif
        movups  XMMWORD PTR [esi],xmm4
        movups  XMMWORD PTR [esi+16],xmm5
        movups  XMMWORD PTR [esi+32],xmm6
        sub     ebp,16
        jb      .L\Mode\().OutputMasked16x1Block
.ifeqs "\Mode\()","Add"
        movups  xmm3,XMMWORD PTR [esi+48]
        addps   xmm7,xmm3
.endif
        movups  XMMWORD PTR [esi+48],xmm7
        add     esi,16*4                    # advance matrix C by 16 columns
        cmp     ebp,12
        ja      .L\Mode\().ProcessNextColumnLoop16x1
        test    ebp,ebp
        jnz     .L\Mode\().ProcessRemainingCountN

//
// Restore non-volatile registers and return.
//

.L\Mode\().ExitKernel:
        mov     eax,1                       # return 1 row handled
        pop     edi
        pop     esi
        pop     ebx
        pop     ebp
        ret

//
// Process the remaining 1 to 12 columns of the matrices.
//

.L\Mode\().ProcessRemainingCountN:
        mov     edi,eax                     # reload CountK
        mov     ecx,ebx                     # reload matrix A
        movss   xmm4,DWORD PTR SgemmKernelFrame_alpha[esp]
        shufps  xmm4,xmm4,0
        xorps   xmm5,xmm5                   # clear block accumulators
        xorps   xmm6,xmm6
        xorps   xmm7,xmm7
        cmp     ebp,4
        jbe     .L\Mode\().ProcessRemainingCountN4OrLess
        cmp     ebp,8
        jbe     .L\Mode\().ProcessRemainingCountN8OrLess

.L\Mode\().ProcessRemainingCountN12OrLess:
        ComputeBlockSseLoop \Mode\(), 3
        mulps   xmm5,xmm4                   # multiply by alpha
        mulps   xmm6,xmm4
        mulps   xmm7,xmm4
.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [esi]
        movups  xmm1,XMMWORD PTR [esi+16]
        addps   xmm5,xmm0
        addps   xmm6,xmm1
.endif
        movups  XMMWORD PTR [esi],xmm5
        movups  XMMWORD PTR [esi+16],xmm6
        add     esi,8*4                     # advance matrix C by 8 columns
        jmp     .L\Mode\().OutputTrailingBlock

.L\Mode\().ProcessRemainingCountN8OrLess:
        ComputeBlockSseLoop \Mode\(), 2
        mulps   xmm6,xmm4                   # multiply by alpha
        mulps   xmm7,xmm4
.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [esi]
        addps   xmm6,xmm0
.endif
        movups  XMMWORD PTR [esi],xmm6
        add     esi,4*4                     # advance matrix C by 4 columns
        jmp     .L\Mode\().OutputTrailingBlock

.L\Mode\().ProcessRemainingCountN4OrLess:
        ComputeBlockSseLoop \Mode\(), 1
        mulps   xmm7,xmm4                   # multiply by alpha
        jmp     .L\Mode\().OutputTrailingBlock

.L\Mode\().OutputMasked16x1Block:
        add     esi,12*4                    # advance matrix C by 12 columns

.L\Mode\().OutputTrailingBlock:
        test    ebp,3
        jz      .L\Mode\().OutputTrailingBlock4Elements
        test    ebp,2
        jz      .L\Mode\().OutputTrailingBlock1Element

.L\Mode\().OutputTrailingBlock2Elements:
.ifeqs "\Mode\()","Add"
        movsd   xmm0,QWORD PTR [esi]
        addps   xmm7,xmm0
.endif
        movsd   QWORD PTR [esi],xmm7
        test    ebp,1
        jz      .L\Mode\().ExitKernel
        shufps  xmm7,xmm7,0xAA              # shuffle third float down
        add     esi,2*4                     # advance matrix C by 2 columns

.L\Mode\().OutputTrailingBlock1Element:
.ifeqs "\Mode\()","Add"
        movss   xmm0,DWORD PTR [esi]
        addss   xmm7,xmm0
.endif
        movss   DWORD PTR [esi],xmm7
        jmp     .L\Mode\().ExitKernel

.L\Mode\().OutputTrailingBlock4Elements:
.ifeqs "\Mode\()","Add"
        movups  xmm0,XMMWORD PTR [esi]
        addps   xmm7,xmm0
.endif
        movups  XMMWORD PTR [esi],xmm7
        jmp     .L\Mode\().ExitKernel

        .endm

        SgemmKernelSseFunction Zero
        SgemmKernelSseFunction Add

        .end
