;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelSse2.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;   This implementation uses SSE2 instructions.
;
;--

        .686
        .xmm

        .xlist
INCLUDE mlasi.inc
INCLUDE SgemmKernelCommon.inc
        .list

        ASSUME  DS:FLAT,ES:FLAT,SS:NOTHING,FS:NOTHING,GS:NOTHING

_TEXT   SEGMENT DWORD PUBLIC 'CODE'

;
; Macro Description:
;
;   This macro multiplies and accumulates for a Nx1 block of the output matrix.
;
; Arguments:
;
;   VectorOffset - Supplies the byte offset from matrix B to fetch elements.
;
;   Shuffle - Supplies the shuffle mask to extract the element from matrix A.
;
; Implicit Arguments:
;
;   ebx - Supplies the length in bytes of a row from matrix A.
;
;   ecx - Supplies the address into the matrix A data.
;
;   edx - Supplies the address into the matrix B data.
;
;   xmm2 - Supplies up to four elements loaded from matrix A.
;
;   xmm4-xmm7 - Supplies the block accumulators.
;

ComputeBlockSseBy4 MACRO VectorOffset, Shuffle

        pshufd  xmm3,xmm1,Shuffle
        movaps  xmm0,XMMWORD PTR [edx+VectorOffset]
        mulps   xmm0,xmm3
        addps   xmm4,xmm0
        movaps  xmm0,XMMWORD PTR [edx+VectorOffset+16]
        mulps   xmm0,xmm3
        addps   xmm5,xmm0
        movaps  xmm0,XMMWORD PTR [edx+VectorOffset+32]
        mulps   xmm0,xmm3
        addps   xmm6,xmm0
        movaps  xmm0,XMMWORD PTR [edx+VectorOffset+48]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

        ENDM

ComputeBlockSseBy3 MACRO VectorOffset, Shuffle

        pshufd  xmm3,xmm1,Shuffle
        movaps  xmm0,XMMWORD PTR [edx+VectorOffset]
        mulps   xmm0,xmm3
        addps   xmm5,xmm0
        movaps  xmm0,XMMWORD PTR [edx+VectorOffset+16]
        mulps   xmm0,xmm3
        addps   xmm6,xmm0
        movaps  xmm0,XMMWORD PTR [edx+VectorOffset+32]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

        ENDM

ComputeBlockSseBy2 MACRO VectorOffset, Shuffle

        pshufd  xmm3,xmm1,Shuffle
        movaps  xmm0,XMMWORD PTR [edx+VectorOffset]
        mulps   xmm0,xmm3
        addps   xmm6,xmm0
        movaps  xmm0,XMMWORD PTR [edx+VectorOffset+16]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

        ENDM

ComputeBlockSseBy1 MACRO VectorOffset, Shuffle

        pshufd  xmm3,xmm1,Shuffle
        movaps  xmm0,XMMWORD PTR [edx+VectorOffset]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

        ENDM

;
; Macro Description:
;
;   This macro generates code to execute the block compute macro multiple
;   times and advancing the matrix A and matrix B data pointers.
;
; Arguments:
;
;   ComputeBlock - Supplies the macro to compute a single block.
;
;   RowCount - Supplies the number of rows to process.
;
; Implicit Arguments:
;
;   ebx - Supplies the number of bytes to the next row of matrix A.
;
;   ecx - Supplies the address into the matrix A data.
;
;   edx - Supplies the address into the matrix B data.
;
;   edi - Supplies the number of columns from matrix A and the number of rows
;       from matrix B to iterate over.
;
;   xmm4-xmm7 - Supplies the block accumulators.
;

ComputeBlockSseLoop MACRO RowCount

        LOCAL   ComputeBlockBy4Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   OutputBlock

        sub     edi,4
        jb      ProcessRemainingBlocks

ComputeBlockBy4Loop:
        movups  xmm1,XMMWORD PTR [ecx]
        ComputeBlockSseBy&RowCount 0,000h
        ComputeBlockSseBy&RowCount 16*4,055h
        sub     edx,-32*4                   ; advance matrix B by 32 columns
        ComputeBlockSseBy&RowCount 0,0AAh
        ComputeBlockSseBy&RowCount 16*4,0FFh
        sub     edx,-32*4                   ; advance matrix B by 32 columns
        add     ecx,4*4                     ; advance matrix A by 4 columns
        sub     edi,4
        jae     ComputeBlockBy4Loop

ProcessRemainingBlocks:
        add     edi,4                       ; correct for over-subtract above
        jz      OutputBlock

ComputeBlockBy1Loop:
        movss   xmm1,DWORD PTR [ecx]
        ComputeBlockSseBy&RowCount 0,000h
        add     edx,16*4                    ; advance matrix B by 16 columns
        add     ecx,4                       ; advance matrix A by 1 column
        dec     edi
        jne     ComputeBlockBy1Loop

OutputBlock:

        ENDM

;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows.
;
;   This implementation uses SSE2 instructions.
;
; Arguments:
;
;   A - Supplies the address of matrix A.
;
;   B - Supplies the address of matrix B. The matrix data has been packed using
;       MlasSgemmCopyPackB or MlasSgemmTransposePackB.
;
;   C - Supplies the address of matrix C.
;
;   CountK - Supplies the number of columns from matrix A and the number of rows
;       from matrix B to iterate over.
;
;   CountM - Supplies the maximum number of rows that can be processed for
;       matrix A and matrix C. The actual number of rows handled for this
;       invocation depends on the kernel implementation.
;
;   CountN - Supplies the number of columns from matrix B and matrix C to iterate
;       over.
;
;   lda - Supplies the first dimension of matrix A.
;
;   ldc - Supplies the first dimension of matrix C.
;
;   Alpha - Supplies the scaler alpha multiplier (see SGEMM definition).
;
;   ZeroMode - Supplies true if the output matrix must be zero initialized,
;       else false if the output matrix is accumulated into.
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

cPublicProc _MlasGemmFloatKernelSse,10

        SgemmKernelEntry

;
; Process 1 row of the matrices.
;

        mov     eax,SgemmKernelFrame.CountK[esp]
        mov     ebx,SgemmKernelFrame.MatrixA[esp]
        cmp     ebp,12
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop16x1:
        mov     edi,eax                     ; reload CountK
        mov     ecx,ebx                     ; reload matrix A
        xorps   xmm4,xmm4                   ; clear block accumulators
        xorps   xmm5,xmm5
        xorps   xmm6,xmm6
        xorps   xmm7,xmm7
        ComputeBlockSseLoop 4
        movss   xmm2,DWORD PTR SgemmKernelFrame.Alpha[esp]
        shufps  xmm2,xmm2,0
        mulps   xmm4,xmm2                   ; multiply by alpha
        mulps   xmm5,xmm2
        mulps   xmm6,xmm2
        mulps   xmm7,xmm2
IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [esi]
        movups  xmm1,XMMWORD PTR [esi+16]
        movups  xmm2,XMMWORD PTR [esi+32]
        addps   xmm4,xmm0
        addps   xmm5,xmm1
        addps   xmm6,xmm2
ENDIF
        movups  XMMWORD PTR [esi],xmm4
        movups  XMMWORD PTR [esi+16],xmm5
        movups  XMMWORD PTR [esi+32],xmm6
        sub     ebp,16
        jb      OutputMasked16x1Block
IFIDNI <Mode>, <Add>
        movups  xmm3,XMMWORD PTR [esi+48]
        addps   xmm7,xmm3
ENDIF
        movups  XMMWORD PTR [esi+48],xmm7
        add     esi,16*4                    ; advance matrix C by 16 columns
        cmp     ebp,12
        ja      ProcessNextColumnLoop16x1
        test    ebp,ebp
        jnz     ProcessRemainingCountN

;
; Restore non-volatile registers and return.
;

ExitKernel:
        mov     eax,1                       ; return 1 row handled
        SgemmKernelExit
        stdRET  _MlasGemmFloatKernelSse

;
; Process the remaining 1 to 12 columns of the matrices.
;

ProcessRemainingCountN:
        mov     edi,eax                     ; reload CountK
        mov     ecx,ebx                     ; reload matrix A
        movss   xmm4,DWORD PTR SgemmKernelFrame.Alpha[esp]
        shufps  xmm4,xmm4,0
        xorps   xmm5,xmm5                   ; clear block accumulators
        xorps   xmm6,xmm6
        xorps   xmm7,xmm7
        cmp     ebp,4
        jbe     ProcessRemainingCountN4OrLess
        cmp     ebp,8
        jbe     ProcessRemainingCountN8OrLess

ProcessRemainingCountN12OrLess:
        ComputeBlockSseLoop 3
        mulps   xmm5,xmm4                   ; multiply by alpha
        mulps   xmm6,xmm4
        mulps   xmm7,xmm4
IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [esi]
        movups  xmm1,XMMWORD PTR [esi+16]
        addps   xmm5,xmm0
        addps   xmm6,xmm1
ENDIF
        movups  XMMWORD PTR [esi],xmm5
        movups  XMMWORD PTR [esi+16],xmm6
        add     esi,8*4                     ; advance matrix C by 8 columns
        jmp     OutputTrailingBlock

ProcessRemainingCountN8OrLess:
        ComputeBlockSseLoop 2
        mulps   xmm6,xmm4                   ; multiply by alpha
        mulps   xmm7,xmm4
IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [esi]
        addps   xmm6,xmm0
ENDIF
        movups  XMMWORD PTR [esi],xmm6
        add     esi,4*4                     ; advance matrix C by 4 columns
        jmp     OutputTrailingBlock

ProcessRemainingCountN4OrLess:
        ComputeBlockSseLoop 1
        mulps   xmm7,xmm4                   ; multiply by alpha
        jmp     OutputTrailingBlock

OutputMasked16x1Block:
        add     esi,12*4                    ; advance matrix C by 12 columns

OutputTrailingBlock:
        test    ebp,3
        jz      OutputTrailingBlock4Elements
        test    ebp,2
        jz      OutputTrailingBlock1Element

OutputTrailingBlock2Elements:
IFIDNI <Mode>, <Add>
        movsd   xmm0,MMWORD PTR [esi]
        addps   xmm7,xmm0
ENDIF
        movsd   MMWORD PTR [esi],xmm7
        test    ebp,1
        jz      ExitKernel
        shufps  xmm7,xmm7,0AAh              ; shuffle third float down
        add     esi,2*4                     ; advance matrix C by 2 columns

OutputTrailingBlock1Element:
IFIDNI <Mode>, <Add>
        movss   xmm0,DWORD PTR [esi]
        addss   xmm7,xmm0
ENDIF
        movss   DWORD PTR [esi],xmm7
        jmp     ExitKernel

OutputTrailingBlock4Elements:
IFIDNI <Mode>, <Add>
        movups  xmm0,XMMWORD PTR [esi]
        addps   xmm7,xmm0
ENDIF
        movups  XMMWORD PTR [esi],xmm7
        jmp     ExitKernel

stdENDP _MlasGemmFloatKernelSse

_TEXT   ENDS

        END
