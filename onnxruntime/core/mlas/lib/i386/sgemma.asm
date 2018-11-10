;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   sgemma.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;--

        .686
        .xmm

        .xlist
INCLUDE callconv.inc
        .list

        ASSUME  DS:FLAT,ES:FLAT,SS:NOTHING,FS:NOTHING,GS:NOTHING

        EXTERN  _MlasMaskMoveAvx:NEAR

;
; Stack frame layout for the SGEMM kernels.
;

SgemmKernelFrame STRUCT

        SavedEdi DWORD ?
        SavedEsi DWORD ?
        SavedEbx DWORD ?
        SavedEbp DWORD ?
        ReturnAddress DWORD ?
        MatrixA DWORD ?
        MatrixB DWORD ?
        MatrixC DWORD ?
        CountK DWORD ?
        CountM DWORD ?
        CountN DWORD ?
        lda DWORD ?
        ldc DWORD ?
        Alpha DWORD ?

SgemmKernelFrame ENDS

_TEXT   SEGMENT DWORD PUBLIC 'CODE'

;
; SgemmKernelEntry
;
; Macro Description:
;
;   This macro implements the common prologue code for the SGEMM kernels.
;
; Arguments:
;
;   None.
;
; Return Registers:
;
;   ecx - Stores the address of the matrix A data from the stack frame.
;
;   edx - Stores the address of the matrix B data from the stack frame.
;
;   ebp - Stores the CountN argument from the stack frame.
;
;   ebx, esi, edi - Previous values stored on the stack and the registers are
;       available as temporaries.
;

SgemmKernelEntry MACRO

        push    ebp
        push    ebx
        push    esi
        push    edi
        mov     edx,SgemmKernelFrame.MatrixB[esp]
        mov     esi,SgemmKernelFrame.MatrixC[esp]
        mov     ebp,SgemmKernelFrame.CountN[esp]

cPublicFpo ((SgemmKernelFrame.ReturnAddress)/4),9

        ENDM

;
; SgemmKernelExit
;
; Macro Description:
;
;   This macro implements the common epilogue code for the SGEMM kernels.
;
; Arguments:
;
;   None.
;

SgemmKernelExit MACRO

        pop     edi
        pop     esi
        pop     ebx
        pop     ebp

        ENDM

;
; ComputeBlockSseBy4
; ComputeBlockSseBy3
; ComputeBlockSseBy2
; ComputeBlockSseBy1
;
; Macro Description:
;
;   This macro multiplies and accumulates for a Nx1 block (where N is 1,2,3,4)
;   of the output matrix.
;
;   This implementation uses SSE2 instructions.
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
; ComputeBlockSseLoop
;
;   This macro generates code to execute the block compute macro multiple
;   times and advancing the matrix A and matrix B data pointers.
;
; Arguments:
;
;   ComputeBlock - Supplies the macro to compute a single block.
;
;   Count - Supplies the number of rows to access from matrix A.
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

ComputeBlockSseLoop MACRO Count

        LOCAL   ComputeBlockBy4Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   OutputBlock

        sub     edi,4
        jl      ProcessRemainingBlocks

ComputeBlockBy4Loop:
        movups  xmm1,XMMWORD PTR [ecx]
        ComputeBlockSseBy&Count 0,000h
        ComputeBlockSseBy&Count 16*4,055h
        sub     edx,-32*4                   ; advance matrix B by 32 columns
        ComputeBlockSseBy&Count 0,0AAh
        ComputeBlockSseBy&Count 16*4,0FFh
        sub     edx,-32*4                   ; advance matrix B by 32 columns
        add     ecx,4*4                     ; advance matrix A by 4 columns
        sub     edi,4
        jge     ComputeBlockBy4Loop

ProcessRemainingBlocks:
        add     edi,4                       ; correct for over-subtract above
        jz      OutputBlock

ComputeBlockBy1Loop:
        movss   xmm1,DWORD PTR [ecx]
        ComputeBlockSseBy&Count 0,000h
        add     edx,16*4                    ; advance matrix B by 16 columns
        add     ecx,4                       ; advance matrix A by 1 column
        dec     edi
        jne     ComputeBlockBy1Loop

OutputBlock:

        ENDM

        SUBTTL  "SGEMM kernel for processors supporting SSE2 instructions"
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
;   Alpha - Supplies the scaler multiplier (see SGEMM definition).
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

SgemmKernelSseFunction MACRO Mode

cPublicProc _MlasSgemmKernel&Mode&Sse,9

        SgemmKernelEntry

;
; Process 1 row of the matrices.
;

        mov     eax,SgemmKernelFrame.CountK[esp]
        mov     ebx,SgemmKernelFrame.MatrixA[esp]
        cmp     ebp,12
        jle     ProcessRemainingCountN

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
        jl      OutputMasked16x1Block
IFIDNI <Mode>, <Add>
        movups  xmm3,XMMWORD PTR [esi+48]
        addps   xmm7,xmm3
ENDIF
        movups  XMMWORD PTR [esi+48],xmm7
        add     esi,16*4                    ; advance matrix C by 16 columns
        cmp     ebp,12
        jg      ProcessNextColumnLoop16x1
        test    ebp,ebp
        jnz     ProcessRemainingCountN

;
; Restore non-volatile registers and return.
;

ExitKernel:
        mov     eax,1                       ; return 1 row handled
        SgemmKernelExit
        stdRET  _MlasSgemmKernel&Mode&Sse

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
        jle     ProcessRemainingCountN4OrLess
        cmp     ebp,8
        jle     ProcessRemainingCountN8OrLess

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

stdENDP _MlasSgemmKernel&Mode&Sse

        ENDM

SgemmKernelSseFunction Zero
SgemmKernelSseFunction Add

;
; ComputeBlockAvxBy16
;
; Macro Description:
;
;   This macro multiplies and accumulates for a 16xN block (where N is 1,2)
;   of the output matrix.
;
;   This implementation uses AVX instructions.
;
; Arguments:
;
;   Count - Supplies the number of rows to access from matrix A.
;
;   VectorOffset - Supplies the byte offset from matrix B to fetch elements.
;
;   BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.
;
; Implicit Arguments:
;
;   ebx - Supplies the length in bytes of a row from matrix A.
;
;   ecx - Supplies the address into the matrix A data.
;
;   edx - Supplies the address into the matrix B data.
;
;   ymm4-ymm7 - Supplies the block accumulators.
;

ComputeBlockAvxBy16 MACRO Count, VectorOffset, BroadcastOffset

IF Count EQ 1
        vbroadcastss ymm3,DWORD PTR [ecx+BroadcastOffset]
        vmulps  ymm1,ymm3,YMMWORD PTR [edx+VectorOffset]
        vaddps  ymm4,ymm1,ymm4
        vmulps  ymm3,ymm3,YMMWORD PTR [edx+VectorOffset+32]
        vaddps  ymm5,ymm3,ymm5
ELSE
        vmovaps ymm0,YMMWORD PTR [edx+VectorOffset]
        vmovaps ymm1,YMMWORD PTR [edx+VectorOffset+32]
        vbroadcastss ymm3,DWORD PTR [ecx+BroadcastOffset]
        vmulps  ymm2,ymm3,ymm0
        vaddps  ymm4,ymm2,ymm4
        vmulps  ymm2,ymm3,ymm1
        vaddps  ymm5,ymm2,ymm5
        vbroadcastss ymm3,DWORD PTR [ecx+ebx+BroadcastOffset]
        vmulps  ymm2,ymm3,ymm0
        vaddps  ymm6,ymm2,ymm6
        vmulps  ymm2,ymm3,ymm1
        vaddps  ymm7,ymm2,ymm7
ENDIF

        ENDM

;
; ComputeBlockAvxBy8
;
; Macro Description:
;
;   This macro multiplies and accumulates for a 8xN block (where N is 1,2)
;   of the output matrix.
;
;   This implementation uses AVX instructions.
;
; Arguments:
;
;   Count - Supplies the number of rows to access from matrix A.
;
;   VectorOffset - Supplies the byte offset from matrix B to fetch elements.
;
;   BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.
;
; Implicit Arguments:
;
;   ebx - Supplies the length in bytes of a row from matrix A.
;
;   ecx - Supplies the address into the matrix A data.
;
;   edx - Supplies the address into the matrix B data.
;
;   ymm4-ymm7 - Supplies the block accumulators.
;

ComputeBlockAvxBy8 MACRO Count, VectorOffset, BroadcastOffset

IF Count EQ 1
        vbroadcastss ymm3,DWORD PTR [ecx+BroadcastOffset]
        vmulps  ymm3,ymm3,YMMWORD PTR [edx+VectorOffset]
        vaddps  ymm5,ymm3,ymm5
ELSE
        vmovaps ymm0,YMMWORD PTR [edx+VectorOffset]
        vbroadcastss ymm3,DWORD PTR [ecx+BroadcastOffset]
        vmulps  ymm3,ymm3,ymm0
        vaddps  ymm5,ymm3,ymm5
        vbroadcastss ymm3,DWORD PTR [ecx+ebx+BroadcastOffset]
        vmulps  ymm3,ymm3,ymm0
        vaddps  ymm7,ymm3,ymm7
ENDIF

        ENDM

;
; ComputeBlockAvxLoop
;
;   This macro generates code to execute the block compute macro multiple
;   times and advancing the matrix A and matrix B data pointers.
;
; Arguments:
;
;   ComputeBlock - Supplies the macro to compute a single block.
;
;   Count - Supplies the number of rows to access from matrix A.
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
;   ymm4-ymm7 - Supplies the block accumulators.
;

ComputeBlockAvxLoop MACRO ComputeBlock, Count

        LOCAL   ComputeBlockBy4Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   OutputBlock

        sub     edi,4
        jl      ProcessRemainingBlocks

ComputeBlockBy4Loop:
        ComputeBlock Count, 0, 0
        ComputeBlock Count, 16*4, 4
        sub     edx,-32*4                   ; advance matrix B by 32 columns
        ComputeBlock Count, 0, 8
        ComputeBlock Count, 16*4, 12
        sub     edx,-32*4                   ; advance matrix B by 32 columns
        add     ecx,4*4                     ; advance matrix A by 4 columns
        sub     edi,4
        jge     ComputeBlockBy4Loop

ProcessRemainingBlocks:
        add     edi,4                       ; correct for over-subtract above
        jz      OutputBlock

ComputeBlockBy1Loop:
        ComputeBlock Count, 0, 0
        add     edx,16*4                    ; advance matrix B by 16 columns
        add     ecx,4                       ; advance matrix A by 1 column
        dec     edi
        jne     ComputeBlockBy1Loop

OutputBlock:

        ENDM

        SUBTTL  "SGEMM kernel for processors supporting AVX instructions"
;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows.
;
;   This implementation uses AVX instructions.
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
;   Alpha - Supplies the scaler multiplier (see SGEMM definition).
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

SgemmKernelAvxFunction MACRO Mode

cPublicProc _MlasSgemmKernel&Mode&Avx,9

        vzeroall
        SgemmKernelEntry

;
; Process 2 rows of the matrices.
;

        cmp     SgemmKernelFrame.CountM[esp],2
        jl      ProcessCountMLessThan2
        mov     BYTE PTR SgemmKernelFrame.CountM[esp],2
        mov     eax,SgemmKernelFrame.ldc[esp]
        mov     ebx,SgemmKernelFrame.lda[esp]
        shl     eax,2                       ; convert ldc to bytes
        shl     ebx,2                       ; convert lda to bytes
        cmp     ebp,8
        jle     ProcessRemainingCountN2

ProcessNextColumnLoop16x2:
        mov     edi,SgemmKernelFrame.CountK[esp]
        mov     ecx,SgemmKernelFrame.MatrixA[esp]
        vxorps  xmm4,xmm4,xmm4              ; clear block accumulators
        vxorps  xmm5,xmm5,xmm5
        vxorps  xmm6,xmm6,xmm6
        vxorps  xmm7,xmm7,xmm7
        ComputeBlockAvxLoop ComputeBlockAvxBy16,2
        vbroadcastss ymm2,DWORD PTR SgemmKernelFrame.Alpha[esp]
        vmulps  ymm4,ymm4,ymm2              ; multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        vmulps  ymm6,ymm6,ymm2
        vmulps  ymm7,ymm7,ymm2
        sub     ebp,16
        jl      OutputMasked16x2Block
IFIDNI <Mode>, <Add>
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,YMMWORD PTR [esi+32]
        vaddps  ymm6,ymm6,YMMWORD PTR [esi+eax]
        vaddps  ymm7,ymm7,YMMWORD PTR [esi+eax+32]
ENDIF
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+32],ymm5
        vmovups YMMWORD PTR [esi+eax],ymm6
        vmovups YMMWORD PTR [esi+eax+32],ymm7
        add     esi,16*4                    ; advance matrix C by 16 columns
        cmp     ebp,8
        jg      ProcessNextColumnLoop16x2
        test    ebp,ebp
        jz      ExitKernel

ProcessRemainingCountN2:
        mov     edi,SgemmKernelFrame.CountK[esp]
        mov     ecx,SgemmKernelFrame.MatrixA[esp]
        vxorps  xmm5,xmm5,xmm5              ; clear block accumulators
        vxorps  xmm7,xmm7,xmm7
        ComputeBlockAvxLoop ComputeBlockAvxBy8,2
        vbroadcastss ymm2,DWORD PTR SgemmKernelFrame.Alpha[esp]
        vmulps  ymm5,ymm5,ymm2              ; multiply by alpha
        vmulps  ymm7,ymm7,ymm2
        cmp     ebp,8
        jl      OutputMasked8x2Block
IFIDNI <Mode>, <Add>
        vaddps  ymm5,ymm5,YMMWORD PTR [esi]
        vaddps  ymm7,ymm7,YMMWORD PTR [esi+eax]
ENDIF
        vmovups YMMWORD PTR [esi],ymm5
        vmovups YMMWORD PTR [esi+eax],ymm7

;
; Restore non-volatile registers and return.
;

ExitKernel:
        movzx   eax,BYTE PTR SgemmKernelFrame.CountM[esp]
        vzeroupper
        SgemmKernelExit
        stdRET  _MlasSgemmKernel&Mode&Avx

OutputMasked16x2Block:
IFIDNI <Mode>, <Add>
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm6,ymm6,YMMWORD PTR [esi+eax]
ENDIF
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+eax],ymm6
        add     esi,8*4                     ; advance matrix C by 8 columns
        add     ebp,8                       ; correct for over-subtract above

OutputMasked8x2Block:
        mov     SgemmKernelFrame.CountN[esp],ebp
        vbroadcastss xmm0,SgemmKernelFrame.CountN[esp]
        vpcmpgtd xmm1,xmm0,XMMWORD PTR [_MlasMaskMoveAvx+16]
        vpcmpgtd xmm0,xmm0,XMMWORD PTR [_MlasMaskMoveAvx]
        vinsertf128 ymm0,ymm0,xmm1,1
IFIDNI <Mode>, <Add>
        vmaskmovps ymm4,ymm0,YMMWORD PTR [esi]
        vmaskmovps ymm6,ymm0,YMMWORD PTR [esi+eax]
        vaddps  ymm5,ymm5,ymm4
        vaddps  ymm7,ymm7,ymm6
ENDIF
        vmaskmovps YMMWORD PTR [esi],ymm0,ymm5
        vmaskmovps YMMWORD PTR [esi+eax],ymm0,ymm7
        jmp     ExitKernel

;
; Process 1 row of the matrices.
;

ProcessCountMLessThan2:
        mov     BYTE PTR SgemmKernelFrame.CountM[esp],1
        mov     ebx,SgemmKernelFrame.MatrixA[esp]
        vbroadcastss ymm2,DWORD PTR SgemmKernelFrame.Alpha[esp]
        cmp     ebp,8
        jle     ProcessRemainingCountN1

ProcessNextColumnLoop16x1:
        mov     edi,SgemmKernelFrame.CountK[esp]
        mov     ecx,ebx                     ; reload matrix A
        vxorps  xmm4,xmm4,xmm4              ; clear block accumulators
        vxorps  xmm5,xmm5,xmm5
        ComputeBlockAvxLoop ComputeBlockAvxBy16,1
        vmulps  ymm4,ymm4,ymm2              ; multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        sub     ebp,16
        jl      OutputMasked16x1Block
IFIDNI <Mode>, <Add>
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,YMMWORD PTR [esi+32]
ENDIF
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+32],ymm5
        add     esi,16*4                    ; advance matrix C by 16 columns
        cmp     ebp,8
        jg      ProcessNextColumnLoop16x1
        test    ebp,ebp
        jz      ExitKernel

ProcessRemainingCountN1:
        mov     edi,SgemmKernelFrame.CountK[esp]
        mov     ecx,ebx                     ; reload matrix A
        vxorps  xmm5,xmm5,xmm5              ; clear block accumulators
        ComputeBlockAvxLoop ComputeBlockAvxBy8,1
        vmulps  ymm5,ymm5,ymm2              ; multiply by alpha
        cmp     ebp,8
        jl      OutputMasked8x1Block
IFIDNI <Mode>, <Add>
        vaddps  ymm5,ymm5,YMMWORD PTR [esi]
ENDIF
        vmovups YMMWORD PTR [esi],ymm5
        jmp     ExitKernel

OutputMasked16x1Block:
IFIDNI <Mode>, <Add>
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
ENDIF
        vmovups YMMWORD PTR [esi],ymm4
        add     esi,8*4                     ; advance matrix C by 8 columns
        add     ebp,8                       ; correct for over-subtract above

OutputMasked8x1Block:
        mov     SgemmKernelFrame.CountN[esp],ebp
        vbroadcastss xmm0,SgemmKernelFrame.CountN[esp]
        vpcmpgtd xmm1,xmm0,XMMWORD PTR [_MlasMaskMoveAvx+16]
        vpcmpgtd xmm0,xmm0,XMMWORD PTR [_MlasMaskMoveAvx]
        vinsertf128 ymm0,ymm0,xmm1,1
IFIDNI <Mode>, <Add>
        vmaskmovps ymm4,ymm0,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,ymm4
ENDIF
        vmaskmovps YMMWORD PTR [esi],ymm0,ymm5
        jmp     ExitKernel

stdENDP _MlasSgemmKernel&Mode&Avx

        ENDM

SgemmKernelAvxFunction Zero
SgemmKernelAvxFunction Add

_TEXT   ENDS

        END
