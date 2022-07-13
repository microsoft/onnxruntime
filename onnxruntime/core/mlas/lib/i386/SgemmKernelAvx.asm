;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelAvx.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;   This implementation uses AVX instructions.
;
;--

        .686
        .xmm

        .xlist
INCLUDE mlasi.inc
INCLUDE SgemmKernelCommon.inc
        .list

        ASSUME  DS:FLAT,ES:FLAT,SS:NOTHING,FS:NOTHING,GS:NOTHING

        EXTERN  _MlasMaskMoveTableAvx:NEAR

_TEXT   SEGMENT DWORD PUBLIC 'CODE'

;
; Macro Description:
;
;   This macro multiplies and accumulates for a 16xN block of the output matrix.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
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

ComputeBlockAvxBy16 MACRO RowCount, VectorOffset, BroadcastOffset

IF RowCount EQ 1
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
; Macro Description:
;
;   This macro multiplies and accumulates for a 8xN block of the output matrix.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
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

ComputeBlockAvxBy8 MACRO RowCount, VectorOffset, BroadcastOffset

IF RowCount EQ 1
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
;   ymm4-ymm7 - Supplies the block accumulators.
;

ComputeBlockAvxLoop MACRO ComputeBlock, RowCount

        LOCAL   ComputeBlockBy4Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   OutputBlock

        sub     edi,4
        jb      ProcessRemainingBlocks

ComputeBlockBy4Loop:
        ComputeBlock RowCount, 0, 0
        ComputeBlock RowCount, 16*4, 4
        sub     edx,-32*4                   ; advance matrix B by 32 columns
        ComputeBlock RowCount, 0, 8
        ComputeBlock RowCount, 16*4, 12
        sub     edx,-32*4                   ; advance matrix B by 32 columns
        add     ecx,4*4                     ; advance matrix A by 4 columns
        sub     edi,4
        jae     ComputeBlockBy4Loop

ProcessRemainingBlocks:
        add     edi,4                       ; correct for over-subtract above
        jz      OutputBlock

ComputeBlockBy1Loop:
        ComputeBlock RowCount, 0, 0
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
;   Alpha - Supplies the scalar alpha multiplier (see SGEMM definition).
;
;   ZeroMode - Supplies true if the output matrix must be zero initialized,
;       else false if the output matrix is accumulated into.
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

cPublicProc _MlasGemmFloatKernelAvx,10

        SgemmKernelEntry

;
; Process 2 rows of the matrices.
;

        cmp     SgemmKernelFrame.CountM[esp],2
        jb      ProcessCountMLessThan2
        mov     BYTE PTR SgemmKernelFrame.CountM[esp],2
        mov     eax,SgemmKernelFrame.ldc[esp]
        mov     ebx,SgemmKernelFrame.lda[esp]
        shl     eax,2                       ; convert ldc to bytes
        shl     ebx,2                       ; convert lda to bytes
        cmp     ebp,8
        jbe     ProcessRemainingCountN2

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
        jb      OutputMasked16x2Block
        cmp     BYTE PTR SgemmKernelFrame.ZeroMode[esp],0
        jnz     SkipAccumulateOutput16x2
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,YMMWORD PTR [esi+32]
        vaddps  ymm6,ymm6,YMMWORD PTR [esi+eax]
        vaddps  ymm7,ymm7,YMMWORD PTR [esi+eax+32]

SkipAccumulateOutput16x2:
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+32],ymm5
        vmovups YMMWORD PTR [esi+eax],ymm6
        vmovups YMMWORD PTR [esi+eax+32],ymm7
        add     esi,16*4                    ; advance matrix C by 16 columns
        cmp     ebp,8
        ja      ProcessNextColumnLoop16x2
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
        jb      OutputMasked8x2Block
        cmp     BYTE PTR SgemmKernelFrame.ZeroMode[esp],0
        jnz     SkipAccumulateOutput8x2
        vaddps  ymm5,ymm5,YMMWORD PTR [esi]
        vaddps  ymm7,ymm7,YMMWORD PTR [esi+eax]

SkipAccumulateOutput8x2:
        vmovups YMMWORD PTR [esi],ymm5
        vmovups YMMWORD PTR [esi+eax],ymm7

;
; Restore non-volatile registers and return.
;

ExitKernel:
        movzx   eax,BYTE PTR SgemmKernelFrame.CountM[esp]
        vzeroupper
        SgemmKernelExit
        stdRET  _MlasGemmFloatKernelAvx

OutputMasked16x2Block:
        cmp     BYTE PTR SgemmKernelFrame.ZeroMode[esp],0
        jnz     SkipAccumulateMasked16x2Block
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm6,ymm6,YMMWORD PTR [esi+eax]

SkipAccumulateMasked16x2Block:
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+eax],ymm6
        add     esi,8*4                     ; advance matrix C by 8 columns
        add     ebp,8                       ; correct for over-subtract above

OutputMasked8x2Block:
        neg     ebp
        vmovdqu ymm0,YMMWORD PTR [_MlasMaskMoveTableAvx+ebp*4+8*4]
        cmp     BYTE PTR SgemmKernelFrame.ZeroMode[esp],0
        jnz     SkipAccumulateMasked8x2Block
        vmaskmovps ymm4,ymm0,YMMWORD PTR [esi]
        vmaskmovps ymm6,ymm0,YMMWORD PTR [esi+eax]
        vaddps  ymm5,ymm5,ymm4
        vaddps  ymm7,ymm7,ymm6

SkipAccumulateMasked8x2Block:
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
        jbe     ProcessRemainingCountN1

ProcessNextColumnLoop16x1:
        mov     edi,SgemmKernelFrame.CountK[esp]
        mov     ecx,ebx                     ; reload matrix A
        vxorps  xmm4,xmm4,xmm4              ; clear block accumulators
        vxorps  xmm5,xmm5,xmm5
        ComputeBlockAvxLoop ComputeBlockAvxBy16,1
        vmulps  ymm4,ymm4,ymm2              ; multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        sub     ebp,16
        jb      OutputMasked16x1Block
        cmp     BYTE PTR SgemmKernelFrame.ZeroMode[esp],0
        jnz     SkipAccumulate16x1Block
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,YMMWORD PTR [esi+32]

SkipAccumulate16x1Block:
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+32],ymm5
        add     esi,16*4                    ; advance matrix C by 16 columns
        cmp     ebp,8
        ja      ProcessNextColumnLoop16x1
        test    ebp,ebp
        jz      ExitKernel

ProcessRemainingCountN1:
        mov     edi,SgemmKernelFrame.CountK[esp]
        mov     ecx,ebx                     ; reload matrix A
        vxorps  xmm5,xmm5,xmm5              ; clear block accumulators
        ComputeBlockAvxLoop ComputeBlockAvxBy8,1
        vmulps  ymm5,ymm5,ymm2              ; multiply by alpha
        cmp     ebp,8
        jb      OutputMasked8x1Block
        cmp     BYTE PTR SgemmKernelFrame.ZeroMode[esp],0
        jnz     SkipAccumulate8x1Block
        vaddps  ymm5,ymm5,YMMWORD PTR [esi]

SkipAccumulate8x1Block:
        vmovups YMMWORD PTR [esi],ymm5
        jmp     ExitKernel

OutputMasked16x1Block:
        cmp     BYTE PTR SgemmKernelFrame.ZeroMode[esp],0
        jnz     SkipAccumulateMasked16x1Block
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]

SkipAccumulateMasked16x1Block:
        vmovups YMMWORD PTR [esi],ymm4
        add     esi,8*4                     ; advance matrix C by 8 columns
        add     ebp,8                       ; correct for over-subtract above

OutputMasked8x1Block:
        neg     ebp
        vmovdqu ymm0,YMMWORD PTR [_MlasMaskMoveTableAvx+ebp*4+8*4]
        cmp     BYTE PTR SgemmKernelFrame.ZeroMode[esp],0
        jnz     SkipAccumulateMasked8x1Block
        vmaskmovps ymm4,ymm0,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,ymm4

SkipAccumulateMasked8x1Block:
        vmaskmovps YMMWORD PTR [esi],ymm0,ymm5
        jmp     ExitKernel

stdENDP _MlasGemmFloatKernelAvx

_TEXT   ENDS

        END
