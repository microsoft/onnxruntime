;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   FgemmKernelAvxCommon.inc
;
; Abstract:
;
;   This module implements the kernels for the floating point matrix/matrix
;   multiply operation (SGEMM and DGEMM).
;
;   This implementation uses AVX instructions.
;
;--

        EXTERN  MlasMaskMoveTableAvx:NEAR

;
; Macro Description:
;
;   This macro multiplies and accumulates for 2 YMMWORDs by N rows of the output
;   matrix.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
;
;   VectorOffset - Supplies the byte offset from matrix B to fetch elements.
;
;   BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.
;
;   PrefetchOffset - Optionally supplies the byte offset from matrix B to
;       prefetch elements.
;
; Implicit Arguments:
;
;   rbx - Supplies the address into the matrix A data plus 2 rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   ymm8-ymm15 - Supplies the block accumulators.
;

ComputeBlockAvxBy2 MACRO RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

IF RowCount EQ 1
        vbroadcastsf ymm3,FgemmElementPtr [rcx+BroadcastOffset]
        vmulpf  ymm4,ymm3,YMMWORD PTR [rdx+VectorOffset]
        vaddpf  ymm8,ymm8,ymm4
        vmulpf  ymm5,ymm3,YMMWORD PTR [rdx+VectorOffset+32]
        vaddpf  ymm9,ymm9,ymm5
ELSE
        vmovapf ymm0,YMMWORD PTR [rdx+VectorOffset]
        vmovapf ymm1,YMMWORD PTR [rdx+VectorOffset+32]
        EmitIfCountGE RowCount, 1, <vbroadcastsf ymm3,FgemmElementPtr [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 1, <vmulpf ymm4,ymm3,ymm0>
        EmitIfCountGE RowCount, 1, <vaddpf ymm8,ymm8,ymm4>
        EmitIfCountGE RowCount, 1, <vmulpf ymm5,ymm3,ymm1>
        EmitIfCountGE RowCount, 1, <vaddpf ymm9,ymm9,ymm5>
        EmitIfCountGE RowCount, 2, <vbroadcastsf ymm3,FgemmElementPtr [rcx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <vmulpf ymm6,ymm3,ymm0>
        EmitIfCountGE RowCount, 2, <vaddpf ymm10,ymm10,ymm6>
        EmitIfCountGE RowCount, 2, <vmulpf ymm7,ymm3,ymm1>
        EmitIfCountGE RowCount, 2, <vaddpf ymm11,ymm11,ymm7>
        EmitIfCountGE RowCount, 3, <vbroadcastsf ymm3,FgemmElementPtr [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <vmulpf ymm4,ymm3,ymm0>
        EmitIfCountGE RowCount, 3, <vaddpf ymm12,ymm12,ymm4>
        EmitIfCountGE RowCount, 3, <vmulpf ymm5,ymm3,ymm1>
        EmitIfCountGE RowCount, 3, <vaddpf ymm13,ymm13,ymm5>
        EmitIfCountGE RowCount, 4, <vbroadcastsf ymm3,FgemmElementPtr [rbx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <vmulpf ymm6,ymm3,ymm0>
        EmitIfCountGE RowCount, 4, <vaddpf ymm14,ymm14,ymm6>
        EmitIfCountGE RowCount, 4, <vmulpf ymm7,ymm3,ymm1>
        EmitIfCountGE RowCount, 4, <vaddpf ymm15,ymm15,ymm7>
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro multiplies and accumulates for 1 YMMWORD by N rows of the output
;   matrix.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
;
;   VectorOffset - Supplies the byte offset from matrix B to fetch elements.
;
;   BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.
;
;   PrefetchOffset - Optionally supplies the byte offset from matrix B to
;       prefetch elements.
;
; Implicit Arguments:
;
;   rbx - Supplies the address into the matrix A data plus 2 rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   ymm8-ymm15 - Supplies the block accumulators.
;

ComputeBlockAvxBy1 MACRO RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

IF RowCount EQ 1
        vbroadcastsf ymm3,FgemmElementPtr [rcx+BroadcastOffset]
        vmulpf  ymm5,ymm3,YMMWORD PTR [rdx+VectorOffset]
        vaddpf  ymm9,ymm9,ymm5
ELSE
        vmovapf ymm0,YMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE RowCount, 1, <vbroadcastsf ymm3,FgemmElementPtr [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 1, <vmulpf ymm5,ymm3,ymm0>
        EmitIfCountGE RowCount, 1, <vaddpf ymm9,ymm9,ymm5>
        EmitIfCountGE RowCount, 2, <vbroadcastsf ymm3,FgemmElementPtr [rcx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <vmulpf ymm7,ymm3,ymm0>
        EmitIfCountGE RowCount, 2, <vaddpf ymm11,ymm11,ymm7>
        EmitIfCountGE RowCount, 3, <vbroadcastsf ymm3,FgemmElementPtr [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <vmulpf ymm5,ymm3,ymm0>
        EmitIfCountGE RowCount, 3, <vaddpf ymm13,ymm13,ymm5>
        EmitIfCountGE RowCount, 4, <vbroadcastsf ymm3,FgemmElementPtr [rbx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <vmulpf ymm7,ymm3,ymm0>
        EmitIfCountGE RowCount, 4, <vaddpf ymm15,ymm15,ymm7>
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
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r9 - Supplies the number of columns from matrix A and the number of rows
;       from matrix B to iterate over.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockAvxLoop MACRO ComputeBlock, RowCount

IF RowCount GT 2
        lea     rbx,[rcx+r10*2]             ; compute matrix A plus 2 rows
ENDIF
        ComputeBlockLoop ComputeBlock, RowCount, <RowCount GT 2>
IF RowCount GT 2
        lea     rbx,[r8+rax*2]              ; compute matrix C plus 2 rows
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to compute matrix multiplication for a fixed set
;   of rows.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
;
;   Fallthrough - Supplies a non-blank value if the macro may fall through to
;       the ExitKernel label.
;
; Implicit Arguments:
;
;   rax - Supplies the length in bytes of a row from matrix C.
;
;   rcx - Supplies the address of matrix A.
;
;   rdx - Supplies the address of matrix B.
;
;   rsi - Supplies the address of matrix A.
;
;   rbp - Supplies the number of columns from matrix B and matrix C to iterate
;       over.
;
;   r8 - Supplies the address of matrix C.
;
;   r9 - Supplies the number of columns from matrix A and the number of rows
;       from matrix B to iterate over.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   r15 - Stores the ZeroMode argument from the stack frame.
;

ProcessCountM MACRO RowCount, Fallthrough

        LOCAL   ProcessNextColumnLoop2xN
        LOCAL   Store2xNBlock
        LOCAL   ProcessRemainingCountN
        LOCAL   Store1xNBlock
        LOCAL   OutputMasked2xNBlock
        LOCAL   StoreMasked2xNBlock
        LOCAL   OutputMasked1xNBlock
        LOCAL   StoreMasked1xNBlock

        cmp     rbp,FgemmYmmElementCount
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop2xN:
        EmitIfCountGE RowCount, 1, <vxorpf xmm8,xmm8,xmm8>
        EmitIfCountGE RowCount, 1, <vxorpf xmm9,xmm9,xmm9>
        EmitIfCountGE RowCount, 2, <vxorpf xmm10,xmm10,xmm10>
        EmitIfCountGE RowCount, 2, <vxorpf xmm11,xmm11,xmm11>
        EmitIfCountGE RowCount, 3, <vxorpf xmm12,xmm12,xmm12>
        EmitIfCountGE RowCount, 3, <vxorpf xmm13,xmm13,xmm13>
        EmitIfCountGE RowCount, 4, <vxorpf xmm14,xmm14,xmm14>
        EmitIfCountGE RowCount, 4, <vxorpf xmm15,xmm15,xmm15>
        ComputeBlockAvxLoop ComputeBlockAvxBy2, RowCount
        EmitIfCountGE RowCount, 1, <vmulpf ymm8,ymm8,ymm2>
        EmitIfCountGE RowCount, 1, <vmulpf ymm9,ymm9,ymm2>
        EmitIfCountGE RowCount, 2, <vmulpf ymm10,ymm10,ymm2>
        EmitIfCountGE RowCount, 2, <vmulpf ymm11,ymm11,ymm2>
        EmitIfCountGE RowCount, 3, <vmulpf ymm12,ymm12,ymm2>
        EmitIfCountGE RowCount, 3, <vmulpf ymm13,ymm13,ymm2>
        EmitIfCountGE RowCount, 4, <vmulpf ymm14,ymm14,ymm2>
        EmitIfCountGE RowCount, 4, <vmulpf ymm15,ymm15,ymm2>
        sub     rbp,2*FgemmYmmElementCount
        jb      OutputMasked2xNBlock
        test    r15b,r15b                   ; ZeroMode?
        jnz     Store2xNBlock
        EmitIfCountGE RowCount, 1, <vaddpf ymm8,ymm8,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 1, <vaddpf ymm9,ymm9,YMMWORD PTR [r8+32]>
        EmitIfCountGE RowCount, 2, <vaddpf ymm10,ymm10,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 2, <vaddpf ymm11,ymm11,YMMWORD PTR [r8+rax+32]>
        EmitIfCountGE RowCount, 3, <vaddpf ymm12,ymm12,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 3, <vaddpf ymm13,ymm13,YMMWORD PTR [rbx+32]>
        EmitIfCountGE RowCount, 4, <vaddpf ymm14,ymm14,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 4, <vaddpf ymm15,ymm15,YMMWORD PTR [rbx+rax+32]>

Store2xNBlock:
        EmitIfCountGE RowCount, 1, <vmovupf YMMWORD PTR [r8],ymm8>
        EmitIfCountGE RowCount, 1, <vmovupf YMMWORD PTR [r8+32],ymm9>
        EmitIfCountGE RowCount, 2, <vmovupf YMMWORD PTR [r8+rax],ymm10>
        EmitIfCountGE RowCount, 2, <vmovupf YMMWORD PTR [r8+rax+32],ymm11>
        EmitIfCountGE RowCount, 3, <vmovupf YMMWORD PTR [rbx],ymm12>
        EmitIfCountGE RowCount, 3, <vmovupf YMMWORD PTR [rbx+32],ymm13>
        EmitIfCountGE RowCount, 4, <vmovupf YMMWORD PTR [rbx+rax],ymm14>
        EmitIfCountGE RowCount, 4, <vmovupf YMMWORD PTR [rbx+rax+32],ymm15>
        add     r8,2*32                     ; advance matrix C by 2 YMMWORDs
        mov     rcx,rsi                     ; reload matrix A
        cmp     rbp,FgemmYmmElementCount
        ja      ProcessNextColumnLoop2xN
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN:
        EmitIfCountGE RowCount, 1, <vxorpf xmm9,xmm9,xmm9>
        EmitIfCountGE RowCount, 2, <vxorpf xmm11,xmm11,xmm11>
        EmitIfCountGE RowCount, 3, <vxorpf xmm13,xmm13,xmm13>
        EmitIfCountGE RowCount, 4, <vxorpf xmm15,xmm15,xmm15>
        ComputeBlockAvxLoop ComputeBlockAvxBy1, RowCount
        EmitIfCountGE RowCount, 1, <vmulpf ymm9,ymm9,ymm2>
        EmitIfCountGE RowCount, 2, <vmulpf ymm11,ymm11,ymm2>
        EmitIfCountGE RowCount, 3, <vmulpf ymm13,ymm13,ymm2>
        EmitIfCountGE RowCount, 4, <vmulpf ymm15,ymm15,ymm2>
        cmp     rbp,FgemmYmmElementCount
        jb      OutputMasked1xNBlock
        test    r15b,r15b                   ; ZeroMode?
        jnz     Store1xNBlock
        EmitIfCountGE RowCount, 1, <vaddpf ymm9,ymm9,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vaddpf ymm11,ymm11,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vaddpf ymm13,ymm13,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 4, <vaddpf ymm15,ymm15,YMMWORD PTR [rbx+rax]>

Store1xNBlock:
        EmitIfCountGE RowCount, 1, <vmovupf YMMWORD PTR [r8],ymm9>
        EmitIfCountGE RowCount, 2, <vmovupf YMMWORD PTR [r8+rax],ymm11>
        EmitIfCountGE RowCount, 3, <vmovupf YMMWORD PTR [rbx],ymm13>
        EmitIfCountGE RowCount, 4, <vmovupf YMMWORD PTR [rbx+rax],ymm15>
        jmp     ExitKernel

OutputMasked2xNBlock:
        test    r15b,r15b                   ; ZeroMode?
        jnz     StoreMasked2xNBlock
        EmitIfCountGE RowCount, 1, <vaddpf ymm8,ymm8,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vaddpf ymm10,ymm10,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vaddpf ymm12,ymm12,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 4, <vaddpf ymm14,ymm14,YMMWORD PTR [rbx+rax]>

StoreMasked2xNBlock:
        EmitIfCountGE RowCount, 1, <vmovupf YMMWORD PTR [r8],ymm8>
        EmitIfCountGE RowCount, 2, <vmovupf YMMWORD PTR [r8+rax],ymm10>
        EmitIfCountGE RowCount, 3, <vmovupf YMMWORD PTR [rbx],ymm12>
        EmitIfCountGE RowCount, 4, <vmovupf YMMWORD PTR [rbx+rax],ymm14>
        add     r8,32                       ; advance matrix C by YMMWORD
IF RowCount GT 2
        add     rbx,32                      ; advance matrix C plus 2 rows by YMMWORD
ENDIF
        add     rbp,FgemmYmmElementCount    ; correct for over-subtract above

OutputMasked1xNBlock:
        neg     rbp
        lea     rcx,MlasMaskMoveTableAvx+8*4
        vmovdqu ymm0,YMMWORD PTR [rcx+rbp*FgemmElementSize]
        test    r15b,r15b                   ; ZeroMode?
        jnz     StoreMasked1xNBlock
        EmitIfCountGE RowCount, 1, <vmaskmovpf ymm8,ymm0,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vmaskmovpf ymm10,ymm0,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vmaskmovpf ymm12,ymm0,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 4, <vmaskmovpf ymm14,ymm0,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 1, <vaddpf ymm9,ymm9,ymm8>
        EmitIfCountGE RowCount, 2, <vaddpf ymm11,ymm11,ymm10>
        EmitIfCountGE RowCount, 3, <vaddpf ymm13,ymm13,ymm12>
        EmitIfCountGE RowCount, 4, <vaddpf ymm15,ymm15,ymm14>

StoreMasked1xNBlock:
        EmitIfCountGE RowCount, 1, <vmaskmovpf YMMWORD PTR [r8],ymm0,ymm9>
        EmitIfCountGE RowCount, 2, <vmaskmovpf YMMWORD PTR [r8+rax],ymm0,ymm11>
        EmitIfCountGE RowCount, 3, <vmaskmovpf YMMWORD PTR [rbx],ymm0,ymm13>
        EmitIfCountGE RowCount, 4, <vmaskmovpf YMMWORD PTR [rbx+rax],ymm0,ymm15>
IFB <Fallthrough>
        jmp     ExitKernel
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates the inner kernel to compute matrix multiplication.
;
; Arguments:
;
;   Type - Supplies the element type string for function tags.
;

FgemmKernelAvxFunction MACRO Type

;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows.
;
; Arguments:
;
;   A (rcx) - Supplies the address of matrix A.
;
;   B (rdx) - Supplies the address of matrix B. The matrix data has been packed
;       using MlasSgemmCopyPackB or MlasSgemmTransposePackB.
;
;   C (r8) - Supplies the address of matrix C.
;
;   CountK (r9) - Supplies the number of columns from matrix A and the number
;       of rows from matrix B to iterate over.
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

        NESTED_ENTRY MlasGemm&Type&KernelAvx, _TEXT

        FgemmKernelEntry Avx

        vbroadcastsf ymm2,FgemmElementPtr FgemmKernelFrame.Alpha[rsp]

;
; Process 4 rows of the matrices.
;

        cmp     r11,4
        jb      ProcessCountMLessThan4
        mov     r11d,4                      ; return 4 rows handled
        ProcessCountM 4, Fallthrough

;
; Restore non-volatile registers and return.
;

ExitKernel:
        vzeroupper
        FgemmKernelExit Avx

;
; Process 2 rows of the matrices.
;

ProcessCountMLessThan4:
        cmp     r11,2
        jb      ProcessCountMLessThan2
        mov     r11d,2                      ; return 2 rows handled
        ProcessCountM 2

;
; Process 1 row of the matrices.
;

ProcessCountMLessThan2:
        ProcessCountM 1

        NESTED_END MlasGemm&Type&KernelAvx, _TEXT

        ENDM
