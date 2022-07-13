;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   FgemmKernelFma3Common.inc
;
; Abstract:
;
;   This module implements the kernels for the floating point matrix/matrix
;   multiply operation (SGEMM and DGEMM).
;
;   This implementation uses AVX fused multiply/add instructions.
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
;   rbx - Supplies the address into the matrix A data plus 3 rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockFma3By2 MACRO RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

IFNB <PrefetchOffset>
        prefetcht0 [rdx+VectorOffset+PrefetchOffset]
ENDIF
IF RowCount EQ 1
        vbroadcastsf ymm3,FgemmElementPtr [rcx+BroadcastOffset]
        vfmadd231pf ymm4,ymm3,YMMWORD PTR [rdx+VectorOffset]
        vfmadd231pf ymm5,ymm3,YMMWORD PTR [rdx+VectorOffset+32]
ELSE
        vmovapf ymm0,YMMWORD PTR [rdx+VectorOffset]
        vmovapf ymm1,YMMWORD PTR [rdx+VectorOffset+32]
        EmitIfCountGE RowCount, 1, <vbroadcastsf ymm3,FgemmElementPtr [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 1, <vfmadd231pf ymm4,ymm3,ymm0>
        EmitIfCountGE RowCount, 1, <vfmadd231pf ymm5,ymm3,ymm1>
        EmitIfCountGE RowCount, 2, <vbroadcastsf ymm3,FgemmElementPtr [rcx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <vfmadd231pf ymm6,ymm3,ymm0>
        EmitIfCountGE RowCount, 2, <vfmadd231pf ymm7,ymm3,ymm1>
        EmitIfCountGE RowCount, 3, <vbroadcastsf ymm3,FgemmElementPtr [rcx+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <vfmadd231pf ymm8,ymm3,ymm0>
        EmitIfCountGE RowCount, 3, <vfmadd231pf ymm9,ymm3,ymm1>
        EmitIfCountGE RowCount, 4, <vbroadcastsf ymm3,FgemmElementPtr [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <vfmadd231pf ymm10,ymm3,ymm0>
        EmitIfCountGE RowCount, 4, <vfmadd231pf ymm11,ymm3,ymm1>
        EmitIfCountGE RowCount, 5, <vbroadcastsf ymm3,FgemmElementPtr [rbx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 5, <vfmadd231pf ymm12,ymm3,ymm0>
        EmitIfCountGE RowCount, 5, <vfmadd231pf ymm13,ymm3,ymm1>
        EmitIfCountGE RowCount, 6, <vbroadcastsf ymm3,FgemmElementPtr [rbx+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 6, <vfmadd231pf ymm14,ymm3,ymm0>
        EmitIfCountGE RowCount, 6, <vfmadd231pf ymm15,ymm3,ymm1>
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
;   rbx - Supplies the address into the matrix A data plus 3 rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockFma3By1 MACRO RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

IFNB <PrefetchOffset>
        prefetcht0 [rdx+VectorOffset+PrefetchOffset]
ENDIF
IF RowCount EQ 1
        vbroadcastsf ymm3,FgemmElementPtr [rcx+BroadcastOffset]
        vfmadd231pf ymm5,ymm3,YMMWORD PTR [rdx+VectorOffset]
ELSE
        vmovapf ymm0,YMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE RowCount, 1, <vbroadcastsf ymm3,FgemmElementPtr [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 1, <vfmadd231pf ymm5,ymm3,ymm0>
        EmitIfCountGE RowCount, 2, <vbroadcastsf ymm3,FgemmElementPtr [rcx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <vfmadd231pf ymm7,ymm3,ymm0>
        EmitIfCountGE RowCount, 3, <vbroadcastsf ymm3,FgemmElementPtr [rcx+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <vfmadd231pf ymm9,ymm3,ymm0>
        EmitIfCountGE RowCount, 4, <vbroadcastsf ymm3,FgemmElementPtr [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <vfmadd231pf ymm11,ymm3,ymm0>
        EmitIfCountGE RowCount, 5, <vbroadcastsf ymm3,FgemmElementPtr [rbx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 5, <vfmadd231pf ymm13,ymm3,ymm0>
        EmitIfCountGE RowCount, 6, <vbroadcastsf ymm3,FgemmElementPtr [rbx+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 6, <vfmadd231pf ymm15,ymm3,ymm0>
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

ComputeBlockFma3Loop MACRO ComputeBlock, RowCount

IF RowCount GT 3
        lea     rbx,[r10*2+r10]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF
        ComputeBlockLoop ComputeBlock, RowCount, <RowCount GT 3>
        vbroadcastsf ymm2,FgemmElementPtr FgemmKernelFrame.Alpha[rsp]
IF RowCount GT 3
        lea     rbx,[rax*2+rax]
        add     rbx,r8                      ; compute matrix C plus 3 rows
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
;       the ExitKernelAndZeroUpper label.
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
        LOCAL   MultiplyAlpha2xNBlock
        LOCAL   Store2xNBlock
        LOCAL   ProcessRemainingCountN
        LOCAL   MultiplyAlpha1xNBlock
        LOCAL   Store1xNBlock
        LOCAL   OutputMasked2xNBlock
        LOCAL   MultiplyAlphaMasked2xNBlock
        LOCAL   StoreMasked2xNBlock
        LOCAL   OutputMasked1xNBlock
        LOCAL   MultiplyAlphaMasked1xNBlock
        LOCAL   StoreMasked1xNBlock

        cmp     rbp,FgemmYmmElementCount
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop2xN:
        ComputeBlockFma3Loop ComputeBlockFma3By2, RowCount
        EmitIfCountGE RowCount, 1, <prefetcht0 [r8+64]>
        EmitIfCountGE RowCount, 2, <prefetcht0 [r8+rax+64]>
        EmitIfCountGE RowCount, 3, <prefetcht0 [r8+rax*2+64]>
        EmitIfCountGE RowCount, 4, <prefetcht0 [rbx+64]>
        EmitIfCountGE RowCount, 5, <prefetcht0 [rbx+rax+64]>
        EmitIfCountGE RowCount, 6, <prefetcht0 [rbx+rax*2+64]>
        sub     rbp,2*FgemmYmmElementCount
        jb      OutputMasked2xNBlock
        test    r15b,r15b                   ; ZeroMode?
        jnz     MultiplyAlpha2xNBlock
        EmitIfCountGE RowCount, 1, <vfmadd213pf ymm4,ymm2,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 1, <vfmadd213pf ymm5,ymm2,YMMWORD PTR [r8+32]>
        EmitIfCountGE RowCount, 2, <vfmadd213pf ymm6,ymm2,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 2, <vfmadd213pf ymm7,ymm2,YMMWORD PTR [r8+rax+32]>
        EmitIfCountGE RowCount, 3, <vfmadd213pf ymm8,ymm2,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 3, <vfmadd213pf ymm9,ymm2,YMMWORD PTR [r8+rax*2+32]>
        EmitIfCountGE RowCount, 4, <vfmadd213pf ymm10,ymm2,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 4, <vfmadd213pf ymm11,ymm2,YMMWORD PTR [rbx+32]>
        EmitIfCountGE RowCount, 5, <vfmadd213pf ymm12,ymm2,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 5, <vfmadd213pf ymm13,ymm2,YMMWORD PTR [rbx+rax+32]>
        EmitIfCountGE RowCount, 6, <vfmadd213pf ymm14,ymm2,YMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE RowCount, 6, <vfmadd213pf ymm15,ymm2,YMMWORD PTR [rbx+rax*2+32]>
        jmp     Store2xNBlock

MultiplyAlpha2xNBlock:
        EmitIfCountGE RowCount, 1, <vmulpf ymm4,ymm4,ymm2>
        EmitIfCountGE RowCount, 1, <vmulpf ymm5,ymm5,ymm2>
        EmitIfCountGE RowCount, 2, <vmulpf ymm6,ymm6,ymm2>
        EmitIfCountGE RowCount, 2, <vmulpf ymm7,ymm7,ymm2>
        EmitIfCountGE RowCount, 3, <vmulpf ymm8,ymm8,ymm2>
        EmitIfCountGE RowCount, 3, <vmulpf ymm9,ymm9,ymm2>
        EmitIfCountGE RowCount, 4, <vmulpf ymm10,ymm10,ymm2>
        EmitIfCountGE RowCount, 4, <vmulpf ymm11,ymm11,ymm2>
        EmitIfCountGE RowCount, 5, <vmulpf ymm12,ymm12,ymm2>
        EmitIfCountGE RowCount, 5, <vmulpf ymm13,ymm13,ymm2>
        EmitIfCountGE RowCount, 6, <vmulpf ymm14,ymm14,ymm2>
        EmitIfCountGE RowCount, 6, <vmulpf ymm15,ymm15,ymm2>

Store2xNBlock:
        EmitIfCountGE RowCount, 1, <vmovupf YMMWORD PTR [r8],ymm4>
        EmitIfCountGE RowCount, 1, <vmovupf YMMWORD PTR [r8+32],ymm5>
        EmitIfCountGE RowCount, 2, <vmovupf YMMWORD PTR [r8+rax],ymm6>
        EmitIfCountGE RowCount, 2, <vmovupf YMMWORD PTR [r8+rax+32],ymm7>
        EmitIfCountGE RowCount, 3, <vmovupf YMMWORD PTR [r8+rax*2],ymm8>
        EmitIfCountGE RowCount, 3, <vmovupf YMMWORD PTR [r8+rax*2+32],ymm9>
        EmitIfCountGE RowCount, 4, <vmovupf YMMWORD PTR [rbx],ymm10>
        EmitIfCountGE RowCount, 4, <vmovupf YMMWORD PTR [rbx+32],ymm11>
        EmitIfCountGE RowCount, 5, <vmovupf YMMWORD PTR [rbx+rax],ymm12>
        EmitIfCountGE RowCount, 5, <vmovupf YMMWORD PTR [rbx+rax+32],ymm13>
        EmitIfCountGE RowCount, 6, <vmovupf YMMWORD PTR [rbx+rax*2],ymm14>
        EmitIfCountGE RowCount, 6, <vmovupf YMMWORD PTR [rbx+rax*2+32],ymm15>
        add     r8,2*32                     ; advance matrix C by 2 YMMWORDs
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,FgemmYmmElementCount
        ja      ProcessNextColumnLoop2xN
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN:
        ComputeBlockFma3Loop ComputeBlockFma3By1, RowCount
        cmp     rbp,FgemmYmmElementCount
        jb      OutputMasked1xNBlock
        test    r15b,r15b                   ; ZeroMode?
        jnz     MultiplyAlpha1xNBlock
        EmitIfCountGE RowCount, 1, <vfmadd213pf ymm5,ymm2,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vfmadd213pf ymm7,ymm2,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vfmadd213pf ymm9,ymm2,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vfmadd213pf ymm11,ymm2,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vfmadd213pf ymm13,ymm2,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vfmadd213pf ymm15,ymm2,YMMWORD PTR [rbx+rax*2]>
        jmp     Store1xNBlock

MultiplyAlpha1xNBlock:
        EmitIfCountGE RowCount, 1, <vmulpf ymm5,ymm5,ymm2>
        EmitIfCountGE RowCount, 2, <vmulpf ymm7,ymm7,ymm2>
        EmitIfCountGE RowCount, 3, <vmulpf ymm9,ymm9,ymm2>
        EmitIfCountGE RowCount, 4, <vmulpf ymm11,ymm11,ymm2>
        EmitIfCountGE RowCount, 5, <vmulpf ymm13,ymm13,ymm2>
        EmitIfCountGE RowCount, 6, <vmulpf ymm15,ymm15,ymm2>

Store1xNBlock:
        EmitIfCountGE RowCount, 1, <vmovupf YMMWORD PTR [r8],ymm5>
        EmitIfCountGE RowCount, 2, <vmovupf YMMWORD PTR [r8+rax],ymm7>
        EmitIfCountGE RowCount, 3, <vmovupf YMMWORD PTR [r8+rax*2],ymm9>
        EmitIfCountGE RowCount, 4, <vmovupf YMMWORD PTR [rbx],ymm11>
        EmitIfCountGE RowCount, 5, <vmovupf YMMWORD PTR [rbx+rax],ymm13>
        EmitIfCountGE RowCount, 6, <vmovupf YMMWORD PTR [rbx+rax*2],ymm15>
        jmp     ExitKernelAndZeroUpper

OutputMasked2xNBlock:
        test    r15b,r15b                   ; ZeroMode?
        jnz     MultiplyAlphaMasked2xNBlock
        EmitIfCountGE RowCount, 1, <vfmadd213pf ymm4,ymm2,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vfmadd213pf ymm6,ymm2,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vfmadd213pf ymm8,ymm2,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vfmadd213pf ymm10,ymm2,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vfmadd213pf ymm12,ymm2,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vfmadd213pf ymm14,ymm2,YMMWORD PTR [rbx+rax*2]>
        jmp     StoreMasked2xNBlock

MultiplyAlphaMasked2xNBlock:
        EmitIfCountGE RowCount, 1, <vmulpf ymm4,ymm4,ymm2>
        EmitIfCountGE RowCount, 2, <vmulpf ymm6,ymm6,ymm2>
        EmitIfCountGE RowCount, 3, <vmulpf ymm8,ymm8,ymm2>
        EmitIfCountGE RowCount, 4, <vmulpf ymm10,ymm10,ymm2>
        EmitIfCountGE RowCount, 5, <vmulpf ymm12,ymm12,ymm2>
        EmitIfCountGE RowCount, 6, <vmulpf ymm14,ymm14,ymm2>

StoreMasked2xNBlock:
        EmitIfCountGE RowCount, 1, <vmovupf YMMWORD PTR [r8],ymm4>
        EmitIfCountGE RowCount, 2, <vmovupf YMMWORD PTR [r8+rax],ymm6>
        EmitIfCountGE RowCount, 3, <vmovupf YMMWORD PTR [r8+rax*2],ymm8>
        EmitIfCountGE RowCount, 4, <vmovupf YMMWORD PTR [rbx],ymm10>
        EmitIfCountGE RowCount, 5, <vmovupf YMMWORD PTR [rbx+rax],ymm12>
        EmitIfCountGE RowCount, 6, <vmovupf YMMWORD PTR [rbx+rax*2],ymm14>
        add     r8,32                       ; advance matrix C by YMMWORD
IF RowCount GT 3
        add     rbx,32                      ; advance matrix C plus 3 rows by YMMWORD
ENDIF
        add     rbp,FgemmYmmElementCount    ; correct for over-subtract above

OutputMasked1xNBlock:
        neg     rbp
        lea     rcx,MlasMaskMoveTableAvx+8*4
        vmovdqu ymm0,YMMWORD PTR [rcx+rbp*FgemmElementSize]
        test    r15b,r15b                   ; ZeroMode?
        jnz     MultiplyAlphaMasked1xNBlock
        EmitIfCountGE RowCount, 1, <vmaskmovpf ymm4,ymm0,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vmaskmovpf ymm6,ymm0,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vmaskmovpf ymm8,ymm0,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vmaskmovpf ymm10,ymm0,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vmaskmovpf ymm12,ymm0,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vmaskmovpf ymm14,ymm0,YMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE RowCount, 1, <vfmadd213pf ymm5,ymm2,ymm4>
        EmitIfCountGE RowCount, 2, <vfmadd213pf ymm7,ymm2,ymm6>
        EmitIfCountGE RowCount, 3, <vfmadd213pf ymm9,ymm2,ymm8>
        EmitIfCountGE RowCount, 4, <vfmadd213pf ymm11,ymm2,ymm10>
        EmitIfCountGE RowCount, 5, <vfmadd213pf ymm13,ymm2,ymm12>
        EmitIfCountGE RowCount, 6, <vfmadd213pf ymm15,ymm2,ymm14>
        jmp     StoreMasked1xNBlock

MultiplyAlphaMasked1xNBlock:
        EmitIfCountGE RowCount, 1, <vmulpf ymm5,ymm5,ymm2>
        EmitIfCountGE RowCount, 2, <vmulpf ymm7,ymm7,ymm2>
        EmitIfCountGE RowCount, 3, <vmulpf ymm9,ymm9,ymm2>
        EmitIfCountGE RowCount, 4, <vmulpf ymm11,ymm11,ymm2>
        EmitIfCountGE RowCount, 5, <vmulpf ymm13,ymm13,ymm2>
        EmitIfCountGE RowCount, 6, <vmulpf ymm15,ymm15,ymm2>

StoreMasked1xNBlock:
        EmitIfCountGE RowCount, 1, <vmaskmovpf YMMWORD PTR [r8],ymm0,ymm5>
        EmitIfCountGE RowCount, 2, <vmaskmovpf YMMWORD PTR [r8+rax],ymm0,ymm7>
        EmitIfCountGE RowCount, 3, <vmaskmovpf YMMWORD PTR [r8+rax*2],ymm0,ymm9>
        EmitIfCountGE RowCount, 4, <vmaskmovpf YMMWORD PTR [rbx],ymm0,ymm11>
        EmitIfCountGE RowCount, 5, <vmaskmovpf YMMWORD PTR [rbx+rax],ymm0,ymm13>
        EmitIfCountGE RowCount, 6, <vmaskmovpf YMMWORD PTR [rbx+rax*2],ymm0,ymm15>
IFB <Fallthrough>
        jmp     ExitKernelAndZeroUpper
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

FgemmKernelFma3Function MACRO Type

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
;   Alpha - Supplies the scalar alpha multiplier (see GEMM definition).
;
;   ZeroMode - Supplies true if the output matrix must be zero initialized,
;       else false if the output matrix is accumulated into.
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

        NESTED_ENTRY MlasGemm&Type&KernelFma3, _TEXT

        FgemmKernelEntry Fma3

;
; Process CountM rows of the matrices.
;

        cmp     r11,5
        ja      ProcessCountM6
        je      ProcessCountM5
        cmp     r11,3
        ja      ProcessCountM4
        je      ProcessCountM3
        cmp     r11,1
        je      ProcessCountM1

ProcessCountM2:
        ProcessCountM 2

ProcessCountM4:
        ProcessCountM 4

ProcessCountM6:
        mov     r11d,6                      ; return 6 rows handled
        ProcessCountM 6, Fallthrough

;
; Restore non-volatile registers and return.
;

ExitKernelAndZeroUpper:
        vzeroupper

ExitKernel:
        FgemmKernelExit Fma3

ProcessCountM1:
        ProcessCountM 1

ProcessCountM3:
        ProcessCountM 3

ProcessCountM5:
        ProcessCountM 5

        NESTED_END MlasGemm&Type&KernelFma3, _TEXT

        ENDM
