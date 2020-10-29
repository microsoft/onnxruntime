;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   FgemmKernelSse2Common.inc
;
; Abstract:
;
;   This module implements the kernels for the floating point matrix/matrix
;   multiply operation (SGEMM and DGEMM).
;
;   This implementation uses SSE2 instructions.
;
;--

;
; Macro Description:
;
;   This stores the block accumulators to the output matrix with an optional
;   accumulation of the existing contents of the output matrix.
;
; Arguments:
;
;   RowCount - Supplies the number of rows to process.
;
;   VectorCount - Supplies the number of vector columns to process.
;
; Implicit Arguments:
;
;   rax - Supplies the length in bytes of a row from matrix C.
;
;   r8 - Supplies the address of matrix C.
;
;   r15 - Stores the ZeroMode argument from the stack frame.
;
;   xmm8-xmm15 - Supplies the block accumulators.
;

AccumulateAndStoreBlock MACRO RowCount, VectorCount

        LOCAL   SkipAccumulateOutput

        test    r15b,r15b                   ; ZeroMode?
        jnz     SkipAccumulateOutput
        EmitIfCount2GE RowCount, 1, VectorCount, 1, <movupf xmm0,XMMWORD PTR [r8]>
        EmitIfCount2GE RowCount, 1, VectorCount, 2, <movupf xmm1,XMMWORD PTR [r8+16]>
        EmitIfCount2GE RowCount, 1, VectorCount, 3, <movupf xmm2,XMMWORD PTR [r8+32]>
        EmitIfCount2GE RowCount, 1, VectorCount, 4, <movupf xmm3,XMMWORD PTR [r8+48]>
        EmitIfCount2GE RowCount, 2, VectorCount, 1, <movupf xmm4,XMMWORD PTR [r8+rax]>
        EmitIfCount2GE RowCount, 2, VectorCount, 2, <movupf xmm5,XMMWORD PTR [r8+rax+16]>
        EmitIfCount2GE RowCount, 2, VectorCount, 3, <movupf xmm6,XMMWORD PTR [r8+rax+32]>
        EmitIfCount2GE RowCount, 2, VectorCount, 4, <movupf xmm7,XMMWORD PTR [r8+rax+48]>
        EmitIfCount2GE RowCount, 1, VectorCount, 1, <addpf xmm8,xmm0>
        EmitIfCount2GE RowCount, 1, VectorCount, 2, <addpf xmm9,xmm1>
        EmitIfCount2GE RowCount, 1, VectorCount, 3, <addpf xmm10,xmm2>
        EmitIfCount2GE RowCount, 1, VectorCount, 4, <addpf xmm11,xmm3>
        EmitIfCount2GE RowCount, 2, VectorCount, 1, <addpf xmm12,xmm4>
        EmitIfCount2GE RowCount, 2, VectorCount, 2, <addpf xmm13,xmm5>
        EmitIfCount2GE RowCount, 2, VectorCount, 3, <addpf xmm14,xmm6>
        EmitIfCount2GE RowCount, 2, VectorCount, 4, <addpf xmm15,xmm7>

SkipAccumulateOutput:
        EmitIfCount2GE RowCount, 1, VectorCount, 1, <movupf XMMWORD PTR [r8],xmm8>
        EmitIfCount2GE RowCount, 1, VectorCount, 2, <movupf XMMWORD PTR [r8+16],xmm9>
        EmitIfCount2GE RowCount, 1, VectorCount, 3, <movupf XMMWORD PTR [r8+32],xmm10>
        EmitIfCount2GE RowCount, 1, VectorCount, 4, <movupf XMMWORD PTR [r8+48],xmm11>
        EmitIfCount2GE RowCount, 2, VectorCount, 1, <movupf XMMWORD PTR [r8+rax],xmm12>
        EmitIfCount2GE RowCount, 2, VectorCount, 2, <movupf XMMWORD PTR [r8+rax+16],xmm13>
        EmitIfCount2GE RowCount, 2, VectorCount, 3, <movupf XMMWORD PTR [r8+rax+32],xmm14>
        EmitIfCount2GE RowCount, 2, VectorCount, 4, <movupf XMMWORD PTR [r8+rax+48],xmm15>

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

FgemmKernelSse2Function MACRO Type

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
;   CountK (r9d) - Supplies the number of columns from matrix A and the number
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

        NESTED_ENTRY MlasGemm&Type&KernelSse, _TEXT

        FgemmKernelEntry Sse

;
; Process CountM rows of the matrices.
;

        cmp     r11,2
        jb      ProcessCountM1
        mov     r11d,2                      ; return 2 rows handled
        ProcessCountM 2, Fallthrough

;
; Restore non-volatile registers and return.
;

ExitKernel:
        FgemmKernelExit Sse

ProcessCountM1:
        ProcessCountM 1

        NESTED_END MlasGemm&Type&KernelSse, _TEXT

        ENDM
