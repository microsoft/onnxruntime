/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelSse2Common.h

Abstract:

    This module implements the kernels for the floating point matrix/matrix
    multiply operation (SGEMM and DGEMM).

    This implementation uses SSE2 instructions.

--*/

/*++

Macro Description:

    This macro generates the inner kernel to compute matrix multiplication.

Arguments:

    FunctionName - Supplies the name for the generated function.

--*/

        .macro FgemmKernelSse2Function FunctionName

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

    Alpha (xmm0) - Supplies the scalar alpha multiplier (see GEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        .globl  \FunctionName\()
\FunctionName\():

        push    rbp
        push    rbx
        push    r15
        mov     r11,rdi
        mov     r10,.LFgemmKernelFrame_lda[rsp]
        shl     r10,.LFgemmElementShift     # convert lda to bytes
        mov     rax,.LFgemmKernelFrame_ldc[rsp]
        shl     rax,.LFgemmElementShift     # convert ldc to bytes
        movzx   r15,BYTE PTR .LFgemmKernelFrame_ZeroMode[rsp]
        movsf   .LFgemmKernelFrame_alpha[rsp],xmm0

//
// Process CountM rows of the matrices.
//

        cmp     r8,2
        jb      .LProcessCountM1
        mov     r8d,2                       # return 2 rows handled
        ProcessCountM 2, Fallthrough

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        mov     eax,r8d
        pop     r15
        pop     rbx
        pop     rbp
        ret

//
// Process 1 row of the matrices.
//

.LProcessCountM1:
        ProcessCountM 1

        .endm
