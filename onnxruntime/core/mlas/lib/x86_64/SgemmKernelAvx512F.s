/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelAvx512F.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses AVX512F instructions.

--*/

        .intel_syntax noprefix

        .equ    SgemmKernelFrame_alpha, -8
        .equ    SgemmKernelFrame_SavedR14, 0
        .equ    SgemmKernelFrame_SavedR13, 8
        .equ    SgemmKernelFrame_SavedR12, 16
        .equ    SgemmKernelFrame_SavedRbx, 24
        .equ    SgemmKernelFrame_SavedRbp, 32
        .equ    SgemmKernelFrame_ReturnAddress, 40
        .equ    SgemmKernelFrame_lda, 48
        .equ    SgemmKernelFrame_ldc, 56

        .text

/*++

Macro Description:

    This macro multiplies and accumulates for a 32xN block (where N is 1,3,6,12)
    of the output matrix.

Arguments:

    Count - Supplies the number of rows to access from matrix A.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    rcx - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    r13 - Supplies the address into the matrix A data plus 6 rows.

    r14 - Supplies the address into the matrix A data plus 9 rows.

    rdx - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    zmm4-zmm27 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvx512FBy32 Count, VectorOffset, BroadcastOffset

.if \Count\() == 1
        vbroadcastss zmm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vfmadd231ps zmm4,zmm3,ZMMWORD PTR [rsi+\VectorOffset\()]
        vfmadd231ps zmm5,zmm3,ZMMWORD PTR [rsi+r12+\VectorOffset\()]
.else
        vmovaps zmm0,ZMMWORD PTR [rsi+\VectorOffset\()]
        vmovaps zmm1,ZMMWORD PTR [rsi+r12+\VectorOffset\()]
        vbroadcastss zmm3,DWORD PTR [rdi+\BroadcastOffset\()]
        vfmadd231ps zmm4,zmm3,zmm0
        vfmadd231ps zmm5,zmm3,zmm1
.if \Count\() >= 3
        vbroadcastss zmm3,DWORD PTR [rdi+r10+\BroadcastOffset\()]
        vfmadd231ps zmm6,zmm3,zmm0
        vfmadd231ps zmm7,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [rdi+r10*2+\BroadcastOffset\()]
        vfmadd231ps zmm8,zmm3,zmm0
        vfmadd231ps zmm9,zmm3,zmm1
.endif
.if \Count\() >= 6
        vbroadcastss zmm3,DWORD PTR [rbx+\BroadcastOffset\()]
        vfmadd231ps zmm10,zmm3,zmm0
        vfmadd231ps zmm11,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [rbx+r10+\BroadcastOffset\()]
        vfmadd231ps zmm12,zmm3,zmm0
        vfmadd231ps zmm13,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [rbx+r10*2+\BroadcastOffset\()]
        vfmadd231ps zmm14,zmm3,zmm0
        vfmadd231ps zmm15,zmm3,zmm1
.endif
.if \Count\() >= 12
        vbroadcastss zmm3,DWORD PTR [r13+\BroadcastOffset\()]
        vfmadd231ps zmm16,zmm3,zmm0
        vfmadd231ps zmm17,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [r13+r10+\BroadcastOffset\()]
        vfmadd231ps zmm18,zmm3,zmm0
        vfmadd231ps zmm19,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [r13+r10*2+\BroadcastOffset\()]
        vfmadd231ps zmm20,zmm3,zmm0
        vfmadd231ps zmm21,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [r14+\BroadcastOffset\()]
        vfmadd231ps zmm22,zmm3,zmm0
        vfmadd231ps zmm23,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [r14+r10+\BroadcastOffset\()]
        vfmadd231ps zmm24,zmm3,zmm0
        vfmadd231ps zmm25,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [r14+r10*2+\BroadcastOffset\()]
        vfmadd231ps zmm26,zmm3,zmm0
        vfmadd231ps zmm27,zmm3,zmm1
.endif
.endif

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for a 16xN block (where N is 1,3,6,12)
    of the output matrix.

Arguments:

    Count - Supplies the number of rows to access from matrix A.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    r13 - Supplies the address into the matrix A data plus 6 rows.

    r14 - Supplies the address into the matrix A data plus 9 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    zmm4-zmm27 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvx512FBy16 Count, VectorOffset, BroadcastOffset

        vmovaps zmm0,ZMMWORD PTR [rsi+\VectorOffset\()]
        vfmadd231ps zmm5,zmm0,DWORD PTR [rdi+\BroadcastOffset\()]{1to16}
.if \Count\() >= 3
        vfmadd231ps zmm7,zmm0,DWORD PTR [rdi+r10+\BroadcastOffset\()]{1to16}
        vfmadd231ps zmm9,zmm0,DWORD PTR [rdi+r10*2+\BroadcastOffset\()]{1to16}
.endif
.if \Count\() >= 6
        vfmadd231ps zmm11,zmm0,DWORD PTR [rbx+\BroadcastOffset\()]{1to16}
        vfmadd231ps zmm13,zmm0,DWORD PTR [rbx+r10+\BroadcastOffset\()]{1to16}
        vfmadd231ps zmm15,zmm0,DWORD PTR [rbx+r10*2+\BroadcastOffset\()]{1to16}
.endif
.if \Count\() >= 12
        vfmadd231ps zmm17,zmm0,DWORD PTR [r13+\BroadcastOffset\()]{1to16}
        vfmadd231ps zmm19,zmm0,DWORD PTR [r13+r10+\BroadcastOffset\()]{1to16}
        vfmadd231ps zmm21,zmm0,DWORD PTR [r13+r10*2+\BroadcastOffset\()]{1to16}
        vfmadd231ps zmm23,zmm0,DWORD PTR [r14+\BroadcastOffset\()]{1to16}
        vfmadd231ps zmm25,zmm0,DWORD PTR [r14+r10+\BroadcastOffset\()]{1to16}
        vfmadd231ps zmm27,zmm0,DWORD PTR [r14+r10*2+\BroadcastOffset\()]{1to16}
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

    r13 - Supplies the address into the matrix A data plus 6 rows.

    r14 - Supplies the address into the matrix A data plus 9 rows.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    zmm4-zmm27 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvx512FLoop Mode, ComputeBlock, Count

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
.if \Count\() == 12
        add     r13,4*4
        add     r14,4*4
.endif
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
.if \Count\() == 12
        add     r13,4
        add     r14,4
.endif
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

        .macro  SgemmKernelAvx512FFunction Mode

        .globl  MlasSgemmKernel\Mode\()Avx512F
MlasSgemmKernel\Mode\()Avx512F:

        push    rbp
        push    rbx
        push    r12
        push    r13
        push    r14
        mov     r11,rdi
        mov     r10,[rsp+SgemmKernelFrame_lda]
        shl     r10,2                       # convert lda to bytes
        mov     rax,[rsp+SgemmKernelFrame_ldc]
        shl     rax,2                       # convert ldc to bytes
        mov     r12,rcx
        shl     r12,6                       # compute 16*CountK*sizeof(float)
        mov     ebx,-1
        kmovw   k1,ebx                      # update mask to write all columns
        vmovss  DWORD PTR [rsp+SgemmKernelFrame_alpha],xmm0
        vzeroall
        vbroadcastss zmm31,DWORD PTR [rsp+SgemmKernelFrame_alpha]

//
// Process 12 rows of the matrices.
//

        cmp     r8,12
        jb      .L\Mode\().ProcessCountMLessThan12
        mov     r8d,12                      # return 12 rows handled
        cmp     r9,16
        jbe     .L\Mode\().ProcessRemainingCountN12

.L\Mode\().ProcessNextColumnLoop32x12:
        vmovaps zmm16,zmm4                  # clear upper block accumulators
        vmovaps zmm17,zmm5
        vmovaps zmm18,zmm4
        vmovaps zmm19,zmm5
        vmovaps zmm20,zmm4
        vmovaps zmm21,zmm5
        vmovaps zmm22,zmm4
        vmovaps zmm23,zmm5
        vmovaps zmm24,zmm4
        vmovaps zmm25,zmm5
        vmovaps zmm26,zmm4
        vmovaps zmm27,zmm5
        lea     rbx,[r10*2+r10]
        lea     r13,[rdi+rbx*2]             # compute matrix A plus 6 rows
        lea     r14,[r13+rbx]               # compute matrix A plus 9 rows
        add     rbx,rdi                     # compute matrix A plus 3 rows
        ComputeBlockAvx512FLoop \Mode\(), ComputeBlockAvx512FBy32, 12
        add     rsi,r12                     # advance matrix B by 16*CountK floats
        lea     rbx,[rax*2+rax]
        lea     r13,[rdx+rbx*2]             # compute matrix C plus 6 rows
        lea     r14,[r13+rbx]               # compute matrix C plus 9 rows
        add     rbx,rdx                     # compute matrix C plus 3 rows
.ifnes "\Mode\()","Add"
        vmulps  zmm4,zmm4,zmm31             # multiply by alpha
        vmulps  zmm6,zmm6,zmm31
        vmulps  zmm8,zmm8,zmm31
        vmulps  zmm10,zmm10,zmm31
        vmulps  zmm12,zmm12,zmm31
        vmulps  zmm14,zmm14,zmm31
        vmulps  zmm16,zmm16,zmm31
        vmulps  zmm18,zmm18,zmm31
        vmulps  zmm20,zmm20,zmm31
        vmulps  zmm22,zmm22,zmm31
        vmulps  zmm24,zmm24,zmm31
        vmulps  zmm26,zmm26,zmm31
.else
        vfmadd213ps zmm4,zmm31,ZMMWORD PTR [rdx]
        vfmadd213ps zmm6,zmm31,ZMMWORD PTR [rdx+rax]
        vfmadd213ps zmm8,zmm31,ZMMWORD PTR [rdx+rax*2]
        vfmadd213ps zmm10,zmm31,ZMMWORD PTR [rbx]
        vfmadd213ps zmm12,zmm31,ZMMWORD PTR [rbx+rax]
        vfmadd213ps zmm14,zmm31,ZMMWORD PTR [rbx+rax*2]
        vfmadd213ps zmm16,zmm31,ZMMWORD PTR [r13]
        vfmadd213ps zmm18,zmm31,ZMMWORD PTR [r13+rax]
        vfmadd213ps zmm20,zmm31,ZMMWORD PTR [r13+rax*2]
        vfmadd213ps zmm22,zmm31,ZMMWORD PTR [r14]
        vfmadd213ps zmm24,zmm31,ZMMWORD PTR [r14+rax]
        vfmadd213ps zmm26,zmm31,ZMMWORD PTR [r14+rax*2]
.endif
        vmovups ZMMWORD PTR [rdx],zmm4
        vmovups ZMMWORD PTR [rdx+rax],zmm6
        vmovups ZMMWORD PTR [rdx+rax*2],zmm8
        vmovups ZMMWORD PTR [rbx],zmm10
        vmovups ZMMWORD PTR [rbx+rax],zmm12
        vmovups ZMMWORD PTR [rbx+rax*2],zmm14
        vmovups ZMMWORD PTR [r13],zmm16
        vmovups ZMMWORD PTR [r13+rax],zmm18
        vmovups ZMMWORD PTR [r13+rax*2],zmm20
        vmovups ZMMWORD PTR [r14],zmm22
        vmovups ZMMWORD PTR [r14+rax],zmm24
        vmovups ZMMWORD PTR [r14+rax*2],zmm26
        add     rdx,16*4                    # advance matrix C by 16 columns
        sub     r9,16

.L\Mode\().Output16x12Block:
        sub     r9,16
        jae     .L\Mode\().Output16x12BlockWithMask
        lea     rcx,[r9+16]                 # correct for over-subtract above
        mov     ebx,1
        shl     ebx,cl
        dec     ebx
        kmovw   k1,ebx                      # update mask for remaining columns
        xor     r9,r9                       # no more columns remaining

.L\Mode\().Output16x12BlockWithMask:
        lea     rbx,[rax*2+rax]
        lea     r13,[rdx+rbx*2]             # compute matrix C plus 6 rows
        lea     r14,[r13+rbx]               # compute matrix C plus 9 rows
        add     rbx,rdx                     # compute matrix C plus 3 rows
.ifnes "\Mode\()","Add"
        vmulps  zmm5,zmm5,zmm31             # multiply by alpha
        vmulps  zmm7,zmm7,zmm31
        vmulps  zmm9,zmm9,zmm31
        vmulps  zmm11,zmm11,zmm31
        vmulps  zmm13,zmm13,zmm31
        vmulps  zmm15,zmm15,zmm31
        vmulps  zmm17,zmm17,zmm31
        vmulps  zmm19,zmm19,zmm31
        vmulps  zmm21,zmm21,zmm31
        vmulps  zmm23,zmm23,zmm31
        vmulps  zmm25,zmm25,zmm31
        vmulps  zmm27,zmm27,zmm31
.else
        vfmadd213ps zmm5{k1},zmm31,ZMMWORD PTR [rdx]
        vfmadd213ps zmm7{k1},zmm31,ZMMWORD PTR [rdx+rax]
        vfmadd213ps zmm9{k1},zmm31,ZMMWORD PTR [rdx+rax*2]
        vfmadd213ps zmm11{k1},zmm31,ZMMWORD PTR [rbx]
        vfmadd213ps zmm13{k1},zmm31,ZMMWORD PTR [rbx+rax]
        vfmadd213ps zmm15{k1},zmm31,ZMMWORD PTR [rbx+rax*2]
        vfmadd213ps zmm17{k1},zmm31,ZMMWORD PTR [r13]
        vfmadd213ps zmm19{k1},zmm31,ZMMWORD PTR [r13+rax]
        vfmadd213ps zmm21{k1},zmm31,ZMMWORD PTR [r13+rax*2]
        vfmadd213ps zmm23{k1},zmm31,ZMMWORD PTR [r14]
        vfmadd213ps zmm25{k1},zmm31,ZMMWORD PTR [r14+rax]
        vfmadd213ps zmm27{k1},zmm31,ZMMWORD PTR [r14+rax*2]
.endif
        vmovups ZMMWORD PTR [rdx]{k1},zmm5
        vmovups ZMMWORD PTR [rdx+rax]{k1},zmm7
        vmovups ZMMWORD PTR [rdx+rax*2]{k1},zmm9
        vmovups ZMMWORD PTR [rbx]{k1},zmm11
        vmovups ZMMWORD PTR [rbx+rax]{k1},zmm13
        vmovups ZMMWORD PTR [rbx+rax*2]{k1},zmm15
        vmovups ZMMWORD PTR [r13]{k1},zmm17
        vmovups ZMMWORD PTR [r13+rax]{k1},zmm19
        vmovups ZMMWORD PTR [r13+rax*2]{k1},zmm21
        vmovups ZMMWORD PTR [r14]{k1},zmm23
        vmovups ZMMWORD PTR [r14+rax]{k1},zmm25
        vmovups ZMMWORD PTR [r14+rax*2]{k1},zmm27
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,16
        ja      .L\Mode\().ProcessNextColumnLoop32x12
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN12:
        vmovaps zmm17,zmm5                  # clear upper block accumulators
        vmovaps zmm19,zmm5
        vmovaps zmm21,zmm5
        vmovaps zmm23,zmm5
        vmovaps zmm25,zmm5
        vmovaps zmm27,zmm5
        lea     rbx,[r10*2+r10]
        lea     r13,[rdi+rbx*2]             # compute matrix A plus 6 rows
        lea     r14,[r13+rbx]               # compute matrix A plus 9 rows
        add     rbx,rdi                     # compute matrix A plus 3 rows
        ComputeBlockAvx512FLoop \Mode\(), ComputeBlockAvx512FBy16, 12
        jmp     .L\Mode\().Output16x12Block

//
// Restore non-volatile registers and return.
//

.L\Mode\().ExitKernel:
        mov     eax,r8d
        pop     r14
        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret

//
// Process 6 rows of the matrices.
//

.L\Mode\().ProcessCountMLessThan12:
        cmp     r8,6
        jb      .L\Mode\().ProcessCountMLessThan6
        mov     r8d,6                       # return 6 rows handled
        cmp     r9,16
        jbe     .L\Mode\().ProcessRemainingCountN6

.L\Mode\().ProcessNextColumnLoop32x6:
        lea     rbx,[r10*2+r10]
        add     rbx,rdi                     # compute matrix A plus 3 rows
        ComputeBlockAvx512FLoop \Mode\(), ComputeBlockAvx512FBy32, 6
        add     rsi,r12                     # advance matrix B by 16*CountK floats
        lea     rdi,[rdx+rax*2]             # compute matrix C plus 2 rows
        lea     rbx,[rdx+rax*4]             # compute matrix C plus 4 rows
.ifnes "\Mode\()","Add"
        vmulps  zmm4,zmm4,zmm31             # multiply by alpha
        vmulps  zmm6,zmm6,zmm31
        vmulps  zmm8,zmm8,zmm31
        vmulps  zmm10,zmm10,zmm31
        vmulps  zmm12,zmm12,zmm31
        vmulps  zmm14,zmm14,zmm31
.else
        vfmadd213ps zmm4,zmm31,ZMMWORD PTR [rdx]
        vfmadd213ps zmm6,zmm31,ZMMWORD PTR [rdx+rax]
        vfmadd213ps zmm8,zmm31,ZMMWORD PTR [rdi]
        vfmadd213ps zmm10,zmm31,ZMMWORD PTR [rdi+rax]
        vfmadd213ps zmm12,zmm31,ZMMWORD PTR [rbx]
        vfmadd213ps zmm14,zmm31,ZMMWORD PTR [rbx+rax]
.endif
        vmovups ZMMWORD PTR [rdx],zmm4
        vmovups ZMMWORD PTR [rdx+rax],zmm6
        vmovups ZMMWORD PTR [rdi],zmm8
        vmovups ZMMWORD PTR [rdi+rax],zmm10
        vmovups ZMMWORD PTR [rbx],zmm12
        vmovups ZMMWORD PTR [rbx+rax],zmm14
        add     rdx,16*4                    # advance matrix C by 16 columns
        sub     r9,16

.L\Mode\().Output16x6Block:
        sub     r9,16
        jae     .L\Mode\().Output16x6BlockWithMask
        lea     rcx,[r9+16]                 # correct for over-subtract above
        mov     ebx,1
        shl     ebx,cl
        dec     ebx
        kmovw   k1,ebx                      # update mask for remaining columns
        xor     r9,r9                       # no more columns remaining

.L\Mode\().Output16x6BlockWithMask:
        lea     rdi,[rdx+rax*2]             # compute matrix C plus 2 rows
        lea     rbx,[rdx+rax*4]             # compute matrix C plus 4 rows
.ifnes "\Mode\()","Add"
        vmulps  zmm5,zmm5,zmm31             # multiply by alpha
        vmulps  zmm7,zmm7,zmm31
        vmulps  zmm9,zmm9,zmm31
        vmulps  zmm11,zmm11,zmm31
        vmulps  zmm13,zmm13,zmm31
        vmulps  zmm15,zmm15,zmm31
.else
        vfmadd213ps zmm5{k1},zmm31,ZMMWORD PTR [rdx]
        vfmadd213ps zmm7{k1},zmm31,ZMMWORD PTR [rdx+rax]
        vfmadd213ps zmm9{k1},zmm31,ZMMWORD PTR [rdi]
        vfmadd213ps zmm11{k1},zmm31,ZMMWORD PTR [rdi+rax]
        vfmadd213ps zmm13{k1},zmm31,ZMMWORD PTR [rbx]
        vfmadd213ps zmm15{k1},zmm31,ZMMWORD PTR [rbx+rax]
.endif
        vmovups ZMMWORD PTR [rdx]{k1},zmm5
        vmovups ZMMWORD PTR [rdx+rax]{k1},zmm7
        vmovups ZMMWORD PTR [rdi]{k1},zmm9
        vmovups ZMMWORD PTR [rdi+rax]{k1},zmm11
        vmovups ZMMWORD PTR [rbx]{k1},zmm13
        vmovups ZMMWORD PTR [rbx+rax]{k1},zmm15
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,16
        ja      .L\Mode\().ProcessNextColumnLoop32x6
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN6:
        lea     rbx,[r10*2+r10]
        add     rbx,rdi                     # compute matrix A plus 3 rows
        ComputeBlockAvx512FLoop \Mode\(), ComputeBlockAvx512FBy16, 6
        jmp     .L\Mode\().Output16x6Block

//
// Process 3 rows of the matrices.
//

.L\Mode\().ProcessCountMLessThan6:
        cmp     r8,3
        jb      .L\Mode\().ProcessCountMLessThan3
        mov     r8d,3                       # return 3 rows handled
        cmp     r9,16
        jbe     .L\Mode\().ProcessRemainingCountN3

.L\Mode\().ProcessNextColumnLoop32x3:
        ComputeBlockAvx512FLoop \Mode\(), ComputeBlockAvx512FBy32, 3
        add     rsi,r12                     # advance matrix B by 16*CountK floats
.ifnes "\Mode\()","Add"
        vmulps  zmm4,zmm4,zmm31             # multiply by alpha
        vmulps  zmm6,zmm6,zmm31
        vmulps  zmm8,zmm8,zmm31
.else
        vfmadd213ps zmm4,zmm31,ZMMWORD PTR [rdx]
        vfmadd213ps zmm6,zmm31,ZMMWORD PTR [rdx+rax]
        vfmadd213ps zmm8,zmm31,ZMMWORD PTR [rdx+rax*2]
.endif
        vmovups ZMMWORD PTR [rdx],zmm4
        vmovups ZMMWORD PTR [rdx+rax],zmm6
        vmovups ZMMWORD PTR [rdx+rax*2],zmm8
        add     rdx,16*4                    # advance matrix C by 16 columns
        sub     r9,16

.L\Mode\().Output16x3Block:
        sub     r9,16
        jae     .L\Mode\().Output16x3BlockWithMask
        lea     rcx,[r9+16]                 # correct for over-subtract above
        mov     ebx,1
        shl     ebx,cl
        dec     ebx
        kmovw   k1,ebx                      # update mask for remaining columns
        xor     r9,r9                       # no more columns remaining

.L\Mode\().Output16x3BlockWithMask:
.ifnes "\Mode\()","Add"
        vmulps  zmm5,zmm5,zmm31             # multiply by alpha
        vmulps  zmm7,zmm7,zmm31
        vmulps  zmm9,zmm9,zmm31
.else
        vfmadd213ps zmm5{k1},zmm31,ZMMWORD PTR [rdx]
        vfmadd213ps zmm7{k1},zmm31,ZMMWORD PTR [rdx+rax]
        vfmadd213ps zmm9{k1},zmm31,ZMMWORD PTR [rdx+rax*2]
.endif
        vmovups ZMMWORD PTR [rdx]{k1},zmm5
        vmovups ZMMWORD PTR [rdx+rax]{k1},zmm7
        vmovups ZMMWORD PTR [rdx+rax*2]{k1},zmm9
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,16
        ja      .L\Mode\().ProcessNextColumnLoop32x3
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN3:
        ComputeBlockAvx512FLoop \Mode\(), ComputeBlockAvx512FBy16, 3
        jmp     .L\Mode\().Output16x3Block

//
// Process 1 row of the matrices.
//

.L\Mode\().ProcessCountMLessThan3:
        mov     r8d,1                       # return 1 row handled
        cmp     r9,16
        jbe     .L\Mode\().ProcessRemainingCountN1

.L\Mode\().ProcessNextColumnLoop32x1:
        ComputeBlockAvx512FLoop \Mode\(), ComputeBlockAvx512FBy32, 1
        add     rsi,r12                     # advance matrix B by 16*CountK floats
.ifnes "\Mode\()","Add"
        vmulps  zmm4,zmm4,zmm31             # multiply by alpha
.else
        vfmadd213ps zmm4,zmm31,ZMMWORD PTR [rdx]
.endif
        vmovups ZMMWORD PTR [rdx],zmm4
        add     rdx,16*4                    # advance matrix C by 16 columns
        sub     r9,16

.L\Mode\().Output16x1Block:
        sub     r9,16
        jae     .L\Mode\().Output16x1BlockWithMask
        lea     rcx,[r9+16]                 # correct for over-subtract above
        mov     ebx,1
        shl     ebx,cl
        dec     ebx
        kmovw   k1,ebx                      # update mask for remaining columns
        xor     r9,r9                       # no more columns remaining

.L\Mode\().Output16x1BlockWithMask:
.ifnes "\Mode\()","Add"
        vmulps  zmm5,zmm5,zmm31             # multiply by alpha
.else
        vfmadd213ps zmm5{k1},zmm31,ZMMWORD PTR [rdx]
.endif
        vmovups ZMMWORD PTR [rdx]{k1},zmm5
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,16
        ja      .L\Mode\().ProcessNextColumnLoop32x1
        test    r9,r9
        jz      .L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN1:
        ComputeBlockAvx512FLoop \Mode\(), ComputeBlockAvx512FBy16, 1
        jmp     .L\Mode\().Output16x1Block

        .endm

        SgemmKernelAvx512FFunction Zero
        SgemmKernelAvx512FFunction Add

        .end
