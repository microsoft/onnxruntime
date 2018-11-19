;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelAvx512F.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;   This implementation uses AVX512F instructions.
;
;--

        .xlist
INCLUDE macamd64.inc
INCLUDE SgemmKernelCommon.inc
        .list

;
; ComputeBlockAvx512FBy32
;
;   This macro multiplies and accumulates for a 32xN block (where N is 1,3,6,12)
;   of the output matrix.
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
;   rbx - Supplies the address into the matrix A data plus 3 rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   r13 - Supplies the address into the matrix A data plus 6 rows.
;
;   r14 - Supplies the address into the matrix A data plus 9 rows.
;
;   zmm4-zmm27 - Supplies the block accumulators.
;

ComputeBlockAvx512FBy32 MACRO Count, VectorOffset, BroadcastOffset

IF Count EQ 1
        vbroadcastss zmm3,DWORD PTR [rcx+BroadcastOffset]
        vfmadd231ps zmm4,zmm3,ZMMWORD PTR [rdx+VectorOffset]
        vfmadd231ps zmm5,zmm3,ZMMWORD PTR [rdx+r12+VectorOffset]
ELSE
        vmovaps zmm0,ZMMWORD PTR [rdx+VectorOffset]
        vmovaps zmm1,ZMMWORD PTR [rdx+r12+VectorOffset]
        vbroadcastss zmm3,DWORD PTR [rcx+BroadcastOffset]
        vfmadd231ps zmm4,zmm3,zmm0
        vfmadd231ps zmm5,zmm3,zmm1
IF Count GE 3
        vbroadcastss zmm3,DWORD PTR [rcx+r10+BroadcastOffset]
        vfmadd231ps zmm6,zmm3,zmm0
        vfmadd231ps zmm7,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [rcx+r10*2+BroadcastOffset]
        vfmadd231ps zmm8,zmm3,zmm0
        vfmadd231ps zmm9,zmm3,zmm1
ENDIF
IF Count GE 6
        vbroadcastss zmm3,DWORD PTR [rbx+BroadcastOffset]
        vfmadd231ps zmm10,zmm3,zmm0
        vfmadd231ps zmm11,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [rbx+r10+BroadcastOffset]
        vfmadd231ps zmm12,zmm3,zmm0
        vfmadd231ps zmm13,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [rbx+r10*2+BroadcastOffset]
        vfmadd231ps zmm14,zmm3,zmm0
        vfmadd231ps zmm15,zmm3,zmm1
ENDIF
IF Count GE 12
        vbroadcastss zmm3,DWORD PTR [r13+BroadcastOffset]
        vfmadd231ps zmm16,zmm3,zmm0
        vfmadd231ps zmm17,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [r13+r10+BroadcastOffset]
        vfmadd231ps zmm18,zmm3,zmm0
        vfmadd231ps zmm19,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [r13+r10*2+BroadcastOffset]
        vfmadd231ps zmm20,zmm3,zmm0
        vfmadd231ps zmm21,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [r14+BroadcastOffset]
        vfmadd231ps zmm22,zmm3,zmm0
        vfmadd231ps zmm23,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [r14+r10+BroadcastOffset]
        vfmadd231ps zmm24,zmm3,zmm0
        vfmadd231ps zmm25,zmm3,zmm1
        vbroadcastss zmm3,DWORD PTR [r14+r10*2+BroadcastOffset]
        vfmadd231ps zmm26,zmm3,zmm0
        vfmadd231ps zmm27,zmm3,zmm1
ENDIF
ENDIF

        ENDM

;
; ComputeBlockAvx512FBy16
;
;   This macro multiplies and accumulates for a 16xN block (where N is 1,3,6,12)
;   of the output matrix.
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
;   rbx - Supplies the address into the matrix A data plus 3 rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   r13 - Supplies the address into the matrix A data plus 6 rows.
;
;   r14 - Supplies the address into the matrix A data plus 9 rows.
;
;   zmm4-zmm27 - Supplies the block accumulators.
;

ComputeBlockAvx512FBy16 MACRO Count, VectorOffset, BroadcastOffset

        vmovaps zmm0,ZMMWORD PTR [rdx+VectorOffset]
        vfmadd231ps zmm5,zmm0,DWORD BCST [rcx+BroadcastOffset]
IF Count GE 3
        vfmadd231ps zmm7,zmm0,DWORD BCST [rcx+r10+BroadcastOffset]
        vfmadd231ps zmm9,zmm0,DWORD BCST [rcx+r10*2+BroadcastOffset]
ENDIF
IF Count GE 6
        vfmadd231ps zmm11,zmm0,DWORD BCST [rbx+BroadcastOffset]
        vfmadd231ps zmm13,zmm0,DWORD BCST [rbx+r10+BroadcastOffset]
        vfmadd231ps zmm15,zmm0,DWORD BCST [rbx+r10*2+BroadcastOffset]
ENDIF
IF Count GE 12
        vfmadd231ps zmm17,zmm0,DWORD BCST [r13+BroadcastOffset]
        vfmadd231ps zmm19,zmm0,DWORD BCST [r13+r10+BroadcastOffset]
        vfmadd231ps zmm21,zmm0,DWORD BCST [r13+r10*2+BroadcastOffset]
        vfmadd231ps zmm23,zmm0,DWORD BCST [r14+BroadcastOffset]
        vfmadd231ps zmm25,zmm0,DWORD BCST [r14+r10+BroadcastOffset]
        vfmadd231ps zmm27,zmm0,DWORD BCST [r14+r10*2+BroadcastOffset]
ENDIF

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
;   Alpha - Supplies the scaler multiplier (see SGEMM definition).
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

SgemmKernelAvx512FFunction MACRO Mode

        NESTED_ENTRY MlasSgemmKernel&Mode&Avx512F, _TEXT

        SgemmKernelAvxEntry SaveExtra

        mov     r12,r9
        shl     r12,6                       ; compute 16*CountK*sizeof(float)
        mov     edi,-1
        kmovw   k1,edi                      ; update mask to write all columns
        vbroadcastss zmm31,DWORD PTR SgemmKernelFrame.Alpha[rsp]

;
; Process 12 rows of the matrices.
;

        cmp     r11,12
        jb      ProcessCountMLessThan12
        mov     r11d,12                     ; return 12 rows handled
        cmp     rbp,16
        jbe     ProcessRemainingCountN12

ProcessNextColumnLoop32x12:
        vmovaps zmm16,zmm4                  ; clear upper block accumulators
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
        lea     r13,[rcx+rbx*2]             ; compute matrix A plus 6 rows
        lea     r14,[r13+rbx]               ; compute matrix A plus 9 rows
        add     rbx,rcx                     ; compute matrix A plus 3 rows
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy32, 12
        add     rdx,r12                     ; advance matrix B by 16*CountK floats
        lea     rbx,[rax*2+rax]
        lea     r13,[r8+rbx*2]              ; compute matrix C plus 6 rows
        lea     r14,[r13+rbx]               ; compute matrix C plus 9 rows
        add     rbx,r8                      ; compute matrix C plus 3 rows
IFDIFI <Mode>, <Add>
        vmulps  zmm4,zmm4,zmm31             ; multiply by alpha
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
ELSE
        vfmadd213ps zmm4,zmm31,ZMMWORD PTR [r8]
        vfmadd213ps zmm6,zmm31,ZMMWORD PTR [r8+rax]
        vfmadd213ps zmm8,zmm31,ZMMWORD PTR [r8+rax*2]
        vfmadd213ps zmm10,zmm31,ZMMWORD PTR [rbx]
        vfmadd213ps zmm12,zmm31,ZMMWORD PTR [rbx+rax]
        vfmadd213ps zmm14,zmm31,ZMMWORD PTR [rbx+rax*2]
        vfmadd213ps zmm16,zmm31,ZMMWORD PTR [r13]
        vfmadd213ps zmm18,zmm31,ZMMWORD PTR [r13+rax]
        vfmadd213ps zmm20,zmm31,ZMMWORD PTR [r13+rax*2]
        vfmadd213ps zmm22,zmm31,ZMMWORD PTR [r14]
        vfmadd213ps zmm24,zmm31,ZMMWORD PTR [r14+rax]
        vfmadd213ps zmm26,zmm31,ZMMWORD PTR [r14+rax*2]
ENDIF
        vmovups ZMMWORD PTR [r8],zmm4
        vmovups ZMMWORD PTR [r8+rax],zmm6
        vmovups ZMMWORD PTR [r8+rax*2],zmm8
        vmovups ZMMWORD PTR [rbx],zmm10
        vmovups ZMMWORD PTR [rbx+rax],zmm12
        vmovups ZMMWORD PTR [rbx+rax*2],zmm14
        vmovups ZMMWORD PTR [r13],zmm16
        vmovups ZMMWORD PTR [r13+rax],zmm18
        vmovups ZMMWORD PTR [r13+rax*2],zmm20
        vmovups ZMMWORD PTR [r14],zmm22
        vmovups ZMMWORD PTR [r14+rax],zmm24
        vmovups ZMMWORD PTR [r14+rax*2],zmm26
        add     r8,16*4                     ; advance matrix C by 16 columns
        sub     rbp,16

Output16x12Block:
        sub     rbp,16
        jae     Output16x12BlockWithMask
        lea     ecx,[ebp+16]                ; correct for over-subtract above
        mov     edi,1
        shl     edi,cl
        dec     edi
        kmovw   k1,edi                      ; update mask for remaining columns
        xor     ebp,ebp                     ; no more columns remaining

Output16x12BlockWithMask:
        lea     rbx,[rax*2+rax]
        lea     r13,[r8+rbx*2]              ; compute matrix C plus 6 rows
        lea     r14,[r13+rbx]               ; compute matrix C plus 9 rows
        add     rbx,r8                      ; compute matrix C plus 3 rows
IFDIFI <Mode>, <Add>
        vmulps  zmm5,zmm5,zmm31             ; multiply by alpha
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
ELSE
        vfmadd213ps zmm5{k1},zmm31,ZMMWORD PTR [r8]
        vfmadd213ps zmm7{k1},zmm31,ZMMWORD PTR [r8+rax]
        vfmadd213ps zmm9{k1},zmm31,ZMMWORD PTR [r8+rax*2]
        vfmadd213ps zmm11{k1},zmm31,ZMMWORD PTR [rbx]
        vfmadd213ps zmm13{k1},zmm31,ZMMWORD PTR [rbx+rax]
        vfmadd213ps zmm15{k1},zmm31,ZMMWORD PTR [rbx+rax*2]
        vfmadd213ps zmm17{k1},zmm31,ZMMWORD PTR [r13]
        vfmadd213ps zmm19{k1},zmm31,ZMMWORD PTR [r13+rax]
        vfmadd213ps zmm21{k1},zmm31,ZMMWORD PTR [r13+rax*2]
        vfmadd213ps zmm23{k1},zmm31,ZMMWORD PTR [r14]
        vfmadd213ps zmm25{k1},zmm31,ZMMWORD PTR [r14+rax]
        vfmadd213ps zmm27{k1},zmm31,ZMMWORD PTR [r14+rax*2]
ENDIF
        vmovups ZMMWORD PTR [r8]{k1},zmm5
        vmovups ZMMWORD PTR [r8+rax]{k1},zmm7
        vmovups ZMMWORD PTR [r8+rax*2]{k1},zmm9
        vmovups ZMMWORD PTR [rbx]{k1},zmm11
        vmovups ZMMWORD PTR [rbx+rax]{k1},zmm13
        vmovups ZMMWORD PTR [rbx+rax*2]{k1},zmm15
        vmovups ZMMWORD PTR [r13]{k1},zmm17
        vmovups ZMMWORD PTR [r13+rax]{k1},zmm19
        vmovups ZMMWORD PTR [r13+rax*2]{k1},zmm21
        vmovups ZMMWORD PTR [r14]{k1},zmm23
        vmovups ZMMWORD PTR [r14+rax]{k1},zmm25
        vmovups ZMMWORD PTR [r14+rax*2]{k1},zmm27
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,16
        ja      ProcessNextColumnLoop32x12
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN12:
        vmovaps zmm17,zmm5                  ; clear upper block accumulators
        vmovaps zmm19,zmm5
        vmovaps zmm21,zmm5
        vmovaps zmm23,zmm5
        vmovaps zmm25,zmm5
        vmovaps zmm27,zmm5
        lea     rbx,[r10*2+r10]
        lea     r13,[rcx+rbx*2]             ; compute matrix A plus 6 rows
        lea     r14,[r13+rbx]               ; compute matrix A plus 9 rows
        add     rbx,rcx                     ; compute matrix A plus 3 rows
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy16, 12
        jmp     Output16x12Block

;
; Restore non-volatile registers and return.
;

ExitKernel:
        SgemmKernelAvxExit RestoreExtra

;
; Process 6 rows of the matrices.
;

ProcessCountMLessThan12:
        cmp     r11,6
        jb      ProcessCountMLessThan6
        mov     r11d,6                      ; return 6 rows handled
        cmp     rbp,16
        jbe     ProcessRemainingCountN6

ProcessNextColumnLoop32x6:
        lea     rbx,[r10*2+r10]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy32, 6
        add     rdx,r12                     ; advance matrix B by 16*CountK floats
        lea     rdi,[r8+rax*2]              ; compute matrix C plus 2 rows
        lea     rbx,[r8+rax*4]              ; compute matrix C plus 4 rows
IFDIFI <Mode>, <Add>
        vmulps  zmm4,zmm4,zmm31             ; multiply by alpha
        vmulps  zmm6,zmm6,zmm31
        vmulps  zmm8,zmm8,zmm31
        vmulps  zmm10,zmm10,zmm31
        vmulps  zmm12,zmm12,zmm31
        vmulps  zmm14,zmm14,zmm31
ELSE
        vfmadd213ps zmm4,zmm31,ZMMWORD PTR [r8]
        vfmadd213ps zmm6,zmm31,ZMMWORD PTR [r8+rax]
        vfmadd213ps zmm8,zmm31,ZMMWORD PTR [rdi]
        vfmadd213ps zmm10,zmm31,ZMMWORD PTR [rdi+rax]
        vfmadd213ps zmm12,zmm31,ZMMWORD PTR [rbx]
        vfmadd213ps zmm14,zmm31,ZMMWORD PTR [rbx+rax]
ENDIF
        vmovups ZMMWORD PTR [r8],zmm4
        vmovups ZMMWORD PTR [r8+rax],zmm6
        vmovups ZMMWORD PTR [rdi],zmm8
        vmovups ZMMWORD PTR [rdi+rax],zmm10
        vmovups ZMMWORD PTR [rbx],zmm12
        vmovups ZMMWORD PTR [rbx+rax],zmm14
        add     r8,16*4                     ; advance matrix C by 16 columns
        sub     rbp,16

Output16x6Block:
        sub     rbp,16
        jae     Output16x6BlockWithMask
        lea     ecx,[ebp+16]                ; correct for over-subtract above
        mov     edi,1
        shl     edi,cl
        dec     edi
        kmovw   k1,edi                      ; update mask for remaining columns
        xor     ebp,ebp                     ; no more columns remaining

Output16x6BlockWithMask:
        lea     rdi,[r8+rax*2]              ; compute matrix C plus 2 rows
        lea     rbx,[r8+rax*4]              ; compute matrix C plus 4 rows
IFDIFI <Mode>, <Add>
        vmulps  zmm5,zmm5,zmm31             ; multiply by alpha
        vmulps  zmm7,zmm7,zmm31
        vmulps  zmm9,zmm9,zmm31
        vmulps  zmm11,zmm11,zmm31
        vmulps  zmm13,zmm13,zmm31
        vmulps  zmm15,zmm15,zmm31
ELSE
        vfmadd213ps zmm5{k1},zmm31,ZMMWORD PTR [r8]
        vfmadd213ps zmm7{k1},zmm31,ZMMWORD PTR [r8+rax]
        vfmadd213ps zmm9{k1},zmm31,ZMMWORD PTR [rdi]
        vfmadd213ps zmm11{k1},zmm31,ZMMWORD PTR [rdi+rax]
        vfmadd213ps zmm13{k1},zmm31,ZMMWORD PTR [rbx]
        vfmadd213ps zmm15{k1},zmm31,ZMMWORD PTR [rbx+rax]
ENDIF
        vmovups ZMMWORD PTR [r8]{k1},zmm5
        vmovups ZMMWORD PTR [r8+rax]{k1},zmm7
        vmovups ZMMWORD PTR [rdi]{k1},zmm9
        vmovups ZMMWORD PTR [rdi+rax]{k1},zmm11
        vmovups ZMMWORD PTR [rbx]{k1},zmm13
        vmovups ZMMWORD PTR [rbx+rax]{k1},zmm15
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,16
        ja      ProcessNextColumnLoop32x6
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN6:
        lea     rbx,[r10*2+r10]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy16, 6
        jmp     Output16x6Block

;
; Process 3 rows of the matrices.
;

ProcessCountMLessThan6:
        cmp     r11,3
        jb      ProcessCountMLessThan3
        mov     r11d,3                      ; return 3 rows handled
        cmp     rbp,16
        jbe     ProcessRemainingCountN3

ProcessNextColumnLoop32x3:
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy32, 3
        add     rdx,r12                     ; advance matrix B by 16*CountK floats
IFDIFI <Mode>, <Add>
        vmulps  zmm4,zmm4,zmm31             ; multiply by alpha
        vmulps  zmm6,zmm6,zmm31
        vmulps  zmm8,zmm8,zmm31
ELSE
        vfmadd213ps zmm4,zmm31,ZMMWORD PTR [r8]
        vfmadd213ps zmm6,zmm31,ZMMWORD PTR [r8+rax]
        vfmadd213ps zmm8,zmm31,ZMMWORD PTR [r8+rax*2]
ENDIF
        vmovups ZMMWORD PTR [r8],zmm4
        vmovups ZMMWORD PTR [r8+rax],zmm6
        vmovups ZMMWORD PTR [r8+rax*2],zmm8
        add     r8,16*4                     ; advance matrix C by 16 columns
        sub     rbp,16

Output16x3Block:
        sub     rbp,16
        jae     Output16x3BlockWithMask
        lea     ecx,[ebp+16]                ; correct for over-subtract above
        mov     edi,1
        shl     edi,cl
        dec     edi
        kmovw   k1,edi                      ; update mask for remaining columns
        xor     ebp,ebp                     ; no more columns remaining

Output16x3BlockWithMask:
IFDIFI <Mode>, <Add>
        vmulps  zmm5,zmm5,zmm31             ; multiply by alpha
        vmulps  zmm7,zmm7,zmm31
        vmulps  zmm9,zmm9,zmm31
ELSE
        vfmadd213ps zmm5{k1},zmm31,ZMMWORD PTR [r8]
        vfmadd213ps zmm7{k1},zmm31,ZMMWORD PTR [r8+rax]
        vfmadd213ps zmm9{k1},zmm31,ZMMWORD PTR [r8+rax*2]
ENDIF
        vmovups ZMMWORD PTR [r8]{k1},zmm5
        vmovups ZMMWORD PTR [r8+rax]{k1},zmm7
        vmovups ZMMWORD PTR [r8+rax*2]{k1},zmm9
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,16
        ja      ProcessNextColumnLoop32x3
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN3:
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy16, 3
        jmp     Output16x3Block

;
; Process 1 row of the matrices.
;

ProcessCountMLessThan3:
        mov     r11d,1                      ; return 1 row handled
        cmp     rbp,16
        jbe     ProcessRemainingCountN1

ProcessNextColumnLoop32x1:
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy32, 1
        add     rdx,r12                     ; advance matrix B by 16*CountK floats
IFDIFI <Mode>, <Add>
        vmulps  zmm4,zmm4,zmm31             ; multiply by alpha
ELSE
        vfmadd213ps zmm4,zmm31,ZMMWORD PTR [r8]
ENDIF
        vmovups ZMMWORD PTR [r8],zmm4
        add     r8,16*4                     ; advance matrix C by 16 columns
        sub     rbp,16

Output16x1Block:
        sub     rbp,16
        jae     Output16x1BlockWithMask
        lea     ecx,[ebp+16]                ; correct for over-subtract above
        mov     edi,1
        shl     edi,cl
        dec     edi
        kmovw   k1,edi                      ; update mask for remaining columns
        xor     ebp,ebp                     ; no more columns remaining

Output16x1BlockWithMask:
IFDIFI <Mode>, <Add>
        vmulps  zmm5,zmm5,zmm31             ; multiply by alpha
ELSE
        vfmadd213ps zmm5{k1},zmm31,ZMMWORD PTR [r8]
ENDIF
        vmovups ZMMWORD PTR [r8]{k1},zmm5
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,16
        ja      ProcessNextColumnLoop32x1
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN1:
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy16, 1
        jmp     Output16x1Block

        NESTED_END MlasSgemmKernel&Mode&Avx512F, _TEXT

        ENDM

SgemmKernelAvx512FFunction Zero
SgemmKernelAvx512FFunction Add

        END
