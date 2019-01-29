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
INCLUDE mlasi.inc
INCLUDE SgemmKernelCommon.inc
        .list

;
; ComputeBlockAvx512FBy32
;
;   This macro multiplies and accumulates for a 32xN block of the output matrix.
;
; Arguments:
;
;   Count - Supplies the number of rows to access from matrix A.
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
;   r13 - Supplies the address into the matrix A data plus 6 rows.
;
;   r14 - Supplies the address into the matrix A data plus 9 rows.
;
;   zmm4-zmm27 - Supplies the block accumulators.
;

ComputeBlockAvx512FBy32 MACRO Count, VectorOffset, BroadcastOffset, PrefetchOffset

IFNB <PrefetchOffset>
        prefetcht0 [rdx+VectorOffset+PrefetchOffset]
        prefetcht0 [rdx+r12+VectorOffset+PrefetchOffset]
ENDIF
IF Count EQ 1
        vbroadcastss zmm3,DWORD PTR [rcx+BroadcastOffset]
        vfmadd231ps zmm4,zmm3,ZMMWORD PTR [rdx+VectorOffset]
        vfmadd231ps zmm5,zmm3,ZMMWORD PTR [rdx+r12+VectorOffset]
ELSE
        vmovaps zmm0,ZMMWORD PTR [rdx+VectorOffset]
        vmovaps zmm1,ZMMWORD PTR [rdx+r12+VectorOffset]
        EmitIfCountGE Count, 1, <vbroadcastss zmm3,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE Count, 1, <vfmadd231ps zmm4,zmm3,zmm0>
        EmitIfCountGE Count, 1, <vfmadd231ps zmm5,zmm3,zmm1>
        EmitIfCountGE Count, 2, <vbroadcastss zmm3,DWORD PTR [rcx+r10+BroadcastOffset]>
        EmitIfCountGE Count, 2, <vfmadd231ps zmm6,zmm3,zmm0>
        EmitIfCountGE Count, 2, <vfmadd231ps zmm7,zmm3,zmm1>
        EmitIfCountGE Count, 3, <vbroadcastss zmm3,DWORD PTR [rcx+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 3, <vfmadd231ps zmm8,zmm3,zmm0>
        EmitIfCountGE Count, 3, <vfmadd231ps zmm9,zmm3,zmm1>
        EmitIfCountGE Count, 4, <vbroadcastss zmm3,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCountGE Count, 4, <vfmadd231ps zmm10,zmm3,zmm0>
        EmitIfCountGE Count, 4, <vfmadd231ps zmm11,zmm3,zmm1>
        EmitIfCountGE Count, 5, <vbroadcastss zmm3,DWORD PTR [rbx+r10+BroadcastOffset]>
        EmitIfCountGE Count, 5, <vfmadd231ps zmm12,zmm3,zmm0>
        EmitIfCountGE Count, 5, <vfmadd231ps zmm13,zmm3,zmm1>
        EmitIfCountGE Count, 6, <vbroadcastss zmm3,DWORD PTR [rbx+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 6, <vfmadd231ps zmm14,zmm3,zmm0>
        EmitIfCountGE Count, 6, <vfmadd231ps zmm15,zmm3,zmm1>
        EmitIfCountGE Count, 12, <vbroadcastss zmm3,DWORD PTR [r13+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm16,zmm3,zmm0>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm17,zmm3,zmm1>
        EmitIfCountGE Count, 12, <vbroadcastss zmm3,DWORD PTR [r13+r10+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm18,zmm3,zmm0>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm19,zmm3,zmm1>
        EmitIfCountGE Count, 12, <vbroadcastss zmm3,DWORD PTR [r13+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm20,zmm3,zmm0>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm21,zmm3,zmm1>
        EmitIfCountGE Count, 12, <vbroadcastss zmm3,DWORD PTR [r14+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm22,zmm3,zmm0>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm23,zmm3,zmm1>
        EmitIfCountGE Count, 12, <vbroadcastss zmm3,DWORD PTR [r14+r10+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm24,zmm3,zmm0>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm25,zmm3,zmm1>
        EmitIfCountGE Count, 12, <vbroadcastss zmm3,DWORD PTR [r14+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm26,zmm3,zmm0>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm27,zmm3,zmm1>
ENDIF

        ENDM

;
; ComputeBlockAvx512FBy16
;
;   This macro multiplies and accumulates for a 16xN block of the output matrix.
;
; Arguments:
;
;   Count - Supplies the number of rows to access from matrix A.
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
;   r13 - Supplies the address into the matrix A data plus 6 rows.
;
;   r14 - Supplies the address into the matrix A data plus 9 rows.
;
;   zmm4-zmm27 - Supplies the block accumulators.
;

ComputeBlockAvx512FBy16 MACRO Count, VectorOffset, BroadcastOffset, PrefetchOffset

IFNB <PrefetchOffset>
        prefetcht0 [rdx+VectorOffset+PrefetchOffset]
ENDIF
        vmovaps zmm0,ZMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE Count, 1, <vfmadd231ps zmm5,zmm0,DWORD BCST [rcx+BroadcastOffset]>
        EmitIfCountGE Count, 2, <vfmadd231ps zmm7,zmm0,DWORD BCST [rcx+r10+BroadcastOffset]>
        EmitIfCountGE Count, 3, <vfmadd231ps zmm9,zmm0,DWORD BCST [rcx+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 4, <vfmadd231ps zmm11,zmm0,DWORD BCST [rbx+BroadcastOffset]>
        EmitIfCountGE Count, 5, <vfmadd231ps zmm13,zmm0,DWORD BCST [rbx+r10+BroadcastOffset]>
        EmitIfCountGE Count, 6, <vfmadd231ps zmm15,zmm0,DWORD BCST [rbx+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm17,zmm0,DWORD BCST [r13+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm19,zmm0,DWORD BCST [r13+r10+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm21,zmm0,DWORD BCST [r13+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm23,zmm0,DWORD BCST [r14+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm25,zmm0,DWORD BCST [r14+r10+BroadcastOffset]>
        EmitIfCountGE Count, 12, <vfmadd231ps zmm27,zmm0,DWORD BCST [r14+r10*2+BroadcastOffset]>

        ENDM

;
; ComputeBlockAvx512FLoop
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
;   rbx - Supplies the address into the matrix A data plus N rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r9 - Supplies the number of columns from matrix A and the number of rows
;       from matrix B to iterate over.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockAvx512FLoop MACRO ComputeBlock, Count

IF Count GT 3
        lea     rbx,[r10*2+r10]
IF Count EQ 12
        lea     r13,[rcx+rbx*2]             ; compute matrix A plus 6 rows
        lea     r14,[r13+rbx]               ; compute matrix A plus 9 rows
ENDIF
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF
        ComputeBlockLoop ComputeBlock, Count, <Count GT 3>
IF Count GT 3
        lea     rbx,[rax*2+rax]
IF Count EQ 12
        lea     r13,[r8+rbx*2]              ; compute matrix C plus 6 rows
        lea     r14,[r13+rbx]               ; compute matrix C plus 9 rows
ENDIF
        add     rbx,r8                      ; compute matrix C plus 3 rows
ENDIF
        ENDM

;
; ProcessCountMAvx512F
;
; Macro Description:
;
;   This macro generates code to compute matrix multiplication for a fixed set
;   of rows.
;
; Arguments:
;
;   Mode - Supplies the mode of operation for updating the contents of matrix C.
;
;   Count - Supplies the number of rows to process.
;
; Implicit Arguments:
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

ProcessCountMAvx512F MACRO Mode, Count

        LOCAL   ProcessNextColumnLoop32xN
        LOCAL   Output16xNBlock
        LOCAL   Output16xNBlockWithMask
        LOCAL   ProcessRemainingCountN

        cmp     rbp,16
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop32xN:
        EmitIfCountGE Count, 12, <vmovaps zmm16,zmm4>
                                            ; clear upper block accumulators
        EmitIfCountGE Count, 12, <vmovaps zmm17,zmm5>
        EmitIfCountGE Count, 12, <vmovaps zmm18,zmm4>
        EmitIfCountGE Count, 12, <vmovaps zmm19,zmm5>
        EmitIfCountGE Count, 12, <vmovaps zmm20,zmm4>
        EmitIfCountGE Count, 12, <vmovaps zmm21,zmm5>
        EmitIfCountGE Count, 12, <vmovaps zmm22,zmm4>
        EmitIfCountGE Count, 12, <vmovaps zmm23,zmm5>
        EmitIfCountGE Count, 12, <vmovaps zmm24,zmm4>
        EmitIfCountGE Count, 12, <vmovaps zmm25,zmm5>
        EmitIfCountGE Count, 12, <vmovaps zmm26,zmm4>
        EmitIfCountGE Count, 12, <vmovaps zmm27,zmm5>
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy32, Count
        add     rdx,r12                     ; advance matrix B by 16*CountK floats
IFDIFI <Mode>, <Add>
        EmitIfCountGE Count, 1, <vmulps zmm4,zmm4,zmm31>
        EmitIfCountGE Count, 2, <vmulps zmm6,zmm6,zmm31>
        EmitIfCountGE Count, 3, <vmulps zmm8,zmm8,zmm31>
        EmitIfCountGE Count, 4, <vmulps zmm10,zmm10,zmm31>
        EmitIfCountGE Count, 5, <vmulps zmm12,zmm12,zmm31>
        EmitIfCountGE Count, 6, <vmulps zmm14,zmm14,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm16,zmm16,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm18,zmm18,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm20,zmm20,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm22,zmm22,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm24,zmm24,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm26,zmm26,zmm31>
ELSE
        EmitIfCountGE Count, 1, <vfmadd213ps zmm4,zmm31,ZMMWORD PTR [r8]>
        EmitIfCountGE Count, 2, <vfmadd213ps zmm6,zmm31,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE Count, 3, <vfmadd213ps zmm8,zmm31,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE Count, 4, <vfmadd213ps zmm10,zmm31,ZMMWORD PTR [rbx]>
        EmitIfCountGE Count, 5, <vfmadd213ps zmm12,zmm31,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE Count, 6, <vfmadd213ps zmm14,zmm31,ZMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm16,zmm31,ZMMWORD PTR [r13]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm18,zmm31,ZMMWORD PTR [r13+rax]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm20,zmm31,ZMMWORD PTR [r13+rax*2]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm22,zmm31,ZMMWORD PTR [r14]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm24,zmm31,ZMMWORD PTR [r14+rax]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm26,zmm31,ZMMWORD PTR [r14+rax*2]>
ENDIF
        EmitIfCountGE Count, 1, <vmovups ZMMWORD PTR [r8],zmm4>
        EmitIfCountGE Count, 2, <vmovups ZMMWORD PTR [r8+rax],zmm6>
        EmitIfCountGE Count, 3, <vmovups ZMMWORD PTR [r8+rax*2],zmm8>
        EmitIfCountGE Count, 4, <vmovups ZMMWORD PTR [rbx],zmm10>
        EmitIfCountGE Count, 5, <vmovups ZMMWORD PTR [rbx+rax],zmm12>
        EmitIfCountGE Count, 6, <vmovups ZMMWORD PTR [rbx+rax*2],zmm14>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r13],zmm16>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r13+rax],zmm18>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r13+rax*2],zmm20>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r14],zmm22>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r14+rax],zmm24>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r14+rax*2],zmm26>
        add     r8,16*4                     ; advance matrix C by 16 columns
IF Count GT 3
        add     rbx,16*4                    ; advance matrix C plus 3 rows by 16 columns
        add     r13,16*4                    ; advance matrix C plus 6 rows by 16 columns
        add     r14,16*4                    ; advance matrix C plus 9 rows by 16 columns
ENDIF
        sub     rbp,16

Output16xNBlock:
        sub     rbp,16
        jae     Output16xNBlockWithMask
        lea     ecx,[ebp+16]                ; correct for over-subtract above
        mov     edi,1
        shl     edi,cl
        dec     edi
        kmovw   k1,edi                      ; update mask for remaining columns
        xor     ebp,ebp                     ; no more columns remaining

Output16xNBlockWithMask:
IFDIFI <Mode>, <Add>
        EmitIfCountGE Count, 1, <vmulps zmm5,zmm5,zmm31>
        EmitIfCountGE Count, 2, <vmulps zmm7,zmm7,zmm31>
        EmitIfCountGE Count, 3, <vmulps zmm9,zmm9,zmm31>
        EmitIfCountGE Count, 4, <vmulps zmm11,zmm11,zmm31>
        EmitIfCountGE Count, 5, <vmulps zmm13,zmm13,zmm31>
        EmitIfCountGE Count, 6, <vmulps zmm15,zmm15,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm17,zmm17,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm19,zmm19,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm21,zmm21,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm23,zmm23,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm25,zmm25,zmm31>
        EmitIfCountGE Count, 12, <vmulps zmm27,zmm27,zmm31>
ELSE
        EmitIfCountGE Count, 1, <vfmadd213ps zmm5{k1},zmm31,ZMMWORD PTR [r8]>
        EmitIfCountGE Count, 2, <vfmadd213ps zmm7{k1},zmm31,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE Count, 3, <vfmadd213ps zmm9{k1},zmm31,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE Count, 4, <vfmadd213ps zmm11{k1},zmm31,ZMMWORD PTR [rbx]>
        EmitIfCountGE Count, 5, <vfmadd213ps zmm13{k1},zmm31,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE Count, 6, <vfmadd213ps zmm15{k1},zmm31,ZMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm17{k1},zmm31,ZMMWORD PTR [r13]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm19{k1},zmm31,ZMMWORD PTR [r13+rax]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm21{k1},zmm31,ZMMWORD PTR [r13+rax*2]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm23{k1},zmm31,ZMMWORD PTR [r14]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm25{k1},zmm31,ZMMWORD PTR [r14+rax]>
        EmitIfCountGE Count, 12, <vfmadd213ps zmm27{k1},zmm31,ZMMWORD PTR [r14+rax*2]>
ENDIF
        EmitIfCountGE Count, 1, <vmovups ZMMWORD PTR [r8]{k1},zmm5>
        EmitIfCountGE Count, 2, <vmovups ZMMWORD PTR [r8+rax]{k1},zmm7>
        EmitIfCountGE Count, 3, <vmovups ZMMWORD PTR [r8+rax*2]{k1},zmm9>
        EmitIfCountGE Count, 4, <vmovups ZMMWORD PTR [rbx]{k1},zmm11>
        EmitIfCountGE Count, 5, <vmovups ZMMWORD PTR [rbx+rax]{k1},zmm13>
        EmitIfCountGE Count, 6, <vmovups ZMMWORD PTR [rbx+rax*2]{k1},zmm15>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r13]{k1},zmm17>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r13+rax]{k1},zmm19>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r13+rax*2]{k1},zmm21>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r14]{k1},zmm23>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r14+rax]{k1},zmm25>
        EmitIfCountGE Count, 12, <vmovups ZMMWORD PTR [r14+rax*2]{k1},zmm27>
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,16
        ja      ProcessNextColumnLoop32xN
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN:
        EmitIfCountGE Count, 12, <vmovaps zmm17,zmm5>
                                            ; clear upper block accumulators
        EmitIfCountGE Count, 12, <vmovaps zmm19,zmm5>
        EmitIfCountGE Count, 12, <vmovaps zmm21,zmm5>
        EmitIfCountGE Count, 12, <vmovaps zmm23,zmm5>
        EmitIfCountGE Count, 12, <vmovaps zmm25,zmm5>
        EmitIfCountGE Count, 12, <vmovaps zmm27,zmm5>
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy16, Count
        jmp     Output16xNBlock

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
; Process N rows of the matrices.
;

        cmp     r11,12
        jb      ProcessCountMLessThan12
        mov     r11d,12                     ; return 12 rows handled
        ProcessCountMAvx512F Mode, 12

ProcessCountMLessThan12:
        cmp     r11,5
        ja      ProcessCountM6
        je      ProcessCountM5
        cmp     r11,3
        ja      ProcessCountM4
        je      ProcessCountM3
        cmp     r11,1
        je      ProcessCountM1

ProcessCountM2:
        ProcessCountMAvx512F Mode, 2

ProcessCountM4:
        ProcessCountMAvx512F Mode, 4

ProcessCountM6:
        mov     r11d,6                      ; return 6 rows handled
        ProcessCountMAvx512F Mode, 6

;
; Restore non-volatile registers and return.
;

ExitKernel:
        SgemmKernelAvxExit RestoreExtra

ProcessCountM1:
        ProcessCountMAvx512F Mode, 1

ProcessCountM3:
        ProcessCountMAvx512F Mode, 3

ProcessCountM5:
        ProcessCountMAvx512F Mode, 5

        NESTED_END MlasSgemmKernel&Mode&Avx512F, _TEXT

        ENDM

SgemmKernelAvx512FFunction Zero
SgemmKernelAvx512FFunction Add

        END
