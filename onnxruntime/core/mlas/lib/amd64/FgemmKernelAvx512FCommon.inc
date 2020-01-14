;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   FgemmKernelAvx512FCommon.inc
;
; Abstract:
;
;   This module implements the kernels for the floating point matrix/matrix
;   multiply operation (SGEMM and DGEMM).
;
;   This implementation uses AVX512F instructions.
;
;--

;
; Macro Description:
;
;   This macro multiplies and accumulates for 2 ZMMWORDs by N rows of the output
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
;   r13 - Supplies the address into the matrix A data plus 6 rows.
;
;   r14 - Supplies the address into the matrix A data plus 9 rows.
;
;   zmm4-zmm27 - Supplies the block accumulators.
;

ComputeBlockAvx512FBy2 MACRO RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

IFNB <PrefetchOffset>
        prefetcht0 [rdx+VectorOffset+PrefetchOffset]
        prefetcht0 [rdx+r12+VectorOffset+PrefetchOffset]
ENDIF
IF RowCount EQ 1
        vbroadcastsf zmm3,FgemmElementPtr [rcx+BroadcastOffset]
        vfmadd231pf zmm4,zmm3,ZMMWORD PTR [rdx+VectorOffset]
        vfmadd231pf zmm5,zmm3,ZMMWORD PTR [rdx+r12+VectorOffset]
ELSE
        vmovapf zmm0,ZMMWORD PTR [rdx+VectorOffset]
        vmovapf zmm1,ZMMWORD PTR [rdx+r12+VectorOffset]
        EmitIfCountGE RowCount, 1, <vbroadcastsf zmm3,FgemmElementPtr [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 1, <vfmadd231pf zmm4,zmm3,zmm0>
        EmitIfCountGE RowCount, 1, <vfmadd231pf zmm5,zmm3,zmm1>
        EmitIfCountGE RowCount, 2, <vbroadcastsf zmm3,FgemmElementPtr [rcx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <vfmadd231pf zmm6,zmm3,zmm0>
        EmitIfCountGE RowCount, 2, <vfmadd231pf zmm7,zmm3,zmm1>
        EmitIfCountGE RowCount, 3, <vbroadcastsf zmm3,FgemmElementPtr [rcx+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <vfmadd231pf zmm8,zmm3,zmm0>
        EmitIfCountGE RowCount, 3, <vfmadd231pf zmm9,zmm3,zmm1>
        EmitIfCountGE RowCount, 4, <vbroadcastsf zmm3,FgemmElementPtr [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <vfmadd231pf zmm10,zmm3,zmm0>
        EmitIfCountGE RowCount, 4, <vfmadd231pf zmm11,zmm3,zmm1>
        EmitIfCountGE RowCount, 5, <vbroadcastsf zmm3,FgemmElementPtr [rbx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 5, <vfmadd231pf zmm12,zmm3,zmm0>
        EmitIfCountGE RowCount, 5, <vfmadd231pf zmm13,zmm3,zmm1>
        EmitIfCountGE RowCount, 6, <vbroadcastsf zmm3,FgemmElementPtr [rbx+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 6, <vfmadd231pf zmm14,zmm3,zmm0>
        EmitIfCountGE RowCount, 6, <vfmadd231pf zmm15,zmm3,zmm1>
        EmitIfCountGE RowCount, 12, <vbroadcastsf zmm3,FgemmElementPtr [r13+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm16,zmm3,zmm0>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm17,zmm3,zmm1>
        EmitIfCountGE RowCount, 12, <vbroadcastsf zmm3,FgemmElementPtr [r13+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm18,zmm3,zmm0>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm19,zmm3,zmm1>
        EmitIfCountGE RowCount, 12, <vbroadcastsf zmm3,FgemmElementPtr [r13+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm20,zmm3,zmm0>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm21,zmm3,zmm1>
        EmitIfCountGE RowCount, 12, <vbroadcastsf zmm3,FgemmElementPtr [r14+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm22,zmm3,zmm0>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm23,zmm3,zmm1>
        EmitIfCountGE RowCount, 12, <vbroadcastsf zmm3,FgemmElementPtr [r14+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm24,zmm3,zmm0>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm25,zmm3,zmm1>
        EmitIfCountGE RowCount, 12, <vbroadcastsf zmm3,FgemmElementPtr [r14+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm26,zmm3,zmm0>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm27,zmm3,zmm1>
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro multiplies and accumulates for 1 ZMMWORD by N rows of the output
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
;   r13 - Supplies the address into the matrix A data plus 6 rows.
;
;   r14 - Supplies the address into the matrix A data plus 9 rows.
;
;   zmm4-zmm27 - Supplies the block accumulators.
;

ComputeBlockAvx512FBy1 MACRO RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

IFNB <PrefetchOffset>
        prefetcht0 [rdx+VectorOffset+PrefetchOffset]
ENDIF
        vmovapf zmm0,ZMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE RowCount, 1, <vfmadd231pf zmm5,zmm0,FgemmElementBcst [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <vfmadd231pf zmm7,zmm0,FgemmElementBcst [rcx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <vfmadd231pf zmm9,zmm0,FgemmElementBcst [rcx+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <vfmadd231pf zmm11,zmm0,FgemmElementBcst [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 5, <vfmadd231pf zmm13,zmm0,FgemmElementBcst [rbx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 6, <vfmadd231pf zmm15,zmm0,FgemmElementBcst [rbx+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm17,zmm0,FgemmElementBcst [r13+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm19,zmm0,FgemmElementBcst [r13+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm21,zmm0,FgemmElementBcst [r13+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm23,zmm0,FgemmElementBcst [r14+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm25,zmm0,FgemmElementBcst [r14+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 12, <vfmadd231pf zmm27,zmm0,FgemmElementBcst [r14+r10*2+BroadcastOffset]>

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

ComputeBlockAvx512FLoop MACRO ComputeBlock, RowCount

IF RowCount GT 3
        lea     rbx,[r10*2+r10]
IF RowCount EQ 12
        lea     r13,[rcx+rbx*2]             ; compute matrix A plus 6 rows
        lea     r14,[r13+rbx]               ; compute matrix A plus 9 rows
ENDIF
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF
        ComputeBlockLoop ComputeBlock, RowCount, <RowCount GT 3>
IF RowCount GT 3
        lea     rbx,[rax*2+rax]
IF RowCount EQ 12
        lea     r13,[r8+rbx*2]              ; compute matrix C plus 6 rows
        lea     r14,[r13+rbx]               ; compute matrix C plus 9 rows
ENDIF
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

ProcessCountM MACRO RowCount

        LOCAL   ProcessNextColumnLoop2xN
        LOCAL   MultiplyAlpha2xNBlock
        LOCAL   Store2xNBlock
        LOCAL   Output1xNBlock
        LOCAL   Output1xNBlockWithMask
        LOCAL   MultiplyAlpha1xNBlockWithMask
        LOCAL   Store1xNBlockWithMask
        LOCAL   ProcessRemainingCountN

        cmp     rbp,FgemmZmmElementCount
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop2xN:
        EmitIfCountGE RowCount, 12, <vmovapf zmm16,zmm4>
                                            ; clear upper block accumulators
        EmitIfCountGE RowCount, 12, <vmovapf zmm17,zmm5>
        EmitIfCountGE RowCount, 12, <vmovapf zmm18,zmm4>
        EmitIfCountGE RowCount, 12, <vmovapf zmm19,zmm5>
        EmitIfCountGE RowCount, 12, <vmovapf zmm20,zmm4>
        EmitIfCountGE RowCount, 12, <vmovapf zmm21,zmm5>
        EmitIfCountGE RowCount, 12, <vmovapf zmm22,zmm4>
        EmitIfCountGE RowCount, 12, <vmovapf zmm23,zmm5>
        EmitIfCountGE RowCount, 12, <vmovapf zmm24,zmm4>
        EmitIfCountGE RowCount, 12, <vmovapf zmm25,zmm5>
        EmitIfCountGE RowCount, 12, <vmovapf zmm26,zmm4>
        EmitIfCountGE RowCount, 12, <vmovapf zmm27,zmm5>
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy2, RowCount
        add     rdx,r12                     ; advance matrix B by 64*CountK bytes
        test    r15b,r15b                   ; ZeroMode?
        jnz     MultiplyAlpha2xNBlock
        EmitIfCountGE RowCount, 1, <vfmadd213pf zmm4,zmm31,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vfmadd213pf zmm6,zmm31,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vfmadd213pf zmm8,zmm31,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vfmadd213pf zmm10,zmm31,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vfmadd213pf zmm12,zmm31,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vfmadd213pf zmm14,zmm31,ZMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm16,zmm31,ZMMWORD PTR [r13]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm18,zmm31,ZMMWORD PTR [r13+rax]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm20,zmm31,ZMMWORD PTR [r13+rax*2]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm22,zmm31,ZMMWORD PTR [r14]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm24,zmm31,ZMMWORD PTR [r14+rax]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm26,zmm31,ZMMWORD PTR [r14+rax*2]>
        jmp     Store2xNBlock

MultiplyAlpha2xNBlock:
        EmitIfCountGE RowCount, 1, <vmulpf zmm4,zmm4,zmm31>
        EmitIfCountGE RowCount, 2, <vmulpf zmm6,zmm6,zmm31>
        EmitIfCountGE RowCount, 3, <vmulpf zmm8,zmm8,zmm31>
        EmitIfCountGE RowCount, 4, <vmulpf zmm10,zmm10,zmm31>
        EmitIfCountGE RowCount, 5, <vmulpf zmm12,zmm12,zmm31>
        EmitIfCountGE RowCount, 6, <vmulpf zmm14,zmm14,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm16,zmm16,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm18,zmm18,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm20,zmm20,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm22,zmm22,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm24,zmm24,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm26,zmm26,zmm31>

Store2xNBlock:
        EmitIfCountGE RowCount, 1, <vmovupf ZMMWORD PTR [r8],zmm4>
        EmitIfCountGE RowCount, 2, <vmovupf ZMMWORD PTR [r8+rax],zmm6>
        EmitIfCountGE RowCount, 3, <vmovupf ZMMWORD PTR [r8+rax*2],zmm8>
        EmitIfCountGE RowCount, 4, <vmovupf ZMMWORD PTR [rbx],zmm10>
        EmitIfCountGE RowCount, 5, <vmovupf ZMMWORD PTR [rbx+rax],zmm12>
        EmitIfCountGE RowCount, 6, <vmovupf ZMMWORD PTR [rbx+rax*2],zmm14>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r13],zmm16>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r13+rax],zmm18>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r13+rax*2],zmm20>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r14],zmm22>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r14+rax],zmm24>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r14+rax*2],zmm26>
        add     r8,64                       ; advance matrix C by ZMMWORD
IF RowCount GT 3
        add     rbx,64                      ; advance matrix C plus 3 rows by ZMMWORD
IF RowCount EQ 12
        add     r13,64                      ; advance matrix C plus 6 rows by ZMMWORD
        add     r14,64                      ; advance matrix C plus 9 rows by ZMMWORD
ENDIF
ENDIF
        sub     rbp,FgemmZmmElementCount

Output1xNBlock:
        sub     rbp,FgemmZmmElementCount
        jae     Output1xNBlockWithMask
        lea     ecx,[ebp+FgemmZmmElementCount]
                                            ; correct for over-subtract above
        mov     edi,1
        shl     edi,cl
        dec     edi
        kmovw   k1,edi                      ; update mask for remaining columns
        xor     ebp,ebp                     ; no more columns remaining

Output1xNBlockWithMask:
        test    r15b,r15b                   ; ZeroMode?
        jnz     MultiplyAlpha1xNBlockWithMask
        EmitIfCountGE RowCount, 1, <vfmadd213pf zmm5{k1},zmm31,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vfmadd213pf zmm7{k1},zmm31,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vfmadd213pf zmm9{k1},zmm31,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vfmadd213pf zmm11{k1},zmm31,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vfmadd213pf zmm13{k1},zmm31,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vfmadd213pf zmm15{k1},zmm31,ZMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm17{k1},zmm31,ZMMWORD PTR [r13]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm19{k1},zmm31,ZMMWORD PTR [r13+rax]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm21{k1},zmm31,ZMMWORD PTR [r13+rax*2]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm23{k1},zmm31,ZMMWORD PTR [r14]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm25{k1},zmm31,ZMMWORD PTR [r14+rax]>
        EmitIfCountGE RowCount, 12, <vfmadd213pf zmm27{k1},zmm31,ZMMWORD PTR [r14+rax*2]>
        jmp     Store1xNBlockWithMask

MultiplyAlpha1xNBlockWithMask:
        EmitIfCountGE RowCount, 1, <vmulpf zmm5,zmm5,zmm31>
        EmitIfCountGE RowCount, 2, <vmulpf zmm7,zmm7,zmm31>
        EmitIfCountGE RowCount, 3, <vmulpf zmm9,zmm9,zmm31>
        EmitIfCountGE RowCount, 4, <vmulpf zmm11,zmm11,zmm31>
        EmitIfCountGE RowCount, 5, <vmulpf zmm13,zmm13,zmm31>
        EmitIfCountGE RowCount, 6, <vmulpf zmm15,zmm15,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm17,zmm17,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm19,zmm19,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm21,zmm21,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm23,zmm23,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm25,zmm25,zmm31>
        EmitIfCountGE RowCount, 12, <vmulpf zmm27,zmm27,zmm31>

Store1xNBlockWithMask:
        EmitIfCountGE RowCount, 1, <vmovupf ZMMWORD PTR [r8]{k1},zmm5>
        EmitIfCountGE RowCount, 2, <vmovupf ZMMWORD PTR [r8+rax]{k1},zmm7>
        EmitIfCountGE RowCount, 3, <vmovupf ZMMWORD PTR [r8+rax*2]{k1},zmm9>
        EmitIfCountGE RowCount, 4, <vmovupf ZMMWORD PTR [rbx]{k1},zmm11>
        EmitIfCountGE RowCount, 5, <vmovupf ZMMWORD PTR [rbx+rax]{k1},zmm13>
        EmitIfCountGE RowCount, 6, <vmovupf ZMMWORD PTR [rbx+rax*2]{k1},zmm15>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r13]{k1},zmm17>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r13+rax]{k1},zmm19>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r13+rax*2]{k1},zmm21>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r14]{k1},zmm23>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r14+rax]{k1},zmm25>
        EmitIfCountGE RowCount, 12, <vmovupf ZMMWORD PTR [r14+rax*2]{k1},zmm27>
        add     r8,64                       ; advance matrix C by ZMMWORD
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,FgemmZmmElementCount
        ja      ProcessNextColumnLoop2xN
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN:
        EmitIfCountGE RowCount, 12, <vmovapf zmm17,zmm5>
                                            ; clear upper block accumulators
        EmitIfCountGE RowCount, 12, <vmovapf zmm19,zmm5>
        EmitIfCountGE RowCount, 12, <vmovapf zmm21,zmm5>
        EmitIfCountGE RowCount, 12, <vmovapf zmm23,zmm5>
        EmitIfCountGE RowCount, 12, <vmovapf zmm25,zmm5>
        EmitIfCountGE RowCount, 12, <vmovapf zmm27,zmm5>
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy1, RowCount
        jmp     Output1xNBlock

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

FgemmKernelAvx512FFunction MACRO Type

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

        NESTED_ENTRY MlasGemm&Type&KernelAvx512F, _TEXT

        FgemmKernelEntry Avx512F

        mov     r12,r9
        shl     r12,6                       ; compute 64*CountK bytes
        mov     edi,-1
        kmovw   k1,edi                      ; update mask to write all columns
        vbroadcastsf zmm31,FgemmElementPtr FgemmKernelFrame.Alpha[rsp]

;
; Process CountM rows of the matrices.
;

        cmp     r11,12
        jb      ProcessCountMLessThan12
        mov     r11d,12                     ; return 12 rows handled
        ProcessCountM 12

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
        ProcessCountM 2

ProcessCountM4:
        ProcessCountM 4

ProcessCountM6:
        mov     r11d,6                      ; return 6 rows handled
        ProcessCountM 6

;
; Restore non-volatile registers and return.
;

ExitKernel:
        FgemmKernelExit Avx512F

ProcessCountM1:
        ProcessCountM 1

ProcessCountM3:
        ProcessCountM 3

ProcessCountM5:
        ProcessCountM 5

        NESTED_END MlasGemm&Type&KernelAvx512F, _TEXT

        ENDM
