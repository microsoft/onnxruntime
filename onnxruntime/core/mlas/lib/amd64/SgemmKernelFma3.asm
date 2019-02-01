;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelFma3.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;   This implementation uses AVX fused multiply/add instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE SgemmKernelCommon.inc
        .list

        EXTERN  MlasMaskMoveAvx:NEAR

;
; ComputeBlockFma3By16
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
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockFma3By16 MACRO Count, VectorOffset, BroadcastOffset, PrefetchOffset

IFNB <PrefetchOffset>
        prefetcht0 [rdx+VectorOffset+PrefetchOffset]
ENDIF
IF Count EQ 1
        vbroadcastss ymm3,DWORD PTR [rcx+BroadcastOffset]
        vfmadd231ps ymm4,ymm3,YMMWORD PTR [rdx+VectorOffset]
        vfmadd231ps ymm5,ymm3,YMMWORD PTR [rdx+VectorOffset+32]
ELSE
        vmovaps ymm0,YMMWORD PTR [rdx+VectorOffset]
        vmovaps ymm1,YMMWORD PTR [rdx+VectorOffset+32]
        EmitIfCountGE Count, 1, <vbroadcastss ymm3,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE Count, 1, <vfmadd231ps ymm4,ymm3,ymm0>
        EmitIfCountGE Count, 1, <vfmadd231ps ymm5,ymm3,ymm1>
        EmitIfCountGE Count, 2, <vbroadcastss ymm3,DWORD PTR [rcx+r10+BroadcastOffset]>
        EmitIfCountGE Count, 2, <vfmadd231ps ymm6,ymm3,ymm0>
        EmitIfCountGE Count, 2, <vfmadd231ps ymm7,ymm3,ymm1>
        EmitIfCountGE Count, 3, <vbroadcastss ymm3,DWORD PTR [rcx+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 3, <vfmadd231ps ymm8,ymm3,ymm0>
        EmitIfCountGE Count, 3, <vfmadd231ps ymm9,ymm3,ymm1>
        EmitIfCountGE Count, 4, <vbroadcastss ymm3,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCountGE Count, 4, <vfmadd231ps ymm10,ymm3,ymm0>
        EmitIfCountGE Count, 4, <vfmadd231ps ymm11,ymm3,ymm1>
        EmitIfCountGE Count, 5, <vbroadcastss ymm3,DWORD PTR [rbx+r10+BroadcastOffset]>
        EmitIfCountGE Count, 5, <vfmadd231ps ymm12,ymm3,ymm0>
        EmitIfCountGE Count, 5, <vfmadd231ps ymm13,ymm3,ymm1>
        EmitIfCountGE Count, 6, <vbroadcastss ymm3,DWORD PTR [rbx+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 6, <vfmadd231ps ymm14,ymm3,ymm0>
        EmitIfCountGE Count, 6, <vfmadd231ps ymm15,ymm3,ymm1>
ENDIF

        ENDM

;
; ComputeBlockFma3By8
;
; Macro Description:
;
;   This macro multiplies and accumulates for a 8xN block of the output matrix.
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
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockFma3By8 MACRO Count, VectorOffset, BroadcastOffset, PrefetchOffset

IFNB <PrefetchOffset>
        prefetcht0 [rdx+VectorOffset+PrefetchOffset]
ENDIF
IF Count EQ 1
        vbroadcastss ymm3,DWORD PTR [rcx+BroadcastOffset]
        vfmadd231ps ymm5,ymm3,YMMWORD PTR [rdx+VectorOffset]
ELSE
        vmovaps ymm0,YMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE Count, 1, <vbroadcastss ymm3,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE Count, 1, <vfmadd231ps ymm5,ymm3,ymm0>
        EmitIfCountGE Count, 2, <vbroadcastss ymm3,DWORD PTR [rcx+r10+BroadcastOffset]>
        EmitIfCountGE Count, 2, <vfmadd231ps ymm7,ymm3,ymm0>
        EmitIfCountGE Count, 3, <vbroadcastss ymm3,DWORD PTR [rcx+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 3, <vfmadd231ps ymm9,ymm3,ymm0>
        EmitIfCountGE Count, 4, <vbroadcastss ymm3,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCountGE Count, 4, <vfmadd231ps ymm11,ymm3,ymm0>
        EmitIfCountGE Count, 5, <vbroadcastss ymm3,DWORD PTR [rbx+r10+BroadcastOffset]>
        EmitIfCountGE Count, 5, <vfmadd231ps ymm13,ymm3,ymm0>
        EmitIfCountGE Count, 6, <vbroadcastss ymm3,DWORD PTR [rbx+r10*2+BroadcastOffset]>
        EmitIfCountGE Count, 6, <vfmadd231ps ymm15,ymm3,ymm0>
ENDIF

        ENDM

;
; ComputeBlockFma3Loop
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

ComputeBlockFma3Loop MACRO ComputeBlock, Count

IF Count GT 3
        lea     rbx,[r10*2+r10]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF
        ComputeBlockLoop ComputeBlock, Count, <Count GT 3>
        vbroadcastss ymm2,DWORD PTR SgemmKernelFrame.Alpha[rsp]
IF Count GT 3
        lea     rbx,[rax*2+rax]
        add     rbx,r8                      ; compute matrix C plus 3 rows
ENDIF

        ENDM

;
; ProcessCountMFma3
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

ProcessCountMFma3 MACRO Mode, Count, Fallthrough

        LOCAL   ProcessNextColumnLoop16xN
        LOCAL   ProcessRemainingCountN
        LOCAL   OutputMasked8xNBlock
        LOCAL   OutputMasked16xNBlock

        cmp     rbp,8
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop16xN:
        ComputeBlockFma3Loop ComputeBlockFma3By16, Count
        EmitIfCountGE Count, 1, <prefetcht0 [r8+64]>
        EmitIfCountGE Count, 2, <prefetcht0 [r8+rax+64]>
        EmitIfCountGE Count, 3, <prefetcht0 [r8+rax*2+64]>
        EmitIfCountGE Count, 4, <prefetcht0 [rbx+64]>
        EmitIfCountGE Count, 5, <prefetcht0 [rbx+rax+64]>
        EmitIfCountGE Count, 6, <prefetcht0 [rbx+rax*2+64]>
IFDIFI <Mode>, <Add>
        EmitIfCountGE Count, 1, <vmulps ymm4,ymm4,ymm2>
        EmitIfCountGE Count, 1, <vmulps ymm5,ymm5,ymm2>
        EmitIfCountGE Count, 2, <vmulps ymm6,ymm6,ymm2>
        EmitIfCountGE Count, 2, <vmulps ymm7,ymm7,ymm2>
        EmitIfCountGE Count, 3, <vmulps ymm8,ymm8,ymm2>
        EmitIfCountGE Count, 3, <vmulps ymm9,ymm9,ymm2>
        EmitIfCountGE Count, 4, <vmulps ymm10,ymm10,ymm2>
        EmitIfCountGE Count, 4, <vmulps ymm11,ymm11,ymm2>
        EmitIfCountGE Count, 5, <vmulps ymm12,ymm12,ymm2>
        EmitIfCountGE Count, 5, <vmulps ymm13,ymm13,ymm2>
        EmitIfCountGE Count, 6, <vmulps ymm14,ymm14,ymm2>
        EmitIfCountGE Count, 6, <vmulps ymm15,ymm15,ymm2>
ENDIF
        sub     rbp,16
        jb      OutputMasked16xNBlock
IFIDNI <Mode>, <Add>
        EmitIfCountGE Count, 1, <vfmadd213ps ymm4,ymm2,YMMWORD PTR [r8]>
        EmitIfCountGE Count, 1, <vfmadd213ps ymm5,ymm2,YMMWORD PTR [r8+32]>
        EmitIfCountGE Count, 2, <vfmadd213ps ymm6,ymm2,YMMWORD PTR [r8+rax]>
        EmitIfCountGE Count, 2, <vfmadd213ps ymm7,ymm2,YMMWORD PTR [r8+rax+32]>
        EmitIfCountGE Count, 3, <vfmadd213ps ymm8,ymm2,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE Count, 3, <vfmadd213ps ymm9,ymm2,YMMWORD PTR [r8+rax*2+32]>
        EmitIfCountGE Count, 4, <vfmadd213ps ymm10,ymm2,YMMWORD PTR [rbx]>
        EmitIfCountGE Count, 4, <vfmadd213ps ymm11,ymm2,YMMWORD PTR [rbx+32]>
        EmitIfCountGE Count, 5, <vfmadd213ps ymm12,ymm2,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE Count, 5, <vfmadd213ps ymm13,ymm2,YMMWORD PTR [rbx+rax+32]>
        EmitIfCountGE Count, 6, <vfmadd213ps ymm14,ymm2,YMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE Count, 6, <vfmadd213ps ymm15,ymm2,YMMWORD PTR [rbx+rax*2+32]>
ENDIF
        EmitIfCountGE Count, 1, <vmovups YMMWORD PTR [r8],ymm4>
        EmitIfCountGE Count, 1, <vmovups YMMWORD PTR [r8+32],ymm5>
        EmitIfCountGE Count, 2, <vmovups YMMWORD PTR [r8+rax],ymm6>
        EmitIfCountGE Count, 2, <vmovups YMMWORD PTR [r8+rax+32],ymm7>
        EmitIfCountGE Count, 3, <vmovups YMMWORD PTR [r8+rax*2],ymm8>
        EmitIfCountGE Count, 3, <vmovups YMMWORD PTR [r8+rax*2+32],ymm9>
        EmitIfCountGE Count, 4, <vmovups YMMWORD PTR [rbx],ymm10>
        EmitIfCountGE Count, 4, <vmovups YMMWORD PTR [rbx+32],ymm11>
        EmitIfCountGE Count, 5, <vmovups YMMWORD PTR [rbx+rax],ymm12>
        EmitIfCountGE Count, 5, <vmovups YMMWORD PTR [rbx+rax+32],ymm13>
        EmitIfCountGE Count, 6, <vmovups YMMWORD PTR [rbx+rax*2],ymm14>
        EmitIfCountGE Count, 6, <vmovups YMMWORD PTR [rbx+rax*2+32],ymm15>
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,8
        ja      ProcessNextColumnLoop16xN
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN:
        ComputeBlockFma3Loop ComputeBlockFma3By8, Count
IFDIFI <Mode>, <Add>
        EmitIfCountGE Count, 1, <vmulps ymm5,ymm5,ymm2>
        EmitIfCountGE Count, 2, <vmulps ymm7,ymm7,ymm2>
        EmitIfCountGE Count, 3, <vmulps ymm9,ymm9,ymm2>
        EmitIfCountGE Count, 4, <vmulps ymm11,ymm11,ymm2>
        EmitIfCountGE Count, 5, <vmulps ymm13,ymm13,ymm2>
        EmitIfCountGE Count, 6, <vmulps ymm15,ymm15,ymm2>
ENDIF
        cmp     rbp,8
        jb      OutputMasked8xNBlock
IFIDNI <Mode>, <Add>
        EmitIfCountGE Count, 1, <vfmadd213ps ymm5,ymm2,YMMWORD PTR [r8]>
        EmitIfCountGE Count, 2, <vfmadd213ps ymm7,ymm2,YMMWORD PTR [r8+rax]>
        EmitIfCountGE Count, 3, <vfmadd213ps ymm9,ymm2,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE Count, 4, <vfmadd213ps ymm11,ymm2,YMMWORD PTR [rbx]>
        EmitIfCountGE Count, 5, <vfmadd213ps ymm13,ymm2,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE Count, 6, <vfmadd213ps ymm15,ymm2,YMMWORD PTR [rbx+rax*2]>
ENDIF
        EmitIfCountGE Count, 1, <vmovups YMMWORD PTR [r8],ymm5>
        EmitIfCountGE Count, 2, <vmovups YMMWORD PTR [r8+rax],ymm7>
        EmitIfCountGE Count, 3, <vmovups YMMWORD PTR [r8+rax*2],ymm9>
        EmitIfCountGE Count, 4, <vmovups YMMWORD PTR [rbx],ymm11>
        EmitIfCountGE Count, 5, <vmovups YMMWORD PTR [rbx+rax],ymm13>
        EmitIfCountGE Count, 6, <vmovups YMMWORD PTR [rbx+rax*2],ymm15>
        jmp     ExitKernelAndZeroUpper

OutputMasked16xNBlock:
IFIDNI <Mode>, <Add>
        EmitIfCountGE Count, 1, <vfmadd213ps ymm4,ymm2,YMMWORD PTR [r8]>
        EmitIfCountGE Count, 2, <vfmadd213ps ymm6,ymm2,YMMWORD PTR [r8+rax]>
        EmitIfCountGE Count, 3, <vfmadd213ps ymm8,ymm2,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE Count, 4, <vfmadd213ps ymm10,ymm2,YMMWORD PTR [rbx]>
        EmitIfCountGE Count, 5, <vfmadd213ps ymm12,ymm2,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE Count, 6, <vfmadd213ps ymm14,ymm2,YMMWORD PTR [rbx+rax*2]>
ENDIF
        EmitIfCountGE Count, 1, <vmovups YMMWORD PTR [r8],ymm4>
        EmitIfCountGE Count, 2, <vmovups YMMWORD PTR [r8+rax],ymm6>
        EmitIfCountGE Count, 3, <vmovups YMMWORD PTR [r8+rax*2],ymm8>
        EmitIfCountGE Count, 4, <vmovups YMMWORD PTR [rbx],ymm10>
        EmitIfCountGE Count, 5, <vmovups YMMWORD PTR [rbx+rax],ymm12>
        EmitIfCountGE Count, 6, <vmovups YMMWORD PTR [rbx+rax*2],ymm14>
        add     r8,8*4                      ; advance matrix C by 8 columns
IF Count GT 3
        add     rbx,8*4                     ; advance matrix C plus 3 rows by 8 columns
ENDIF
        add     rbp,8                       ; correct for over-subtract above

OutputMasked8xNBlock:
        mov     DWORD PTR SgemmKernelFrame.CountN[rsp],ebp
        vbroadcastss ymm0,DWORD PTR SgemmKernelFrame.CountN[rsp]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR [MlasMaskMoveAvx]
IFIDNI <Mode>, <Add>
        EmitIfCountGE Count, 1, <vmaskmovps ymm4,ymm0,YMMWORD PTR [r8]>
        EmitIfCountGE Count, 2, <vmaskmovps ymm6,ymm0,YMMWORD PTR [r8+rax]>
        EmitIfCountGE Count, 3, <vmaskmovps ymm8,ymm0,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE Count, 4, <vmaskmovps ymm10,ymm0,YMMWORD PTR [rbx]>
        EmitIfCountGE Count, 5, <vmaskmovps ymm12,ymm0,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE Count, 6, <vmaskmovps ymm14,ymm0,YMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE Count, 1, <vfmadd213ps ymm5,ymm2,ymm4>
        EmitIfCountGE Count, 2, <vfmadd213ps ymm7,ymm2,ymm6>
        EmitIfCountGE Count, 3, <vfmadd213ps ymm9,ymm2,ymm8>
        EmitIfCountGE Count, 4, <vfmadd213ps ymm11,ymm2,ymm10>
        EmitIfCountGE Count, 5, <vfmadd213ps ymm13,ymm2,ymm12>
        EmitIfCountGE Count, 6, <vfmadd213ps ymm15,ymm2,ymm14>
ENDIF
        EmitIfCountGE Count, 1, <vmaskmovps YMMWORD PTR [r8],ymm0,ymm5>
        EmitIfCountGE Count, 2, <vmaskmovps YMMWORD PTR [r8+rax],ymm0,ymm7>
        EmitIfCountGE Count, 3, <vmaskmovps YMMWORD PTR [r8+rax*2],ymm0,ymm9>
        EmitIfCountGE Count, 4, <vmaskmovps YMMWORD PTR [rbx],ymm0,ymm11>
        EmitIfCountGE Count, 5, <vmaskmovps YMMWORD PTR [rbx+rax],ymm0,ymm13>
        EmitIfCountGE Count, 6, <vmaskmovps YMMWORD PTR [rbx+rax*2],ymm0,ymm15>
IFB <Fallthrough>
        jmp     ExitKernelAndZeroUpper
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
;   Beta - Supplies the scaler multiplier (see SGEMM definition).
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

SgemmKernelFma3Function MACRO Mode

        NESTED_ENTRY MlasSgemmKernel&Mode&Fma3, _TEXT

        SgemmKernelAvxEntry

;
; Process N rows of the matrices.
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
        ProcessCountMFma3 Mode, 2

ProcessCountM4:
        ProcessCountMFma3 Mode, 4

ProcessCountM6:
        mov     r11d,6                      ; return 6 rows handled
        ProcessCountMFma3 Mode, 6, Fallthrough

;
; Restore non-volatile registers and return.
;

ExitKernelAndZeroUpper:
        vzeroupper

ExitKernel:
        SgemmKernelAvxExit

ProcessCountM1:
        ProcessCountMFma3 Mode, 1

ProcessCountM3:
        ProcessCountMFma3 Mode, 3

ProcessCountM5:
        ProcessCountMFma3 Mode, 5

        NESTED_END MlasSgemmKernel&Mode&Fma3, _TEXT

        ENDM

SgemmKernelFma3Function Zero
SgemmKernelFma3Function Add

        END
