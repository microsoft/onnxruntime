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
; ComputeBlockFma3By32
;
;   This macro multiplies and accumulates for a 32xN block (where N is 1,3)
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
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockFma3By32 MACRO Count, VectorOffset, BroadcastOffset

IF Count EQ 1
        vbroadcastss ymm3,DWORD PTR [rcx+BroadcastOffset]
        vfmadd231ps ymm4,ymm3,YMMWORD PTR [rdx+VectorOffset]
        vfmadd231ps ymm5,ymm3,YMMWORD PTR [rdx+VectorOffset+32]
        vfmadd231ps ymm6,ymm3,YMMWORD PTR [rdx+rbx+VectorOffset]
        vfmadd231ps ymm7,ymm3,YMMWORD PTR [rdx+rbx+VectorOffset+32]
ENDIF

        ENDM

;
; ComputeBlockFma3By16
;
;   This macro multiplies and accumulates for a 16xN block (where N is 1,3,6)
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
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockFma3By16 MACRO Count, VectorOffset, BroadcastOffset

IF Count EQ 1
        vbroadcastss ymm3,DWORD PTR [rcx+BroadcastOffset]
        vfmadd231ps ymm4,ymm3,YMMWORD PTR [rdx+VectorOffset]
        vfmadd231ps ymm5,ymm3,YMMWORD PTR [rdx+VectorOffset+32]
ELSE
        vmovaps ymm0,YMMWORD PTR [rdx+VectorOffset]
        vmovaps ymm1,YMMWORD PTR [rdx+VectorOffset+32]
        vbroadcastss ymm3,DWORD PTR [rcx+BroadcastOffset]
        vfmadd231ps ymm4,ymm3,ymm0
        vfmadd231ps ymm5,ymm3,ymm1
IF Count GE 3
        vbroadcastss ymm3,DWORD PTR [rcx+r10+BroadcastOffset]
        vfmadd231ps ymm6,ymm3,ymm0
        vfmadd231ps ymm7,ymm3,ymm1
        vbroadcastss ymm3,DWORD PTR [rcx+r10*2+BroadcastOffset]
        vfmadd231ps ymm8,ymm3,ymm0
        vfmadd231ps ymm9,ymm3,ymm1
ENDIF
IF Count GE 6
        vbroadcastss ymm3,DWORD PTR [rbx+BroadcastOffset]
        vfmadd231ps ymm10,ymm3,ymm0
        vfmadd231ps ymm11,ymm3,ymm1
        vbroadcastss ymm3,DWORD PTR [rbx+r10+BroadcastOffset]
        vfmadd231ps ymm12,ymm3,ymm0
        vfmadd231ps ymm13,ymm3,ymm1
        vbroadcastss ymm3,DWORD PTR [rbx+r10*2+BroadcastOffset]
        vfmadd231ps ymm14,ymm3,ymm0
        vfmadd231ps ymm15,ymm3,ymm1
ENDIF
ENDIF

        ENDM

;
; ComputeBlockFma3By8
;
; Macro Description:
;
;   This macro multiplies and accumulates for a 8xN block (where N is 1,3,6)
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
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockFma3By8 MACRO Count, VectorOffset, BroadcastOffset

IF Count EQ 1
        vbroadcastss ymm3,DWORD PTR [rcx+BroadcastOffset]
        vfmadd231ps ymm5,ymm3,YMMWORD PTR [rdx+VectorOffset]
ELSE
        vmovaps ymm0,YMMWORD PTR [rdx+VectorOffset]
        vbroadcastss ymm3,DWORD PTR [rcx+BroadcastOffset]
        vfmadd231ps ymm5,ymm3,ymm0
IF Count GE 3
        vbroadcastss ymm3,DWORD PTR [rcx+r10+BroadcastOffset]
        vfmadd231ps ymm7,ymm3,ymm0
        vbroadcastss ymm3,DWORD PTR [rcx+r10*2+BroadcastOffset]
        vfmadd231ps ymm9,ymm3,ymm0
ENDIF
IF Count GE 6
        vbroadcastss ymm3,DWORD PTR [rbx+BroadcastOffset]
        vfmadd231ps ymm11,ymm3,ymm0
        vbroadcastss ymm3,DWORD PTR [rbx+r10+BroadcastOffset]
        vfmadd231ps ymm13,ymm3,ymm0
        vbroadcastss ymm3,DWORD PTR [rbx+r10*2+BroadcastOffset]
        vfmadd231ps ymm15,ymm3,ymm0
ENDIF
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

SgemmKernelFma3Function MACRO Mode

        NESTED_ENTRY MlasSgemmKernel&Mode&Fma3, _TEXT

        SgemmKernelAvxEntry

;
; Process 6 rows of the matrices.
;

        cmp     r11,6
        jb      ProcessCountMLessThan6
        mov     r11d,6                      ; return 6 rows handled
        cmp     rbp,8
        jbe     ProcessRemainingCountN6

ProcessNextColumnLoop16x6:
        lea     rbx,[r10*2+r10]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
        ComputeBlockFma3Loop ComputeBlockFma3By16, 6
        lea     rcx,[r8+rax*2]              ; compute matrix C plus 2 rows
        lea     rbx,[r8+rax*4]              ; compute matrix C plus 4 rows
IFDIFI <Mode>, <Add>
        vmulps  ymm4,ymm4,ymm2              ; multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        vmulps  ymm6,ymm6,ymm2
        vmulps  ymm7,ymm7,ymm2
        vmulps  ymm8,ymm8,ymm2
        vmulps  ymm9,ymm9,ymm2
        vmulps  ymm10,ymm10,ymm2
        vmulps  ymm11,ymm11,ymm2
        vmulps  ymm12,ymm12,ymm2
        vmulps  ymm13,ymm13,ymm2
        vmulps  ymm14,ymm14,ymm2
        vmulps  ymm15,ymm15,ymm2
ENDIF
        sub     rbp,16
        jb      OutputMasked16x6Block
IFIDNI <Mode>, <Add>
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [r8]
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [r8+32]
        vfmadd213ps ymm6,ymm2,YMMWORD PTR [r8+rax]
        vfmadd213ps ymm7,ymm2,YMMWORD PTR [r8+rax+32]
        vfmadd213ps ymm8,ymm2,YMMWORD PTR [rcx]
        vfmadd213ps ymm9,ymm2,YMMWORD PTR [rcx+32]
        vfmadd213ps ymm10,ymm2,YMMWORD PTR [rcx+rax]
        vfmadd213ps ymm11,ymm2,YMMWORD PTR [rcx+rax+32]
        vfmadd213ps ymm12,ymm2,YMMWORD PTR [rbx]
        vfmadd213ps ymm13,ymm2,YMMWORD PTR [rbx+32]
        vfmadd213ps ymm14,ymm2,YMMWORD PTR [rbx+rax]
        vfmadd213ps ymm15,ymm2,YMMWORD PTR [rbx+rax+32]
ENDIF
        vmovups YMMWORD PTR [r8],ymm4
        vmovups YMMWORD PTR [r8+32],ymm5
        vmovups YMMWORD PTR [r8+rax],ymm6
        vmovups YMMWORD PTR [r8+rax+32],ymm7
        vmovups YMMWORD PTR [rcx],ymm8
        vmovups YMMWORD PTR [rcx+32],ymm9
        vmovups YMMWORD PTR [rcx+rax],ymm10
        vmovups YMMWORD PTR [rcx+rax+32],ymm11
        vmovups YMMWORD PTR [rbx],ymm12
        vmovups YMMWORD PTR [rbx+32],ymm13
        vmovups YMMWORD PTR [rbx+rax],ymm14
        vmovups YMMWORD PTR [rbx+rax+32],ymm15
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,8
        ja      ProcessNextColumnLoop16x6
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN6:
        lea     rbx,[r10*2+r10]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
        ComputeBlockFma3Loop ComputeBlockFma3By8, 6
        lea     rcx,[r8+rax*2]              ; compute matrix C plus 2 rows
        lea     rbx,[r8+rax*4]              ; compute matrix C plus 4 rows
IFDIFI <Mode>, <Add>
        vmulps  ymm5,ymm5,ymm2              ; multiply by alpha
        vmulps  ymm7,ymm7,ymm2
        vmulps  ymm9,ymm9,ymm2
        vmulps  ymm11,ymm11,ymm2
        vmulps  ymm13,ymm13,ymm2
        vmulps  ymm15,ymm15,ymm2
ENDIF
        cmp     rbp,8
        jb      OutputMasked8x6Block
IFIDNI <Mode>, <Add>
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [r8]
        vfmadd213ps ymm7,ymm2,YMMWORD PTR [r8+rax]
        vfmadd213ps ymm9,ymm2,YMMWORD PTR [rcx]
        vfmadd213ps ymm11,ymm2,YMMWORD PTR [rcx+rax]
        vfmadd213ps ymm13,ymm2,YMMWORD PTR [rbx]
        vfmadd213ps ymm15,ymm2,YMMWORD PTR [rbx+rax]
ENDIF
        vmovups YMMWORD PTR [r8],ymm5
        vmovups YMMWORD PTR [r8+rax],ymm7
        vmovups YMMWORD PTR [rcx],ymm9
        vmovups YMMWORD PTR [rcx+rax],ymm11
        vmovups YMMWORD PTR [rbx],ymm13
        vmovups YMMWORD PTR [rbx+rax],ymm15
        jmp     ExitKernelAndZeroUpper

OutputMasked16x6Block:
IFIDNI <Mode>, <Add>
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [r8]
        vfmadd213ps ymm6,ymm2,YMMWORD PTR [r8+rax]
        vfmadd213ps ymm8,ymm2,YMMWORD PTR [rcx]
        vfmadd213ps ymm10,ymm2,YMMWORD PTR [rcx+rax]
        vfmadd213ps ymm12,ymm2,YMMWORD PTR [rbx]
        vfmadd213ps ymm14,ymm2,YMMWORD PTR [rbx+rax]
ENDIF
        vmovups YMMWORD PTR [r8],ymm4
        vmovups YMMWORD PTR [r8+rax],ymm6
        vmovups YMMWORD PTR [rcx],ymm8
        vmovups YMMWORD PTR [rcx+rax],ymm10
        vmovups YMMWORD PTR [rbx],ymm12
        vmovups YMMWORD PTR [rbx+rax],ymm14
        add     r8,8*4                      ; advance matrix C by 8 columns
        add     rcx,8*4                     ; advance matrix C plus 2 rows by 8 columns
        add     rbx,8*4                     ; advance matrix C plus 4 rows by 8 columns
        add     rbp,8                       ; correct for over-subtract above

OutputMasked8x6Block:
        mov     DWORD PTR SgemmKernelFrame.CountN[rsp],ebp
        vbroadcastss ymm0,DWORD PTR SgemmKernelFrame.CountN[rsp]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR [MlasMaskMoveAvx]
IFIDNI <Mode>, <Add>
        vmaskmovps ymm4,ymm0,YMMWORD PTR [r8]
        vmaskmovps ymm6,ymm0,YMMWORD PTR [r8+rax]
        vmaskmovps ymm8,ymm0,YMMWORD PTR [rcx]
        vmaskmovps ymm10,ymm0,YMMWORD PTR [rcx+rax]
        vmaskmovps ymm12,ymm0,YMMWORD PTR [rbx]
        vmaskmovps ymm14,ymm0,YMMWORD PTR [rbx+rax]
        vfmadd213ps ymm5,ymm2,ymm4
        vfmadd213ps ymm7,ymm2,ymm6
        vfmadd213ps ymm9,ymm2,ymm8
        vfmadd213ps ymm11,ymm2,ymm10
        vfmadd213ps ymm13,ymm2,ymm12
        vfmadd213ps ymm15,ymm2,ymm14
ENDIF
        vmaskmovps YMMWORD PTR [r8],ymm0,ymm5
        vmaskmovps YMMWORD PTR [r8+rax],ymm0,ymm7
        vmaskmovps YMMWORD PTR [rcx],ymm0,ymm9
        vmaskmovps YMMWORD PTR [rcx+rax],ymm0,ymm11
        vmaskmovps YMMWORD PTR [rbx],ymm0,ymm13
        vmaskmovps YMMWORD PTR [rbx+rax],ymm0,ymm15

;
; Restore non-volatile registers and return.
;

ExitKernelAndZeroUpper:
        vzeroupper

ExitKernel:
        SgemmKernelAvxExit

;
; Process 3 rows of the matrices.
;

ProcessCountMLessThan6:
        cmp     r11,3
        jb      ProcessCountMLessThan3
        mov     r11d,3                      ; return 3 rows handled
        cmp     rbp,8
        jbe     ProcessRemainingCountN3

ProcessNextColumnLoop16x3:
        ComputeBlockFma3Loop ComputeBlockFma3By16, 3
IFDIFI <Mode>, <Add>
        vmulps  ymm4,ymm4,ymm2              ; multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        vmulps  ymm6,ymm6,ymm2
        vmulps  ymm7,ymm7,ymm2
        vmulps  ymm8,ymm8,ymm2
        vmulps  ymm9,ymm9,ymm2
ENDIF
        sub     rbp,16
        jb      OutputMasked16x3Block
IFIDNI <Mode>, <Add>
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [r8]
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [r8+32]
        vfmadd213ps ymm6,ymm2,YMMWORD PTR [r8+rax]
        vfmadd213ps ymm7,ymm2,YMMWORD PTR [r8+rax+32]
        vfmadd213ps ymm8,ymm2,YMMWORD PTR [r8+rax*2]
        vfmadd213ps ymm9,ymm2,YMMWORD PTR [r8+rax*2+32]
ENDIF
        vmovups YMMWORD PTR [r8],ymm4
        vmovups YMMWORD PTR [r8+32],ymm5
        vmovups YMMWORD PTR [r8+rax],ymm6
        vmovups YMMWORD PTR [r8+rax+32],ymm7
        vmovups YMMWORD PTR [r8+rax*2],ymm8
        vmovups YMMWORD PTR [r8+rax*2+32],ymm9
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,8
        ja      ProcessNextColumnLoop16x3
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN3:
        ComputeBlockFma3Loop ComputeBlockFma3By8, 3
IFDIFI <Mode>, <Add>
        vmulps  ymm5,ymm5,ymm2              ; multiply by alpha
        vmulps  ymm7,ymm7,ymm2
        vmulps  ymm9,ymm9,ymm2
ENDIF
        cmp     rbp,8
        jb      OutputMasked8x3Block
IFIDNI <Mode>, <Add>
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [r8]
        vfmadd213ps ymm7,ymm2,YMMWORD PTR [r8+rax]
        vfmadd213ps ymm9,ymm2,YMMWORD PTR [r8+rax*2]
ENDIF
        vmovups YMMWORD PTR [r8],ymm5
        vmovups YMMWORD PTR [r8+rax],ymm7
        vmovups YMMWORD PTR [r8+rax*2],ymm9
        jmp     ExitKernelAndZeroUpper

OutputMasked16x3Block:
IFIDNI <Mode>, <Add>
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [r8]
        vfmadd213ps ymm6,ymm2,YMMWORD PTR [r8+rax]
        vfmadd213ps ymm8,ymm2,YMMWORD PTR [r8+rax*2]
ENDIF
        vmovups YMMWORD PTR [r8],ymm4
        vmovups YMMWORD PTR [r8+rax],ymm6
        vmovups YMMWORD PTR [r8+rax*2],ymm8
        add     r8,8*4                      ; advance matrix C by 8 columns
        add     rbp,8                       ; correct for over-subtract above

OutputMasked8x3Block:
        mov     DWORD PTR SgemmKernelFrame.CountN[rsp],ebp
        vbroadcastss ymm0,DWORD PTR SgemmKernelFrame.CountN[rsp]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR [MlasMaskMoveAvx]
IFIDNI <Mode>, <Add>
        vmaskmovps ymm4,ymm0,YMMWORD PTR [r8]
        vmaskmovps ymm6,ymm0,YMMWORD PTR [r8+rax]
        vmaskmovps ymm8,ymm0,YMMWORD PTR [r8+rax*2]
        vfmadd213ps ymm5,ymm2,ymm4
        vfmadd213ps ymm7,ymm2,ymm6
        vfmadd213ps ymm9,ymm2,ymm8
ENDIF
        vmaskmovps YMMWORD PTR [r8],ymm0,ymm5
        vmaskmovps YMMWORD PTR [r8+rax],ymm0,ymm7
        vmaskmovps YMMWORD PTR [r8+rax*2],ymm0,ymm9
        jmp     ExitKernelAndZeroUpper

;
; Process 1 row of the matrices.
;

ProcessCountMLessThan3:
        mov     r11d,1                      ; return 1 row handled
        cmp     rbp,32
        jb      ProcessRemainingCountN1LessThan32
        mov     rbx,r9
        shl     rbx,6                       ; compute 16*CountK*sizeof(float)

ProcessNextColumnLoop32x1:
        ComputeBlockFma3Loop ComputeBlockFma3By32, 1
        add     rdx,rbx                     ; advance matrix B by 16*CountK floats
IFDIFI <Mode>, <Add>
        vmulps  ymm4,ymm4,ymm2              ; multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        vmulps  ymm6,ymm6,ymm2
        vmulps  ymm7,ymm7,ymm2
ELSE
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [r8]
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [r8+32]
        vfmadd213ps ymm6,ymm2,YMMWORD PTR [r8+64]
        vfmadd213ps ymm7,ymm2,YMMWORD PTR [r8+96]
ENDIF
        sub     rbp,32
        vmovups YMMWORD PTR [r8],ymm4
        vmovups YMMWORD PTR [r8+32],ymm5
        vmovups YMMWORD PTR [r8+64],ymm6
        vmovups YMMWORD PTR [r8+96],ymm7
        add     r8,32*4                     ; advance matrix C by 32 columns
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,32
        jae     ProcessNextColumnLoop32x1
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN1LessThan32:
        cmp     rbp,8
        jbe     ProcessRemainingCountN1

ProcessNextColumnLoop16x1:
        ComputeBlockFma3Loop ComputeBlockFma3By16, 1
IFDIFI <Mode>, <Add>
        vmulps  ymm4,ymm4,ymm2              ; multiply by alpha
        vmulps  ymm5,ymm5,ymm2
ENDIF
        sub     rbp,16
        jb      OutputMasked16x1Block
IFIDNI <Mode>, <Add>
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [r8]
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [r8+32]
ENDIF
        vmovups YMMWORD PTR [r8],ymm4
        vmovups YMMWORD PTR [r8+32],ymm5
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rsi                     ; reload matrix A
        vzeroall
        cmp     rbp,8
        ja      ProcessNextColumnLoop16x1
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN1:
        ComputeBlockFma3Loop ComputeBlockFma3By8, 1
IFDIFI <Mode>, <Add>
        vmulps  ymm5,ymm5,ymm2              ; multiply by alpha
ENDIF
        cmp     rbp,8
        jb      OutputMasked8x1Block
IFIDNI <Mode>, <Add>
        vfmadd213ps ymm5,ymm2,YMMWORD PTR [r8]
ENDIF
        vmovups YMMWORD PTR [r8],ymm5
        jmp     ExitKernelAndZeroUpper

OutputMasked16x1Block:
IFIDNI <Mode>, <Add>
        vfmadd213ps ymm4,ymm2,YMMWORD PTR [r8]
ENDIF
        vmovups YMMWORD PTR [r8],ymm4
        add     r8,8*4                      ; advance matrix C by 8 columns
        add     rbp,8                       ; correct for over-subtract above

OutputMasked8x1Block:
        mov     DWORD PTR SgemmKernelFrame.CountN[rsp],ebp
        vbroadcastss ymm0,DWORD PTR SgemmKernelFrame.CountN[rsp]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR [MlasMaskMoveAvx]
IFIDNI <Mode>, <Add>
        vmaskmovps ymm4,ymm0,YMMWORD PTR [r8]
        vfmadd213ps ymm5,ymm2,ymm4
ENDIF
        vmaskmovps YMMWORD PTR [r8],ymm0,ymm5
        jmp     ExitKernelAndZeroUpper

        NESTED_END MlasSgemmKernel&Mode&Fma3, _TEXT

        ENDM

SgemmKernelFma3Function Zero
SgemmKernelFma3Function Add

        END
