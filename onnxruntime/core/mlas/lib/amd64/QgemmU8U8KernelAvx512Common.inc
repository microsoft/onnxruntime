;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8U8KernelAvx512Common.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the quantized
;   integer matrix/matrix multiply operation (QGEMM) for the AVX512BW and
;   AVX512VNNI kernels.
;
;--

;
; Stack frame layout for the U8U8 kernel.
;

GemmU8U8KernelFrame STRUCT

        SavedR14 QWORD ?
        SavedR13 QWORD ?
        SavedR12 QWORD ?
        SavedRdi QWORD ?
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        CountM QWORD ?
        CountN QWORD ?
        ldc QWORD ?
        RowSumVector QWORD ?
        ColumnSumVector QWORD ?
        DepthValue QWORD ?
        ZeroMode QWORD ?

GemmU8U8KernelFrame ENDS

;
; Macro Description:
;
;   This macro generates code to produce an output block for a set of columns
;   and rows.
;
; Arguments:
;
;   ColumnCount - Supplies the number of columns to produce.
;
;   RowCount - Supplies the number of rows to produce.
;
; Implicit Arguments:
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r9 - Supplies the number of paired columns from matrix A and the number of
;       paired rows from matrix B to iterate over.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   r12 - Supplies the address of the row sum vector.
;
;   r13 - Supplies the address of the column sum vector.
;

ProduceOutputBlock MACRO ColumnCount, RowCount

        LOCAL   ComputeBlockLoop

;
; Initialize the accumulators with the sum of the global depth value constant,
; the column sums, and the row sums.
;

        vpbroadcastd zmm31,DWORD PTR GemmU8U8KernelFrame.DepthValue[rsp]
IF ColumnCount EQ 32
        vpaddd  zmm30,zmm31,ZMMWORD PTR [r13]
        vpaddd  zmm31,zmm31,ZMMWORD PTR [r13+64]
        add     r13,32*4                    ; advance ColumnSumVector by 32 columns
ELSE
        vpaddd zmm31,zmm31,ZMMWORD PTR [r13]
ENDIF
        EmitIfCount2GE RowCount, 1, ColumnCount, 32, <vpaddd zmm16,zmm30,DWORD BCST [r12]>
        EmitIfCountGE RowCount, 1, <vpaddd zmm17,zmm31,DWORD BCST [r12]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 32, <vpaddd zmm18,zmm30,DWORD BCST [r12+4]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm19,zmm31,DWORD BCST [r12+4]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 32, <vpaddd zmm20,zmm30,DWORD BCST [r12+8]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm21,zmm31,DWORD BCST [r12+8]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 32, <vpaddd zmm22,zmm30,DWORD BCST [r12+12]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm23,zmm31,DWORD BCST [r12+12]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 32, <vpaddd zmm24,zmm30,DWORD BCST [r12+16]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm25,zmm31,DWORD BCST [r12+16]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 32, <vpaddd zmm26,zmm30,DWORD BCST [r12+20]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm27,zmm31,DWORD BCST [r12+20]>

;
; Iterate over PairedCountK elements from matrix A and matrix B.
;

        mov     rsi,r9                      ; reload PairedCountK
IF RowCount GT 3
        lea     rbx,[r10*2+r10]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF

ComputeBlockLoop:
        ComputeBlock ColumnCount, RowCount
        add     rcx,4                       ; advance matrix A by 1 pair
IF RowCount GT 3
        add     rbx,4                       ; advance matrix A plus 3 rows by 1 pair
ENDIF
        add     rdx,32
        dec     rsi                         ; decrement pairs remaining
        jnz     ComputeBlockLoop

IF RowCount GT 3
        lea     rbx,[r8+rax*2]              ; compute matrix C plus 3 rows
        add     rbx,rax
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
;   r8 - Supplies the address of matrix C.
;
;   rdi - Supplies the address of matrix A.
;
;   rbp - Supplies the number of columns from matrix B and matrix C to iterate
;       over.
;
;   r9 - Supplies the number of paired columns from matrix A and the number of
;       paired rows from matrix B to iterate over.
;
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   r12 - Supplies the address of the row sum vector.
;
;   r13 - Supplies the address of the column sum vector.
;
;   r14b - Supplies the zero mode flag.
;

ProcessCountM MACRO RowCount

        LOCAL   ProcessNextColumnLoop32xN
        LOCAL   SkipAccumulateOutput32xNBlock
        LOCAL   Output16xNBlock
        LOCAL   Output16xNBlockWithMask
        LOCAL   SkipAccumulateOutput16xNBlockWithMask
        LOCAL   ProcessRemainingCountN

        cmp     rbp,16
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop32xN:
        ProduceOutputBlock 32, RowCount
        lea     rdx,[rdx+r10*8]             ; advance matrix B by 8*PairedCountK
        test    r14b,r14b                   ; ZeroMode?
        jnz     SkipAccumulateOutput32xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd zmm16,zmm16,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm18,zmm18,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm20,zmm20,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm22,zmm22,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm24,zmm24,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm26,zmm26,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput32xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8],zmm16>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax],zmm18>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2],zmm20>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx],zmm22>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax],zmm24>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2],zmm26>
        add     r8,16*4                     ; advance matrix C by 16 columns
IF RowCount GT 3
        add     rbx,16*4                    ; advance matrix C plus 3 rows by 16 columns
ENDIF
        sub     rbp,16

Output16xNBlock:
        sub     rbp,16
        jae     Output16xNBlockWithMask
        lea     ecx,[ebp+16]                ; correct for over-subtract above
        mov     esi,1
        shl     esi,cl
        dec     esi
        kmovw   k1,esi                      ; update mask for remaining columns
        xor     ebp,ebp                     ; no more columns remaining

Output16xNBlockWithMask:
        test    r14b,r14b                   ; ZeroMode?
        jnz     SkipAccumulateOutput16xNBlockWithMask
        EmitIfCountGE RowCount, 1, <vpaddd zmm17{k1},zmm17,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm19{k1},zmm19,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm21{k1},zmm21,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm23{k1},zmm23,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm25{k1},zmm25,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm27{k1},zmm27,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput16xNBlockWithMask:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8]{k1},zmm17>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax]{k1},zmm19>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2]{k1},zmm21>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx]{k1},zmm23>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax]{k1},zmm25>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2]{k1},zmm27>
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rdi                     ; reload matrix A
        cmp     rbp,16
        ja      ProcessNextColumnLoop32xN
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN:
        ProduceOutputBlock 16, RowCount
        jmp     Output16xNBlock

        ENDM

;
; Macro Description:
;
;   This macro generates the common AVX512 code for the inner kernel to compute
;   matrix multiplication.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string for function tags.
;

GemmU8U8KernelAvx512Function MACRO Isa

;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows.
;
; Arguments:
;
;   A (rcx) - Supplies the address of matrix A. The matrix data has been packed
;       using MlasGemmU8U8CopyPackAAvx2.
;
;   B (rdx) - Supplies the address of matrix B. The matrix data has been packed
;       using MlasGemmU8U8CopyPackBAvx2.
;
;   C (r8) - Supplies the address of matrix C.
;
;   PairedCountK (r9) - Supplies the number of paired columns from matrix A and
;       the number of paired rows from matrix B to iterate over.
;
;   CountM - Supplies the maximum number of rows that can be processed for
;       matrix A and matrix C. The actual number of rows handled for this
;       invocation depends on the kernel implementation.
;
;   CountN - Supplies the number of columns from matrix B and matrix C to iterate
;       over.
;
;   ldc - Supplies the first dimension of matrix C.
;
;   RowSumVector - Supplies the sum of each row from matrix A multiplied by the
;       zero point offset of matrix B. These values are accumulated into every
;       row of matrix C.
;
;   ColumnSumVector - Supplies the sum of each column from matrix B multiplied
;       by the zero point offset of matrix A. These values are accumulated into
;       every column of matrix C.
;
;   DepthValue - Supplies the value CountK multiplied by the zero point offset
;       of matrixA multplied by the zero point offset of matrix B. This value is
;       accumulated into every element of matrix C.
;
;   ZeroMode - Supplies true if the output matrix must be zero initialized,
;       else false if the output matrix is accumulated into.
;
; Return Value:
;
;   Returns the number of rows handled.
;
;--

        NESTED_ENTRY MlasGemmU8U8Kernel&Isa&, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        push_reg r14

        END_PROLOGUE

        mov     rdi,rcx
        mov     rbp,GemmU8U8KernelFrame.CountN[rsp]
        mov     rax,GemmU8U8KernelFrame.ldc[rsp]
        shl     rax,2                       ; convert ldc to bytes
        lea     r10,[r9*4]
        mov     r11,GemmU8U8KernelFrame.CountM[rsp]
        mov     r12,GemmU8U8KernelFrame.RowSumVector[rsp]
        mov     r13,GemmU8U8KernelFrame.ColumnSumVector[rsp]
        movzx   r14,BYTE PTR GemmU8U8KernelFrame.ZeroMode[rsp]
        mov     esi,-1
        kmovw   k1,esi                      ; update mask to write all columns

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
        ProcessCountM 6

;
; Restore non-volatile registers and return.
;

ExitKernel:
        mov     eax,r11d

        BEGIN_EPILOGUE

        pop     r14
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

ProcessCountM1:
        ProcessCountM 1

ProcessCountM3:
        ProcessCountM 3

ProcessCountM5:
        ProcessCountM 5

        NESTED_END MlasGemmU8U8Kernel&Isa&, _TEXT

        ENDM
