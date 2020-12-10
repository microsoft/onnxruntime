;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8X8KernelAvx512Common.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the quantized
;   integer matrix/matrix multiply operation (QGEMM) for the AVX512 core and
;   AVX512VNNI kernels.
;
;--

;
; Stack frame layout for the U8S8 and U8U8 kernels.
;

GemmU8X8KernelFrame STRUCT

        SavedXmm14 OWORD ?
        SavedXmm15 OWORD ?
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
        RowSumBuffer QWORD ?
        ColumnSumBuffer QWORD ?
        DepthValue QWORD ?
        ZeroMode QWORD ?

GemmU8X8KernelFrame ENDS

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
;   rax - Supplies the length in bytes of a row from matrix C.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r9 - Supplies the length in bytes of a row from matrix A.
;
;   r12 - Supplies the address of the row sum buffer.
;
;   r13 - Supplies the address of the column sum buffer.
;

ProduceOutputBlock MACRO ColumnCount, RowCount

;
; Initialize the accumulators with the sum of the global depth value constant,
; the column sums, and the row sums.
;

        vpbroadcastd zmm3,DWORD PTR GemmU8X8KernelFrame.DepthValue[rsp]
IF ColumnCount GE 32
IF ColumnCount GE 48
        vpaddd  zmm2,zmm3,ZMMWORD PTR [r13]
        vpaddd  zmm1,zmm3,ZMMWORD PTR [r13+64]
        vpaddd  zmm0,zmm3,ZMMWORD PTR [r13+128]
ELSE
        vpaddd  zmm1,zmm3,ZMMWORD PTR [r13]
        vpaddd  zmm0,zmm3,ZMMWORD PTR [r13+64]
ENDIF
        add_immed r13,ColumnCount*4         ; advance ColumnSumBuffer by N columns
ELSE
        vpaddd zmm0,zmm3,ZMMWORD PTR [r13]
ENDIF
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <vpaddd zmm14,zmm0,DWORD BCST [r12]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 32, <vpaddd zmm20,zmm1,DWORD BCST [r12]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 48, <vpaddd zmm26,zmm2,DWORD BCST [r12]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <vpaddd zmm15,zmm0,DWORD BCST [r12+4]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 32, <vpaddd zmm21,zmm1,DWORD BCST [r12+4]>
        EmitIfCount2GE RowCount, 2, ColumnCount, 48, <vpaddd zmm27,zmm2,DWORD BCST [r12+4]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <vpaddd zmm16,zmm0,DWORD BCST [r12+8]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 32, <vpaddd zmm22,zmm1,DWORD BCST [r12+8]>
        EmitIfCount2GE RowCount, 3, ColumnCount, 48, <vpaddd zmm28,zmm2,DWORD BCST [r12+8]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <vpaddd zmm17,zmm0,DWORD BCST [r12+12]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 32, <vpaddd zmm23,zmm1,DWORD BCST [r12+12]>
        EmitIfCount2GE RowCount, 4, ColumnCount, 48, <vpaddd zmm29,zmm2,DWORD BCST [r12+12]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <vpaddd zmm18,zmm0,DWORD BCST [r12+16]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 32, <vpaddd zmm24,zmm1,DWORD BCST [r12+16]>
        EmitIfCount2GE RowCount, 5, ColumnCount, 48, <vpaddd zmm30,zmm2,DWORD BCST [r12+16]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <vpaddd zmm19,zmm0,DWORD BCST [r12+20]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 32, <vpaddd zmm25,zmm1,DWORD BCST [r12+20]>
        EmitIfCount2GE RowCount, 6, ColumnCount, 48, <vpaddd zmm31,zmm2,DWORD BCST [r12+20]>

;
; Iterate over the length of a matrix A row to produce the output accumulators.
;

IF RowCount GT 3
        lea     rbx,[r9*2+r9]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF
        ComputeBlockLoop ColumnCount, RowCount
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
;   r9 - Supplies the length in bytes of a row from matrix A.
;
;   r10b - Supplies the zero mode flag.
;
;   r12 - Supplies the address of the row sum buffer.
;
;   r13 - Supplies the address of the column sum buffer.
;
;   r14 - Supplies the stride in bytes of between packed blocks of matrix B.
;

ProcessCountM MACRO RowCount

        LOCAL   ProcessNextColumnLoop32xN
        LOCAL   Output32xNBlock
        LOCAL   SkipAccumulateOutput32xNBlock
        LOCAL   Output16xNBlock
        LOCAL   Output16xNBlockWithMask
        LOCAL   SkipAccumulateOutput16xNBlockWithMask
        LOCAL   ProcessRemainingCountN
        LOCAL   ProcessNextColumnLoop48xN
        LOCAL   SkipAccumulateOutput48xNBlock

        cmp     rbp,32
        ja      ProcessNextColumnLoop48xN
        cmp     rbp,16
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop32xN:
        ProduceOutputBlock 32, RowCount
        add     rdx,r14                     ; advance matrix B by packed block stride

Output32xNBlock:
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput32xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd zmm20,zmm20,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm21,zmm21,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm22,zmm22,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm23,zmm23,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm24,zmm24,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm25,zmm25,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput32xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8],zmm20>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax],zmm21>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2],zmm22>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx],zmm23>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax],zmm24>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2],zmm25>
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
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput16xNBlockWithMask
        EmitIfCountGE RowCount, 1, <vpaddd zmm14{k1},zmm14,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm15{k1},zmm15,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm16{k1},zmm16,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm17{k1},zmm17,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm18{k1},zmm18,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm19{k1},zmm19,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput16xNBlockWithMask:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8]{k1},zmm14>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax]{k1},zmm15>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2]{k1},zmm16>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx]{k1},zmm17>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax]{k1},zmm18>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2]{k1},zmm19>
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rdi                     ; reload matrix A
        cmp     rbp,32
        ja      ProcessNextColumnLoop48xN
        cmp     rbp,16
        ja      ProcessNextColumnLoop32xN
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN:
        ProduceOutputBlock 16, RowCount
        jmp     Output16xNBlock

ProcessNextColumnLoop48xN:
        ProduceOutputBlock 48, RowCount
        lea     rdx,[rdx+r14*2]             ; advance matrix B by packed block stride
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput48xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd zmm26,zmm26,ZMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd zmm27,zmm27,ZMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd zmm28,zmm28,ZMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd zmm29,zmm29,ZMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd zmm30,zmm30,ZMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd zmm31,zmm31,ZMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput48xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu32 ZMMWORD PTR [r8],zmm26>
        EmitIfCountGE RowCount, 2, <vmovdqu32 ZMMWORD PTR [r8+rax],zmm27>
        EmitIfCountGE RowCount, 3, <vmovdqu32 ZMMWORD PTR [r8+rax*2],zmm28>
        EmitIfCountGE RowCount, 4, <vmovdqu32 ZMMWORD PTR [rbx],zmm29>
        EmitIfCountGE RowCount, 5, <vmovdqu32 ZMMWORD PTR [rbx+rax],zmm30>
        EmitIfCountGE RowCount, 6, <vmovdqu32 ZMMWORD PTR [rbx+rax*2],zmm31>
        add     r8,16*4                     ; advance matrix C by 16 columns
IF RowCount GT 3
        add     rbx,16*4                    ; advance matrix C plus 3 rows by 16 columns
ENDIF
        sub     rbp,16
        jmp     Output32xNBlock

        ENDM

;
; Macro Description:
;
;   This macro generates the common AVX512 code for the inner kernel to compute
;   matrix multiplication.
;
; Arguments:
;
;   Type - Supplies the kernel type string for function tags.
;
;   Isa - Supplies the instruction set architecture string for function tags.
;

GemmU8X8KernelAvx512Function MACRO Type, Isa

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
;       using MlasGemmU8X8CopyPackAAvx2.
;
;   B (rdx) - Supplies the address of matrix B. The matrix data has been packed
;       using MlasGemmU8X8CopyPackBAvx2.
;
;   C (r8) - Supplies the address of matrix C.
;
;   PackedCountK (r9) - Supplies the number of packed columns from matrix A and
;       the number of packed rows from matrix B to iterate over.
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
;   RowSumBuffer - Supplies the sum of each row from matrix A multiplied by the
;       zero point offset of matrix B. These values are accumulated into every
;       row of matrix C.
;
;   ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
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

        NESTED_ENTRY MlasGemm&Type&Kernel&Isa&, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        push_reg r14
        alloc_stack (GemmU8X8KernelFrame.SavedR14)
        save_xmm128 xmm14,GemmU8X8KernelFrame.SavedXmm14
        save_xmm128 xmm15,GemmU8X8KernelFrame.SavedXmm15

        END_PROLOGUE

        mov     rdi,rcx
        mov     rbp,GemmU8X8KernelFrame.CountN[rsp]
        mov     rax,GemmU8X8KernelFrame.ldc[rsp]
        shl     rax,2                       ; convert ldc to bytes
        shl     r9,2                        ; convert to row length
        movzx   r10,BYTE PTR GemmU8X8KernelFrame.ZeroMode[rsp]
        mov     r11,GemmU8X8KernelFrame.CountM[rsp]
        mov     r12,GemmU8X8KernelFrame.RowSumBuffer[rsp]
        mov     r13,GemmU8X8KernelFrame.ColumnSumBuffer[rsp]
        mov     esi,-1
        kmovw   k1,esi                      ; update mask to write all columns
IFIDNI <Type>, <U8S8>
IFIDNI <Isa>, <Avx512Core>
        neg     esi
        vpbroadcastw zmm5,esi               ; generate 512-bit word vector [0x0001]
ENDIF
        mov     r14,r9
        shl     r14,4                       ; compute matrix B packed stride
ELSE
        lea     r14,[r9*8]                  ; compute matrix B packed stride
ENDIF

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
        vzeroupper
        movaps  xmm14,GemmU8X8KernelFrame.SavedXmm14[rsp]
        movaps  xmm15,GemmU8X8KernelFrame.SavedXmm15[rsp]
        add     rsp,(GemmU8X8KernelFrame.SavedR14)

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

        NESTED_END MlasGemm&Type&Kernel&Isa&, _TEXT

        ENDM
