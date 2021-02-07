;++
;
; Copyright (c) Intel Corporation 2020. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8S8KernelAvxVnni.asm
;
; Abstract:
;
;   This module implements the kernels for the quantized integer matrix/matrix
;   multiply operation (QGEMM).
;
;   This implementation uses AVXVNNI instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE QgemmU8X8KernelAvx2Common.inc
INCLUDE AssembleAvxVnni.inc
        .list

;
; Macro Description:
;
;   This macro generates code to multiply and accumulator a single row of the
;   output block.
;
; Arguments:
;
;   ColumnCount - Supplies the number of columns to produce.
;
;   Vec1Reg - Supplies the high block accumulator register (when ColumnCount
;       is 16).
;
;   Vec2Reg - Supplies the low block accumulator register.
;
; Implicit Arguments:
;
;   ymm0 - Supplies the first vector loaded from matrix B.
;
;   ymm1 - Supplies the second vector loaded from matrix B (when ColumnCount
;       is 16).
;
;   ymm2 - Supplies the broadcast value loaded from matrix A.
;

MultiplyAccumulateRow MACRO ColumnCount, Vec1Reg, Vec2Reg

IF ColumnCount EQ 16
        VpdpbusdsYmmYmmYmm Vec1Reg,ymm2,ymm0
        VpdpbusdsYmmYmmYmm Vec2Reg,ymm2,ymm1
ELSE
        VpdpbusdsYmmYmmYmm Vec2Reg,ymm2,ymm0
ENDIF

        ENDM

;
; Macro Description:
;
;   This macro generates code to multiply and accumulate each row of the output
;   block.
;
; Arguments:
;
;   ColumnCount - Supplies the number of columns to produce.
;
;   RowCount - Supplies the number of rows to produce.
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
;   r9 - Supplies the length in bytes of a row from matrix A.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlock MACRO ColumnCount, RowCount, VectorOffset, BroadcastOffset

        vmovdqu ymm0,YMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE ColumnCount, 16, <vmovdqu ymm1,YMMWORD PTR [rdx+VectorOffset+32]>
        EmitIfCountGE RowCount, 1, <vpbroadcastd ymm2,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 1, <MultiplyAccumulateRow ColumnCount, ymm4, ymm5>
        EmitIfCountGE RowCount, 2, <vpbroadcastd ymm2,DWORD PTR [rcx+r9+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <MultiplyAccumulateRow ColumnCount, ymm6, ymm7>
        EmitIfCountGE RowCount, 3, <vpbroadcastd ymm2,DWORD PTR [rcx+r9*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <MultiplyAccumulateRow ColumnCount, ymm8, ymm9>
        EmitIfCountGE RowCount, 4, <vpbroadcastd ymm2,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <MultiplyAccumulateRow ColumnCount, ymm10, ymm11>
        EmitIfCountGE RowCount, 5, <vpbroadcastd ymm2,DWORD PTR [rbx+r9+BroadcastOffset]>
        EmitIfCountGE RowCount, 5, <MultiplyAccumulateRow ColumnCount, ymm12, ymm13>
        EmitIfCountGE RowCount, 6, <vpbroadcastd ymm2,DWORD PTR [rbx+r9*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 6, <MultiplyAccumulateRow ColumnCount, ymm14, ymm15>

        ENDM

;
; Macro Description:
;
;   This macro generates code to execute the block compute macro multiple
;   times and advancing the matrix A and matrix B data pointers.
;
; Arguments:
;
;   ColumnCount - Supplies the number of columns to produce.
;
;   RowCount - Supplies the number of rows to produce.
;
; Implicit Arguments:
;
;   rbx - Supplies the address into the matrix A data plus 3 rows.
;
;   rcx - Supplies the address into the matrix A data.
;
;   rdx - Supplies the address into the matrix B data.
;
;   r9 - Supplies the length in bytes of a row from matrix A.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlockLoop MACRO ColumnCount, RowCount

        LOCAL   ComputeBlockBy1Loop

        mov     rsi,r9                      ; reload row length remaining

ComputeBlockBy1Loop:
        ComputeBlock ColumnCount, RowCount, 0, 0
        add     rcx,4                       ; advance matrix A by 1 quad
IF RowCount GT 3
        add     rbx,4                       ; advance matrix A plus 3 rows by 1 quad
ENDIF
        add     rdx,64                      ; advance matrix B
        sub     rsi,4
        jnz     ComputeBlockBy1Loop

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
;   A (rcx) - Supplies the address of matrix A. The matrix data has been packed
;       using MlasGemmU8S8CopyPackAAvx2.
;
;   B (rdx) - Supplies the address of matrix B. The matrix data has been packed
;       using MlasGemmU8S8CopyPackBAvx2.
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
;       of matrix A multplied by the zero point offset of matrix B. This value is
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

        NESTED_ENTRY MlasGemmU8S8KernelAvxVnni, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        alloc_stack (GemmU8X8KernelFrame.SavedR13)
        save_xmm128 xmm6,GemmU8X8KernelFrame.SavedXmm6
        save_xmm128 xmm7,GemmU8X8KernelFrame.SavedXmm7
        save_xmm128 xmm8,GemmU8X8KernelFrame.SavedXmm8
        save_xmm128 xmm9,GemmU8X8KernelFrame.SavedXmm9
        save_xmm128 xmm10,GemmU8X8KernelFrame.SavedXmm10
        save_xmm128 xmm11,GemmU8X8KernelFrame.SavedXmm11
        save_xmm128 xmm12,GemmU8X8KernelFrame.SavedXmm12
        save_xmm128 xmm13,GemmU8X8KernelFrame.SavedXmm13
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
        ProcessCountM 6, Fallthrough

;
; Restore non-volatile registers and return.
;

ExitKernel:
        mov     eax,r11d
        vzeroupper
        movaps  xmm6,GemmU8X8KernelFrame.SavedXmm6[rsp]
        movaps  xmm7,GemmU8X8KernelFrame.SavedXmm7[rsp]
        movaps  xmm8,GemmU8X8KernelFrame.SavedXmm8[rsp]
        movaps  xmm9,GemmU8X8KernelFrame.SavedXmm9[rsp]
        movaps  xmm10,GemmU8X8KernelFrame.SavedXmm10[rsp]
        movaps  xmm11,GemmU8X8KernelFrame.SavedXmm11[rsp]
        movaps  xmm12,GemmU8X8KernelFrame.SavedXmm12[rsp]
        movaps  xmm13,GemmU8X8KernelFrame.SavedXmm13[rsp]
        movaps  xmm14,GemmU8X8KernelFrame.SavedXmm14[rsp]
        movaps  xmm15,GemmU8X8KernelFrame.SavedXmm15[rsp]
        add     rsp,(GemmU8X8KernelFrame.SavedR13)

        BEGIN_EPILOGUE

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

        NESTED_END MlasGemmU8S8KernelAvxVnni, _TEXT

        END
