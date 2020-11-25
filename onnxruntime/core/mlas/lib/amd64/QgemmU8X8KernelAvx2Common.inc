;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8X8KernelAvx2Common.inc
;
; Abstract:
;
;   This module contains common kernel macros and structures for the quantized
;   integer matrix/matrix multiply operation (QGEMM) for the AVX2 kernels.
;
;--

        EXTERN  MlasMaskMoveAvx:NEAR

;
; Stack frame layout for the U8S8 and U8U8 kernels.
;

GemmU8X8KernelFrame STRUCT

        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
        SavedXmm11 OWORD ?
        SavedXmm12 OWORD ?
        SavedXmm13 OWORD ?
        SavedXmm14 OWORD ?
        SavedXmm15 OWORD ?
        Padding QWORD ?
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
;   ymm4-ymm15 - Supplies the block accumulators.
;

ProduceOutputBlock MACRO ColumnCount, RowCount

;
; Initialize the accumulators with the sum of the global depth value constant,
; the column sums, and the row sums.
;

        vpbroadcastd ymm1,DWORD PTR GemmU8X8KernelFrame.DepthValue[rsp]
IF ColumnCount EQ 16
        vpaddd  ymm0,ymm1,YMMWORD PTR [r13]
        vpaddd  ymm1,ymm1,YMMWORD PTR [r13+32]
        add     r13,16*4                    ; advance ColumnSumBuffer by 16 columns
ELSE
        vpaddd  ymm1,ymm1,YMMWORD PTR [r13]
ENDIF
        EmitIfCountGE RowCount, 1, <vpbroadcastd ymm5,DWORD PTR [r12]>
        EmitIfCountGE RowCount, 2, <vpbroadcastd ymm7,DWORD PTR [r12+4]>
        EmitIfCountGE RowCount, 3, <vpbroadcastd ymm9,DWORD PTR [r12+8]>
        EmitIfCountGE RowCount, 4, <vpbroadcastd ymm11,DWORD PTR [r12+12]>
        EmitIfCountGE RowCount, 5, <vpbroadcastd ymm13,DWORD PTR [r12+16]>
        EmitIfCountGE RowCount, 6, <vpbroadcastd ymm15,DWORD PTR [r12+20]>
        EmitIfCount2GE RowCount, 1, ColumnCount, 16, <vpaddd ymm4,ymm5,ymm0>
        EmitIfCountGE RowCount, 1, <vpaddd ymm5,ymm5,ymm1>
        EmitIfCount2GE RowCount, 2, ColumnCount, 16, <vpaddd ymm6,ymm7,ymm0>
        EmitIfCountGE RowCount, 2, <vpaddd ymm7,ymm7,ymm1>
        EmitIfCount2GE RowCount, 3, ColumnCount, 16, <vpaddd ymm8,ymm9,ymm0>
        EmitIfCountGE RowCount, 3, <vpaddd ymm9,ymm9,ymm1>
        EmitIfCount2GE RowCount, 4, ColumnCount, 16, <vpaddd ymm10,ymm11,ymm0>
        EmitIfCountGE RowCount, 4, <vpaddd ymm11,ymm11,ymm1>
        EmitIfCount2GE RowCount, 5, ColumnCount, 16, <vpaddd ymm12,ymm13,ymm0>
        EmitIfCountGE RowCount, 5, <vpaddd ymm13,ymm13,ymm1>
        EmitIfCount2GE RowCount, 6, ColumnCount, 16, <vpaddd ymm14,ymm15,ymm0>
        EmitIfCountGE RowCount, 6, <vpaddd ymm15,ymm15,ymm1>

;
; Iterate over the length of a matrix A row to produce the output accumulators.
;

IF RowCount GT 3
        lea     rbx,[r9*2+r9]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF
        ComputeBlockLoop ColumnCount, RowCount
IF RowCount GT 3
        lea     rbx,[rax*2+rax]
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
;   Fallthrough - Supplies a non-blank value if the macro may fall through to
;       the ExitKernel label.
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

ProcessCountM MACRO RowCount, Fallthrough

        LOCAL   ProcessNextColumnLoop16xN
        LOCAL   SkipAccumulateOutput16xNBlock
        LOCAL   OutputMasked16xNBlock
        LOCAL   ProcessRemainingCountN
        LOCAL   SkipAccumulateOutput8xNBlock
        LOCAL   SkipAccumulateOutputMasked16xNBlock
        LOCAL   OutputMasked8xNBlock
        LOCAL   SkipAccumulateOutputMasked8xNBlock

        cmp     rbp,8
        jbe     ProcessRemainingCountN

ProcessNextColumnLoop16xN:
        ProduceOutputBlock 16, RowCount
        sub     rbp,16
        jb      OutputMasked16xNBlock
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput16xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd ymm4,ymm4,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 1, <vpaddd ymm5,ymm5,YMMWORD PTR [r8+32]>
        EmitIfCountGE RowCount, 2, <vpaddd ymm6,ymm6,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 2, <vpaddd ymm7,ymm7,YMMWORD PTR [r8+rax+32]>
        EmitIfCountGE RowCount, 3, <vpaddd ymm8,ymm8,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 3, <vpaddd ymm9,ymm9,YMMWORD PTR [r8+rax*2+32]>
        EmitIfCountGE RowCount, 4, <vpaddd ymm10,ymm10,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 4, <vpaddd ymm11,ymm11,YMMWORD PTR [rbx+32]>
        EmitIfCountGE RowCount, 5, <vpaddd ymm12,ymm12,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 5, <vpaddd ymm13,ymm13,YMMWORD PTR [rbx+rax+32]>
        EmitIfCountGE RowCount, 6, <vpaddd ymm14,ymm14,YMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE RowCount, 6, <vpaddd ymm15,ymm15,YMMWORD PTR [rbx+rax*2+32]>

SkipAccumulateOutput16xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu YMMWORD PTR [r8],ymm4>
        EmitIfCountGE RowCount, 1, <vmovdqu YMMWORD PTR [r8+32],ymm5>
        EmitIfCountGE RowCount, 2, <vmovdqu YMMWORD PTR [r8+rax],ymm6>
        EmitIfCountGE RowCount, 2, <vmovdqu YMMWORD PTR [r8+rax+32],ymm7>
        EmitIfCountGE RowCount, 3, <vmovdqu YMMWORD PTR [r8+rax*2],ymm8>
        EmitIfCountGE RowCount, 3, <vmovdqu YMMWORD PTR [r8+rax*2+32],ymm9>
        EmitIfCountGE RowCount, 4, <vmovdqu YMMWORD PTR [rbx],ymm10>
        EmitIfCountGE RowCount, 4, <vmovdqu YMMWORD PTR [rbx+32],ymm11>
        EmitIfCountGE RowCount, 5, <vmovdqu YMMWORD PTR [rbx+rax],ymm12>
        EmitIfCountGE RowCount, 5, <vmovdqu YMMWORD PTR [rbx+rax+32],ymm13>
        EmitIfCountGE RowCount, 6, <vmovdqu YMMWORD PTR [rbx+rax*2],ymm14>
        EmitIfCountGE RowCount, 6, <vmovdqu YMMWORD PTR [rbx+rax*2+32],ymm15>
        add     r8,16*4                     ; advance matrix C by 16 columns
        mov     rcx,rdi                     ; reload matrix A
        cmp     rbp,8
        ja      ProcessNextColumnLoop16xN
        test    rbp,rbp
        jz      ExitKernel

ProcessRemainingCountN:
        ProduceOutputBlock 8, RowCount
        cmp     rbp,8
        jb      OutputMasked8xNBlock
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutput8xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd ymm5,ymm5,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd ymm7,ymm7,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd ymm9,ymm9,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd ymm11,ymm11,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd ymm13,ymm13,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd ymm15,ymm15,YMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutput8xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu YMMWORD PTR [r8],ymm5>
        EmitIfCountGE RowCount, 2, <vmovdqu YMMWORD PTR [r8+rax],ymm7>
        EmitIfCountGE RowCount, 3, <vmovdqu YMMWORD PTR [r8+rax*2],ymm9>
        EmitIfCountGE RowCount, 4, <vmovdqu YMMWORD PTR [rbx],ymm11>
        EmitIfCountGE RowCount, 5, <vmovdqu YMMWORD PTR [rbx+rax],ymm13>
        EmitIfCountGE RowCount, 6, <vmovdqu YMMWORD PTR [rbx+rax*2],ymm15>
        jmp     ExitKernel

OutputMasked16xNBlock:
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutputMasked16xNBlock
        EmitIfCountGE RowCount, 1, <vpaddd ymm4,ymm4,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpaddd ymm6,ymm6,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpaddd ymm8,ymm8,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpaddd ymm10,ymm10,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpaddd ymm12,ymm12,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpaddd ymm14,ymm14,YMMWORD PTR [rbx+rax*2]>

SkipAccumulateOutputMasked16xNBlock:
        EmitIfCountGE RowCount, 1, <vmovdqu YMMWORD PTR [r8],ymm4>
        EmitIfCountGE RowCount, 2, <vmovdqu YMMWORD PTR [r8+rax],ymm6>
        EmitIfCountGE RowCount, 3, <vmovdqu YMMWORD PTR [r8+rax*2],ymm8>
        EmitIfCountGE RowCount, 4, <vmovdqu YMMWORD PTR [rbx],ymm10>
        EmitIfCountGE RowCount, 5, <vmovdqu YMMWORD PTR [rbx+rax],ymm12>
        EmitIfCountGE RowCount, 6, <vmovdqu YMMWORD PTR [rbx+rax*2],ymm14>
        add     r8,8*4                      ; advance matrix C by 8 columns
IF RowCount GT 3
        add     rbx,8*4                     ; advance matrix C plus 3 rows by 8 columns
ENDIF
        add     rbp,8                       ; correct for over-subtract above

OutputMasked8xNBlock:
        mov     DWORD PTR GemmU8X8KernelFrame.CountN[rsp],ebp
        vpbroadcastd ymm0,DWORD PTR GemmU8X8KernelFrame.CountN[rsp]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR [MlasMaskMoveAvx]
        test    r10b,r10b                   ; ZeroMode?
        jnz     SkipAccumulateOutputMasked8xNBlock
        EmitIfCountGE RowCount, 1, <vpmaskmovd ymm4,ymm0,YMMWORD PTR [r8]>
        EmitIfCountGE RowCount, 2, <vpmaskmovd ymm6,ymm0,YMMWORD PTR [r8+rax]>
        EmitIfCountGE RowCount, 3, <vpmaskmovd ymm8,ymm0,YMMWORD PTR [r8+rax*2]>
        EmitIfCountGE RowCount, 4, <vpmaskmovd ymm10,ymm0,YMMWORD PTR [rbx]>
        EmitIfCountGE RowCount, 5, <vpmaskmovd ymm12,ymm0,YMMWORD PTR [rbx+rax]>
        EmitIfCountGE RowCount, 6, <vpmaskmovd ymm14,ymm0,YMMWORD PTR [rbx+rax*2]>
        EmitIfCountGE RowCount, 1, <vpaddd ymm5,ymm5,ymm4>
        EmitIfCountGE RowCount, 2, <vpaddd ymm7,ymm7,ymm6>
        EmitIfCountGE RowCount, 3, <vpaddd ymm9,ymm9,ymm8>
        EmitIfCountGE RowCount, 4, <vpaddd ymm11,ymm11,ymm10>
        EmitIfCountGE RowCount, 5, <vpaddd ymm13,ymm13,ymm12>
        EmitIfCountGE RowCount, 6, <vpaddd ymm15,ymm15,ymm14>

SkipAccumulateOutputMasked8xNBlock:
        EmitIfCountGE RowCount, 1, <vpmaskmovd YMMWORD PTR [r8],ymm0,ymm5>
        EmitIfCountGE RowCount, 2, <vpmaskmovd YMMWORD PTR [r8+rax],ymm0,ymm7>
        EmitIfCountGE RowCount, 3, <vpmaskmovd YMMWORD PTR [r8+rax*2],ymm0,ymm9>
        EmitIfCountGE RowCount, 4, <vpmaskmovd YMMWORD PTR [rbx],ymm0,ymm11>
        EmitIfCountGE RowCount, 5, <vpmaskmovd YMMWORD PTR [rbx+rax],ymm0,ymm13>
        EmitIfCountGE RowCount, 6, <vpmaskmovd YMMWORD PTR [rbx+rax*2],ymm0,ymm15>
IFB <Fallthrough>
        jmp     ExitKernel
ENDIF

        ENDM
