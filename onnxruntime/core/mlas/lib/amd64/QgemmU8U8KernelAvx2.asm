;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8U8KernelAvx2.asm
;
; Abstract:
;
;   This module implements the kernels for the quantized integer matrix/matrix
;   multiply operation (QGEMM).
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        EXTERN  MlasMaskMoveAvx:NEAR

;
; Stack frame layout for the U8U8 CopyPackA routine.
;

GemmU8U8CopyPackAFrame STRUCT

        PaddedMatrixAData OWORD 4 DUP (?)
        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
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
        CountK QWORD ?
        RowSumVector QWORD ?
        offb QWORD ?

GemmU8U8CopyPackAFrame ENDS

;
; Stack frame layout for the U8U8 CopyPackB routine.
;

GemmU8U8CopyPackBFrame STRUCT

        PaddedMatrixBData OWORD 2 DUP (?)
        SavedRsi QWORD ?
        SavedRbx QWORD ?
        SavedRbp QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        CountK QWORD ?
        ColumnSumVector QWORD ?
        offa QWORD ?

GemmU8U8CopyPackBFrame ENDS

;
; Stack frame layout for the U8U8 kernel.
;

GemmU8U8KernelFrame STRUCT

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

;++
;
; Routine Description:
;
;   This routine copies elements from the source matrix to the destination
;   packed buffer.
;
;   The kernel expects that elements from matrix A have been zero extended to
;   16-bits and padded to a multiple of 32-bits (two pairs of 16-bit values).
;   The kernel can then efficiently broadcast 32-bits from the packed buffer
;   and avoid expensive shuffling inside the kernel.
;
; Arguments:
;
;   D (rcx) - Supplies the address of the destination packed buffer.
;
;   A (rdx) - Supplies the address of the source matrix.
;
;   lda (r8) - Supplies the number of elements per row of the source matrix.
;
;   CountM (r9) - Supplies the number of rows of the source matrix to copy.
;
;   CountK - Supplies the number of columns of the source matrix to copy.
;
;   RowSumVector - Supplies the address of the buffer to receive the sums of
;       the elements from each of the rows. Each sum has also been multiplied
;       by the zero point offset.
;
;   offb - Supplies the zero point offset for the other source matrix of the
;       matrix multiplication.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasGemmU8U8CopyPackAAvx2, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        alloc_stack (GemmU8U8CopyPackAFrame.SavedR13)
        save_xmm128_avx xmm6,GemmU8U8CopyPackAFrame.SavedXmm6
        save_xmm128_avx xmm7,GemmU8U8CopyPackAFrame.SavedXmm7
        save_xmm128_avx xmm8,GemmU8U8CopyPackAFrame.SavedXmm8
        save_xmm128_avx xmm9,GemmU8U8CopyPackAFrame.SavedXmm9

        END_PROLOGUE

        mov     rdi,rcx
        mov     rsi,rdx
        mov     r10,GemmU8U8CopyPackAFrame.CountK[rsp]
        lea     r11,[r10+1]
        and     r11,NOT 1                   ; align CountK up to pair count
        mov     r12,GemmU8U8CopyPackAFrame.RowSumVector[rsp]
        vpbroadcastw xmm8,WORD PTR GemmU8U8CopyPackAFrame.offb[rsp]

;
; Compute the conditional load/store mask for an unaligned CountK.
;

        mov     eax,r10d
        and     eax,15                      ; isolate unaligned count
        inc     eax
        shr     eax,1                       ; align unaligned count to pair count
        mov     DWORD PTR GemmU8U8CopyPackAFrame.CountK[rsp],eax
        vpbroadcastd ymm9,DWORD PTR GemmU8U8CopyPackAFrame.CountK[rsp]
        vpcmpgtd ymm9,ymm9,YMMWORD PTR [MlasMaskMoveAvx]

;
; Zero initialize the padded stack buffers.
;

        vpxor   xmm0,xmm0,xmm0
        vmovdqu YMMWORD PTR GemmU8U8CopyPackAFrame.PaddedMatrixAData[rsp],ymm0
        vmovdqu YMMWORD PTR GemmU8U8CopyPackAFrame.PaddedMatrixAData[rsp+32],ymm0

;
; Process 4 rows of matrix A in a loop.
;
; For each row, zero extend the source bytes to 16-bits and write to the packed
; buffer. The packed buffer has the same data ordering as the source bytes, but
; the stride is CountK aligned up to an even number of 16-bit values.
;
; These 16-bit values are also accumulated into an intermediate per-row
; accumulator. CountK cannot be greater than 256 to avoid overflowing these
; 16-bit accumulators.
;

        sub     r9,4
        jb      ProcessRemainingRows

ProcessNextRowM4:
        vpxor   xmm0,xmm0,xmm0              ; clear row accumulators
        vpxor   xmm1,xmm1,xmm1
        vpxor   xmm2,xmm2,xmm2
        vpxor   xmm3,xmm3,xmm3
        mov     rdx,rsi
        mov     rcx,rdi
        lea     rsi,[rsi+r8*4]              ; advance next matrix A by 4 rows
        lea     rdi,[rdi+r11*(2*4)]         ; advance next matrix D by 4 rows
        mov     rbx,r10                     ; reload columns remaining
        sub     rbx,16
        jb      ProcessRemainingColumnsM4

ProcessNextColumnLoopM4:
        lea     rax,[rdx+r8*2]              ; compute matrix A plus two rows
        vpmovzxbw ymm4,XMMWORD PTR [rdx]
        vpmovzxbw ymm5,XMMWORD PTR [rdx+r8]
        vpmovzxbw ymm6,XMMWORD PTR [rax]
        vpmovzxbw ymm7,XMMWORD PTR [rax+r8]
        lea     rax,[rcx+r11*4]             ; compute matrix D plus two rows
        vmovdqu YMMWORD PTR [rcx],ymm4
        vmovdqu YMMWORD PTR [rcx+r11*2],ymm5
        vmovdqu YMMWORD PTR [rax],ymm6
        vmovdqu YMMWORD PTR [rax+r11*2],ymm7
        vpaddw  ymm0,ymm0,ymm4              ; accumulate per row along columns
        vpaddw  ymm1,ymm1,ymm5
        vpaddw  ymm2,ymm2,ymm6
        vpaddw  ymm3,ymm3,ymm7
        add     rdx,16                      ; advance matrix A by 16 bytes
        add     rcx,16*2                    ; advance matrix D by 16 words
        sub     rbx,16                      ; subtract columns remaining
        jae     ProcessNextColumnLoopM4

ProcessRemainingColumnsM4:
        add     rbx,16                      ; correct for over-subtract above
        jz      ReduceRowSumVectorM4

;
; Copy the unaligned CountK columns to a zero padded stack buffer.
;

.errnz  GemmU8U8CopyPackAFrame.PaddedMatrixAData
        mov     rbp,rsp                     ; GemmU8U8CopyPackAFrame.PaddedMatrixAData
        test    bl,8                        ; (CountK & 8) != 0?
        jz      CopyRemainingCountKLessThan8M4
        lea     r13,[rdx+r8*2]              ; compute matrix A plus two rows
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        mov     rax,QWORD PTR [rdx+r8]
        mov     QWORD PTR [rbp+16],rax
        mov     rax,QWORD PTR [r13]
        mov     QWORD PTR [rbp+32],rax
        mov     rax,QWORD PTR [r13+r8]
        mov     QWORD PTR [rbp+48],rax
        add     rdx,8
        add     rbp,8                       ; advance padded buffer destination

CopyRemainingCountKLessThan8M4:
        test    bl,4                        ; (CountK & 4) != 0?
        jz      CopyRemainingCountKLessThan4M4
        lea     r13,[rdx+r8*2]              ; compute matrix A plus two rows
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        mov     eax,DWORD PTR [rdx+r8]
        mov     DWORD PTR [rbp+16],eax
        mov     eax,DWORD PTR [r13]
        mov     DWORD PTR [rbp+32],eax
        mov     eax,DWORD PTR [r13+r8]
        mov     DWORD PTR [rbp+48],eax
        add     rdx,4
        add     rbp,4                       ; advance padded buffer destination

CopyRemainingCountKLessThan4M4:
        test    bl,2                        ; (CountK & 2) != 0?
        jz      CopyRemainingCountKLessThan2M4
        lea     r13,[rdx+r8*2]              ; compute matrix A plus two rows
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        movzx   eax,WORD PTR [rdx+r8]
        mov     WORD PTR [rbp+16],ax
        movzx   eax,WORD PTR [r13]
        mov     WORD PTR [rbp+32],ax
        movzx   eax,WORD PTR [r13+r8]
        mov     WORD PTR [rbp+48],ax
        add     rdx,2
        add     rbp,2                       ; advance padded buffer destination

CopyRemainingCountKLessThan2M4:
        test    bl,1                        ; (CountK & 1) != 0?
        jz      ProcessPaddedMatrixADataM4
        lea     r13,[rdx+r8*2]              ; compute matrix A plus two rows
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al
        movzx   eax,BYTE PTR [rdx+r8]
        mov     BYTE PTR [rbp+16],al
        movzx   eax,BYTE PTR [r13]
        mov     BYTE PTR [rbp+32],al
        movzx   eax,BYTE PTR [r13+r8]
        mov     BYTE PTR [rbp+48],al

;
; Process the remaining CountK columns using the zero padded stack buffer.
;

ProcessPaddedMatrixADataM4:
        vpmovzxbw ymm4,XMMWORD PTR GemmU8U8CopyPackAFrame.PaddedMatrixAData[rsp]
        vpmovzxbw ymm5,XMMWORD PTR GemmU8U8CopyPackAFrame.PaddedMatrixAData[rsp+16]
        vpmovzxbw ymm6,XMMWORD PTR GemmU8U8CopyPackAFrame.PaddedMatrixAData[rsp+32]
        vpmovzxbw ymm7,XMMWORD PTR GemmU8U8CopyPackAFrame.PaddedMatrixAData[rsp+48]
        lea     rax,[rcx+r11*4]             ; compute matrix D plus two rows
        vpmaskmovd YMMWORD PTR [rcx],ymm9,ymm4
        vpmaskmovd YMMWORD PTR [rcx+r11*2],ymm9,ymm5
        vpmaskmovd YMMWORD PTR [rax],ymm9,ymm6
        vpmaskmovd YMMWORD PTR [rax+r11*2],ymm9,ymm7
        vpaddw  ymm0,ymm0,ymm4              ; accumulate per row along columns
        vpaddw  ymm1,ymm1,ymm5
        vpaddw  ymm2,ymm2,ymm6
        vpaddw  ymm3,ymm3,ymm7

;
; Reduce the sums for the four rows of output. Transpose the intermediate
; accumulators by treating the registers as 32-bit elements containing a pair
; of 16-bit sums. Continue reducing the transposed accumulators to produce the
; final 32-bit vector output.
;

ReduceRowSumVectorM4:
        vpunpckldq ymm4,ymm0,ymm1           ; [A5 B5 A4 B4 A1 B1 A0 B0]
        vpunpckhdq ymm5,ymm0,ymm1           ; [A7 B7 A6 B6 A3 B3 A2 B2]
        vpunpckldq ymm6,ymm2,ymm3           ; [C5 D5 C4 D4 C1 D1 C0 D0]
        vpunpckhdq ymm7,ymm2,ymm3           ; [C7 D7 C6 D6 C3 D3 C2 D2]
        vpunpcklqdq ymm0,ymm4,ymm6          ; [A4 B4 C4 D4 A0 B0 C0 D0]
        vpunpckhqdq ymm1,ymm4,ymm6          ; [A5 B5 C5 D5 A1 B1 C1 D1]
        vpunpcklqdq ymm2,ymm5,ymm7          ; [A6 B6 C6 D6 A2 B2 C2 D2]
        vpunpckhqdq ymm3,ymm5,ymm7          ; [A7 B7 C7 D7 A3 B3 C3 D3]
        vpaddw  ymm0,ymm0,ymm1              ; reduction
        vpaddw  ymm0,ymm0,ymm2
        vpaddw  ymm0,ymm0,ymm3
        vextracti128 xmm1,ymm0,1            ; extract high pairs
        vpaddw  xmm0,xmm0,xmm1              ; reduction
        vpmaddwd xmm0,xmm0,xmm8             ; multiply by offset and reduce
        vmovdqu XMMWORD PTR [r12],xmm0
        add     r12,4*4                     ; advance row sum vector by 4 dwords
        sub     r9,4                        ; subtract rows remaining
        jae     ProcessNextRowM4

ProcessRemainingRows:
        add     r9,4                        ; correct for over-subtract above
        jz      ExitRoutine

;
; Process a single row of matrix A in a loop.
;

ProcessNextRowM1:
        vpxor   xmm0,xmm0,xmm0              ; clear row accumulator
        mov     rdx,rsi
        mov     rcx,rdi
        add     rsi,r8
        lea     rdi,[rdi+r11*2]
        mov     rbx,r10                     ; reload columns remaining
        sub     rbx,16
        jb      ProcessRemainingColumnsM1

ProcessNextColumnLoopM1:
        vpmovzxbw ymm4,XMMWORD PTR [rdx]
        vmovdqu YMMWORD PTR [rcx],ymm4
        vpaddw  ymm0,ymm0,ymm4              ; accumulate per row along columns
        add     rdx,16                      ; advance matrix A by 16 bytes
        add     rcx,16*2                    ; advance matrix D by 16 words
        sub     rbx,16                      ; subtract columns remaining
        jae     ProcessNextColumnLoopM1

ProcessRemainingColumnsM1:
        add     rbx,16                      ; correct for over-subtract above
        jz      ReduceRowSumVectorM1

;
; Copy the unaligned CountK columns to a zero padded stack buffer.
;

.errnz  GemmU8U8CopyPackAFrame.PaddedMatrixAData
        mov     rbp,rsp                     ; GemmU8U8CopyPackAFrame.PaddedMatrixAData
        test    bl,8                        ; (CountK & 8) != 0?
        jz      CopyRemainingCountKLessThan8M1
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        add     rdx,8
        add     rbp,8                       ; advance padded buffer destination

CopyRemainingCountKLessThan8M1:
        test    bl,4                        ; (CountK & 4) != 0?
        jz      CopyRemainingCountKLessThan4M1
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        add     rdx,4
        add     rbp,4                       ; advance padded buffer destination

CopyRemainingCountKLessThan4M1:
        test    bl,2                        ; (CountK & 2) != 0?
        jz      CopyRemainingCountKLessThan2M1
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        add     rdx,2
        add     rbp,2                       ; advance padded buffer destination

CopyRemainingCountKLessThan2M1:
        test    bl,1                        ; (CountK & 1) != 0?
        jz      ProcessPaddedMatrixADataM1
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al

;
; Process the remaining CountK columns using the zero padded stack buffer.
;

ProcessPaddedMatrixADataM1:
        vpmovzxbw ymm4,XMMWORD PTR GemmU8U8CopyPackAFrame.PaddedMatrixAData[rsp]
        vpmaskmovd YMMWORD PTR [rcx],ymm9,ymm4
        vpaddw  ymm0,ymm0,ymm4              ; accumulate per row along columns

;
; Reduce the sum for the single row of output.
;

ReduceRowSumVectorM1:
        vextracti128 xmm1,ymm0,1            ; extract high pairs
        vpaddw  xmm0,xmm0,xmm1              ; reduction
        vphaddw xmm0,xmm0,xmm0
        vphaddw xmm0,xmm0,xmm0
        vpmaddwd xmm0,xmm0,xmm8             ; multiply by offset and reduce
        vmovd   DWORD PTR [r12],xmm0
        add     r12,4                       ; advance row sum vector by 1 DWORD
        dec     r9                          ; decrement rows remaining
        jnz     ProcessNextRowM1

;
; Restore non-volatile registers and return.
;

ExitRoutine:
        vzeroupper
        vmovaps xmm6,GemmU8U8CopyPackAFrame.SavedXmm6[rsp]
        vmovaps xmm7,GemmU8U8CopyPackAFrame.SavedXmm7[rsp]
        vmovaps xmm8,GemmU8U8CopyPackAFrame.SavedXmm8[rsp]
        vmovaps xmm9,GemmU8U8CopyPackAFrame.SavedXmm9[rsp]
        add     rsp,(GemmU8U8CopyPackAFrame.SavedR13)

        BEGIN_EPILOGUE

        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasGemmU8U8CopyPackAAvx2, _TEXT

;++
;
; Routine Description:
;
;   This routine copies elements from the source matrix to the destination
;   packed buffer.
;
; Arguments:
;
;   D (rcx) - Supplies the address of the destination packed buffer.
;
;   B (rdx) - Supplies the address of the source matrix.
;
;   ldb (r8) - Supplies the number of elements per row of the source matrix.
;
;   CountN (r9) - Supplies the number of columns of the source matrix to copy.
;
;   CountK - Supplies the number of rows of the source matrix to copy.
;
;   ColumnSumVector - Supplies the address of the buffer to receive the sums of
;       the elements from each of the columns. Each sum has also been multiplied
;       by the zero point offset.
;
;   offa - Supplies the zero point offset for the other source matrix of the
;       matrix multiplication.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasGemmU8U8CopyPackBAvx2, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        alloc_stack (GemmU8U8CopyPackBFrame.SavedRsi)

        END_PROLOGUE

        mov     rsi,rdx
        mov     r10,GemmU8U8CopyPackBFrame.CountK[rsp]
        mov     r11,GemmU8U8CopyPackBFrame.ColumnSumVector[rsp]
        vpbroadcastw ymm5,WORD PTR GemmU8U8CopyPackBFrame.offa[rsp]

;
; Zero initialize the padded stack buffers.
;

        vpxor   xmm0,xmm0,xmm0
        vmovdqu YMMWORD PTR GemmU8U8CopyPackBFrame.PaddedMatrixBData[rsp],ymm0

;
; Process 16 columns of matrix B in a loop.
;

        sub     r9,16
        jb      ProcessRemainingColumns

ProcessNextColumnN16:
        vpxor   xmm0,xmm0,xmm0              ; clear column accumulators
        vpxor   xmm1,xmm1,xmm1
        mov     rdx,rsi
        add     rsi,16                      ; advance next matrix B by 16 columns
        mov     rbx,r10                     ; reload rows remaining
        sub     rbx,2
        jb      ProcessRemainingRowsN16

ProcessNextRowLoopN16:
        vmovdqu xmm2,XMMWORD PTR [rdx]      ; load two rows
        vmovdqu xmm3,XMMWORD PTR [rdx+r8]
        lea     rdx,[rdx+r8*2]              ; advance matrix B by two rows
        vpunpcklbw xmm4,xmm2,xmm3           ; interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vmovdqu XMMWORD PTR [rcx],xmm4      ; store interleaved rows
        vmovdqu XMMWORD PTR [rcx+16],xmm3
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        add     rcx,32                      ; advance matrix D by 32 bytes
        vpaddw  ymm0,ymm0,ymm4              ; accumulate per column
        vpaddw  ymm1,ymm1,ymm3
        sub     rbx,2                       ; subtract columns remaining
        jae     ProcessNextRowLoopN16

ProcessRemainingRowsN16:
        add     rbx,2                       ; correct for over-subtract above
        jz      ReduceColumnSumVectorN16
        vpmovzxbw ymm4,XMMWORD PTR [rdx]
        vmovdqu YMMWORD PTR [rcx],ymm4      ; store interleaved rows
        vextracti128 xmm3,ymm4,1
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        vpaddw  ymm0,ymm0,ymm4              ; accumulate per column
        vpaddw  ymm1,ymm1,ymm3
        add     rcx,32                      ; advance matrix D by 32 bytes

ReduceColumnSumVectorN16:
        vpmaddwd ymm0,ymm0,ymm5             ; multiply by offset and reduce
        vpmaddwd ymm1,ymm1,ymm5             ; multiply by offset and reduce
        vmovdqu YMMWORD PTR [r11],ymm0
        vmovdqu YMMWORD PTR [r11+32],ymm1
        add     r11,64                      ; advance column sum vector by 16 dwords
        sub     r9,16                       ; subtract columns remaining
        jae     ProcessNextColumnN16

ProcessRemainingColumns:
        add     r9,16                       ; correct for over-subtract above
        jnz     ProcessColumnNUnaligned

;
; Restore non-volatile registers and return.
;

ExitRoutine:
        vzeroupper
        add     rsp,(GemmU8U8CopyPackBFrame.SavedRsi)

        BEGIN_EPILOGUE

        pop     rsi
        pop     rbx
        pop     rbp
        ret

;
; Process the remaining columns of matrix B.
;

ProcessColumnNUnaligned:
        vpxor   xmm0,xmm0,xmm0              ; clear column accumulators
        vpxor   xmm1,xmm1,xmm1
        sub     r10,2
        jb      ProcessRemainingRowsNUnaligned

ProcessNextRowLoopNUnaligned:
        mov     rdx,rsi
.errnz  GemmU8U8CopyPackBFrame.PaddedMatrixBData
        mov     rbp,rsp                     ; GemmU8U8CopyPackBFrame.PaddedMatrixBData
        test    r9b,8                       ; (CountN & 8) != 0?
        jz      CopyRemainingCountNLessThan8K2
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        mov     rax,QWORD PTR [rdx+r8]
        mov     QWORD PTR [rbp+16],rax
        add     rdx,8                       ; advance matrix B
        add     rbp,8                       ; advance padded buffer destination

CopyRemainingCountNLessThan8K2:
        test    r9b,4                       ; (CountN & 4) != 0?
        jz      CopyRemainingCountNLessThan4K2
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        mov     eax,DWORD PTR [rdx+r8]
        mov     DWORD PTR [rbp+16],eax
        add     rdx,4                       ; advance matrix B
        add     rbp,4                       ; advance padded buffer destination

CopyRemainingCountNLessThan4K2:
        test    r9b,2                       ; (CountN & 2) != 0?
        jz      CopyRemainingCountNLessThan2K2
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        movzx   eax,WORD PTR [rdx+r8]
        mov     WORD PTR [rbp+16],ax
        add     rdx,2                       ; advance matrix B
        add     rbp,2                       ; advance padded buffer destination

CopyRemainingCountNLessThan2K2:
        test    r9b,1                       ; (CountN & 1) != 0?
        jz      ProcessPaddedMatrixBDataK2
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al
        movzx   eax,BYTE PTR [rdx+r8]
        mov     BYTE PTR [rbp+16],al

ProcessPaddedMatrixBDataK2:
        vmovdqu xmm2,XMMWORD PTR XMMWORD PTR GemmU8U8CopyPackBFrame.PaddedMatrixBData[rsp]
        vmovdqu xmm3,XMMWORD PTR XMMWORD PTR GemmU8U8CopyPackBFrame.PaddedMatrixBData[rsp+16]
        vpunpcklbw xmm4,xmm2,xmm3           ; interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vmovdqu XMMWORD PTR [rcx],xmm4      ; store interleaved rows
        vmovdqu XMMWORD PTR [rcx+16],xmm3
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        vpaddw  ymm0,ymm0,ymm4              ; accumulate per column
        vpaddw  ymm1,ymm1,ymm3
        lea     rsi,[rsi+r8*2]              ; advance next matrix B by two rows
        add     rcx,32                      ; advance matrix D by 32 bytes
        sub     r10,2                       ; subtract columns remaining
        jae     ProcessNextRowLoopNUnaligned

ProcessRemainingRowsNUnaligned:
        add     r10,2
        jz      ReduceColumnSumVectorNUnaligned
        mov     rdx,rsi
.errnz  GemmU8U8CopyPackBFrame.PaddedMatrixBData
        mov     rbp,rsp                     ; GemmU8U8CopyPackBFrame.PaddedMatrixBData
        test    r9b,8                       ; (CountN & 8) != 0?
        jz      CopyRemainingCountNLessThan8K1
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        add     rdx,8                       ; advance matrix B
        add     rbp,8                       ; advance padded buffer destination

CopyRemainingCountNLessThan8K1:
        test    r9b,4                       ; (CountN & 4) != 0?
        jz      CopyRemainingCountNLessThan4K1
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        add     rdx,4                       ; advance matrix B
        add     rbp,4                       ; advance padded buffer destination

CopyRemainingCountNLessThan4K1:
        test    r9b,2                       ; (CountN & 2) != 0?
        jz      CopyRemainingCountNLessThan2K1
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        add     rdx,2                       ; advance matrix B
        add     rbp,2                       ; advance padded buffer destination

CopyRemainingCountNLessThan2K1:
        test    r9b,1                       ; (CountN & 1) != 0?
        jz      ProcessPaddedMatrixBDataK1
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al

ProcessPaddedMatrixBDataK1:
        vpmovzxbw ymm4,XMMWORD PTR GemmU8U8CopyPackBFrame.PaddedMatrixBData[rsp]
        vmovdqu YMMWORD PTR [rcx],ymm4      ; store interleaved rows
        vextracti128 xmm3,ymm4,1
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        vpaddw  ymm0,ymm0,ymm4              ; accumulate per column
        vpaddw  ymm1,ymm1,ymm3

ReduceColumnSumVectorNUnaligned:
        vpmaddwd ymm0,ymm0,ymm5             ; multiply by offset and reduce
        vpmaddwd ymm1,ymm1,ymm5             ; multiply by offset and reduce
        vmovdqu YMMWORD PTR [r11],ymm0
        vmovdqu YMMWORD PTR [r11+32],ymm1
        jmp     ExitRoutine

        NESTED_END MlasGemmU8U8CopyPackBAvx2, _TEXT

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
        vpmaddwd ymm3,ymm2,ymm0
        vpaddd  Vec1Reg,Vec1Reg,ymm3
        vpmaddwd ymm2,ymm2,ymm1
        vpaddd  Vec2Reg,Vec2Reg,ymm2
ELSE
        vpmaddwd ymm3,ymm2,ymm0
        vpaddd  Vec2Reg,Vec2Reg,ymm3
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
;   r10 - Supplies the length in bytes of a row from matrix A.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlock MACRO ColumnCount, RowCount, VectorOffset, BroadcastOffset

        vpmovzxbw ymm0,XMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE ColumnCount, 16, <vpmovzxbw ymm1,XMMWORD PTR [rdx+VectorOffset+16]>
        EmitIfCountGE RowCount, 1, <vpbroadcastd ymm2,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE RowCount, 1, <MultiplyAccumulateRow ColumnCount, ymm4, ymm5>
        EmitIfCountGE RowCount, 2, <vpbroadcastd ymm2,DWORD PTR [rcx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 2, <MultiplyAccumulateRow ColumnCount, ymm6, ymm7>
        EmitIfCountGE RowCount, 3, <vpbroadcastd ymm2,DWORD PTR [rcx+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 3, <MultiplyAccumulateRow ColumnCount, ymm8, ymm9>
        EmitIfCountGE RowCount, 4, <vpbroadcastd ymm2,DWORD PTR [rbx+BroadcastOffset]>
        EmitIfCountGE RowCount, 4, <MultiplyAccumulateRow ColumnCount, ymm10, ymm11>
        EmitIfCountGE RowCount, 5, <vpbroadcastd ymm2,DWORD PTR [rbx+r10+BroadcastOffset]>
        EmitIfCountGE RowCount, 5, <MultiplyAccumulateRow ColumnCount, ymm12, ymm13>
        EmitIfCountGE RowCount, 6, <vpbroadcastd ymm2,DWORD PTR [rbx+r10*2+BroadcastOffset]>
        EmitIfCountGE RowCount, 6, <MultiplyAccumulateRow ColumnCount, ymm14, ymm15>

        ENDM

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
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockLoopExit

;
; Initialize the accumulators with the sum of the global depth value constant,
; the column sums, and the row sums.
;

        vpbroadcastd ymm1,DWORD PTR GemmU8U8KernelFrame.DepthValue[rsp]
IF ColumnCount EQ 16
        vpaddd  ymm0,ymm1,YMMWORD PTR [r13]
        vpaddd  ymm1,ymm1,YMMWORD PTR [r13+32]
        add     r13,16*4                    ; advance ColumnSumVector by 16 columns
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
; Iterate over PairedCountK elements from matrix A and matrix B.
;
; Unrolling the loop to do two iterations improves performance slightly at the
; cost of larger code size. Balance this by only unrolling for the common case
; of computing 16 columns for an even number of rows.
;

        mov     rsi,r9                      ; reload PairedCountK
IF RowCount GT 3
        lea     rbx,[r10*2+r10]
        add     rbx,rcx                     ; compute matrix A plus 3 rows
ENDIF

IF (ColumnCount EQ 16) AND ((RowCount AND 1) EQ 0)
        sub     rsi,2
        jb      ProcessRemainingBlocks

ComputeBlockLoop:
        ComputeBlock ColumnCount, RowCount, 0, 0
        ComputeBlock ColumnCount, RowCount, 32, 4
        add     rcx,2*4                     ; advance matrix A by 2 pairs
IF RowCount GT 3
        add     rbx,2*4                     ; advance matrix A plus 3 rows by 2 pairs
ENDIF
        add     rdx,2*32                    ; advance matrix B by 64 columns
        sub     rsi,2                       ; subtract pairs remaining
        jae     ComputeBlockLoop

ProcessRemainingBlocks:
        add     rsi,2                       ; correct for over-subtract above
        jz      ComputeBlockLoopExit
        ComputeBlock ColumnCount, RowCount, 0, 0
        add     rdx,32                      ; advance matrix B by 32 columns
ELSE
ComputeBlockLoop:
        ComputeBlock ColumnCount, RowCount, 0, 0
        add     rcx,4                       ; advance matrix A by 1 pair
IF RowCount GT 3
        add     rbx,4                       ; advance matrix A plus 3 rows by 1 pair
ENDIF
        add     rdx,32
        dec     rsi                         ; decrement pairs remaining
        jnz     ComputeBlockLoop
ENDIF

ComputeBlockLoopExit:
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
        test    r14b,r14b                   ; ZeroMode?
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
        test    r14b,r14b                   ; ZeroMode?
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
        test    r14b,r14b                   ; ZeroMode?
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
        mov     DWORD PTR GemmU8U8KernelFrame.CountN[rsp],ebp
        vpbroadcastd ymm0,DWORD PTR GemmU8U8KernelFrame.CountN[rsp]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR [MlasMaskMoveAvx]
        test    r14b,r14b                   ; ZeroMode?
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

        NESTED_ENTRY MlasGemmU8U8KernelAvx2, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        push_reg r14
        alloc_stack (GemmU8U8KernelFrame.SavedR14)
        save_xmm128_avx xmm6,GemmU8U8KernelFrame.SavedXmm6
        save_xmm128_avx xmm7,GemmU8U8KernelFrame.SavedXmm7
        save_xmm128_avx xmm8,GemmU8U8KernelFrame.SavedXmm8
        save_xmm128_avx xmm9,GemmU8U8KernelFrame.SavedXmm9
        save_xmm128_avx xmm10,GemmU8U8KernelFrame.SavedXmm10
        save_xmm128_avx xmm11,GemmU8U8KernelFrame.SavedXmm11
        save_xmm128_avx xmm12,GemmU8U8KernelFrame.SavedXmm12
        save_xmm128_avx xmm13,GemmU8U8KernelFrame.SavedXmm13
        save_xmm128_avx xmm14,GemmU8U8KernelFrame.SavedXmm14
        save_xmm128_avx xmm15,GemmU8U8KernelFrame.SavedXmm15

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
        vmovaps xmm6,GemmU8U8KernelFrame.SavedXmm6[rsp]
        vmovaps xmm7,GemmU8U8KernelFrame.SavedXmm7[rsp]
        vmovaps xmm8,GemmU8U8KernelFrame.SavedXmm8[rsp]
        vmovaps xmm9,GemmU8U8KernelFrame.SavedXmm9[rsp]
        vmovaps xmm10,GemmU8U8KernelFrame.SavedXmm10[rsp]
        vmovaps xmm11,GemmU8U8KernelFrame.SavedXmm11[rsp]
        vmovaps xmm12,GemmU8U8KernelFrame.SavedXmm12[rsp]
        vmovaps xmm13,GemmU8U8KernelFrame.SavedXmm13[rsp]
        vmovaps xmm14,GemmU8U8KernelFrame.SavedXmm14[rsp]
        vmovaps xmm15,GemmU8U8KernelFrame.SavedXmm15[rsp]
        add     rsp,(GemmU8U8KernelFrame.SavedR14)

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

        NESTED_END MlasGemmU8U8KernelAvx2, _TEXT

        END
