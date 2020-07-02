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
;   This implementation uses AVX2 instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE QgemmU8X8KernelAvx2Common.inc
        .list

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
        RowSumBuffer QWORD ?

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
        ColumnSumBuffer QWORD ?

GemmU8U8CopyPackBFrame ENDS

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
;   RowSumBuffer - Supplies the address of the buffer to receive the sums of
;       the elements along each of the rows.
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
        save_xmm128 xmm6,GemmU8U8CopyPackAFrame.SavedXmm6
        save_xmm128 xmm7,GemmU8U8CopyPackAFrame.SavedXmm7
        save_xmm128 xmm8,GemmU8U8CopyPackAFrame.SavedXmm8
        save_xmm128 xmm9,GemmU8U8CopyPackAFrame.SavedXmm9

        END_PROLOGUE

        mov     rdi,rcx
        mov     rsi,rdx
        mov     r10,GemmU8U8CopyPackAFrame.CountK[rsp]
        lea     r11,[r10+1]
        and     r11,NOT 1                   ; align CountK up to pair count
        mov     r12,GemmU8U8CopyPackAFrame.RowSumBuffer[rsp]
        vpcmpeqw ymm8,ymm8,ymm8             ; generate word vector [0xFFFF]
        vpsrlw  ymm8,ymm8,15                ; generate word vector [0x0001]

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
; Zero extend the source bytes to 16-bits and write to the packed buffer.
;
; The packed buffer has the same data ordering as the source bytes, but CountK
; is aligned up to a multiple of 2 to maintain 32-bit alignment. All padding
; bytes are zero filled.
;
; These 16-bit values are also accumulated into an intermediate per-row
; accumulator. CountK cannot be greater than 128 to avoid overflowing these
; signed 16-bit accumulators.
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
        lea     rdi,[rdi+r11*8]             ; advance next matrix D by 4 rows
        mov     rbx,r10                     ; reload columns remaining
        sub     rbx,16
        jb      ProcessRemainingColumnsM4

ProcessNextColumnLoopM4:
        lea     rax,[rdx+r8*2]              ; compute matrix A plus 2 rows
        vpmovzxbw ymm4,XMMWORD PTR [rdx]
        vpmovzxbw ymm5,XMMWORD PTR [rdx+r8]
        vpmovzxbw ymm6,XMMWORD PTR [rax]
        vpmovzxbw ymm7,XMMWORD PTR [rax+r8]
        lea     rax,[rcx+r11*4]             ; compute matrix D plus 2 rows
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
        jz      ReduceRowSumBufferM4

;
; Copy the unaligned CountK columns to a zero padded stack buffer.
;

.errnz  GemmU8U8CopyPackAFrame.PaddedMatrixAData
        mov     rbp,rsp                     ; GemmU8U8CopyPackAFrame.PaddedMatrixAData
        test    bl,8                        ; (CountK & 8) != 0?
        jz      CopyRemainingCountKLessThan8M4
        lea     r13,[rdx+r8*2]              ; compute matrix A plus 2 rows
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
        lea     r13,[rdx+r8*2]              ; compute matrix A plus 2 rows
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
        lea     r13,[rdx+r8*2]              ; compute matrix A plus 2 rows
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
        lea     r13,[rdx+r8*2]              ; compute matrix A plus 2 rows
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
        lea     rax,[rcx+r11*4]             ; compute matrix D plus 2 rows
        vpmaskmovd YMMWORD PTR [rcx],ymm9,ymm4
        vpmaskmovd YMMWORD PTR [rcx+r11*2],ymm9,ymm5
        vpmaskmovd YMMWORD PTR [rax],ymm9,ymm6
        vpmaskmovd YMMWORD PTR [rax+r11*2],ymm9,ymm7
        vpaddw  ymm0,ymm0,ymm4              ; accumulate per row along columns
        vpaddw  ymm1,ymm1,ymm5
        vpaddw  ymm2,ymm2,ymm6
        vpaddw  ymm3,ymm3,ymm7

;
; Reduce the sums for the four rows of output.
;

ReduceRowSumBufferM4:
        vpmaddwd ymm0,ymm0,ymm8             ; horizontal word+word=dword per row
        vpmaddwd ymm1,ymm1,ymm8
        vphaddd ymm0,ymm0,ymm1              ; reduce and interleave Sum1/Sum0
        vpmaddwd ymm2,ymm2,ymm8
        vpmaddwd ymm3,ymm3,ymm8
        vphaddd ymm1,ymm2,ymm3              ; reduce and interleave Sum3/Sum2
        vphaddd ymm0,ymm0,ymm1              ; reduce and interleave Sum3/Sum2/Sum1/Sum0
        vextracti128 xmm1,ymm0,1            ; extract high dwords
        vpaddd  xmm0,xmm0,xmm1              ; reduce low/high dwords
        vmovdqu XMMWORD PTR [r12],xmm0
        add     r12,4*4                     ; advance row sum buffer by 4 dwords
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
        jz      ReduceRowSumBufferM1

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

ReduceRowSumBufferM1:
        vpmaddwd ymm0,ymm0,ymm8             ; horizontal word+word=dword per row
        vextracti128 xmm1,ymm0,1            ; extract high dwords
        vpaddd  xmm0,xmm0,xmm1              ; reduction
        vphaddd xmm0,xmm0,xmm0
        vphaddd xmm0,xmm0,xmm0
        vmovd   DWORD PTR [r12],xmm0
        add     r12,4                       ; advance row sum buffer by 1 dword
        dec     r9                          ; decrement rows remaining
        jnz     ProcessNextRowM1

;
; Restore non-volatile registers and return.
;

ExitRoutine:
        vzeroupper
        movaps  xmm6,GemmU8U8CopyPackAFrame.SavedXmm6[rsp]
        movaps  xmm7,GemmU8U8CopyPackAFrame.SavedXmm7[rsp]
        movaps  xmm8,GemmU8U8CopyPackAFrame.SavedXmm8[rsp]
        movaps  xmm9,GemmU8U8CopyPackAFrame.SavedXmm9[rsp]
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
;   ColumnSumBuffer - Supplies the address of the buffer to receive the sums of
;       the elements along each of the columns.
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
        mov     r11,GemmU8U8CopyPackBFrame.ColumnSumBuffer[rsp]
        vpcmpeqw ymm5,ymm5,ymm5             ; generate word vector [0xFFFF]
        vpsrlw  ymm5,ymm5,15                ; generate word vector [0x0001]

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
        vmovdqu xmm2,XMMWORD PTR [rdx]      ; load 2 rows
        vmovdqu xmm3,XMMWORD PTR [rdx+r8]
        lea     rdx,[rdx+r8*2]              ; advance matrix B by 2 rows
        vpunpcklbw xmm4,xmm2,xmm3           ; interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vmovdqu XMMWORD PTR [rcx],xmm4      ; store interleaved rows
        vmovdqu XMMWORD PTR [rcx+16],xmm3
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        add     rcx,32                      ; advance matrix D by 32 bytes
        vpmaddwd ymm4,ymm4,ymm5             ; horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              ; accumulate per column
        vpmaddwd ymm3,ymm3,ymm5
        vpaddd  ymm1,ymm1,ymm3
        sub     rbx,2                       ; subtract rows remaining
        jae     ProcessNextRowLoopN16

ProcessRemainingRowsN16:
        add     rbx,2                       ; correct for over-subtract above
        jz      StoreColumnSumBufferN16
        vpmovzxbw ymm4,XMMWORD PTR [rdx]
        vmovdqu YMMWORD PTR [rcx],ymm4      ; store interleaved rows
        vextracti128 xmm3,ymm4,1
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        vpmaddwd ymm4,ymm4,ymm5             ; horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              ; accumulate per column
        vpmaddwd ymm3,ymm3,ymm5
        vpaddd  ymm1,ymm1,ymm3
        add     rcx,32                      ; advance matrix D by 32 bytes

StoreColumnSumBufferN16:
        vmovdqu YMMWORD PTR [r11],ymm0
        vmovdqu YMMWORD PTR [r11+32],ymm1
        add     r11,16*4                    ; advance column sum buffer by 16 dwords
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
        vpmaddwd ymm4,ymm4,ymm5             ; horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              ; accumulate per column
        vpmaddwd ymm3,ymm3,ymm5
        vpaddd  ymm1,ymm1,ymm3
        lea     rsi,[rsi+r8*2]              ; advance next matrix B by 2 rows
        add     rcx,32                      ; advance matrix D by 32 bytes
        sub     r10,2                       ; subtract columns remaining
        jae     ProcessNextRowLoopNUnaligned

ProcessRemainingRowsNUnaligned:
        add     r10,2
        jz      StoreColumnSumBufferNUnaligned
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
        vpmaddwd ymm4,ymm4,ymm5             ; horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              ; accumulate per column
        vpmaddwd ymm3,ymm3,ymm5
        vpaddd  ymm1,ymm1,ymm3

StoreColumnSumBufferNUnaligned:
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

        vpmaddwd ymm3,ymm2,ymm0
IF ColumnCount EQ 16
        vpaddd  Vec1Reg,Vec1Reg,ymm3
        vpmaddwd ymm2,ymm2,ymm1
        vpaddd  Vec2Reg,Vec2Reg,ymm2
ELSE
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
;   r9 - Supplies the length in bytes of a row from matrix A.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlock MACRO ColumnCount, RowCount, VectorOffset, BroadcastOffset

        vpmovzxbw ymm0,XMMWORD PTR [rdx+VectorOffset]
        EmitIfCountGE ColumnCount, 16, <vpmovzxbw ymm1,XMMWORD PTR [rdx+VectorOffset+16]>
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

        LOCAL   ComputeBlockBy2Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   ComputeBlockLoopExit

        mov     rsi,r9                      ; reload row length remaining

IF (ColumnCount EQ 16) AND ((RowCount AND 1) EQ 0)
        sub     rsi,2*4
        jb      ProcessRemainingBlocks

ComputeBlockBy2Loop:
        ComputeBlock ColumnCount, RowCount, 0, 0
        ComputeBlock ColumnCount, RowCount, 32, 4
        add     rcx,2*4                     ; advance matrix A by 2 pairs
IF RowCount GT 3
        add     rbx,2*4                     ; advance matrix A plus 3 rows by 2 pairs
ENDIF
        add     rdx,2*32                    ; advance matrix B
        sub     rsi,2*4
        jae     ComputeBlockBy2Loop

ProcessRemainingBlocks:
        add     rsi,2*4                     ; correct for over-subtract above
        jz      ComputeBlockLoopExit
        ComputeBlock ColumnCount, RowCount, 0, 0
        add     rdx,32                      ; advance matrix B
ELSE
ComputeBlockBy1Loop:
        ComputeBlock ColumnCount, RowCount, 0, 0
        add     rcx,4                       ; advance matrix A by 1 pair
IF RowCount GT 3
        add     rbx,4                       ; advance matrix A plus 3 rows by 1 pair
ENDIF
        add     rdx,32                      ; advance matrix B
        sub     rsi,4
        jnz     ComputeBlockBy1Loop
ENDIF

ComputeBlockLoopExit:

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

        NESTED_ENTRY MlasGemmU8U8KernelAvx2, _TEXT

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

        NESTED_END MlasGemmU8U8KernelAvx2, _TEXT

        END
