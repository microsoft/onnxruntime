;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemmU8S8KernelAvx2.asm
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
        .list

        EXTERN  MlasMaskMoveTableAvx:NEAR

;
; Stack frame layout for the U8S8 CopyPackA routine.
;

GemmU8S8CopyPackAFrame STRUCT

        PaddedMatrixAData OWORD 4 DUP (?)
        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
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

GemmU8S8CopyPackAFrame ENDS

;
; Stack frame layout for the U8S8 CopyPackB routine.
;

GemmU8S8CopyPackBFrame STRUCT

        PaddedMatrixBData OWORD 4 DUP (?)
        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        Padding QWORD ?
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
        ColumnSumBuffer QWORD ?
        BIsSigned QWORD ?

GemmU8S8CopyPackBFrame ENDS

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

        NESTED_ENTRY MlasGemmU8S8CopyPackAAvx2, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        alloc_stack (GemmU8S8CopyPackAFrame.SavedR13)
        save_xmm128 xmm6,GemmU8S8CopyPackAFrame.SavedXmm6
        save_xmm128 xmm7,GemmU8S8CopyPackAFrame.SavedXmm7
        save_xmm128 xmm8,GemmU8S8CopyPackAFrame.SavedXmm8
        save_xmm128 xmm9,GemmU8S8CopyPackAFrame.SavedXmm9
        save_xmm128 xmm10,GemmU8S8CopyPackAFrame.SavedXmm10

        END_PROLOGUE

        mov     rdi,rcx
        mov     rsi,rdx
        mov     r10,GemmU8S8CopyPackAFrame.CountK[rsp]
        lea     r11,[r10+3]
        and     r11,NOT 3                   ; align CountK up to quad count
        mov     r12,GemmU8S8CopyPackAFrame.RowSumBuffer[rsp]
        vpcmpeqw ymm8,ymm8,ymm8             ; generate word vector [0xFFFF]
        vpsrlw  ymm8,ymm8,15                ; generate word vector [0x0001]
        vpsllw  ymm9,ymm8,8                 ; generate word vector [0x0100]
        vpor    ymm9,ymm8,ymm9              ; generate word vector [0x0101]

;
; Compute the conditional load/store mask for an unaligned CountK.
;

        mov     eax,r10d
        and     eax,15                      ; isolate unaligned count
        add     eax,3
        shr     eax,2                       ; align unaligned count to quad count
        neg     rax
        lea     rbx,MlasMaskMoveTableAvx+8*4
        vmovdqu xmm10,XMMWORD PTR [rbx+rax*4]

;
; Zero initialize the padded stack buffers.
;

        vpxor   xmm0,xmm0,xmm0
        vmovdqu YMMWORD PTR GemmU8S8CopyPackAFrame.PaddedMatrixAData[rsp],ymm0
        vmovdqu YMMWORD PTR GemmU8S8CopyPackAFrame.PaddedMatrixAData[rsp+32],ymm0

;
; Process 4 rows of matrix A in a loop.
;

        sub     r9,4
        jb      ProcessRemainingRows

ProcessNextRowM4:
        vpxor   xmm0,xmm0,xmm0              ; clear row accumulators
        vpxor   xmm1,xmm1,xmm1
        vpxor   xmm2,xmm2,xmm2
        vpxor   xmm3,xmm3,xmm3
        lea     r13,[r8+r8*2]               ; compute lda * 3
        lea     rax,[r11+r11*2]             ; compute output stride * 3
        mov     rdx,rsi
        mov     rcx,rdi
        lea     rsi,[rsi+r8*4]              ; advance next matrix A by 4 rows
        lea     rdi,[rdi+r11*4]             ; advance next matrix D by 4 rows
        mov     rbx,r10                     ; reload columns remaining
        sub     rbx,32
        jb      ProcessRemainingColumnsM4

ProcessNextColumnLoopM4:
        vmovdqu ymm4,YMMWORD PTR [rdx]
        vmovdqu ymm5,YMMWORD PTR [rdx+r8]
        vmovdqu ymm6,YMMWORD PTR [rdx+r8*2]
        vmovdqu ymm7,YMMWORD PTR [rdx+r13]
        vmovdqu YMMWORD PTR [rcx],ymm4
        vmovdqu YMMWORD PTR [rcx+r11],ymm5
        vmovdqu YMMWORD PTR [rcx+r11*2],ymm6
        vmovdqu YMMWORD PTR [rcx+rax],ymm7
        vpmaddubsw ymm4,ymm4,ymm9           ; horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              ; add words to row accumulators
        vpmaddubsw ymm5,ymm5,ymm9
        vpaddw  ymm1,ymm1,ymm5
        vpmaddubsw ymm6,ymm6,ymm9
        vpaddw  ymm2,ymm2,ymm6
        vpmaddubsw ymm7,ymm7,ymm9
        vpaddw  ymm3,ymm3,ymm7
        add     rdx,32                      ; advance matrix A by 32 bytes
        add     rcx,32                      ; advance matrix D by 32 bytes
        sub     rbx,32                      ; subtract columns remaining
        jae     ProcessNextColumnLoopM4

ProcessRemainingColumnsM4:
        add     rbx,32                      ; correct for over-subtract above
        jz      ReduceRowSumBufferM4
        test    bl,16                       ; (CountK & 16) != 0?
        jz      CopyRemainingCountKLessThan16M4
        vmovdqu xmm4,XMMWORD PTR [rdx]
        vmovdqu xmm5,XMMWORD PTR [rdx+r8]
        vmovdqu xmm6,XMMWORD PTR [rdx+r8*2]
        vmovdqu xmm7,XMMWORD PTR [rdx+r13]
        vmovdqu XMMWORD PTR [rcx],xmm4
        vmovdqu XMMWORD PTR [rcx+r11],xmm5
        vmovdqu XMMWORD PTR [rcx+r11*2],xmm6
        vmovdqu XMMWORD PTR [rcx+rax],xmm7
        vpmaddubsw xmm4,xmm4,xmm9           ; horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              ; add words to row accumulators
        vpmaddubsw xmm5,xmm5,xmm9
        vpaddw  ymm1,ymm1,ymm5
        vpmaddubsw xmm6,xmm6,xmm9
        vpaddw  ymm2,ymm2,ymm6
        vpmaddubsw xmm7,xmm7,xmm9
        vpaddw  ymm3,ymm3,ymm7
        add     rdx,16                      ; advance matrix A by 16 bytes
        add     rcx,16                      ; advance matrix D by 16 bytes
        test    bl,15                       ; test for unaligned columns
        jz      ReduceRowSumBufferM4

;
; Copy the unaligned CountK columns to a zero padded stack buffer.
;

CopyRemainingCountKLessThan16M4:
.errnz  GemmU8S8CopyPackAFrame.PaddedMatrixAData
        mov     rbp,rsp                     ; GemmU8S8CopyPackAFrame.PaddedMatrixAData
        test    bl,8                        ; (CountK & 8) != 0?
        jz      CopyRemainingCountKLessThan8M4
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        mov     rax,QWORD PTR [rdx+r8]
        mov     QWORD PTR [rbp+16],rax
        mov     rax,QWORD PTR [rdx+r8*2]
        mov     QWORD PTR [rbp+32],rax
        mov     rax,QWORD PTR [rdx+r13]
        mov     QWORD PTR [rbp+48],rax
        add     rdx,8
        add     rbp,8                       ; advance padded buffer destination

CopyRemainingCountKLessThan8M4:
        test    bl,4                        ; (CountK & 4) != 0?
        jz      CopyRemainingCountKLessThan4M4
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        mov     eax,DWORD PTR [rdx+r8]
        mov     DWORD PTR [rbp+16],eax
        mov     eax,DWORD PTR [rdx+r8*2]
        mov     DWORD PTR [rbp+32],eax
        mov     eax,DWORD PTR [rdx+r13]
        mov     DWORD PTR [rbp+48],eax
        add     rdx,4
        add     rbp,4                       ; advance padded buffer destination

CopyRemainingCountKLessThan4M4:
        test    bl,2                        ; (CountK & 2) != 0?
        jz      CopyRemainingCountKLessThan2M4
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        movzx   eax,WORD PTR [rdx+r8]
        mov     WORD PTR [rbp+16],ax
        movzx   eax,WORD PTR [rdx+r8*2]
        mov     WORD PTR [rbp+32],ax
        movzx   eax,WORD PTR [rdx+r13]
        mov     WORD PTR [rbp+48],ax
        add     rdx,2
        add     rbp,2                       ; advance padded buffer destination

CopyRemainingCountKLessThan2M4:
        test    bl,1                        ; (CountK & 1) != 0?
        jz      ProcessPaddedMatrixADataM4
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al
        movzx   eax,BYTE PTR [rdx+r8]
        mov     BYTE PTR [rbp+16],al
        movzx   eax,BYTE PTR [rdx+r8*2]
        mov     BYTE PTR [rbp+32],al
        movzx   eax,BYTE PTR [rdx+r13]
        mov     BYTE PTR [rbp+48],al

;
; Process the remaining CountK columns using the zero padded stack buffer.
;

ProcessPaddedMatrixADataM4:
        vmovdqu xmm4,XMMWORD PTR GemmU8S8CopyPackAFrame.PaddedMatrixAData[rsp]
        vmovdqu xmm5,XMMWORD PTR GemmU8S8CopyPackAFrame.PaddedMatrixAData[rsp+16]
        vmovdqu xmm6,XMMWORD PTR GemmU8S8CopyPackAFrame.PaddedMatrixAData[rsp+32]
        vmovdqu xmm7,XMMWORD PTR GemmU8S8CopyPackAFrame.PaddedMatrixAData[rsp+48]
        lea     rax,[rcx+r11*2]             ; compute matrix D plus 2 rows
        vpmaskmovd XMMWORD PTR [rcx],xmm10,xmm4
        vpmaskmovd XMMWORD PTR [rcx+r11],xmm10,xmm5
        vpmaskmovd XMMWORD PTR [rax],xmm10,xmm6
        vpmaskmovd XMMWORD PTR [rax+r11],xmm10,xmm7
        vpmaddubsw xmm4,xmm4,xmm9           ; horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              ; add words to row accumulators
        vpmaddubsw xmm5,xmm5,xmm9
        vpaddw  ymm1,ymm1,ymm5
        vpmaddubsw xmm6,xmm6,xmm9
        vpaddw  ymm2,ymm2,ymm6
        vpmaddubsw xmm7,xmm7,xmm9
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
        add     rdi,r11
        mov     rbx,r10                     ; reload columns remaining
        sub     rbx,32
        jb      ProcessRemainingColumnsM1

ProcessNextColumnLoopM1:
        vmovdqu ymm4,YMMWORD PTR [rdx]
        vmovdqu YMMWORD PTR [rcx],ymm4
        vpmaddubsw ymm4,ymm4,ymm9           ; horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              ; add words to row accumulators
        add     rdx,32                      ; advance matrix A by 32 bytes
        add     rcx,32                      ; advance matrix D by 32 bytes
        sub     rbx,32                      ; subtract columns remaining
        jae     ProcessNextColumnLoopM1

ProcessRemainingColumnsM1:
        add     rbx,32                      ; correct for over-subtract above
        jz      ReduceRowSumBufferM1
        test    bl,16                       ; (CountK & 16) != 0?
        jz      CopyRemainingCountKLessThan16M1
        vmovdqu xmm4,XMMWORD PTR [rdx]
        vmovdqu XMMWORD PTR [rcx],xmm4
        vpmaddubsw xmm4,xmm4,xmm9           ; horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              ; add words to row accumulators
        add     rdx,16                      ; advance matrix A by 16 bytes
        add     rcx,16                      ; advance matrix D by 16 bytes
        test    bl,15                       ; test for unaligned columns
        jz      ReduceRowSumBufferM1

;
; Copy the unaligned CountK columns to a zero padded stack buffer.
;

CopyRemainingCountKLessThan16M1:
.errnz  GemmU8S8CopyPackAFrame.PaddedMatrixAData
        mov     rbp,rsp                     ; GemmU8S8CopyPackAFrame.PaddedMatrixAData
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
        vmovdqu xmm4,XMMWORD PTR GemmU8S8CopyPackAFrame.PaddedMatrixAData[rsp]
        vpmaskmovd XMMWORD PTR [rcx],xmm10,xmm4
        vpmaddubsw ymm4,ymm4,ymm9           ; horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              ; add words to row accumulators

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
        movaps  xmm6,GemmU8S8CopyPackAFrame.SavedXmm6[rsp]
        movaps  xmm7,GemmU8S8CopyPackAFrame.SavedXmm7[rsp]
        movaps  xmm8,GemmU8S8CopyPackAFrame.SavedXmm8[rsp]
        movaps  xmm9,GemmU8S8CopyPackAFrame.SavedXmm9[rsp]
        movaps  xmm10,GemmU8S8CopyPackAFrame.SavedXmm10[rsp]
        add     rsp,(GemmU8S8CopyPackAFrame.SavedR13)

        BEGIN_EPILOGUE

        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasGemmU8S8CopyPackAAvx2, _TEXT

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
;   BIsSigned - Supplies true if the source matrix is signed data, else false
;       if the source matrix is unsigned data.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasGemmU8S8CopyPackBAvx2, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        alloc_stack (GemmU8S8CopyPackBFrame.SavedRdi)
        save_xmm128 xmm6,GemmU8S8CopyPackBFrame.SavedXmm6
        save_xmm128 xmm7,GemmU8S8CopyPackBFrame.SavedXmm7
        save_xmm128 xmm8,GemmU8S8CopyPackBFrame.SavedXmm8
        save_xmm128 xmm9,GemmU8S8CopyPackBFrame.SavedXmm9

        END_PROLOGUE

        mov     rsi,rdx
        lea     rdi,[r8+r8*2]               ; compute ldb * 3
        mov     r10,GemmU8S8CopyPackBFrame.CountK[rsp]
        mov     r11,GemmU8S8CopyPackBFrame.ColumnSumBuffer[rsp]
        vpcmpeqw ymm7,ymm7,ymm7             ; generate word vector [0xFFFF]
        vpsrlw  ymm7,ymm7,15                ; generate word vector [0x0001]
        vpsllw  ymm8,ymm7,8                 ; generate word vector [0x0100]
        vpor    ymm8,ymm7,ymm8              ; generate word vector [0x0101]

;
; Compute the bit flip vector to adjust input from U8 to S8.
;

        vpxor   xmm9,xmm9,xmm9              ; generate word vector [0x0000]
        cmp     BYTE PTR GemmU8S8CopyPackBFrame.BIsSigned[rsp],0
        jnz     SkipUnsignedBitFlipVector
        vpsllw  ymm9,ymm8,7                 ; generate word vector [0x8080]

SkipUnsignedBitFlipVector:

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
        sub     rbx,4
        jb      ProcessRemainingRowsN16

ProcessNextRowLoopN16:
        vmovdqu xmm2,XMMWORD PTR [rdx]      ; load 4 rows
        vmovdqu xmm3,XMMWORD PTR [rdx+r8]
        vmovdqu xmm4,XMMWORD PTR [rdx+r8*2]
        vmovdqu xmm5,XMMWORD PTR [rdx+rdi]
        lea     rdx,[rdx+r8*4]              ; advance matrix B by 4 rows

InterleaveRowDataN16:
        vpunpcklbw xmm6,xmm2,xmm3           ; interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vpunpcklbw xmm2,xmm4,xmm5
        vpunpckhbw xmm5,xmm4,xmm5
        vpunpcklwd xmm4,xmm6,xmm2
        vpunpckhwd xmm6,xmm6,xmm2
        vpunpcklwd xmm2,xmm3,xmm5
        vpunpckhwd xmm3,xmm3,xmm5
        vinserti128 ymm4,ymm4,xmm6,1
        vinserti128 ymm2,ymm2,xmm3,1
        vpxor   ymm4,ymm4,ymm9              ; optionally adjust unsigned data
        vpxor   ymm2,ymm2,ymm9
        vmovdqu YMMWORD PTR [rcx],ymm4      ; store interleaved rows
        vmovdqu YMMWORD PTR [rcx+32],ymm2
        vpmaddubsw ymm4,ymm8,ymm4           ; horizontal byte+byte=word per row
        vpmaddwd ymm4,ymm4,ymm7             ; horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              ; accumulate per column
        vpmaddubsw ymm2,ymm8,ymm2
        vpmaddwd ymm2,ymm2,ymm7
        vpaddd  ymm1,ymm1,ymm2
        add     rcx,64                      ; advance matrix D by 64 bytes
        sub     rbx,4                       ; subtract rows remaining
        jae     ProcessNextRowLoopN16

;
; Process the less than 4 remaining rows where the row has 16 columns.
;

ProcessRemainingRowsN16:
        add     rbx,4                       ; correct for over-subtract above
        jz      StoreColumnSumBufferN16
        vmovdqu xmm2,XMMWORD PTR [rdx]
        vmovaps xmm3,xmm9
        vmovaps xmm4,xmm9
        vmovaps xmm5,xmm9
        xor     ebx,ebx                     ; no more rows remaining
        test    r10b,2                      ; (CountK & 2) != 0?
        jz      InterleaveRowDataN16
        vmovdqu xmm3,XMMWORD PTR [rdx+r8]
        test    r10b,1                      ; (CountK & 1) != 0?
        jz      InterleaveRowDataN16
        vmovdqu xmm4,XMMWORD PTR [rdx+r8*2]
        jmp     InterleaveRowDataN16

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
        movaps  xmm6,GemmU8S8CopyPackBFrame.SavedXmm6[rsp]
        movaps  xmm7,GemmU8S8CopyPackBFrame.SavedXmm7[rsp]
        movaps  xmm8,GemmU8S8CopyPackBFrame.SavedXmm8[rsp]
        movaps  xmm9,GemmU8S8CopyPackBFrame.SavedXmm9[rsp]
        add     rsp,(GemmU8S8CopyPackBFrame.SavedRdi)

        BEGIN_EPILOGUE

        pop     rdi
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
        vmovdqu YMMWORD PTR GemmU8S8CopyPackBFrame.PaddedMatrixBData[rsp],ymm9
        vmovdqu YMMWORD PTR GemmU8S8CopyPackBFrame.PaddedMatrixBData[rsp+32],ymm9
        sub     r10,4
        jb      ProcessRemainingRowsNUnaligned

ProcessNextRowLoopNUnaligned:
        mov     rdx,rsi
.errnz  GemmU8S8CopyPackBFrame.PaddedMatrixBData
        mov     rbp,rsp                     ; GemmU8S8CopyPackBFrame.PaddedMatrixBData
        test    r9b,8                       ; (CountN & 8) != 0?
        jz      CopyRemainingCountNLessThan8K4
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        mov     rax,QWORD PTR [rdx+r8]
        mov     QWORD PTR [rbp+16],rax
        mov     rax,QWORD PTR [rdx+r8*2]
        mov     QWORD PTR [rbp+32],rax
        mov     rax,QWORD PTR [rdx+rdi]
        mov     QWORD PTR [rbp+48],rax
        add     rdx,8                       ; advance matrix B
        add     rbp,8                       ; advance padded buffer destination

CopyRemainingCountNLessThan8K4:
        test    r9b,4                       ; (CountN & 4) != 0?
        jz      CopyRemainingCountNLessThan4K4
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        mov     eax,DWORD PTR [rdx+r8]
        mov     DWORD PTR [rbp+16],eax
        mov     eax,DWORD PTR [rdx+r8*2]
        mov     DWORD PTR [rbp+32],eax
        mov     eax,DWORD PTR [rdx+rdi]
        mov     DWORD PTR [rbp+48],eax
        add     rdx,4                       ; advance matrix B
        add     rbp,4                       ; advance padded buffer destination

CopyRemainingCountNLessThan4K4:
        test    r9b,2                       ; (CountN & 2) != 0?
        jz      CopyRemainingCountNLessThan2K4
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        movzx   eax,WORD PTR [rdx+r8]
        mov     WORD PTR [rbp+16],ax
        movzx   eax,WORD PTR [rdx+r8*2]
        mov     WORD PTR [rbp+32],ax
        movzx   eax,WORD PTR [rdx+rdi]
        mov     WORD PTR [rbp+48],ax
        add     rdx,2                       ; advance matrix B
        add     rbp,2                       ; advance padded buffer destination

CopyRemainingCountNLessThan2K4:
        test    r9b,1                       ; (CountN & 1) != 0?
        jz      ProcessPaddedMatrixBData
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al
        movzx   eax,BYTE PTR [rdx+r8]
        mov     BYTE PTR [rbp+16],al
        movzx   eax,BYTE PTR [rdx+r8*2]
        mov     BYTE PTR [rbp+32],al
        movzx   eax,BYTE PTR [rdx+rdi]
        mov     BYTE PTR [rbp+48],al

ProcessPaddedMatrixBData:
        vmovdqu xmm2,XMMWORD PTR GemmU8S8CopyPackBFrame.PaddedMatrixBData[rsp]
        vmovdqu xmm3,XMMWORD PTR GemmU8S8CopyPackBFrame.PaddedMatrixBData[rsp+16]
        vmovdqu xmm4,XMMWORD PTR GemmU8S8CopyPackBFrame.PaddedMatrixBData[rsp+32]
        vmovdqu xmm5,XMMWORD PTR GemmU8S8CopyPackBFrame.PaddedMatrixBData[rsp+48]
        vpunpcklbw xmm6,xmm2,xmm3           ; interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vpunpcklbw xmm2,xmm4,xmm5
        vpunpckhbw xmm5,xmm4,xmm5
        vpunpcklwd xmm4,xmm6,xmm2
        vpunpckhwd xmm6,xmm6,xmm2
        vpunpcklwd xmm2,xmm3,xmm5
        vpunpckhwd xmm3,xmm3,xmm5
        vinserti128 ymm4,ymm4,xmm6,1
        vinserti128 ymm2,ymm2,xmm3,1
        vpxor   ymm4,ymm4,ymm9              ; optionally adjust unsigned data
        vpxor   ymm2,ymm2,ymm9
        vmovdqu YMMWORD PTR [rcx],ymm4      ; store interleaved rows
        vmovdqu YMMWORD PTR [rcx+32],ymm2
        vpmaddubsw ymm4,ymm8,ymm4           ; horizontal byte+byte=word per row
        vpmaddwd ymm4,ymm4,ymm7             ; horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              ; accumulate per column
        vpmaddubsw ymm2,ymm8,ymm2
        vpmaddwd ymm2,ymm2,ymm7
        vpaddd  ymm1,ymm1,ymm2
        lea     rsi,[rsi+r8*4]              ; advance next matrix B by 4 rows
        add     rcx,64                      ; advance matrix D by 64 bytes
        sub     r10,4                       ; subtract rows remaining
        jae     ProcessNextRowLoopNUnaligned

ProcessRemainingRowsNUnaligned:
        add     r10,4
        jz      StoreColumnSumBufferNUnaligned

;
; Process the less than 4 remaining rows where the row has less than 16 columns.
;

.errnz  GemmU8S8CopyPackBFrame.PaddedMatrixBData
        mov     rbp,rsp                     ; GemmU8S8CopyPackBFrame.PaddedMatrixBData
        vmovdqu YMMWORD PTR [rbp],ymm9
        vmovdqu YMMWORD PTR [rbp+32],ymm9

CopyUnalignedRowLoop:
        lea     rdi,[rbp+16]                ; advance next padded buffer by 16 bytes
        mov     rdx,rsi
        test    r9b,8                       ; (CountN & 8) != 0?
        jz      CopyRemainingCountNLessThan8KSmall
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        add     rdx,8                       ; advance matrix B
        add     rbp,8                       ; advance padded buffer destination

CopyRemainingCountNLessThan8KSmall:
        test    r9b,4                       ; (CountN & 4) != 0?
        jz      CopyRemainingCountNLessThan4KSmall
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        add     rdx,4                       ; advance matrix B
        add     rbp,4                       ; advance padded buffer destination

CopyRemainingCountNLessThan4KSmall:
        test    r9b,2                       ; (CountN & 2) != 0?
        jz      CopyRemainingCountNLessThan2KSmall
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        add     rdx,2                       ; advance matrix B
        add     rbp,2                       ; advance padded buffer destination

CopyRemainingCountNLessThan2KSmall:
        test    r9b,1                       ; (CountN & 1) != 0?
        jz      DoneCopyRemainingCountNKSmall
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al

DoneCopyRemainingCountNKSmall:
        dec     r10
        jz      ProcessPaddedMatrixBData
        add     rsi,r8                      ; advance next matrix B by 1 row
        mov     rbp,rdi
        jmp     CopyUnalignedRowLoop

StoreColumnSumBufferNUnaligned:
        vmovdqu YMMWORD PTR [r11],ymm0
        vmovdqu YMMWORD PTR [r11+32],ymm1
        jmp     ExitRoutine

        NESTED_END MlasGemmU8S8CopyPackBAvx2, _TEXT

        END
