;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   QgemvU8S8KernelAvx2.asm
;
; Abstract:
;
;   This module implements the kernels for the quantized integer matrix/vector
;   multiply operation (QGEMV).
;
;   This implementation uses AVX2 instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        EXTERN  MlasMaskMoveAvx:NEAR
        EXTERN  MlasTranspose4x4BytesAvx:NEAR

;
; Stack frame layout for the U8S8 kernel.
;

GemvU8S8KernelFrame STRUCT

        SavedXmm6 OWORD ?
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
        CountN QWORD ?
        ldb QWORD ?

GemvU8S8KernelFrame ENDS

;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix/vector multiplication.
;
; Arguments:
;
;   A (rcx) - Supplies the address of vector A.
;
;   B (rdx) - Supplies the address of matrix B.
;
;   C (r8) - Supplies the address of matrix C.
;
;   CountK (r9) - Supplies the number of columns from vector A and the number
;       of rows from matrix B to iterate over.
;
;   CountN - Supplies the number of columns from matrix B and matrix C to iterate
;       over.
;
;   ldb - Supplies the first dimension of matrix B.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasGemvU8S8KernelAvx2, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        alloc_stack (GemvU8S8KernelFrame.SavedRdi)
        save_xmm128 xmm6,GemvU8S8KernelFrame.SavedXmm6

        END_PROLOGUE

        mov     rsi,rdx
        mov     rdi,GemvU8S8KernelFrame.ldb[rsp]
        mov     r10,GemvU8S8KernelFrame.CountN[rsp]
        mov     r11,rsp                     ; set ZeroMode to any non-zero value
        vpcmpeqw ymm6,ymm6,ymm6             ; generate word vector [0xFFFF]
        vpsrlw  ymm6,ymm6,15                ; generate word vector [0x0001]

;
; Process 4 rows of matrix B in a loop.
;

        sub     r9,4
        jb      ProcessRemainingRows

ProcessRowLoop4:
        mov     rdx,rsi                     ; reload matrix B
        lea     rsi,[rsi+rdi*4]             ; advance matrix B by 4 rows
        mov     rbx,r8                      ; reload matrix C
        mov     rbp,r10                     ; reload CountN
        vpbroadcastd ymm0,DWORD PTR [rcx]
        add     rcx,4                       ; advance matrix A by 4 bytes

;
; Process sets of 32 columns from the 4 rows in a loop.
;
; Some permute operations are deferred until the final store of the 4x32 block
; as these permutes are expensive.
;

ProcessColumnLoop4By32:
        cmp     rbp,32
        jb      ProcessColumnLoop4By8
        lea     rax,[rdx+rdi*2]             ; compute matrix B plus 2 rows
        vmovdqu ymm2,YMMWORD PTR [rdx]
        vmovdqu ymm3,YMMWORD PTR [rdx+rdi]
        vmovdqu ymm4,YMMWORD PTR [rax]
        vmovdqu ymm5,YMMWORD PTR [rax+rdi]
        vpunpcklbw ymm1,ymm2,ymm3           ; interleave row data bytes
        vpunpckhbw ymm2,ymm2,ymm3
        vpunpcklbw ymm3,ymm4,ymm5
        vpunpckhbw ymm4,ymm4,ymm5
        vpunpcklwd ymm5,ymm1,ymm3           ; interleave row data words
        vpunpckhwd ymm1,ymm1,ymm3
        vpunpcklwd ymm3,ymm2,ymm4
        vpunpckhwd ymm2,ymm2,ymm4
        vpmaddubsw ymm5,ymm0,ymm5           ; multiply and reduce
        vpmaddwd ymm5,ymm5,ymm6
        vpmaddubsw ymm1,ymm0,ymm1
        vpmaddwd ymm1,ymm1,ymm6
        vpmaddubsw ymm3,ymm0,ymm3
        vpmaddwd ymm3,ymm3,ymm6
        vpmaddubsw ymm2,ymm0,ymm2
        vpmaddwd ymm2,ymm2,ymm6
        test    r11,r11                     ; ZeroMode?
        jnz     SkipAccumulateOutput4By32
        vpaddd  ymm5,ymm5,YMMWORD PTR [rbx]
        vpaddd  ymm1,ymm1,YMMWORD PTR [rbx+32]
        vpaddd  ymm3,ymm3,YMMWORD PTR [rbx+64]
        vpaddd  ymm2,ymm2,YMMWORD PTR [rbx+96]

SkipAccumulateOutput4By32:
        cmp     r9,4                        ; final 4x32 block?
        jae     StoreOutput4By32
        vperm2i128 ymm4,ymm5,ymm1,31h       ; interleave vector results
        vperm2i128 ymm5,ymm5,ymm1,20h
        vperm2i128 ymm1,ymm3,ymm2,20h
        vperm2i128 ymm2,ymm3,ymm2,31h
        vmovaps ymm3,ymm4

StoreOutput4By32:
        vmovdqu YMMWORD PTR [rbx],ymm5
        vmovdqu YMMWORD PTR [rbx+32],ymm1
        vmovdqu YMMWORD PTR [rbx+64],ymm3
        vmovdqu YMMWORD PTR [rbx+96],ymm2
        add     rdx,32                      ; advance matrix B by 32 bytes
        add     rbx,32*4                    ; advance matrix C by 32 columns
        sub     rbp,32                      ; decrement CountN
        jnz     ProcessColumnLoop4By32

AdvanceRowLoop4:
        xor     r11,r11                     ; clear ZeroMode
        sub     r9,4                        ; decrement CountK
        jae     ProcessRowLoop4

ProcessRemainingRows:
        add     r9,4                        ; correct for over-subtract above
        jnz     ProcessRemainingSmallK

;
; Restore non-volatile registers and return.
;

ExitKernel:
        vzeroupper
        movaps  xmm6,GemvU8S8KernelFrame.SavedXmm6[rsp]
        add     rsp,(GemvU8S8KernelFrame.SavedRdi)

        BEGIN_EPILOGUE

        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

;
; Process sets of 8 columns from the 4 rows in a loop.
;

ProcessColumnLoop4By8:
        cmp     ebp,8
        jb      ProcessColumn4By4
        lea     rax,[rdx+rdi*2]             ; compute matrix B plus 2 rows
        vmovq   xmm2,QWORD PTR [rdx]
        vmovq   xmm3,QWORD PTR [rdx+rdi]
        vmovq   xmm4,QWORD PTR [rax]
        vmovq   xmm5,QWORD PTR [rax+rdi]
        vpunpcklbw xmm2,xmm2,xmm3           ; interleave row data bytes
        vpunpcklbw xmm4,xmm4,xmm5
        vpunpcklwd xmm1,xmm2,xmm4           ; interleave row data words
        vpunpckhwd xmm2,xmm2,xmm4
        vinserti128 ymm1,ymm1,xmm2,1        ; concatenate vector
        vpmaddubsw ymm1,ymm0,ymm1           ; multiply and reduce
        vpmaddwd ymm1,ymm1,ymm6
        test    r11,r11                     ; ZeroMode?
        jnz     SkipAccumulateOutput4By8
        vpaddd  ymm1,ymm1,YMMWORD PTR [rbx]

SkipAccumulateOutput4By8:
        vmovdqu YMMWORD PTR [rbx],ymm1
        add     rdx,8                       ; advance matrix B by 8 bytes
        add     rbx,8*4                     ; advance matrix C by 8 columns
        sub     ebp,8                       ; decrement CountN
        jnz     ProcessColumnLoop4By8
        jmp     AdvanceRowLoop4

;
; Process a set of 4 columns from the 4 rows.
;

ProcessColumn4By4:
        test    ebp,4                       ; (CountN & 4) != 0?
        jz      ProcessColumn4BySmallN
        lea     rax,[rdx+rdi*2]             ; compute matrix B plus 2 rows
        vmovd   xmm1,DWORD PTR [rdx]
        vpinsrd xmm1,xmm1,DWORD PTR [rdx+rdi],1
        vpinsrd xmm1,xmm1,DWORD PTR [rax],2
        vpinsrd xmm1,xmm1,DWORD PTR [rax+rdi],3
        vpshufb xmm1,xmm1,XMMWORD PTR [MlasTranspose4x4BytesAvx]
        vpmaddubsw xmm1,xmm0,xmm1           ; multiply and reduce
        vpmaddwd xmm1,xmm1,xmm6
        test    r11,r11                     ; ZeroMode?
        jnz     SkipAccumulateOutput4By4
        vpaddd  xmm1,xmm1,XMMWORD PTR [rbx]

SkipAccumulateOutput4By4:
        vmovdqu XMMWORD PTR [rbx],xmm1
        and     ebp,3                       ; (CountN & 3) != 0?
        jz      AdvanceRowLoop4
        add     rdx,4                       ; advance matrix B by 4 bytes
        add     rbx,4*4                     ; advance matrix C by 4 columns

;
; Process the remaining 1 to 3 columns from the 4 rows.
;

ProcessColumn4BySmallN:
        mov     DWORD PTR GemvU8S8KernelFrame.CountN[rsp],ebp
        vbroadcastss xmm2,DWORD PTR GemvU8S8KernelFrame.CountN[rsp]
        vpcmpgtd xmm2,xmm2,XMMWORD PTR [MlasMaskMoveAvx]
        vpxor   xmm1,xmm1,xmm1
        lea     rax,[rdx+rdi*2]             ; compute matrix B plus 2 rows
        cmp     ebp,2                       ; (CountN & 2) != 0?
        jb      ProcessColumn4By1
        vpinsrw xmm1,xmm1,WORD PTR [rdx],0
        vpinsrw xmm1,xmm1,WORD PTR [rdx+rdi],2
        vpinsrw xmm1,xmm1,WORD PTR [rax],4
        vpinsrw xmm1,xmm1,WORD PTR [rax+rdi],6
        je      ComputeOutput4BySmallN
        vpinsrb xmm1,xmm1,BYTE PTR [rdx+2],2
        vpinsrb xmm1,xmm1,BYTE PTR [rdx+rdi+2],6
        vpinsrb xmm1,xmm1,BYTE PTR [rax+2],10
        vpinsrb xmm1,xmm1,BYTE PTR [rax+rdi+2],14
        jmp     ComputeOutput4BySmallN

ProcessColumn4By1:
        vpinsrb xmm1,xmm1,BYTE PTR [rdx],0
        vpinsrb xmm1,xmm1,BYTE PTR [rdx+rdi],4
        vpinsrb xmm1,xmm1,BYTE PTR [rax],8
        vpinsrb xmm1,xmm1,BYTE PTR [rax+rdi],12

ComputeOutput4BySmallN:
        vpshufb xmm1,xmm1,XMMWORD PTR [MlasTranspose4x4BytesAvx]
        vpmaddubsw xmm1,xmm0,xmm1           ; multiply and reduce
        vpmaddwd xmm1,xmm1,xmm6
        test    r11,r11                     ; ZeroMode?
        jnz     StoreOutput4BySmallN
        vpmaskmovd xmm3,xmm2,XMMWORD PTR [rbx]
        vpaddd  xmm1,xmm1,xmm3

StoreOutput4BySmallN:
        vpmaskmovd XMMWORD PTR [rbx],xmm2,xmm1
        jmp     AdvanceRowLoop4

;
; Broadcast the remaining 1 to 3 values from vector A.
;

ProcessRemainingSmallK:
        vpxor   xmm5,xmm5,xmm5              ; keep zero vector for vpinsrb/vpinsrw
        cmp     r9d,2
        jb      LoadVectorASingleRemainingByte
        vpinsrw xmm0,xmm5,WORD PTR [rcx],0
        je      BroadcastVectorARemainingBytes
        vpinsrb xmm0,xmm0,BYTE PTR [rcx+2],2
        jmp     BroadcastVectorARemainingBytes

LoadVectorASingleRemainingByte:
        vpinsrb xmm0,xmm5,BYTE PTR [rcx],0

BroadcastVectorARemainingBytes:
        vpshufd xmm0,xmm0,0                 ; broadcast values

;
; Process a set of 4 columns from the remaining rows.
;

ProcessColumnLoopSmallKBy4:
        cmp     r10,4
        jb      ProcessColumnLoopSmallKBySmallN
        vmovd   xmm1,DWORD PTR [rsi]
        cmp     r9d,2
        jb      ComputeOutputSmallKBy4
        vpinsrd xmm1,xmm1,DWORD PTR [rsi+rdi],1
        je      ComputeOutputSmallKBy4
        vpinsrd xmm1,xmm1,DWORD PTR [rsi+rdi*2],2

ComputeOutputSmallKBy4:
        vpshufb xmm1,xmm1,XMMWORD PTR [MlasTranspose4x4BytesAvx]
        vpmaddubsw xmm1,xmm0,xmm1           ; multiply and reduce
        vpmaddwd xmm1,xmm1,xmm6
        test    r11,r11                     ; ZeroMode?
        jnz     SkipAccumulateOutputSmallKBy4
        vpaddd  xmm1,xmm1,XMMWORD PTR [r8]

SkipAccumulateOutputSmallKBy4:
        vmovdqu XMMWORD PTR [r8],xmm1
        add     rsi,4                       ; advance matrix B by 4 bytes
        add     r8,4*4                      ; advance matrix C by 4 columns
        sub     r10,4                       ; decrement CountN
        jnz     ProcessColumnLoopSmallKBy4
        jmp     ExitKernel

;
; Process the remaining 1 to 3 columns from the remaining rows.
;
; Single step through each of the columns to keep code size small for the
; uncommon path (typically the row count is a multiple of 4).
;

ProcessColumnLoopSmallKBySmallN:
        vpinsrb xmm1,xmm5,BYTE PTR [rsi],0
        cmp     r9d,2
        jb      ComputeOutputSmallKBySmallN
        vpinsrb xmm1,xmm1,BYTE PTR [rsi+rdi],1
        je      ComputeOutputSmallKBySmallN
        vpinsrb xmm1,xmm1,BYTE PTR [rsi+rdi*2],2

ComputeOutputSmallKBySmallN:
        vpmaddubsw xmm1,xmm0,xmm1           ; multiply and reduce
        vpmaddwd xmm1,xmm1,xmm6
        test    r11,r11                     ; ZeroMode?
        jnz     SkipAccumulateOutputSmallKBySmallN
        vmovd   xmm3,DWORD PTR [r8]
        vpaddd  xmm1,xmm1,xmm3

SkipAccumulateOutputSmallKBySmallN:
        vmovd   DWORD PTR [r8],xmm1
        inc     rsi                         ; advance matrix B by 1 byte
        add     r8,4                        ; advance matrix C by 1 column
        dec     r10
        jnz     ProcessColumnLoopSmallKBySmallN
        jmp     ExitKernel

        NESTED_END MlasGemvU8S8KernelAvx2, _TEXT

        END
