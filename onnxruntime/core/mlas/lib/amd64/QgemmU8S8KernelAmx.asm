;++
;
;Copyright (c) Microsoft Corporation. All rights reserved.
;
;Licensed under the MIT License.
;
;Module Name:
;
;    QgemmU8S8KernelAmx.asm
;
;Abstract:
;
;    This module implements the packing functions for the quantized integer matrix/matrix
;    multiply operation (QGEMM).
;
;    These packing functions are suited for AMX Qgemm kernel. The implementation only
;    uses AVX2 instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list


;
; Stack frame layout for the U8S8 CopyPackB routine.
;

GemmU8S8CopyPackBFrame STRUCT
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

        NESTED_ENTRY MlasGemmU8S8CopyPackBAmx, _TEXT
        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        END_PROLOGUE

        mov     rsi,rdx                     ; Save B
        lea     rdi,[r8+r8*2]               ; compute ldb * 3
        mov     r10,GemmU8S8CopyPackBFrame.CountK[rsp]
        mov     r11,GemmU8S8CopyPackBFrame.ColumnSumBuffer[rsp]
        lea     r12,[r10+3]                 ; compute extra padding for 64|K
        shr     r12,2
        neg     r12
        and     r12,15
        vpcmpeqw ymm0,ymm0,ymm0             ; generate word vector [0xFFFF]
        vpsrlw  ymm0,ymm0,15                ; generate word vector [0x0001]
        vpsllw  ymm1,ymm0,8                 ; generate word vector [0x0100]
        vpor    ymm1,ymm0,ymm1              ; generate word vector [0x0101]

;
; Compute the bit flip vector to adjust input from U8 to S8.
;

        vpxor   xmm2,xmm2,xmm2              ; generate word vector [0x0000]
        cmp     BYTE PTR GemmU8S8CopyPackBFrame.BIsSigned[rsp],0
        jnz     CopyPackB_SkipUnsignedBitFlipVector
        vpsllw  ymm2,ymm1,7                 ; generate word vector [0x8080]

CopyPackB_SkipUnsignedBitFlipVector:

;
; Process 16 columns of matrix B in a loop.
;

        sub     r9,16                       ; CountN -= 16
        jb      CopyPackB_ProcessRemainingColumns

CopyPackB_ProcessNextColumnN16:
        vpxord  xmm30,xmm30,xmm30           ; clear column accumulators
        vpxord  xmm31,xmm31,xmm31
        mov     rdx,rsi                     ; rdx -> B start of 16 columns
        add     rsi,16                      ; advance next matrix B by 16 columns
        mov     rbx,r10                     ; reload rows remaining
        sub     rbx,4
        jb      CopyPackB_ProcessRemainingRowsN16

CopyPackB_ProcessNextRowLoopN16:
        vmovdqu64 xmm16,XMMWORD PTR [rdx]   ; load 4 rows
        vmovdqu64 xmm17,XMMWORD PTR [rdx+r8]
        vmovdqu64 xmm18,XMMWORD PTR [rdx+r8*2]
        vmovdqu64 xmm19,XMMWORD PTR [rdx+rdi]
        lea     rdx,[rdx+r8*4]              ; advance matrix B by 4 rows

CopyPackB_InterleaveRowDataN16:
        vpunpcklbw xmm3,xmm16,xmm17         ; interleave row data
        vpunpckhbw xmm17,xmm16,xmm17
        vpunpcklbw xmm16,xmm18,xmm19
        vpunpckhbw xmm19,xmm18,xmm19
        vpunpcklwd xmm18,xmm3,xmm16
        vpunpckhwd xmm3,xmm3,xmm16
        vpunpcklwd xmm16,xmm17,xmm19
        vpunpckhwd xmm17,xmm17,xmm19
        vinserti64x2 ymm18,ymm18,xmm3,1
        vinserti64x2 ymm16,ymm16,xmm17,1
        vpxord  ymm18,ymm18,ymm2            ; optionally adjust unsigned data
        vpxord  ymm16,ymm16,ymm2
        vmovdqu64 YMMWORD PTR [rcx],ymm18   ; store interleaved rows
        vmovdqu64 YMMWORD PTR [rcx+32],ymm16
        vpmaddubsw ymm18,ymm1,ymm18         ; horizontal byte+byte=word per row
        vpmaddwd ymm18,ymm18,ymm0           ; horizontal word+word=dword per row
        vpaddd  ymm30,ymm30,ymm18           ; accumulate per column
        vpmaddubsw ymm16,ymm1,ymm16
        vpmaddwd ymm16,ymm16,ymm0
        vpaddd  ymm31,ymm31,ymm16
        add     rcx,64                      ; advance matrix D by 64 bytes
        sub     rbx,4                       ; subtract rows remaining
        jae     CopyPackB_ProcessNextRowLoopN16

;
; Process the less than 4 remaining rows where the row has 16 columns.
;

CopyPackB_ProcessRemainingRowsN16:
        add     rbx,4                       ; correct for over-subtract above
        jz      CopyPackB_StoreColumnSumBufferN16
        vmovdqu64 xmm16,XMMWORD PTR [rdx]
        vmovaps xmm17,xmm2
        vmovaps xmm18,xmm2
        vmovaps xmm19,xmm2
        xor     ebx,ebx                     ; no more rows remaining
        test    r10b,2                      ; (CountK & 2) != 0?
        jz      CopyPackB_InterleaveRowDataN16
        vmovdqu64 xmm17,XMMWORD PTR [rdx+r8]
        test    r10b,1                      ; (CountK & 1) != 0?
        jz      CopyPackB_InterleaveRowDataN16
        vmovdqu64 xmm18,XMMWORD PTR [rdx+r8*2]
        jmp     CopyPackB_InterleaveRowDataN16

CopyPackB_StoreColumnSumBufferN16:
        vmovdqu64 YMMWORD PTR [r11],ymm30
        vmovdqu64 YMMWORD PTR [r11+32],ymm31
        test    r12,r12
        jz      CopyPackB_N16K64PaddingFinished
        mov     rax, r12
        vpxord  xmm30,xmm30,xmm30

CopyPackB_N16K64Padding:
        vmovdqu64 YMMWORD PTR [rcx],ymm30   ; store 0
        vmovdqu64 YMMWORD PTR [rcx+32],ymm30
        add     rcx,64
        dec     rax
        jnz     CopyPackB_N16K64Padding

CopyPackB_N16K64PaddingFinished:
        add     r11,16*4                    ; advance column sum buffer by 16 dwords
        sub     r9,16                       ; subtract columns remaining
        jae     CopyPackB_ProcessNextColumnN16

CopyPackB_ProcessRemainingColumns:
        add     r9,16                       ; correct for over-subtract above
        jnz     CopyPackB_ProcessColumnNUnaligned

;
; Restore non-volatile registers and return.
;

CopyPackB_ExitRoutine:
        vzeroupper

        BEGIN_EPILOGUE
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

;
; Process the remaining columns of matrix B.
;

CopyPackB_ProcessColumnNUnaligned:
        vpxord  xmm30,xmm30,xmm30           ; clear column accumulators
        vpxord  xmm31,xmm31,xmm31
        mov     rax,rcx                     ; save rcx (D)
        mov     rcx,r9                      ; load left over N
        neg     ecx                         ; compute load mask for left over N
        and     ecx,63
        mov     rbx,-1
        shr     rbx,cl
        kmovq   k1,rbx
        mov     rcx,rax                     ; restore rcx (D)
        sub     r10,4
        jb      CopyPackB_ProcessRemainingRowsNUnaligned

CopyPackB_ProcessNextRowLoopNUnaligned:
        vmovdqu64 xmm16,xmm2
        vmovdqu8 xmm16 {k1},XMMWORD PTR [rsi]
        vmovdqu64 xmm17,xmm2
        vmovdqu8 xmm17 {k1},XMMWORD PTR [rsi+r8]
        vmovdqu64 xmm18,xmm2
        vmovdqu8 xmm18 {k1},XMMWORD PTR [rsi+r8*2]
        vmovdqu64 xmm19,xmm2
        vmovdqu8 xmm19 {k1},XMMWORD PTR [rsi+rdi]
        lea     rsi,[rsi+r8*4]              ; advance next matrix B by 4 rows

CopyPackB_InterleaveRowDataUnaligned:
        vpunpcklbw xmm3,xmm16,xmm17         ; interleave row data
        vpunpckhbw xmm17,xmm16,xmm17
        vpunpcklbw xmm16,xmm18,xmm19
        vpunpckhbw xmm19,xmm18,xmm19
        vpunpcklwd xmm18,xmm3,xmm16
        vpunpckhwd xmm3,xmm3,xmm16
        vpunpcklwd xmm16,xmm17,xmm19
        vpunpckhwd xmm17,xmm17,xmm19
        vinserti64x2 ymm18,ymm18,xmm3,1
        vinserti64x2 ymm16,ymm16,xmm17,1
        vpxord  ymm18,ymm18,ymm2            ; optionally adjust unsigned data
        vpxord  ymm16,ymm16,ymm2
        vmovdqu64 YMMWORD PTR [rcx],ymm18   ; store interleaved rows
        vmovdqu64 YMMWORD PTR [rcx+32],ymm16
        vpmaddubsw ymm18,ymm1,ymm18         ; horizontal byte+byte=word per row
        vpmaddwd ymm18,ymm18,ymm0           ; horizontal word+word=dword per row
        vpaddd  ymm30,ymm30,ymm18           ; accumulate per column
        vpmaddubsw ymm16,ymm1,ymm16
        vpmaddwd ymm16,ymm16,ymm0
        vpaddd  ymm31,ymm31,ymm16
        add     rcx,64                      ; advance matrix D by 64 bytes
        sub     r10,4                       ; subtract rows remaining
        jae     CopyPackB_ProcessNextRowLoopNUnaligned

;
; Process the less than 4 remaining rows where the row has less than 16 columns.
;

CopyPackB_ProcessRemainingRowsNUnaligned:
        add     r10,4
        jz      CopyPackB_StoreColumnSumBufferNUnaligned

        vmovaps xmm16,xmm2
        vmovdqu8 xmm16 {k1},XMMWORD PTR [rsi]
        vmovaps xmm17,xmm2
        vmovaps xmm18,xmm2
        vmovaps xmm19,xmm2
        mov     rbx,r10
        xor     r10b,r10b                   ; no more rows remaining
        test    bl,2                        ; (CountK & 2) != 0?
        jz      CopyPackB_InterleaveRowDataUnaligned
        vmovdqu8 xmm17 {k1},XMMWORD PTR [rsi+r8]
        test    bl,1                        ; (CountK & 1) != 0?
        jz      CopyPackB_InterleaveRowDataUnaligned
        vmovdqu8 xmm18 {k1},XMMWORD PTR [rsi+r8*2]
        jmp     CopyPackB_InterleaveRowDataUnaligned

CopyPackB_StoreColumnSumBufferNUnaligned:
        vmovdqu64 YMMWORD PTR [r11],ymm30
        vmovdqu64 YMMWORD PTR [r11+32],ymm31
        test    r12,r12
        jz      CopyPackB_ExitRoutine
        mov     rax, r12
        vpxord  xmm30,xmm30,xmm30

CopyPackB_K64Padding:
        vmovdqu64 YMMWORD PTR [rcx],ymm30   ; store 0
        vmovdqu64 YMMWORD PTR [rcx+32],ymm30
        add     rcx,64
        dec     rax
        jne     CopyPackB_K64Padding
        jmp     CopyPackB_ExitRoutine

        NESTED_END MlasGemmU8S8CopyPackBAmx, _TEXT


;
; Stack frame layout for the U8S8 CopyPackA routine.
;

GemmU8S8CopyPackAFrame STRUCT

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

        NESTED_ENTRY MlasGemmU8S8CopyPackAAmx, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        END_PROLOGUE

        mov     rdi,rcx                         ; save D
        mov     rsi,rdx                         ; save A
        mov     r10,GemmU8S8CopyPackAFrame.CountK[rsp]
        mov     r12,GemmU8S8CopyPackAFrame.RowSumBuffer[rsp]
        lea     r11,[r10+63]
        and     r11,NOT 63                      ; align CountK up to 64
        vpternlogd zmm30,zmm30,zmm30,255        ; generate word vector [0xFFFF]
        vpsrlw  zmm30,zmm30,15                  ; generate word vector [0x0001]
        vpsllw  zmm31,zmm30,8                   ; generate word vector [0x0100]
        vpord   zmm31,zmm30,zmm31               ; generate word vector [0x0101]
        lea     r13,[r8+r8*2]                   ; compute ldb * 3
        lea     rax,[r11+r11*2]                 ; compute AlignedCountK * 3
        mov     ecx,r10d                        ; CountK
        neg     ecx
        and     ecx,63
        mov     rbx,-1
        shr     rbx,cl                          ; mask for left over k < 64
        kmovq   k1,rbx                          ; mask

;
; Process 4 rows of matrix A in a loop.
;

        sub     r9,4                            ; m -= 4
        jb      CopyPackA_ProcessRemainingRows

CopyPackA_ProcessNextRowM4:
        vpxor   xmm0,xmm0,xmm0                  ; clear row accumulators
        vpxor   xmm1,xmm1,xmm1
        vpxor   xmm2,xmm2,xmm2
        vpxor   xmm3,xmm3,xmm3
        mov     rdx,rsi                         ; src = A row beginning
        mov     rcx,rdi                         ; dst = D row beginning
        lea     rsi,[rsi+r8*4]                  ; advance next matrix A by 4 rows
        lea     rdi,[rdi+r11*4]                 ; advance next matrix D by 4 rows
        mov     rbx,r10                         ; k = CountK
        sub     rbx,64
        jb      CopyPackA_ProcessRemainingColumnsM4

CopyPackA_ProcessNextColumnLoopM4:
        vmovdqu64  zmm16,ZMMWORD PTR [rdx]
        vmovdqu64  zmm17,ZMMWORD PTR [rdx+r8]
        vmovdqu64  zmm18,ZMMWORD PTR [rdx+r8*2]
        vmovdqu64  zmm19,ZMMWORD PTR [rdx+r13]
        vmovdqu64  ZMMWORD PTR [rcx],zmm16
        vmovdqu64  ZMMWORD PTR [rcx+r11],zmm17
        vmovdqu64  ZMMWORD PTR [rcx+r11*2],zmm18
        vmovdqu64  ZMMWORD PTR [rcx+rax],zmm19
        vpmaddubsw zmm16,zmm16,zmm31            ; horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm16              ; add words to row accumulators
        vpmaddubsw zmm17,zmm17,zmm31
        vpaddw     zmm1,zmm1,zmm17
        vpmaddubsw zmm18,zmm18,zmm31
        vpaddw     zmm2,zmm2,zmm18
        vpmaddubsw zmm19,zmm19,zmm31
        vpaddw     zmm3,zmm3,zmm19
        add     rdx,64                          ; src += 64
        add     rcx,64                          ; dst += 64
        sub     rbx,64                          ; k -= 64
        jae     CopyPackA_ProcessNextColumnLoopM4

CopyPackA_ProcessRemainingColumnsM4:
        add     rbx,64                          ; correct for over-subtract above
        jz      CopyPackA_ReduceRowSumBufferM4

        vmovdqu8   zmm16{k1}{z},ZMMWORD PTR [rdx]
        vmovdqu8   zmm17{k1}{z},ZMMWORD PTR [rdx+r8]
        vmovdqu8   zmm18{k1}{z},ZMMWORD PTR [rdx+r8*2]
        vmovdqu8   zmm19{k1}{z},ZMMWORD PTR [rdx+r13]
        vmovdqu64  ZMMWORD PTR [rcx],zmm16
        vmovdqu64  ZMMWORD PTR [rcx+r11],zmm17
        vmovdqu64  ZMMWORD PTR [rcx+r11*2],zmm18
        vmovdqu64  ZMMWORD PTR [rcx+rax],zmm19
        vpmaddubsw zmm16,zmm16,zmm31            ; horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm16              ; add words to row accumulators
        vpmaddubsw zmm17,zmm17,zmm31
        vpaddw     zmm1,zmm1,zmm17
        vpmaddubsw zmm18,zmm18,zmm31
        vpaddw     zmm2,zmm2,zmm18
        vpmaddubsw zmm19,zmm19,zmm31
        vpaddw     zmm3,zmm3,zmm19

;
; Reduce the sums for the four rows of output.
;

CopyPackA_ReduceRowSumBufferM4:
        vpmaddwd       zmm0,zmm0,zmm30          ; horizontal word+word=dword per row
        vpmaddwd       zmm1,zmm1,zmm30
        vpmaddwd       zmm2,zmm2,zmm30
        vpmaddwd       zmm3,zmm3,zmm30
        vextracti64x4  ymm16,zmm0,1             ; fold zmm -> ymm
        vextracti64x4  ymm17,zmm1,1
        vextracti64x4  ymm18,zmm2,1
        vextracti64x4  ymm19,zmm3,1
        vpaddd         ymm0,ymm0,ymm16
        vpaddd         ymm1,ymm1,ymm17
        vpaddd         ymm2,ymm2,ymm18
        vpaddd         ymm3,ymm3,ymm19
        vphaddd        ymm0,ymm0,ymm1           ; reduce and interleave Sum1/Sum0
        vphaddd        ymm1,ymm2,ymm3           ; reduce and interleave Sum3/Sum2
        vphaddd        ymm0,ymm0,ymm1           ; reduce and interleave Sum3/Sum2/Sum1/Sum0
        vextracti128   xmm1,ymm0,1              ; fold ymm -> xmm
        vpaddd         xmm0,xmm0,xmm1
        vmovdqu        XMMWORD PTR [r12],xmm0
        add     r12,4*4                         ; advance row sum buffer by 4 dwords
        sub     r9,4                            ; m -= 4
        jae     CopyPackA_ProcessNextRowM4

CopyPackA_ProcessRemainingRows:
        add     r9,4                            ; correct for over-subtract above
        jz      CopyPackA_ExitRoutine

;
; Process a single row of matrix A in a loop.
;

CopyPackA_ProcessNextRowM1:
        vpxor   xmm0,xmm0,xmm0                  ; clear row accumulator
        mov     rdx,rsi                         ; src = A
        mov     rcx,rdi                         ; dst = D
        add     rsi,r8                          ; A to next row
        add     rdi,r11                         ; D to next row
        mov     rbx,r10                         ; k = CountK
        sub     rbx,64                          ; k -= 64
        jb      CopyPackA_ProcessRemainingColumnsM1

CopyPackA_ProcessNextColumnLoopM1:
        vmovdqu64  zmm16,ZMMWORD PTR [rdx]
        vmovdqu64  ZMMWORD PTR [rcx],zmm16
        vpmaddubsw zmm16,zmm16,zmm31            ; horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm16              ; add words to row accumulators
        add     rdx,64                          ; src += 64
        add     rcx,64                          ; dst += 64
        sub     rbx,64                          ; k -= 64
        jae     CopyPackA_ProcessNextColumnLoopM1

CopyPackA_ProcessRemainingColumnsM1:
        add     rbx,64                          ; correct for over-subtract above
        jz      CopyPackA_ReduceRowSumBufferM1

        vmovdqu8   zmm16{k1}{z},ZMMWORD PTR [rdx]
        vmovdqu64  ZMMWORD PTR [rcx],zmm16
        vpmaddubsw zmm16,zmm16,zmm31            ; horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm16              ; add words to row accumulators

;
; Reduce the sum for the single row of output.
;

CopyPackA_ReduceRowSumBufferM1:
        vpmaddwd       zmm0,zmm0,zmm30          ; horizontal word+word=dword per row
        vextracti64x4  ymm16,zmm0,1             ; fold zmm -> ymm
        vpaddd         ymm0,ymm0,ymm16
        vextracti128   xmm1,ymm0,1              ; fold ymm -> xmm
        vpaddd         xmm0,xmm0,xmm1           ; reduction
        vphaddd        xmm0,xmm0,xmm0
        vphaddd        xmm0,xmm0,xmm0
        vmovd          DWORD PTR [r12],xmm0
        add     r12,4                           ; advance row sum buffer by 1 dword
        dec     r9                              ; decrement rows remaining
        jnz     CopyPackA_ProcessNextRowM1

;
; Restore non-volatile registers and return.
;

CopyPackA_ExitRoutine:
        vzeroupper

        BEGIN_EPILOGUE
        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasGemmU8S8CopyPackAAmx, _TEXT

        END
