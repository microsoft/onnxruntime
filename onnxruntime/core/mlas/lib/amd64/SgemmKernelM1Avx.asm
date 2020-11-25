;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelM1Avx.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM). This handles the special case of M=1.
;
;   This implementation uses AVX instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        EXTERN  MlasMaskMoveAvx:NEAR

;
; Stack frame layout for the SGEMM M=1 kernels.
;

SgemmKernelM1Frame STRUCT

        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
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
        Beta QWORD ?

SgemmKernelM1Frame ENDS

;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows. This handles the special case of M=1.
;
;   The elements in matrix B are not transposed.
;
; Arguments:
;
;   A (rcx) - Supplies the address of matrix A.
;
;   B (rdx) - Supplies the address of matrix B.
;
;   C (r8) - Supplies the address of matrix C.
;
;   CountK (r9) - Supplies the number of columns from matrix A and the number
;       of rows from matrix B to iterate over.
;
;   CountN - Supplies the number of columns from matrix B and matrix C to iterate
;       over.
;
;   ldb - Supplies the first dimension of matrix B.
;
;   Beta - Supplies the scalar beta multiplier (see SGEMM definition).
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasSgemmKernelM1Avx, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        alloc_stack (SgemmKernelM1Frame.SavedRsi)
        save_xmm128 xmm6,SgemmKernelM1Frame.SavedXmm6
        save_xmm128 xmm7,SgemmKernelM1Frame.SavedXmm7
        save_xmm128 xmm8,SgemmKernelM1Frame.SavedXmm8

        END_PROLOGUE

        mov     rbx,SgemmKernelM1Frame.ldb[rsp]
        shl     rbx,2                       ; convert ldb to bytes
        mov     r10,r8
        mov     r11,rdx
        mov     rbp,SgemmKernelM1Frame.CountN[rsp]

;
; Compute the initial results mask for zeroing or accumulate mode.
;

        vxorps  xmm0,xmm0,xmm0
        vcmpeqss xmm0,xmm0,DWORD PTR SgemmKernelM1Frame.Beta[rsp]
        vshufps xmm0,xmm0,xmm0,0
        vinsertf128 ymm0,ymm0,xmm0,1

;
; Compute the conditional load/store mask for an unaligned CountN.
;

        mov     eax,ebp
        and     eax,7
        vmovd   xmm7,eax
        vshufps xmm7,xmm7,xmm7,0
        vpcmpgtd xmm6,xmm7,XMMWORD PTR [MlasMaskMoveAvx+16]
        vpcmpgtd xmm7,xmm7,XMMWORD PTR [MlasMaskMoveAvx]
        vinsertf128 ymm7,ymm7,xmm6,1

;
; Process 4 rows of the matrices in a loop.
;

        sub     r9,4
        jb      ProcessRemainingCountK

ProcessRowLoop4:
        vbroadcastss ymm2,DWORD PTR [rcx]
        mov     rax,rbp                     ; reload CountN
        vbroadcastss ymm3,DWORD PTR [rcx+4]
        mov     rdx,r11                     ; reload matrix B
        vbroadcastss ymm4,DWORD PTR [rcx+8]
        mov     r8,r10                      ; reload matrix C
        vbroadcastss ymm5,DWORD PTR [rcx+12]
        add     rcx,4*4                     ; advance matrix A by 4 columns
        lea     r11,[rdx+rbx*4]             ; advance matrix B by 4 rows
        sub     rax,16
        jb      ProcessRemainingCountN4

ProcessColumnLoop4:
        lea     rsi,[rdx+rbx*2]             ; compute matrix B plus 2 rows
        vmulps  ymm1,ymm2,YMMWORD PTR [rdx]
        vmulps  ymm6,ymm2,YMMWORD PTR [rdx+32]
        vmulps  ymm8,ymm3,YMMWORD PTR [rdx+rbx]
        vaddps  ymm1,ymm1,ymm8
        vmulps  ymm8,ymm3,YMMWORD PTR [rdx+rbx+32]
        vaddps  ymm6,ymm6,ymm8
        vmulps  ymm8,ymm4,YMMWORD PTR [rsi]
        vaddps  ymm1,ymm1,ymm8
        vmulps  ymm8,ymm4,YMMWORD PTR [rsi+32]
        vaddps  ymm6,ymm6,ymm8
        vmulps  ymm8,ymm5,YMMWORD PTR [rsi+rbx]
        vaddps  ymm1,ymm1,ymm8
        vmulps  ymm8,ymm5,YMMWORD PTR [rsi+rbx+32]
        vaddps  ymm6,ymm6,ymm8
        vandnps ymm8,ymm0,YMMWORD PTR [r8]
        vaddps  ymm1,ymm1,ymm8
        vandnps ymm8,ymm0,YMMWORD PTR [r8+32]
        vaddps  ymm6,ymm6,ymm8
        vmovups YMMWORD PTR [r8],ymm1
        vmovups YMMWORD PTR [r8+32],ymm6
        add     rdx,16*4                    ; advance matrix B by 16 columns
        add     r8,16*4                     ; advance matrix C by 16 columns
        sub     rax,16
        jae     ProcessColumnLoop4

ProcessRemainingCountN4:
        test    al,15                       ; test for unaligned columns
        jz      ProcessedRemainingCountN4
        test    al,8                        ; CountN >= 8?
        jz      ProcessRemainingCountNSmall4
        lea     rsi,[rdx+rbx*2]             ; compute matrix B plus 2 rows
        vmulps  ymm1,ymm2,YMMWORD PTR [rdx]
        vmulps  ymm8,ymm3,YMMWORD PTR [rdx+rbx]
        vaddps  ymm1,ymm1,ymm8
        vmulps  ymm8,ymm4,YMMWORD PTR [rsi]
        vaddps  ymm1,ymm1,ymm8
        vmulps  ymm8,ymm5,YMMWORD PTR [rsi+rbx]
        vaddps  ymm1,ymm1,ymm8
        vandnps ymm8,ymm0,YMMWORD PTR [r8]
        vaddps  ymm1,ymm1,ymm8
        vmovups YMMWORD PTR [r8],ymm1
        add     rdx,8*4                     ; advance matrix B by 8 columns
        add     r8,8*4                      ; advance matrix C by 8 columns
        test    al,7
        jz      ProcessedRemainingCountN4

ProcessRemainingCountNSmall4:
        lea     rsi,[rdx+rbx*2]             ; compute matrix B plus 2 rows
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx]
        vmulps  ymm1,ymm2,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx+rbx]
        vmulps  ymm8,ymm3,ymm6
        vaddps  ymm1,ymm1,ymm8
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi]
        vmulps  ymm8,ymm4,ymm6
        vaddps  ymm1,ymm1,ymm8
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi+rbx]
        vmulps  ymm8,ymm5,ymm6
        vaddps  ymm1,ymm1,ymm8
        vmaskmovps ymm6,ymm7,YMMWORD PTR [r8]
        vandnps ymm6,ymm0,ymm6
        vaddps  ymm1,ymm1,ymm6
        vmaskmovps YMMWORD PTR [r8],ymm7,ymm1

ProcessedRemainingCountN4:
        vxorps  xmm0,xmm0,xmm0              ; switch to accumulate mode
        sub     r9,4
        jae     ProcessRowLoop4

ProcessRemainingCountK:
        test    r9d,2
        jnz     ProcessRowLoop2
        test    r9d,1
        jnz     ProcessRowLoop1

ExitKernel:
        vzeroupper
        movaps  xmm6,SgemmKernelM1Frame.SavedXmm6[rsp]
        movaps  xmm7,SgemmKernelM1Frame.SavedXmm7[rsp]
        movaps  xmm8,SgemmKernelM1Frame.SavedXmm8[rsp]
        add     rsp,(SgemmKernelM1Frame.SavedRsi)

        BEGIN_EPILOGUE

        pop     rsi
        pop     rbx
        pop     rbp
        ret

;
; Process 2 rows of the matrices.
;

ProcessRowLoop2:
        vbroadcastss ymm2,DWORD PTR [rcx]
        mov     rax,rbp                     ; reload CountN
        vbroadcastss ymm3,DWORD PTR [rcx+4]
        mov     rdx,r11                     ; reload matrix B
        mov     r8,r10                      ; reload matrix C
        add     rcx,2*4                     ; advance matrix A by 2 columns
        lea     r11,[rdx+rbx*2]             ; advance matrix B by 2 rows
        sub     rax,8
        jb      ProcessRemainingCountN2

ProcessColumnLoop2:
        vmulps  ymm1,ymm2,YMMWORD PTR [rdx]
        vmulps  ymm8,ymm3,YMMWORD PTR [rdx+rbx]
        vaddps  ymm1,ymm1,ymm8
        vandnps ymm6,ymm0,YMMWORD PTR [r8]
        vaddps  ymm1,ymm1,ymm6
        vmovups YMMWORD PTR [r8],ymm1
        add     rdx,8*4                     ; advance matrix B by 8 columns
        add     r8,8*4                      ; advance matrix C by 8 columns
        sub     rax,8
        jae     ProcessColumnLoop2

ProcessRemainingCountN2:
        test    al,7                        ; test for unaligned columns
        jz      ProcessedRemainingCountN2
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx]
        vmulps  ymm1,ymm2,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx+rbx]
        vmulps  ymm8,ymm3,ymm6
        vaddps  ymm1,ymm1,ymm8
        vmaskmovps ymm6,ymm7,YMMWORD PTR [r8]
        vandnps ymm6,ymm0,ymm6
        vaddps  ymm1,ymm1,ymm6
        vmaskmovps YMMWORD PTR [r8],ymm7,ymm1

ProcessedRemainingCountN2:
        test    r9d,1
        jz      ExitKernel
        vxorps  xmm0,xmm0,xmm0              ; switch to accumulate mode

;
; Process 1 row of the matrices.
;

ProcessRowLoop1:
        vbroadcastss ymm2,DWORD PTR [rcx]
        mov     rax,rbp                     ; reload CountN
        mov     rdx,r11                     ; reload matrix B
        mov     r8,r10                      ; reload matrix C
        sub     rax,8
        jb      ProcessRemainingCountN1

ProcessColumnLoop1:
        vmulps  ymm1,ymm2,YMMWORD PTR [rdx]
        vandnps ymm6,ymm0,YMMWORD PTR [r8]
        vaddps  ymm1,ymm1,ymm6
        vmovups YMMWORD PTR [r8],ymm1
        add     rdx,8*4                     ; advance matrix B by 8 columns
        add     r8,8*4                      ; advance matrix C by 8 columns
        sub     rax,8
        jae     ProcessColumnLoop1

ProcessRemainingCountN1:
        test    al,7                        ; test for unaligned columns
        jz      ExitKernel
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx]
        vmulps  ymm1,ymm2,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [r8]
        vandnps ymm6,ymm0,ymm6
        vaddps  ymm1,ymm1,ymm6
        vmaskmovps YMMWORD PTR [r8],ymm7,ymm1
        jmp     ExitKernel

        NESTED_END MlasSgemmKernelM1Avx, _TEXT

;++
;
; Routine Description:
;
;   This routine is an inner kernel to compute matrix multiplication for a
;   set of rows. This handles the special case of M=1.
;
;   The elements in matrix B are transposed.
;
; Arguments:
;
;   A (rcx) - Supplies the address of matrix A.
;
;   B (rdx) - Supplies the address of matrix B. The elements are transposed.
;
;   C (r8) - Supplies the address of matrix C.
;
;   CountK (r9) - Supplies the number of columns from matrix A and the number
;       of columns from matrix B to iterate over.
;
;   CountN - Supplies the number of rows from matrix B and the number of columns
;       from matrix C to iterate over.
;
;   ldb - Supplies the first dimension of matrix B.
;
;   Beta - Supplies the scalar beta multiplier (see SGEMM definition).
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasSgemmKernelM1TransposeBAvx, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        alloc_stack (SgemmKernelM1Frame.SavedRsi)
        save_xmm128 xmm6,SgemmKernelM1Frame.SavedXmm6
        save_xmm128 xmm7,SgemmKernelM1Frame.SavedXmm7

        END_PROLOGUE

        mov     rbx,SgemmKernelM1Frame.ldb[rsp]
        shl     rbx,2                       ; convert ldb to bytes
        mov     r10,rcx
        mov     r11,rdx
        mov     rbp,SgemmKernelM1Frame.CountN[rsp]

;
; Compute the results mask for zeroing or accumulate mode.
;

        vxorps  xmm0,xmm0,xmm0
        vcmpeqss xmm0,xmm0,DWORD PTR SgemmKernelM1Frame.Beta[rsp]
        vshufps xmm0,xmm0,xmm0,0

;
; Compute the conditional load/store mask for an unaligned CountK.
;

        mov     eax,r9d
        and     eax,7
        vmovd   xmm7,eax
        vshufps xmm7,xmm7,xmm7,0
        vpcmpgtd xmm6,xmm7,XMMWORD PTR [MlasMaskMoveAvx+16]
        vpcmpgtd xmm7,xmm7,XMMWORD PTR [MlasMaskMoveAvx]
        vinsertf128 ymm7,ymm7,xmm6,1

;
; Process 4 rows of the matrices in a loop.
;

        sub     rbp,4
        jb      ProcessRemainingCountN

ProcessRowLoop4:
        vxorps  xmm2,xmm2,xmm2              ; clear row accumulators
        vxorps  xmm3,xmm3,xmm3
        vxorps  xmm4,xmm4,xmm4
        vxorps  xmm5,xmm5,xmm5
        mov     rcx,r10                     ; reload matrix A
        mov     rdx,r11                     ; reload matrix B
        mov     rax,r9                      ; reload CountK
        lea     r11,[rdx+rbx*4]             ; advance matrix B by 4 rows
        sub     rax,8
        jb      ProcessRemainingCountK4

ProcessColumnLoop4:
        lea     rsi,[rdx+rbx*2]             ; compute matrix B plus 2 rows
        vmovups ymm1,YMMWORD PTR [rcx]
        vmulps  ymm6,ymm1,YMMWORD PTR [rdx]
        vaddps  ymm2,ymm2,ymm6
        vmulps  ymm6,ymm1,YMMWORD PTR [rdx+rbx]
        vaddps  ymm3,ymm3,ymm6
        vmulps  ymm6,ymm1,YMMWORD PTR [rsi]
        vaddps  ymm4,ymm4,ymm6
        vmulps  ymm6,ymm1,YMMWORD PTR [rsi+rbx]
        vaddps  ymm5,ymm5,ymm6
        add     rcx,8*4                     ; advance matrix A by 8 columns
        add     rdx,8*4                     ; advance matrix B by 8 columns
        sub     rax,8
        jae     ProcessColumnLoop4

ProcessRemainingCountK4:
        test    al,7                        ; test for unaligned columns
        jz      Output4x1Block
        lea     rsi,[rdx+rbx*2]             ; compute matrix B plus 2 rows
        vmaskmovps ymm1,ymm7,YMMWORD PTR [rcx]
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm2,ymm2,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx+rbx]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm3,ymm3,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm4,ymm4,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rsi+rbx]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm5,ymm5,ymm6

;
; Reduce and output the row accumulators.
;

Output4x1Block:
        vunpcklps ymm6,ymm2,ymm3            ; transpose row accumulators
        vunpckhps ymm1,ymm2,ymm3
        vunpcklps ymm2,ymm4,ymm5
        vunpckhps ymm3,ymm4,ymm5
        vunpcklpd ymm4,ymm6,ymm2
        vunpckhpd ymm5,ymm6,ymm2
        vaddps  ymm4,ymm4,ymm5
        vunpcklpd ymm6,ymm1,ymm3
        vunpckhpd ymm2,ymm1,ymm3
        vaddps  ymm4,ymm4,ymm6
        vaddps  ymm4,ymm4,ymm2
        vextractf128 xmm5,ymm4,1
        vaddps  xmm4,xmm4,xmm5
        vandnps xmm6,xmm0,XMMWORD PTR [r8]
        vaddps  xmm4,xmm4,xmm6
        vmovups XMMWORD PTR [r8],xmm4
        add     r8,4*4                      ; advance matrix C by 4 columns
        sub     rbp,4
        jae     ProcessRowLoop4

ProcessRemainingCountN:
        test    ebp,2
        jnz     ProcessRowLoop2
        test    ebp,1
        jnz     ProcessRowLoop1

ExitKernel:
        vzeroupper
        movaps  xmm6,SgemmKernelM1Frame.SavedXmm6[rsp]
        movaps  xmm7,SgemmKernelM1Frame.SavedXmm7[rsp]
        add     rsp,(SgemmKernelM1Frame.SavedRsi)

        BEGIN_EPILOGUE

        pop     rsi
        pop     rbx
        pop     rbp
        ret

;
; Process 2 rows of the matrices.
;

ProcessRowLoop2:
        vxorps  xmm2,xmm2,xmm2              ; clear row accumulators
        vxorps  xmm3,xmm3,xmm3
        mov     rcx,r10                     ; reload matrix A
        mov     rdx,r11                     ; reload matrix B
        mov     rax,r9                      ; reload CountK
        lea     r11,[rdx+rbx*2]             ; advance matrix B by 2 rows
        sub     rax,8
        jb      ProcessRemainingCountK2

ProcessColumnLoop2:
        vmovups ymm1,YMMWORD PTR [rcx]
        vmulps  ymm6,ymm1,YMMWORD PTR [rdx]
        vaddps  ymm2,ymm2,ymm6
        vmulps  ymm6,ymm1,YMMWORD PTR [rdx+rbx]
        vaddps  ymm3,ymm3,ymm6
        add     rcx,8*4                     ; advance matrix A by 8 columns
        add     rdx,8*4                     ; advance matrix B by 8 columns
        sub     rax,8
        jae     ProcessColumnLoop2

ProcessRemainingCountK2:
        test    al,7                        ; test for unaligned columns
        jz      Output2x1Block
        vmaskmovps ymm1,ymm7,YMMWORD PTR [rcx]
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm2,ymm2,ymm6
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx+rbx]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm3,ymm3,ymm6

;
; Reduce and output the row accumulators.
;

Output2x1Block:
        vunpcklps ymm4,ymm2,ymm3            ; reduce row accumulators
        vunpckhps ymm2,ymm2,ymm3
        vaddps  ymm2,ymm2,ymm4
        vextractf128 xmm4,ymm2,1
        vaddps  xmm2,xmm2,xmm4
        vmovhlps xmm4,xmm2,xmm2
        vaddps  xmm2,xmm2,xmm4
        vmovsd  xmm3,QWORD PTR [r8]
        vandnps xmm3,xmm0,xmm3
        vaddps  xmm2,xmm2,xmm3
        vmovsd  QWORD PTR [r8],xmm2
        add     r8,2*4                      ; advance matrix C by 2 columns
        test    ebp,1
        jz      ExitKernel

;
; Process 1 row of the matrices.
;

ProcessRowLoop1:
        vxorps  xmm2,xmm2,xmm2              ; clear row accumulators
        mov     rcx,r10                     ; reload matrix A
        mov     rdx,r11                     ; reload matrix B
        mov     rax,r9                      ; reload CountK
        sub     rax,8
        jb      ProcessRemainingCountK1

ProcessColumnLoop1:
        vmovups ymm1,YMMWORD PTR [rcx]
        vmulps  ymm6,ymm1,YMMWORD PTR [rdx]
        vaddps  ymm2,ymm2,ymm6
        add     rcx,8*4                     ; advance matrix A by 8 columns
        add     rdx,8*4                     ; advance matrix B by 8 columns
        sub     rax,8
        jae     ProcessColumnLoop1

ProcessRemainingCountK1:
        test    al,7                        ; test for unaligned columns
        jz      Output1x1Block
        vmaskmovps ymm1,ymm7,YMMWORD PTR [rcx]
        vmaskmovps ymm6,ymm7,YMMWORD PTR [rdx]
        vmulps  ymm6,ymm1,ymm6
        vaddps  ymm2,ymm2,ymm6

;
; Reduce and output the row accumulators.
;

Output1x1Block:
        vhaddps ymm2,ymm2,ymm2              ; reduce row accumulators
        vhaddps ymm2,ymm2,ymm2
        vextractf128 xmm4,ymm2,1
        vaddss  xmm2,xmm2,xmm4
        vmovss  xmm3,DWORD PTR [r8]
        vandnps xmm3,xmm0,xmm3
        vaddss  xmm2,xmm2,xmm3
        vmovss  DWORD PTR [r8],xmm2
        jmp     ExitKernel

        NESTED_END MlasSgemmKernelM1TransposeBAvx, _TEXT

        END
