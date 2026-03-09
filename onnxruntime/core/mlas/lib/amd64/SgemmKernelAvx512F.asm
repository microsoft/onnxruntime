;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SgemmKernelAvx512F.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;   This implementation uses AVX512F instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE SgemmKernelCommon.inc
INCLUDE FgemmKernelAvx512FCommon.inc
        .list

;
; Generate the GEMM kernel.
;

FgemmKernelAvx512FFunction Float

;
; Stack frame layout for the SGEMM M=1 AVX512 kernels.
;

SgemmKernelM1Avx512Frame STRUCT

        Reserved QWORD ?
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
        Beta QWORD ?

SgemmKernelM1Avx512Frame ENDS

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

        NESTED_ENTRY MlasSgemmKernelM1Avx512F, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        alloc_stack (SgemmKernelM1Avx512Frame.SavedRdi)

        END_PROLOGUE

        mov     rdi,rcx
        mov     rbx,SgemmKernelM1Avx512Frame.ldb[rsp]
        shl     rbx,2                       ; convert ldb to bytes
        mov     rbp,SgemmKernelM1Avx512Frame.CountN[rsp]
        xor     rsi,rsi                     ; processed CountN columns

M1Avx512LoopN:
        cmp     rsi,rbp
        jae     M1Avx512ExitKernel

        mov     r10,rbp
        sub     r10,rsi                     ; remaining CountN columns
        cmp     r10,16
        jb      M1Avx512TailColumns

        mov     r10,16
        mov     eax,0FFFFh
        kmovw   k1,eax
        jmp     M1Avx512ComputeBlock

M1Avx512TailColumns:
        mov     ecx,r10d
        mov     eax,1
        shl     eax,cl
        dec     eax
        kmovw   k1,eax

M1Avx512ComputeBlock:
        vxorps  zmm1,zmm1,zmm1
        mov     rcx,rdi                     ; reload matrix A
        lea     rax,[rdx+rsi*4]             ; matrix B at current column offset
        mov     r11,r9                      ; reload CountK
        test    r11,r11
        jz      M1Avx512ApplyBeta

M1Avx512LoopK:
        vbroadcastss zmm2,DWORD PTR [rcx]
        vmovups zmm3{k1}{z},ZMMWORD PTR [rax]
        vmulps  zmm3,zmm3,zmm2
        vaddps  zmm1,zmm1,zmm3
        add     rcx,4
        add     rax,rbx
        dec     r11
        jnz     M1Avx512LoopK

M1Avx512ApplyBeta:
        mov     eax,DWORD PTR SgemmKernelM1Avx512Frame.Beta[rsp]
        and     eax,7FFFFFFFh
        jz      M1Avx512StoreBlock
        vmovups zmm4{k1}{z},ZMMWORD PTR [r8+rsi*4]
        vaddps  zmm1,zmm1,zmm4

M1Avx512StoreBlock:
        vmovups ZMMWORD PTR [r8+rsi*4]{k1},zmm1
        add     rsi,r10
        jmp     M1Avx512LoopN

M1Avx512ExitKernel:
        vzeroupper
        add     rsp,(SgemmKernelM1Avx512Frame.SavedRdi)

        BEGIN_EPILOGUE

        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasSgemmKernelM1Avx512F, _TEXT

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

        NESTED_ENTRY MlasSgemmKernelM1TransposeBAvx512F, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        alloc_stack (SgemmKernelM1Avx512Frame.SavedRdi)

        END_PROLOGUE

        mov     rdi,rcx
        mov     rbx,SgemmKernelM1Avx512Frame.ldb[rsp]
        shl     rbx,2                       ; convert ldb to bytes
        mov     rbp,SgemmKernelM1Avx512Frame.CountN[rsp]
        xor     rsi,rsi

M1TransposeAvx512LoopN:
        cmp     rsi,rbp
        jae     M1TransposeAvx512ExitKernel

        vxorps  zmm1,zmm1,zmm1
        mov     r10,rdi                     ; reload matrix A
        mov     r11,rdx                     ; reload matrix B row
        mov     rcx,r9                      ; reload CountK
        sub     rcx,16
        jb      M1TransposeAvx512TailK

M1TransposeAvx512LoopK:
        vmovups zmm2,ZMMWORD PTR [r10]
        vmovups zmm3,ZMMWORD PTR [r11]
        vmulps  zmm4,zmm2,zmm3
        vaddps  zmm1,zmm1,zmm4
        add     r10,16*4
        add     r11,16*4
        sub     rcx,16
        jae     M1TransposeAvx512LoopK

M1TransposeAvx512TailK:
        add     rcx,16
        jz      M1TransposeAvx512Reduce
        mov     eax,1
        shl     eax,cl
        dec     eax
        kmovw   k1,eax
        vmovups zmm2{k1}{z},ZMMWORD PTR [r10]
        vmovups zmm3{k1}{z},ZMMWORD PTR [r11]
        vmulps  zmm4,zmm2,zmm3
        vaddps  zmm1,zmm1,zmm4

M1TransposeAvx512Reduce:
        vextractf64x4 ymm2,zmm1,1
        vaddps  ymm1,ymm1,ymm2
        vextractf128 xmm2,ymm1,1
        vaddps  xmm1,xmm1,xmm2
        vhaddps xmm1,xmm1,xmm1
        vhaddps xmm1,xmm1,xmm1

        mov     eax,DWORD PTR SgemmKernelM1Avx512Frame.Beta[rsp]
        and     eax,7FFFFFFFh
        jz      M1TransposeAvx512Store
        vmovss  xmm3,DWORD PTR [r8+rsi*4]
        vaddss  xmm1,xmm1,xmm3

M1TransposeAvx512Store:
        vmovss  DWORD PTR [r8+rsi*4],xmm1
        add     rdx,rbx                     ; advance matrix B by 1 row
        inc     rsi
        jmp     M1TransposeAvx512LoopN

M1TransposeAvx512ExitKernel:
        vzeroupper
        add     rsp,(SgemmKernelM1Avx512Frame.SavedRdi)

        BEGIN_EPILOGUE

        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasSgemmKernelM1TransposeBAvx512F, _TEXT

        END
