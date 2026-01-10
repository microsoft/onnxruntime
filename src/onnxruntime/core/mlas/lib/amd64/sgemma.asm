;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   sgemma.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision matrix/matrix
;   multiply operation (SGEMM).
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

;++
;
; Routine Description:
;
;   This routine transposes elements from the source matrix to the destination
;   packed buffer.
;
;   4 columns of 16 rows from the source matrix are transposed to 16 columns of 4
;   rows in the destination packed buffer.
;
;   This implementation uses SSE2 instructions.
;
; Arguments:
;
;   D (rcx) - Supplies the address of the destination packed buffer.
;
;   B (rdx) - Supplies the address of the source matrix.
;
;   ldb (r8d) - Supplies the number of elements per row of the source matrix.
;
; Return Value:
;
;   None.
;
;--

        LEAF_ENTRY MlasSgemmTransposePackB16x4Sse, _TEXT

        shl     r8,2                        ; convert ldb to bytes
        mov     r9d,4                       ; transpose four 4x4 blocks

TransposeBlockLoop:
        lea     rax,[rdx+r8*2]
        movups  xmm0,XMMWORD PTR [rdx]
        movups  xmm1,XMMWORD PTR [rdx+r8]
        movups  xmm2,XMMWORD PTR [rax]
        movups  xmm3,XMMWORD PTR [rax+r8]
        movaps  xmm4,xmm0
        unpcklps xmm4,xmm1
        unpckhps xmm0,xmm1
        movaps  xmm5,xmm2
        unpcklps xmm5,xmm3
        unpckhps xmm2,xmm3
        movaps  xmm1,xmm4
        unpcklpd xmm1,xmm5
        unpckhpd xmm4,xmm5
        movaps  xmm3,xmm0
        unpcklpd xmm3,xmm2
        unpckhpd xmm0,xmm2
        movaps  XMMWORD PTR [rcx+16*4*0],xmm1
        movaps  XMMWORD PTR [rcx+16*4*1],xmm4
        movaps  XMMWORD PTR [rcx+16*4*2],xmm3
        movaps  XMMWORD PTR [rcx+16*4*3],xmm0
        add     rcx,4*4
        lea     rdx,[rax+r8*2]
        dec     r9d
        jnz     TransposeBlockLoop
        ret

        LEAF_END MlasSgemmTransposePackB16x4Sse, _TEXT

;
; Transpose8x4BlockAvx
;
;   4 columns of 8 rows from the source matrix are transposed to 8 columns of 4
;   rows in the destination packed buffer.
;
;   This implementation uses AVX instructions.
;
; Arguments:
;
;   StoreOffset - Supplies the relative byte offset into the destination packed
;       buffer.
;
; Implicit Arguments:
;
;   rcx - Supplies the address of the destination packed buffer.
;
;   rdx - Supplies the address of the source matrix.
;
;   r8 - Supplies the number of elements per row of the source matrix.
;

TransposePackB8x4BlockAvx MACRO StoreOffset

;
; Load 4 columns from 8 rows of the source matrix into the lower and upper
; halves of 4 YMM registers.
;

        lea     rax,[rdx+r8*2]
        vmovups xmm0,XMMWORD PTR [rdx]
        vmovups xmm1,XMMWORD PTR [rdx+r8]
        lea     rdx,[rax+r8*2]
        vmovups xmm2,XMMWORD PTR [rax]
        vmovups xmm3,XMMWORD PTR [rax+r8]
        lea     rax,[rdx+r8*2]
        vinsertf128 ymm0,ymm0,XMMWORD PTR [rdx],1
        vinsertf128 ymm1,ymm1,XMMWORD PTR [rdx+r8],1
        vinsertf128 ymm2,ymm2,XMMWORD PTR [rax],1
        vinsertf128 ymm3,ymm3,XMMWORD PTR [rax+r8],1

;
; Transpose the lower and upper halves of the 4 YMM registers as two 4x4
; matrices and store the output to the destination packed buffer.
;

        vunpcklps ymm4,ymm0,ymm1
        vunpckhps ymm5,ymm0,ymm1
        vunpcklps ymm0,ymm2,ymm3
        vunpckhps ymm1,ymm2,ymm3
        vunpcklpd ymm2,ymm4,ymm0
        vunpckhpd ymm3,ymm4,ymm0
        vmovaps YMMWORD PTR [rcx+16*4*0+StoreOffset],ymm2
        vmovaps YMMWORD PTR [rcx+16*4*1+StoreOffset],ymm3
        vunpcklpd ymm0,ymm5,ymm1
        vunpckhpd ymm4,ymm5,ymm1
        vmovaps YMMWORD PTR [rcx+16*4*2+StoreOffset],ymm0
        vmovaps YMMWORD PTR [rcx+16*4*3+StoreOffset],ymm4

        ENDM

;++
;
; Routine Description:
;
;   This routine transposes elements from the source matrix to the destination
;   packed buffer.
;
;   4 columns of 16 rows from the source matrix are transposed to 16 columns of 4
;   rows in the destination packed buffer.
;
;   This implementation uses AVX instructions.
;
; Arguments:
;
;   D (rcx) - Supplies the address of the destination packed buffer.
;
;   B (rdx) - Supplies the address of the source matrix.
;
;   ldb (r8d) - Supplies the number of elements per row of the source matrix.
;
; Return Value:
;
;   None.
;
;--

        LEAF_ENTRY MlasSgemmTransposePackB16x4Avx, _TEXT

        shl     r8,2                        ; convert ldb to bytes
        TransposePackB8x4BlockAvx 0*4
        lea     rdx,[rax+r8*2]
        TransposePackB8x4BlockAvx 8*4
        vzeroupper
        ret

        LEAF_END MlasSgemmTransposePackB16x4Avx, _TEXT

        END
