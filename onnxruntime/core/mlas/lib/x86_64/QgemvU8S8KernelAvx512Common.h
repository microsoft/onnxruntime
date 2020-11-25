/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemvU8S8KernelAvx512Common.h

Abstract:

    This module contains common kernel macros and structures for the quantized
    integer matrix/vector multiply operation (QGEMV) for the AVX512 core and
    AVX512VNNI kernels.

--*/

//
// Stack frame layout for the U8S8 kernel.
//

        .equ    .LGemvU8S8KernelFrame_SavedRbx, 0
        .equ    .LGemvU8S8KernelFrame_SavedRbp, 8
        .equ    .LGemvU8S8KernelFrame_ReturnAddress, 16

/*++

Macro Description:

    This macro generates the common AVX512 code for the inner kernel to compute
    matrix/vector multiplication.

Arguments:

    Isa - Supplies the instruction set architecture string for function tags.

--*/

        .macro GemvU8S8KernelAvx512Function Isa

/*++

Routine Description:

    This routine is an inner kernel to compute matrix/vector multiplication.

Arguments:

    A (rdi) - Supplies the address of vector A.

    B (rsi) - Supplies the address of matrix B.

    C (rdx) - Supplies the address of matrix C.

    CountK (rcx) - Supplies the number of columns from vector A and the number
        of rows from matrix B to iterate over.

    CountN (r8) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldb (r9) - Supplies the first dimension of matrix B.

Return Value:

    None.

--*/

        .globl  C_UNDERSCORE(MlasGemvU8S8Kernel\Isa\())
C_UNDERSCORE(MlasGemvU8S8Kernel\Isa\()):

        push    rbp
        push    rbx

        mov     rbx,rcx
        mov     ecx,r8d
        and     ecx,15                      # isolate unaligned count
        mov     eax,1
        shl     eax,cl
        dec     eax
        kmovw   k1,eax                      # compute vector load/store mask
        mov     rcx,rbx
        mov     r10,rdx
        mov     r11,rsp                     # set ZeroMode to any non-zero value
.ifeqs "\Isa\()", "Avx512Core"
        mov     eax,1
        vpbroadcastw zmm29,eax
.endif

//
// Process 4 rows of matrix B in a loop.
//

        sub     rcx,4
        jb      .LProcessRemainingRows

.LProcessRowLoop4:
        mov     rdx,rsi                     # reload matrix B
        lea     rsi,[rsi+r9*4]              # advance matrix B by 4 rows
        mov     rbx,r10                     # reload matrix C
        mov     rbp,r8                      # reload CountN
        vpbroadcastd zmm28,DWORD PTR [rdi]
        add     rdi,4                       # advance matrix A by 4 bytes

//
// Process sets of 64 columns from the 4 rows in a loop.
//
// Some permute operations are deferred until the final store of the 4x64 block
// as these permutes are expensive.
//

.LProcessColumnLoop4By64:
        cmp     rbp,64
        jb      .LProcessColumnLoop4By16
        lea     rax,[rdx+r9*2]              # compute matrix B plus 2 rows
        vmovdqu32 zmm16,ZMMWORD PTR [rdx]
        vmovdqu32 zmm17,ZMMWORD PTR [rdx+r9]
        vmovdqu32 zmm18,ZMMWORD PTR [rax]
        vmovdqu32 zmm19,ZMMWORD PTR [rax+r9]
        vpunpcklbw zmm20,zmm16,zmm17        # interleave row data bytes
        vpunpckhbw zmm21,zmm16,zmm17
        vpunpcklbw zmm22,zmm18,zmm19
        vpunpckhbw zmm23,zmm18,zmm19
        vpunpcklwd zmm16,zmm20,zmm22        # interleave row data words
        vpunpckhwd zmm17,zmm20,zmm22
        vpunpcklwd zmm18,zmm21,zmm23
        vpunpckhwd zmm19,zmm21,zmm23
.ifeqs "\Isa\()", "Avx512Core"
        vpmaddubsw zmm16,zmm28,zmm16
        vpmaddwd zmm20,zmm16,zmm29
        vpmaddubsw zmm17,zmm28,zmm17
        vpmaddwd zmm21,zmm17,zmm29
        vpmaddubsw zmm18,zmm28,zmm18
        vpmaddwd zmm22,zmm18,zmm29
        vpmaddubsw zmm19,zmm28,zmm19
        vpmaddwd zmm23,zmm19,zmm29
.else
        vpxord zmm20,zmm20,zmm20
        vpxord zmm21,zmm21,zmm21
        vpxord zmm22,zmm22,zmm22
        vpxord zmm23,zmm23,zmm23
        VpdpbusdsZmmZmmZmm zmm20,zmm28,zmm16
        VpdpbusdsZmmZmmZmm zmm21,zmm28,zmm17
        VpdpbusdsZmmZmmZmm zmm22,zmm28,zmm18
        VpdpbusdsZmmZmmZmm zmm23,zmm28,zmm19
.endif
        test    r11,r11                     # ZeroMode?
        jnz     .LSkipAccumulateOutput4By64
        vpaddd  zmm20,zmm20,ZMMWORD PTR [rbx]
        vpaddd  zmm21,zmm21,ZMMWORD PTR [rbx+16*4]
        vpaddd  zmm22,zmm22,ZMMWORD PTR [rbx+32*4]
        vpaddd  zmm23,zmm23,ZMMWORD PTR [rbx+48*4]

.LSkipAccumulateOutput4By64:
        cmp     rcx,4                       # final 4x64 block?
        jae     .LStoreOutput4By64
        vextracti32x4 XMMWORD PTR [rbx],zmm20,0
        vextracti32x4 XMMWORD PTR [rbx+4*4],zmm21,0
        vextracti32x4 XMMWORD PTR [rbx+8*4],zmm22,0
        vextracti32x4 XMMWORD PTR [rbx+12*4],zmm23,0
        vextracti32x4 XMMWORD PTR [rbx+16*4],zmm20,1
        vextracti32x4 XMMWORD PTR [rbx+20*4],zmm21,1
        vextracti32x4 XMMWORD PTR [rbx+24*4],zmm22,1
        vextracti32x4 XMMWORD PTR [rbx+28*4],zmm23,1
        vextracti32x4 XMMWORD PTR [rbx+32*4],zmm20,2
        vextracti32x4 XMMWORD PTR [rbx+36*4],zmm21,2
        vextracti32x4 XMMWORD PTR [rbx+40*4],zmm22,2
        vextracti32x4 XMMWORD PTR [rbx+44*4],zmm23,2
        vextracti32x4 XMMWORD PTR [rbx+48*4],zmm20,3
        vextracti32x4 XMMWORD PTR [rbx+52*4],zmm21,3
        vextracti32x4 XMMWORD PTR [rbx+56*4],zmm22,3
        vextracti32x4 XMMWORD PTR [rbx+60*4],zmm23,3
        jmp     .LAdvanceColumnLoop64

.LStoreOutput4By64:
        vmovdqu32 ZMMWORD PTR [rbx],zmm20
        vmovdqu32 ZMMWORD PTR [rbx+16*4],zmm21
        vmovdqu32 ZMMWORD PTR [rbx+32*4],zmm22
        vmovdqu32 ZMMWORD PTR [rbx+48*4],zmm23

.LAdvanceColumnLoop64:
        add     rdx,64                      # advance matrix B by 64 bytes
        add     rbx,64*4                    # advance matrix C by 64 columns
        sub     rbp,64                      # decrement CountN
        jnz     .LProcessColumnLoop4By64

.LAdvanceRowLoop4:
        xor     r11,r11                     # clear ZeroMode
        sub     rcx,4                       # decrement CountK
        jae     .LProcessRowLoop4

.LProcessRemainingRows:
        add     rcx,4                       # correct for over-subtract above
        jnz     .LProcessRemainingSmallK

.LExitKernel:
        vzeroupper

        pop     rbx
        pop     rbp
        ret

//
// Process sets of 16 columns from the 4 rows in a loop or process the remaining
// 1 to 15 columns.
//

.LProcessColumnLoop4By16:
        lea     rax,[rdx+r9*2]             # compute matrix B plus 2 rows
        cmp     ebp,16
        jb      .LLoadPartialVector4BySmallN
        vmovdqu xmm2,XMMWORD PTR [rdx]
        vmovdqu xmm3,XMMWORD PTR [rdx+r9]
        vmovdqu xmm4,XMMWORD PTR [rax]
        vmovdqu xmm5,XMMWORD PTR [rax+r9]
        jmp     .LComputeOutput4By16

.LLoadPartialVector4BySmallN:
        vmovdqu8 zmm2{k1}{z},ZMMWORD PTR [rdx]
        vmovdqu8 zmm3{k1}{z},ZMMWORD PTR [rdx+r9]
        vmovdqu8 zmm4{k1}{z},ZMMWORD PTR [rax]
        vmovdqu8 zmm5{k1}{z},ZMMWORD PTR [rax+r9]

.LComputeOutput4By16:
        vpunpcklbw xmm1,xmm2,xmm3           # interleave row data bytes
        vpunpckhbw xmm2,xmm2,xmm3
        vpunpcklbw xmm3,xmm4,xmm5
        vpunpckhbw xmm4,xmm4,xmm5
        vpunpcklwd xmm5,xmm1,xmm3           # interleave row data words
        vpunpckhwd xmm1,xmm1,xmm3
        vpunpcklwd xmm3,xmm2,xmm4
        vpunpckhwd xmm2,xmm2,xmm4
        vinserti128 ymm5,ymm5,xmm1,1        # concatenate 256-bit vector
        vinserti128 ymm3,ymm3,xmm2,1
        vshufi32x4 zmm16,zmm5,zmm3,0x44     # concatenate 512-bit vector
.ifeqs "\Isa\()", "Avx512Core"
        vpmaddubsw zmm16,zmm28,zmm16
        vpmaddwd zmm20,zmm16,zmm29
.else
        vpxord zmm20,zmm20,zmm20
        VpdpbusdsZmmZmmZmm zmm20,zmm28,zmm16
.endif
        cmp     ebp,16
        jb      .LStorePartialVector4BySmallN
        test    r11,r11                     # ZeroMode?
        jnz     .LSkipAccumulateOutput4By16
        vpaddd  zmm20,zmm20,ZMMWORD PTR [rbx]

.LSkipAccumulateOutput4By16:
        vmovdqu32 ZMMWORD PTR [rbx],zmm20
        add     rdx,16                      # advance matrix B by 16 bytes
        add     rbx,16*4                    # advance matrix C by 16 columns
        sub     ebp,16                      # decrement CountN
        jnz     .LProcessColumnLoop4By16
        jmp     .LAdvanceRowLoop4

.LStorePartialVector4BySmallN:
        test    r11,r11                     # ZeroMode?
        jnz     .LSkipAccumulateOutput4BySmallN
        vpaddd  zmm20{k1}{z},zmm20,ZMMWORD PTR [rbx]

.LSkipAccumulateOutput4BySmallN:
        vmovdqu32 ZMMWORD PTR [rbx]{k1},zmm20
        jmp     .LAdvanceRowLoop4

//
// Broadcast the remaining 1 to 3 values from vector A.
//

.LProcessRemainingSmallK:
        vpxor   xmm0,xmm0,xmm0
        cmp     ecx,2
        jb      .LLoadVectorASingleRemainingByte
        vpinsrw xmm0,xmm0,WORD PTR [rdi],0
        je      .LBroadcastVectorARemainingBytes
        vpinsrb xmm0,xmm0,BYTE PTR [rdi+2],2
        jmp     .LBroadcastVectorARemainingBytes

.LLoadVectorASingleRemainingByte:
        vpinsrb xmm0,xmm0,BYTE PTR [rdi],0

.LBroadcastVectorARemainingBytes:
        vpbroadcastd zmm28,xmm0             # broadcast values

//
// Process sets of 16 columns from the remaining rows in a loop or process the
// remaining 1 to 15 columns.
//

.LProcessColumnLoopSmallKBy16:
        vpxor   xmm3,xmm3,xmm3              # clear optional row vectors
        vpxor   xmm4,xmm4,xmm4
        vpxor   xmm5,xmm5,xmm5
        cmp     r8d,16
        jb      .LLoadPartialVectorSmallKBySmallN
        vmovdqu xmm2,XMMWORD PTR [rsi]
        cmp     ecx,2
        jb      .LComputeOutputSmallKBy16
        vmovdqu xmm3,XMMWORD PTR [rsi+r9]
        je      .LComputeOutputSmallKBy16
        vmovdqu xmm4,XMMWORD PTR [rsi+r9*2]
        jmp     .LComputeOutputSmallKBy16

.LLoadPartialVectorSmallKBySmallN:
        vmovdqu8 zmm2{k1}{z},ZMMWORD PTR [rsi]
        cmp     ecx,2
        jb      .LComputeOutputSmallKBy16
        vmovdqu8 zmm3{k1}{z},ZMMWORD PTR [rsi+r9]
        je      .LComputeOutputSmallKBy16
        vmovdqu8 zmm4{k1}{z},ZMMWORD PTR [rsi+r9*2]
        jmp     .LComputeOutputSmallKBy16

.LComputeOutputSmallKBy16:
        vpunpcklbw xmm1,xmm2,xmm3           # interleave row data bytes
        vpunpckhbw xmm2,xmm2,xmm3
        vpunpcklbw xmm3,xmm4,xmm5
        vpunpckhbw xmm4,xmm4,xmm5
        vpunpcklwd xmm5,xmm1,xmm3           # interleave row data words
        vpunpckhwd xmm1,xmm1,xmm3
        vpunpcklwd xmm3,xmm2,xmm4
        vpunpckhwd xmm2,xmm2,xmm4
        vinserti128 ymm5,ymm5,xmm1,1        # concatenate 256-bit vector
        vinserti128 ymm3,ymm3,xmm2,1
        vshufi32x4 zmm16,zmm5,zmm3,0x44     # concatenate 512-bit vector
.ifeqs "\Isa\()", "Avx512Core"
        vpmaddubsw zmm16,zmm28,zmm16
        vpmaddwd zmm20,zmm16,zmm29
.else
        vpxord zmm20,zmm20,zmm20
        VpdpbusdsZmmZmmZmm zmm20,zmm28,zmm16
.endif
        cmp     r8d,16
        jb      .LStorePartialVectorSmallKBySmallN
        test    r11,r11                     # ZeroMode?
        jnz     .LSkipAccumulateOutputSmallKBy16
        vpaddd  zmm20,zmm20,ZMMWORD PTR [r10]

.LSkipAccumulateOutputSmallKBy16:
        vmovdqu32 ZMMWORD PTR [r10],zmm20
        add     rsi,16                      # advance matrix B by 16 bytes
        add     r10,16*4                    # advance matrix C by 16 columns
        sub     r8d,16                      # decrement CountN
        jnz     .LProcessColumnLoopSmallKBy16
        jmp     .LExitKernel

.LStorePartialVectorSmallKBySmallN:
        test    r11,r11                     # ZeroMode?
        jnz     .LSkipAccumulateOutputSmallKBySmallN
        vpaddd  zmm20{k1}{z},zmm20,ZMMWORD PTR [r10]

.LSkipAccumulateOutputSmallKBySmallN:
        vmovdqu32 ZMMWORD PTR [r10]{k1},zmm20
        jmp     .LExitKernel

        .endm
