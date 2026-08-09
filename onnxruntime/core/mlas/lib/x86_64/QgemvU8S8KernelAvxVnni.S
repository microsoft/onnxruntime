/*++

Copyright (c) 2020 Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemvU8S8KernelAvxVnni.s

Abstract:

    This module implements the kernels for the quantized integer matrix/vector
    multiply operation (QGEMV).

    This implementation uses AVXVNNI instructions.

--*/

#include "asmmacro.h"
#include "AssembleAvxVnni.h"

        .intel_syntax noprefix

//
// Stack frame layout for the U8S8 kernel.
//

        .equ    .LGemvU8S8KernelFrame_mask, -8
        .equ    .LGemvU8S8KernelFrame_SavedRbx, 0
        .equ    .LGemvU8S8KernelFrame_SavedRbp, 8
        .equ    .LGemvU8S8KernelFrame_ReturnAddress, 16

        .text

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

        .globl  C_UNDERSCORE(MlasGemvU8S8KernelAvxVnni)
C_UNDERSCORE(MlasGemvU8S8KernelAvxVnni):

        push    rbp
        push    rbx

        mov     r10,rdx
        mov     r11,rsp                     # set ZeroMode to any non-zero value

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
        vpbroadcastd ymm0,DWORD PTR [rdi]
        add     rdi,4                       # advance matrix A by 4 bytes

//
// Process sets of 32 columns from the 4 rows in a loop.
//
// Some permute operations are deferred until the final store of the 4x32 block
// as these permutes are expensive.
//

.LProcessColumnLoop4By32:
        cmp     rbp,32
        jb      .LProcessColumnLoop4By8
        lea     rax,[rdx+r9*2]              # compute matrix B plus 2 rows
        vmovdqu ymm2,YMMWORD PTR [rdx]
        vmovdqu ymm3,YMMWORD PTR [rdx+r9]
        vmovdqu ymm4,YMMWORD PTR [rax]
        vmovdqu ymm5,YMMWORD PTR [rax+r9]
        vpunpcklbw ymm1,ymm2,ymm3           # interleave row data bytes
        vpunpckhbw ymm2,ymm2,ymm3
        vpxor ymm7,ymm7,ymm7
        vpunpcklbw ymm3,ymm4,ymm5
        vpunpckhbw ymm4,ymm4,ymm5
        vpxor ymm8,ymm8,ymm8
        vpunpcklwd ymm5,ymm1,ymm3           # interleave row data words
        vpunpckhwd ymm1,ymm1,ymm3
        vpxor ymm9,ymm9,ymm9
        vpunpcklwd ymm3,ymm2,ymm4
        vpunpckhwd ymm2,ymm2,ymm4
        vpxor ymm10,ymm10,ymm10
        VpdpbusdsYmmYmmYmm ymm7,ymm0,ymm5
        VpdpbusdsYmmYmmYmm ymm8,ymm0,ymm1
        VpdpbusdsYmmYmmYmm ymm9,ymm0,ymm3
        VpdpbusdsYmmYmmYmm ymm10,ymm0,ymm2
        test    r11,r11                     # ZeroMode?
        jnz     .LSkipAccumulateOutput4By32
        vpaddd  ymm7,ymm7,YMMWORD PTR [rbx]
        vpaddd  ymm8,ymm8,YMMWORD PTR [rbx+32]
        vpaddd  ymm9,ymm9,YMMWORD PTR [rbx+64]
        vpaddd  ymm10,ymm10,YMMWORD PTR [rbx+96]

.LSkipAccumulateOutput4By32:
        cmp     rcx,4                       # final 4x32 block?
        jae     .LStoreOutput4By32
        vperm2i128 ymm4,ymm7,ymm8,0x31      # interleave vector results
        vperm2i128 ymm7,ymm7,ymm8,0x20
        vperm2i128 ymm8,ymm9,ymm10,0x20
        vperm2i128 ymm10,ymm9,ymm10,0x31
        vmovaps ymm9,ymm4

.LStoreOutput4By32:
        vmovdqu YMMWORD PTR [rbx],ymm7
        vmovdqu YMMWORD PTR [rbx+32],ymm8
        vmovdqu YMMWORD PTR [rbx+64],ymm9
        vmovdqu YMMWORD PTR [rbx+96],ymm10
        add     rdx,32                      # advance matrix B by 32 bytes
        add     rbx,32*4                    # advance matrix C by 32 columns
        sub     rbp,32                      # decrement CountN
        jnz     .LProcessColumnLoop4By32

.LAdvanceRowLoop4:
        xor     r11,r11                     # clear ZeroMode
        sub     rcx,4                       # decrement CountK
        jae     .LProcessRowLoop4

.LProcessRemainingRows:
        add     rcx,4                       # correct for over-subtract above
        jnz     .LProcessRemainingSmallK

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        vzeroupper

        pop     rbx
        pop     rbp
        ret

//
// Process sets of 8 columns from the 4 rows in a loop.
//

.LProcessColumnLoop4By8:
        cmp     ebp,8
        jb      .LProcessColumn4By4
        lea     rax,[rdx+r9*2]              # compute matrix B plus 2 rows
        vmovq   xmm2,QWORD PTR [rdx]
        vmovq   xmm3,QWORD PTR [rdx+r9]
        vmovq   xmm4,QWORD PTR [rax]
        vmovq   xmm5,QWORD PTR [rax+r9]
        vpunpcklbw xmm2,xmm2,xmm3           # interleave row data bytes
        vpunpcklbw xmm4,xmm4,xmm5
        vpunpcklwd xmm1,xmm2,xmm4           # interleave row data words
        vpunpckhwd xmm2,xmm2,xmm4
        vinserti128 ymm1,ymm1,xmm2,1        # concatenate vector
        vpxor ymm8,ymm8,ymm8
        VpdpbusdsYmmYmmYmm ymm8,ymm0,ymm1
        test    r11,r11                     # ZeroMode?
        jnz     .LSkipAccumulateOutput4By8
        vpaddd  ymm8,ymm8,YMMWORD PTR [rbx]

.LSkipAccumulateOutput4By8:
        vmovdqu YMMWORD PTR [rbx],ymm8
        add     rdx,8                       # advance matrix B by 8 bytes
        add     rbx,8*4                     # advance matrix C by 8 columns
        sub     ebp,8                       # decrement CountN
        jnz     .LProcessColumnLoop4By8
        jmp     .LAdvanceRowLoop4

//
// Process a set of 4 columns from the 4 rows.
//

.LProcessColumn4By4:
        test    ebp,4                       # (CountN & 4) != 0?
        jz      .LProcessColumn4BySmallN
        lea     rax,[rdx+r9*2]              # compute matrix B plus 2 rows
        vmovd   xmm1,DWORD PTR [rdx]
        vpinsrd xmm1,xmm1,DWORD PTR [rdx+r9],1
        vpinsrd xmm1,xmm1,DWORD PTR [rax],2
        vpinsrd xmm1,xmm1,DWORD PTR [rax+r9],3
        vpshufb xmm1,xmm1,XMMWORD PTR C_UNDERSCORE(MlasTranspose4x4BytesAvx)[rip]
        vpxor xmm8,xmm8,xmm8
        VpdpbusdsXmmXmmXmm xmm8,xmm0,xmm1
        test    r11,r11                     # ZeroMode?
        jnz     .LSkipAccumulateOutput4By4
        vpaddd  xmm8,xmm8,XMMWORD PTR [rbx]

.LSkipAccumulateOutput4By4:
        vmovdqu XMMWORD PTR [rbx],xmm8
        and     ebp,3                       # (CountN & 3) != 0?
        jz      .LAdvanceRowLoop4
        add     rdx,4                       # advance matrix B by 4 bytes
        add     rbx,4*4                     # advance matrix C by 4 columns

//
// Process the remaining 1 to 3 columns from the 4 rows.
//

.LProcessColumn4BySmallN:
        mov     DWORD PTR .LGemvU8S8KernelFrame_mask[rsp],ebp
        vbroadcastss xmm2,DWORD PTR .LGemvU8S8KernelFrame_mask[rsp]
        vpcmpgtd xmm2,xmm2,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]
        vpxor   xmm1,xmm1,xmm1
        lea     rax,[rdx+r9*2]              # compute matrix B plus 2 rows
        cmp     ebp,2                       # (CountN & 2) != 0?
        jb      .LProcessColumn4By1
        vpinsrw xmm1,xmm1,WORD PTR [rdx],0
        vpinsrw xmm1,xmm1,WORD PTR [rdx+r9],2
        vpinsrw xmm1,xmm1,WORD PTR [rax],4
        vpinsrw xmm1,xmm1,WORD PTR [rax+r9],6
        je      .LComputeOutput4BySmallN
        vpinsrb xmm1,xmm1,BYTE PTR [rdx+2],2
        vpinsrb xmm1,xmm1,BYTE PTR [rdx+r9+2],6
        vpinsrb xmm1,xmm1,BYTE PTR [rax+2],10
        vpinsrb xmm1,xmm1,BYTE PTR [rax+r9+2],14
        jmp     .LComputeOutput4BySmallN

.LProcessColumn4By1:
        vpinsrb xmm1,xmm1,BYTE PTR [rdx],0
        vpinsrb xmm1,xmm1,BYTE PTR [rdx+r9],4
        vpinsrb xmm1,xmm1,BYTE PTR [rax],8
        vpinsrb xmm1,xmm1,BYTE PTR [rax+r9],12

.LComputeOutput4BySmallN:
        vpshufb xmm1,xmm1,XMMWORD PTR C_UNDERSCORE(MlasTranspose4x4BytesAvx)[rip]
        vpxor xmm8,xmm8,xmm8
        VpdpbusdsXmmXmmXmm xmm8,xmm0,xmm1
        test    r11,r11                     # ZeroMode?
        jnz     .LStoreOutput4BySmallN
        vpmaskmovd xmm3,xmm2,XMMWORD PTR [rbx]
        vpaddd  xmm8,xmm8,xmm3

.LStoreOutput4BySmallN:
        vpmaskmovd XMMWORD PTR [rbx],xmm2,xmm8
        jmp     .LAdvanceRowLoop4

//
// Broadcast the remaining 1 to 3 values from vector A.
//

.LProcessRemainingSmallK:
        vpxor   xmm5,xmm5,xmm5              # keep zero vector for vpinsrb/vpinsrw
        cmp     ecx,2
        jb      .LLoadVectorASingleRemainingByte
        vpinsrw xmm0,xmm5,WORD PTR [rdi],0
        je      .LBroadcastVectorARemainingBytes
        vpinsrb xmm0,xmm0,BYTE PTR [rdi+2],2
        jmp     .LBroadcastVectorARemainingBytes

.LLoadVectorASingleRemainingByte:
        vpinsrb xmm0,xmm5,BYTE PTR [rdi],0

.LBroadcastVectorARemainingBytes:
        vpshufd xmm0,xmm0,0                 # broadcast values

//
// Process a set of 4 columns from the remaining rows.
//

.LProcessColumnLoopSmallKBy4:
        cmp     r8d,4
        jb      .LProcessColumnLoopSmallKBySmallN
        vmovd   xmm1,DWORD PTR [rsi]
        cmp     ecx,2
        jb      .LComputeOutputSmallKBy4
        vpinsrd xmm1,xmm1,DWORD PTR [rsi+r9],1
        je      .LComputeOutputSmallKBy4
        vpinsrd xmm1,xmm1,DWORD PTR [rsi+r9*2],2

.LComputeOutputSmallKBy4:
        vpshufb xmm1,xmm1,XMMWORD PTR C_UNDERSCORE(MlasTranspose4x4BytesAvx)[rip]
        vpxor xmm8,xmm8,xmm8
        VpdpbusdsXmmXmmXmm xmm8,xmm0,xmm1
        test    r11,r11                     # ZeroMode?
        jnz     .LSkipAccumulateOutputSmallKBy4
        vpaddd  xmm8,xmm8,XMMWORD PTR [r10]

.LSkipAccumulateOutputSmallKBy4:
        vmovdqu XMMWORD PTR [r10],xmm8
        add     rsi,4                       # advance matrix B by 4 bytes
        add     r10,4*4                     # advance matrix C by 4 columns
        sub     r8d,4                       # decrement CountN
        jnz     .LProcessColumnLoopSmallKBy4
        jmp     .LExitKernel

//
// Process the remaining 1 to 3 columns from the remaining rows.
//
// Single step through each of the columns to keep code size small for the
// uncommon path (typically the row count is a multiple of 4).
//

.LProcessColumnLoopSmallKBySmallN:
        vpinsrb xmm1,xmm5,BYTE PTR [rsi],0
        cmp     ecx,2
        jb      .LComputeOutputSmallKBySmallN
        vpinsrb xmm1,xmm1,BYTE PTR [rsi+r9],1
        je      .LComputeOutputSmallKBySmallN
        vpinsrb xmm1,xmm1,BYTE PTR [rsi+r9*2],2

.LComputeOutputSmallKBySmallN:
        vpxor xmm8,xmm8,xmm8
        VpdpbusdsXmmXmmXmm xmm8,xmm0,xmm1
        test    r11,r11                     # ZeroMode?
        jnz     .LSkipAccumulateOutputSmallKBySmallN
        vmovd   xmm3,DWORD PTR [r10]
        vpaddd  xmm8,xmm8,xmm3

.LSkipAccumulateOutputSmallKBySmallN:
        vmovd   DWORD PTR [r10],xmm8
        inc     rsi                         # advance matrix B by 1 byte
        add     r10,4                       # advance matrix C by 1 column
        dec     r8
        jnz     .LProcessColumnLoopSmallKBySmallN
        jmp     .LExitKernel

        .end
