/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8U8KernelAvx2.S

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

        .text

//
// Stack frame layout for the U8U8 CopyPackA routine.
//

        .equ    .LGemmU8U8CopyPackAFrame_PaddedMatrixAData, -72
        .equ    .LGemmU8U8CopyPackAFrame_mask, -8
        .equ    .LGemmU8U8CopyPackAFrame_SavedR13, 0
        .equ    .LGemmU8U8CopyPackAFrame_SavedR12, 8
        .equ    .LGemmU8U8CopyPackAFrame_SavedRbx, 16
        .equ    .LGemmU8U8CopyPackAFrame_SavedRbp, 24
        .equ    .LGemmU8U8CopyPackAFrame_ReturnAddress, 32
        .equ    .LGemmU8U8CopyPackAFrame_offb, 40

//
// Stack frame layout for the U8U8 CopyPackB routine.
//

        .equ    .LGemmU8U8CopyPackBFrame_PaddedMatrixBData, -40
        .equ    .LGemmU8U8CopyPackBFrame_Padding, -8
        .equ    .LGemmU8U8CopyPackBFrame_SavedRbx, 0
        .equ    .LGemmU8U8CopyPackBFrame_SavedRbp, 8
        .equ    .LGemmU8U8CopyPackBFrame_ReturnAddress, 16
        .equ    .LGemmU8U8CopyPackBFrame_offa, 24

//
// Stack frame layout for the U8U8 kernel.
//

        .equ    .LGemmU8U8KernelFrame_mask, -8
        .equ    .LGemmU8U8KernelFrame_SavedR14, 0
        .equ    .LGemmU8U8KernelFrame_SavedR13, 8
        .equ    .LGemmU8U8KernelFrame_SavedR12, 16
        .equ    .LGemmU8U8KernelFrame_SavedRbx, 24
        .equ    .LGemmU8U8KernelFrame_SavedRbp, 32
        .equ    .LGemmU8U8KernelFrame_ReturnAddress, 40
        .equ    .LGemmU8U8KernelFrame_ldc, 48
        .equ    .LGemmU8U8KernelFrame_RowSumVector, 56
        .equ    .LGemmU8U8KernelFrame_ColumnSumVector, 64
        .equ    .LGemmU8U8KernelFrame_DepthValue, 72
        .equ    .LGemmU8U8KernelFrame_ZeroMode, 80

/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

    The kernel expects that elements from matrix A have been zero extended to
    16-bits and padded to a multiple of 32-bits (two pairs of 16-bit values).
    The kernel can then efficiently broadcast 32-bits from the packed buffer
    and avoid expensive shuffling inside the kernel.

Arguments:

    D (rdi) - Supplies the address of the destination packed buffer.

    A (rsi) - Supplies the address of the source matrix.

    lda (rdx) - Supplies the number of elements per row of the source matrix.

    CountM (rcx) - Supplies the number of rows of the source matrix to copy.

    CountK (r8) - Supplies the number of columns of the source matrix to copy.

    RowSumVector (r9) - Supplies the address of the buffer to receive the sums
        of the elements from each of the rows. Each sum has also been multiplied
        by the zero point offset.

    offb - Supplies the zero point offset for the other source matrix of the
        matrix multiplication.

Return Value:

    None.

--*/

        .globl  C_UNDERSCORE(MlasGemmU8U8CopyPackAAvx2)
C_UNDERSCORE(MlasGemmU8U8CopyPackAAvx2):

        push    rbp
        push    rbx
        push    r12
        push    r13

        mov     r10,rdx
        mov     r11,rcx
        lea     r12,[r8+1]
        and     r12,NOT 1                   # align CountK up to pair count
        vpbroadcastw xmm8,WORD PTR .LGemmU8U8CopyPackAFrame_offb[rsp]

//
// Compute the conditional load/store mask for an unaligned CountK.
//

        mov     eax,r8d
        and     eax,15                      # isolate unaligned count
        inc     eax
        shr     eax,1                       # align unaligned count to pair count
        mov     DWORD PTR .LGemmU8U8CopyPackAFrame_mask[rsp],eax
        mov     rbp,QWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)@GOTPCREL[rip]
        vpbroadcastd ymm9,DWORD PTR .LGemmU8U8CopyPackAFrame_mask[rsp]
        vpcmpgtd ymm9,ymm9,YMMWORD PTR [rbp]

//
// Zero initialize the padded stack buffers.
//

        vpxor   xmm0,xmm0,xmm0
        vmovdqu YMMWORD PTR .LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp],ymm0
        vmovdqu YMMWORD PTR .LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp+32],ymm0

//
// Process 4 rows of matrix A in a loop.
//
// For each row, zero extend the source bytes to 16-bits and write to the packed
// buffer. The packed buffer has the same data ordering as the source bytes, but
// the stride is CountK aligned up to an even number of 16-bit values.
//
// These 16-bit values are also accumulated into an intermediate per-row
// accumulator. CountK cannot be greater than 256 to avoid overflowing these
// 16-bit accumulators.
//

        sub     r11,4
        jb      .LCopyPackA.ProcessRemainingRows

.LCopyPackA.ProcessNextRowM4:
        vpxor   xmm0,xmm0,xmm0              # clear row accumulators
        vpxor   xmm1,xmm1,xmm1
        vpxor   xmm2,xmm2,xmm2
        vpxor   xmm3,xmm3,xmm3
        mov     rdx,rsi
        mov     rcx,rdi
        lea     rsi,[rsi+r10*4]             # advance next matrix A by 4 rows
        lea     rdi,[rdi+r12*(2*4)]         # advance next matrix D by 4 rows
        mov     rbx,r8                      # reload columns remaining
        sub     rbx,16
        jb      .LCopyPackA.ProcessRemainingColumnsM4

.LCopyPackA.ProcessNextColumnLoopM4:
        lea     rax,[rdx+r10*2]             # compute matrix A plus two rows
        vpmovzxbw ymm4,XMMWORD PTR [rdx]
        vpmovzxbw ymm5,XMMWORD PTR [rdx+r10]
        vpmovzxbw ymm6,XMMWORD PTR [rax]
        vpmovzxbw ymm7,XMMWORD PTR [rax+r10]
        lea     rax,[rcx+r12*4]             # compute matrix D plus two rows
        vmovdqu YMMWORD PTR [rcx],ymm4
        vmovdqu YMMWORD PTR [rcx+r12*2],ymm5
        vmovdqu YMMWORD PTR [rax],ymm6
        vmovdqu YMMWORD PTR [rax+r12*2],ymm7
        vpaddw  ymm0,ymm0,ymm4              # accumulate per row along columns
        vpaddw  ymm1,ymm1,ymm5
        vpaddw  ymm2,ymm2,ymm6
        vpaddw  ymm3,ymm3,ymm7
        add     rdx,16                      # advance matrix A by 16 bytes
        add     rcx,16*2                    # advance matrix D by 16 words
        sub     rbx,16                      # subtract columns remaining
        jae     .LCopyPackA.ProcessNextColumnLoopM4

.LCopyPackA.ProcessRemainingColumnsM4:
        add     rbx,16                      # correct for over-subtract above
        jz      .LCopyPackA.ReduceRowSumVectorM4

//
// Copy the unaligned CountK columns to a zero padded stack buffer.
//

        lea     rbp,.LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp]
        test    bl,8                        # (CountK & 8) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan8M4
        lea     r13,[rdx+r10*2]             # compute matrix A plus two rows
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        mov     rax,QWORD PTR [rdx+r10]
        mov     QWORD PTR [rbp+16],rax
        mov     rax,QWORD PTR [r13]
        mov     QWORD PTR [rbp+32],rax
        mov     rax,QWORD PTR [r13+r10]
        mov     QWORD PTR [rbp+48],rax
        add     rdx,8
        add     rbp,8                       # advance padded buffer destination

.LCopyPackA.CopyRemainingCountKLessThan8M4:
        test    bl,4                        # (CountK & 4) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan4M4
        lea     r13,[rdx+r10*2]             # compute matrix A plus two rows
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        mov     eax,DWORD PTR [rdx+r10]
        mov     DWORD PTR [rbp+16],eax
        mov     eax,DWORD PTR [r13]
        mov     DWORD PTR [rbp+32],eax
        mov     eax,DWORD PTR [r13+r10]
        mov     DWORD PTR [rbp+48],eax
        add     rdx,4
        add     rbp,4                       # advance padded buffer destination

.LCopyPackA.CopyRemainingCountKLessThan4M4:
        test    bl,2                        # (CountK & 2) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan2M4
        lea     r13,[rdx+r10*2]             # compute matrix A plus two rows
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        movzx   eax,WORD PTR [rdx+r10]
        mov     WORD PTR [rbp+16],ax
        movzx   eax,WORD PTR [r13]
        mov     WORD PTR [rbp+32],ax
        movzx   eax,WORD PTR [r13+r10]
        mov     WORD PTR [rbp+48],ax
        add     rdx,2
        add     rbp,2                       # advance padded buffer destination

.LCopyPackA.CopyRemainingCountKLessThan2M4:
        test    bl,1                        # (CountK & 1) != 0?
        jz      .LCopyPackA.ProcessPaddedMatrixADataM4
        lea     r13,[rdx+r10*2]             # compute matrix A plus two rows
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al
        movzx   eax,BYTE PTR [rdx+r10]
        mov     BYTE PTR [rbp+16],al
        movzx   eax,BYTE PTR [r13]
        mov     BYTE PTR [rbp+32],al
        movzx   eax,BYTE PTR [r13+r10]
        mov     BYTE PTR [rbp+48],al

//
// Process the remaining CountK columns using the zero padded stack buffer.
//

.LCopyPackA.ProcessPaddedMatrixADataM4:
        vpmovzxbw ymm4,XMMWORD PTR .LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp]
        vpmovzxbw ymm5,XMMWORD PTR .LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp+16]
        vpmovzxbw ymm6,XMMWORD PTR .LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp+32]
        vpmovzxbw ymm7,XMMWORD PTR .LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp+48]
        lea     rax,[rcx+r12*4]             # compute matrix D plus two rows
        vpmaskmovd YMMWORD PTR [rcx],ymm9,ymm4
        vpmaskmovd YMMWORD PTR [rcx+r12*2],ymm9,ymm5
        vpmaskmovd YMMWORD PTR [rax],ymm9,ymm6
        vpmaskmovd YMMWORD PTR [rax+r12*2],ymm9,ymm7
        vpaddw  ymm0,ymm0,ymm4              # accumulate per row along columns
        vpaddw  ymm1,ymm1,ymm5
        vpaddw  ymm2,ymm2,ymm6
        vpaddw  ymm3,ymm3,ymm7

//
// Reduce the sums for the four rows of output. Transpose the intermediate
// accumulators by treating the registers as 32-bit elements containing a pair
// of 16-bit sums. Continue reducing the transposed accumulators to produce the
// final 32-bit vector output.
//

.LCopyPackA.ReduceRowSumVectorM4:
        vpunpckldq ymm4,ymm0,ymm1           # [A5 B5 A4 B4 A1 B1 A0 B0]
        vpunpckhdq ymm5,ymm0,ymm1           # [A7 B7 A6 B6 A3 B3 A2 B2]
        vpunpckldq ymm6,ymm2,ymm3           # [C5 D5 C4 D4 C1 D1 C0 D0]
        vpunpckhdq ymm7,ymm2,ymm3           # [C7 D7 C6 D6 C3 D3 C2 D2]
        vpunpcklqdq ymm0,ymm4,ymm6          # [A4 B4 C4 D4 A0 B0 C0 D0]
        vpunpckhqdq ymm1,ymm4,ymm6          # [A5 B5 C5 D5 A1 B1 C1 D1]
        vpunpcklqdq ymm2,ymm5,ymm7          # [A6 B6 C6 D6 A2 B2 C2 D2]
        vpunpckhqdq ymm3,ymm5,ymm7          # [A7 B7 C7 D7 A3 B3 C3 D3]
        vpaddw  ymm0,ymm0,ymm1              # reduction
        vpaddw  ymm0,ymm0,ymm2
        vpaddw  ymm0,ymm0,ymm3
        vextracti128 xmm1,ymm0,1            # extract high pairs
        vpaddw  xmm0,xmm0,xmm1              # reduction
        vpmaddwd xmm0,xmm0,xmm8             # multiply by offset and reduce
        vmovdqu XMMWORD PTR [r9],xmm0
        add     r9,4*4                      # advance row sum vector by 4 dwords
        sub     r11,4                       # subtract rows remaining
        jae     .LCopyPackA.ProcessNextRowM4

.LCopyPackA.ProcessRemainingRows:
        add     r11,4                       # correct for over-subtract above
        jz      .LCopyPackA.ExitRoutine

//
// Process a single row of matrix A in a loop.
//

.LCopyPackA.ProcessNextRowM1:
        vpxor   xmm0,xmm0,xmm0              # clear row accumulator
        mov     rdx,rsi
        mov     rcx,rdi
        add     rsi,r10
        lea     rdi,[rdi+r12*2]
        mov     rbx,r8                      # reload columns remaining
        sub     rbx,16
        jb      .LCopyPackA.ProcessRemainingColumnsM1

.LCopyPackA.ProcessNextColumnLoopM1:
        vpmovzxbw ymm4,XMMWORD PTR [rdx]
        vmovdqu YMMWORD PTR [rcx],ymm4
        vpaddw  ymm0,ymm0,ymm4              # accumulate per row along columns
        add     rdx,16                      # advance matrix A by 16 bytes
        add     rcx,16*2                    # advance matrix D by 16 words
        sub     rbx,16                      # subtract columns remaining
        jae     .LCopyPackA.ProcessNextColumnLoopM1

.LCopyPackA.ProcessRemainingColumnsM1:
        add     rbx,16                      # correct for over-subtract above
        jz      .LCopyPackA.ReduceRowSumVectorM1

//
// Copy the unaligned CountK columns to a zero padded stack buffer.
//

        lea     rbp,.LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp]
        test    bl,8                        # (CountK & 8) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan8M1
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        add     rdx,8
        add     rbp,8                       # advance padded buffer destination

.LCopyPackA.CopyRemainingCountKLessThan8M1:
        test    bl,4                        # (CountK & 4) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan4M1
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        add     rdx,4
        add     rbp,4                       # advance padded buffer destination

.LCopyPackA.CopyRemainingCountKLessThan4M1:
        test    bl,2                        # (CountK & 2) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan2M1
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        add     rdx,2
        add     rbp,2                       # advance padded buffer destination

.LCopyPackA.CopyRemainingCountKLessThan2M1:
        test    bl,1                        # (CountK & 1) != 0?
        jz      .LCopyPackA.ProcessPaddedMatrixADataM1
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al

//
// Process the remaining CountK columns using the zero padded stack buffer.
//

.LCopyPackA.ProcessPaddedMatrixADataM1:
        vpmovzxbw ymm4,XMMWORD PTR .LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp]
        vpmaskmovd YMMWORD PTR [rcx],ymm9,ymm4
        vpaddw  ymm0,ymm0,ymm4              # accumulate per row along columns

//
// Reduce the sum for the single row of output.
//

.LCopyPackA.ReduceRowSumVectorM1:
        vextracti128 xmm1,ymm0,1            # extract high pairs
        vpaddw  xmm0,xmm0,xmm1              # reduction
        vphaddw xmm0,xmm0,xmm0
        vphaddw xmm0,xmm0,xmm0
        vpmaddwd xmm0,xmm0,xmm8             # multiply by offset and reduce
        vmovd   DWORD PTR [r9],xmm0
        add     r9,4                        # advance row sum vector by 1 DWORD
        dec     r11                         # decrement rows remaining
        jnz     .LCopyPackA.ProcessNextRowM1

//
// Restore non-volatile registers and return.
//

.LCopyPackA.ExitRoutine:
        vzeroupper

        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret

/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

Arguments:

    D (rdi) - Supplies the address of the destination packed buffer.

    B (rsi) - Supplies the address of the source matrix.

    ldb (rdx) - Supplies the number of elements per row of the source matrix.

    CountN (rcx) - Supplies the number of columns of the source matrix to copy.

    CountK (r8) - Supplies the number of rows of the source matrix to copy.

    ColumnSumVector (r9) - Supplies the address of the buffer to receive the sums
        of the elements from each of the columns. Each sum has also been
        multiplied by the zero point offset.

    offa - Supplies the zero point offset for the other source matrix of the
        matrix multiplication.

Return Value:

    None.

--*/

        .globl  C_UNDERSCORE(MlasGemmU8U8CopyPackBAvx2)
C_UNDERSCORE(MlasGemmU8U8CopyPackBAvx2):

        push    rbp
        push    rbx

        mov     r10,rdx
        mov     r11,rcx
        vpbroadcastw ymm5,WORD PTR .LGemmU8U8CopyPackBFrame_offa[rsp]

//
// Zero initialize the padded stack buffers.
//

        vpxor   xmm0,xmm0,xmm0
        vmovdqu YMMWORD PTR .LGemmU8U8CopyPackBFrame_PaddedMatrixBData[rsp],ymm0

//
// Process 16 columns of matrix B in a loop.
//

        sub     r11,16
        jb      .LCopyPackB.ProcessRemainingColumns

.LCopyPackB.ProcessNextColumnN16:
        vpxor   xmm0,xmm0,xmm0              # clear column accumulators
        vpxor   xmm1,xmm1,xmm1
        mov     rdx,rsi
        add     rsi,16                      # advance next matrix B by 16 columns
        mov     rbx,r8                      # reload rows remaining
        sub     rbx,2
        jb      .LCopyPackB.ProcessRemainingRowsN16

.LCopyPackB.ProcessNextRowLoopN16:
        vmovdqu xmm2,XMMWORD PTR [rdx]      # load two rows
        vmovdqu xmm3,XMMWORD PTR [rdx+r10]
        lea     rdx,[rdx+r10*2]             # advance matrix B by two rows
        vpunpcklbw xmm4,xmm2,xmm3           # interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vmovdqu XMMWORD PTR [rdi],xmm4      # store interleaved rows
        vmovdqu XMMWORD PTR [rdi+16],xmm3
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        add     rdi,32                      # advance matrix D by 32 bytes
        vpaddw  ymm0,ymm0,ymm4              # accumulate per column
        vpaddw  ymm1,ymm1,ymm3
        sub     rbx,2                       # subtract columns remaining
        jae     .LCopyPackB.ProcessNextRowLoopN16

.LCopyPackB.ProcessRemainingRowsN16:
        add     rbx,2                       # correct for over-subtract above
        jz      .LCopyPackB.ReduceColumnSumVectorN16
        vpmovzxbw ymm4,XMMWORD PTR [rdx]
        vmovdqu YMMWORD PTR [rdi],ymm4      # store interleaved rows
        vextracti128 xmm3,ymm4,1
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        vpaddw  ymm0,ymm0,ymm4              # accumulate per column
        vpaddw  ymm1,ymm1,ymm3
        add     rdi,32                      # advance matrix D by 32 bytes

.LCopyPackB.ReduceColumnSumVectorN16:
        vpmaddwd ymm0,ymm0,ymm5             # multiply by offset and reduce
        vpmaddwd ymm1,ymm1,ymm5             # multiply by offset and reduce
        vmovdqu YMMWORD PTR [r9],ymm0
        vmovdqu YMMWORD PTR [r9+32],ymm1
        add     r9,64                       # advance column sum vector by 16 dwords
        sub     r11,16                      # subtract columns remaining
        jae     .LCopyPackB.ProcessNextColumnN16

.LCopyPackB.ProcessRemainingColumns:
        add     r11,16                      # correct for over-subtract above
        jnz     .LCopyPackB.ProcessColumnNUnaligned

//
// Restore non-volatile registers and return.
//

.LCopyPackB.ExitRoutine:
        vzeroupper

        pop     rbx
        pop     rbp
        ret

//
// Process the remaining columns of matrix B.
//

.LCopyPackB.ProcessColumnNUnaligned:
        vpxor   xmm0,xmm0,xmm0              # clear column accumulators
        vpxor   xmm1,xmm1,xmm1
        sub     r8,2
        jb      .LCopyPackB.ProcessRemainingRowsNUnaligned

.LCopyPackB.ProcessNextRowLoopNUnaligned:
        mov     rdx,rsi
        lea     rbp,.LGemmU8U8CopyPackBFrame_PaddedMatrixBData[rsp]
        test    r11b,8                      # (CountN & 8) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan8K2
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        mov     rax,QWORD PTR [rdx+r10]
        mov     QWORD PTR [rbp+16],rax
        add     rdx,8                       # advance matrix B
        add     rbp,8                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan8K2:
        test    r11b,4                      # (CountN & 4) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan4K2
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        mov     eax,DWORD PTR [rdx+r10]
        mov     DWORD PTR [rbp+16],eax
        add     rdx,4                       # advance matrix B
        add     rbp,4                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan4K2:
        test    r11b,2                      # (CountN & 2) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan2K2
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        movzx   eax,WORD PTR [rdx+r10]
        mov     WORD PTR [rbp+16],ax
        add     rdx,2                       # advance matrix B
        add     rbp,2                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan2K2:
        test    r11b,1                      # (CountN & 1) != 0?
        jz      .LCopyPackB.ProcessPaddedMatrixBDataK2
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al
        movzx   eax,BYTE PTR [rdx+r10]
        mov     BYTE PTR [rbp+16],al

.LCopyPackB.ProcessPaddedMatrixBDataK2:
        vmovdqu xmm2,XMMWORD PTR .LGemmU8U8CopyPackBFrame_PaddedMatrixBData[rsp]
        vmovdqu xmm3,XMMWORD PTR .LGemmU8U8CopyPackBFrame_PaddedMatrixBData[rsp+16]
        vpunpcklbw xmm4,xmm2,xmm3           # interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vmovdqu XMMWORD PTR [rdi],xmm4      # store interleaved rows
        vmovdqu XMMWORD PTR [rdi+16],xmm3
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        vpaddw  ymm0,ymm0,ymm4              # accumulate per column
        vpaddw  ymm1,ymm1,ymm3
        lea     rsi,[rsi+r10*2]             # advance next matrix B by two rows
        add     rdi,32                      # advance matrix D by 32 bytes
        sub     r8,2                        # subtract columns remaining
        jae     .LCopyPackB.ProcessNextRowLoopNUnaligned

.LCopyPackB.ProcessRemainingRowsNUnaligned:
        add     r8,2
        jz      .LCopyPackB.ReduceColumnSumVectorNUnaligned
        mov     rdx,rsi
        lea     rbp,.LGemmU8U8CopyPackBFrame_PaddedMatrixBData[rsp]
        test    r11b,8                      # (CountN & 8) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan8K1
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        add     rdx,8                       # advance matrix B
        add     rbp,8                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan8K1:
        test    r11b,4                      # (CountN & 4) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan4K1
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        add     rdx,4                       # advance matrix B
        add     rbp,4                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan4K1:
        test    r11b,2                      # (CountN & 2) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan2K1
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        add     rdx,2                       # advance matrix B
        add     rbp,2                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan2K1:
        test    r11b,1                      # (CountN & 1) != 0?
        jz      .LCopyPackB.ProcessPaddedMatrixBDataK1
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al

.LCopyPackB.ProcessPaddedMatrixBDataK1:
        vpmovzxbw ymm4,XMMWORD PTR .LGemmU8U8CopyPackBFrame_PaddedMatrixBData[rsp]
        vmovdqu YMMWORD PTR [rdi],ymm4      # store interleaved rows
        vextracti128 xmm3,ymm4,1
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        vpaddw  ymm0,ymm0,ymm4              # accumulate per column
        vpaddw  ymm1,ymm1,ymm3

.LCopyPackB.ReduceColumnSumVectorNUnaligned:
        vpmaddwd ymm0,ymm0,ymm5             # multiply by offset and reduce
        vpmaddwd ymm1,ymm1,ymm5             # multiply by offset and reduce
        vmovdqu YMMWORD PTR [r9],ymm0
        vmovdqu YMMWORD PTR [r9+32],ymm1
        jmp     .LCopyPackB.ExitRoutine

/*++

Macro Description:

    This macro generates code to multiply and accumulator a single row of the
    output block.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    Vec1Reg - Supplies the high block accumulator register (when ColumnCount
        is 16).

    Vec2Reg - Supplies the low block accumulator register.

Implicit Arguments:

    ymm0 - Supplies the first vector loaded from matrix B.

    ymm1 - Supplies the second vector loaded from matrix B (when ColumnCount
        is 16).

    ymm2 - Supplies the broadcast value loaded from matrix A.

--*/

        .macro MultiplyAccumulateRow ColumnCount, Vec1Reg, Vec2Reg

.if \ColumnCount\() == 16
        vpmaddwd ymm3,ymm2,ymm0
        vpaddd  \Vec1Reg\(),\Vec1Reg\(),ymm3
        vpmaddwd ymm2,ymm2,ymm1
        vpaddd  \Vec2Reg\(),\Vec2Reg\(),ymm2
.else
        vpmaddwd ymm3,ymm2,ymm0
        vpaddd  \Vec2Reg\(),\Vec2Reg\(),ymm3
.endif

        .endm

/*++

Macro Description:

    This macro generates code to multiply and accumulate each row of the output
    block.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    RowCount - Supplies the number of rows to produce.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlock ColumnCount, RowCount, VectorOffset, BroadcastOffset

        vpmovzxbw ymm0,XMMWORD PTR [rsi+\VectorOffset\()]
        EmitIfCountGE \ColumnCount\(), 16, "vpmovzxbw ymm1,XMMWORD PTR [rsi+\VectorOffset\()+16]"
        EmitIfCountGE \RowCount\(), 1, "vpbroadcastd ymm2,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "MultiplyAccumulateRow \ColumnCount\(), ymm4, ymm5"
        EmitIfCountGE \RowCount\(), 2, "vpbroadcastd ymm2,DWORD PTR [rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "MultiplyAccumulateRow \ColumnCount\(), ymm6, ymm7"
        EmitIfCountGE \RowCount\(), 3, "vpbroadcastd ymm2,DWORD PTR [rdi+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "MultiplyAccumulateRow \ColumnCount\(), ymm8, ymm9"
        EmitIfCountGE \RowCount\(), 4, "vpbroadcastd ymm2,DWORD PTR [rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "MultiplyAccumulateRow \ColumnCount\(), ymm10, ymm11"
        EmitIfCountGE \RowCount\(), 5, "vpbroadcastd ymm2,DWORD PTR [rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 5, "MultiplyAccumulateRow \ColumnCount\(), ymm12, ymm13"
        EmitIfCountGE \RowCount\(), 6, "vpbroadcastd ymm2,DWORD PTR [rbx+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 6, "MultiplyAccumulateRow \ColumnCount\(), ymm14, ymm15"

        .endm

/*++

Macro Description:

    This macro generates code to produce an output block for a set of columns
    and rows.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    RowCount - Supplies the number of rows to produce.

Implicit Arguments:

    rax - Supplies the length in bytes of a row from matrix C.

    rdi - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of paired columns from matrix A and the number of
        paired rows from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    r12 - Supplies the address of the row sum vector.

    r13 - Supplies the address of the column sum vector.

--*/

        .macro ProduceOutputBlock ColumnCount, RowCount

//
// Initialize the accumulators with the sum of the global depth value constant,
// the column sums, and the row sums.
//

        vpbroadcastd ymm1,DWORD PTR .LGemmU8U8KernelFrame_DepthValue[rsp]
.if \ColumnCount\() == 16
        vpaddd  ymm0,ymm1,YMMWORD PTR [r13]
        vpaddd  ymm1,ymm1,YMMWORD PTR [r13+32]
        add     r13,16*4                    # advance ColumnSumVector by 16 columns
.else
        vpaddd  ymm1,ymm1,YMMWORD PTR [r13]
.endif
        EmitIfCountGE \RowCount\(), 1, "vpbroadcastd ymm5,DWORD PTR [r12]"
        EmitIfCountGE \RowCount\(), 2, "vpbroadcastd ymm7,DWORD PTR [r12+4]"
        EmitIfCountGE \RowCount\(), 3, "vpbroadcastd ymm9,DWORD PTR [r12+8]"
        EmitIfCountGE \RowCount\(), 4, "vpbroadcastd ymm11,DWORD PTR [r12+12]"
        EmitIfCountGE \RowCount\(), 5, "vpbroadcastd ymm13,DWORD PTR [r12+16]"
        EmitIfCountGE \RowCount\(), 6, "vpbroadcastd ymm15,DWORD PTR [r12+20]"
        EmitIfCount2GE \RowCount\(), 1, \ColumnCount\(), 16, "vpaddd ymm4,ymm5,ymm0"
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm5,ymm5,ymm1"
        EmitIfCount2GE \RowCount\(), 2, \ColumnCount\(), 16, "vpaddd ymm6,ymm7,ymm0"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm7,ymm7,ymm1"
        EmitIfCount2GE \RowCount\(), 3, \ColumnCount\(), 16, "vpaddd ymm8,ymm9,ymm0"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm9,ymm9,ymm1"
        EmitIfCount2GE \RowCount\(), 4, \ColumnCount\(), 16, "vpaddd ymm10,ymm11,ymm0"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm11,ymm11,ymm1"
        EmitIfCount2GE \RowCount\(), 5, \ColumnCount\(), 16, "vpaddd ymm12,ymm13,ymm0"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm13,ymm13,ymm1"
        EmitIfCount2GE \RowCount\(), 6, \ColumnCount\(), 16, "vpaddd ymm14,ymm15,ymm0"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm15,ymm15,ymm1"

//
// Iterate over PairedCountK elements from matrix A and matrix B.
//
// Unrolling the loop to do two iterations improves performance slightly at the
// cost of larger code size. Balance this by only unrolling for the common case
// of computing 16 columns for an even number of rows.
//

        mov     rbp,rcx                     # reload PairedCountK
.if \RowCount\() > 3
        lea     rbx,[r10*2+r10]
        add     rbx,rdi                     # compute matrix A plus 3 rows
.endif

.if (\ColumnCount\() == 16) && ((\RowCount\() & 1) == 0)
        sub     rbp,2
        jb      .LProcessRemainingBlocks.\ColumnCount\().\RowCount\()

.LComputeBlockLoop.\ColumnCount\().\RowCount\():
        ComputeBlock \ColumnCount\(), \RowCount\(), 0, 0
        ComputeBlock \ColumnCount\(), \RowCount\(), 32, 4
        add     rdi,2*4                     # advance matrix A by 2 pairs
.if \RowCount\() > 3
        add     rbx,2*4                     # advance matrix A plus 3 rows by 2 pairs
.endif
        add     rsi,2*32                    # advance matrix B by 64 columns
        sub     rbp,2                       # subtract pairs remaining
        jae     .LComputeBlockLoop.\ColumnCount\().\RowCount\()

.LProcessRemainingBlocks.\ColumnCount\().\RowCount\():
        add     rbp,2                       # correct for over-subtract above
        jz      .LComputeBlockLoopExit.\ColumnCount\().\RowCount\()
        ComputeBlock \ColumnCount\(), \RowCount\(), 0, 0
        add     rsi,32                      # advance matrix B by 32 columns
.else
.LComputeBlockLoop.\ColumnCount\().\RowCount\():
        ComputeBlock \ColumnCount\(), \RowCount\(), 0, 0
        add     rdi,4                       # advance matrix A by 1 pair
.if \RowCount\() > 3
        add     rbx,4                       # advance matrix A plus 3 rows by 1 pair
.endif
        add     rsi,32
        dec     rbp                         # decrement pairs remaining
        jnz     .LComputeBlockLoop.\ColumnCount\().\RowCount\()
.endif

.LComputeBlockLoopExit.\ColumnCount\().\RowCount\():
.if \RowCount\() > 3
        lea     rbx,[rdx+rax*2]             # compute matrix C plus 3 rows
        add     rbx,rax
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute matrix multiplication for a fixed set
    of rows.

Arguments:

    RowCount - Supplies the number of rows to process.

    Fallthrough - Supplies a non-blank value if the macro may fall through to
        the ExitKernel label.

Implicit Arguments:

    rax - Supplies the length in bytes of a row from matrix C.

    rdi - Supplies the address of matrix A.

    rsi - Supplies the address of matrix B.

    rdx - Supplies the address of matrix C.

    r11 - Supplies the address of matrix A.

    r9 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    rcx - Supplies the number of paired columns from matrix A and the number of
        paired rows from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    r12 - Supplies the address of the row sum vector.

    r13 - Supplies the address of the column sum vector.

    r14b - Supplies the zero mode flag.

--*/

        .macro ProcessCountM RowCount, Fallthrough

        cmp     r9,8
        jbe     .LProcessRemainingCountN.\RowCount\()

.LProcessNextColumnLoop16xN.\RowCount\():
        ProduceOutputBlock 16, \RowCount\()
        sub     r9,16
        jb      .LOutputMasked16xNBlock.\RowCount\()
        test    r14b,r14b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput16xNBlock.\RowCount\()
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm4,ymm4,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm5,ymm5,YMMWORD PTR [rdx+32]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm6,ymm6,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm7,ymm7,YMMWORD PTR [rdx+rax+32]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm8,ymm8,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm9,ymm9,YMMWORD PTR [rdx+rax*2+32]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm10,ymm10,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm11,ymm11,YMMWORD PTR [rbx+32]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm12,ymm12,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm13,ymm13,YMMWORD PTR [rbx+rax+32]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm14,ymm14,YMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm15,ymm15,YMMWORD PTR [rbx+rax*2+32]"

.LSkipAccumulateOutput16xNBlock.\RowCount\():
        EmitIfCountGE \RowCount\(), 1, "vmovdqu YMMWORD PTR [rdx],ymm4"
        EmitIfCountGE \RowCount\(), 1, "vmovdqu YMMWORD PTR [rdx+32],ymm5"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu YMMWORD PTR [rdx+rax],ymm6"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu YMMWORD PTR [rdx+rax+32],ymm7"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu YMMWORD PTR [rdx+rax*2],ymm8"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu YMMWORD PTR [rdx+rax*2+32],ymm9"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu YMMWORD PTR [rbx],ymm10"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu YMMWORD PTR [rbx+32],ymm11"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu YMMWORD PTR [rbx+rax],ymm12"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu YMMWORD PTR [rbx+rax+32],ymm13"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu YMMWORD PTR [rbx+rax*2],ymm14"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu YMMWORD PTR [rbx+rax*2+32],ymm15"
        add     rdx,16*4                    # advance matrix C by 16 columns
        mov     rdi,r11                     # reload matrix A
        cmp     r9,8
        ja      .LProcessNextColumnLoop16xN.\RowCount\()
        test    r9,r9
        jz      .LExitKernel

.LProcessRemainingCountN.\RowCount\():
        ProduceOutputBlock 8, \RowCount\()
        cmp     r9,8
        jb      .LOutputMasked8xNBlock.\RowCount\()
        test    r14b,r14b                   # ZeroMode?
        jnz     .LSkipAccumulateOutput8xNBlock.\RowCount\()
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm5,ymm5,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm7,ymm7,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm9,ymm9,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm11,ymm11,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm13,ymm13,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm15,ymm15,YMMWORD PTR [rbx+rax*2]"

.LSkipAccumulateOutput8xNBlock.\RowCount\():
        EmitIfCountGE \RowCount\(), 1, "vmovdqu YMMWORD PTR [rdx],ymm5"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu YMMWORD PTR [rdx+rax],ymm7"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu YMMWORD PTR [rdx+rax*2],ymm9"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu YMMWORD PTR [rbx],ymm11"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu YMMWORD PTR [rbx+rax],ymm13"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu YMMWORD PTR [rbx+rax*2],ymm15"
        jmp     .LExitKernel

.LOutputMasked16xNBlock.\RowCount\():
        test    r14b,r14b                   # ZeroMode?
        jnz     .LSkipAccumulateOutputMasked16xNBlock.\RowCount\()
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm4,ymm4,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm6,ymm6,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm8,ymm8,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm10,ymm10,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm12,ymm12,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm14,ymm14,YMMWORD PTR [rbx+rax*2]"

.LSkipAccumulateOutputMasked16xNBlock.\RowCount\():
        EmitIfCountGE \RowCount\(), 1, "vmovdqu YMMWORD PTR [rdx],ymm4"
        EmitIfCountGE \RowCount\(), 2, "vmovdqu YMMWORD PTR [rdx+rax],ymm6"
        EmitIfCountGE \RowCount\(), 3, "vmovdqu YMMWORD PTR [rdx+rax*2],ymm8"
        EmitIfCountGE \RowCount\(), 4, "vmovdqu YMMWORD PTR [rbx],ymm10"
        EmitIfCountGE \RowCount\(), 5, "vmovdqu YMMWORD PTR [rbx+rax],ymm12"
        EmitIfCountGE \RowCount\(), 6, "vmovdqu YMMWORD PTR [rbx+rax*2],ymm14"
        add     rdx,8*4                     # advance matrix C by 8 columns
.if \RowCount\() > 3
        add     rbx,8*4                     # advance matrix C plus 3 rows by 8 columns
.endif
        add     r9,8                        # correct for over-subtract above

.LOutputMasked8xNBlock.\RowCount\():
        mov     DWORD PTR .LGemmU8U8KernelFrame_mask[rsp],r9d
        mov     rbp,QWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)@GOTPCREL[rip]
        vpbroadcastd ymm0,DWORD PTR .LGemmU8U8KernelFrame_mask[rsp]
        vpcmpgtd ymm0,ymm0,YMMWORD PTR [rbp]
        test    r14b,r14b                   # ZeroMode?
        jnz     .LSkipAccumulateOutputMasked8xNBlock.\RowCount\()
        EmitIfCountGE \RowCount\(), 1, "vpmaskmovd ymm4,ymm0,YMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vpmaskmovd ymm6,ymm0,YMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vpmaskmovd ymm8,ymm0,YMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vpmaskmovd ymm10,ymm0,YMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vpmaskmovd ymm12,ymm0,YMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vpmaskmovd ymm14,ymm0,YMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 1, "vpaddd ymm5,ymm5,ymm4"
        EmitIfCountGE \RowCount\(), 2, "vpaddd ymm7,ymm7,ymm6"
        EmitIfCountGE \RowCount\(), 3, "vpaddd ymm9,ymm9,ymm8"
        EmitIfCountGE \RowCount\(), 4, "vpaddd ymm11,ymm11,ymm10"
        EmitIfCountGE \RowCount\(), 5, "vpaddd ymm13,ymm13,ymm12"
        EmitIfCountGE \RowCount\(), 6, "vpaddd ymm15,ymm15,ymm14"

.LSkipAccumulateOutputMasked8xNBlock.\RowCount\():
        EmitIfCountGE \RowCount\(), 1, "vpmaskmovd YMMWORD PTR [rdx],ymm0,ymm5"
        EmitIfCountGE \RowCount\(), 2, "vpmaskmovd YMMWORD PTR [rdx+rax],ymm0,ymm7"
        EmitIfCountGE \RowCount\(), 3, "vpmaskmovd YMMWORD PTR [rdx+rax*2],ymm0,ymm9"
        EmitIfCountGE \RowCount\(), 4, "vpmaskmovd YMMWORD PTR [rbx],ymm0,ymm11"
        EmitIfCountGE \RowCount\(), 5, "vpmaskmovd YMMWORD PTR [rbx+rax],ymm0,ymm13"
        EmitIfCountGE \RowCount\(), 6, "vpmaskmovd YMMWORD PTR [rbx+rax*2],ymm0,ymm15"
.ifb \Fallthrough\()
        jmp     .LExitKernel
.endif

        .endm

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (rdi) - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8U8CopyPackAAvx2.

    B (rsi) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8U8CopyPackBAvx2.

    C (rdx) - Supplies the address of matrix C.

    PairedCountK (rcx) - Supplies the number of paired columns from matrix A and
        the number of paired rows from matrix B to iterate over.

    CountM (r8) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (r9) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldc - Supplies the first dimension of matrix C.

    RowSumVector - Supplies the sum of each row from matrix A multiplied by the
        zero point offset of matrix B. These values are accumulated into every
        row of matrix C.

    ColumnSumVector - Supplies the sum of each column from matrix B multiplied
        by the zero point offset of matrix A. These values are accumulated into
        every column of matrix C.

    DepthValue - Supplies the value CountK multiplied by the zero point offset
        of matrixA multplied by the zero point offset of matrix B. This value is
        accumulated into every element of matrix C.

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        .globl  C_UNDERSCORE(MlasGemmU8U8KernelAvx2)
C_UNDERSCORE(MlasGemmU8U8KernelAvx2):

        push    rbp
        push    rbx
        push    r12
        push    r13
        push    r14

        mov     rax,.LGemmU8U8KernelFrame_ldc[rsp]
        shl     rax,2                       # convert ldc to bytes
        lea     r10,[rcx*4]
        mov     r11,rdi
        mov     r12,.LGemmU8U8KernelFrame_RowSumVector[rsp]
        mov     r13,.LGemmU8U8KernelFrame_ColumnSumVector[rsp]
        movzx   r14,BYTE PTR .LGemmU8U8KernelFrame_ZeroMode[rsp]

//
// Process CountM rows of the matrices.
//

        cmp     r8,5
        ja      .LProcessCountM6
        je      .LProcessCountM5
        cmp     r8,3
        ja      .LProcessCountM4
        je      .LProcessCountM3
        cmp     r8,1
        je      .LProcessCountM1

.LProcessCountM2:
        ProcessCountM 2

.LProcessCountM4:
        ProcessCountM 4

.LProcessCountM6:
        mov     r8d,6                       # return 6 rows handled
        ProcessCountM 6, Fallthrough

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        mov     eax,r8d
        vzeroupper

        pop     r14
        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret

.LProcessCountM1:
        ProcessCountM 1

.LProcessCountM3:
        ProcessCountM 3

.LProcessCountM5:
        ProcessCountM 5

        .end
