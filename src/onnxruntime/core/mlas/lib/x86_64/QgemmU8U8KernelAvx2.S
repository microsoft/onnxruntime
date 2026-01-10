/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8U8KernelAvx2.s

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

    This implementation uses AVX2 instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

//
// Stack frame layout for the U8U8 CopyPackA routine.
//

        .equ    .LGemmU8U8CopyPackAFrame_PaddedMatrixAData, -72
        .equ    .LGemmU8U8CopyPackAFrame_Padding, -8
        .equ    .LGemmU8U8CopyPackAFrame_SavedR13, 0
        .equ    .LGemmU8U8CopyPackAFrame_SavedR12, 8
        .equ    .LGemmU8U8CopyPackAFrame_SavedRbx, 16
        .equ    .LGemmU8U8CopyPackAFrame_SavedRbp, 24
        .equ    .LGemmU8U8CopyPackAFrame_ReturnAddress, 32

//
// Stack frame layout for the U8U8 CopyPackB routine.
//

        .equ    .LGemmU8U8CopyPackBFrame_PaddedMatrixBData, -40
        .equ    .LGemmU8U8CopyPackBFrame_Padding, -8
        .equ    .LGemmU8U8CopyPackBFrame_SavedRbx, 0
        .equ    .LGemmU8U8CopyPackBFrame_SavedRbp, 8
        .equ    .LGemmU8U8CopyPackBFrame_ReturnAddress, 16

        .text

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

    RowSumBuffer (r9) - Supplies the address of the buffer to receive the sums
        of the elements along each of the rows.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasGemmU8U8CopyPackAAvx2

        push    rbp
        push    rbx
        push    r12
        push    r13

        mov     r10,rdx
        mov     r11,rcx
        lea     r12,[r8+1]
        and     r12,NOT 1                   # align CountK up to pair count
        vpcmpeqw ymm8,ymm8,ymm8             # generate word vector [0xFFFF]
        vpsrlw  ymm8,ymm8,15                # generate word vector [0x0001]

//
// Compute the conditional load/store mask for an unaligned CountK.
//

        mov     eax,r8d
        and     eax,15                      # isolate unaligned count
        inc     eax
        shr     eax,1                       # align unaligned count to pair count
        neg     rax
        lea     rbx,C_UNDERSCORE(MlasMaskMoveTableAvx)[rip+8*4]
        vmovdqu ymm9,YMMWORD PTR [rbx+rax*4]

//
// Zero initialize the padded stack buffers.
//

        vpxor   xmm0,xmm0,xmm0
        vmovdqu YMMWORD PTR .LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp],ymm0
        vmovdqu YMMWORD PTR .LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp+32],ymm0

//
// Process 4 rows of matrix A in a loop.
//
// Zero extend the source bytes to 16-bits and write to the packed buffer.
//
// The packed buffer has the same data ordering as the source bytes, but CountK
// is aligned up to a multiple of 2 to maintain 32-bit alignment. All padding
// bytes are zero filled.
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
        lea     rdi,[rdi+r12*8]             # advance next matrix D by 4 rows
        mov     rbx,r8                      # reload columns remaining
        sub     rbx,16
        jb      .LCopyPackA.ProcessRemainingColumnsM4

.LCopyPackA.ProcessNextColumnLoopM4:
        lea     rax,[rdx+r10*2]             # compute matrix A plus 2 rows
        vpmovzxbw ymm4,XMMWORD PTR [rdx]
        vpmovzxbw ymm5,XMMWORD PTR [rdx+r10]
        vpmovzxbw ymm6,XMMWORD PTR [rax]
        vpmovzxbw ymm7,XMMWORD PTR [rax+r10]
        lea     rax,[rcx+r12*4]             # compute matrix D plus 2 rows
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
        jz      .LCopyPackA.ReduceRowSumBufferM4

//
// Copy the unaligned CountK columns to a zero padded stack buffer.
//

        lea     rbp,.LGemmU8U8CopyPackAFrame_PaddedMatrixAData[rsp]
        test    bl,8                        # (CountK & 8) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan8M4
        lea     r13,[rdx+r10*2]             # compute matrix A plus 2 rows
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
        lea     r13,[rdx+r10*2]             # compute matrix A plus 2 rows
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
        lea     r13,[rdx+r10*2]             # compute matrix A plus 2 rows
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
        lea     r13,[rdx+r10*2]             # compute matrix A plus 2 rows
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
        lea     rax,[rcx+r12*4]             # compute matrix D plus 2 rows
        vpmaskmovd YMMWORD PTR [rcx],ymm9,ymm4
        vpmaskmovd YMMWORD PTR [rcx+r12*2],ymm9,ymm5
        vpmaskmovd YMMWORD PTR [rax],ymm9,ymm6
        vpmaskmovd YMMWORD PTR [rax+r12*2],ymm9,ymm7
        vpaddw  ymm0,ymm0,ymm4              # accumulate per row along columns
        vpaddw  ymm1,ymm1,ymm5
        vpaddw  ymm2,ymm2,ymm6
        vpaddw  ymm3,ymm3,ymm7

//
// Reduce the sums for the four rows of output.
//

.LCopyPackA.ReduceRowSumBufferM4:
        vpmaddwd ymm0,ymm0,ymm8             # horizontal word+word=dword per row
        vpmaddwd ymm1,ymm1,ymm8
        vphaddd ymm0,ymm0,ymm1              # reduce and interleave Sum1/Sum0
        vpmaddwd ymm2,ymm2,ymm8
        vpmaddwd ymm3,ymm3,ymm8
        vphaddd ymm1,ymm2,ymm3              # reduce and interleave Sum3/Sum2
        vphaddd ymm0,ymm0,ymm1              # reduce and interleave Sum3/Sum2/Sum1/Sum0
        vextracti128 xmm1,ymm0,1            # extract high dwords
        vpaddd  xmm0,xmm0,xmm1              # reduce low/high dwords
        vmovdqu XMMWORD PTR [r9],xmm0
        add     r9,4*4                      # advance row sum buffer by 4 dwords
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
        jz      .LCopyPackA.ReduceRowSumBufferM1

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

.LCopyPackA.ReduceRowSumBufferM1:
        vpmaddwd ymm0,ymm0,ymm8             # horizontal word+word=dword per row
        vextracti128 xmm1,ymm0,1            # extract high dwords
        vpaddd  xmm0,xmm0,xmm1              # reduction
        vphaddd xmm0,xmm0,xmm0
        vphaddd xmm0,xmm0,xmm0
        vmovd   DWORD PTR [r9],xmm0
        add     r9,4                        # advance row sum buffer by 1 dword
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

    ColumnSumBuffer (r9) - Supplies the address of the buffer to receive the sums
        of the elements along each of the columns.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasGemmU8U8CopyPackBAvx2

        push    rbp
        push    rbx

        mov     r10,rdx
        vpcmpeqw ymm5,ymm5,ymm5             # generate word vector [0xFFFF]
        vpsrlw  ymm5,ymm5,15                # generate word vector [0x0001]

//
// Zero initialize the padded stack buffers.
//

        vpxor   xmm0,xmm0,xmm0
        vmovdqu YMMWORD PTR .LGemmU8U8CopyPackBFrame_PaddedMatrixBData[rsp],ymm0

//
// Process 16 columns of matrix B in a loop.
//

        sub     rcx,16
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
        vmovdqu xmm2,XMMWORD PTR [rdx]      # load 2 rows
        vmovdqu xmm3,XMMWORD PTR [rdx+r10]
        lea     rdx,[rdx+r10*2]             # advance matrix B by 2 rows
        vpunpcklbw xmm4,xmm2,xmm3           # interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vmovdqu XMMWORD PTR [rdi],xmm4      # store interleaved rows
        vmovdqu XMMWORD PTR [rdi+16],xmm3
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        add     rdi,32                      # advance matrix D by 32 bytes
        vpmaddwd ymm4,ymm4,ymm5             # horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              # accumulate per column
        vpmaddwd ymm3,ymm3,ymm5
        vpaddd  ymm1,ymm1,ymm3
        sub     rbx,2                       # subtract rows remaining
        jae     .LCopyPackB.ProcessNextRowLoopN16

.LCopyPackB.ProcessRemainingRowsN16:
        add     rbx,2                       # correct for over-subtract above
        jz      .LCopyPackB.StoreColumnSumBufferN16
        vpmovzxbw ymm4,XMMWORD PTR [rdx]
        vmovdqu YMMWORD PTR [rdi],ymm4      # store interleaved rows
        vextracti128 xmm3,ymm4,1
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        vpmaddwd ymm4,ymm4,ymm5             # horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              # accumulate per column
        vpmaddwd ymm3,ymm3,ymm5
        vpaddd  ymm1,ymm1,ymm3
        add     rdi,32                      # advance matrix D by 32 bytes

.LCopyPackB.StoreColumnSumBufferN16:
        vmovdqu YMMWORD PTR [r9],ymm0
        vmovdqu YMMWORD PTR [r9+32],ymm1
        add     r9,64                       # advance column sum buffer by 16 dwords
        sub     rcx,16                      # subtract columns remaining
        jae     .LCopyPackB.ProcessNextColumnN16

.LCopyPackB.ProcessRemainingColumns:
        add     rcx,16                      # correct for over-subtract above
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
        test    cl,8                        # (CountN & 8) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan8K2
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        mov     rax,QWORD PTR [rdx+r10]
        mov     QWORD PTR [rbp+16],rax
        add     rdx,8                       # advance matrix B
        add     rbp,8                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan8K2:
        test    cl,4                        # (CountN & 4) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan4K2
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        mov     eax,DWORD PTR [rdx+r10]
        mov     DWORD PTR [rbp+16],eax
        add     rdx,4                       # advance matrix B
        add     rbp,4                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan4K2:
        test    cl,2                        # (CountN & 2) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan2K2
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        movzx   eax,WORD PTR [rdx+r10]
        mov     WORD PTR [rbp+16],ax
        add     rdx,2                       # advance matrix B
        add     rbp,2                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan2K2:
        test    cl,1                        # (CountN & 1) != 0?
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
        vpmaddwd ymm4,ymm4,ymm5             # horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              # accumulate per column
        vpmaddwd ymm3,ymm3,ymm5
        vpaddd  ymm1,ymm1,ymm3
        lea     rsi,[rsi+r10*2]             # advance next matrix B by 2 rows
        add     rdi,32                      # advance matrix D by 32 bytes
        sub     r8,2                        # subtract rows remaining
        jae     .LCopyPackB.ProcessNextRowLoopNUnaligned

.LCopyPackB.ProcessRemainingRowsNUnaligned:
        add     r8,2
        jz      .LCopyPackB.StoreColumnSumBufferNUnaligned
        mov     rdx,rsi
        lea     rbp,.LGemmU8U8CopyPackBFrame_PaddedMatrixBData[rsp]
        test    cl,8                        # (CountN & 8) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan8K1
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        add     rdx,8                       # advance matrix B
        add     rbp,8                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan8K1:
        test    cl,4                        # (CountN & 4) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan4K1
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        add     rdx,4                       # advance matrix B
        add     rbp,4                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan4K1:
        test    cl,2                        # (CountN & 2) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan2K1
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        add     rdx,2                       # advance matrix B
        add     rbp,2                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan2K1:
        test    cl,1                        # (CountN & 1) != 0?
        jz      .LCopyPackB.ProcessPaddedMatrixBDataK1
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al

.LCopyPackB.ProcessPaddedMatrixBDataK1:
        vpmovzxbw ymm4,XMMWORD PTR .LGemmU8U8CopyPackBFrame_PaddedMatrixBData[rsp]
        vmovdqu YMMWORD PTR [rdi],ymm4      # store interleaved rows
        vextracti128 xmm3,ymm4,1
        vpmovzxbw ymm4,xmm4
        vpmovzxbw ymm3,xmm3
        vpmaddwd ymm4,ymm4,ymm5             # horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              # accumulate per column
        vpmaddwd ymm3,ymm3,ymm5
        vpaddd  ymm1,ymm1,ymm3

.LCopyPackB.StoreColumnSumBufferNUnaligned:
        vmovdqu YMMWORD PTR [r9],ymm0
        vmovdqu YMMWORD PTR [r9+32],ymm1
        jmp     .LCopyPackB.ExitRoutine

        .end
