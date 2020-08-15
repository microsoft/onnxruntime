/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8S8KernelAvx2.s

Abstract:

    This module implements the kernels for the quantized integer matrix/matrix
    multiply operation (QGEMM).

    This implementation uses AVX2 instructions.

--*/

#include "asmmacro.h"
#include "QgemmU8X8KernelAvx2Common.h"

        .intel_syntax noprefix

//
// Stack frame layout for the U8S8 CopyPackA routine.
//

        .equ    .LGemmU8S8CopyPackAFrame_PaddedMatrixAData, -72
        .equ    .LGemmU8S8CopyPackAFrame_mask, -8
        .equ    .LGemmU8S8CopyPackAFrame_SavedR13, 0
        .equ    .LGemmU8S8CopyPackAFrame_SavedR12, 8
        .equ    .LGemmU8S8CopyPackAFrame_SavedRbx, 16
        .equ    .LGemmU8S8CopyPackAFrame_SavedRbp, 24
        .equ    .LGemmU8S8CopyPackAFrame_ReturnAddress, 32

//
// Stack frame layout for the U8S8 CopyPackB routine.
//

        .equ    .LGemmU8S8CopyPackBFrame_PaddedMatrixBData, -72
        .equ    .LGemmU8S8CopyPackBFrame_Padding, -8
        .equ    .LGemmU8S8CopyPackBFrame_SavedRbx, 0
        .equ    .LGemmU8S8CopyPackBFrame_SavedRbp, 8
        .equ    .LGemmU8S8CopyPackBFrame_ReturnAddress, 16
        .equ    .LGemmU8S8CopyPackBFrame_BIsSigned, 24

        .text

/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

Arguments:

    D (rdi) - Supplies the address of the destination packed buffer.

    A (rsi) - Supplies the address of the source matrix.

    lda (rdx) - Supplies the number of elements per row of the source matrix.

    CountM (rcx) - Supplies the number of rows of the source matrix to copy.

    CountK (r8) - Supplies the number of columns of the source matrix to copy.

    RowSumBuffer (r9) - Supplies the address of the buffer to receive the sums
        of the elements along each of the rows.
        by the zero point offset.

Return Value:

    None.

--*/

        .globl  C_UNDERSCORE(MlasGemmU8S8CopyPackAAvx2)
C_UNDERSCORE(MlasGemmU8S8CopyPackAAvx2):

        push    rbp
        push    rbx
        push    r12
        push    r13

        mov     r10,rdx
        mov     r11,rcx
        lea     r12,[r8+3]
        and     r12,NOT 3                   # align CountK up to quad count
        vpcmpeqw ymm8,ymm8,ymm8             # generate word vector [0xFFFF]
        vpsrlw  ymm8,ymm8,15                # generate word vector [0x0001]
        vpsllw  ymm9,ymm8,8                 # generate word vector [0x0100]
        vpor    ymm9,ymm8,ymm9              # generate word vector [0x0101]

//
// Compute the conditional load/store mask for an unaligned CountK.
//

        mov     eax,r8d
        and     eax,15                      # isolate unaligned count
        add     eax,3
        shr     eax,2                       # align unaligned count to quad count
        mov     DWORD PTR .LGemmU8S8CopyPackAFrame_mask[rsp],eax
        vpbroadcastd xmm10,DWORD PTR .LGemmU8S8CopyPackAFrame_mask[rsp]
        vpcmpgtd xmm10,xmm10,XMMWORD PTR C_UNDERSCORE(MlasMaskMoveAvx)[rip]

//
// Zero initialize the padded stack buffers.
//

        vpxor   xmm0,xmm0,xmm0
        vmovdqu YMMWORD PTR .LGemmU8S8CopyPackAFrame_PaddedMatrixAData[rsp],ymm0
        vmovdqu YMMWORD PTR .LGemmU8S8CopyPackAFrame_PaddedMatrixAData[rsp+32],ymm0

//
// Process 4 rows of matrix A in a loop.
//

        sub     r11,4
        jb      .LCopyPackA.ProcessRemainingRows

.LCopyPackA.ProcessNextRowM4:
        vpxor   xmm0,xmm0,xmm0              # clear row accumulators
        vpxor   xmm1,xmm1,xmm1
        vpxor   xmm2,xmm2,xmm2
        vpxor   xmm3,xmm3,xmm3
        lea     r13,[r10+r10*2]             # compute ldb * 3
        lea     rax,[r12+r12*2]             # compute output stride * 3
        mov     rdx,rsi
        mov     rcx,rdi
        lea     rsi,[rsi+r10*4]             # advance next matrix A by 4 rows
        lea     rdi,[rdi+r12*4]             # advance next matrix D by 4 rows
        mov     rbx,r8                      # reload columns remaining
        sub     rbx,32
        jb      .LCopyPackA.ProcessRemainingColumnsM4

.LCopyPackA.ProcessNextColumnLoopM4:
        vmovdqu ymm4,YMMWORD PTR [rdx]
        vmovdqu ymm5,YMMWORD PTR [rdx+r10]
        vmovdqu ymm6,YMMWORD PTR [rdx+r10*2]
        vmovdqu ymm7,YMMWORD PTR [rdx+r13]
        vmovdqu YMMWORD PTR [rcx],ymm4
        vmovdqu YMMWORD PTR [rcx+r12],ymm5
        vmovdqu YMMWORD PTR [rcx+r12*2],ymm6
        vmovdqu YMMWORD PTR [rcx+rax],ymm7
        vpmaddubsw ymm4,ymm4,ymm9           # horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              # add words to row accumulators
        vpmaddubsw ymm5,ymm5,ymm9
        vpaddw  ymm1,ymm1,ymm5
        vpmaddubsw ymm6,ymm6,ymm9
        vpaddw  ymm2,ymm2,ymm6
        vpmaddubsw ymm7,ymm7,ymm9
        vpaddw  ymm3,ymm3,ymm7
        add     rdx,32                      # advance matrix A by 32 bytes
        add     rcx,32                      # advance matrix D by 32 bytes
        sub     rbx,32                      # subtract columns remaining
        jae     .LCopyPackA.ProcessNextColumnLoopM4

.LCopyPackA.ProcessRemainingColumnsM4:
        add     rbx,32                      # correct for over-subtract above
        jz      .LCopyPackA.ReduceRowSumBufferM4
        test    bl,16                       # (CountK & 16) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan16M4
        vmovdqu xmm4,XMMWORD PTR [rdx]
        vmovdqu xmm5,XMMWORD PTR [rdx+r10]
        vmovdqu xmm6,XMMWORD PTR [rdx+r10*2]
        vmovdqu xmm7,XMMWORD PTR [rdx+r13]
        vmovdqu XMMWORD PTR [rcx],xmm4
        vmovdqu XMMWORD PTR [rcx+r12],xmm5
        vmovdqu XMMWORD PTR [rcx+r12*2],xmm6
        vmovdqu XMMWORD PTR [rcx+rax],xmm7
        vpmaddubsw xmm4,xmm4,xmm9           # horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              # add words to row accumulators
        vpmaddubsw xmm5,xmm5,xmm9
        vpaddw  ymm1,ymm1,ymm5
        vpmaddubsw xmm6,xmm6,xmm9
        vpaddw  ymm2,ymm2,ymm6
        vpmaddubsw xmm7,xmm7,xmm9
        vpaddw  ymm3,ymm3,ymm7
        add     rdx,16                      # advance matrix A by 16 bytes
        add     rcx,16                      # advance matrix D by 16 bytes
        test    bl,15                       # test for unaligned columns
        jz      .LCopyPackA.ReduceRowSumBufferM4

//
// Copy the unaligned CountK columns to a zero padded stack buffer.
//

.LCopyPackA.CopyRemainingCountKLessThan16M4:
        lea     rbp,.LGemmU8S8CopyPackAFrame_PaddedMatrixAData[rsp]
        test    bl,8                        # (CountK & 8) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan8M4
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        mov     rax,QWORD PTR [rdx+r10]
        mov     QWORD PTR [rbp+16],rax
        mov     rax,QWORD PTR [rdx+r10*2]
        mov     QWORD PTR [rbp+32],rax
        mov     rax,QWORD PTR [rdx+r13]
        mov     QWORD PTR [rbp+48],rax
        add     rdx,8
        add     rbp,8                       # advance padded buffer destination

.LCopyPackA.CopyRemainingCountKLessThan8M4:
        test    bl,4                        # (CountK & 4) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan4M4
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        mov     eax,DWORD PTR [rdx+r10]
        mov     DWORD PTR [rbp+16],eax
        mov     eax,DWORD PTR [rdx+r10*2]
        mov     DWORD PTR [rbp+32],eax
        mov     eax,DWORD PTR [rdx+r13]
        mov     DWORD PTR [rbp+48],eax
        add     rdx,4
        add     rbp,4                       # advance padded buffer destination

.LCopyPackA.CopyRemainingCountKLessThan4M4:
        test    bl,2                        # (CountK & 2) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan2M4
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        movzx   eax,WORD PTR [rdx+r10]
        mov     WORD PTR [rbp+16],ax
        movzx   eax,WORD PTR [rdx+r10*2]
        mov     WORD PTR [rbp+32],ax
        movzx   eax,WORD PTR [rdx+r13]
        mov     WORD PTR [rbp+48],ax
        add     rdx,2
        add     rbp,2                       # advance padded buffer destination

.LCopyPackA.CopyRemainingCountKLessThan2M4:
        test    bl,1                        # (CountK & 1) != 0?
        jz      .LCopyPackA.ProcessPaddedMatrixADataM4
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al
        movzx   eax,BYTE PTR [rdx+r10]
        mov     BYTE PTR [rbp+16],al
        movzx   eax,BYTE PTR [rdx+r10*2]
        mov     BYTE PTR [rbp+32],al
        movzx   eax,BYTE PTR [rdx+r13]
        mov     BYTE PTR [rbp+48],al

//
// Process the remaining CountK columns using the zero padded stack buffer.
//

.LCopyPackA.ProcessPaddedMatrixADataM4:
        vmovdqu xmm4,XMMWORD PTR .LGemmU8S8CopyPackAFrame_PaddedMatrixAData[rsp]
        vmovdqu xmm5,XMMWORD PTR .LGemmU8S8CopyPackAFrame_PaddedMatrixAData[rsp+16]
        vmovdqu xmm6,XMMWORD PTR .LGemmU8S8CopyPackAFrame_PaddedMatrixAData[rsp+32]
        vmovdqu xmm7,XMMWORD PTR .LGemmU8S8CopyPackAFrame_PaddedMatrixAData[rsp+48]
        lea     rax,[rcx+r12*2]             # compute matrix D plus 2 rows
        vpmaskmovd XMMWORD PTR [rcx],xmm10,xmm4
        vpmaskmovd XMMWORD PTR [rcx+r12],xmm10,xmm5
        vpmaskmovd XMMWORD PTR [rax],xmm10,xmm6
        vpmaskmovd XMMWORD PTR [rax+r12],xmm10,xmm7
        vpmaddubsw xmm4,xmm4,xmm9           # horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              # add words to row accumulators
        vpmaddubsw xmm5,xmm5,xmm9
        vpaddw  ymm1,ymm1,ymm5
        vpmaddubsw xmm6,xmm6,xmm9
        vpaddw  ymm2,ymm2,ymm6
        vpmaddubsw xmm7,xmm7,xmm9
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
        add     rdi,r12
        mov     rbx,r8                      # reload columns remaining
        sub     rbx,32
        jb      .LCopyPackA.ProcessRemainingColumnsM1

.LCopyPackA.ProcessNextColumnLoopM1:
        vmovdqu ymm4,YMMWORD PTR [rdx]
        vmovdqu YMMWORD PTR [rcx],ymm4
        vpmaddubsw ymm4,ymm4,ymm9           # horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              # add words to row accumulators
        add     rdx,32                      # advance matrix A by 32 bytes
        add     rcx,32                      # advance matrix D by 32 bytes
        sub     rbx,32                      # subtract columns remaining
        jae     .LCopyPackA.ProcessNextColumnLoopM1

.LCopyPackA.ProcessRemainingColumnsM1:
        add     rbx,32                      # correct for over-subtract above
        jz      .LCopyPackA.ReduceRowSumBufferM1
        test    bl,16                       # (CountK & 16) != 0?
        jz      .LCopyPackA.CopyRemainingCountKLessThan16M1
        vmovdqu xmm4,XMMWORD PTR [rdx]
        vmovdqu XMMWORD PTR [rcx],xmm4
        vpmaddubsw xmm4,xmm4,xmm9           # horizontal byte+byte=word per row
        vpaddw  ymm0,ymm0,ymm4              # add words to row accumulators
        add     rdx,16                      # advance matrix A by 16 bytes
        add     rcx,16                      # advance matrix D by 16 bytes
        test    bl,15                       # test for unaligned columns
        jz      .LCopyPackA.ReduceRowSumBufferM1

//
// Copy the unaligned CountK columns to a zero padded stack buffer.
//

.LCopyPackA.CopyRemainingCountKLessThan16M1:
        lea     rbp,.LGemmU8S8CopyPackAFrame_PaddedMatrixAData[rsp]
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
        vmovdqu xmm4,XMMWORD PTR .LGemmU8S8CopyPackAFrame_PaddedMatrixAData[rsp]
        vpmaskmovd XMMWORD PTR [rcx],xmm10,xmm4
        vpmaddubsw ymm4,ymm4,ymm9           # horizontal byte+byte=word per row
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

    BIsSigned - Supplies true if the source matrix is signed data, else false if
        the source matrix is unsigned data.

Return Value:

    None.

--*/

        .globl  C_UNDERSCORE(MlasGemmU8S8CopyPackBAvx2)
C_UNDERSCORE(MlasGemmU8S8CopyPackBAvx2):

        push    rbp
        push    rbx

        mov     r10,rdx
        lea     r11,[r10+r10*2]             # compute ldb * 3
        vpcmpeqw ymm7,ymm7,ymm7             # generate word vector [0xFFFF]
        vpsrlw  ymm7,ymm7,15                # generate word vector [0x0001]
        vpsllw  ymm8,ymm7,8                 # generate word vector [0x0100]
        vpor    ymm8,ymm7,ymm8              # generate word vector [0x0101]

//
// Compute the bit flip vector to adjust input from U8 to S8.
//

        vpxor   xmm9,xmm9,xmm9              # generate word vector [0x0000]
        cmp     BYTE PTR .LGemmU8S8CopyPackBFrame_BIsSigned[rsp],0
        jnz     .LCopyPackB.SkipUnsignedBitFlipVector
        vpsllw  ymm9,ymm8,7                 # generate word vector [0x8080]

.LCopyPackB.SkipUnsignedBitFlipVector:

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
        sub     rbx,4
        jb      .LCopyPackB.ProcessRemainingRowsN16

.LCopyPackB.ProcessNextRowLoopN16:
        vmovdqu xmm2,XMMWORD PTR [rdx]      # load 4 rows
        vmovdqu xmm3,XMMWORD PTR [rdx+r10]
        vmovdqu xmm4,XMMWORD PTR [rdx+r10*2]
        vmovdqu xmm5,XMMWORD PTR [rdx+r11]
        lea     rdx,[rdx+r10*4]             # advance matrix B by 4 rows

.LCopyPackB.InterleaveRowDataN16:
        vpunpcklbw xmm6,xmm2,xmm3           # interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vpunpcklbw xmm2,xmm4,xmm5
        vpunpckhbw xmm5,xmm4,xmm5
        vpunpcklwd xmm4,xmm6,xmm2
        vpunpckhwd xmm6,xmm6,xmm2
        vpunpcklwd xmm2,xmm3,xmm5
        vpunpckhwd xmm3,xmm3,xmm5
        vinserti128 ymm4,ymm4,xmm6,1
        vinserti128 ymm2,ymm2,xmm3,1
        vpxor   ymm4,ymm4,ymm9              # optionally adjust unsigned data
        vpxor   ymm2,ymm2,ymm9
        vmovdqu YMMWORD PTR [rdi],ymm4      # store interleaved rows
        vmovdqu YMMWORD PTR [rdi+32],ymm2
        vpmaddubsw ymm4,ymm8,ymm4           # horizontal byte+byte=word per row
        vpmaddwd ymm4,ymm4,ymm7             # horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              # accumulate per column
        vpmaddubsw ymm2,ymm8,ymm2
        vpmaddwd ymm2,ymm2,ymm7
        vpaddd  ymm1,ymm1,ymm2
        add     rdi,64                      # advance matrix D by 64 bytes
        sub     rbx,4                       # subtract rows remaining
        jae     .LCopyPackB.ProcessNextRowLoopN16

//
// Process the less than 4 remaining rows where the row has 16 columns.
//

.LCopyPackB.ProcessRemainingRowsN16:
        add     rbx,4                       # correct for over-subtract above
        jz      .LCopyPackB.StoreColumnSumBufferN16
        vmovdqu xmm2,XMMWORD PTR [rdx]
        vmovaps xmm3,xmm9
        vmovaps xmm4,xmm9
        vmovaps xmm5,xmm9
        xor     ebx,ebx                     # no more rows remaining
        test    r8b,2                       # (CountK & 2) != 0?
        jz      .LCopyPackB.InterleaveRowDataN16
        vmovdqu xmm3,XMMWORD PTR [rdx+r10]
        test    r8b,1                       # (CountK & 1) != 0?
        jz      .LCopyPackB.InterleaveRowDataN16
        vmovdqu xmm4,XMMWORD PTR [rdx+r10*2]
        jmp     .LCopyPackB.InterleaveRowDataN16

.LCopyPackB.StoreColumnSumBufferN16:
        vmovdqu YMMWORD PTR [r9],ymm0
        vmovdqu YMMWORD PTR [r9+32],ymm1
        add     r9,16*4                     # advance column sum buffer by 16 dwords
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
        vmovdqu YMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp],ymm9
        vmovdqu YMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp+32],ymm9
        sub     r8,4
        jb      .LCopyPackB.ProcessRemainingRowsNUnaligned

.LCopyPackB.ProcessNextRowLoopNUnaligned:
        mov     rdx,rsi
        lea     rbp,.LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp]
        test    cl,8                        # (CountN & 8) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan8K4
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        mov     rax,QWORD PTR [rdx+r10]
        mov     QWORD PTR [rbp+16],rax
        mov     rax,QWORD PTR [rdx+r10*2]
        mov     QWORD PTR [rbp+32],rax
        mov     rax,QWORD PTR [rdx+r11]
        mov     QWORD PTR [rbp+48],rax
        add     rdx,8                       # advance matrix B
        add     rbp,8                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan8K4:
        test    cl,4                        # (CountN & 4) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan4K4
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        mov     eax,DWORD PTR [rdx+r10]
        mov     DWORD PTR [rbp+16],eax
        mov     eax,DWORD PTR [rdx+r10*2]
        mov     DWORD PTR [rbp+32],eax
        mov     eax,DWORD PTR [rdx+r11]
        mov     DWORD PTR [rbp+48],eax
        add     rdx,4                       # advance matrix B
        add     rbp,4                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan4K4:
        test    cl,2                        # (CountN & 2) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan2K4
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        movzx   eax,WORD PTR [rdx+r10]
        mov     WORD PTR [rbp+16],ax
        movzx   eax,WORD PTR [rdx+r10*2]
        mov     WORD PTR [rbp+32],ax
        movzx   eax,WORD PTR [rdx+r11]
        mov     WORD PTR [rbp+48],ax
        add     rdx,2                       # advance matrix B
        add     rbp,2                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan2K4:
        test    cl,1                        # (CountN & 1) != 0?
        jz      .LCopyPackB.ProcessPaddedMatrixBData
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al
        movzx   eax,BYTE PTR [rdx+r10]
        mov     BYTE PTR [rbp+16],al
        movzx   eax,BYTE PTR [rdx+r10*2]
        mov     BYTE PTR [rbp+32],al
        movzx   eax,BYTE PTR [rdx+r11]
        mov     BYTE PTR [rbp+48],al

.LCopyPackB.ProcessPaddedMatrixBData:
        vmovdqu xmm2,XMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp]
        vmovdqu xmm3,XMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp+16]
        vmovdqu xmm4,XMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp+32]
        vmovdqu xmm5,XMMWORD PTR .LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp+48]
        vpunpcklbw xmm6,xmm2,xmm3           # interleave row data
        vpunpckhbw xmm3,xmm2,xmm3
        vpunpcklbw xmm2,xmm4,xmm5
        vpunpckhbw xmm5,xmm4,xmm5
        vpunpcklwd xmm4,xmm6,xmm2
        vpunpckhwd xmm6,xmm6,xmm2
        vpunpcklwd xmm2,xmm3,xmm5
        vpunpckhwd xmm3,xmm3,xmm5
        vinserti128 ymm4,ymm4,xmm6,1
        vinserti128 ymm2,ymm2,xmm3,1
        vpxor   ymm4,ymm4,ymm9              # optionally adjust unsigned data
        vpxor   ymm2,ymm2,ymm9
        vmovdqu YMMWORD PTR [rdi],ymm4      # store interleaved rows
        vmovdqu YMMWORD PTR [rdi+32],ymm2
        vpmaddubsw ymm4,ymm8,ymm4           # horizontal byte+byte=word per row
        vpmaddwd ymm4,ymm4,ymm7             # horizontal word+word=dword per row
        vpaddd  ymm0,ymm0,ymm4              # accumulate per column
        vpmaddubsw ymm2,ymm8,ymm2
        vpmaddwd ymm2,ymm2,ymm7
        vpaddd  ymm1,ymm1,ymm2
        lea     rsi,[rsi+r10*4]             # advance next matrix B by 4 rows
        add     rdi,64                      # advance matrix D by 64 bytes
        sub     r8,4                        # subtract rows remaining
        jae     .LCopyPackB.ProcessNextRowLoopNUnaligned

.LCopyPackB.ProcessRemainingRowsNUnaligned:
        add     r8,4
        jz      .LCopyPackB.StoreColumnSumBufferNUnaligned

//
// Process the less than 4 remaining rows where the row has less than 16 columns.
//

        lea     rbp,.LGemmU8S8CopyPackBFrame_PaddedMatrixBData[rsp]
        vmovdqu YMMWORD PTR [rbp],ymm9
        vmovdqu YMMWORD PTR [rbp+32],ymm9

.LCopyPackB.CopyUnalignedRowLoop:
        lea     r11,[rbp+16]                # advance next padded buffer by 16 bytes
        mov     rdx,rsi
        test    cl,8                        # (CountN & 8) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan8KSmall
        mov     rax,QWORD PTR [rdx]
        mov     QWORD PTR [rbp],rax
        add     rdx,8                       # advance matrix B
        add     rbp,8                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan8KSmall:
        test    cl,4                        # (CountN & 4) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan4KSmall
        mov     eax,DWORD PTR [rdx]
        mov     DWORD PTR [rbp],eax
        add     rdx,4                       # advance matrix B
        add     rbp,4                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan4KSmall:
        test    cl,2                      # (CountN & 2) != 0?
        jz      .LCopyPackB.CopyRemainingCountNLessThan2KSmall
        movzx   eax,WORD PTR [rdx]
        mov     WORD PTR [rbp],ax
        add     rdx,2                       # advance matrix B
        add     rbp,2                       # advance padded buffer destination

.LCopyPackB.CopyRemainingCountNLessThan2KSmall:
        test    cl,1                        # (CountN & 1) != 0?
        jz      .LCopyPackB.DoneCopyRemainingCountNKSmall
        movzx   eax,BYTE PTR [rdx]
        mov     BYTE PTR [rbp],al

.LCopyPackB.DoneCopyRemainingCountNKSmall:
        dec     r8
        jz      .LCopyPackB.ProcessPaddedMatrixBData
        add     rsi,r10                     # advance next matrix B by 1 row
        mov     rbp,r11
        jmp     .LCopyPackB.CopyUnalignedRowLoop

.LCopyPackB.StoreColumnSumBufferNUnaligned:
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

    ymm12 - Supplies a 256-bit with the broadcasted word value 0x0001.

--*/

        .macro MultiplyAccumulateRow ColumnCount, Vec1Reg, Vec2Reg

        vpmaddubsw ymm3,ymm2,ymm0
        vpmaddwd ymm3,ymm3,ymm12
.if \ColumnCount\() == 16
        vpaddd  \Vec1Reg\(),\Vec1Reg\(),ymm3
        vpmaddubsw ymm2,ymm2,ymm1
        vpmaddwd ymm2,ymm2,ymm12
        vpaddd  \Vec2Reg\(),\Vec2Reg\(),ymm2
.else
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

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rcx - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm11 - Supplies the block accumulators.

    ymm12 - Supplies a 256-bit with the broadcasted word value 0x0001.

--*/

        .macro ComputeBlock ColumnCount, RowCount, VectorOffset, BroadcastOffset

.if \RowCount\() == 1
        vpbroadcastd ymm2,DWORD PTR [rdi+\BroadcastOffset\()]
        vpmaddubsw ymm3,ymm2,YMMWORD PTR [rsi+\VectorOffset\()]
        vpmaddwd ymm3,ymm3,ymm12
.if \ColumnCount\() == 16
        vpaddd  ymm4,ymm4,ymm3
        vpmaddubsw ymm2,ymm2,YMMWORD PTR [rsi+\VectorOffset\()+32]
        vpmaddwd ymm2,ymm2,ymm12
        vpaddd  ymm5,ymm5,ymm2
.else
        vpaddd  ymm5,ymm5,ymm3
.endif
.else
        vmovdqu ymm0,YMMWORD PTR [rsi+\VectorOffset\()]
        EmitIfCountGE \ColumnCount\(), 16, "vmovdqu ymm1,YMMWORD PTR [rsi+\VectorOffset\()+32]"
        EmitIfCountGE \RowCount\(), 1, "vpbroadcastd ymm2,DWORD PTR [rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "MultiplyAccumulateRow \ColumnCount\(), ymm4, ymm5"
        EmitIfCountGE \RowCount\(), 2, "vpbroadcastd ymm2,DWORD PTR [rdi+rcx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "MultiplyAccumulateRow \ColumnCount\(), ymm6, ymm7"
        EmitIfCountGE \RowCount\(), 3, "vpbroadcastd ymm2,DWORD PTR [rdi+rcx*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "MultiplyAccumulateRow \ColumnCount\(), ymm8, ymm9"
        EmitIfCountGE \RowCount\(), 4, "vpbroadcastd ymm2,DWORD PTR [rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "MultiplyAccumulateRow \ColumnCount\(), ymm10, ymm11"
.endif

        .endm

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple
    times and advancing the matrix A and matrix B data pointers.

Arguments:

    ColumnCount - Supplies the number of columns to produce.

    RowCount - Supplies the number of rows to produce.

Implicit Arguments:

    rbx - Supplies the address into the matrix A data plus 3 rows.

    rdi - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the length in bytes of a row from matrix A.

    ymm4-ymm11 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLoop ColumnCount, RowCount

        mov     rbp,rcx                     # reload row length remaining

.LComputeBlockBy1Loop\@:
        ComputeBlock \ColumnCount\(), \RowCount\(), 0, 0
        add     rdi,4                       # advance matrix A by 1 quad
.if \RowCount\() > 3
        add     rbx,4                       # advance matrix A plus 3 rows by 1 quad
.endif
        add     rsi,64                      # advance matrix B
        sub     rbp,4
        jnz     .LComputeBlockBy1Loop\@

        .endm

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (rdi) - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8S8CopyPackAAvx2.

    B (rsi) - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8S8CopyPackBAvx2.

    C (rdx) - Supplies the address of matrix C.

    PackedCountK (rcx) - Supplies the number of packed columns from matrix A
        and the number of packed rows from matrix B to iterate over.

    CountM (r8) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (r9) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldc - Supplies the first dimension of matrix C.

    RowSumBuffer - Supplies the sum of each row from matrix A multiplied by the
        zero point offset of matrix B. These values are accumulated into every
        row of matrix C.

    ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
        by the zero point offset of matrix A. These values are accumulated into
        every column of matrix C.

    DepthValue - Supplies the value CountK multiplied by the zero point offset
        of matrix A multplied by the zero point offset of matrix B. This value
        is accumulated into every element of matrix C.

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        .globl  C_UNDERSCORE(MlasGemmU8S8KernelAvx2)
C_UNDERSCORE(MlasGemmU8S8KernelAvx2):

        push    rbp
        push    rbx
        push    r12
        push    r13

        mov     rax,.LGemmU8X8KernelFrame_ldc[rsp]
        shl     rax,2                       # convert ldc to bytes
        shl     rcx,2                       # convert to row length
        movzx   r10,BYTE PTR .LGemmU8X8KernelFrame_ZeroMode[rsp]
        mov     r11,rdi
        mov     r12,.LGemmU8X8KernelFrame_RowSumBuffer[rsp]
        mov     r13,.LGemmU8X8KernelFrame_ColumnSumBuffer[rsp]
        vpcmpeqw ymm12,ymm12,ymm12          # generate 256-bit word vector [0xFFFF]
        vpsrlw  ymm12,ymm12,15              # generate 256-bit word vector [0x0001]

//
// Process CountM rows of the matrices.
//

        cmp     r8,3
        ja      .LProcessCountM4
        je      .LProcessCountM3
        cmp     r8,1
        je      .LProcessCountM1

.LProcessCountM2:
        ProcessCountM 2

.LProcessCountM4:
        mov     r8d,4                       # return 4 rows handled
        ProcessCountM 4, Fallthrough

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        mov     eax,r8d
        vzeroupper

        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret

.LProcessCountM1:
        ProcessCountM 1

.LProcessCountM3:
        ProcessCountM 3

        .end
