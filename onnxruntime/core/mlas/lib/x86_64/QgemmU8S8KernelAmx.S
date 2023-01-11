/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    QgemmU8S8KernelAmx.s

Abstract:

    This module implements the packing functions for the quantized integer matrix/matrix
    multiply operation (QGEMM).

    These packing functions are suited for AMX Qgemm kernel. The implementation only
    uses AVX2 instructions.

--*/

#include "asmmacro.h"

        .intel_syntax noprefix

//
// Stack frame layout for the U8S8 CopyPackB routine.
//

        .equ    .LGemmU8S8CopyPackBFrame_SavedR12, 0
        .equ    .LGemmU8S8CopyPackBFrame_SavedRbx, 8
        .equ    .LGemmU8S8CopyPackBFrame_SavedRbp, 16
        .equ    .LGemmU8S8CopyPackBFrame_ReturnAddress, 24
        .equ    .LGemmU8S8CopyPackBFrame_BIsSigned, 32

        .text

/*++

Routine Description:

    This routine copies elements from the source B matrix to the destination
    packed buffer.

    This implementation is almost identical to MlasGemmU8S8CopyPackBAvx2
    where it traverse B vertically, take a block of 4 row 16 col, transpose
    and store it, then go down 4 row to grab the next 4x16 block. The only
    difference here is that we need K to be aligned to 64 to the fill
    an AMX tile.

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

        FUNCTION_ENTRY MlasGemmU8S8CopyPackBAmx

        push    rbp
        push    rbx
        push    r12

        mov     r10,rdx
        lea     r11,[r10+r10*2]             # compute ldb * 3
        lea     r12,[r8+3]                  # compute extra padding for 64|K
        shr     r12,2
        neg     r12
        and     r12,15
        vpcmpeqw ymm0,ymm0,ymm0             # generate word vector [0xFFFF]
        vpsrlw  ymm0,ymm0,15                # generate word vector [0x0001]
        vpsllw  ymm1,ymm0,8                 # generate word vector [0x0100]
        vpor    ymm1,ymm0,ymm1              # generate word vector [0x0101]

//
// Compute the bit flip vector to adjust input from U8 to S8.
//

        vpxor   xmm2,xmm2,xmm2              # generate word vector [0x0000]
        cmp     BYTE PTR .LGemmU8S8CopyPackBFrame_BIsSigned[rsp],0
        jnz     .LCopyPackB.SkipUnsignedBitFlipVector
        vpsllw  ymm2,ymm1,7                 # generate word vector [0x8080]

.LCopyPackB.SkipUnsignedBitFlipVector:

//
// Process 16 columns of matrix B in a loop.
//

        sub     rcx,16                      # CountN -= 16
        jb      .LCopyPackB.ProcessRemainingColumns

.LCopyPackB.ProcessNextColumnN16:
        vpxord  xmm30,xmm30,xmm30           # clear column accumulators
        vpxord  xmm31,xmm31,xmm31
        mov     rdx,rsi                     # rdx -> B start of 16 columns
        add     rsi,16                      # advance next matrix B by 16 columns
        mov     rbx,r8                      # reload rows remaining
        sub     rbx,4
        jb      .LCopyPackB.ProcessRemainingRowsN16

.LCopyPackB.ProcessNextRowLoopN16:
        vmovdqu64 xmm16,XMMWORD PTR [rdx]   # load 4 rows
        vmovdqu64 xmm17,XMMWORD PTR [rdx+r10]
        vmovdqu64 xmm18,XMMWORD PTR [rdx+r10*2]
        vmovdqu64 xmm19,XMMWORD PTR [rdx+r11]
        lea     rdx,[rdx+r10*4]             # advance matrix B by 4 rows

.LCopyPackB.InterleaveRowDataN16:
        vpunpcklbw xmm3,xmm16,xmm17         # interleave row data
        vpunpckhbw xmm17,xmm16,xmm17
        vpunpcklbw xmm16,xmm18,xmm19
        vpunpckhbw xmm19,xmm18,xmm19
        vpunpcklwd xmm18,xmm3,xmm16
        vpunpckhwd xmm3,xmm3,xmm16
        vpunpcklwd xmm16,xmm17,xmm19
        vpunpckhwd xmm17,xmm17,xmm19
        vinserti64x2 ymm18,ymm18,xmm3,1
        vinserti64x2 ymm16,ymm16,xmm17,1
        vpxord  ymm18,ymm18,ymm2            # optionally adjust unsigned data
        vpxord  ymm16,ymm16,ymm2
        vmovdqu64 YMMWORD PTR [rdi],ymm18   # store interleaved rows
        vmovdqu64 YMMWORD PTR [rdi+32],ymm16
        vpmaddubsw ymm18,ymm1,ymm18         # horizontal byte+byte=word per row
        vpmaddwd ymm18,ymm18,ymm0           # horizontal word+word=dword per row
        vpaddd  ymm30,ymm30,ymm18           # accumulate per column
        vpmaddubsw ymm16,ymm1,ymm16
        vpmaddwd ymm16,ymm16,ymm0
        vpaddd  ymm31,ymm31,ymm16
        add     rdi,64                      # advance matrix D by 64 bytes
        sub     rbx,4                       # subtract rows remaining
        jae     .LCopyPackB.ProcessNextRowLoopN16

//
// Process the less than 4 remaining rows where the row has 16 columns.
//

.LCopyPackB.ProcessRemainingRowsN16:
        add     rbx,4                       # correct for over-subtract above
        jz      .LCopyPackB.StoreColumnSumBufferN16
        vmovdqu64 xmm16,XMMWORD PTR [rdx]
        vmovaps xmm17,xmm2
        vmovaps xmm18,xmm2
        vmovaps xmm19,xmm2
        xor     ebx,ebx                     # no more rows remaining
        test    r8b,2                       # (CountK & 2) != 0?
        jz      .LCopyPackB.InterleaveRowDataN16
        vmovdqu64 xmm17,XMMWORD PTR [rdx+r10]
        test    r8b,1                       # (CountK & 1) != 0?
        jz      .LCopyPackB.InterleaveRowDataN16
        vmovdqu64 xmm18,XMMWORD PTR [rdx+r10*2]
        jmp     .LCopyPackB.InterleaveRowDataN16

.LCopyPackB.StoreColumnSumBufferN16:
        vmovdqu64 YMMWORD PTR [r9],ymm30
        vmovdqu64 YMMWORD PTR [r9+32],ymm31
        test    r12,r12
        jz      .LCopyPackB.N16K64PaddingFinished
        mov     rax, r12
        vpxord  xmm30,xmm30,xmm30

.LCopyPackB.N16K64Padding:
        vmovdqu64 YMMWORD PTR [rdi],ymm30   # store 0
        vmovdqu64 YMMWORD PTR [rdi+32],ymm30
        add     rdi,64
        dec     rax
        jnz     .LCopyPackB.N16K64Padding

.LCopyPackB.N16K64PaddingFinished:
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

        pop     r12
        pop     rbx
        pop     rbp
        ret

//
// Process the remaining columns of matrix B.
//

.LCopyPackB.ProcessColumnNUnaligned:
        vpxord  xmm30,xmm30,xmm30           # clear column accumulators
        vpxord  xmm31,xmm31,xmm31
        neg     ecx
        and     ecx,63
        mov     rbx,-1
        shr     rbx,cl                      # mask for left over N
        kmovq   k1,rbx                      # mask
        sub     r8,4
        jb      .LCopyPackB.ProcessRemainingRowsNUnaligned

.LCopyPackB.ProcessNextRowLoopNUnaligned:
        vmovdqu64 xmm16,xmm2
        vmovdqu8 xmm16 {k1},XMMWORD PTR [rsi]   # load 4 rows
        vmovdqu64 xmm17,xmm2
        vmovdqu8 xmm17 {k1},XMMWORD PTR [rsi+r10]
        vmovdqu64 xmm18,xmm2
        vmovdqu8 xmm18 {k1},XMMWORD PTR [rsi+r10*2]
        vmovdqu64 xmm19,xmm2
        vmovdqu8 xmm19 {k1},XMMWORD PTR [rsi+r11]
        lea     rsi,[rsi+r10*4]             # advance next matrix B by 4 rows

.LCopyPackB.InterleaveRowDataUnaligned:
        vpunpcklbw xmm3,xmm16,xmm17         # interleave row data
        vpunpckhbw xmm17,xmm16,xmm17
        vpunpcklbw xmm16,xmm18,xmm19
        vpunpckhbw xmm19,xmm18,xmm19
        vpunpcklwd xmm18,xmm3,xmm16
        vpunpckhwd xmm3,xmm3,xmm16
        vpunpcklwd xmm16,xmm17,xmm19
        vpunpckhwd xmm17,xmm17,xmm19
        vinserti64x2 ymm18,ymm18,xmm3,1
        vinserti64x2 ymm16,ymm16,xmm17,1
        vpxord  ymm18,ymm18,ymm2            # optionally adjust unsigned data
        vpxord  ymm16,ymm16,ymm2
        vmovdqu64 YMMWORD PTR [rdi],ymm18   # store interleaved rows
        vmovdqu64 YMMWORD PTR [rdi+32],ymm16
        vpmaddubsw ymm18,ymm1,ymm18         # horizontal byte+byte=word per row
        vpmaddwd ymm18,ymm18,ymm0           # horizontal word+word=dword per row
        vpaddd  ymm30,ymm30,ymm18           # accumulate per column
        vpmaddubsw ymm16,ymm1,ymm16
        vpmaddwd ymm16,ymm16,ymm0
        vpaddd  ymm31,ymm31,ymm16
        add     rdi,64                      # advance matrix D by 64 bytes
        sub     r8,4                        # subtract rows remaining
        jae     .LCopyPackB.ProcessNextRowLoopNUnaligned

//
// Process the less than 4 remaining rows where the row has less than 16 columns.
//

.LCopyPackB.ProcessRemainingRowsNUnaligned:
        add     r8,4
        jz      .LCopyPackB.StoreColumnSumBufferNUnaligned

        vmovaps xmm16,xmm2
        vmovdqu8 xmm16 {k1},XMMWORD PTR [rsi]
        vmovaps xmm17,xmm2
        vmovaps xmm18,xmm2
        vmovaps xmm19,xmm2
        mov     rbx,r8
        xor     r8b,r8b                     # no more rows remaining
        test    bl,2                        # (CountK & 2) != 0?
        jz      .LCopyPackB.InterleaveRowDataUnaligned
        vmovdqu8 xmm17 {k1},XMMWORD PTR [rsi+r10]
        test    bl,1                        # (CountK & 1) != 0?
        jz      .LCopyPackB.InterleaveRowDataUnaligned
        vmovdqu8 xmm18 {k1},XMMWORD PTR [rsi+r10*2]
        jmp     .LCopyPackB.InterleaveRowDataUnaligned

.LCopyPackB.StoreColumnSumBufferNUnaligned:
        vmovdqu64 YMMWORD PTR [r9],ymm30
        vmovdqu64 YMMWORD PTR [r9+32],ymm31
        test    r12,r12
        jz      .LCopyPackB.ExitRoutine
        mov     rax, r12
        vpxord   xmm30,xmm30,xmm30

.LCopyPackB.K64Padding:
        vmovdqu64 YMMWORD PTR [rdi],ymm30   # store 0
        vmovdqu64 YMMWORD PTR [rdi+32],ymm30
        add     rdi,64
        dec     rax
        jne     .LCopyPackB.K64Padding
        jmp     .LCopyPackB.ExitRoutine


//
// Stack frame layout for the U8S8 CopyPackA routine.
//
        .equ    .LGemmU8S8CopyPackAFrame_SavedR13, 0
        .equ    .LGemmU8S8CopyPackAFrame_SavedR12, 8
        .equ    .LGemmU8S8CopyPackAFrame_SavedRbx, 16
        .equ    .LGemmU8S8CopyPackAFrame_SavedRbp, 24
        .equ    .LGemmU8S8CopyPackAFrame_ReturnAddress, 32

/*++

Routine Description:

    This routine copies elements from the source matrix A to the destination
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

        FUNCTION_ENTRY MlasGemmU8S8CopyPackAAmx

        push    rbp
        push    rbx
        push    r12
        push    r13

        mov     r10,rdx                         # lda
        mov     r11,rcx                         # m = CountM
        lea     r12,[r8+63]
        and     r12,NOT 63                      # align CountK up to 64
        vpternlogd zmm30,zmm30,zmm30,255        # generate word vector [0xFFFF]
        vpsrlw  zmm30,zmm30,15                  # generate word vector [0x0001]
        vpsllw  zmm31,zmm30,8                   # generate word vector [0x0100]
        vpord   zmm31,zmm30,zmm31               # generate word vector [0x0101]
        lea     r13,[r10+r10*2]                 # compute ldb * 3
        lea     rax,[r12+r12*2]                 # compute AlignedCountK * 3
        mov     ecx,r8d                         # CountK
        neg     ecx
        and     ecx,63
        mov     rbx,-1
        shr     rbx,cl                          # mask for left over k < 64
        kmovq   k1,rbx                          # mask

//
// Process 4 rows of matrix A in a loop.
//

        sub     r11,4                           # m -= 4
        jb      .LCopyPackA.ProcessRemainingRows

.LCopyPackA.ProcessNextRowM4:
        vpxor   xmm0,xmm0,xmm0                  # clear row accumulators
        vpxor   xmm1,xmm1,xmm1
        vpxor   xmm2,xmm2,xmm2
        vpxor   xmm3,xmm3,xmm3
        mov     rdx,rsi                         # src = A
        mov     rcx,rdi                         # dst = D
        lea     rsi,[rsi+r10*4]                 # advance next matrix A by 4 rows
        lea     rdi,[rdi+r12*4]                 # advance next matrix D by 4 rows
        mov     rbx,r8                          # k = CountK
        sub     rbx,64
        jb      .LCopyPackA.ProcessRemainingColumnsM4

.LCopyPackA.ProcessNextColumnLoopM4:
        vmovdqu64  zmm16,ZMMWORD PTR [rdx]
        vmovdqu64  zmm17,ZMMWORD PTR [rdx+r10]
        vmovdqu64  zmm18,ZMMWORD PTR [rdx+r10*2]
        vmovdqu64  zmm19,ZMMWORD PTR [rdx+r13]
        vmovdqu64  ZMMWORD PTR [rcx],zmm16
        vmovdqu64  ZMMWORD PTR [rcx+r12],zmm17
        vmovdqu64  ZMMWORD PTR [rcx+r12*2],zmm18
        vmovdqu64  ZMMWORD PTR [rcx+rax],zmm19
        vpmaddubsw zmm16,zmm16,zmm31             # horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm16               # add words to row accumulators
        vpmaddubsw zmm17,zmm17,zmm31
        vpaddw     zmm1,zmm1,zmm17
        vpmaddubsw zmm18,zmm18,zmm31
        vpaddw     zmm2,zmm2,zmm18
        vpmaddubsw zmm19,zmm19,zmm31
        vpaddw     zmm3,zmm3,zmm19
        add     rdx,64                          # src += 64
        add     rcx,64                          # dst += 64
        sub     rbx,64                          # k -= 64
        jae     .LCopyPackA.ProcessNextColumnLoopM4

.LCopyPackA.ProcessRemainingColumnsM4:
        add     rbx,64                          # correct for over-subtract above
        jz      .LCopyPackA.ReduceRowSumBufferM4
        vmovdqu8   zmm16{k1}{z},ZMMWORD PTR [rdx]
        vmovdqu8   zmm17{k1}{z},ZMMWORD PTR [rdx+r10]
        vmovdqu8   zmm18{k1}{z},ZMMWORD PTR [rdx+r10*2]
        vmovdqu8   zmm19{k1}{z},ZMMWORD PTR [rdx+r13]
        vmovdqu64  ZMMWORD PTR [rcx],zmm16
        vmovdqu64  ZMMWORD PTR [rcx+r12],zmm17
        vmovdqu64  ZMMWORD PTR [rcx+r12*2],zmm18
        vmovdqu64  ZMMWORD PTR [rcx+rax],zmm19
        vpmaddubsw zmm16,zmm16,zmm31            # horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm16              # add words to row accumulators
        vpmaddubsw zmm17,zmm17,zmm31
        vpaddw     zmm1,zmm1,zmm17
        vpmaddubsw zmm18,zmm18,zmm31
        vpaddw     zmm2,zmm2,zmm18
        vpmaddubsw zmm19,zmm19,zmm31
        vpaddw     zmm3,zmm3,zmm19

//
// Reduce the sums for the four rows of output.
//

.LCopyPackA.ReduceRowSumBufferM4:
        vpmaddwd       zmm0,zmm0,zmm30           # horizontal word+word=dword per row
        vpmaddwd       zmm1,zmm1,zmm30
        vpmaddwd       zmm2,zmm2,zmm30
        vpmaddwd       zmm3,zmm3,zmm30
        vextracti64x4  ymm16,zmm0,1              # fold zmm -> ymm
        vextracti64x4  ymm17,zmm1,1
        vextracti64x4  ymm18,zmm2,1
        vextracti64x4  ymm19,zmm3,1
        vpaddd         ymm0,ymm0,ymm16
        vpaddd         ymm1,ymm1,ymm17
        vpaddd         ymm2,ymm2,ymm18
        vpaddd         ymm3,ymm3,ymm19
        vphaddd        ymm0,ymm0,ymm1           # reduce and interleave Sum1/Sum0
        vphaddd        ymm1,ymm2,ymm3           # reduce and interleave Sum3/Sum2
        vphaddd        ymm0,ymm0,ymm1           # reduce and interleave Sum3/Sum2/Sum1/Sum0
        vextracti128   xmm1,ymm0,1              # fold ymm -> xmm
        vpaddd         xmm0,xmm0,xmm1
        vmovdqu        XMMWORD PTR [r9],xmm0
        add     r9,4*4                          # advance row sum buffer by 4 dwords
        sub     r11,4                           # m -= 4
        jae     .LCopyPackA.ProcessNextRowM4

.LCopyPackA.ProcessRemainingRows:
        add     r11,4                           # correct for over-subtract above
        jz      .LCopyPackA.ExitRoutine

//
// Process a single row of matrix A in a loop.
//

.LCopyPackA.ProcessNextRowM1:
        vpxor   xmm0,xmm0,xmm0                  # clear row accumulator
        mov     rdx,rsi                         # src = A
        mov     rcx,rdi                         # dst = D
        add     rsi,r10                         # A to next row
        add     rdi,r12                         # D to next row
        mov     rbx,r8                          # k = CountK
        sub     rbx,64                          # k -= 64
        jb      .LCopyPackA.ProcessRemainingColumnsM1

.LCopyPackA.ProcessNextColumnLoopM1:
        vmovdqu64  zmm16,ZMMWORD PTR [rdx]
        vmovdqu64  ZMMWORD PTR [rcx],zmm16
        vpmaddubsw zmm16,zmm16,zmm31            # horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm16              # add words to row accumulators
        add     rdx,64                          # src += 64
        add     rcx,64                          # dst += 64
        sub     rbx,64                          # k -= 64
        jae     .LCopyPackA.ProcessNextColumnLoopM1

.LCopyPackA.ProcessRemainingColumnsM1:
        add     rbx,64                          # correct for over-subtract above
        jz      .LCopyPackA.ReduceRowSumBufferM1

        vmovdqu8   zmm16{k1}{z},ZMMWORD PTR [rdx]
        vmovdqu64  ZMMWORD PTR [rcx],zmm16
        vpmaddubsw zmm16,zmm16,zmm31            # horizontal byte+byte=word per row
        vpaddw     zmm0,zmm0,zmm16              # add words to row accumulators

//
// Reduce the sum for the single row of output.
//

.LCopyPackA.ReduceRowSumBufferM1:
        vpmaddwd       zmm0,zmm0,zmm30          # horizontal word+word=dword per row
        vextracti64x4  ymm16,zmm0,1             # fold zmm -> ymm
        vpaddd         ymm0,ymm0,ymm16
        vextracti128   xmm1,ymm0,1              # fold ymm -> xmm
        vpaddd         xmm0,xmm0,xmm1           # reduction
        vphaddd        xmm0,xmm0,xmm0
        vphaddd        xmm0,xmm0,xmm0
        vmovd          DWORD PTR [r9],xmm0
        add     r9,4                            # advance row sum buffer by 1 dword
        dec     r11                             # decrement rows remaining
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


        .end
