/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    ConvSymKernelAvx2.asm

Abstract:

    This module implements the kernels for the symmetric quantized integer
    convolution operation.

    This implementation uses AVX2 and AVX VNNI instructions.

--*/

#include "asmmacro.h"
#include "ConvSymKernelCommon.h"
#include "AssembleAvxVnni.h"

        .intel_syntax noprefix

/*++

Macro Description:

    This macro generates code to multiply and accumulate a single row of the
    output block.

Arguments:

    Vec1Reg - Supplies the low block accumulator register.

    Vec2Reg - Supplies the high block accumulator register.

Implicit Arguments:

    ymm0 - Supplies the first vector loaded from the filter buffer.

    ymm1 - Supplies the second vector loaded from the filter buffer.

    ymm2 - Supplies the broadcast value loaded from the input buffer.

    ymm3 - Supplies a scratch register for intermediate results.

    ymm12 - Supplies a 256-bit with the broadcasted word value 0x0001.

--*/

        .macro MultiplyAccumulateRowAvx2 Vec1Reg, Vec2Reg

        vpmaddubsw ymm3,ymm2,ymm0
        vpmaddwd ymm3,ymm3,ymm12
        vpaddd \Vec1Reg\(),\Vec1Reg\(),ymm3
        vpmaddubsw ymm2,ymm2,ymm1
        vpmaddwd ymm2,ymm2,ymm12
        vpaddd \Vec2Reg\(),\Vec2Reg\(),ymm2

        .endm

        .macro MultiplyAccumulateRowAvxVnni Vec1Reg, Vec2Reg

        VpdpbusdsYmmYmmYmm \Vec1Reg\(),ymm2,ymm0
        VpdpbusdsYmmYmmYmm \Vec2Reg\(),ymm2,ymm1

        .endm

/*++

Macro Description:

    This macro generates code to multiply and accumulate each row of the output
    block.

Arguments:

    Isa - Supplies the instruction set architecture string.

    RowCount - Supplies the number of rows to produce.

    VectorOffset - Supplies the byte offset from the filter to fetch elements.

    BroadcastOffset - Supplies the byte offset from the input to fetch elements.

Implicit Arguments:

    rdx - Supplies the address of the filter buffer.

    r10 - Supplies the address of the base of the input buffer.

Implicit Arguments (Avx2):

    r11-r13 - Supplies the relative byte offsets from the base of the input
        buffer to access the second through fourth rows.

    ymm4-ymm11 - Supplies the block accumulators.

    ymm12 - Supplies a 256-bit with the broadcasted word value 0x0001.

Implicit Arguments (AvxVnni):

    r11-r15 - Supplies the relative byte offsets from the base of the input
        buffer to access the second through sixth rows.

    ymm4-ymm15 - Supplies the block accumulators.

--*/

        .macro ComputeBlock Isa, RowCount, VectorOffset, BroadcastOffset

        vmovdqu ymm0,YMMWORD PTR [rdx+\VectorOffset\()]
        vmovdqu ymm1,YMMWORD PTR [rdx+\VectorOffset\()+32]
        EmitIfCountGE \RowCount\(),1,"vpbroadcastd ymm2,DWORD PTR [r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(),1,"MultiplyAccumulateRow\Isa\() ymm4,ymm5"
        EmitIfCountGE \RowCount\(),2,"vpbroadcastd ymm2,DWORD PTR [r10+r11+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(),2,"MultiplyAccumulateRow\Isa\() ymm6,ymm7"
        EmitIfCountGE \RowCount\(),3,"vpbroadcastd ymm2,DWORD PTR [r10+r12+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(),3,"MultiplyAccumulateRow\Isa\() ymm8,ymm9"
        EmitIfCountGE \RowCount\(),4,"vpbroadcastd ymm2,DWORD PTR [r10+r13+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(),4,"MultiplyAccumulateRow\Isa\() ymm10,ymm11"
        EmitIfCountGE \RowCount\(),5,"vpbroadcastd ymm2,DWORD PTR [r10+r14+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(),5,"MultiplyAccumulateRow\Isa\() ymm12,ymm13"
        EmitIfCountGE \RowCount\(),6,"vpbroadcastd ymm2,DWORD PTR [r10+r15+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(),6,"MultiplyAccumulateRow\Isa\() ymm14,ymm15"

        .endm

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple times
    and advancing the input and filter data pointers.

Arguments:

    Isa - Supplies the instruction set architecture string.

    RowCount - Supplies the number of rows to produce.

    UnrollLoop - Supplies a non-blank value if the loop should be unrolled to
        improve performance.

Implicit Arguments:

    rax - Supplies the number of input channels.

    rdx - Supplies the address of the filter buffer.

    r10 - Supplies the address of the base of the input buffer.

--*/

        .macro ComputeBlockLoop Isa, RowCount, UnrollLoop

.ifeqs "\UnrollLoop\()","UnrollLoop"
        sub     rax,4*4
        jb      .LProcessRemainingBlocks\@

.LComputeBlockBy4Loop\@:
        ComputeBlock \Isa\(),\RowCount\(),0*64,0
        ComputeBlock \Isa\(),\RowCount\(),1*64,4
        ComputeBlock \Isa\(),\RowCount\(),2*64,8
        ComputeBlock \Isa\(),\RowCount\(),3*64,12
        add     r10,4*4                     # advance input base address
        add     rdx,4*16*4                  # advance filter address
        sub     rax,4*4                     # decrement elements remaining
        jae     .LComputeBlockBy4Loop\@

.LProcessRemainingBlocks\@:
        add     rax,4*4                     # correct for over-subtract above
        jz      .LComputeBlockLoopExit\@
.endif

.LComputeBlockBy1Loop\@:
        ComputeBlock \Isa\(),\RowCount\(),0*64,0
        add     r10,4                       # advance input base address
        add     rdx,16*4                    # advance filter address
        sub     rax,4                       # decrement elements remaining
        jnz     .LComputeBlockBy1Loop\@

.LComputeBlockLoopExit\@:

        .endm

/*++

Macro Description:

    This macro generates code to convert the block accumulators from the matrix
    multiply loop to float values.

Arguments:

    RegList - Supplies the list of vector registers to operate on.

Implicit Arguments:

    ymm0 - Supplies the integer bias vector.

    ymm1 - Supplies the output scale vector.

--*/

        .macro ConvertAccumulatorToFloatRegList RegList

//
// Offset each value by the per-channel bias value, convert to floating point,
// and apply the output scale.
//

        EmitForEachRegister "\RegList\()","vpaddd \RegItem\(),\RegItem\(),ymm0"
        EmitForEachRegister "\RegList\()","vcvtdq2ps \RegItem\(),\RegItem\()"
        EmitForEachRegister "\RegList\()","vmulps \RegItem\(),\RegItem\(),ymm1"

        .endm

/*++

Macro Description:

    This macro generates code to convert float values to 32-bit integers in the
    range 0 to 255.

Arguments:

    RegList - Supplies the list of vector registers to operate on.

Implicit Arguments:

    ymm0 - Supplies the broadcasted minimum clip float value.

        This is set to static_cast<float>(0 - ZeroPointValue).

    ymm1 - Supplies the broadcasted maximum clip float value.

        This is set to static_cast<float>(255 - ZeroPointValue).

    ymm2 - Supplies the broadcasted zero point integer value.

--*/

        .macro ConvertFloatToIntegerRegList RegList

//
// Clip the float values to the integer range covered by the output zero point.
// This also keeps values outside the range INT_MIN to INT_MAX from converting
// to INT_MIN.
//

        EmitForEachRegister "\RegList\()","vmaxps \RegItem\(),\RegItem\(),ymm0"
        EmitForEachRegister "\RegList\()","vminps \RegItem\(),\RegItem\(),ymm1"

//
// Convert the float value to integer and add the zero point offset.
//

        EmitForEachRegister "\RegList\()","vcvtps2dq \RegItem\(),\RegItem\()"
        EmitForEachRegister "\RegList\()","vpaddd \RegItem\(),\RegItem\(),ymm2"

        .endm

/*++

Macro Description:

    This macro generates code for the inner kernel to compute a convolution
    for the elements of an output row for a set of filter rows.

Arguments:

    Isa - Supplies the instruction set architecture string.

--*/

        .macro ConvSymKernelFunction Isa

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

Arguments:

    Input (rdi) - Supplies the address of the input buffer.

        If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is set, then the input buffer points
        directly at the input tensor.

        If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is clear, then the input buffer is an
        indirection buffer. Every pointer in the indirection buffer points at a
        InputChannels length vector (either from the input tensor or a vector of
        padding values). These are grouped in batches of length KernelSize.
        These batches are then repeated OutputCount times.

    Filter (rsi) - Supplies the address of the filter buffer.

    Output (rdx) - Supplies the address of the output buffer.

    KernelSize (rcx) - Supplies the size of the kernel.

        If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is set, then kernel size should be 1.

    InputChannels (r8) - Supplies the number of input channels.

        This implementation requires the count to be a multiple of 4.

    OutputChannels (r9) - Supplies the number of output channels.

    ChannelCount - Supplies the number of channels this iteration produces.

        This implementation requires the count to be 8 or 16.

    OutputCount - Supplies the number of output elements this iteration produces.

.ifeqs "\Isa\()","AvxVnni"
        This implementation requires the count to be in the range 1 to 6.
.else
        This implementation requires the count to be in the range 1 to 4.
.endif

    PostProcessParams - Supplies the address of the post process parameter block.

    KernelFlags - Supplies additional flags controlling the operation.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvSymKernel\Isa\()

        push    rbp
        push    rbx
        push    r12
        push    r13
        sub     rsp,.LConvSymKernelFrame_SavedR13
.ifeqs "\Isa\()","AvxVnni"
        mov     .LConvSymKernelFrame_SavedR14[rsp],r14
        mov     .LConvSymKernelFrame_SavedR15[rsp],r15
.endif

        mov     .LConvSymKernelFrame_InputChannels[rsp],r8
        mov     .LConvSymKernelFrame_OutputChannels[rsp],r9
        mov     r8,rdx                      # shuffle registers to Windows ABI
        mov     r9,rcx
        mov     rcx,rdi
        mov     rdx,rsi

        lea     rdi,[r9*8]
        mov     ebx,DWORD PTR .LConvSymKernelFrame_OutputCount[rsp]
        mov     rsi,.LConvSymKernelFrame_InputChannels[rsp]
        mov     ebp,DWORD PTR .LConvSymKernelFrame_KernelFlags[rsp]
        vpxor   xmm4,xmm4,xmm4
        vpxor   xmm5,xmm5,xmm5
        vpxor   xmm6,xmm6,xmm6
        vpxor   xmm7,xmm7,xmm7
        vpxor   xmm8,xmm8,xmm8
        vpxor   xmm9,xmm9,xmm9
        vpxor   xmm10,xmm10,xmm10
        vpxor   xmm11,xmm11,xmm11
.ifeqs "\Isa\()","AvxVnni"
        vpxor   xmm12,xmm12,xmm12
        vpxor   xmm13,xmm13,xmm13
        vpxor   xmm14,xmm14,xmm14
        vpxor   xmm15,xmm15,xmm15
.else
        vpcmpeqw ymm12,ymm12,ymm12          # generate 256-bit word vector [0xFFFF]
        vpsrlw  ymm12,ymm12,15              # generate 256-bit word vector [0x0001]
.endif

//
// Process an input block of length InputChannels for each element of the kernel.
//

.LProcessNextInputBlock\@:
        test    bpl,MLAS_CONV_SYM_FLAG_INPUT_DIRECT
        jz      .LInputIndirection\@

//
// The input buffer points directly at the input data and this is effectively a
// GEMM operation (such as a pointwise convolution or an Im2Col transform).
//

.LInputDirect\@:
        xor     r10,r10
        mov     r11,rsi
        lea     r12,[r11+r11]
        lea     r13,[r12+r11]
.ifeqs "\Isa\()","AvxVnni"
        lea     r14,[r13+r11]
        lea     r15,[r14+r11]
.endif
        cmp     ebx,2
        cmovb   r11,r10                     # use first row if output count is small
        cmovbe  r12,r10
        cmp     ebx,4
        cmovb   r13,r10
.ifeqs "\Isa\()","AvxVnni"
        cmovbe  r14,r10
        cmp     ebx,6
        cmovb   r15,r10
.endif
        mov     r10,rcx
        jmp     .LComputeBlockLoopStart\@

.LInputIndirection\@:
        lea     r11,[rcx+rdi]
        lea     r12,[rcx+rdi*2]
        lea     r13,[r11+rdi*2]
.ifeqs "\Isa\()","AvxVnni"
        lea     r14,[r12+rdi*2]
        lea     r15,[r13+rdi*2]
.endif
        cmp     ebx,2
        cmovb   r11,rcx                     # use first row if output count is small
        cmovbe  r12,rcx
        cmp     ebx,4
        cmovb   r13,rcx
.ifeqs "\Isa\()","AvxVnni"
        cmovbe  r14,rcx
        cmp     ebx,6
        cmovb   r15,rcx
.endif
        mov     r10,QWORD PTR [rcx]
        mov     r11,QWORD PTR [r11]
        mov     r12,QWORD PTR [r12]
        mov     r13,QWORD PTR [r13]
.ifeqs "\Isa\()","AvxVnni"
        mov     r14,QWORD PTR [r14]
        mov     r15,QWORD PTR [r15]
.endif
        add     rcx,8                       # advance indirection buffer address
        sub     r11,r10                     # compute deltas from base address
        sub     r12,r10
        sub     r13,r10
.ifeqs "\Isa\()","AvxVnni"
        sub     r14,r10
        sub     r15,r10
.endif

.LComputeBlockLoopStart\@:
        mov     rax,rsi                     # reload input channels
        cmp     ebx,2                       # output count <= 2?
        jbe     .LComputeBlockLoopBy2\@
.ifeqs "\Isa\()","AvxVnni"
        cmp     ebx,4                       # output count <= 4?
        jbe     .LComputeBlockLoopBy4\@
        ComputeBlockLoop \Isa\(),6,UnrollLoop
.else
        ComputeBlockLoop \Isa\(),4,UnrollLoop
.endif

.LComputeBlockLoopDone\@:
        dec     r9                          # decrement input blocks remaining
        jnz     .LProcessNextInputBlock\@

//
// Apply the bias and convert the block accumulators to intermediate float values.
//

        mov     rdx,.LConvSymKernelFrame_PostProcessParams[rsp]
        mov     rsi,.LConvSymKernelFrame_OutputChannels[rsp]
        mov     r11d,DWORD PTR .LConvSymKernelFrame_ChannelCount[rsp]
        mov     rcx,.LConvSymPostProcessParams_Bias[rdx]
        mov     r9,.LConvSymPostProcessParams_Scale[rdx]
        lea     r10,[rsi*2+rsi]             # compute fourth row output offset
        add     r10,r8
        vmovdqu ymm0,YMMWORD PTR [rcx]      # load low bias vector
        test    bpl,MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        jz      .LBroadcastScaleValue\@
        vmovups ymm1,YMMWORD PTR [r9]       # load low scale vector
        jmp     .LConvertLowAccumulatorsToFloat\@

.LBroadcastScaleValue\@:
        vbroadcastss ymm1,DWORD PTR [r9]

.LConvertLowAccumulatorsToFloat\@:
.ifeqs "\Isa\()","AvxVnni"
        ConvertAccumulatorToFloatRegList "ymm4,ymm6,ymm8,ymm10,ymm12,ymm14"
.else
        ConvertAccumulatorToFloatRegList "ymm4,ymm6,ymm8,ymm10"
.endif
        cmp     r11d,8                      # output single vector?
        jbe     .LConvertFloatsToIntegers\@
        vmovdqu ymm0,YMMWORD PTR [rcx+8*4]  # load high bias vector
        test    bpl,MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        jz      .LConvertHighAccumulatorsToFloat\@
        vmovups ymm1,YMMWORD PTR [r9+8*4]   # load high scale vector

.LConvertHighAccumulatorsToFloat\@:
.ifeqs "\Isa\()","AvxVnni"
        ConvertAccumulatorToFloatRegList "ymm5,ymm7,ymm9,ymm11,ymm13,ymm15"
.else
        ConvertAccumulatorToFloatRegList "ymm5,ymm7,ymm9,ymm11"
.endif

//
// Convert the intermediate float values to 32-bit integers in the range 0 to 255.
//

.LConvertFloatsToIntegers\@:
        vbroadcastss ymm0,DWORD PTR .LConvSymPostProcessParams_MinimumValue[rdx]
        vbroadcastss ymm1,DWORD PTR .LConvSymPostProcessParams_MaximumValue[rdx]
        vpbroadcastd ymm2,DWORD PTR .LConvSymPostProcessParams_OutputZeroPoint[rdx]
.ifeqs "\Isa\()","AvxVnni"
        ConvertFloatToIntegerRegList "ymm4,ymm6,ymm8,ymm10,ymm12,ymm14"
.else
        ConvertFloatToIntegerRegList "ymm4,ymm6,ymm8,ymm10"
.endif
        cmp     r11d,8                      # output single vector?
        jbe     .LStoreQuantizedOutputBy8\@
.ifeqs "\Isa\()","AvxVnni"
        ConvertFloatToIntegerRegList "ymm5,ymm7,ymm9,ymm11,ymm13,ymm15"
.else
        ConvertFloatToIntegerRegList "ymm5,ymm7,ymm9,ymm11"
.endif

//
// Pack with saturation and store 16 bytes to the output buffer.
//

.LStoreQuantizedOutputBy16\@:
.ifeqs "\Isa\()","AvxVnni"
        cmp     ebx,5
        ja      .LStoreQuantizedOutput6By16\@
        je      .LStoreQuantizedOutput5By16\@
.endif
        cmp     ebx,3
        ja      .LStoreQuantizedOutput4By16\@
        je      .LStoreQuantizedOutput3By16\@
        cmp     ebx,1
        ja      .LStoreQuantizedOutput2By16\@
        jmp     .LStoreQuantizedOutput1By16\@

.ifeqs "\Isa\()","AvxVnni"
.LStoreQuantizedOutput6By16\@:
        vextracti128 xmm0,ymm14,1
        vpackusdw xmm14,xmm14,xmm0
        vextracti128 xmm1,ymm15,1
        vpackusdw xmm15,xmm15,xmm1
        vpackuswb xmm14,xmm14,xmm15
        vmovdqu XMMWORD PTR [r10+rsi*2],xmm14

.LStoreQuantizedOutput5By16\@:
        vextracti128 xmm0,ymm12,1
        vpackusdw xmm12,xmm12,xmm0
        vextracti128 xmm1,ymm13,1
        vpackusdw xmm13,xmm13,xmm1
        vpackuswb xmm12,xmm12,xmm13
        vmovdqu XMMWORD PTR [r10+rsi],xmm12
.endif

.LStoreQuantizedOutput4By16\@:
        vextracti128 xmm0,ymm10,1
        vpackusdw xmm10,xmm10,xmm0
        vextracti128 xmm1,ymm11,1
        vpackusdw xmm11,xmm11,xmm1
        vpackuswb xmm10,xmm10,xmm11
        vmovdqu XMMWORD PTR [r10],xmm10

.LStoreQuantizedOutput3By16\@:
        vextracti128 xmm0,ymm8,1
        vpackusdw xmm8,xmm8,xmm0
        vextracti128 xmm1,ymm9,1
        vpackusdw xmm9,xmm9,xmm1
        vpackuswb xmm8,xmm8,xmm9
        vmovdqu XMMWORD PTR [r8+rsi*2],xmm8

.LStoreQuantizedOutput2By16\@:
        vextracti128 xmm0,ymm6,1
        vpackusdw xmm6,xmm6,xmm0
        vextracti128 xmm1,ymm7,1
        vpackusdw xmm7,xmm7,xmm1
        vpackuswb xmm6,xmm6,xmm7
        vmovdqu XMMWORD PTR [r8+rsi],xmm6

.LStoreQuantizedOutput1By16\@:
        vextracti128 xmm0,ymm4,1
        vpackusdw xmm4,xmm4,xmm0
        vextracti128 xmm1,ymm5,1
        vpackusdw xmm5,xmm5,xmm1
        vpackuswb xmm4,xmm4,xmm5
        vmovdqu XMMWORD PTR [r8],xmm4

//
// Restore non-volatile registers and return.
//

.LExitKernel\@:
        vzeroupper
.ifeqs "\Isa\()","AvxVnni"
        mov     r14,.LConvSymKernelFrame_SavedR14[rsp]
        mov     r15,.LConvSymKernelFrame_SavedR15[rsp]
.endif
        add     rsp,.LConvSymKernelFrame_SavedR13
        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret

//
// Pack with saturation and store 8 bytes to the output buffer.
//

.LStoreQuantizedOutputBy8\@:
.ifeqs "\Isa\()","AvxVnni"
        cmp     ebx,5
        ja      .LStoreQuantizedOutput6By8\@
        je      .LStoreQuantizedOutput5By8\@
.endif
        cmp     ebx,3
        ja      .LStoreQuantizedOutput4By8\@
        je      .LStoreQuantizedOutput3By8\@
        cmp     ebx,1
        ja      .LStoreQuantizedOutput2By8\@
        jmp     .LStoreQuantizedOutput1By8\@

.ifeqs "\Isa\()","AvxVnni"
.LStoreQuantizedOutput6By8\@:
        vextracti128 xmm0,ymm14,1
        vpackusdw xmm14,xmm14,xmm0
        vpackuswb xmm14,xmm14,xmm14
        vmovq   QWORD PTR [r10+rsi*2],xmm14

.LStoreQuantizedOutput5By8\@:
        vextracti128 xmm0,ymm12,1
        vpackusdw xmm12,xmm12,xmm0
        vpackuswb xmm12,xmm12,xmm12
        vmovq   QWORD PTR [r10+rsi],xmm12
.endif

.LStoreQuantizedOutput4By8\@:
        vextracti128 xmm0,ymm10,1
        vpackusdw xmm10,xmm10,xmm0
        vpackuswb xmm10,xmm10,xmm10
        vmovq   QWORD PTR [r10],xmm10

.LStoreQuantizedOutput3By8\@:
        vextracti128 xmm0,ymm8,1
        vpackusdw xmm8,xmm8,xmm0
        vpackuswb xmm8,xmm8,xmm8
        vmovq   QWORD PTR [r8+rsi*2],xmm8

.LStoreQuantizedOutput2By8\@:
        vextracti128 xmm0,ymm6,1
        vpackusdw xmm6,xmm6,xmm0
        vpackuswb xmm6,xmm6,xmm6
        vmovq   QWORD PTR [r8+rsi],xmm6

.LStoreQuantizedOutput1By8\@:
        vextracti128 xmm0,ymm4,1
        vpackusdw xmm4,xmm4,xmm0
        vpackuswb xmm4,xmm4,xmm4
        vmovq   QWORD PTR [r8],xmm4
        jmp     .LExitKernel\@

//
// Process the tail output counts out of line with a reduced block size.
//

.ifeqs "\Isa\()","AvxVnni"
.LComputeBlockLoopBy4\@:
        ComputeBlockLoop \Isa\(),4
        jmp     .LComputeBlockLoopDone\@
.endif

.LComputeBlockLoopBy2\@:
        ComputeBlockLoop \Isa\(),2
        jmp     .LComputeBlockLoopDone\@

        .endm

/*++

Macro Description:

    This macro generates code to multiply and accumulate a single cell of the
    output block.

Arguments:

    AccumReg - Supplies the register to accumulate into.

    Mult1Reg - Supplies the first multiplication operand register. This register
        may be trashed on return.

    Mult2Reg - Supplies the second multiplication operand register.

--*/

        .macro DepthwiseMultiplyAccumulateCellAvx2 AccumReg, Mult1Reg, Mult2Reg

        vpmaddwd \Mult1Reg\(),\Mult1Reg\(),\Mult2Reg\()
        vpaddd  \AccumReg\(),\AccumReg\(),\Mult1Reg\()

        .endm

        .macro DepthwiseMultiplyAccumulateCellAvxVnni AccumReg, Mult1Reg, Mult2Reg

        VpdpbusdsYmmYmmYmm \AccumReg\(),\Mult1Reg\(),\Mult2Reg\()

        .endm

/*++

Macro Description:

    This macro generates code for the inner kernel to compute a depthwise
    convolution for the elements of an output row for a set of filter rows.

Arguments:

    Isa - Supplies the instruction set architecture string.

--*/

        .macro ConvSymDepthwiseKernelFunction Isa

/*++

Routine Description:

    This routine is the inner kernel to compute a depthwise convolution for the
    elements of an output row for a set of filter rows.

Arguments:

    Input (rdi) - Supplies the address of the indirection buffer.

    Filter (rsi) - Supplies the address of the filter buffer.

    Output (rdx) - Supplies the address of the output buffer.

    KernelSize (rcx) - Supplies the size of the kernel.

    Channels (r8) - Supplies the number of input and output channels.

    ChannelOffset (r9) - Supplies the byte offset from the indirection buffer base
        address for this iteration.

    ChannelCount - Supplies the number of channels this iteration produces.

        This implementation requires the count to be 16.

    OutputCount - Supplies the number of output elements this iteration produces.

        This implementation requires the count to be in the range 1 to 4.

    PostProcessParams - Supplies the address of the post process parameter block.

    KernelFlags - Supplies additional flags controlling the operation.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvSymDepthwiseKernel\Isa\()

        push    rbp
        push    rbx
        push    r12
        push    r13
        sub     rsp,.LConvSymDepthwiseKernelFrame_SavedR13

        mov     .LConvSymDepthwiseKernelFrame_Channels[rsp],r8
        mov     .LConvSymDepthwiseKernelFrame_ChannelOffset[rsp],r9
        mov     r8,rdx                      # shuffle registers to Windows ABI
        mov     r9,rcx
        mov     rcx,rdi
        mov     rdx,rsi

        lea     rdi,[r9*8]
        mov     ebx,DWORD PTR .LConvSymDepthwiseKernelFrame_OutputCount[rsp]
        mov     rsi,.LConvSymDepthwiseKernelFrame_Channels[rsp]
        mov     rax,.LConvSymDepthwiseKernelFrame_ChannelOffset[rsp]
        mov     ebp,DWORD PTR .LConvSymDepthwiseKernelFrame_KernelFlags[rsp]
        vpxor   xmm4,xmm4,xmm4
        vpxor   xmm5,xmm5,xmm5
        vpxor   xmm6,xmm6,xmm6
        vpxor   xmm7,xmm7,xmm7
        vpxor   xmm8,xmm8,xmm8
        vpxor   xmm9,xmm9,xmm9
        vpxor   xmm10,xmm10,xmm10
        vpxor   xmm11,xmm11,xmm11

//
// Process an input block of length Channels for each element of the kernel.
//

.LProcessNextInputBlock\@:
        vpmovsxbd ymm0,QWORD PTR [rdx]
        vpmovsxbd ymm1,QWORD PTR [rdx+8]
        lea     r11,[rcx+rdi]
        lea     r12,[rcx+rdi*2]
        lea     r13,[r11+rdi*2]
        cmp     ebx,2
        cmovb   r11,rcx                     # use first row if output count is small
        cmovbe  r12,rcx
        cmp     ebx,4
        cmovb   r13,rcx
        mov     r10,QWORD PTR [rcx]
        mov     r11,QWORD PTR [r11]
        mov     r12,QWORD PTR [r12]
        mov     r13,QWORD PTR [r13]
        add     rcx,8                       # advance indirection buffer address
        vpmovzxbd ymm2,QWORD PTR [r10+rax]
        vpmovzxbd ymm3,QWORD PTR [r10+rax+8]
        DepthwiseMultiplyAccumulateCell\Isa\() ymm4,ymm2,ymm0
        vpmovzxbd ymm2,QWORD PTR [r11+rax]
        DepthwiseMultiplyAccumulateCell\Isa\() ymm5,ymm3,ymm1
        vpmovzxbd ymm3,QWORD PTR [r11+rax+8]
        DepthwiseMultiplyAccumulateCell\Isa\() ymm6,ymm2,ymm0
        vpmovzxbd ymm2,QWORD PTR [r12+rax]
        DepthwiseMultiplyAccumulateCell\Isa\() ymm7,ymm3,ymm1
        vpmovzxbd ymm3,QWORD PTR [r12+rax+8]
        DepthwiseMultiplyAccumulateCell\Isa\() ymm8,ymm2,ymm0
        vpmovzxbd ymm2,QWORD PTR [r13+rax]
        DepthwiseMultiplyAccumulateCell\Isa\() ymm9,ymm3,ymm1
        vpmovzxbd ymm3,QWORD PTR [r13+rax+8]
        DepthwiseMultiplyAccumulateCell\Isa\() ymm10,ymm2,ymm0
        add     rdx,rsi                     # advance filter to next kernel
        DepthwiseMultiplyAccumulateCell\Isa\() ymm11,ymm3,ymm1
        dec     r9                          # decrement input blocks remaining
        jnz     .LProcessNextInputBlock\@

//
// Apply the bias and convert the block accumulators to intermediate float values.
//

        mov     rdx,.LConvSymDepthwiseKernelFrame_PostProcessParams[rsp]
        mov     rcx,.LConvSymPostProcessParams_Bias[rdx]
        mov     r9,.LConvSymPostProcessParams_Scale[rdx]
        vmovdqu ymm0,YMMWORD PTR [rcx]      # load low bias vector
        test    bpl,MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        jz      .LBroadcastScaleValue\@
        vmovups ymm1,YMMWORD PTR [r9]       # load low scale vector
        jmp     .LConvertLowAccumulatorsToFloat\@

.LBroadcastScaleValue\@:
        vbroadcastss ymm1,DWORD PTR [r9]

.LConvertLowAccumulatorsToFloat\@:
        ConvertAccumulatorToFloatRegList "ymm4,ymm6,ymm8,ymm10"
        vmovdqu ymm0,YMMWORD PTR [rcx+8*4]  # load high bias vector
        test    bpl,MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        jz      .LConvertHighAccumulatorsToFloat\@
        vmovups ymm1,YMMWORD PTR [r9+8*4]   # load high scale vector

.LConvertHighAccumulatorsToFloat\@:
        ConvertAccumulatorToFloatRegList "ymm5,ymm7,ymm9,ymm11"

//
// Convert the intermediate float values to 32-bit integers in the range 0 to 255.
//

.LConvertFloatsToIntegers\@:
        vbroadcastss ymm0,DWORD PTR .LConvSymPostProcessParams_MinimumValue[rdx]
        vbroadcastss ymm1,DWORD PTR .LConvSymPostProcessParams_MaximumValue[rdx]
        vpbroadcastd ymm2,DWORD PTR .LConvSymPostProcessParams_OutputZeroPoint[rdx]
        ConvertFloatToIntegerRegList "ymm4,ymm6,ymm8,ymm10"
        ConvertFloatToIntegerRegList "ymm5,ymm7,ymm9,ymm11"

//
// Pack with saturation and store 16 bytes to the output buffer.
//

.LStoreQuantizedOutputBy16\@:
        lea     r10,[rsi*2+rsi]
        cmp     ebx,3
        ja      .LStoreQuantizedOutput4By16\@
        je      .LStoreQuantizedOutput3By16\@
        cmp     ebx,1
        ja      .LStoreQuantizedOutput2By16\@
        jmp     .LStoreQuantizedOutput1By16\@

.LStoreQuantizedOutput4By16\@:
        vextracti128 xmm0,ymm10,1
        vpackusdw xmm10,xmm10,xmm0
        vextracti128 xmm1,ymm11,1
        vpackusdw xmm11,xmm11,xmm1
        vpackuswb xmm10,xmm10,xmm11
        vmovdqu XMMWORD PTR [r8+r10],xmm10

.LStoreQuantizedOutput3By16\@:
        vextracti128 xmm0,ymm8,1
        vpackusdw xmm8,xmm8,xmm0
        vextracti128 xmm1,ymm9,1
        vpackusdw xmm9,xmm9,xmm1
        vpackuswb xmm8,xmm8,xmm9
        vmovdqu XMMWORD PTR [r8+rsi*2],xmm8

.LStoreQuantizedOutput2By16\@:
        vextracti128 xmm0,ymm6,1
        vpackusdw xmm6,xmm6,xmm0
        vextracti128 xmm1,ymm7,1
        vpackusdw xmm7,xmm7,xmm1
        vpackuswb xmm6,xmm6,xmm7
        vmovdqu XMMWORD PTR [r8+rsi],xmm6

.LStoreQuantizedOutput1By16\@:
        vextracti128 xmm0,ymm4,1
        vpackusdw xmm4,xmm4,xmm0
        vextracti128 xmm1,ymm5,1
        vpackusdw xmm5,xmm5,xmm1
        vpackuswb xmm4,xmm4,xmm5
        vmovdqu XMMWORD PTR [r8],xmm4

//
// Restore non-volatile registers and return.
//

.LExitKernel\@:
        vzeroupper
        add     rsp,.LConvSymDepthwiseKernelFrame_SavedR13
        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret

        .endm

//
// Generate the convolution kernels.
//

ConvSymKernelFunction Avx2
ConvSymDepthwiseKernelFunction Avx2

ConvSymKernelFunction AvxVnni
ConvSymDepthwiseKernelFunction AvxVnni

        .end
