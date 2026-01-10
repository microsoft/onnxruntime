/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    ConvSymKernelAvx512Core.asm

Abstract:

    This module implements the kernels for the symmetric quantized integer
    convolution operation.

    This implementation uses AVX512 core (BW/DQ/VL) and AVX512 VNNI instructions.

--*/

#include "asmmacro.h"
#include "ConvSymKernelCommon.h"
#include "AssembleAvx512Vnni.h"

        .intel_syntax noprefix

/*++

Macro Description:

    This macro generates code to setup registers that is common between
    convolution kernel types.

Arguments:

    Isa - Supplies the instruction set architecture string.

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

Implicit Arguments:

    rcx - Supplies the address of the input buffer.

    r9 - Supplies the size of the kernel.

Output:

    rbx - Supplies the address of the input buffer.

    rdi - Supplies the input indirection buffer stride.

.ifeqs <Isa>, <Avx512Core>
    zmm7 - Supplies a 512-bit with the broadcasted word value 0x0001.
.endif

    zmm8-zmm31 - Supplies the zeroed block accumulators.

    k1-k4 - Supplies the opmask registers loaded with a 64-bit channel bitmask
        for KernelFrame.ChannelCount.

--*/

        .macro SetupRegistersCommon Isa, KernelFrame

        mov     rbx,rcx                     # preserve base input address
        lea     rdi,[r9*8]                  # indirection buffer offset to next output
.ifeqs "\Isa\()","Avx512Core"
        mov     esi,1
        vpbroadcastw zmm7,esi               # generate 512-bit word vector [0x0001]
.endif
        EmitForEachRegister "zmm8,zmm9,zmm10,zmm11","vpxord \RegItem\(),\RegItem\(),\RegItem\()"
        mov     ecx,DWORD PTR \KernelFrame\()_ChannelCount[rsp]
        EmitForEachRegister "zmm12,zmm13,zmm14,zmm15","vpxord \RegItem\(),\RegItem\(),\RegItem\()"
        dec     ecx                         # convert shift count to 0..63
        mov     eax,2
        shl     rax,cl                      # compute 2 << ChannelShiftCount
        dec     rax                         # convert to 64-bit channel bitmask
        EmitForEachRegister "zmm16,zmm17,zmm18,zmm19","vpxord \RegItem\(),\RegItem\(),\RegItem\()"
        kmovw   k1,eax                      # k1 = channel bitmask[0..15]
        shr     rax,16
        EmitForEachRegister "zmm20,zmm21,zmm22,zmm23","vpxord \RegItem\(),\RegItem\(),\RegItem\()"
        kmovw   k2,eax                      # k2 = channel bitmask[16..31]
        shr     rax,16
        EmitForEachRegister "zmm24,zmm25,zmm26,zmm27","vpxord \RegItem\(),\RegItem\(),\RegItem\()"
        kmovw   k3,eax                      # k3 = channel bitmask[32..47]
        shr     eax,16
        EmitForEachRegister "zmm28,zmm29,zmm30,zmm31","vpxord \RegItem\(),\RegItem\(),\RegItem\()"
        kmovw   k4,eax                      # k4 = channel bitmask[48..63]

        .endm

/*++

Macro Description:

    This macro generates code to multiply and accumulate a single cell of the
    output block.

Arguments:

    AccumReg - Supplies the register to accumulate into.

    Mult1Reg - Supplies the first multiplication operand register.

    Mult2Reg - Supplies the second multiplication operand register.

Implicit Arguments:

    zmm5 - Supplies a scratch register for intermediate results.

    zmm7 - Supplies a 512-bit with the broadcasted word value 0x0001.

--*/

        .macro MultiplyAccumulateCellAvx512Core AccumReg, Mult1Reg, Mult2Reg

        vpmaddubsw zmm5,\Mult1Reg\(),\Mult2Reg\()
        vpmaddwd zmm5,zmm5,zmm7
        vpaddd  \AccumReg\(),\AccumReg\(),zmm5

        .endm

        .macro MultiplyAccumulateCellAvx512Vnni AccumReg, Mult1Reg, Mult2Reg

        VpdpbusdsZmmZmmZmm \AccumReg\(),\Mult1Reg\(),\Mult2Reg\()

        .endm

/*++

Macro Description:

    This macro generates code to multiply and accumulate each row of the output
    block.

Arguments:

    Isa - Supplies the instruction set architecture string.

    ColumnCount - Supplies the number of columns to produce.

    VectorOffset - Supplies the byte offset from the filter to fetch elements.

    BroadcastOffset - Supplies the byte offset from the input to fetch elements.

Implicit Arguments:

    rdx - Supplies the address of the filter buffer.

    rsi - Supplies the filter stride to access the packed data for the next 16
        output channels.

    rbp - Supplies three times the above filter stride.

    r10 - Supplies the address of the base of the input buffer.

    r11-r15 - Supplies the relative byte offsets from the base of the input
        buffer to access the second through sixth rows.

    zmm8-zmm31 - Supplies the block accumulators.

--*/

        .macro ComputeBlock Isa, ColumnCount, VectorOffset, BroadcastOffset

        EmitIfCountGE \ColumnCount\(),16,"vmovdqu32 zmm0,ZMMWORD PTR [rdx+\VectorOffset\()]"
        EmitIfCountGE \ColumnCount\(),32,"vmovdqu32 zmm1,ZMMWORD PTR [rdx+rsi+\VectorOffset\()]"
        EmitIfCountGE \ColumnCount\(),48,"vmovdqu32 zmm2,ZMMWORD PTR [rdx+rsi*2+\VectorOffset\()]"
        EmitIfCountGE \ColumnCount\(),64,"vmovdqu32 zmm3,ZMMWORD PTR [rdx+rbp+\VectorOffset\()]"
        vpbroadcastd zmm4,DWORD PTR [r10+\BroadcastOffset\()]
        EmitIfCountGE \ColumnCount\(),16,"MultiplyAccumulateCell\Isa\() zmm8,zmm4,zmm0"
        EmitIfCountGE \ColumnCount\(),32,"MultiplyAccumulateCell\Isa\() zmm9,zmm4,zmm1"
        EmitIfCountGE \ColumnCount\(),48,"MultiplyAccumulateCell\Isa\() zmm10,zmm4,zmm2"
        EmitIfCountGE \ColumnCount\(),64,"MultiplyAccumulateCell\Isa\() zmm11,zmm4,zmm3"
        vpbroadcastd zmm4,DWORD PTR [r10+r11+\BroadcastOffset\()]
        EmitIfCountGE \ColumnCount\(),16,"MultiplyAccumulateCell\Isa\() zmm12,zmm4,zmm0"
        EmitIfCountGE \ColumnCount\(),32,"MultiplyAccumulateCell\Isa\() zmm13,zmm4,zmm1"
        EmitIfCountGE \ColumnCount\(),48,"MultiplyAccumulateCell\Isa\() zmm14,zmm4,zmm2"
        EmitIfCountGE \ColumnCount\(),64,"MultiplyAccumulateCell\Isa\() zmm15,zmm4,zmm3"
        vpbroadcastd zmm4,DWORD PTR [r10+r12+\BroadcastOffset\()]
        EmitIfCountGE \ColumnCount\(),16,"MultiplyAccumulateCell\Isa\() zmm16,zmm4,zmm0"
        EmitIfCountGE \ColumnCount\(),32,"MultiplyAccumulateCell\Isa\() zmm17,zmm4,zmm1"
        EmitIfCountGE \ColumnCount\(),48,"MultiplyAccumulateCell\Isa\() zmm18,zmm4,zmm2"
        EmitIfCountGE \ColumnCount\(),64,"MultiplyAccumulateCell\Isa\() zmm19,zmm4,zmm3"
        vpbroadcastd zmm4,DWORD PTR [r10+r13+\BroadcastOffset\()]
        EmitIfCountGE \ColumnCount\(),16,"MultiplyAccumulateCell\Isa\() zmm20,zmm4,zmm0"
        EmitIfCountGE \ColumnCount\(),32,"MultiplyAccumulateCell\Isa\() zmm21,zmm4,zmm1"
        EmitIfCountGE \ColumnCount\(),48,"MultiplyAccumulateCell\Isa\() zmm22,zmm4,zmm2"
        EmitIfCountGE \ColumnCount\(),64,"MultiplyAccumulateCell\Isa\() zmm23,zmm4,zmm3"
        vpbroadcastd zmm4,DWORD PTR [r10+r14+\BroadcastOffset\()]
        EmitIfCountGE \ColumnCount\(),16,"MultiplyAccumulateCell\Isa\() zmm24,zmm4,zmm0"
        EmitIfCountGE \ColumnCount\(),32,"MultiplyAccumulateCell\Isa\() zmm25,zmm4,zmm1"
        EmitIfCountGE \ColumnCount\(),48,"MultiplyAccumulateCell\Isa\() zmm26,zmm4,zmm2"
        EmitIfCountGE \ColumnCount\(),64,"MultiplyAccumulateCell\Isa\() zmm27,zmm4,zmm3"
        vpbroadcastd zmm4,DWORD PTR [r10+r15+\BroadcastOffset\()]
        EmitIfCountGE \ColumnCount\(),16,"MultiplyAccumulateCell\Isa\() zmm28,zmm4,zmm0"
        EmitIfCountGE \ColumnCount\(),32,"MultiplyAccumulateCell\Isa\() zmm29,zmm4,zmm1"
        EmitIfCountGE \ColumnCount\(),48,"MultiplyAccumulateCell\Isa\() zmm30,zmm4,zmm2"
        EmitIfCountGE \ColumnCount\(),64,"MultiplyAccumulateCell\Isa\() zmm31,zmm4,zmm3"

        .endm

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple times
    and advancing the input and filter data pointers.

Arguments:

    Isa - Supplies the instruction set architecture string.

    ColumnCount - Supplies the number of columns to produce.

Implicit Arguments:

    rax - Supplies the number of byte elements to process (multiple of 4).

    rdx - Supplies the address of the filter buffer.

    rsi - Supplies the filter stride to access the packed data for the next 16
        output channels.

    rbp - Supplies three times the above filter stride.

    r10 - Supplies the address of the base of the input buffer.

    r11-r15 - Supplies the relative byte offsets from the base of the input
        buffer to access the second through sixth rows.

    zmm8-zmm31 - Supplies the block accumulators.

--*/

        .macro ComputeBlockLoop Isa, ColumnCount

.LComputeBlockBy1Loop\@:
        ComputeBlock \Isa\(),\ColumnCount\(),0*64,0
        add     r10,4                       # advance input base address
        add     rdx,16*4                    # advance filter address
        sub     rax,4                       # decrement elements remaining
        jnz     .LComputeBlockBy1Loop\@

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

        This implementation requires the count to be in the range 1 to 64.

    OutputCount - Supplies the number of output elements this iteration produces.

        This implementation requires the count to be in the range 1 to 6.

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
        push    r14
        push    r15
        sub     rsp,.LConvSymKernelFrame_SavedR15

        mov     .LConvSymKernelFrame_InputChannels[rsp],r8
        mov     .LConvSymKernelFrame_OutputChannels[rsp],r9
        mov     r8,rdx                      # shuffle registers to Windows ABI
        mov     r9,rcx
        mov     rcx,rdi
        mov     rdx,rsi

        SetupRegistersCommon \Isa\(),.LConvSymKernelFrame

        mov     rsi,.LConvSymKernelFrame_InputChannels[rsp]
        mov     ecx,DWORD PTR .LConvSymKernelFrame_ChannelCount[rsp]
        shl     rsi,4                       # 16 output channels per filter block
        imul    rsi,r9                      # compute filter stride
        lea     rbp,[rsi*2+rsi]

//
// Process an input block of length InputChannels for each element of the kernel.
//
// To keep code size small, this kernel always computes a fixed number of output
// rows. If the output count is less than this fixed number, then the first row
// is duplicated into the unused slots and the results are discarded.
//

.LProcessNextInputBlock\@:
        mov     eax,DWORD PTR .LConvSymKernelFrame_OutputCount[rsp]
        test    BYTE PTR .LConvSymKernelFrame_KernelFlags[rsp],MLAS_CONV_SYM_FLAG_INPUT_DIRECT
        jz      .LInputIndirection\@

//
// The input buffer points directly at the input data and this is effectively a
// GEMM operation (such as a pointwise convolution or an Im2Col transform).
//

.LInputDirect\@:
        xor     r10,r10
        mov     r11,.LConvSymKernelFrame_InputChannels[rsp]
        lea     r12,[r11+r11]
        lea     r13,[r12+r11]
        lea     r14,[r13+r11]
        lea     r15,[r14+r11]
        cmp     eax,2
        cmovb   r11,r10                     # use first row if output count is small
        cmovbe  r12,r10
        cmp     eax,4
        cmovb   r13,r10
        cmovbe  r14,r10
        cmp     eax,6
        cmovb   r15,r10
        mov     r10,rbx
        jmp     .LComputeBlockLoopStart\@

.LInputIndirection\@:
        lea     r11,[rbx+rdi]
        lea     r12,[rbx+rdi*2]
        lea     r13,[r11+rdi*2]
        lea     r14,[r12+rdi*2]
        lea     r15,[r13+rdi*2]
        cmp     eax,2
        cmovb   r11,rbx                     # use first row if output count is small
        cmovbe  r12,rbx
        cmp     eax,4
        cmovb   r13,rbx
        cmovbe  r14,rbx
        cmp     eax,6
        cmovb   r15,rbx
        mov     r10,QWORD PTR [rbx]
        mov     r11,QWORD PTR [r11]
        mov     r12,QWORD PTR [r12]
        mov     r13,QWORD PTR [r13]
        mov     r14,QWORD PTR [r14]
        mov     r15,QWORD PTR [r15]
        add     rbx,8                       # advance indirection buffer address
        sub     r11,r10                     # compute deltas from base address
        sub     r12,r10
        sub     r13,r10
        sub     r14,r10
        sub     r15,r10

.LComputeBlockLoopStart\@:
        mov     rax,.LConvSymKernelFrame_InputChannels[rsp]
        cmp     ecx,16
        jbe     .LComputeBlockLoopBy16\@
        cmp     ecx,32
        jbe     .LComputeBlockLoopBy32\@
        cmp     ecx,48
        jbe     .LComputeBlockLoopBy48\@

.LComputeBlockLoopBy64\@:
        ComputeBlockLoop \Isa\(),64
        jmp     .LComputeBlockLoopDone\@

.LComputeBlockLoopBy48\@:
        ComputeBlockLoop \Isa\(),48
        jmp     .LComputeBlockLoopDone\@

.LComputeBlockLoopBy32\@:
        ComputeBlockLoop \Isa\(),32
        jmp     .LComputeBlockLoopDone\@

.LComputeBlockLoopBy16\@:
        ComputeBlockLoop \Isa\(),16

.LComputeBlockLoopDone\@:
        dec     r9                          # decrement input blocks remaining
        jnz     .LProcessNextInputBlock\@

//
// Post-process the block accumulators.
//

        mov     ebx,DWORD PTR .LConvSymKernelFrame_OutputCount[rsp]
        mov     rsi,.LConvSymKernelFrame_OutputChannels[rsp]
        mov     rdx,.LConvSymKernelFrame_PostProcessParams[rsp]
        mov     ebp,DWORD PTR .LConvSymKernelFrame_KernelFlags[rsp]
        call    MlasConvSymPostProcessAvx512Core

//
// Restore non-volatile registers and return.
//

.LExitKernel\@:
        vzeroupper
        add     rsp,.LConvSymKernelFrame_SavedR15
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret

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

    Input (rdi) - Supplies the address of the input indirection buffer.

    Filter (rsi) - Supplies the address of the filter buffer.

    Output (rdx) - Supplies the address of the output buffer.

    KernelSize (rcx) - Supplies the size of the kernel.

    Channels (r8) - Supplies the number of input and output channels.

    ChannelOffset (r9) - Supplies the byte offset from the indirection buffer base
        address for this iteration.

    ChannelCount - Supplies the number of channels this iteration produces.

        This implementation requires the count to be in the range 1 to 64.

    OutputCount - Supplies the number of output elements this iteration produces.

        This implementation requires the count to be in the range 1 to 6.

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
        push    r14
        push    r15
        sub     rsp,.LConvSymDepthwiseKernelFrame_SavedR15

        mov     .LConvSymDepthwiseKernelFrame_Channels[rsp],r8
        mov     .LConvSymDepthwiseKernelFrame_ChannelOffset[rsp],r9
        mov     r8,rdx                      # shuffle registers to Windows ABI
        mov     r9,rcx
        mov     rcx,rdi
        mov     rdx,rsi

        SetupRegistersCommon \Isa\(),.LConvSymDepthwiseKernelFrame

        mov     rsi,.LConvSymDepthwiseKernelFrame_Channels[rsp]
        mov     ebp,DWORD PTR .LConvSymDepthwiseKernelFrame_OutputCount[rsp]
        mov     rax,.LConvSymDepthwiseKernelFrame_ChannelOffset[rsp]
        mov     ecx,DWORD PTR .LConvSymDepthwiseKernelFrame_ChannelCount[rsp]

//
// Process an input block of length Channels for each element of the kernel.
//
// To keep code size small, this kernel always computes a fixed number of output
// rows. If the output count is less than this fixed number, then the first row
// is duplicated into the unused slots and the results are discarded.
//

.LProcessNextInputBlock\@:
        lea     r11,[rbx+rdi]
        lea     r12,[rbx+rdi*2]
        lea     r13,[r11+rdi*2]
        lea     r14,[r12+rdi*2]
        lea     r15,[r13+rdi*2]
        cmp     ebp,2
        cmovb   r11,rbx                     # use first row if output count is small
        cmovbe  r12,rbx
        cmp     ebp,4
        cmovb   r13,rbx
        cmovbe  r14,rbx
        cmp     ebp,6
        cmovb   r15,rbx
        mov     r10,QWORD PTR [rbx]
        mov     r11,QWORD PTR [r11]
        mov     r12,QWORD PTR [r12]
        mov     r13,QWORD PTR [r13]
        mov     r14,QWORD PTR [r14]
        mov     r15,QWORD PTR [r15]
        add     rbx,8
        cmp     ecx,16
        jbe     .LComputeDepthwiseBlockBy16\@
        cmp     ecx,32
        jbe     .LComputeDepthwiseBlockBy32\@
        cmp     ecx,48
        jbe     .LComputeDepthwiseBlockBy48\@

.LComputeDepthwiseBlockBy64\@:
        vpmovzxbd zmm2{k4}{z},XMMWORD PTR [rdx+3*16]
        vpmovzxbd zmm0{k4}{z},XMMWORD PTR [r10+rax+3*16]
        vpmovzxbd zmm1{k4}{z},XMMWORD PTR [r11+rax+3*16]
        MultiplyAccumulateCell\Isa\() zmm11,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm15,zmm1,zmm2
        vpmovzxbd zmm0{k4}{z},XMMWORD PTR [r12+rax+3*16]
        vpmovzxbd zmm1{k4}{z},XMMWORD PTR [r13+rax+3*16]
        MultiplyAccumulateCell\Isa\() zmm19,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm23,zmm1,zmm2
        vpmovzxbd zmm0{k4}{z},XMMWORD PTR [r14+rax+3*16]
        vpmovzxbd zmm1{k4}{z},XMMWORD PTR [r15+rax+3*16]
        MultiplyAccumulateCell\Isa\() zmm27,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm31,zmm1,zmm2

.LComputeDepthwiseBlockBy48\@:
        vpmovzxbd zmm2{k3}{z},XMMWORD PTR [rdx+2*16]
        vpmovzxbd zmm0{k3}{z},XMMWORD PTR [r10+rax+2*16]
        vpmovzxbd zmm1{k3}{z},XMMWORD PTR [r11+rax+2*16]
        MultiplyAccumulateCell\Isa\() zmm10,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm14,zmm1,zmm2
        vpmovzxbd zmm0{k3}{z},XMMWORD PTR [r12+rax+2*16]
        vpmovzxbd zmm1{k3}{z},XMMWORD PTR [r13+rax+2*16]
        MultiplyAccumulateCell\Isa\() zmm18,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm22,zmm1,zmm2
        vpmovzxbd zmm0{k3}{z},XMMWORD PTR [r14+rax+2*16]
        vpmovzxbd zmm1{k3}{z},XMMWORD PTR [r15+rax+2*16]
        MultiplyAccumulateCell\Isa\() zmm26,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm30,zmm1,zmm2

.LComputeDepthwiseBlockBy32\@:
        vpmovzxbd zmm2{k2}{z},XMMWORD PTR [rdx+1*16]
        vpmovzxbd zmm0{k2}{z},XMMWORD PTR [r10+rax+1*16]
        vpmovzxbd zmm1{k2}{z},XMMWORD PTR [r11+rax+1*16]
        MultiplyAccumulateCell\Isa\() zmm9,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm13,zmm1,zmm2
        vpmovzxbd zmm0{k2}{z},XMMWORD PTR [r12+rax+1*16]
        vpmovzxbd zmm1{k2}{z},XMMWORD PTR [r13+rax+1*16]
        MultiplyAccumulateCell\Isa\() zmm17,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm21,zmm1,zmm2
        vpmovzxbd zmm0{k2}{z},XMMWORD PTR [r14+rax+1*16]
        vpmovzxbd zmm1{k2}{z},XMMWORD PTR [r15+rax+1*16]
        MultiplyAccumulateCell\Isa\() zmm25,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm29,zmm1,zmm2

.LComputeDepthwiseBlockBy16\@:
        vpmovzxbd zmm2{k1}{z},XMMWORD PTR [rdx]
        vpmovzxbd zmm0{k1}{z},XMMWORD PTR [r10+rax]
        vpmovzxbd zmm1{k1}{z},XMMWORD PTR [r11+rax]
        MultiplyAccumulateCell\Isa\() zmm8,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm12,zmm1,zmm2
        vpmovzxbd zmm0{k1}{z},XMMWORD PTR [r12+rax]
        vpmovzxbd zmm1{k1}{z},XMMWORD PTR [r13+rax]
        MultiplyAccumulateCell\Isa\() zmm16,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm20,zmm1,zmm2
        vpmovzxbd zmm0{k1}{z},XMMWORD PTR [r14+rax]
        vpmovzxbd zmm1{k1}{z},XMMWORD PTR [r15+rax]
        MultiplyAccumulateCell\Isa\() zmm24,zmm0,zmm2
        MultiplyAccumulateCell\Isa\() zmm28,zmm1,zmm2
        add     rdx,rsi                     # advance filter to next kernel
        dec     r9                          # decrement input blocks remaining
        jnz     .LProcessNextInputBlock\@

//
// Post-process the block accumulators.
//

        mov     ebx,ebp
        mov     rdx,.LConvSymDepthwiseKernelFrame_PostProcessParams[rsp]
        mov     ebp,DWORD PTR .LConvSymDepthwiseKernelFrame_KernelFlags[rsp]
        call    MlasConvSymPostProcessAvx512Core

//
// Restore non-volatile registers and return.
//

.LExitKernel\@:
        vzeroupper
        add     rsp,.LConvSymDepthwiseKernelFrame_SavedR15
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rbx
        pop     rbp
        ret

        .endm

/*++

Macro Description:

    This macro generates code to convert the block accumulators from the matrix
    multiply loop to float values.

Arguments:

    RegList - Supplies the list of vector registers to operate on.

    ScaleReg - Supplies the output scale vector.

Implicit Arguments:

    zmm4 - Supplies the integer bias vector.

--*/

        .macro ConvertAccumulatorToFloatRegList RegList, ScaleReg

//
// Offset each value by the per-channel bias value, convert to floating point,
// and apply the output scale.
//

        EmitForEachRegister "\RegList\()","vpaddd \RegItem\(),\RegItem\(),zmm4"
        EmitForEachRegister "\RegList\()","vcvtdq2ps \RegItem\(),\RegItem\()"
        EmitForEachRegister "\RegList\()","vmulps \RegItem\(),\RegItem\(),\ScaleReg\()"

        .endm

/*++

Macro Description:

    This macro generates code to convert float values to 32-bit integers in the
    range 0 to 255.

Arguments:

    RegList - Supplies the list of vector registers to operate on.

Implicit Arguments:

    zmm0 - Supplies the broadcasted minimum clip float value.

        This is set to static_cast<float>(0 - ZeroPointValue).

    zmm1 - Supplies the broadcasted maximum clip float value.

        This is set to static_cast<float>(255 - ZeroPointValue).

    zmm2 - Supplies the broadcasted zero point integer value.

--*/

        .macro ConvertFloatToIntegerRegList RegList

//
// Clip the float values to the integer range covered by the output zero point.
// This also keeps values outside the range INT_MIN to INT_MAX from converting
// to INT_MIN.
//

        EmitForEachRegister "\RegList\()","vmaxps \RegItem\(),\RegItem\(),zmm0"
        EmitForEachRegister "\RegList\()","vminps \RegItem\(),\RegItem\(),zmm1"

//
// Convert the float value to integer and add the zero point offset.
//

        EmitForEachRegister "\RegList\()","vcvtps2dq \RegItem\(),\RegItem\()"
        EmitForEachRegister "\RegList\()","vpaddd \RegItem\(),\RegItem\(),zmm2"

        .endm

/*++

Routine Description:

    This routine post processes the block accumulators produced by the convolution
    kernels, including type conversion, requantization, and storing to the output
    buffer.

Arguments:

Return Value:

    None.

--*/

MlasConvSymPostProcessAvx512Core:

//
// Apply the bias and convert the block accumulators to intermediate float values.
//

        mov     r10,.LConvSymPostProcessParams_Bias[rdx]
        mov     r11,.LConvSymPostProcessParams_Scale[rdx]
        test    bpl,MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        jz      .LPostProcess.BroadcastScaleValue
        vmovups zmm0{k1}{z},ZMMWORD PTR [r11]
        vmovups zmm1{k2}{z},ZMMWORD PTR [r11+16*4]
        vmovups zmm2{k3}{z},ZMMWORD PTR [r11+32*4]
        vmovups zmm3{k4}{z},ZMMWORD PTR [r11+48*4]
        jmp     .LPostProcess.ConvertAccumulatorsToFloat

.LPostProcess.BroadcastScaleValue:
        vbroadcastss zmm0,DWORD PTR [r11]
        vmovups zmm1,zmm0
        vmovups zmm2,zmm0
        vmovups zmm3,zmm0

.LPostProcess.ConvertAccumulatorsToFloat:
        cmp     ecx,16
        jbe     .LPostProcess.ConvertAccumulatorsToFloatBy16
        cmp     ecx,32
        jbe     .LPostProcess.ConvertAccumulatorsToFloatBy32
        cmp     ecx,48
        jbe     .LPostProcess.ConvertAccumulatorsToFloatBy48

.LPostProcess.ConvertAccumulatorsToFloatBy64:
        vmovdqu32 zmm4{k4}{z},ZMMWORD PTR [r10+48*4]
        ConvertAccumulatorToFloatRegList "zmm11,zmm15,zmm19,zmm23,zmm27,zmm31",zmm3

.LPostProcess.ConvertAccumulatorsToFloatBy48:
        vmovdqu32 zmm4{k3}{z},ZMMWORD PTR [r10+32*4]
        ConvertAccumulatorToFloatRegList "zmm10,zmm14,zmm18,zmm22,zmm26,zmm30",zmm2

.LPostProcess.ConvertAccumulatorsToFloatBy32:
        vmovdqu32 zmm4{k2}{z},ZMMWORD PTR [r10+16*4]
        ConvertAccumulatorToFloatRegList "zmm9,zmm13,zmm17,zmm21,zmm25,zmm29",zmm1

.LPostProcess.ConvertAccumulatorsToFloatBy16:
        vmovdqu32 zmm4{k1}{z},ZMMWORD PTR [r10]
        ConvertAccumulatorToFloatRegList "zmm8,zmm12,zmm16,zmm20,zmm24,zmm28",zmm0

//
// Convert the intermediate float values to 32-bit integers in the range 0 to 255.
//

        vbroadcastss zmm0,DWORD PTR .LConvSymPostProcessParams_MinimumValue[rdx]
        vbroadcastss zmm1,DWORD PTR .LConvSymPostProcessParams_MaximumValue[rdx]
        vpbroadcastd zmm2,DWORD PTR .LConvSymPostProcessParams_OutputZeroPoint[rdx]
        cmp     ecx,16
        jbe     .LPostProcess.ConvertFloatsToIntegerBy16
        cmp     ecx,32
        jbe     .LPostProcess.ConvertFloatsToIntegerBy32
        cmp     ecx,48
        jbe     .LPostProcess.ConvertFloatsToIntegerBy48

.LPostProcess.ConvertFloatsToIntegerBy64:
        ConvertFloatToIntegerRegList "zmm11,zmm15,zmm19,zmm23,zmm27,zmm31"

.LPostProcess.ConvertFloatsToIntegerBy48:
        ConvertFloatToIntegerRegList "zmm10,zmm14,zmm18,zmm22,zmm26,zmm30"

.LPostProcess.ConvertFloatsToIntegerBy32:
        ConvertFloatToIntegerRegList "zmm9,zmm13,zmm17,zmm21,zmm25,zmm29"

.LPostProcess.ConvertFloatsToIntegerBy16:
        ConvertFloatToIntegerRegList "zmm8,zmm12,zmm16,zmm20,zmm24,zmm28"

//
// Pack with saturation and store 1 to 64 bytes to the output buffer.
//

.LPostProcess.StoreQuantizedOutput:
        lea     r9,[rsi*2+rsi]
        add     r9,r8
        cmp     ebx,5
        ja      .LPostProcess.StoreQuantizedOutput6
        je      .LPostProcess.StoreQuantizedOutput5
        cmp     ebx,3
        ja      .LPostProcess.StoreQuantizedOutput4
        je      .LPostProcess.StoreQuantizedOutput3
        cmp     ebx,1
        ja      .LPostProcess.StoreQuantizedOutput2
        jmp     .LPostProcess.StoreQuantizedOutput1

.LPostProcess.StoreQuantizedOutput6:
        vpmovusdb XMMWORD PTR [r9+rsi*2]{k1},zmm28
        vpmovusdb XMMWORD PTR [r9+rsi*2+16]{k2},zmm29
        vpmovusdb XMMWORD PTR [r9+rsi*2+32]{k3},zmm30
        vpmovusdb XMMWORD PTR [r9+rsi*2+48]{k4},zmm31

.LPostProcess.StoreQuantizedOutput5:
        vpmovusdb XMMWORD PTR [r9+rsi]{k1},zmm24
        vpmovusdb XMMWORD PTR [r9+rsi+16]{k2},zmm25
        vpmovusdb XMMWORD PTR [r9+rsi+32]{k3},zmm26
        vpmovusdb XMMWORD PTR [r9+rsi+48]{k4},zmm27

.LPostProcess.StoreQuantizedOutput4:
        vpmovusdb XMMWORD PTR [r9]{k1},zmm20
        vpmovusdb XMMWORD PTR [r9+16]{k2},zmm21
        vpmovusdb XMMWORD PTR [r9+32]{k3},zmm22
        vpmovusdb XMMWORD PTR [r9+48]{k4},zmm23

.LPostProcess.StoreQuantizedOutput3:
        vpmovusdb XMMWORD PTR [r8+rsi*2]{k1},zmm16
        vpmovusdb XMMWORD PTR [r8+rsi*2+16]{k2},zmm17
        vpmovusdb XMMWORD PTR [r8+rsi*2+32]{k3},zmm18
        vpmovusdb XMMWORD PTR [r8+rsi*2+48]{k4},zmm19

.LPostProcess.StoreQuantizedOutput2:
        vpmovusdb XMMWORD PTR [r8+rsi]{k1},zmm12
        vpmovusdb XMMWORD PTR [r8+rsi+16]{k2},zmm13
        vpmovusdb XMMWORD PTR [r8+rsi+32]{k3},zmm14
        vpmovusdb XMMWORD PTR [r8+rsi+48]{k4},zmm15

.LPostProcess.StoreQuantizedOutput1:
        vpmovusdb XMMWORD PTR [r8]{k1},zmm8
        vpmovusdb XMMWORD PTR [r8+16]{k2},zmm9
        vpmovusdb XMMWORD PTR [r8+32]{k3},zmm10
        vpmovusdb XMMWORD PTR [r8+48]{k4},zmm11
        ret

//
// Generate the convolution kernels.
//

ConvSymKernelFunction Avx512Core
ConvSymDepthwiseKernelFunction Avx512Core

ConvSymKernelFunction Avx512Vnni
ConvSymDepthwiseKernelFunction Avx512Vnni

        .end
