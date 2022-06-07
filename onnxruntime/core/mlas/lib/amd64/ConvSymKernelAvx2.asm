;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   ConvSymKernelAvx2.asm
;
; Abstract:
;
;   This module implements the kernels for the symmetric quantized integer
;   convolution operation.
;
;   This implementation uses AVX2 and AVX VNNI instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE ConvSymKernelCommon.inc
INCLUDE AssembleAvxVnni.inc
        .list

;
; Macro Description:
;
;   This macro generates code to multiply and accumulate a single row of the
;   output block.
;
; Arguments:
;
;   Vec1Reg - Supplies the low block accumulator register.
;
;   Vec2Reg - Supplies the high block accumulator register.
;
; Implicit Arguments:
;
;   ymm0 - Supplies the first vector loaded from the filter buffer.
;
;   ymm1 - Supplies the second vector loaded from the filter buffer.
;
;   ymm2 - Supplies the broadcast value loaded from the input buffer.
;
;   ymm3 - Supplies a scratch register for intermediate results.
;
;   ymm12 - Supplies a 256-bit with the broadcasted word value 0x0001.
;

MultiplyAccumulateRowAvx2 MACRO Vec1Reg, Vec2Reg

        vpmaddubsw ymm3,ymm2,ymm0
        vpmaddwd ymm3,ymm3,ymm12
        vpaddd Vec1Reg,Vec1Reg,ymm3
        vpmaddubsw ymm2,ymm2,ymm1
        vpmaddwd ymm2,ymm2,ymm12
        vpaddd Vec2Reg,Vec2Reg,ymm2

        ENDM

MultiplyAccumulateRowAvxVnni MACRO Vec1Reg, Vec2Reg

        VpdpbusdsYmmYmmYmm Vec1Reg,ymm2,ymm0
        VpdpbusdsYmmYmmYmm Vec2Reg,ymm2,ymm1

        ENDM

;
; Macro Description:
;
;   This macro generates code to multiply and accumulate each row of the output
;   block.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string.
;
;   RowCount - Supplies the number of rows to produce.
;
;   VectorOffset - Supplies the byte offset from the filter to fetch elements.
;
;   BroadcastOffset - Supplies the byte offset from the input to fetch elements.
;
; Implicit Arguments:
;
;   rdx - Supplies the address of the filter buffer.
;
;   r10 - Supplies the address of the base of the input buffer.
;
; Implicit Arguments (Avx2):
;
;   r11-r13 - Supplies the relative byte offsets from the base of the input
;       buffer to access the second through fourth rows.
;
;   ymm4-ymm11 - Supplies the block accumulators.
;
;   ymm12 - Supplies a 256-bit with the broadcasted word value 0x0001.
;
; Implicit Arguments (AvxVnni):
;
;   r11-r15 - Supplies the relative byte offsets from the base of the input
;       buffer to access the second through sixth rows.
;
;   ymm4-ymm15 - Supplies the block accumulators.
;

ComputeBlock MACRO Isa, RowCount, VectorOffset, BroadcastOffset

        vmovdqu ymm0,YMMWORD PTR [rdx+VectorOffset]
        vmovdqu ymm1,YMMWORD PTR [rdx+VectorOffset+32]
        EmitIfCountGE RowCount,1,<vpbroadcastd ymm2,DWORD PTR [r10+BroadcastOffset]>
        EmitIfCountGE RowCount,1,<MultiplyAccumulateRow&Isa& ymm4,ymm5>
        EmitIfCountGE RowCount,2,<vpbroadcastd ymm2,DWORD PTR [r10+r11+BroadcastOffset]>
        EmitIfCountGE RowCount,2,<MultiplyAccumulateRow&Isa& ymm6,ymm7>
        EmitIfCountGE RowCount,3,<vpbroadcastd ymm2,DWORD PTR [r10+r12+BroadcastOffset]>
        EmitIfCountGE RowCount,3,<MultiplyAccumulateRow&Isa& ymm8,ymm9>
        EmitIfCountGE RowCount,4,<vpbroadcastd ymm2,DWORD PTR [r10+r13+BroadcastOffset]>
        EmitIfCountGE RowCount,4,<MultiplyAccumulateRow&Isa& ymm10,ymm11>
        EmitIfCountGE RowCount,5,<vpbroadcastd ymm2,DWORD PTR [r10+r14+BroadcastOffset]>
        EmitIfCountGE RowCount,5,<MultiplyAccumulateRow&Isa& ymm12,ymm13>
        EmitIfCountGE RowCount,6,<vpbroadcastd ymm2,DWORD PTR [r10+r15+BroadcastOffset]>
        EmitIfCountGE RowCount,6,<MultiplyAccumulateRow&Isa& ymm14,ymm15>

        ENDM

;
; Macro Description:
;
;   This macro generates code to execute the block compute macro multiple times
;   and advancing the input and filter data pointers.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string.
;
;   RowCount - Supplies the number of rows to produce.
;
;   UnrollLoop - Supplies a non-blank value if the loop should be unrolled to
;       improve performance.
;
; Implicit Arguments:
;
;   rax - Supplies the number of input channels.
;
;   rdx - Supplies the address of the filter buffer.
;
;   r10 - Supplies the address of the base of the input buffer.
;

ComputeBlockLoop MACRO Isa, RowCount, UnrollLoop

        LOCAL   ComputeBlockBy4Loop
        LOCAL   ProcessRemainingBlocks
        LOCAL   ComputeBlockBy1Loop
        LOCAL   ComputeBlockLoopExit

IFNB <UnrollLoop>
        sub     rax,4*4
        jb      ProcessRemainingBlocks

ComputeBlockBy4Loop:
        ComputeBlock Isa,RowCount,0*64,0
        ComputeBlock Isa,RowCount,1*64,4
        ComputeBlock Isa,RowCount,2*64,8
        ComputeBlock Isa,RowCount,3*64,12
        add     r10,4*4                     ; advance input base address
        add     rdx,4*16*4                  ; advance filter address
        sub     rax,4*4                     ; decrement elements remaining
        jae     ComputeBlockBy4Loop

ProcessRemainingBlocks:
        add     rax,4*4                     ; correct for over-subtract above
        jz      ComputeBlockLoopExit
ENDIF

ComputeBlockBy1Loop:
        ComputeBlock Isa,RowCount,0*64,0
        add     r10,4                       ; advance input base address
        add     rdx,16*4                    ; advance filter address
        sub     rax,4                       ; decrement elements remaining
        jnz     ComputeBlockBy1Loop

ComputeBlockLoopExit:

        ENDM

;
; Macro Description:
;
;   This macro generates code to convert the block accumulators from the matrix
;   multiply loop to float values.
;
; Arguments:
;
;   RegList - Supplies the list of vector registers to operate on.
;
; Implicit Arguments:
;
;   ymm0 - Supplies the integer bias vector.
;
;   ymm1 - Supplies the output scale vector.
;

ConvertAccumulatorToFloatRegList MACRO RegList

;
; Offset each value by the per-channel bias value, convert to floating point,
; and apply the output scale.
;

        EmitForEachRegister <RegList>,<vpaddd RegItem,RegItem,ymm0>
        EmitForEachRegister <RegList>,<vcvtdq2ps RegItem,RegItem>
        EmitForEachRegister <RegList>,<vmulps RegItem,RegItem,ymm1>

        ENDM

;
; Macro Description:
;
;   This macro generates code to convert float values to 32-bit integers in the
;   range 0 to 255.
;
; Arguments:
;
;   RegList - Supplies the list of vector registers to operate on.
;
; Implicit Arguments:
;
;   ymm0 - Supplies the broadcasted minimum clip float value.
;
;       This is set to static_cast<float>(0 - ZeroPointValue).
;
;   ymm1 - Supplies the broadcasted maximum clip float value.
;
;       This is set to static_cast<float>(255 - ZeroPointValue).
;
;   ymm2 - Supplies the broadcasted zero point integer value.
;

ConvertFloatToIntegerRegList MACRO RegList

;
; Clip the float values to the integer range covered by the output zero point.
; This also keeps values outside the range INT_MIN to INT_MAX from converting
; to INT_MIN.
;

        EmitForEachRegister <RegList>,<vmaxps RegItem,RegItem,ymm0>
        EmitForEachRegister <RegList>,<vminps RegItem,RegItem,ymm1>

;
; Convert the float value to integer and add the zero point offset.
;

        EmitForEachRegister <RegList>,<vcvtps2dq RegItem,RegItem>
        EmitForEachRegister <RegList>,<vpaddd RegItem,RegItem,ymm2>

        ENDM

;
; Macro Description:
;
;   This macro generates code for the inner kernel to compute a convolution
;   for the elements of an output row for a set of filter rows.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string.
;

ConvSymKernelFunction MACRO Isa

;++
;
; Routine Description:
;
;   This routine is the inner kernel to compute a convolution for the elements
;   of an output row for a set of filter rows.
;
; Arguments:
;
;   Input (rcx) - Supplies the address of the input buffer.
;
;       If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is set, then the input buffer points
;       directly at the input tensor.
;
;       If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is clear, then the input buffer is an
;       indirection buffer. Every pointer in the indirection buffer points at a
;       InputChannels length vector (either from the input tensor or a vector of
;       padding values). These are grouped in batches of length KernelSize.
;       These batches are then repeated OutputCount times.
;
;   Filter (rdx) - Supplies the address of the filter buffer.
;
;   Output (r8) - Supplies the address of the output buffer.
;
;   KernelSize (r9) - Supplies the size of the kernel.
;
;       If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is set, then kernel size should be 1.
;
;   InputChannels - Supplies the number of input channels.
;
;       This implementation requires the count to be a multiple of 4.
;
;   OutputChannels - Supplies the number of output channels.
;
;   ChannelCount - Supplies the number of channels this iteration produces.
;
;       This implementation requires the count to be 8 or 16.
;
;   OutputCount - Supplies the number of output elements this iteration produces.
;
IFIDNI <Isa>, <AvxVnni>
;       This implementation requires the count to be in the range 1 to 6.
ELSE
;       This implementation requires the count to be in the range 1 to 4.
ENDIF
;
;   PostProcessParams - Supplies the address of the post process parameter block.
;
;   KernelFlags - Supplies additional flags controlling the operation.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasConvSymKernel&Isa&, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        alloc_stack (ConvSymKernelFrame.SavedR13)
IFIDNI <Isa>, <AvxVnni>
        save_reg r14,ConvSymKernelFrame.SavedR14
        save_reg r15,ConvSymKernelFrame.SavedR15
ENDIF
        save_xmm128 xmm6,ConvSymKernelFrame.SavedXmm6
        save_xmm128 xmm7,ConvSymKernelFrame.SavedXmm7
        save_xmm128 xmm8,ConvSymKernelFrame.SavedXmm8
        save_xmm128 xmm9,ConvSymKernelFrame.SavedXmm9
        save_xmm128 xmm10,ConvSymKernelFrame.SavedXmm10
        save_xmm128 xmm11,ConvSymKernelFrame.SavedXmm11
        save_xmm128 xmm12,ConvSymKernelFrame.SavedXmm12
IFIDNI <Isa>, <AvxVnni>
        save_xmm128 xmm13,ConvSymKernelFrame.SavedXmm13
        save_xmm128 xmm14,ConvSymKernelFrame.SavedXmm14
        save_xmm128 xmm15,ConvSymKernelFrame.SavedXmm15
ENDIF

        END_PROLOGUE

        lea     rdi,[r9*8]
        mov     ebx,DWORD PTR ConvSymKernelFrame.OutputCount[rsp]
        mov     rsi,ConvSymKernelFrame.InputChannels[rsp]
        mov     ebp,DWORD PTR ConvSymKernelFrame.KernelFlags[rsp]
        vpxor   xmm4,xmm4,xmm4
        vpxor   xmm5,xmm5,xmm5
        vpxor   xmm6,xmm6,xmm6
        vpxor   xmm7,xmm7,xmm7
        vpxor   xmm8,xmm8,xmm8
        vpxor   xmm9,xmm9,xmm9
        vpxor   xmm10,xmm10,xmm10
        vpxor   xmm11,xmm11,xmm11
IFIDNI <Isa>, <AvxVnni>
        vpxor   xmm12,xmm12,xmm12
        vpxor   xmm13,xmm13,xmm13
        vpxor   xmm14,xmm14,xmm14
        vpxor   xmm15,xmm15,xmm15
ELSE
        vpcmpeqw ymm12,ymm12,ymm12          ; generate 256-bit word vector [0xFFFF]
        vpsrlw  ymm12,ymm12,15              ; generate 256-bit word vector [0x0001]
ENDIF

;
; Process an input block of length InputChannels for each element of the kernel.
;

ProcessNextInputBlock:
        test    bpl,MLAS_CONV_SYM_FLAG_INPUT_DIRECT
        jz      InputIndirection

;
; The input buffer points directly at the input data and this is effectively a
; GEMM operation (such as a pointwise convolution or an Im2Col transform).
;

InputDirect:
        xor     r10,r10
        mov     r11,rsi
        lea     r12,[r11+r11]
        lea     r13,[r12+r11]
IFIDNI <Isa>, <AvxVnni>
        lea     r14,[r13+r11]
        lea     r15,[r14+r11]
ENDIF
        cmp     ebx,2
        cmovb   r11,r10                     ; use first row if output count is small
        cmovbe  r12,r10
        cmp     ebx,4
        cmovb   r13,r10
IFIDNI <Isa>, <AvxVnni>
        cmovbe  r14,r10
        cmp     ebx,6
        cmovb   r15,r10
ENDIF
        mov     r10,rcx
        jmp     ComputeBlockLoopStart

InputIndirection:
        lea     r11,[rcx+rdi]
        lea     r12,[rcx+rdi*2]
        lea     r13,[r11+rdi*2]
IFIDNI <Isa>, <AvxVnni>
        lea     r14,[r12+rdi*2]
        lea     r15,[r13+rdi*2]
ENDIF
        cmp     ebx,2
        cmovb   r11,rcx                     ; use first row if output count is small
        cmovbe  r12,rcx
        cmp     ebx,4
        cmovb   r13,rcx
IFIDNI <Isa>, <AvxVnni>
        cmovbe  r14,rcx
        cmp     ebx,6
        cmovb   r15,rcx
ENDIF
        mov     r10,QWORD PTR [rcx]
        mov     r11,QWORD PTR [r11]
        mov     r12,QWORD PTR [r12]
        mov     r13,QWORD PTR [r13]
IFIDNI <Isa>, <AvxVnni>
        mov     r14,QWORD PTR [r14]
        mov     r15,QWORD PTR [r15]
ENDIF
        add     rcx,8                       ; advance indirection buffer address
        sub     r11,r10                     ; compute deltas from base address
        sub     r12,r10
        sub     r13,r10
IFIDNI <Isa>, <AvxVnni>
        sub     r14,r10
        sub     r15,r10
ENDIF

ComputeBlockLoopStart:
        mov     rax,rsi                     ; reload input channels
        cmp     ebx,2                       ; output count <= 2?
        jbe     ComputeBlockLoopBy2
IFIDNI <Isa>, <AvxVnni>
        cmp     ebx,4                       ; output count <= 4?
        jbe     ComputeBlockLoopBy4
        ComputeBlockLoop Isa,6,UnrollLoop
ELSE
        ComputeBlockLoop Isa,4,UnrollLoop
ENDIF

ComputeBlockLoopDone:
        dec     r9                          ; decrement input blocks remaining
        jnz     ProcessNextInputBlock

;
; Apply the bias and convert the block accumulators to intermediate float values.
;

        mov     rdx,ConvSymKernelFrame.PostProcessParams[rsp]
        mov     rsi,ConvSymKernelFrame.OutputChannels[rsp]
        mov     r11d,DWORD PTR ConvSymKernelFrame.ChannelCount[rsp]
        mov     rcx,ConvSymPostProcessParams.Bias[rdx]
        mov     r9,ConvSymPostProcessParams.Scale[rdx]
        lea     r10,[rsi*2+rsi]             ; compute fourth row output offset
        add     r10,r8
        vmovdqu ymm0,YMMWORD PTR [rcx]      ; load low bias vector
        test    bpl,MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        jz      BroadcastScaleValue
        vmovups ymm1,YMMWORD PTR [r9]       ; load low scale vector
        jmp     ConvertLowAccumulatorsToFloat

BroadcastScaleValue:
        vbroadcastss ymm1,DWORD PTR [r9]

ConvertLowAccumulatorsToFloat:
IFIDNI <Isa>, <AvxVnni>
        ConvertAccumulatorToFloatRegList <ymm4,ymm6,ymm8,ymm10,ymm12,ymm14>
ELSE
        ConvertAccumulatorToFloatRegList <ymm4,ymm6,ymm8,ymm10>
ENDIF
        cmp     r11d,8                      ; output single vector?
        jbe     ConvertFloatsToIntegers
        vmovdqu ymm0,YMMWORD PTR [rcx+8*4]  ; load high bias vector
        test    bpl,MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        jz      ConvertHighAccumulatorsToFloat
        vmovups ymm1,YMMWORD PTR [r9+8*4]   ; load high scale vector

ConvertHighAccumulatorsToFloat:
IFIDNI <Isa>, <AvxVnni>
        ConvertAccumulatorToFloatRegList <ymm5,ymm7,ymm9,ymm11,ymm13,ymm15>
ELSE
        ConvertAccumulatorToFloatRegList <ymm5,ymm7,ymm9,ymm11>
ENDIF

;
; Convert the intermediate float values to 32-bit integers in the range 0 to 255.
;

ConvertFloatsToIntegers:
        vbroadcastss ymm0,DWORD PTR ConvSymPostProcessParams.MinimumValue[rdx]
        vbroadcastss ymm1,DWORD PTR ConvSymPostProcessParams.MaximumValue[rdx]
        vpbroadcastd ymm2,DWORD PTR ConvSymPostProcessParams.OutputZeroPoint[rdx]
IFIDNI <Isa>, <AvxVnni>
        ConvertFloatToIntegerRegList <ymm4,ymm6,ymm8,ymm10,ymm12,ymm14>
ELSE
        ConvertFloatToIntegerRegList <ymm4,ymm6,ymm8,ymm10>
ENDIF
        cmp     r11d,8                      ; output single vector?
        jbe     StoreQuantizedOutputBy8
IFIDNI <Isa>, <AvxVnni>
        ConvertFloatToIntegerRegList <ymm5,ymm7,ymm9,ymm11,ymm13,ymm15>
ELSE
        ConvertFloatToIntegerRegList <ymm5,ymm7,ymm9,ymm11>
ENDIF

;
; Pack with saturation and store 16 bytes to the output buffer.
;

StoreQuantizedOutputBy16:
IFIDNI <Isa>, <AvxVnni>
        cmp     ebx,5
        ja      StoreQuantizedOutput6By16
        je      StoreQuantizedOutput5By16
ENDIF
        cmp     ebx,3
        ja      StoreQuantizedOutput4By16
        je      StoreQuantizedOutput3By16
        cmp     ebx,1
        ja      StoreQuantizedOutput2By16
        jmp     StoreQuantizedOutput1By16

IFIDNI <Isa>, <AvxVnni>
StoreQuantizedOutput6By16:
        vextracti128 xmm0,ymm14,1
        vpackusdw xmm14,xmm14,xmm0
        vextracti128 xmm1,ymm15,1
        vpackusdw xmm15,xmm15,xmm1
        vpackuswb xmm14,xmm14,xmm15
        vmovdqu XMMWORD PTR [r10+rsi*2],xmm14

StoreQuantizedOutput5By16:
        vextracti128 xmm0,ymm12,1
        vpackusdw xmm12,xmm12,xmm0
        vextracti128 xmm1,ymm13,1
        vpackusdw xmm13,xmm13,xmm1
        vpackuswb xmm12,xmm12,xmm13
        vmovdqu XMMWORD PTR [r10+rsi],xmm12
ENDIF

StoreQuantizedOutput4By16:
        vextracti128 xmm0,ymm10,1
        vpackusdw xmm10,xmm10,xmm0
        vextracti128 xmm1,ymm11,1
        vpackusdw xmm11,xmm11,xmm1
        vpackuswb xmm10,xmm10,xmm11
        vmovdqu XMMWORD PTR [r10],xmm10

StoreQuantizedOutput3By16:
        vextracti128 xmm0,ymm8,1
        vpackusdw xmm8,xmm8,xmm0
        vextracti128 xmm1,ymm9,1
        vpackusdw xmm9,xmm9,xmm1
        vpackuswb xmm8,xmm8,xmm9
        vmovdqu XMMWORD PTR [r8+rsi*2],xmm8

StoreQuantizedOutput2By16:
        vextracti128 xmm0,ymm6,1
        vpackusdw xmm6,xmm6,xmm0
        vextracti128 xmm1,ymm7,1
        vpackusdw xmm7,xmm7,xmm1
        vpackuswb xmm6,xmm6,xmm7
        vmovdqu XMMWORD PTR [r8+rsi],xmm6

StoreQuantizedOutput1By16:
        vextracti128 xmm0,ymm4,1
        vpackusdw xmm4,xmm4,xmm0
        vextracti128 xmm1,ymm5,1
        vpackusdw xmm5,xmm5,xmm1
        vpackuswb xmm4,xmm4,xmm5
        vmovdqu XMMWORD PTR [r8],xmm4

;
; Restore non-volatile registers and return.
;

ExitKernel:
        vzeroupper
        movaps  xmm6,ConvSymKernelFrame.SavedXmm6[rsp]
        movaps  xmm7,ConvSymKernelFrame.SavedXmm7[rsp]
        movaps  xmm8,ConvSymKernelFrame.SavedXmm8[rsp]
        movaps  xmm9,ConvSymKernelFrame.SavedXmm9[rsp]
        movaps  xmm10,ConvSymKernelFrame.SavedXmm10[rsp]
        movaps  xmm11,ConvSymKernelFrame.SavedXmm11[rsp]
        movaps  xmm12,ConvSymKernelFrame.SavedXmm12[rsp]
IFIDNI <Isa>, <AvxVnni>
        movaps  xmm13,ConvSymKernelFrame.SavedXmm13[rsp]
        movaps  xmm14,ConvSymKernelFrame.SavedXmm14[rsp]
        movaps  xmm15,ConvSymKernelFrame.SavedXmm15[rsp]
        mov     r14,ConvSymKernelFrame.SavedR14[rsp]
        mov     r15,ConvSymKernelFrame.SavedR15[rsp]
ENDIF
        add     rsp,(ConvSymKernelFrame.SavedR13)

        BEGIN_EPILOGUE

        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

;
; Pack with saturation and store 8 bytes to the output buffer.
;

StoreQuantizedOutputBy8:
IFIDNI <Isa>, <AvxVnni>
        cmp     ebx,5
        ja      StoreQuantizedOutput6By8
        je      StoreQuantizedOutput5By8
ENDIF
        cmp     ebx,3
        ja      StoreQuantizedOutput4By8
        je      StoreQuantizedOutput3By8
        cmp     ebx,1
        ja      StoreQuantizedOutput2By8
        jmp     StoreQuantizedOutput1By8

IFIDNI <Isa>, <AvxVnni>
StoreQuantizedOutput6By8:
        vextracti128 xmm0,ymm14,1
        vpackusdw xmm14,xmm14,xmm0
        vpackuswb xmm14,xmm14,xmm14
        vmovq   QWORD PTR [r10+rsi*2],xmm14

StoreQuantizedOutput5By8:
        vextracti128 xmm0,ymm12,1
        vpackusdw xmm12,xmm12,xmm0
        vpackuswb xmm12,xmm12,xmm12
        vmovq   QWORD PTR [r10+rsi],xmm12
ENDIF

StoreQuantizedOutput4By8:
        vextracti128 xmm0,ymm10,1
        vpackusdw xmm10,xmm10,xmm0
        vpackuswb xmm10,xmm10,xmm10
        vmovq   QWORD PTR [r10],xmm10

StoreQuantizedOutput3By8:
        vextracti128 xmm0,ymm8,1
        vpackusdw xmm8,xmm8,xmm0
        vpackuswb xmm8,xmm8,xmm8
        vmovq   QWORD PTR [r8+rsi*2],xmm8

StoreQuantizedOutput2By8:
        vextracti128 xmm0,ymm6,1
        vpackusdw xmm6,xmm6,xmm0
        vpackuswb xmm6,xmm6,xmm6
        vmovq   QWORD PTR [r8+rsi],xmm6

StoreQuantizedOutput1By8:
        vextracti128 xmm0,ymm4,1
        vpackusdw xmm4,xmm4,xmm0
        vpackuswb xmm4,xmm4,xmm4
        vmovq   QWORD PTR [r8],xmm4
        jmp     ExitKernel

;
; Process the tail output counts out of line with a reduced block size.
;

IFIDNI <Isa>, <AvxVnni>
ComputeBlockLoopBy4:
        ComputeBlockLoop Isa,4
        jmp     ComputeBlockLoopDone
ENDIF

ComputeBlockLoopBy2:
        ComputeBlockLoop Isa,2
        jmp     ComputeBlockLoopDone

        NESTED_END MlasConvSymKernel&Isa&, _TEXT

        ENDM

;
; Macro Description:
;
;   This macro generates code to multiply and accumulate a single cell of the
;   output block.
;
; Arguments:
;
;   AccumReg - Supplies the register to accumulate into.
;
;   Mult1Reg - Supplies the first multiplication operand register. This register
;       may be trashed on return.
;
;   Mult2Reg - Supplies the second multiplication operand register.
;

DepthwiseMultiplyAccumulateCellAvx2 MACRO AccumReg, Mult1Reg, Mult2Reg

        vpmaddwd Mult1Reg,Mult1Reg,Mult2Reg
        vpaddd  AccumReg,AccumReg,Mult1Reg

        ENDM

DepthwiseMultiplyAccumulateCellAvxVnni MACRO AccumReg, Mult1Reg, Mult2Reg

        VpdpbusdsYmmYmmYmm AccumReg,Mult1Reg,Mult2Reg

        ENDM

;
; Macro Description:
;
;   This macro generates code for the inner kernel to compute a depthwise
;   convolution for the elements of an output row for a set of filter rows.
;
; Arguments:
;
;   Isa - Supplies the instruction set architecture string.
;

ConvSymDepthwiseKernelFunction MACRO Isa

;++
;
; Routine Description:
;
;   This routine is the inner kernel to compute a depthwise convolution for the
;   elements of an output row for a set of filter rows.
;
; Arguments:
;
;   Input (rcx) - Supplies the address of the indirection buffer.
;
;   Filter (rdx) - Supplies the address of the filter buffer.
;
;   Output (r8) - Supplies the address of the output buffer.
;
;   KernelSize (r9) - Supplies the size of the kernel.
;
;   Channels - Supplies the number of input and output channels.
;
;   ChannelOffset - Supplies the byte offset from the indirection buffer base
;       address for this iteration.
;
;   ChannelCount - Supplies the number of channels this iteration produces.
;
;       This implementation requires the count to be 16.
;
;   OutputCount - Supplies the number of output elements this iteration produces.
;
;       This implementation requires the count to be in the range 1 to 4.
;
;   PostProcessParams - Supplies the address of the post process parameter block.
;
;   KernelFlags - Supplies additional flags controlling the operation.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY MlasConvSymDepthwiseKernel&Isa&, _TEXT

        rex_push_reg rbp
        push_reg rbx
        push_reg rsi
        push_reg rdi
        push_reg r12
        push_reg r13
        alloc_stack (ConvSymDepthwiseKernelFrame.SavedR13)
        save_xmm128 xmm6,ConvSymDepthwiseKernelFrame.SavedXmm6
        save_xmm128 xmm7,ConvSymDepthwiseKernelFrame.SavedXmm7
        save_xmm128 xmm8,ConvSymDepthwiseKernelFrame.SavedXmm8
        save_xmm128 xmm9,ConvSymDepthwiseKernelFrame.SavedXmm9
        save_xmm128 xmm10,ConvSymDepthwiseKernelFrame.SavedXmm10
        save_xmm128 xmm11,ConvSymDepthwiseKernelFrame.SavedXmm11

        END_PROLOGUE

        lea     rdi,[r9*8]
        mov     ebx,DWORD PTR ConvSymDepthwiseKernelFrame.OutputCount[rsp]
        mov     rsi,ConvSymDepthwiseKernelFrame.Channels[rsp]
        mov     rax,ConvSymDepthwiseKernelFrame.ChannelOffset[rsp]
        mov     ebp,DWORD PTR ConvSymDepthwiseKernelFrame.KernelFlags[rsp]
        vpxor   xmm4,xmm4,xmm4
        vpxor   xmm5,xmm5,xmm5
        vpxor   xmm6,xmm6,xmm6
        vpxor   xmm7,xmm7,xmm7
        vpxor   xmm8,xmm8,xmm8
        vpxor   xmm9,xmm9,xmm9
        vpxor   xmm10,xmm10,xmm10
        vpxor   xmm11,xmm11,xmm11

;
; Process an input block of length Channels for each element of the kernel.
;

ProcessNextInputBlock:
        vpmovsxbd ymm0,QWORD PTR [rdx]
        vpmovsxbd ymm1,QWORD PTR [rdx+8]
        lea     r11,[rcx+rdi]
        lea     r12,[rcx+rdi*2]
        lea     r13,[r11+rdi*2]
        cmp     ebx,2
        cmovb   r11,rcx                     ; use first row if output count is small
        cmovbe  r12,rcx
        cmp     ebx,4
        cmovb   r13,rcx
        mov     r10,QWORD PTR [rcx]
        mov     r11,QWORD PTR [r11]
        mov     r12,QWORD PTR [r12]
        mov     r13,QWORD PTR [r13]
        add     rcx,8                       ; advance indirection buffer address
        vpmovzxbd ymm2,QWORD PTR [r10+rax]
        vpmovzxbd ymm3,QWORD PTR [r10+rax+8]
        DepthwiseMultiplyAccumulateCell&Isa& ymm4,ymm2,ymm0
        vpmovzxbd ymm2,QWORD PTR [r11+rax]
        DepthwiseMultiplyAccumulateCell&Isa& ymm5,ymm3,ymm1
        vpmovzxbd ymm3,QWORD PTR [r11+rax+8]
        DepthwiseMultiplyAccumulateCell&Isa& ymm6,ymm2,ymm0
        vpmovzxbd ymm2,QWORD PTR [r12+rax]
        DepthwiseMultiplyAccumulateCell&Isa& ymm7,ymm3,ymm1
        vpmovzxbd ymm3,QWORD PTR [r12+rax+8]
        DepthwiseMultiplyAccumulateCell&Isa& ymm8,ymm2,ymm0
        vpmovzxbd ymm2,QWORD PTR [r13+rax]
        DepthwiseMultiplyAccumulateCell&Isa& ymm9,ymm3,ymm1
        vpmovzxbd ymm3,QWORD PTR [r13+rax+8]
        DepthwiseMultiplyAccumulateCell&Isa& ymm10,ymm2,ymm0
        add     rdx,rsi                     ; advance filter to next kernel
        DepthwiseMultiplyAccumulateCell&Isa& ymm11,ymm3,ymm1
        dec     r9                          ; decrement input blocks remaining
        jnz     ProcessNextInputBlock

;
; Apply the bias and convert the block accumulators to intermediate float values.
;

        mov     rdx,ConvSymDepthwiseKernelFrame.PostProcessParams[rsp]
        mov     rcx,ConvSymPostProcessParams.Bias[rdx]
        mov     r9,ConvSymPostProcessParams.Scale[rdx]
        vmovdqu ymm0,YMMWORD PTR [rcx]      ; load low bias vector
        test    bpl,MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        jz      BroadcastScaleValue
        vmovups ymm1,YMMWORD PTR [r9]       ; load low scale vector
        jmp     ConvertLowAccumulatorsToFloat

BroadcastScaleValue:
        vbroadcastss ymm1,DWORD PTR [r9]

ConvertLowAccumulatorsToFloat:
        ConvertAccumulatorToFloatRegList <ymm4,ymm6,ymm8,ymm10>
        vmovdqu ymm0,YMMWORD PTR [rcx+8*4]  ; load high bias vector
        test    bpl,MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        jz      ConvertHighAccumulatorsToFloat
        vmovups ymm1,YMMWORD PTR [r9+8*4]   ; load high scale vector

ConvertHighAccumulatorsToFloat:
        ConvertAccumulatorToFloatRegList <ymm5,ymm7,ymm9,ymm11>

;
; Convert the intermediate float values to 32-bit integers in the range 0 to 255.
;

ConvertFloatsToIntegers:
        vbroadcastss ymm0,DWORD PTR ConvSymPostProcessParams.MinimumValue[rdx]
        vbroadcastss ymm1,DWORD PTR ConvSymPostProcessParams.MaximumValue[rdx]
        vpbroadcastd ymm2,DWORD PTR ConvSymPostProcessParams.OutputZeroPoint[rdx]
        ConvertFloatToIntegerRegList <ymm4,ymm6,ymm8,ymm10>
        ConvertFloatToIntegerRegList <ymm5,ymm7,ymm9,ymm11>

;
; Pack with saturation and store 16 bytes to the output buffer.
;

StoreQuantizedOutputBy16:
        lea     r10,[rsi*2+rsi]
        cmp     ebx,3
        ja      StoreQuantizedOutput4By16
        je      StoreQuantizedOutput3By16
        cmp     ebx,1
        ja      StoreQuantizedOutput2By16
        jmp     StoreQuantizedOutput1By16

StoreQuantizedOutput4By16:
        vextracti128 xmm0,ymm10,1
        vpackusdw xmm10,xmm10,xmm0
        vextracti128 xmm1,ymm11,1
        vpackusdw xmm11,xmm11,xmm1
        vpackuswb xmm10,xmm10,xmm11
        vmovdqu XMMWORD PTR [r8+r10],xmm10

StoreQuantizedOutput3By16:
        vextracti128 xmm0,ymm8,1
        vpackusdw xmm8,xmm8,xmm0
        vextracti128 xmm1,ymm9,1
        vpackusdw xmm9,xmm9,xmm1
        vpackuswb xmm8,xmm8,xmm9
        vmovdqu XMMWORD PTR [r8+rsi*2],xmm8

StoreQuantizedOutput2By16:
        vextracti128 xmm0,ymm6,1
        vpackusdw xmm6,xmm6,xmm0
        vextracti128 xmm1,ymm7,1
        vpackusdw xmm7,xmm7,xmm1
        vpackuswb xmm6,xmm6,xmm7
        vmovdqu XMMWORD PTR [r8+rsi],xmm6

StoreQuantizedOutput1By16:
        vextracti128 xmm0,ymm4,1
        vpackusdw xmm4,xmm4,xmm0
        vextracti128 xmm1,ymm5,1
        vpackusdw xmm5,xmm5,xmm1
        vpackuswb xmm4,xmm4,xmm5
        vmovdqu XMMWORD PTR [r8],xmm4

;
; Restore non-volatile registers and return.
;

ExitKernel:
        vzeroupper
        movaps  xmm6,ConvSymDepthwiseKernelFrame.SavedXmm6[rsp]
        movaps  xmm7,ConvSymDepthwiseKernelFrame.SavedXmm7[rsp]
        movaps  xmm8,ConvSymDepthwiseKernelFrame.SavedXmm8[rsp]
        movaps  xmm9,ConvSymDepthwiseKernelFrame.SavedXmm9[rsp]
        movaps  xmm10,ConvSymDepthwiseKernelFrame.SavedXmm10[rsp]
        movaps  xmm11,ConvSymDepthwiseKernelFrame.SavedXmm11[rsp]
        add     rsp,(ConvSymDepthwiseKernelFrame.SavedR13)

        BEGIN_EPILOGUE

        pop     r13
        pop     r12
        pop     rdi
        pop     rsi
        pop     rbx
        pop     rbp
        ret

        NESTED_END MlasConvSymDepthwiseKernel&Isa&, _TEXT

        ENDM

;
; Generate the convolution kernels.
;

ConvSymKernelFunction Avx2
ConvSymDepthwiseKernelFunction Avx2

ConvSymKernelFunction AvxVnni
ConvSymDepthwiseKernelFunction AvxVnni

        END
