/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvKernelCommon.h

Abstract:

    This module contains common kernel macros and structures for the single
    precision convolution operation.

--*/

//
// Define the convolution kernel flags.
//

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT     0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION         0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION       0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION      0x00000008

//
// Stack frame layout for the convolution kernels.
//

        .equ    .LSconvKernelFrame_Filter, 0
        .equ    .LSconvKernelFrame_SavedR12, 8
        .equ    .LSconvKernelFrame_SavedR13, 16
        .equ    .LSconvKernelFrame_SavedR14, 24
        .equ    .LSconvKernelFrame_SavedR15, 32
        .equ    .LSconvKernelFrame_SavedRbx, 40
        .equ    .LSconvKernelFrame_SavedRbp, 48
        .equ    .LSconvKernelFrame_ReturnAddress, 56
        .equ    .LSconvKernelFrame_InputStride, 64
        .equ    .LSconvKernelFrame_FilterStride, 72
        .equ    .LSconvKernelFrame_OutputStride, 80
        .equ    .LSconvKernelFrame_KernelHeight, 88
        .equ    .LSconvKernelFrame_KernelWidth, 96
        .equ    .LSconvKernelFrame_InputBase, 104
        .equ    .LSconvKernelFrame_InputWidth, 112
        .equ    .LSconvKernelFrame_DilatedInputWidth, 120
        .equ    .LSconvKernelFrame_OutputCountLeftPad, 128
        .equ    .LSconvKernelFrame_OutputCount, 136
        .equ    .LSconvKernelFrame_OutputCountRightPad, 144
        .equ    .LSconvKernelFrame_Bias, 152
        .equ    .LSconvKernelFrame_Flags, 160

        .equ    .LSconvKernelSingleFrame_ReturnAddress, 0
        .equ    .LSconvKernelSingleFrame_Filter, 8
        .equ    .LSconvKernelSingleFrame_SavedR12, 16
        .equ    .LSconvKernelSingleFrame_SavedR13, 24
        .equ    .LSconvKernelSingleFrame_SavedR14, 32
        .equ    .LSconvKernelSingleFrame_SavedR15, 40
        .equ    .LSconvKernelSingleFrame_SavedRbx, 48
        .equ    .LSconvKernelSingleFrame_SavedRbp, 56
        .equ    .LSconvKernelSingleFrame_ParentReturnAddress, 64
        .equ    .LSconvKernelSingleFrame_InputStride, 72
        .equ    .LSconvKernelSingleFrame_FilterStride, 80
        .equ    .LSconvKernelSingleFrame_OutputStride, 88
        .equ    .LSconvKernelSingleFrame_KernelHeight, 96
        .equ    .LSconvKernelSingleFrame_KernelWidth, 104
        .equ    .LSconvKernelSingleFrame_InputBase, 112
        .equ    .LSconvKernelSingleFrame_InputWidth, 120
        .equ    .LSconvKernelSingleFrame_DilatedInputWidth, 128
        .equ    .LSconvKernelSingleFrame_OutputCountLeftPad, 136
        .equ    .LSconvKernelSingleFrame_OutputCount, 144
        .equ    .LSconvKernelSingleFrame_OutputCountRightPad, 152
        .equ    .LSconvKernelSingleFrame_Bias, 160
        .equ    .LSconvKernelSingleFrame_Flags, 168

        .equ    .LSconvKernelDepthwiseFrame_SavedR12, 0
        .equ    .LSconvKernelDepthwiseFrame_SavedR13, 8
        .equ    .LSconvKernelDepthwiseFrame_SavedR14, 16
        .equ    .LSconvKernelDepthwiseFrame_SavedR15, 24
        .equ    .LSconvKernelDepthwiseFrame_SavedRbx, 32
        .equ    .LSconvKernelDepthwiseFrame_SavedRbp, 40
        .equ    .LSconvKernelDepthwiseFrame_ReturnAddress, 48
        .equ    .LSconvKernelDepthwiseFrame_KernelHeight, 56
        .equ    .LSconvKernelDepthwiseFrame_KernelWidth, 64
        .equ    .LSconvKernelDepthwiseFrame_InputBase, 72
        .equ    .LSconvKernelDepthwiseFrame_InputWidth, 80
        .equ    .LSconvKernelDepthwiseFrame_DilatedInputWidth, 88
        .equ    .LSconvKernelDepthwiseFrame_OutputCountLeftPad, 96
        .equ    .LSconvKernelDepthwiseFrame_OutputCount, 104
        .equ    .LSconvKernelDepthwiseFrame_OutputCountRightPad, 112
        .equ    .LSconvKernelDepthwiseFrame_Bias, 120
        .equ    .LSconvKernelDepthwiseFrame_Flags, 128

        .equ    .LSconvKernelDepthwiseSingleFrame_ReturnAddress, 0
        .equ    .LSconvKernelDepthwiseSingleFrame_SavedR12, 8
        .equ    .LSconvKernelDepthwiseSingleFrame_SavedR13, 16
        .equ    .LSconvKernelDepthwiseSingleFrame_SavedR14, 24
        .equ    .LSconvKernelDepthwiseSingleFrame_SavedR15, 32
        .equ    .LSconvKernelDepthwiseSingleFrame_SavedRbx, 40
        .equ    .LSconvKernelDepthwiseSingleFrame_SavedRbp, 48
        .equ    .LSconvKernelDepthwiseSingleFrame_ParentReturnAddress, 56
        .equ    .LSconvKernelDepthwiseSingleFrame_KernelHeight, 64
        .equ    .LSconvKernelDepthwiseSingleFrame_KernelWidth, 72
        .equ    .LSconvKernelDepthwiseSingleFrame_InputBase, 80
        .equ    .LSconvKernelDepthwiseSingleFrame_InputWidth, 88
        .equ    .LSconvKernelDepthwiseSingleFrame_DilatedInputWidth, 96
        .equ    .LSconvKernelDepthwiseSingleFrame_OutputCountLeftPad, 104
        .equ    .LSconvKernelDepthwiseSingleFrame_OutputCount, 112
        .equ    .LSconvKernelDepthwiseSingleFrame_OutputCountRightPad, 120
        .equ    .LSconvKernelDepthwiseSingleFrame_Bias, 128
        .equ    .LSconvKernelDepthwiseSingleFrame_Flags, 136

        .equ    .LSconvKernelPointwiseFrame_InputChannels, 0
        .equ    .LSconvKernelPointwiseFrame_SavedR12, 8
        .equ    .LSconvKernelPointwiseFrame_SavedR14, 16
        .equ    .LSconvKernelPointwiseFrame_SavedRbx, 24
        .equ    .LSconvKernelPointwiseFrame_SavedRbp, 32
        .equ    .LSconvKernelPointwiseFrame_ReturnAddress, 40
        .equ    .LSconvKernelPointwiseFrame_InputStride, 48
        .equ    .LSconvKernelPointwiseFrame_FilterStride, 56
        .equ    .LSconvKernelPointwiseFrame_OutputStride, 64
        .equ    .LSconvKernelPointwiseFrame_OutputCount, 72
        .equ    .LSconvKernelPointwiseFrame_Bias, 80
        .equ    .LSconvKernelPointwiseFrame_Flags, 88

/*++

Macro Description:

    This macro generates code to compute the convolution for a vector of input
    blocks and a vector of filter blocks to produce a matrix of output blocks.

    OutputCount=1 generates special case code to handle padding blocks. All
    other output counts assume no padding.

Arguments:

    Isa - Supplies the instruction set architecture string for function tags.

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    KernelType - Supplies the type of kernel to be generated.

    BlockSize - Supplies the number of elements per block.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    rdi - Supplies the address of the input buffer.

    rsi - Supplies the FilterStride parameter (see function description) when
        KernelType!=Depthwise. Supplies the address of the filter buffer when
        KernelType=Depthwise.

    rbp - Supplies the DilationWidth parameter (see function description).

    r8 - Supplies the address of the output buffer.

    r9 - Supplies the StrideWidth parameter (see function description).

    r15 - Supplies the InputStride parameter (see function description).

--*/

        .macro ProcessOutputCountN Isa, KernelFrame, KernelType, BlockSize, FilterCount, OutputCount

        mov     rcx,rdi
.ifeqs "\KernelType\()","Depthwise"
        mov     rdx,rsi
.else
        mov     rdx,\KernelFrame\()_Filter[rsp]
.endif
        mov     r11,\KernelFrame\()_KernelHeight[rsp]
        mov     r12,\KernelFrame\()_KernelWidth[rsp]
.if \OutputCount\() == 1
        mov     r13,\KernelFrame\()_InputBase[rsp]
        mov     r14,\KernelFrame\()_InputWidth[rsp]
        neg     r13                         # keep negative for lea usage below
.endif
        ClearBlock \FilterCount\(), \OutputCount\()
        test    r11,r11                     # zero sized kernel?
        jz      .L\KernelType\().\FilterCount\().\OutputCount\().HandlePostProcessing

.L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextRow:
        mov     rax,r12                     # reload kernel width remaining

.L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextColumn:
.if \OutputCount\() == 1
        lea     rbx,[rcx+r13]               # compute (Input - InputBase)
        cmp     rbx,r14                     # (Input - InputBase) >= InputWidth?
        jae     .L\KernelType\().\FilterCount\().\OutputCount\().SkipOverPadding
.endif
.if \OutputCount\() > 3
        lea     r14,[r9+r9*2]
        add     r14,rcx                     # compute input plus 3 blocks
.endif
.if \FilterCount\() > 2
        lea     rbx,[rdx+rsi*2]             # compute filter plus 2 rows
.endif
.ifeqs "\KernelType\()","Nchwc"
.if \BlockSize\() == 16
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), \Index\()*16*4, \Index\()*4
        .endr
.else
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7
            ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), (\Index\()-4)*8*4, \Index\()*4
        .endr
.endif
.else
        ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), 0, 0
.endif

.L\KernelType\().\FilterCount\().\OutputCount\().SkipOverPadding:
        add     rcx,rbp                     # advance input by dilation width
.ifeqs "\KernelType\()","Nchwc"
        add     rdx,\BlockSize\()*\BlockSize\()*4
                                            # advance filter by 8i8o/16i16o block
.else
        add     rdx,\BlockSize\()*4         # advance filter by 8o/16o block
.endif
        dec     rax                         # decrement columns remaining
        jnz     .L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextColumn
        add     rcx,r15                     # advance input to next row
.if \OutputCount\() == 1
        sub     r13,\KernelFrame\()_DilatedInputWidth[rsp]
                                            # advance input base to next row
.endif
        dec     r11                         # decrement rows remaining
        jnz     .L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextRow

//
// Handle post processing of the output block.
//

.L\KernelType\().\FilterCount\().\OutputCount\().HandlePostProcessing:
        mov     edx,DWORD PTR \KernelFrame\()_Flags[rsp]
.if \FilterCount\() > 1
        mov     rax,\KernelFrame\()_OutputStride[rsp]
.endif
        mov     rcx,\KernelFrame\()_Bias[rsp]
        call    MlasConvPostProcessFloat\Isa\()Filter\FilterCount\()Output\OutputCount\()

        .endm

/*++

Macro Description:

    This macro generates code for the inner convolution kernel.

Arguments:

    KernelType - Supplies the type of kernel to be generated.

    BlockSize - Supplies the number of elements per block.

    Isa - Supplies the instruction set architecture string for function tags.

    BiasFilter - Supplies a non-blank value if the address of the filter buffer
        should be biased to point to the middle of a OIhw8i8o block in order to
        reduce the code size from relative byte offsets.

--*/

        .macro SconvKernelFunction KernelType, BlockSize, Isa, BiasFilter

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

Arguments:

    Input (rdi) - Supplies the address of the input buffer.

        The address is biased to include padding blocks for the left width
        dimension. The address is not biased to include padding rows for the
        left height dimension  these are accounted for in the outer kernel.

    Filter (rsi) - Supplies the address of the filter buffer.

    Output (rdx) - Supplies the address of the output buffer.

    StrideWidth (rcx) - Supplies the length in bytes of the blocked stride width.

    DilationWidth (r8) - Supplies the length in bytes of the blocked dilation
        width.

    FilterCount (r9) - Supplies the number of filters to process in this
        iteration.

    InputStride - Supplies the length in bytes to advance the input buffer to
        the next input row.

    FilterStride - Supplies the length in bytes to advance the filter buffer
        to the next set of filters.

    OutputStride - Supplies the length in bytes to advance the output buffer
        to the next output address associated with the next set of filters.

    KernelHeight - Supplies the height of the kernel to apply. This height may
        be less than the original kernel height after removing any padding
        rows.

    KernelWidth - Supplies the width of the kernel to apply.

    InputBase - Supplies the address of the valid input buffer.

        This parameter is similar to the Input parameter, but does not include
        the padding blocks for the left width dimension. This parameter is used
        with the following InputWidth parameter in order to validate that the
        current input buffer address in bounds and not in the left or right
        width padding region.

    InputWidth - Supplies the length in bytes of the blocked input width.

    DilatedInputWidth - Supplies the length in bytes to advance the input base
        buffer to the next input row including dilation.

    OutputCountLeftPad - Supplies the number of output elements that include
        one or more padding elements from the left edge.

    OutputCount - Supplies the number of output elements that do not include
        any padding elements.

    OutputCountRightPad - Supplies the number of output elements that include
        one or more padding elements from the right edge.

    Bias - Supplies the address of the bias buffer.

    Flags - Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConv\KernelType\()FloatKernel\Isa\()

        push    rbp
        push    rbx
        push    r15
        push    r14
        push    r13
        push    r12
.ifeqs "\BiasFilter\()","BiasFilter"
        add_immed rsi,4*8*4
.endif
        push    rsi
        mov     rsi,.LSconvKernelFrame_FilterStride[rsp]
        mov     r15,.LSconvKernelFrame_InputStride[rsp]
        mov     rbp,r8                      # shuffle to Win64 register usage
        mov     r11,r9
        mov     r8,rdx
        mov     r9,rcx

//
// Process the specified number of filter rows.
//

        cmp     r11,3
        je      .L\KernelType\().ProcessFilterCount3
        jb      .L\KernelType\().ProcessFilterCountLessThan3
        ProcessFilterCountN .LSconvKernelFrame, \KernelType\(), 4
        jmp     .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCount3:
        ProcessFilterCountN .LSconvKernelFrame, \KernelType\(), 3
        jmp     .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCountLessThan3:
        cmp     r11,2
        jb      .L\KernelType\().ProcessFilterCount1
        ProcessFilterCountN .LSconvKernelFrame, \KernelType\(), 2
        jmp     .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCount1:
        ProcessFilterCountN .LSconvKernelFrame, \KernelType\(), 1

//
// Restore non-volatile registers and return.
//

.L\KernelType\().ExitKernel:
.ifnes "\Isa\()","Sse"
        vzeroupper
.endif
        pop     rsi                         # clear Filter local
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rbx
        pop     rbp
        ret

.ifnes "\Isa\()","Sse"

//
// Generate out-of-band helpers for handling output blocks involving padding.
//

        .irp FilterCount, 1, 2, 3, 4

MlasConv\KernelType\()FloatSingle\Isa\()Filter\FilterCount\():
        ProcessOutputCountN \Isa\(), .LSconvKernelSingleFrame, \KernelType\(), \BlockSize\(), \FilterCount\(), 1
        add     rdi,r9                      # advance input by 1 element
        dec     r10                         # decrement output count remaining
        jnz     MlasConv\KernelType\()FloatSingle\Isa\()Filter\FilterCount\()
        ret

        .endr

.endif

        .endm

/*++

Macro Description:

    This macro generates code for the inner convolution kernel for the special
    case of a depthwise separable convolution.

Arguments:

    BlockSize - Supplies the number of elements per block.

    Isa - Supplies the instruction set architecture string for function tags.

--*/

        .macro SconvKernelDepthwiseFunction BlockSize, Isa

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

    Depthwise separable convolutions are a form of grouped convolution where
    the number of input and output channels per group are one.

Arguments:

    Input (rdi) - Supplies the address of the input buffer.

        The address is biased to include padding blocks for the left width
        dimension. The address is not biased to include padding rows for the
        left height dimension  these are accounted for in the outer kernel.

    Filter (rsi) - Supplies the address of the filter buffer.

    Output (rdx) - Supplies the address of the output buffer.

    StrideWidth (rcx) - Supplies the length in bytes of the blocked stride width.

    DilationWidth (r8) - Supplies the length in bytes of the blocked dilation
        width.

    InputStride (r9) - Supplies the length in bytes to advance the input buffer
        to the next input row.

    KernelHeight - Supplies the height of the kernel to apply. This height may
        be less than the original kernel height after removing any padding
        rows.

    KernelWidth - Supplies the width of the kernel to apply.

    InputBase - Supplies the address of the valid input buffer.

        This parameter is similar to the Input parameter, but does not include
        the padding blocks for the left width dimension. This parameter is used
        with the following InputWidth parameter in order to validate that the
        current input buffer address in bounds and not in the left or right
        width padding region.

    InputWidth - Supplies the length in bytes of the blocked input width.

    DilatedInputWidth - Supplies the length in bytes to advance the input base
        buffer to the next input row including dilation.

    OutputCountLeftPad - Supplies the number of output elements that include
        one or more padding elements from the left edge.

    OutputCount - Supplies the number of output elements that do not include
        any padding elements.

    OutputCountRightPad - Supplies the number of output elements that include
        one or more padding elements from the right edge.

    Bias - Supplies the address of the bias buffer.

    Flags - Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvDepthwiseFloatKernel\Isa\()

        push    rbp
        push    rbx
        push    r15
        push    r14
        push    r13
        push    r12
        mov     rbp,r8                      # shuffle to Win64 register usage
        mov     r15,r9
        mov     r8,rdx
        mov     r9,rcx

//
// Process the specified number of filter rows.
//

        ProcessFilterCountN .LSconvKernelDepthwiseFrame, Depthwise, 1

//
// Restore non-volatile registers and return.
//

.LDepthwise.ExitKernel:
.ifnes "\Isa\()","Sse"
        vzeroupper
.endif
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        pop     rbx
        pop     rbp
        ret

.ifnes "\Isa\()","Sse"

//
// Generate out-of-band helpers for handling output blocks involving padding.
//

MlasConvDepthwiseFloatSingle\Isa\()Filter1:
        ProcessOutputCountN \Isa\(), .LSconvKernelDepthwiseSingleFrame, Depthwise, \BlockSize\(), 1, 1
        add     rdi,r9                      # advance input by 1 element
        dec     r10                         # decrement output count remaining
        jnz     MlasConvDepthwiseFloatSingle\Isa\()Filter1
        ret

.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute the convolution for a vector of input
    blocks and a vector of filter blocks to produce a matrix of output blocks
    for a pointwise convolution.

Arguments:

    Isa - Supplies the instruction set architecture string for function tags.

    BlockSize - Supplies the number of elements per block.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    rdi - Supplies the address of the input buffer.

    rsi - Supplies the FilterStride parameter (see function description).

    rbp - Supplies the InputStride parameter (see function description).

    r8 - Supplies the address of the output buffer.

    r9 - Supplies the StrideWidth parameter (see function description).

    r12 - Supplies the address of the filter buffer.

--*/

        .macro ProcessPointwiseOutputCountN Isa, BlockSize, FilterCount, OutputCount

        mov     rcx,rdi
        mov     rdx,r12
        mov     r11,.LSconvKernelPointwiseFrame_InputChannels[rsp]
        ClearBlock \FilterCount\(), \OutputCount\()

.LPointwise.\FilterCount\().\OutputCount\().ProcessNextInputBlock:
.if \OutputCount\() > 3
        lea     r14,[r9+r9*2]
        add     r14,rcx                     # compute input plus 3 blocks
.endif
.if \FilterCount\() > 2
        lea     rbx,[rdx+rsi*2]             # compute filter plus 2 rows
.endif
.if \BlockSize\() == 16
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            ComputeBlock Pointwise, \FilterCount\(), \OutputCount\(), \Index\()*16*4, \Index\()*4
        .endr
.else
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7
            ComputeBlock Pointwise, \FilterCount\(), \OutputCount\(), (\Index\()-4)*8*4, \Index\()*4
        .endr
.endif
        add     rcx,rbp                     # advance input to next channel block
        add     rdx,\BlockSize\()*\BlockSize\()*4
                                            # advance filter by 8i8o/16i16o block
        dec     r11                         # decrement input blocks remaining
        jnz     .LPointwise.\FilterCount\().\OutputCount\().ProcessNextInputBlock

//
// Handle post processing of the output block.
//

        mov     edx,DWORD PTR .LSconvKernelPointwiseFrame_Flags[rsp]
.if \FilterCount\() > 1
        mov     rax,.LSconvKernelPointwiseFrame_OutputStride[rsp]
.endif
        mov     rcx,.LSconvKernelPointwiseFrame_Bias[rsp]
        call    MlasConvPostProcessFloat\Isa\()Filter\FilterCount\()Output\OutputCount\()

        .endm

/*++

Macro Description:

    This macro generates code for the inner convolution kernel for the special
    case where the kernel dimensions are 1.

Arguments:

    Isa - Supplies the instruction set architecture string for function tags.

    BiasFilter - Supplies a non-blank value if the address of the filter buffer
        should be biased to point to the middle of a OIhw8i8o block in order to
        reduce the code size from relative byte offsets.

--*/

        .macro SconvKernelPointwiseFunction Isa, BiasFilter

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

    Pointwise convolutions have a kernel size of one. To simplify this
    implementation, no input padding is allowed, which matches typical usage in
    models.

Arguments:

    Input (rdi) - Supplies the address of the input buffer.

    Filter (rsi) - Supplies the address of the filter buffer.

    Output (rdx) - Supplies the address of the output buffer.

    StrideWidth (rcx) - Supplies the length in bytes of the blocked stride width.

    InputChannels (r8) - Supplies the number of input channels to process.

    FilterCount (r9) - Supplies the number of rows from the filter to process.

    InputStride - Supplies the length in bytes to advance the input buffer to
        the next input channel of the same input row.

    FilterStride - Supplies the length in bytes to advance the filter buffer
        to the next set of filters.

    OutputStride - Supplies the length in bytes to advance the output buffer
        to the next output address associated with the next set of filters.

    OutputCount - Supplies the number of output elements.

    Bias - Supplies the address of the bias buffer.

    Flags - Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvPointwiseFloatKernel\Isa\()

        push    rbp
        push    rbx
        push    r14
        push    r12
        push    r8
.ifeqs "\BiasFilter\()","BiasFilter"
        lea     r12,[rsi+4*8*4]
.else
        mov     r12,rsi
.endif
        mov     r10,.LSconvKernelPointwiseFrame_OutputCount[rsp]
        mov     rsi,.LSconvKernelPointwiseFrame_FilterStride[rsp]
        mov     rbp,.LSconvKernelPointwiseFrame_InputStride[rsp]
        mov     r11,r9                      # shuffle to Win64 register usage
        mov     r8,rdx
        mov     r9,rcx

//
// Process the specified number of filter rows.
//

        cmp     r11,3
        je      .LPointwise.ProcessFilterCount3
        jb      .LPointwise.ProcessFilterCountLessThan3
        ProcessPointwiseFilterCountN 4
        jmp     .LPointwise.ExitKernel

.LPointwise.ProcessFilterCount3:
        ProcessPointwiseFilterCountN 3
        jmp     .LPointwise.ExitKernel

.LPointwise.ProcessFilterCountLessThan3:
        cmp     r11,2
        jb      .LPointwise.ProcessFilterCount1
        ProcessPointwiseFilterCountN 2
        jmp     .LPointwise.ExitKernel

.LPointwise.ProcessFilterCount1:
        ProcessPointwiseFilterCountN 1

//
// Restore non-volatile registers and return.
//

.LPointwise.ExitKernel:
.ifnes "\Isa\()","Sse"
        vzeroupper
.endif
        pop     r8                          # clear InputChannels local
        pop     r12
        pop     r14
        pop     rbx
        pop     rbp
        ret

        .endm
