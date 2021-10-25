/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    ConvSymKernelNeon.s

Abstract:

    This module implements the kernels for the symmetric quantized integer
    convolution operation.

--*/

#include "asmmacro.h"

        .equ    .LMLAS_CONV_SYM_FLAG_INPUT_DIRECT,      1
        .equ    .LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE, 2

//
// Stack frame layout for the symmetric convolution kernel.
// d8-d15, x19-x30 need to be preserved if used
//
        .equ    .LConvSymFrame_SavedNeonRegisters, (8 * 12)
        .equ    .LConvSymFrame_SavedRegisters, .LConvSymFrame_SavedNeonRegisters
        .equ    .LConvSymFrame_PostProcessParams, 0 + .LConvSymFrame_SavedRegisters
        .equ    .LConvSymFrame_KernelFlags, 8 + .LConvSymFrame_SavedRegisters

        .equ    .LConvSymPostProcessParams_Bias,      0
        .equ    .LConvSymPostProcessParams_Scale,     8
        .equ    .LConvSymPostProcessParams_Min,       16
        .equ    .LConvSymPostProcessParams_Max,       20
        .equ    .LConvSymPostProcessParams_ZeroPoint, 24

        .text

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

Arguments:

    Input (x0) - Supplies the address of the input buffer.

        If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is set, then the input buffer points
        directly at the input tensor.

        If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is clear, then the input buffer is an
        indirection buffer. Every pointer in the indirection buffer points at a
        InputChannels length vector (either from the input tensor or a vector of
        padding values). These are grouped in batches of length KernelSize.
        These batches are then repeated OutputCount times.

    Filter (x1) - Supplies the address of the filter buffer.

    Output (x2) - Supplies the address of the output buffer.

    KernelSize (x3) - Supplies the size of the kernel.

        If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is set, then kernel size should be 1.

    InputChannels (x4) - Supplies the number of input channels.

        This implementation requires the count to be a multiple of 8.

    OutputChannels (x5) - Supplies the number of output channels.

    ChannelCount (x6) - Supplies the number of channels this iteration produces.

        This implementation requires the count to be 8.

    OutputCount (x7) - Supplies the number of output elements this iteration produces.

        This implementation requires the count to be 1 or 2.

    PostProcessParams - Supplies the address of the post process parameter block.

    KernelFlags - Supplies additional flags controlling the operation.

Return Value:

    None.

--*/

// input: 
// filter:
// output:


        FUNCTION_ENTRY MlasConvSymKernelNeon

        stp     d8,d9,[sp,#-96]!
        ldr     x8,[sp,#.LConvSymFrame_PostProcessParams]
        ldrb    w10,[sp,#.LConvSymFrame_KernelFlags]
        stp     d10,d11,[sp,#16]
        stp     d12,d13,[sp,#32]
        stp     d14,d15,[sp,#48]
        stp     x19,x20,[sp,#64]
        stp     x21,x22,[sp,#80]

        add     x16,x2,x5,lsl#1         // x16 -> C2
        mov     x22,x0                  // store A

        lsl     x20,x3,#4               // x20 = ks*16
        mov     w14, #128
        add     x17,x16,x5              // x17 -> C3

        ldr     w13,[x8, #.LConvSymPostProcessParams_ZeroPoint]
        ldr     x11,[x8,#.LConvSymPostProcessParams_Bias]
        ldr     x12,[x8,#.LConvSymPostProcessParams_Scale]

        add     x21,x20,x3,lsl#3        // x21 = ks*24
        add     x5,x2,x5                // x5 -> C1

        dup     v8.16b,w14
        dup     v9.8h,w13

 .LConvSym.OutputChannelLoop:
        ldp     s16,s17,[x11],8         // init accumulators with bias
        ldp     s18,s19,[x11],8

        mov     x0, x22                // reload A
        mov     x8, x3                 // reload kernel size

        mov     v20.16b,v16.16b
        mov     v24.16b,v16.16b
        mov     v28.16b,v16.16b
        mov     v21.16b,v17.16b
        mov     v25.16b,v17.16b
        mov     v29.16b,v17.16b
        mov     v22.16b,v18.16b
        mov     v26.16b,v18.16b
        mov     v30.16b,v18.16b
        mov     v23.16b,v19.16b
        mov     v27.16b,v19.16b
        mov     v31.16b,v19.16b

// Nested loops, inner loop: input channel; outter loop: kernel size
// Each inner iteration processes 8 input channels, 2 output pixels, 8 output channels.
//
//                                            B 16x4
//                                  ---------------------------------------------
//                                  |v4.[0-7]b  v5.[0-7]b  v6.[0-7]b  v7.[0-7]b  |
//                                  |v4.[8-15]b v5.[8-15]b v6.[8-15]b v7.[8-15]b |
//            A 4x16                ---------------------------------------------
//       --------------             ---------------------------------------------
// x13-> |v0.[0-7]b,v0.[8-15]b|     |v16.4s     v17.4s     v18.4s     v19.4s |
// x14-> |v1.[0-7]b,v1.[8-15]b|     |v20.4s     v21.4s     v22.4s     v23.4s |
// x15-> |v2.[0-7]b,v2.[8-15]b|     |v24.4s     v25.4s     v26.4s     v27.4s |
// x19-> |v3.[0-7]b,v3.[8-15]b|     |v28.4s     v29.4s     v30.4s     v31.4s |
//       --------------             ---------------------------------------------
// When Input Channels greater than 16, unroll:
// A registers v6 v7,
// B registers v8 v9
//

.LConvSym.KernelSizeLoop:

        mov     x9,x4
        # Load next 4 A pointers
        tst     w10,#.LMLAS_CONV_SYM_FLAG_INPUT_DIRECT
        beq     .LConvSym.InputIndirection
.LConvSym.InputDirect:
        add     x14,x0,x4               // x14 -> A1
        cmp     x7,3                    // test OutputCount with 3
        mov     x13,x0                  // x13 -> A0
        add     x15,x0,x4,lsl#1         // x15 -> A2
        add     x19,x14,x4,lsl#1        // x19 -> A3
        bhi     .LConvSym.Blockloop
        mov     x19,x0
        bhs     .LConvSym.Blockloop
        cmp     x7,2
        mov     x15,x0
        bhs     .LConvSym.Blockloop
        mov     x14,x0
.LConvSym.InputIndirection:
        ldr     x13,[x0]                // x13 -> A0
        cmp     x7,2                    // test if OutputCount < 2
        mov     x14,x13                 // x14 -> A0
        mov     x15,x13                 // x15 -> A0
        mov     x19,x13                 // x19 -> A0
        blo     .LConvSym.SkipLoadA1
        ldr     x14,[x0,x3,lsl#3]       // x14 -> A1
        bls     .LConvSym.SkipLoadA1
        ldr     x15,[x0,x20]       // x15 -> A2
        cmp     x7,4
        blo     .LConvSym.SkipLoadA1
        ldr     x19,[x0,x21]            // x19 -> A3
.LConvSym.SkipLoadA1:
        add     x0,x0,8                 // indirect A advance to next pointer, prepare for kernel size loop

.LConvSym.Blockloop:
        subs    x9, x9, 16
        ld1     {v0.16b}, [x13],#16
        ld1     {v4.16b}, [x1],#16
        ld1     {v5.16b}, [x1],#16
        ld1     {v6.16b}, [x1],#16
        ld1     {v7.16b}, [x1],#16
        eor     v0.16b, v0.16b, v8.16b
        ld1     {v1.16b}, [x14],#16
        ld1     {v2.16b}, [x15],#16
        ld1     {v3.16b}, [x19],#16
        smull   v12.8h, v0.8b, v4.8b
        smull   v13.8h, v0.8b, v5.8b
        smull   v14.8h, v0.8b, v6.8b
        smull   v15.8h, v0.8b, v7.8b
        eor     v1.16b, v1.16b, v8.16b
        eor     v2.16b, v2.16b, v8.16b
        eor     v3.16b, v3.16b, v8.16b
        smlal2  v12.8h, v0.16b, v4.16b
        smlal2  v13.8h, v0.16b, v5.16b
        smlal2  v14.8h, v0.16b, v6.16b
        smlal2  v15.8h, v0.16b, v7.16b
        sadalp  v16.4s, v12.8h
        sadalp  v17.4s, v13.8h
        sadalp  v18.4s, v14.8h
        sadalp  v19.4s, v15.8h

        smull   v12.8h, v1.8b, v4.8b
        smull   v13.8h, v1.8b, v5.8b
        smull   v14.8h, v1.8b, v6.8b
        smull   v15.8h, v1.8b, v7.8b
        smlal2  v12.8h, v1.16b, v4.16b
        smlal2  v13.8h, v1.16b, v5.16b
        smlal2  v14.8h, v1.16b, v6.16b
        smlal2  v15.8h, v1.16b, v7.16b
        sadalp  v20.4s, v12.8h
        sadalp  v21.4s, v13.8h
        sadalp  v22.4s, v14.8h
        sadalp  v23.4s, v15.8h

        smull   v12.8h, v2.8b, v4.8b
        smull   v13.8h, v2.8b, v5.8b
        smull   v14.8h, v2.8b, v6.8b
        smull   v15.8h, v2.8b, v7.8b
        smlal2  v12.8h, v2.16b, v4.16b
        smlal2  v13.8h, v2.16b, v5.16b
        smlal2  v14.8h, v2.16b, v6.16b
        smlal2  v15.8h, v2.16b, v7.16b
        sadalp  v24.4s, v12.8h
        sadalp  v25.4s, v13.8h
        sadalp  v26.4s, v14.8h
        sadalp  v27.4s, v15.8h

        smull   v12.8h, v3.8b, v4.8b
        smull   v13.8h, v3.8b, v5.8b
        smull   v14.8h, v3.8b, v6.8b
        smull   v15.8h, v3.8b, v7.8b
        smlal2  v12.8h, v3.16b, v4.16b
        smlal2  v13.8h, v3.16b, v5.16b
        smlal2  v14.8h, v3.16b, v6.16b
        smlal2  v15.8h, v3.16b, v7.16b
        sadalp  v28.4s, v12.8h
        sadalp  v29.4s, v13.8h
        sadalp  v30.4s, v14.8h
        sadalp  v31.4s, v15.8h
        bhi    .LConvSym.Blockloop

        subs    x8, x8, 1
        bhi    .LConvSym.KernelSizeLoop

.LConvSym.Requantize:
        tst     w10,#.LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        beq     .LConvSym.BroadcastScaleValue
        ld1     {v4.4s}, [x12]   // load scale vector
        b       .LConvSym.AccumulatorsToFloat

.LConvSym.BroadcastScaleValue:
        ld1r   {v4.4s},[x12]                // load scale Value

.LConvSym.AccumulatorsToFloat:
        addp    v16.4s,v16.4s,v17.4s
        addp    v18.4s,v18.4s,v19.4s
        addp    v20.4s,v20.4s,v21.4s
        addp    v22.4s,v22.4s,v23.4s
        addp    v24.4s,v24.4s,v25.4s
        addp    v26.4s,v26.4s,v27.4s
        addp    v28.4s,v28.4s,v29.4s
        addp    v30.4s,v30.4s,v31.4s
        addp    v0.4s,v16.4s,v18.4s
        addp    v1.4s,v20.4s,v22.4s
        addp    v2.4s,v24.4s,v26.4s
        addp    v3.4s,v28.4s,v30.4s
        scvtf   v0.4s,v0.4s                // convert to float
        scvtf   v1.4s,v1.4s
        scvtf   v2.4s,v2.4s
        scvtf   v3.4s,v3.4s
        fmul    v0.4s,v0.4s,v4.4s         // multiply by scale
        fmul    v1.4s,v1.4s,v4.4s
        fmul    v2.4s,v2.4s,v4.4s
        fmul    v3.4s,v3.4s,v4.4s
        fcvtns   v0.4s,v0.4s          // convert to int
        fcvtns   v1.4s,v1.4s
        fcvtns   v2.4s,v2.4s
        fcvtns   v3.4s,v3.4s
        sqxtn   v0.4h,v0.4s
        sqxtn2  v0.8h,v1.4s
        sqxtn   v2.4h,v2.4s
        sqxtn2  v2.8h,v3.4s
        sqadd   v0.8h,v0.8h,v9.8h
        sqadd   v2.8h,v2.8h,v9.8h
        sqxtun   v0.8b,v0.8h                 // shorten to int8
        sqxtun2  v0.16b,v2.8h

        cmp     x7,2
        st1     {v0.s}[0],[x2],4               // full 4byte store to c
        b.lo    .LConvSym.OutputChannelLoop.Check
        st1     {v0.s}[1],[x5],4               // full 4byte store to c + outputchannel
        b.ls    .LConvSym.OutputChannelLoop.Check
        cmp     x7,4
        st1     {v0.s}[2],[x16],4              // full 4byte store to c + 2*outputchannel
        b.lo    .LConvSym.OutputChannelLoop.Check
        st1     {v0.s}[3],[x17],4              // full 4byte store to c + 3*outputchannel

.LConvSym.OutputChannelLoop.Check:
        subs    x6,x6,4
        b.hi    .LConvSym.OutputChannelLoop

.LConvSym.ExitKernel:
        ldp     x21,x22,[sp,#80]
        ldp     x19,x20,[sp,#64]
        ldp     d14,d15,[sp,#48]
        ldp     d12,d13,[sp,#32]
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#96
        ret

        .end
