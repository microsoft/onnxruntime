/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DepthwiseConvSymKernelNeon.asm

Abstract:

    This module implements the kernels for the depthwise convolution
    operation with symmetrically quantized integer values

--*/

#include "kxarm64.h"

//
// Stack frame layout for the depthwise conv kernel.
// d8-d15, x19-x30 need to be preserved if used
//

#define  ConvSymDepthwiseKernelFrame_SavedNeonRegisters    (8 * 8)
#define  ConvSymDepthwiseKernelFrame_SavedRegisters            ConvSymDepthwiseKernelFrame_SavedNeonRegisters
#define  ConvSymDepthwiseKernelFrame_PostProcessParams     0 + ConvSymDepthwiseKernelFrame_SavedRegisters
#define  ConvSymDepthwiseKernelFrame_KernelFlags           8 + ConvSymDepthwiseKernelFrame_SavedRegisters

#define  ConvSymDepthwisePostProcessParams_Bias            0
#define  ConvSymDepthwisePostProcessParams_Scale           8
#define  ConvSymDepthwisePostProcessParams_Min             16
#define  ConvSymDepthwisePostProcessParams_Max             20
#define  ConvSymDepthwisePostProcessParams_ZeroPoint       24

#define  MLAS_CONV_SYM_FLAG_INPUT_DIRECT                   1
#define  MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE              2

        TEXTAREA

/*++

Routine Description:

    This routine is the inner kernel to compute a depthwise convolution for the
    elements of an output row for a set of filter rows.

Arguments:

    Input (x0) - Supplies the address of the indirection buffer.
 
    Filter (x1) - Supplies the address of the filter buffer.

    Output (x2) - Supplies the address of the output buffer.

    KernelSize (x3) - Supplies the size of the kernel.
 
    Channels (x4) - Supplies the number of input and output channels.
 
    ChannelOffset (x5) - Supplies the byte offset from the indirection buffer base
        address for this iteration.
 
    ChannelCount (x6) - Supplies the number of channels this iteration produces.
 
        This implementation requires the count to be 16 or 8
 
    OutputCount (x7)- Supplies the number of output elements this iteration produces.
 
        This implementation requires the count to be in the range 1 to 2.
 
    PostProcessParams - Supplies the address of the post process parameter block.
 
    KernelFlags - Supplies additional flags controlling the operation.

Return Value:

    None.

--*/

        NESTED_ENTRY MlasConvSymDepthwiseKernelNeon

        PROLOG_SAVE_REG_PAIR      d8,d9,#-64!
        PROLOG_NOP        ldr     x8,[sp,#ConvSymDepthwiseKernelFrame_PostProcessParams]
        PROLOG_NOP        mov     w10,#0x80808080
        PROLOG_SAVE_REG_PAIR      d10,d11,#16
        PROLOG_SAVE_REG_PAIR      d12,d13,#32
        PROLOG_SAVE_REG_PAIR      d14,d15,#48
        dup     v8.4s,w10                   // bit flip vector
        ldr     x16,[x8,#ConvSymDepthwisePostProcessParams_Bias]
        cmp     x7,2
        add     x9,x0,x3,lsl#3              // x9 -> &A1
        add     x14,x0,x3,lsl#4             // x14 -> &A2
        add     x15,x9,x3,lsl#4             // x15 -> &A3
        csel    x9,x0,x9,lo                 // x9 -> &A0 if OutputCount < 2
        csel    x14,x0,x14,ls               // x14 -> &A0 if OutputCount <= 2
        ldr     x11,[x9],#8                 // x11 -> A1 iter 0
        cmp     x7,4
        ldp     q24,q25,[x16],#32           // init accumulators with bias
        csel    x15,x0,x15,lo               // x15 -> &A0 if OutputCount < 4
        cmp     x6,16
        ldr     x10,[x0],#8                 // x10 -> A0 iter 0
        b.lo    Process8Channels

//
// Process an input block of length Channels for each element of the kernel.
//
// Filter:  v0,
//          v1       // unroll
// Input:
// x0  -> x10 -> v4
//     -> x12 -> v2  // unroll
// x9  -> x11 -> v6
//     -> x13 -> v10 // unroll
// x14 -> x10 -> v4
//     -> x12 -> v2  // unroll
// x15 -> x11 -> v6
//     -> x13 -> v10 // unroll
//

Process16Channels
        cmp     x3,1
        ldp     q26,q27,[x16]
        b.eq    ProcC16P1

        ldr     x12,[x0],#8                 // x12 -> A0 iter 1
        ldr     x13,[x9],#8                 // x13 -> A1 iter 1
        mov     v28.16b,v24.16b
        mov     v29.16b,v25.16b
        ld1     {v0.16b},[x1],x4            // filter iter 0
        ld1     {v1.16b},[x1],x4            // filter iter 1
        mov     v16.16b,v24.16b
        mov     v17.16b,v25.16b
        ldr     q4,[x10,x5]                 // A0 iter 0
        mov     v20.16b,v24.16b
        ldr     x10,[x14],#8                // x10 -> A2 iter 0
        mov     v21.16b,v25.16b
        ldr     q6,[x11,x5]                 // A1 iter 0
        mov     v30.16b,v26.16b
        ldr     x11,[x15],#8                // x11 -> A3 iter 0
        mov     v31.16b,v27.16b
        ldr     q2,[x12,x5]                 // A0 iter 1
        subs    x3,x3,2                     // decrement input blocks remaining
        mov     v18.16b,v26.16b
        ldr     x12,[x14],#8                // x12 -> A2 iter 1
        mov     v19.16b,v27.16b
        ldr     q10,[x13,x5]                // A1 iter 1
        mov     v22.16b,v26.16b
        ldr     x13,[x15],#8                // x13 -> A3 iter 1
        mov     v23.16b,v27.16b

BlockLoopC16

        //
        // Process 2 pixels, and load next two pixels
        //
        eor     v4.16b,v4.16b,v8.16b        // fix sign bits
        smull   v12.8h,v0.8b,v4.8b
        smull2  v13.8h,v0.16b,v4.16b
        eor     v6.16b,v6.16b,v8.16b
        ldr     q4,[x10,x5]                 // A2 iter 0
        b.eq    EpilogueC16P2
        smull   v14.8h,v0.8b,v6.8b
        ldr     x10,[x0],#8                 // x10 -> A0 iter 2
        smull2  v15.8h,v0.16b,v6.16b
        eor     v2.16b,v2.16b,v8.16b
        cmp     x3,1
        ldr     q6,[x11,x5]                 // A3 iter 0
        smlal   v12.8h,v1.8b,v2.8b
        ldr     x11,[x9],#8                 // x11 -> A1 iter 2
        smlal2  v13.8h,v1.16b,v2.16b
        b.eq    EpilogueC16P3             // 3 pixel remains      
        eor     v10.16b,v10.16b,v8.16b
        ldr     q2,[x12,x5]                 // A2 iter 1
        smlal   v14.8h,v1.8b,v10.8b
        ldr     x12,[x0],#8                 // x12 -> A0 iter 3
        smlal2  v15.8h,v1.16b,v10.16b
        ldr     q10,[x13,x5]                // A3 iter 1
        saddw   v24.4s,v24.4s,v12.4h
        saddw2  v25.4s,v25.4s,v12.8h
        ldr     x13,[x9],#8                 // x13 -> A1 iter 3
        saddw   v26.4s,v26.4s,v13.4h
        saddw2  v27.4s,v27.4s,v13.8h
        saddw   v28.4s,v28.4s,v14.4h
        saddw2  v29.4s,v29.4s,v14.8h
        saddw   v30.4s,v30.4s,v15.4h
        saddw2  v31.4s,v31.4s,v15.8h
        eor     v4.16b,v4.16b,v8.16b
        subs    x3,x3,2                     // decrement input blocks remaining
        smull   v12.8h,v0.8b,v4.8b
        smull2  v13.8h,v0.16b,v4.16b
        eor     v6.16b,v6.16b,v8.16b
        ldr     q4,[x10,x5]                 // A0 iter 2
        smull   v14.8h,v0.8b,v6.8b
        ldr     x10,[x14],#8                // x10 -> A2 iter 2
        smull2  v15.8h,v0.16b,v6.16b
        ldr     q6,[x11,x5]                 // A1 iter 2
        eor     v2.16b,v2.16b,v8.16b
        ld1     {v0.16b},[x1],x4            // filter iter 2
        smlal   v12.8h,v1.8b,v2.8b
        ldr     x11,[x15],#8                // x11 -> A3 iter 2
        smlal2  v13.8h,v1.16b,v2.16b
        eor     v10.16b,v10.16b,v8.16b
        ldr     q2,[x12,x5]                 // A0 iter 3
        smlal   v14.8h,v1.8b,v10.8b
        ldr     x12,[x14],#8                // x12 -> A2 iter 3
        smlal2  v15.8h,v1.16b,v10.16b
        ldr     q10,[x13,x5]                // A1 iter 3
        saddw   v16.4s,v16.4s,v12.4h
        saddw2  v17.4s,v17.4s,v12.8h
        ld1     {v1.16b},[x1],x4            // filter iter 3
        saddw   v18.4s,v18.4s,v13.4h
        saddw2  v19.4s,v19.4s,v13.8h
        ldr     x13,[x15],#8                // x13 -> A3 iter 3
        saddw   v20.4s,v20.4s,v14.4h
        saddw2  v21.4s,v21.4s,v14.8h
        saddw   v22.4s,v22.4s,v15.4h
        saddw2  v23.4s,v23.4s,v15.8h
        b       BlockLoopC16

EpilogueC16P2
        //
        // Loop epilogue (process last 2 pixels) mixed
        // with loading of dequantization params
        //
        smull   v14.8h,v0.8b,v6.8b
        smull2  v15.8h,v0.16b,v6.16b
        ldr     q6,[x11,x5]                 // A3 iter 0
        eor     v2.16b,v2.16b,v8.16b
        smlal   v12.8h,v1.8b,v2.8b
        smlal2  v13.8h,v1.16b,v2.16b
        eor     v10.16b,v10.16b,v8.16b
        ldr     q2,[x12,x5]                 // A2 iter 1
        smlal   v14.8h,v1.8b,v10.8b
        smlal2  v15.8h,v1.16b,v10.16b
        ldr     q10,[x13,x5]                // A3 iter 1
        saddw   v24.4s,v24.4s,v12.4h
        saddw2  v25.4s,v25.4s,v12.8h
        saddw   v26.4s,v26.4s,v13.4h
        saddw2  v27.4s,v27.4s,v13.8h
        saddw   v28.4s,v28.4s,v14.4h
        saddw2  v29.4s,v29.4s,v14.8h
        saddw   v30.4s,v30.4s,v15.4h
        saddw2  v31.4s,v31.4s,v15.8h
        ldr     w9,[sp,#ConvSymDepthwiseKernelFrame_KernelFlags]
        eor     v4.16b,v4.16b,v8.16b
        ldr     x12,[x8,#ConvSymDepthwisePostProcessParams_Scale]
        smull   v12.8h,v0.8b,v4.8b
        smull2  v13.8h,v0.16b,v4.16b
        eor     v6.16b,v6.16b,v8.16b
        ldr     w15,[x8,#ConvSymDepthwisePostProcessParams_ZeroPoint]
        smull   v14.8h,v0.8b,v6.8b
        smull2  v15.8h,v0.16b,v6.16b
        eor     v2.16b,v2.16b,v8.16b
        smlal   v12.8h,v1.8b,v2.8b
        smlal2  v13.8h,v1.16b,v2.16b
        eor     v10.16b,v10.16b,v8.16b
        smlal   v14.8h,v1.8b,v10.8b
        smlal2  v15.8h,v1.16b,v10.16b
        tst     w9,#MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        ld1r    {v4.4s},[x12]               // load scale val
        b.eq    SkipScaleVecLoad2
        ldp     q4,q11,[x12],#32            // load scale vector if per channel
        ldp     q6,q9,[x12]
SkipScaleVecLoad2
        saddw   v16.4s,v16.4s,v12.4h
        saddw2  v17.4s,v17.4s,v12.8h
        saddw   v18.4s,v18.4s,v13.4h
        saddw2  v19.4s,v19.4s,v13.8h
        saddw   v20.4s,v20.4s,v14.4h
        saddw2  v21.4s,v21.4s,v14.8h
        saddw   v22.4s,v22.4s,v15.4h
        saddw2  v23.4s,v23.4s,v15.8h
        b       Dequantization

ProcC16P1
        //
        // Channel 16 kernel size 1
        // TODO!! is this reachable at all?
        //
        ldr     x12,[x14],#8                // x12 -> A2
        ldr     x13,[x15],#8                // x13 -> A3
        mov     v28.16b,v24.16b
        mov     v29.16b,v25.16b
        ld1     {v0.16b},[x1]
        mov     v16.16b,v24.16b
        mov     v17.16b,v25.16b
        ldr     q4,[x10,x5]
        mov     v20.16b,v24.16b
        mov     v21.16b,v25.16b
        ldr     q6,[x11,x5]
        mov     v30.16b,v26.16b
        mov     v31.16b,v27.16b
        ldr     q2,[x12,x5]
        subs    x3,x3,2                     // decrement input blocks remaining
        mov     v18.16b,v26.16b
        mov     v19.16b,v27.16b
        ldr     q10,[x13,x5]
        mov     v22.16b,v26.16b
        mov     v23.16b,v27.16b
        b       EpilogueC16P1

EpilogueC16P3
        //
        // Loop epilogue (process last 2 pixels) mixed
        // with loading of dequantization params
        //
        eor     v10.16b,v10.16b,v8.16b
        ldr     q2,[x12,x5]                 // A2 iter 1
        smlal   v14.8h,v1.8b,v10.8b
        ldr     x12,[x14],#8                // x12 -> A2 iter 2
        smlal2  v15.8h,v1.16b,v10.16b
        ldr     q10,[x13,x5]                // A3 iter 1
        saddw   v24.4s,v24.4s,v12.4h
        saddw2  v25.4s,v25.4s,v12.8h
        ldr     x13,[x15],#8                // x13 -> A3 iter 2
        saddw   v26.4s,v26.4s,v13.4h
        saddw2  v27.4s,v27.4s,v13.8h
        saddw   v28.4s,v28.4s,v14.4h
        saddw2  v29.4s,v29.4s,v14.8h
        saddw   v30.4s,v30.4s,v15.4h
        saddw2  v31.4s,v31.4s,v15.8h
        eor     v4.16b,v4.16b,v8.16b
        smull   v12.8h,v0.8b,v4.8b
        smull2  v13.8h,v0.16b,v4.16b
        eor     v6.16b,v6.16b,v8.16b
        ldr     q4,[x10,x5]                 // A0 iter 2
        smull   v14.8h,v0.8b,v6.8b
        smull2  v15.8h,v0.16b,v6.16b
        ld1     {v0.16b},[x1]               // filter iter 2
        ldr     q6,[x11,x5]                 // A1 iter 2
        eor     v2.16b,v2.16b,v8.16b
        smlal   v12.8h,v1.8b,v2.8b
        smlal2  v13.8h,v1.16b,v2.16b
        eor     v10.16b,v10.16b,v8.16b
        ldr     q2,[x12,x5]                 // A2 iter 2
        smlal   v14.8h,v1.8b,v10.8b
        smlal2  v15.8h,v1.16b,v10.16b
        ldr     q10,[x13,x5]                // A3 iter 2
        saddw   v16.4s,v16.4s,v12.4h
        saddw2  v17.4s,v17.4s,v12.8h
        saddw   v18.4s,v18.4s,v13.4h
        saddw2  v19.4s,v19.4s,v13.8h
        saddw   v20.4s,v20.4s,v14.4h
        saddw2  v21.4s,v21.4s,v14.8h
        saddw   v22.4s,v22.4s,v15.4h
        saddw2  v23.4s,v23.4s,v15.8h

EpilogueC16P1
        //
        // Loop epilogue (process last single pixel) mixed with loading of dequantization params
        //
        ldr     w9,[sp,#ConvSymDepthwiseKernelFrame_KernelFlags]
        eor     v4.16b,v4.16b,v8.16b
        ldr     x12,[x8,#ConvSymDepthwisePostProcessParams_Scale]
        smull   v12.8h,v0.8b,v4.8b
        smull2  v13.8h,v0.16b,v4.16b
        eor     v6.16b,v6.16b,v8.16b
        ldr     w15,[x8,#ConvSymDepthwisePostProcessParams_ZeroPoint]
        smull   v14.8h,v0.8b,v6.8b
        smull2  v15.8h,v0.16b,v6.16b
        saddw   v24.4s,v24.4s,v12.4h
        saddw2  v25.4s,v25.4s,v12.8h
        saddw   v26.4s,v26.4s,v13.4h
        saddw2  v27.4s,v27.4s,v13.8h
        saddw   v28.4s,v28.4s,v14.4h
        saddw2  v29.4s,v29.4s,v14.8h
        saddw   v30.4s,v30.4s,v15.4h
        saddw2  v31.4s,v31.4s,v15.8h
        eor     v2.16b,v2.16b,v8.16b
        smull   v12.8h,v0.8b,v2.8b
        smull2  v13.8h,v0.16b,v2.16b
        eor     v10.16b,v10.16b,v8.16b
        smull   v14.8h,v0.8b,v10.8b
        smull2  v15.8h,v0.16b,v10.16b
        tst     w9,#MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        ld1r    {v4.4s},[x12]               // load scale val
        b.eq    SkipScaleVecLoad
        ldp     q4,q11,[x12],#32            // load scale vector if per channel
        ldp     q6,q9,[x12]
SkipScaleVecLoad
        saddw   v16.4s,v16.4s,v12.4h
        saddw2  v17.4s,v17.4s,v12.8h
        saddw   v18.4s,v18.4s,v13.4h
        saddw2  v19.4s,v19.4s,v13.8h
        saddw   v20.4s,v20.4s,v14.4h
        saddw2  v21.4s,v21.4s,v14.8h
        saddw   v22.4s,v22.4s,v15.4h
        saddw2  v23.4s,v23.4s,v15.8h

Dequantization
        scvtf   v24.4s,v24.4s               // convert to float
        scvtf   v25.4s,v25.4s
        scvtf   v26.4s,v26.4s
        scvtf   v27.4s,v27.4s
        scvtf   v28.4s,v28.4s
        scvtf   v29.4s,v29.4s
        scvtf   v30.4s,v30.4s
        scvtf   v31.4s,v31.4s
        scvtf   v16.4s,v16.4s
        scvtf   v17.4s,v17.4s
        scvtf   v18.4s,v18.4s
        scvtf   v19.4s,v19.4s
        scvtf   v20.4s,v20.4s
        scvtf   v21.4s,v21.4s
        scvtf   v22.4s,v22.4s
        scvtf   v23.4s,v23.4s
        b.ne    SkipScaleBroadcast
        mov     v11.16b,v4.16b               // broadcast scale val if not per channel
        mov     v6.16b,v4.16b
        mov     v9.16b,v4.16b
SkipScaleBroadcast
        fmul    v24.4s,v24.4s,v4.4s         // multiply by scale
        fmul    v25.4s,v25.4s,v11.4s
        fmul    v26.4s,v26.4s,v6.4s
        fmul    v27.4s,v27.4s,v9.4s
        fmul    v28.4s,v28.4s,v4.4s
        fmul    v29.4s,v29.4s,v11.4s
        fmul    v30.4s,v30.4s,v6.4s
        fmul    v31.4s,v31.4s,v9.4s
        fmul    v16.4s,v16.4s,v4.4s
        fmul    v17.4s,v17.4s,v11.4s
        fmul    v18.4s,v18.4s,v6.4s
        fmul    v19.4s,v19.4s,v9.4s
        fmul    v20.4s,v20.4s,v4.4s
        fmul    v21.4s,v21.4s,v11.4s
        fmul    v22.4s,v22.4s,v6.4s
        fmul    v23.4s,v23.4s,v9.4s
        fcvtns  v24.4s,v24.4s               // convert to int
        fcvtns  v25.4s,v25.4s
        fcvtns  v26.4s,v26.4s
        fcvtns  v27.4s,v27.4s
        fcvtns  v28.4s,v28.4s
        fcvtns  v29.4s,v29.4s
        fcvtns  v30.4s,v30.4s
        fcvtns  v31.4s,v31.4s
        fcvtns  v16.4s,v16.4s
        fcvtns  v17.4s,v17.4s
        fcvtns  v18.4s,v18.4s
        fcvtns  v19.4s,v19.4s
        fcvtns  v20.4s,v20.4s
        fcvtns  v21.4s,v21.4s
        fcvtns  v22.4s,v22.4s
        fcvtns  v23.4s,v23.4s
        sqxtn   v24.4h,v24.4s               // shorten to int16
        sqxtn   v26.4h,v26.4s
        sqxtn2  v24.8h,v25.4s
        sqxtn2  v26.8h,v27.4s
        sqxtn   v28.4h,v28.4s
        sqxtn   v30.4h,v30.4s
        sqxtn2  v28.8h,v29.4s
        sqxtn2  v30.8h,v31.4s
        dup     v0.8h,w15
        sqxtn   v16.4h,v16.4s
        sqxtn   v18.4h,v18.4s
        sqxtn2  v16.8h,v17.4s
        sqxtn2  v18.8h,v19.4s
        sqxtn   v20.4h,v20.4s
        sqxtn   v22.4h,v22.4s
        sqxtn2  v20.8h,v21.4s
        sqxtn2  v22.8h,v23.4s
        sqadd   v24.8h,v24.8h,v0.8h         // add zero point
        sqadd   v26.8h,v26.8h,v0.8h
        sqadd   v28.8h,v28.8h,v0.8h
        sqadd   v30.8h,v30.8h,v0.8h
        sqadd   v16.8h,v16.8h,v0.8h
        sqadd   v18.8h,v18.8h,v0.8h
        sqadd   v20.8h,v20.8h,v0.8h
        sqadd   v22.8h,v22.8h,v0.8h
        sqxtun  v24.8b,v24.8h               // shorten to int8
        sqxtun2 v24.16b,v26.8h
        sqxtun  v28.8b,v28.8h
        sqxtun2 v28.16b,v30.8h
        sqxtun  v16.8b,v16.8h
        sqxtun2 v16.16b,v18.8h
        sqxtun  v20.8b,v20.8h
        sqxtun2 v20.16b,v22.8h
        cmp     x7,2                        // OutputCount < 2 ?
        st1     {v24.16b},[x2],x4
        b.lo    ExitKernel                // exit if OutputCount < 2
        st1     {v28.16b},[x2],x4
        b.ls    ExitKernel                // exit if OutputCount <=2
        cmp     x7,4                        // OutputCount < 4 ?
        st1     {v16.16b},[x2],x4
        b.lo    ExitKernel                // exit if OutputCount < 4
        str     q20,[x2]

ExitKernel
        EPILOG_RESTORE_REG_PAIR d14,d15,#48
        EPILOG_RESTORE_REG_PAIR d12,d13,#32
        EPILOG_RESTORE_REG_PAIR d10,d11,#16
        EPILOG_RESTORE_REG_PAIR d8,d9,#64!
        EPILOG_RETURN

Process8Channels
        cmp     x3,1
        b.eq    ProcC8P1

        ldr     x12,[x0],#8                 // x12 -> A0 iter 1
        ldr     x13,[x9],#8                 // x13 -> A1 iter 1
        ld1     {v0.8b},[x1],x4             // filter iter 0
        ld1     {v1.8b},[x1],x4             // filter iter 1
        ldr     d4,[x10,x5]                 // A0 iter 0
        ldr     x10,[x14],#8                // x10 -> A2 iter 0
        mov     v28.16b,v24.16b
        ldr     d6,[x11,x5]                 // A1 iter 0
        mov     v29.16b,v25.16b
        ldr     x11,[x15],#8                // x11 -> A3 iter 0
        mov     v16.16b,v24.16b
        ldr     d2,[x12,x5]                 // A0 iter 1
        mov     v17.16b,v25.16b
        ldr     x12,[x14],#8                // x12 -> A2 iter 1
        subs    x3,x3,2                     // decrement input blocks remaining
        ldr     d10,[x13,x5]                // A1 iter 1
        mov     v20.16b,v24.16b
        ldr     x13,[x15],#8                // x13 -> A3 iter 1
        mov     v21.16b,v25.16b

BlockLoopC8
        //
        // Process 2 pixels, and load next two pixels
        //
        eor     v4.8b,v4.8b,v8.8b           // fix sign bits
        eor     v6.8b,v6.8b,v8.8b
        smull   v12.8h,v0.8b,v4.8b
        ldr     d4,[x10,x5]                 // A2 iter 0
        smull   v14.8h,v0.8b,v6.8b
        b.eq    EpilogueC8P2
        ldr     x10,[x0],#8                 // x10 -> A0 iter 2
        eor     v2.8b,v2.8b,v8.8b
        eor     v10.8b,v10.8b,v8.8b
        ldr     d6,[x11,x5]                 // A3 iter 0
        cmp     x3,1
        smlal   v12.8h,v1.8b,v2.8b
        ldr     x11,[x9],#8                 // x11 -> A1 iter 2
        smlal   v14.8h,v1.8b,v10.8b
        ldr     d2,[x12,x5]                 // A2 iter 1
        b.eq    EpilogueC8P3                // 3 pixel remains      
        ldr     d10,[x13,x5]                // A3 iter 1
        saddw   v24.4s,v24.4s,v12.4h
        ldr     x12,[x0],#8                 // x12 -> A0 iter 3
        saddw2  v25.4s,v25.4s,v12.8h
        ldr     x13,[x9],#8                 // x13 -> A1 iter 3
        saddw   v28.4s,v28.4s,v14.4h
        saddw2  v29.4s,v29.4s,v14.8h
        eor     v4.8b,v4.8b,v8.8b
        eor     v6.8b,v6.8b,v8.8b
        subs    x3,x3,2                     // decrement input blocks remaining
        smull   v12.8h,v0.8b,v4.8b
        ldr     d4,[x10,x5]                 // A0 iter 2
        smull   v14.8h,v0.8b,v6.8b
        ldr     x10,[x14],#8                // x10 -> A2 iter 2
        ldr     d6,[x11,x5]                 // A1 iter 2
        eor     v2.8b,v2.8b,v8.8b
        eor     v10.8b,v10.8b,v8.8b
        ld1     {v0.8b},[x1],x4             // filter iter 2
        smlal   v12.8h,v1.8b,v2.8b
        ldr     x11,[x15],#8                // x11 -> A3 iter 2
        ldr     d2,[x12,x5]                 // A0 iter 3
        smlal   v14.8h,v1.8b,v10.8b
        ldr     x12,[x14],#8                // x12 -> A2 iter 3
        saddw   v16.4s,v16.4s,v12.4h
        ldr     d10,[x13,x5]                // A1 iter 3
        saddw2  v17.4s,v17.4s,v12.8h
        ld1     {v1.8b},[x1],x4             // filter iter 3
        saddw   v20.4s,v20.4s,v14.4h
        ldr     x13,[x15],#8                // x13 -> A3 iter 3
        saddw2  v21.4s,v21.4s,v14.8h
        b       BlockLoopC8

EpilogueC8P2
        //
        // Loop epilogue (process last 2 pixels) mixed
        // with loading of dequantization params
        //
        ldr     d6,[x11,x5]                 // A3 iter 0
        eor     v2.8b,v2.8b,v8.8b
        eor     v10.8b,v10.8b,v8.8b
        smlal   v12.8h,v1.8b,v2.8b
        ldr     d2,[x12,x5]                 // A2 iter 1
        smlal   v14.8h,v1.8b,v10.8b
        ldr     d10,[x13,x5]                // A3 iter 1
        saddw   v24.4s,v24.4s,v12.4h
        saddw2  v25.4s,v25.4s,v12.8h
        saddw   v28.4s,v28.4s,v14.4h
        saddw2  v29.4s,v29.4s,v14.8h
        ldr     w9,[sp,#ConvSymDepthwiseKernelFrame_KernelFlags]
        eor     v4.8b,v4.8b,v8.8b
        eor     v6.8b,v6.8b,v8.8b
        smull   v12.8h,v0.8b,v4.8b
        ldr     x12,[x8,#ConvSymDepthwisePostProcessParams_Scale]
        smull   v14.8h,v0.8b,v6.8b
        ldr     w15,[x8,#ConvSymDepthwisePostProcessParams_ZeroPoint]
        eor     v2.8b,v2.8b,v8.8b
        eor     v10.8b,v10.8b,v8.8b
        smlal   v12.8h,v1.8b,v2.8b
        smlal   v14.8h,v1.8b,v10.8b
        tst     w9,#MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        ld1r    {v4.4s},[x12]               // load scale val
        b.eq    SkipScaleVecLoad2C8
        ldp     q4,q11,[x12],#32            // load scale vector if per channel
SkipScaleVecLoad2C8
        saddw   v16.4s,v16.4s,v12.4h
        saddw2  v17.4s,v17.4s,v12.8h
        saddw   v20.4s,v20.4s,v14.4h
        saddw2  v21.4s,v21.4s,v14.8h
        b       DequantC8

ProcC8P1
        //
        // Channel 8 kernel size 1
        // TODO!! is this reachable at all?
        //
        ldr     x12,[x14],#8                // x12 -> A2
        mov     v28.16b,v24.16b
        ldr     x13,[x15],#8                // x13 -> A3
        mov     v29.16b,v25.16b
        ld1     {v0.8b},[x1]
        mov     v16.16b,v24.16b
        ldr     d4,[x10,x5]
        mov     v17.16b,v25.16b
        ldr     d6,[x11,x5]
        mov     v20.16b,v24.16b
        ldr     d2,[x12,x5]
        subs    x3,x3,2                     // decrement input blocks remaining
        ldr     d10,[x13,x5]
        mov     v21.16b,v25.16b
        b       EpilogueC8P1

EpilogueC8P3
        //
        // Loop epilogue (process 2 of last 3 pixels)
        //
        ldr     x12,[x14],#8                // x12 -> A2 iter 2
        ldr     d10,[x13,x5]                // A3 iter 1
        saddw   v24.4s,v24.4s,v12.4h
        saddw2  v25.4s,v25.4s,v12.8h
        ldr     x13,[x15],#8                // x13 -> A3 iter 2
        saddw   v28.4s,v28.4s,v14.4h
        saddw2  v29.4s,v29.4s,v14.8h
        eor     v4.8b,v4.8b,v8.8b
        eor     v6.8b,v6.8b,v8.8b
        smull   v12.8h,v0.8b,v4.8b
        ldr     d4,[x10,x5]                 // A0 iter 2
        smull   v14.8h,v0.8b,v6.8b
        ld1     {v0.8b},[x1]                // filter iter 2
        eor     v2.8b,v2.8b,v8.8b
        eor     v10.8b,v10.8b,v8.8b
        ldr     d6,[x11,x5]                 // A1 iter 2
        smlal   v12.8h,v1.8b,v2.8b
        ldr     d2,[x12,x5]                 // A2 iter 2
        smlal   v14.8h,v1.8b,v10.8b
        ldr     d10,[x13,x5]                // A3 iter 2
        saddw   v16.4s,v16.4s,v12.4h
        saddw2  v17.4s,v17.4s,v12.8h
        saddw   v20.4s,v20.4s,v14.4h
        saddw2  v21.4s,v21.4s,v14.8h

EpilogueC8P1
        //
        // Loop epilogue (process last single pixel) mixed with loading of dequantization params
        //
        ldr     w9,[sp,#ConvSymDepthwiseKernelFrame_KernelFlags]
        eor     v4.8b,v4.8b,v8.8b
        eor     v6.8b,v6.8b,v8.8b
        ldr     x12,[x8,#ConvSymDepthwisePostProcessParams_Scale]
        smull   v12.8h,v0.8b,v4.8b
        ldr     w15,[x8,#ConvSymDepthwisePostProcessParams_ZeroPoint]
        smull   v14.8h,v0.8b,v6.8b
        saddw   v24.4s,v24.4s,v12.4h
        saddw2  v25.4s,v25.4s,v12.8h
        saddw   v28.4s,v28.4s,v14.4h
        saddw2  v29.4s,v29.4s,v14.8h
        eor     v2.8b,v2.8b,v8.8b
        eor     v10.8b,v10.8b,v8.8b
        smull   v12.8h,v0.8b,v2.8b
        smull   v14.8h,v0.8b,v10.8b
        tst     w9,#MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        ld1r    {v4.4s},[x12]               // load scale val
        b.eq    SkipScaleVecLoadC8
        ldp     q4,q11,[x12]                // load scale vector if per channel
SkipScaleVecLoadC8
        saddw   v16.4s,v16.4s,v12.4h
        saddw2  v17.4s,v17.4s,v12.8h
        saddw   v20.4s,v20.4s,v14.4h
        saddw2  v21.4s,v21.4s,v14.8h

DequantC8
        scvtf   v24.4s,v24.4s               // convert to float
        scvtf   v25.4s,v25.4s
        scvtf   v28.4s,v28.4s
        scvtf   v29.4s,v29.4s
        scvtf   v16.4s,v16.4s
        scvtf   v17.4s,v17.4s
        scvtf   v20.4s,v20.4s
        scvtf   v21.4s,v21.4s
        b.ne    SkipScaleBroadcastC8
        mov     v11.16b,v4.16b              // broadcast scale val if not per channel
SkipScaleBroadcastC8
        fmul    v24.4s,v24.4s,v4.4s         // multiply by scale
        fmul    v25.4s,v25.4s,v11.4s
        fmul    v28.4s,v28.4s,v4.4s
        fmul    v29.4s,v29.4s,v11.4s
        fmul    v16.4s,v16.4s,v4.4s
        fmul    v17.4s,v17.4s,v11.4s
        fmul    v20.4s,v20.4s,v4.4s
        fmul    v21.4s,v21.4s,v11.4s
        fcvtns  v24.4s,v24.4s               // convert to int
        fcvtns  v25.4s,v25.4s
        fcvtns  v28.4s,v28.4s
        fcvtns  v29.4s,v29.4s
        fcvtns  v16.4s,v16.4s
        fcvtns  v17.4s,v17.4s
        fcvtns  v20.4s,v20.4s
        fcvtns  v21.4s,v21.4s
        dup     v0.8h,w15
        sqxtn   v24.4h,v24.4s               // shorten to int16
        sqxtn2  v24.8h,v25.4s
        sqxtn   v28.4h,v28.4s
        sqxtn2  v28.8h,v29.4s
        sqxtn   v16.4h,v16.4s
        sqxtn2  v16.8h,v17.4s
        sqxtn   v20.4h,v20.4s
        sqxtn2  v20.8h,v21.4s
        sqadd   v24.8h,v24.8h,v0.8h         // add zero point
        sqadd   v28.8h,v28.8h,v0.8h
        sqadd   v16.8h,v16.8h,v0.8h
        sqadd   v20.8h,v20.8h,v0.8h
        sqxtun  v24.8b,v24.8h               // shorten to int8
        sqxtun  v28.8b,v28.8h
        sqxtun  v16.8b,v16.8h
        sqxtun  v20.8b,v20.8h
        cmp     x7,2                        // OutputCount < 2 ?
        st1     {v24.8b},[x2],x4
        b.lo    ExitKernel                // exit if OutputCount < 2
        st1     {v28.8b},[x2],x4
        b.ls    ExitKernel                // exit if OutputCount <=2
        cmp     x7,4                        // OutputCount < 4 ?
        st1     {v16.8b},[x2],x4
        b.lo    ExitKernel                // exit if OutputCount < 4
        str     d20,[x2]
        b       ExitKernel
        NESTED_END MlasConvSymDepthwiseKernelNeon

        END
