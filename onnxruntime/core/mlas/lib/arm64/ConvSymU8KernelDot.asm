/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    ConvSymKernelNeonDot.asm

Abstract:

    This module implements the kernels for the symmetric quantized integer
    convolution operation.

--*/

#include "kxarm64.h"

#define     MLAS_CONV_SYM_FLAG_INPUT_DIRECT      1
#define     MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE 2

//
// Stack frame layout for the symmetric convolution kernel.
// d8-d15, x19-x30 need to be preserved if used
//
#define     ConvSymFrame_SavedNeonRegisters   (8 * 8)
#define     ConvSymFrame_SavedRegisters           ConvSymFrame_SavedNeonRegisters
#define     ConvSymFrame_PostProcessParams    0 + ConvSymFrame_SavedRegisters
#define     ConvSymFrame_KernelFlags          8 + ConvSymFrame_SavedRegisters

#define     ConvSymPostProcessParams_Bias      0
#define     ConvSymPostProcessParams_Scale     8
#define     ConvSymPostProcessParams_Min       16
#define     ConvSymPostProcessParams_Max       20
#define     ConvSymPostProcessParams_ZeroPoint 24

        TEXTAREA

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

Arguments:

    Input (x0) - Points to the input buffer.

        If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is set, then the input buffer points
        directly at the input tensor.

        If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is clear, then the input buffer is an
        indirection buffer. Every pointer in the indirection buffer points at a
        InputChannels length vector (either from the input tensor or a vector of
        padding values). These are grouped in batches of length KernelSize.
        These batches are then repeated OutputCount times.

    Filter (x1) - Points to the filter buffer.

    Output (x2) - Points the output buffer.

    KernelSize (x3/x9) - Size of the kernel (most commonly. 3x3=9, 5x5=25).

        If MLAS_CONV_SYM_FLAG_INPUT_DIRECT is set, then kernel size should be 1.

    InputChannels (x4/x7) - Number of input channels.

    OutputChannels (x5) - Number of output channels.

    ChannelCount (x6) - Number of output channels this iteration produces.

    OutputCount (x7) - Number of output elements this iteration produces.

        This implementation requires the count to be no larger than 4.

    PostProcessParams (x8) - Points to the post process parameter block.

    KernelFlags - (w10) Additional flags controlling the operation.

Return Value:

    None.

--*/
        NESTED_ENTRY MlasConvSymU8KernelDot

        PROLOG_SAVE_REG_PAIR  d8,d9,#-64!
        PROLOG_NOP    ldr     x8,[sp,#ConvSymFrame_PostProcessParams]
        PROLOG_NOP    ldr     w10,[sp,#ConvSymFrame_KernelFlags]
        PROLOG_SAVE_REG_PAIR  d10,d11,#16
        PROLOG_SAVE_REG_PAIR  d12,d13,#32
        PROLOG_SAVE_REG_PAIR  x19,x20,#48

        // compute C pointers: x2, x16, x17, x5
        cmp     x7,2                    // OutputCount < 2 ?
        add     x16,x2,x5               // x16 -> C1
        lsl     x3,x3,#3                // KernelSize * sizeof(int8_t*)
        csel    x16,x2,x16,lo           // if OutputCount < 2  x16/C1 -> C0
        mov     x20,x4
        add     x4,x4,3                 // InputChannels align to 4
        add     x17,x16,x5              // x17 -> C2
        ldr     x11,[x8,#ConvSymPostProcessParams_Bias]
        csel    x17,x16,x17,ls          // if OutputCount <= 2  x17/C2 -> C1
        bic     x4,x4,3
        cmp     x7,4                    // OutputCount < 4 ?
        add     x5,x17,x5               // x5 -> C3
        ldr     x19,[x8,#ConvSymPostProcessParams_Scale]
        csel    x5,x17,x5,lo            // if OutputCount < 4  x5/C3 -> C2
        movi    v12.16b,128             // for top bit flipping

OutputChannelLoop
        ldp     q16,q20,[x11],32        // Init accumulators with biases
        mov     v17.16b,v16.16b
        mov     v18.16b,v16.16b
        ldp     q24,q28,[x11],32
        mov     v19.16b,v16.16b
        mov     v21.16b,v20.16b
        mov     v22.16b,v20.16b
        mov     v23.16b,v20.16b
        mov     v25.16b,v24.16b
        mov     v26.16b,v24.16b
        mov     v27.16b,v24.16b
        mov     v29.16b,v28.16b
        mov     v30.16b,v28.16b
        mov     v31.16b,v28.16b
        mov     x9,x3                   // restore KernelSize * sizeof(int8_t*)

KernelSizeLoop
        tst     w10,#MLAS_CONV_SYM_FLAG_INPUT_DIRECT
        beq     InputIndirection

InputDirect
        cmp     x16,x2
        mov     x12,x0                  // x12 -> A0
        add     x13,x0,x20              // x13 -> A1 = A0 + input channels
        csel    x13,x0,x13,eq
        cmp     x17,x16
        add     x14,x0,x20,lsl#1        // x14 -> A2
        csel    x14,x13,x14,eq
        cmp     x5,x17
        add     x15,x13,x20,lsl#1       // x15 -> A3
        csel    x15,x14,x15,eq
        b       FinishLoadAPtr

InputIndirection
        ldr     x12,[x0]                // x12 -> A0
        cmp     x16,x2
        b.eq    SkipLoadA1              // C1==C0 -> A0=A1=A2=A3
        cmp     x17,x16
        lsl     x14,x3,#1
        ldr     x13,[x0,x3]             // x13 -> A1
        b.eq    SkipLoadA2              // C2==C1 -> A1=A2=A3
        cmp     x5,x17
        add     x15,x3,x3,lsl#1
        ldr     x14,[x0,x14]            // x14 -> A2
        b.eq    SkipLoadA3              // C3==C2 -> A2=A3
        ldr     x15,[x0,x15]            // x15 -> A3
        b       FinishLoadAPtr
SkipLoadA1
        mov     x13,x12
SkipLoadA2
        mov     x14,x13
SkipLoadA3
        mov     x15,x14

// Register Usage
//                                            B (x1) -> 4x16
//                        ----------------------------------------------------------------------------
//                        |v4.b[0]..v4.b[12] v5.b[0]..v5.b[12]  v6.b[0]..v6.b[12]   v7.b[0]..v7.b[12]|
//                        |  ...      ...     ...       ...       ...       ...       ...     ...    |
//                        |v4.b[3]..v4.b[15] v5.b[3]..v5.b[15]  v6.b[3]..v6.b[15]   v7.b[3]..v7.b[15]|
//            A 4x4       ----------------------------------------------------------------------------
//     ------------------ ----------------------------------------------------------------------------
// x12 |v0.b[0]..v0.b[3]| |v16.s[0]_v16.s[3] v20.s[0]_v20.s[3]  v24.s[0]_v24.s[3]   v28.s[0]_v28.s[3]|  x2
// x13 |v1.b[0]..v1.b[3]| |v17.s[0]_v17.s[3] v21.s[0]_v21.s[3]  v25.s[0]_v25.s[3]   v29.s[0]_v29.s[3]| x16
// x14 |v2.b[0]..v2.b[3]| |v18.s[0]_v18.s[3] v22.s[0]_v23.s[3]  v26.s[0]_v26.s[3]   v30.s[0]_v31.s[3]| x17
// x15 |v3.b[0]..v3.b[3]| |v19.s[0]_v19.s[3] v23.s[0]_v23.s[3]  v27.s[0]_v27.s[3]   v31.s[0]_v31.s[3]|  x5
//     ------------------ ----------------------------------------------------------------------------

FinishLoadAPtr
        subs    x7,x4,16                // Need 16 input channels for loop
        add     x0,x0,8                 // indirect A advance to next pointer, prepare for kernel size loop
        b.lo    InChannels8

        ldr     d0,[x12],8
        ldr     q4,[x1],16
        ldr     d1,[x13],8
        subs    x7,x7,16
        ldr     d2,[x14],8
        ldr     d3,[x15],8
        ldr     q5,[x1],16
        ldr     q6,[x1],16
        ldr     q7,[x1],16
        b.lo    InChLoopEpilogue        // Need 32 input channels for main loop

InputChannelLoop
        eor     v0.8b,v0.8b,v12.8b
        eor     v1.8b,v1.8b,v12.8b
        sdot    v16.4s,v4.16b,v0.4b[0]
        eor     v2.8b,v2.8b,v12.8b
        sdot    v17.4s,v4.16b,v1.4b[0]
        eor     v3.8b,v3.8b,v12.8b
        ldr     d8,[x12],8
        sdot    v18.4s,v4.16b,v2.4b[0]
        sdot    v19.4s,v4.16b,v3.4b[0]
        ldr     q4,[x1],16
        sdot    v20.4s,v5.16b,v0.4b[0]
        sdot    v21.4s,v5.16b,v1.4b[0]
        ldr     d9,[x13],8
        sdot    v22.4s,v5.16b,v2.4b[0]
        sdot    v23.4s,v5.16b,v3.4b[0]
        ldr     q5,[x1],16
        sdot    v24.4s,v6.16b,v0.4b[0]
        sdot    v25.4s,v6.16b,v1.4b[0]
        ldr     d10,[x14],8
        sdot    v26.4s,v6.16b,v2.4b[0]
        sdot    v27.4s,v6.16b,v3.4b[0]
        ldr     q6,[x1],16
        sdot    v28.4s,v7.16b,v0.4b[0]
        sdot    v29.4s,v7.16b,v1.4b[0]
        ldr     d11,[x15],8
        sdot    v30.4s,v7.16b,v2.4b[0]
        sdot    v31.4s,v7.16b,v3.4b[0]
        ldr     q7,[x1],16
        sdot    v16.4s,v4.16b,v0.4b[1]
        sdot    v17.4s,v4.16b,v1.4b[1]
        sdot    v18.4s,v4.16b,v2.4b[1]
        sdot    v19.4s,v4.16b,v3.4b[1]
        ldr     q4,[x1],16
        sdot    v20.4s,v5.16b,v0.4b[1]
        sdot    v21.4s,v5.16b,v1.4b[1]
        sdot    v22.4s,v5.16b,v2.4b[1]
        sdot    v23.4s,v5.16b,v3.4b[1]
        ldr     q5,[x1],16
        sdot    v24.4s,v6.16b,v0.4b[1]
        sdot    v25.4s,v6.16b,v1.4b[1]
        sdot    v26.4s,v6.16b,v2.4b[1]
        sdot    v27.4s,v6.16b,v3.4b[1]
        ldr     q6,[x1],16
        sdot    v28.4s,v7.16b,v0.4b[1]
        sdot    v29.4s,v7.16b,v1.4b[1]
        sdot    v30.4s,v7.16b,v2.4b[1]
        sdot    v31.4s,v7.16b,v3.4b[1]
        eor     v8.8b,v8.8b,v12.8b
        ldr     q7,[x1],16
        eor     v9.8b,v9.8b,v12.8b
        sdot    v16.4s,v4.16b,v8.4b[0]
        eor     v10.8b,v10.8b,v12.8b
        sdot    v17.4s,v4.16b,v9.4b[0]
        ldr     d0,[x12],8
        eor     v11.8b,v11.8b,v12.8b
        sdot    v18.4s,v4.16b,v10.4b[0]
        sdot    v19.4s,v4.16b,v11.4b[0]
        ldr     q4,[x1],16
        sdot    v20.4s,v5.16b,v8.4b[0]
        sdot    v21.4s,v5.16b,v9.4b[0]
        ldr     d1,[x13],8
        sdot    v22.4s,v5.16b,v10.4b[0]
        sdot    v23.4s,v5.16b,v11.4b[0]
        ldr     q5,[x1],16
        sdot    v24.4s,v6.16b,v8.4b[0]
        sdot    v25.4s,v6.16b,v9.4b[0]
        ldr     d2,[x14],8
        sdot    v26.4s,v6.16b,v10.4b[0]
        sdot    v27.4s,v6.16b,v11.4b[0]
        ldr     q6,[x1],16
        sdot    v28.4s,v7.16b,v8.4b[0]
        sdot    v29.4s,v7.16b,v9.4b[0]
        ldr     d3,[x15],8
        sdot    v30.4s,v7.16b,v10.4b[0]
        sdot    v31.4s,v7.16b,v11.4b[0]
        ldr     q7,[x1],16
        sdot    v16.4s,v4.16b,v8.4b[1]
        sdot    v17.4s,v4.16b,v9.4b[1]
        sdot    v18.4s,v4.16b,v10.4b[1]
        sdot    v19.4s,v4.16b,v11.4b[1]
        ldr     q4,[x1],16
        sdot    v20.4s,v5.16b,v8.4b[1]
        sdot    v21.4s,v5.16b,v9.4b[1]
        sdot    v22.4s,v5.16b,v10.4b[1]
        sdot    v23.4s,v5.16b,v11.4b[1]
        ldr     q5,[x1],16
        sdot    v24.4s,v6.16b,v8.4b[1]
        sdot    v25.4s,v6.16b,v9.4b[1]
        sdot    v26.4s,v6.16b,v10.4b[1]
        sdot    v27.4s,v6.16b,v11.4b[1]
        ldr     q6,[x1],16
        sdot    v28.4s,v7.16b,v8.4b[1]
        sdot    v29.4s,v7.16b,v9.4b[1]
        subs    x7,x7,16                // InputChannels -= 16
        sdot    v30.4s,v7.16b,v10.4b[1]
        sdot    v31.4s,v7.16b,v11.4b[1]
        ldr     q7,[x1],16
        b.hs    InputChannelLoop

InChLoopEpilogue
        eor     v0.8b,v0.8b,v12.8b
        eor     v1.8b,v1.8b,v12.8b
        sdot    v16.4s,v4.16b,v0.4b[0]
        eor     v2.8b,v2.8b,v12.8b
        sdot    v17.4s,v4.16b,v1.4b[0]
        eor     v3.8b,v3.8b,v12.8b
        ldr     d8,[x12],8
        sdot    v18.4s,v4.16b,v2.4b[0]
        sdot    v19.4s,v4.16b,v3.4b[0]
        ldr     q4,[x1],16
        sdot    v20.4s,v5.16b,v0.4b[0]
        sdot    v21.4s,v5.16b,v1.4b[0]
        ldr     d9,[x13],8
        sdot    v22.4s,v5.16b,v2.4b[0]
        sdot    v23.4s,v5.16b,v3.4b[0]
        ldr     q5,[x1],16
        sdot    v24.4s,v6.16b,v0.4b[0]
        sdot    v25.4s,v6.16b,v1.4b[0]
        ldr     d10,[x14],8
        sdot    v26.4s,v6.16b,v2.4b[0]
        sdot    v27.4s,v6.16b,v3.4b[0]
        ldr     q6,[x1],16
        sdot    v28.4s,v7.16b,v0.4b[0]
        sdot    v29.4s,v7.16b,v1.4b[0]
        ldr     d11,[x15],8
        sdot    v30.4s,v7.16b,v2.4b[0]
        sdot    v31.4s,v7.16b,v3.4b[0]
        ldr     q7,[x1],16
        sdot    v16.4s,v4.16b,v0.4b[1]
        sdot    v17.4s,v4.16b,v1.4b[1]
        sdot    v18.4s,v4.16b,v2.4b[1]
        sdot    v19.4s,v4.16b,v3.4b[1]
        ldr     q4,[x1],16
        sdot    v20.4s,v5.16b,v0.4b[1]
        sdot    v21.4s,v5.16b,v1.4b[1]
        sdot    v22.4s,v5.16b,v2.4b[1]
        sdot    v23.4s,v5.16b,v3.4b[1]
        ldr     q5,[x1],16
        sdot    v24.4s,v6.16b,v0.4b[1]
        sdot    v25.4s,v6.16b,v1.4b[1]
        sdot    v26.4s,v6.16b,v2.4b[1]
        sdot    v27.4s,v6.16b,v3.4b[1]
        ldr     q6,[x1],16
        sdot    v28.4s,v7.16b,v0.4b[1]
        sdot    v29.4s,v7.16b,v1.4b[1]
        sdot    v30.4s,v7.16b,v2.4b[1]
        sdot    v31.4s,v7.16b,v3.4b[1]
        eor     v8.8b,v8.8b,v12.8b
        ldr     q7,[x1],16
        eor     v9.8b,v9.8b,v12.8b
        sdot    v16.4s,v4.16b,v8.4b[0]
        eor     v10.8b,v10.8b,v12.8b
        sdot    v17.4s,v4.16b,v9.4b[0]
        eor     v11.8b,v11.8b,v12.8b
        sdot    v18.4s,v4.16b,v10.4b[0]
        sdot    v19.4s,v4.16b,v11.4b[0]
        ldr     q4,[x1],16
        sdot    v20.4s,v5.16b,v8.4b[0]
        sdot    v21.4s,v5.16b,v9.4b[0]
        sdot    v22.4s,v5.16b,v10.4b[0]
        sdot    v23.4s,v5.16b,v11.4b[0]
        ldr     q5,[x1],16
        sdot    v24.4s,v6.16b,v8.4b[0]
        sdot    v25.4s,v6.16b,v9.4b[0]
        sdot    v26.4s,v6.16b,v10.4b[0]
        sdot    v27.4s,v6.16b,v11.4b[0]
        ldr     q6,[x1],16
        sdot    v28.4s,v7.16b,v8.4b[0]
        sdot    v29.4s,v7.16b,v9.4b[0]
        sdot    v30.4s,v7.16b,v10.4b[0]
        sdot    v31.4s,v7.16b,v11.4b[0]
        ldr     q7,[x1],16
        sdot    v16.4s,v4.16b,v8.4b[1]
        sdot    v17.4s,v4.16b,v9.4b[1]
        sdot    v18.4s,v4.16b,v10.4b[1]
        sdot    v19.4s,v4.16b,v11.4b[1]
        sdot    v20.4s,v5.16b,v8.4b[1]
        sdot    v21.4s,v5.16b,v9.4b[1]
        sdot    v22.4s,v5.16b,v10.4b[1]
        sdot    v23.4s,v5.16b,v11.4b[1]
        sdot    v24.4s,v6.16b,v8.4b[1]
        sdot    v25.4s,v6.16b,v9.4b[1]
        sdot    v26.4s,v6.16b,v10.4b[1]
        sdot    v27.4s,v6.16b,v11.4b[1]
        sdot    v28.4s,v7.16b,v8.4b[1]
        sdot    v29.4s,v7.16b,v9.4b[1]
        sdot    v30.4s,v7.16b,v10.4b[1]
        sdot    v31.4s,v7.16b,v11.4b[1]

        TST     x7,15
        B.NE    InChannels8             // 4 ~ 12 InputChannels

        subs    x9,x9,8                 // KernelSize-=1
        b.hi    KernelSizeLoop

Requantize
        tst     w10,#MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        ldr     w13,[x8,#ConvSymPostProcessParams_ZeroPoint]
        beq     BroadcastScaleValue
        ldp     q0,q1,[x19],32          // load scale vector
        ldp     q2,q3,[x19],32
        b       AccumulatorsToFloat

BroadcastScaleValue
        ld1r    {v0.4s},[x19]           // load scale Value
        mov     v1.16b, v0.16b
        mov     v2.16b, v0.16b
        mov     v3.16b, v0.16b

AccumulatorsToFloat
        scvtf   v16.4s,v16.4s           // convert to float
        scvtf   v17.4s,v17.4s
        scvtf   v18.4s,v18.4s
        scvtf   v19.4s,v19.4s
        scvtf   v20.4s,v20.4s
        scvtf   v21.4s,v21.4s
        scvtf   v22.4s,v22.4s
        scvtf   v23.4s,v23.4s
        scvtf   v24.4s,v24.4s
        scvtf   v25.4s,v25.4s
        scvtf   v26.4s,v26.4s
        scvtf   v27.4s,v27.4s
        scvtf   v28.4s,v28.4s
        scvtf   v29.4s,v29.4s
        scvtf   v30.4s,v30.4s
        scvtf   v31.4s,v31.4s
        fmul    v16.4s,v16.4s,v0.4s     // multiply by scale
        fmul    v17.4s,v17.4s,v0.4s
        fmul    v18.4s,v18.4s,v0.4s
        fmul    v19.4s,v19.4s,v0.4s
        fmul    v20.4s,v20.4s,v1.4s
        fmul    v21.4s,v21.4s,v1.4s
        fmul    v22.4s,v22.4s,v1.4s
        fmul    v23.4s,v23.4s,v1.4s
        fmul    v24.4s,v24.4s,v2.4s
        fmul    v25.4s,v25.4s,v2.4s
        fmul    v26.4s,v26.4s,v2.4s
        fmul    v27.4s,v27.4s,v2.4s
        fmul    v28.4s,v28.4s,v3.4s
        fmul    v29.4s,v29.4s,v3.4s
        fmul    v30.4s,v30.4s,v3.4s
        fmul    v31.4s,v31.4s,v3.4s
        fcvtns  v16.4s,v16.4s           // convert to int
        fcvtns  v17.4s,v17.4s
        fcvtns  v18.4s,v18.4s
        fcvtns  v19.4s,v19.4s
        fcvtns  v20.4s,v20.4s
        fcvtns  v21.4s,v21.4s
        fcvtns  v22.4s,v22.4s
        fcvtns  v23.4s,v23.4s
        fcvtns  v24.4s,v24.4s
        fcvtns  v25.4s,v25.4s
        fcvtns  v26.4s,v26.4s
        fcvtns  v27.4s,v27.4s
        fcvtns  v28.4s,v28.4s
        fcvtns  v29.4s,v29.4s
        fcvtns  v30.4s,v30.4s
        fcvtns  v31.4s,v31.4s

        sqxtn   v16.4h,v16.4s
        sqxtn   v17.4h,v17.4s
        sqxtn   v18.4h,v18.4s
        sqxtn   v19.4h,v19.4s
        sqxtn   v24.4h,v24.4s
        sqxtn   v25.4h,v25.4s
        sqxtn   v26.4h,v26.4s
        sqxtn   v27.4h,v27.4s
        dup     v4.8h,w13               // zero point
        sqxtn2  v16.8h,v20.4s
        sqxtn2  v17.8h,v21.4s
        sqxtn2  v18.8h,v22.4s
        sqxtn2  v19.8h,v23.4s
        sqxtn2  v24.8h,v28.4s
        sqxtn2  v25.8h,v29.4s
        sqxtn2  v26.8h,v30.4s
        sqxtn2  v27.8h,v31.4s
        sqadd   v16.8h,v16.8h,v4.8h
        sqadd   v17.8h,v17.8h,v4.8h
        sqadd   v18.8h,v18.8h,v4.8h
        sqadd   v19.8h,v19.8h,v4.8h
        sqadd   v24.8h,v24.8h,v4.8h
        sqadd   v25.8h,v25.8h,v4.8h
        sqadd   v26.8h,v26.8h,v4.8h
        sqadd   v27.8h,v27.8h,v4.8h
        sqxtun   v0.8b,v16.8h
        sqxtun   v1.8b,v17.8h
        sqxtun   v2.8b,v18.8h
        sqxtun   v3.8b,v19.8h
        sqxtun2  v0.16b,v24.8h
        sqxtun2  v1.16b,v25.8h
        subs    x6,x6,16            // processed 16 output channels
        sqxtun2  v2.16b,v26.8h
        sqxtun2  v3.16b,v27.8h
        b.lo    PartialStore

        st1     {v3.16b},[x5],16    // Store full 4 x 16
        st1     {v2.16b},[x17],16
        sub     x0,x0,x3            // Restore pointer to A: a -= ks
        st1     {v1.16b},[x16],16
        st1     {v0.16b},[x2],16
        b.hi    OutputChannelLoop

ExitKernel
        EPILOG_RESTORE_REG_PAIR  x19,x20,#48
        EPILOG_RESTORE_REG_PAIR  d12,d13,#32
        EPILOG_RESTORE_REG_PAIR  d10,d11,#16
        EPILOG_RESTORE_REG_PAIR  d8,d9,#64!
        EPILOG_RETURN

InChannels8
        tbz     x7,3,InChannels4
        ldr     d0,[x12],8
        ldr     q4,[x1],16
        ldr     d1,[x13],8
        ldr     d2,[x14],8
        ldr     d3,[x15],8
        eor     v0.8b,v0.8b,v12.8b
        ldr     q5,[x1],16
        eor     v1.8b,v1.8b,v12.8b
        sdot    v16.4s,v4.16b,v0.4b[0]
        sdot    v17.4s,v4.16b,v1.4b[0]
        eor     v2.8b,v2.8b,v12.8b
        ldp     q6,q7,[x1],32
        eor     v3.8b,v3.8b,v12.8b
        sdot    v18.4s,v4.16b,v2.4b[0]
        sdot    v19.4s,v4.16b,v3.4b[0]
        sdot    v20.4s,v5.16b,v0.4b[0]
        sdot    v21.4s,v5.16b,v1.4b[0]
        sdot    v22.4s,v5.16b,v2.4b[0]
        sdot    v23.4s,v5.16b,v3.4b[0]
        sdot    v24.4s,v6.16b,v0.4b[0]
        sdot    v25.4s,v6.16b,v1.4b[0]
        ldp     q4,q5,[x1],32
        sdot    v26.4s,v6.16b,v2.4b[0]
        sdot    v27.4s,v6.16b,v3.4b[0]
        sdot    v28.4s,v7.16b,v0.4b[0]
        sdot    v29.4s,v7.16b,v1.4b[0]
        sdot    v30.4s,v7.16b,v2.4b[0]
        sdot    v31.4s,v7.16b,v3.4b[0]
        sdot    v16.4s,v4.16b,v0.4b[1]
        sdot    v17.4s,v4.16b,v1.4b[1]
        ldp     q6,q7,[x1],32
        sdot    v18.4s,v4.16b,v2.4b[1]
        sdot    v19.4s,v4.16b,v3.4b[1]
        sdot    v20.4s,v5.16b,v0.4b[1]
        sdot    v21.4s,v5.16b,v1.4b[1]
        sdot    v22.4s,v5.16b,v2.4b[1]
        sdot    v23.4s,v5.16b,v3.4b[1]
        sdot    v24.4s,v6.16b,v0.4b[1]
        sdot    v25.4s,v6.16b,v1.4b[1]
        sdot    v26.4s,v6.16b,v2.4b[1]
        sdot    v27.4s,v6.16b,v3.4b[1]
        sdot    v28.4s,v7.16b,v0.4b[1]
        sdot    v29.4s,v7.16b,v1.4b[1]
        sdot    v30.4s,v7.16b,v2.4b[1]
        sdot    v31.4s,v7.16b,v3.4b[1]
        tbz     x7,2,SkipInCh4

InChannels4
        ldr     s0,[x12],4
        ldr     q4,[x1],16
        ldr     s1,[x13],4
        ldr     s2,[x14],4
        ldr     s3,[x15],4
        eor     v0.8b,v0.8b,v12.8b
        ldr     q5,[x1],16
        eor     v1.8b,v1.8b,v12.8b
        sdot    v16.4s,v4.16b,v0.4b[0]
        sdot    v17.4s,v4.16b,v1.4b[0]
        eor     v2.8b,v2.8b,v12.8b
        ldp     q6,q7,[x1],32
        eor     v3.8b,v3.8b,v12.8b
        sdot    v18.4s,v4.16b,v2.4b[0]
        sdot    v19.4s,v4.16b,v3.4b[0]
        sdot    v20.4s,v5.16b,v0.4b[0]
        sdot    v21.4s,v5.16b,v1.4b[0]
        sdot    v22.4s,v5.16b,v2.4b[0]
        sdot    v23.4s,v5.16b,v3.4b[0]
        sdot    v24.4s,v6.16b,v0.4b[0]
        sdot    v25.4s,v6.16b,v1.4b[0]
        sdot    v26.4s,v6.16b,v2.4b[0]
        sdot    v27.4s,v6.16b,v3.4b[0]
        sdot    v28.4s,v7.16b,v0.4b[0]
        sdot    v29.4s,v7.16b,v1.4b[0]
        sdot    v30.4s,v7.16b,v2.4b[0]
        sdot    v31.4s,v7.16b,v3.4b[0]

SkipInCh4
        subs    x9,x9,8             // ks -= 1
        b.hi    KernelSizeLoop
        b       Requantize

PartialStore
        tbz     x6,3,LT8Store
        str     d3,[x5],8           // no less than 8 channels
        str     d2,[x17],8
        dup     d3,v3.d[1]
        dup     d2,v2.d[1]
        str     d1,[x16],8
        str     d0,[x2],8
        dup     d1,v1.d[1]
        dup     d0,v0.d[1]
LT8Store
        tbz     x6,2,LT4Store
        str     s3,[x5],4
        str     s2,[x17],4
        dup     s3,v3.s[1]
        dup     s2,v2.s[1]
        str     s1,[x16],4
        str     s0,[x2],4
        dup     s1,v1.s[1]
        dup     s0,v0.s[1]
LT4Store
        tbz     x6,1, LT2Store
        str     h3,[x5],2
        str     h2,[x17],2
        dup     h3,v3.h[1]
        dup     h2,v2.h[1]
        str     h1,[x16],2
        str     h0,[x2],2
        dup     h1,v1.h[1]
        dup     h0,v0.h[1]
LT2Store
        tbz     x6,0,ExitKernel
        str     b3,[x5]
        str     b2,[x17]
        str     b1,[x16]
        str     b0,[x2]
        b       ExitKernel

        NESTED_END MlasConvSymU8KernelDot

        END
