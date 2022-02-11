/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    ConvSymU8KernelNeon.S

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
        .equ    .LConvSymFrame_SavedNeonRegisters, (8 * 8)
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
        FUNCTION_ENTRY MlasConvSymU8KernelNeon

        stp     d8,d9,[sp,#-64]!
        ldr     x8,[sp,#.LConvSymFrame_PostProcessParams]
        ldrb    w10,[sp,#.LConvSymFrame_KernelFlags]
        stp     d10,d11,[sp,#16]
        stp     d12,d13,[sp,#32]
        stp     d14,d15,[sp,#48]
        mov     x9,x3                   // save kernel size
        ldr     x11,[x8,#.LConvSymPostProcessParams_Bias]
        mov     x16,x4                  // save input channels
        ldr     x12,[x8,#.LConvSymPostProcessParams_Scale]
        cmp     x7,2                    // if OutputCount < 2
        add     x5,x2,x5                // c1 = c0 + ldc
        add     x4,x4,7                 // kc = (kc + 7) & ~7
        csel    x5,x2,x5,lo             // if OutputCount < 2  c1 = c0
        bic     x4,x4,7
        ldp     s16,s18,[x11],8         // init accumulators with bias
        ldp     s20,s22,[x11],8
        ldp     s24,s26,[x11],8
        ldp     s28,s30,[x11],8
        mov     v17.16b,v16.16b
        mov     v19.16b,v18.16b
        mov     v21.16b,v20.16b
        mov     v23.16b,v22.16b
        mov     v25.16b,v24.16b
        mov     v27.16b,v26.16b
        mov     v29.16b,v28.16b
        mov     v31.16b,v30.16b

// Nested loops, inner loop: input channel; outter loop: kernel size
// Each inner iteration processes 8 input channels, 2 output pixels, 8 output channels.
//
//                                            B 8x8
//                           ------------------------------------------------------------------
//                           |v4.b[0] v5.b[0] v4.b[0] v5.b[0] v4.b[0] v5.b[0] v4.b[0] v5.b[0] |
//                           |  ...    ...     ...     ...       ...    ...     ...     ...   |
//                           |v4.b[7] v5.b[7] v4.b[7] v5.b[7] v4.b[7] v5.b[7] v4.b[7] v5.b[7] |
//            A 2x8          ------------------------------------------------------------------
//       ------------------  ------------------------------------------------------------------
// x13-> |v0.b[0]..v0.b[7]|  |v16.4s   v18.4s  v20.4s  v22.4s  v24.4s  v26.4s  v28.4s  v30.4s |
// x15-> |v1.b[0]..v1.b[7]|  |v17.4s   v19.4s  v21.4s  v23.4s  v25.4s  v27.4s  v29.4s  v31.4s |
//       ------------------  ------------------------------------------------------------------
// When Input Channels greater than 16, unroll:
// A registers v6 v7,
// B registers v8 v9
//

.LConvSym.KernelSizeLoop:

        # Load next 2 A pointers
        tst     w10,#.LMLAS_CONV_SYM_FLAG_INPUT_DIRECT
        ldr     d4,[x1]
        ldr     d5,[x1,8]
        beq     .LConvSym.InputIndirection

.LConvSym.InputDirect:
        mov     x13,x0                  // x13 -> A0
        add     x15,x0,x16              // x15 -> A1 = A0 + input channels
        b       .LConvSym.BlockLoopPrologue

.LConvSym.InputIndirection:
        cmp     x7,2                    // test if OutputCount < 2
        ldr     x13,[x0]                // x13 -> A0
        blo     .LConvSym.SkipLoadA1
        ldr     x15,[x0,x3,lsl#3]       // x15 -> A1
.LConvSym.SkipLoadA1:

.LConvSym.BlockLoopPrologue:
        cmp     x7,2                    // test if OutputCount < 2
        add     x0,x0,8                 // indirect A advance to next pointer, prepare for kernel size loop
        csel    x15,x13,x15,lo          // if OutputCount < 2  x15 -> A0
        subs    x14,x4,16               // input channel - 16
        movi    v12.8b,128
        blo     .LConvSym.8InputChannels     // less than 16 deep, no unroll

        ldr     d0,[x13],8
        ldr     d1,[x15],8
        ldr     d8,[x1,64]
        ldr     d9,[x1,72]
        ldr     d6,[x13],8
        subs    x14,x14,16              // input channel - 16
        ldr     d7,[x15],8
        blo     .LConvSym.BlockLoopEpilogue  // need 32 input channel for full unrolled loop

.LConvSym.Blockloop:
        eor     v0.8b,v0.8b,v12.8b
        eor     v1.8b,v1.8b,v12.8b
        smull   v2.8h,v4.8b,v0.8b
        smull   v3.8h,v4.8b,v1.8b
        ldr     d4,[x1,16]
        smull   v10.8h,v5.8b,v0.8b
        smull   v11.8h,v5.8b,v1.8b
        ldr     d5,[x1,24]
        eor     v6.8b,v6.8b,v12.8b
        eor     v7.8b,v7.8b,v12.8b
        smlal   v2.8h,v8.8b,v6.8b
        smlal   v3.8h,v8.8b,v7.8b
        ldr     d8,[x1,80]
        smlal   v10.8h,v9.8b,v6.8b
        smlal   v11.8h,v9.8b,v7.8b
        ldr     d9,[x1,88]
        smull   v12.8h,v4.8b,v0.8b
        sadalp  v16.4s,v2.8h
        smull   v13.8h,v4.8b,v1.8b
        ldr     d4,[x1,32]
        sadalp  v17.4s,v3.8h
        smull   v14.8h,v5.8b,v0.8b
        sadalp  v18.4s,v10.8h
        smull   v15.8h,v5.8b,v1.8b
        ldr     d5,[x1,40]
        sadalp  v19.4s,v11.8h
        smlal   v12.8h,v8.8b,v6.8b
        smlal   v13.8h,v8.8b,v7.8b
        ldr     d8,[x1,96]
        smlal   v14.8h,v9.8b,v6.8b
        smlal   v15.8h,v9.8b,v7.8b
        ldr     d9,[x1,104]
        smull   v2.8h,v4.8b,v0.8b
        sadalp  v20.4s,v12.8h
        smull   v3.8h,v4.8b,v1.8b
        ldr     d4,[x1,48]
        sadalp  v21.4s,v13.8h
        smull   v10.8h,v5.8b,v0.8b
        sadalp  v22.4s,v14.8h
        smull   v11.8h,v5.8b,v1.8b
        ldr     d5,[x1,56]
        sadalp  v23.4s, v15.8h
        smlal   v2.8h,v8.8b,v6.8b
        smlal   v3.8h,v8.8b,v7.8b
        ldr     d8,[x1,112]
        smlal   v10.8h,v9.8b,v6.8b
        smlal   v11.8h,v9.8b,v7.8b
        ldr     d9,[x1,120]
        smull   v12.8h,v4.8b,v0.8b
        add     x1,x1,128
        sadalp  v24.4s,v2.8h
        smull   v13.8h,v4.8b,v1.8b
        ldr     d4,[x1]                 // Read B
        sadalp  v25.4s,v3.8h
        smull   v14.8h,v5.8b,v0.8b
        ldr     d0,[x13],8              // Read A0
        sadalp  v26.4s,v10.8h
        smull   v15.8h,v5.8b,v1.8b
        ldr     d1,[x15],8              // Read A1
        sadalp  v27.4s,v11.8h
        smlal   v12.8h,v8.8b,v6.8b
        ldr     d5,[x1,8]               // Read B
        smlal   v13.8h,v8.8b,v7.8b
        ldr     d8,[x1,64]              // Read B
        smlal   v14.8h,v9.8b,v6.8b
        ldr     d6,[x13],8              // Read A0
        smlal   v15.8h,v9.8b,v7.8b
        ldr     d7,[x15],8              // Read A1
        sadalp  v28.4s,v12.8h
        ldr     d9,[x1,72]              // Read B
        sadalp  v29.4s,v13.8h
        subs    x14,x14,16
        sadalp  v30.4s,v14.8h
        movi    v12.8b,128
        sadalp  v31.4s,v15.8h
        b.hs    .LConvSym.Blockloop

.LConvSym.BlockLoopEpilogue:            // remaining 16 input channels
        eor     v0.8b,v0.8b,v12.8b
        eor     v1.8b,v1.8b,v12.8b
        smull   v2.8h,v4.8b,v0.8b
        smull   v3.8h,v4.8b,v1.8b
        ldr     d4,[x1,16]
        smull   v10.8h,v5.8b,v0.8b
        smull   v11.8h,v5.8b,v1.8b
        ldr     d5,[x1,24]
        eor     v6.8b,v6.8b,v12.8b
        eor     v7.8b,v7.8b,v12.8b
        smlal   v2.8h,v8.8b,v6.8b
        smlal   v3.8h,v8.8b,v7.8b
        ldr     d8,[x1,80]
        smlal   v10.8h,v9.8b,v6.8b
        smlal   v11.8h,v9.8b,v7.8b
        ldr     d9,[x1,88]
        smull   v12.8h,v4.8b,v0.8b
        sadalp  v16.4s,v2.8h
        smull   v13.8h,v4.8b,v1.8b
        ldr     d4,[x1,32]
        sadalp  v17.4s,v3.8h
        smull   v14.8h,v5.8b,v0.8b
        sadalp  v18.4s,v10.8h
        smull   v15.8h,v5.8b,v1.8b
        sadalp  v19.4s,v11.8h
        ldr     d5,[x1,40]
        smlal   v12.8h,v8.8b,v6.8b
        smlal   v13.8h,v8.8b,v7.8b
        ldr     d8,[x1,96]
        smlal   v14.8h,v9.8b,v6.8b
        smlal   v15.8h,v9.8b,v7.8b
        ldr     d9,[x1,104]
        smull   v2.8h,v4.8b,v0.8b
        sadalp  v20.4s,v12.8h
        smull   v3.8h,v4.8b,v1.8b
        ldr     d4,[x1,48]
        sadalp  v21.4s,v13.8h
        smull   v10.8h,v5.8b,v0.8b
        sadalp  v22.4s,v14.8h
        smull   v11.8h,v5.8b,v1.8b
        sadalp  v23.4s,v15.8h
        ldr     d5,[x1,56]
        smlal   v2.8h,v8.8b,v6.8b
        smlal   v3.8h,v8.8b,v7.8b
        ldr     d8,[x1,112]
        smlal   v10.8h,v9.8b,v6.8b
        smlal   v11.8h,v9.8b,v7.8b
        ldr     d9,[x1,120]
        smull   v12.8h,v4.8b,v0.8b
        sadalp  v24.4s,v2.8h
        smull   v13.8h,v4.8b,v1.8b
        sadalp  v25.4s,v3.8h
        smull   v14.8h,v5.8b,v0.8b
        sadalp  v26.4s,v10.8h
        smull   v15.8h,v5.8b,v1.8b
        sadalp  v27.4s,v11.8h
        smlal   v12.8h,v8.8b,v6.8b
        smlal   v13.8h,v8.8b,v7.8b
        smlal   v14.8h,v9.8b,v6.8b
        smlal   v15.8h,v9.8b,v7.8b
        add     x1,x1,128

        sadalp  v28.4s,v12.8h
        sadalp  v29.4s,v13.8h
        sadalp  v30.4s,v14.8h
        sadalp  v31.4s,v15.8h
        movi    v12.8b,128
        tbnz    x14,3,.LConvSym.8InputChannels

        subs    x9,x9,1
        b.hi    .LConvSym.KernelSizeLoop

.LConvSym.Requantize:
        ldr     w11, [x8, #.LConvSymPostProcessParams_ZeroPoint]
        tst     w10,#.LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        beq     .LConvSym.BroadcastScaleValue
        ld1     {v4.4s,v5.4s},[x12]     // load scale vector
        b       .LConvSym.AccumulatorsToFloat

.LConvSym.BroadcastScaleValue:
        ld1r    {v4.4s},[x12]           // load scale Value
        mov     v5.16b, v4.16b

.LConvSym.AccumulatorsToFloat:
        addp    v16.4s,v16.4s,v18.4s
        addp    v20.4s,v20.4s,v22.4s
        addp    v24.4s,v24.4s,v26.4s
        addp    v28.4s,v28.4s,v30.4s
        addp    v17.4s,v17.4s,v19.4s
        addp    v21.4s,v21.4s,v23.4s
        addp    v25.4s,v25.4s,v27.4s
        addp    v29.4s,v29.4s,v31.4s
        addp    v0.4s,v16.4s,v20.4s
        addp    v1.4s,v24.4s,v28.4s
        addp    v2.4s,v17.4s,v21.4s
        addp    v3.4s,v25.4s,v29.4s
        scvtf   v0.4s,v0.4s             // convert to float
        scvtf   v1.4s,v1.4s
        scvtf   v2.4s,v2.4s
        scvtf   v3.4s,v3.4s
        fmul    v0.4s,v0.4s,v4.4s       // multiply by scale
        fmul    v1.4s,v1.4s,v5.4s
        fmul    v2.4s,v2.4s,v4.4s
        fmul    v3.4s,v3.4s,v5.4s
        fcvtns  v0.4s,v0.4s             // convert to int
        fcvtns  v1.4s,v1.4s
        dup     v9.8h,w11
        fcvtns  v2.4s,v2.4s
        fcvtns  v3.4s,v3.4s
        sqxtn   v0.4h,v0.4s
        sqxtn2  v0.8h,v1.4s
        sqxtn   v2.4h,v2.4s
        sqxtn2  v2.8h,v3.4s
        subs    x6, x6, 8
        sqadd   v0.8h,v0.8h,v9.8h
        sqadd   v2.8h,v2.8h,v9.8h
        sqxtun  v0.8b,v0.8h             // shorten to int8
        sqxtun2 v0.16b,v2.8h
        b.lo    .LConvSym.PartialStore

        st1     {v0.d}[1],[x5]          // full 2x8 store to c 
        st1     {v0.8b},[x2]

.LConvSym.ExitKernel:
        ldp     d14,d15,[sp,#48]
        ldp     d12,d13,[sp,#32]
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#64
        ret

.LConvSym.8InputChannels:
        ldr     d0,[x13]
        ldr     d1,[x15]
        ldr     d4,[x1]
        ldr     d5,[x1,8]
        ldr     d6,[x1,16]
        ldr     d7,[x1,24]
        eor     v0.8b,v0.8b,v12.8b
        eor     v1.8b,v1.8b,v12.8b
        smull   v2.8h,v4.8b,v0.8b
        smull   v3.8h,v4.8b,v1.8b
        ldr     d4,[x1,32]
        smull   v10.8h,v5.8b,v0.8b
        smull   v11.8h,v5.8b,v1.8b
        ldr     d5,[x1,40]
        smull   v12.8h,v6.8b,v0.8b
        sadalp  v16.4s,v2.8h
        smull   v13.8h,v6.8b,v1.8b
        ldr     d6,[x1,48]
        sadalp  v17.4s,v3.8h
        smull   v14.8h,v7.8b,v0.8b
        sadalp  v18.4s,v10.8h
        smull   v15.8h,v7.8b,v1.8b
        ldr     d7,[x1,56]
        sadalp  v19.4s,v11.8h
        smull   v2.8h,v4.8b,v0.8b
        sadalp  v20.4s,v12.8h
        smull   v3.8h,v4.8b,v1.8b
        sadalp  v21.4s,v13.8h
        smull   v10.8h,v5.8b,v0.8b
        sadalp  v22.4s,v14.8h
        smull   v11.8h,v5.8b,v1.8b
        sadalp  v23.4s,v15.8h
        smull   v12.8h,v6.8b,v0.8b
        sadalp  v24.4s,v2.8h
        smull   v13.8h,v6.8b,v1.8b
        sadalp  v25.4s,v3.8h
        smull   v14.8h,v7.8b,v0.8b
        sadalp  v26.4s,v10.8h
        smull   v15.8h,v7.8b,v1.8b
        sadalp  v27.4s,v11.8h
        add     x1,x1,64
        sadalp  v28.4s,v12.8h
        sadalp  v29.4s,v13.8h
        sadalp  v30.4s,v14.8h
        sadalp  v31.4s,v15.8h

        # ks loop
        subs    x9,x9,1
        b.hi    .LConvSym.KernelSizeLoop
        b       .LConvSym.Requantize

.LConvSym.PartialStore:
        tbz     x6,2,.LConvSym.Store2
        st1     {v0.s}[2],[x5],4
        str     s0,[x2],4
        EXT     v0.16b,v0.16b,v0.16b,4

.LConvSym.Store2:
        tbz     x6, 1, .LConvSym.Store1
        st1     {v0.h}[4], [x5], 2
        str     h0, [x2], 2
        EXT     v0.16b,v0.16b,v0.16b,2
.LConvSym.Store1:
        tbz     x6,0,.LConvSym.ExitKernel
        st1     {v0.b}[8],[x5]
        str     b0,[x2]
        b       .LConvSym.ExitKernel

        .end
