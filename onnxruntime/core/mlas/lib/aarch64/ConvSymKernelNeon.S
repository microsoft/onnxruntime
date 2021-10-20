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
        FUNCTION_ENTRY MlasConvSymKernelNeon

        stp     d8,d9,[sp,#-64]!
        ldr     x8,[sp,#.LConvSymFrame_PostProcessParams]
        ldrb    w10,[sp,#.LConvSymFrame_KernelFlags]
        stp     d10,d11,[sp,#16]
        stp     d12,d13,[sp,#32]
        stp     d14,d15,[sp,#48]

        mov     x9, x3                  // save kernel size
        mov     x16,x4                  // save input channels
        ldr     x11,[x8], #8 // Bias
        ldr     x12,[x8], #8 // Scale
        cmp     x7, 2                   // if OutputCount < 2
        add     x5, x2, x5              // c1 = c0 + ldc
        add     x4, x4, 7               // kc = (kc + 7) & ~7
        csel    x5, x2, x5, lo          // if OutputCount < 2  c1 = c0
        bic     x4, x4, 7

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
        mov     w17, #128

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
        beq     .LConvSym.InputIndirection
.LConvSym.InputDirect:
        cmp     x7,2                    // test if OutputCount < 2
        mov     x13,x0                  // x13 -> A0
        add     x15,x0,x16              // x15 -> A1 = A0 + input channels
        csel    x15, x13, x15, LO       // if OutputCount < 2  x15 -> A0
        b       .LConvSym.BlockloopPrologue
.LConvSym.InputIndirection:
        cmp     x7,2                    // test if OutputCount < 2
        ldr     x13,[x0]                // x13 -> A0
        mov     x15,x13                 // x15 -> A0
        blo     .LConvSym.SkipLoadA1
        ldr     x15,[x0,x3,lsl#3]       // x15 -> A1
.LConvSym.SkipLoadA1:
        add     x0,x0,8                 // indirect A advance to next pointer, prepare for kernel size loop

.LConvSym.BlockloopPrologue:

        subs     x14,x4,16               // input channel - 16
        blo     .LConvSym.8InputChannels     // less than 16 deep, no unroll

        ldp     d4,d5,[x1]
        dup     v2.8b, w17
        ldp     d0,d6,[x13],16
        eor     v0.8b, v0.8b, v2.8b
        eor     v6.8b, v6.8b, v2.8b
        ldp     d1,d7,[x15],16
        eor     v1.8b, v1.8b, v2.8b
        eor     v7.8b, v7.8b, v2.8b
        ldp     d8,d9,[x1,64]

        subs    x14,x14,16              // input channel - 16
        blo     .LConvSym.BlockLoopEpilogue  // need 32 input channel for full unrolled loop

.LConvSym.Blockloop:
        SMULL   v2.8h, v4.8b, v0.8b
        SMULL   v3.8h, v4.8b, v1.8b
        PRFM    PLDL1KEEP, [x1, 448]
        SMULL   v10.8h, v5.8b, v0.8b
        SMULL   v11.8h, v5.8b, v1.8b
        LDP     d4, d5, [x1, 16]
        SMLAL   v2.8h, v8.8b, v6.8b
        SMLAL   v3.8h, v8.8b, v7.8b
        PRFM    PLDL1KEEP, [x1, 512]
        SMLAL   v10.8h, v9.8b, v6.8b
        SMLAL   v11.8h, v9.8b, v7.8b

        LDP     d8, d9, [x1, 80]
        SMULL   v12.8h, v4.8b, v0.8b
        SADALP  v16.4s,  v2.8h
        SMULL   v13.8h, v4.8b, v1.8b
        SADALP  v17.4s,  v3.8h
        SMULL   v14.8h, v5.8b, v0.8b
        SADALP  v18.4s, v10.8h
        SMULL   v15.8h, v5.8b, v1.8b
        SADALP  v19.4s, v11.8h
        LDP     d4, d5, [x1, 32]
        SMLAL   v12.8h, v8.8b, v6.8b
        SMLAL   v13.8h, v8.8b, v7.8b
        PRFM    PLDL1KEEP, [x13, 128]
        SMLAL   v14.8h, v9.8b, v6.8b
        SMLAL   v15.8h, v9.8b, v7.8b

        LDP     d8, d9, [x1, 96]
        SMULL   v2.8h, v4.8b, v0.8b
        SADALP  v20.4s, v12.8h
        SMULL   v3.8h, v4.8b, v1.8b
        SADALP  v21.4s, v13.8h
        SMULL   v10.8h, v5.8b, v0.8b
        SADALP  v22.4s, v14.8h
        SMULL   v11.8h, v5.8b, v1.8b
        SADALP  v23.4s, v15.8h
        LDP     d4, d5, [x1, 48]
        SMLAL   v2.8h, v8.8b, v6.8b
        SMLAL   v3.8h, v8.8b, v7.8b
        PRFM    PLDL1KEEP, [x15, 128]
        SMLAL   v10.8h, v9.8b, v6.8b
        SMLAL   v11.8h, v9.8b, v7.8b

        LDP     d8, d9, [x1, 112]
        SMULL   v12.8h, v4.8b, v0.8b
        ADD     x1, x1, 128
        SADALP  v24.4s,  v2.8h
        SMULL   v13.8h, v4.8b, v1.8b
        SADALP  v25.4s,  v3.8h
        SMULL   v14.8h, v5.8b, v0.8b
        SADALP  v26.4s, v10.8h
        SMULL   v15.8h, v5.8b, v1.8b
        SADALP  v27.4s, v11.8h
        SMLAL   v12.8h, v8.8b, v6.8b
        LDP     d4, d5, [x1]            // Read B
        SMLAL   v13.8h, v8.8b, v7.8b
        SUBS    x14, x14, 16
        SMLAL   v14.8h, v9.8b, v6.8b
        LDP     d0, d6, [x13], 16       // Read A0
        dup     v2.8b, w17
        eor     v0.8b, v0.8b, v2.8b
        eor     v6.8b, v6.8b, v2.8b
        SMLAL   v15.8h, v9.8b, v7.8b

        SADALP  v28.4s, v12.8h
        LDP     d1, d7, [x15], 16       // Read A1
        eor     v1.8b, v1.8b, v2.8b
        eor     v7.8b, v7.8b, v2.8b
        SADALP  v29.4s, v13.8h
        SADALP  v30.4s, v14.8h
        LDP     d8, d9, [x1, 64]        // Read B
        SADALP  v31.4s, v15.8h
        B.HS    .LConvSym.Blockloop

.LConvSym.BlockLoopEpilogue:            // remaining 16 input channels
        SMULL   v2.8h, v4.8b, v0.8b
        SMULL   v3.8h, v4.8b, v1.8b
        SMULL   v10.8h, v5.8b, v0.8b
        SMULL   v11.8h, v5.8b, v1.8b
        LDP     d4, d5, [x1, 16]
        SMLAL   v2.8h, v8.8b, v6.8b
        SMLAL   v3.8h, v8.8b, v7.8b
        SMLAL   v10.8h, v9.8b, v6.8b
        SMLAL   v11.8h, v9.8b, v7.8b

        LDP     d8, d9, [x1, 80]
        SMULL   v12.8h, v4.8b, v0.8b
        SADALP  v16.4s,  v2.8h
        SMULL   v13.8h, v4.8b, v1.8b
        SADALP  v17.4s,  v3.8h
        SMULL   v14.8h, v5.8b, v0.8b
        SADALP  v18.4s, v10.8h
        SMULL   v15.8h, v5.8b, v1.8b
        SADALP  v19.4s, v11.8h
        LDP     d4, d5, [x1, 32]
        SMLAL   v12.8h, v8.8b, v6.8b
        SMLAL   v13.8h, v8.8b, v7.8b
        SMLAL   v14.8h, v9.8b, v6.8b
        SMLAL   v15.8h, v9.8b, v7.8b

        LDP     d8, d9, [x1, 96]
        SMULL   v2.8h, v4.8b, v0.8b
        SADALP  v20.4s, v12.8h
        SMULL   v3.8h, v4.8b, v1.8b
        SADALP  v21.4s, v13.8h
        SMULL   v10.8h, v5.8b, v0.8b
        SADALP  v22.4s, v14.8h
        SMULL   v11.8h, v5.8b, v1.8b
        SADALP  v23.4s, v15.8h
        LDP     d4, d5, [x1, 48]
        SMLAL   v2.8h, v8.8b, v6.8b
        SMLAL   v3.8h, v8.8b, v7.8b
        SMLAL   v10.8h, v9.8b, v6.8b
        SMLAL   v11.8h, v9.8b, v7.8b

        LDP     d8, d9, [x1, 112]
        SMULL   v12.8h, v4.8b, v0.8b
        SADALP  v24.4s,  v2.8h
        SMULL   v13.8h, v4.8b, v1.8b
        SADALP  v25.4s,  v3.8h
        SMULL   v14.8h, v5.8b, v0.8b
        SADALP  v26.4s, v10.8h
        SMULL   v15.8h, v5.8b, v1.8b
        SADALP  v27.4s, v11.8h
        SMLAL   v12.8h, v8.8b, v6.8b
        SMLAL   v13.8h, v8.8b, v7.8b
        SMLAL   v14.8h, v9.8b, v6.8b
        SMLAL   v15.8h, v9.8b, v7.8b
        ADD     x1, x1, 128

        SADALP  v28.4s, v12.8h
        SADALP  v29.4s, v13.8h
        SADALP  v30.4s, v14.8h
        SADALP  v31.4s, v15.8h
        TBNZ    x14, 3, .LConvSym.8InputChannels

        SUBS    x9, x9, 1
        B.HI    .LConvSym.KernelSizeLoop

.LConvSym.Requantize:
        LD1R    {v10.4s},[x8], #4 // #.LConvSymPostProcessParams_Min]
        LD1R    {v6.4s},[x8], #4 // #.LConvSymPostProcessParams_Max]
        LD1R    {v9.4s},[x8], #4 // #.LConvSymPostProcessParams_ZeroPoint]
        tst     w10,#.LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE
        beq     .LConvSym.BroadcastScaleValue
        LD1     {v4.4s,v5.4s}, [x12]   // load scale vector
        b       .LConvSym.AccumulatorsToFloat

.LConvSym.BroadcastScaleValue:
        LD1R   {v4.4s},[x12]                // load scale Value
        mov    v5.16b, v4.16b

.LConvSym.AccumulatorsToFloat:
        ADDP    v16.4s,v16.4s,v18.4s
        ADDP    v20.4s,v20.4s,v22.4s
        ADDP    v24.4s,v24.4s,v26.4s
        ADDP    v28.4s,v28.4s,v30.4s
        ADDP    v17.4s,v17.4s,v19.4s
        ADDP    v21.4s,v21.4s,v23.4s
        ADDP    v25.4s,v25.4s,v27.4s
        ADDP    v29.4s,v29.4s,v31.4s
        ADDP    v0.4s,v16.4s,v20.4s
        ADDP    v1.4s,v24.4s,v28.4s
        ADDP    v2.4s,v17.4s,v21.4s
        ADDP    v3.4s,v25.4s,v29.4s
        scvtf    v0.4s,v0.4s         // convert to float
        scvtf    v1.4s,v1.4s
        scvtf    v2.4s,v2.4s
        scvtf    v3.4s,v3.4s
        FMUL v0.4s,v0.4s,v4.4s          // multiply by scale
        FMUL v1.4s,v1.4s,v5.4s
        FMUL v2.4s,v2.4s,v4.4s
        FMUL v3.4s,v3.4s,v5.4s
        FMIN v0.4s,v0.4s,v6.4s         // clamp max value
        FMIN v1.4s,v1.4s,v6.4s
        FMIN v2.4s,v2.4s,v6.4s
        FMIN v3.4s,v3.4s,v6.4s
        FMAX v0.4s,v0.4s,v10.4s        // clamp min value
        FMAX v1.4s,v1.4s,v10.4s
        FMAX v2.4s,v2.4s,v10.4s
        FMAX v3.4s,v3.4s,v10.4s
        SUBS    x6, x6, 8
        fcvtns   v0.4s,v0.4s          // convert to int
        fcvtns   v1.4s,v1.4s
        fcvtns   v2.4s,v2.4s
        fcvtns   v3.4s,v3.4s
        add     v0.4s,v0.4s,v9.4s           // add zero point
        add     v1.4s,v1.4s,v9.4s
        add     v2.4s,v2.4s,v9.4s
        add     v3.4s,v3.4s,v9.4s
        UQXTN   v0.4h,v0.4s                 // shorten to int16
        UQXTN   v2.4h,v2.4s
        UQXTN2  v0.8h,v1.4s
        UQXTN2  v2.8h,v3.4s
        UQXTN   v0.8b,v0.8h                 // shorten to int8
        UQXTN2  v0.16b,v2.8h
        B.LO    .LConvSym.PartialStore

        ST1     {v0.d}[1],[x5]              // full 2x8 store to c 
        ST1     {v0.8b}, [x2]

.LConvSym.ExitKernel:
        ldp     d14,d15,[sp,#48]
        ldp     d12,d13,[sp,#32]
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#64
        ret

.LConvSym.8InputChannels:
        LDR     d0, [x13]
        dup     v2.8b, w17
        eor     v0.8b, v0.8b, v2.8b
        LDP     d4, d5, [x1]
        LDR     d1, [x15]
        eor     v1.8b, v1.8b, v2.8b
        LDP     d6, d7, [x1, 16]
        SMULL   v2.8h, v4.8b, v0.8b
        SMULL   v3.8h, v4.8b, v1.8b
        SMULL   v10.8h, v5.8b, v0.8b
        SMULL   v11.8h, v5.8b, v1.8b
        SMULL   v12.8h, v6.8b, v0.8b
        SADALP  v16.4s,  v2.8h
        SMULL   v13.8h, v6.8b, v1.8b
        SADALP  v17.4s,  v3.8h
        SMULL   v14.8h, v7.8b, v0.8b
        SADALP  v18.4s, v10.8h
        SMULL   v15.8h, v7.8b, v1.8b
        SADALP  v19.4s, v11.8h
        LDP     d4, d5, [x1, 32]
        SMULL   v2.8h, v4.8b, v0.8b
        SADALP  v20.4s, v12.8h
        SMULL   v3.8h, v4.8b, v1.8b
        SADALP  v21.4s, v13.8h
        SMULL   v10.8h, v5.8b, v0.8b
        SADALP  v22.4s, v14.8h
        SMULL   v11.8h, v5.8b, v1.8b
        SADALP  v23.4s, v15.8h
        LDP     d6, d7, [x1, 48]
        SMULL   v12.8h, v6.8b, v0.8b
        SADALP  v24.4s,  v2.8h
        SMULL   v13.8h, v6.8b, v1.8b
        SADALP  v25.4s,  v3.8h
        SMULL   v14.8h, v7.8b, v0.8b
        SADALP  v26.4s, v10.8h
        SMULL   v15.8h, v7.8b, v1.8b
        SADALP  v27.4s, v11.8h
        ADD     x1, x1, 64
        SADALP  v28.4s, v12.8h
        SADALP  v29.4s, v13.8h
        SADALP  v30.4s, v14.8h
        SADALP  v31.4s, v15.8h

        # ks loop
        SUBS    x9, x9, 1
        B.HI    .LConvSym.KernelSizeLoop
        B       .LConvSym.Requantize

.LConvSym.PartialStore:
        TBZ     x6, 2, 7f
        ST1     {v0.s}[2], [x5], 4
        STR     s0, [x2], 4
        EXT     v0.16b, v0.16b, v0.16b, 4

7:
        TBZ     x6, 1, 8f
        ST1     {v0.h}[4], [x5], 2
        STR     h0, [x2], 2
        EXT     v0.16b, v0.16b, v0.16b, 2
8:
        TBZ     x6, 0, .LConvSym.ExitKernel
        ST1     {v0.b}[8], [x5]
        STR     b0, [x2]
        b       .LConvSym.ExitKernel

        .end
