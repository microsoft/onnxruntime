/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    ConvSymKernelNeon.asm

Abstract:

    This module implements the kernels for the symmetric quantized integer
    convolution operation.

--*/

#include "kxarm64.h"

//
// Stack frame layout for the symmetric convolution kernel.
// d8-d15, x19-x30 need to be preserved if used
//

#define ConvSymFrame_SavedNeonRegisters, (8 * 8)
#define ConvSymFrame_SavedRegisters, ConvSymFrame_SavedNeonRegisters
#define ConvSymFrame_PostProcessParams, 0 + ConvSymFrame_SavedRegisters
#define ConvSymFrame_KernelFlags, 8 + ConvSymFrame_SavedRegisters

#define ConvSymPostProcessParams_Bias,      0
#define ConvSymPostProcessParams_Scale,     8
#define ConvSymPostProcessParams_Min,       16
#define ConvSymPostProcessParams_Max,       20
#define ConvSymPostProcessParams_ZeroPoint, 24

        TEXTAREA

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
        NESTED_ENTRY MlasConvSymKernelNeon

        PROLOG_SAVE_REG_PAIR     d8,d9,#-64!
        PROLOG_SAVE_REG_PAIR d10,d11,#16
        PROLOG_SAVE_REG_PAIR d12,d13,#32
        PROLOG_SAVE_REG_PAIR d14,d15,#48
        ldr     x8,[sp,#ConvSymFrame_PostProcessParams]
        ldrb    w10,[sp,#ConvSymFrame_KernelFlags]

        mov     x9, x3                  // save kernel size
        mov     x16,x4                  // save input channels
        ldr     x11,[x8,#ConvSymPostProcessParams_Bias]
        ldr     x12,[x8,#ConvSymPostProcessParams_Scale]
        cmp     x7, 2                   // if OutputCount < 2
        add     x5, x2, x5              // c1 = c0 + ldc
        add     x4, x4, 7               // kc = (kc + 7) & ~7
        csel    x5, x2, x5, lo          // if OutputCount < 2  c1 = c0
        bic     x4, x4, 7

        LDP     s16,s18,[x11],8         // init accumulators with bias
        LDP     s20,s22,[x11],8
        LDP     s24,s26,[x11],8
        LDP     s28,s30,[x11],8
        MOV     v17.16b,v16.16b
        MOV     v19.16b,v18.16b
        MOV     v21.16b,v20.16b
        MOV     v23.16b,v22.16b
        MOV     v25.16b,v24.16b
        MOV     v27.16b,v26.16b
        MOV     v29.16b,v28.16b
        MOV     v31.16b,v30.16b

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

ConvSym.KernelSizeLoop:

        # Load next 2 A pointers
        tst     w10,MLAS_CONV_SYM_FLAG_INPUT_DIRECT
        beq     ConvSym.InputIndirection
ConvSym.InputDirect:
        cmp     x7,2                    // test if OutputCount < 2
        mov     x13,x0                  // x13 -> A0
        add     x15,x0,x16              // x15 -> A1 = A0 + input channels
        csel    x15, x13, x15, LO       // if OutputCount < 2  x15 -> A0
        b       ConvSym.BlockloopPrologue
ConvSym.InputIndirection:
        cmp     x7,2                    // test if OutputCount < 2
        ldr     x13,[x0]                // x13 -> A0
        mov     x15,x13                 // x13 -> A0
        blo     ConvSym.SkipLoadA1
        ldr     x15,[x0,x3,lsl#3]       // x13 -> A1
ConvSym.SkipLoadA1:
        add     x0,x0,8                 // indirect A advance to next pointer, prepare for kernel size loop

ConvSym.BlockloopPrologue:

        sub     x14,x4,16               // input channel - 16
        blo     ConvSym.8InputChannels     // less than 16 deep, no unroll

        ldp     d4,d5,[x1]
        ldp     d0,d6,[x13],16
        ldp     d1,d7,[x15],16
        ldp     d8,d9,[x1,64]

        subs    x14,x14,16              // input channel - 16
        blo     ConvSym.BlockLoopEpilogue  // need 32 input channel for full unrolled loop

ConvSym.Blockloop:
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
        SMLAL   v15.8h, v9.8b, v7.8b

        SADALP  v28.4s, v12.8h
        LDP     d1, d7, [x15], 16       // Read A1
        SADALP  v29.4s, v13.8h
        SADALP  v30.4s, v14.8h
        LDP     d8, d9, [x1, 64]        // Read B
        SADALP  v31.4s, v15.8h
        B.HS    ConvSym.Blockloop

ConvSym.BlockLoopEpilogue:            // remaining 16 input channels
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
        TBNZ    x14, 3, ConvSym.8InputChannels

        SUBS    x9, x9, 1
        B.HI    ConvSym.KernelSizeLoop

ConvSym.Requantize:
        LD1R    {v10.4s},[x8,#ConvSymPostProcessParams_Min]
        LD1R    {v6.4s},[x8,#ConvSymPostProcessParams_Max]
        LD1R    {v9.4s},[x8,#ConvSymPostProcessParams_ZeroPoint]
        tst     w10,MLAS_CONV_SYM_FLAG_INPUT_DIRECT
        beq     ConvSym.BroadcastScaleValue
        ldp     v5.4s,v8.4s,[x12]           // load scale vector
        b       ConvSym.AccumulatorsToFloat

ConvSym.BroadcastScaleValue:
        LD1R   {v5.4s},[x12]                // load scale Value
        mov    v8.4s, v5.4s

ConvSym.AccumulatorsToFloat:
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
        vcvt.s32.f32    v0.4s,v0.4s         // convert to float
        vcvt.s32.f32    v1.4s,v1.4s
        vcvt.s32.f32    v2.4s,v2.4s
        vcvt.s32.f32    v3.4s,v3.4s
        vmul.f32 v0.4s,v0.4s,v5.4s          // multiply by scale
        vmul.f32 v1.4s,v1.4s,v8.4s
        vmul.f32 v2.4s,v0.4s,v5.4s
        vmul.f32 v3.4s,v3.4s,v8.4s
        vpmin.f32 v0.4s,v0.4s,v6.4s         // clamp max value
        vpmin.f32 v1.4s,v1.4s,v6.4s
        vpmin.f32 v2.4s,v2.4s,v6.4s
        vpmin.f32 v3.4s,v3.4s,v6.4s
        vpmax.f32 v0.4s,v0.4s,v10.4s        // clamp min value
        vpmax.f32 v1.4s,v1.4s,v10.4s
        vpmax.f32 v2.4s,v2.4s,v10.4s
        vpmax.f32 v3.4s,v3.4s,v10.4s
        SUBS    x6, x6, 8
        vcvt.f32.s32   v0.4s,v0.4s          // convert to int
        vcvt.f32.s32   v1.4s,v1.4s
        vcvt.f32.s32   v2.4s,v2.4s
        vcvt.f32.s32   v3.4s,v3.4s
        add     v0.4s,v0.4s,v9.4s           // add zero point
        add     v1.4s,v1.4s,v9.4s
        add     v2.4s,v2.4s,v9.4s
        add     v3.4s,v3.4s,v9.4s
        SQXTN   v0.4h,v0.4s                 // shorten to int16
        SQXTN   v2.4h,v2.4s
        SQXTN2  v0.8h,v1.4s
        SQXTN2  v2.8h,v3.4s
        SQXTN   v0.8b,v0.8h                 // shorten to int8
        SQXTN2  v0.16b,v2.8h
        BO    ConvSym.PartialStore

        ST1     {v0.d}[1],[x5]              // full 2x8 store to c 
        ST1     {v0.8b}, [x2]

ConvSym.ExitKernel:
        ldp     d14,d15,[sp,#48]
        ldp     d12,d13,[sp,#32]
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#64
        ret

ConvSym.8InputChannels:
        LDR     d0, [x13]
        LDP     d4, d5, [x1]
        LDR     d1, [x15]
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
        B.HI    ConvSym.KernelSizeLoop
        B       ConvSym.Requantize

ConvSym.PartialStore:
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
        TBZ     x6, 0, ConvSym.ExitKernel
        ST1     {v0.b}[8], [x5]
        STR     b0, [x2]
        b       ConvSym.ExitKernel

        NESTED_END MlasConvSymKernelNeon

        END

