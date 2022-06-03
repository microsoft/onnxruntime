/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DepthwiseQConvKernelSize9Neon.asm

Abstract:

    This module implements the routine for the depthwise convolution
    operation with symmetrically quantized integer values for kernel
    size 9. ie, 3x3, 1x9, 9x1

--*/

#include "asmmacro.h"

        .equ  .LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE_BIT_INDEX,    1
        .equ  .LMLAS_CONV_SYM_FLAG_FIXED_POINT_SCALE, 4

//
// Stack frame layout for the depthwise conv kernel. d8-d15, x19-x30 need save
//
        .equ  .LMlasConvSymDepthwiseKernelSize9_x19_x20,                 0
        .equ  .LMlasConvSymDepthwiseKernelSize9_x21_x22,                 16
        .equ  .LMlasConvSymDepthwiseKernelSize9_x23_x24,                 32
        .equ  .LMlasConvSymDepthwiseKernelSize9_x25_x26,                 48
        .equ  .LMlasConvSymDepthwiseKernelSize9_x27_x28,                 64
        .equ  .LMlasConvSymDepthwiseKernelSize9_d8_d9,                   80
        .equ  .LMlasConvSymDepthwiseKernelSize9_d10_d11,                 96
        .equ  .LMlasConvSymDepthwiseKernelSize9_d12_d13,                 112
        .equ  .LMlasConvSymDepthwiseKernelSize9_d14_d15,                 128
        .equ  .LMlasConvSymDepthwiseKernelSize9_x0_x4,                   144
        .equ  .LMlasConvSymDepthwiseKernelSize9_multiplier_postshift,    160
        .equ  .LMlasConvSymDepthwiseKernelSize9_SavedRegisters,          176
        .equ  .LMlasConvSymDepthwiseKernelSize9_SavedRegisters_Neg,      -176

/*
struct MLAS_CONV_SYM_POST_PROCESS_PARAMS {
    const int32_t* Bias{nullptr};
    const float* Scale{nullptr};
    float MinimumValue;
    float MaximumValue;
    int32_t OutputZeroPoint;
    int32_t Padding;
    const int32_t* Multiplier{nullptr};
    const int32_t* PreShift{nullptr};
    const int32_t* PostShift{nullptr};
};
*/
        .equ  .LConvSymPostProcessParams_Bias,                     0
        .equ  .LConvSymPostProcessParams_Scale,                    8
        .equ  .LConvSymPostProcessParams_Min,                      16
        .equ  .LConvSymPostProcessParams_Max,                      20
        .equ  .LConvSymPostProcessParams_ZeroPoint,                24
        .equ  .LConvSymPostProcessParams_Multiplier,               32
        .equ  .LConvSymPostProcessParams_PreShift,                 40
        .equ  .LConvSymPostProcessParams_PostShift,                48

        .text

/*++

Routine Description:

    This routine is the inner kernel to compute a depthwise quantized convolution
    on kernel size 9 for u8s8

Arguments:

    Input (x0) - Supplies the address of the indirection buffer.

    Filter (x1) - Supplies the address of the filter buffer.

    Channels (x2) - Supplies the number of input and output channels.

    Output (x3) - Supplies the address of the output buffer.

    OutputCount (x4)- Supplies the number of image pixels.

    PostProcessParams (x5) - Supplies the address of the post process parameter block.

    KernelFlags (x6) - Supplies additional flags controlling the operation.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvSymDepthwiseKernelSize9Arm64U8S8

        stp     x19, x20, [sp, #.LMlasConvSymDepthwiseKernelSize9_SavedRegisters_Neg]!
        stp     x21, x22, [sp, #.LMlasConvSymDepthwiseKernelSize9_x21_x22]
        stp     x23, x24, [sp, #.LMlasConvSymDepthwiseKernelSize9_x23_x24]
        stp     x25, x26, [sp, #.LMlasConvSymDepthwiseKernelSize9_x25_x26]
        stp     x27, x28, [sp, #.LMlasConvSymDepthwiseKernelSize9_x27_x28]
        stp     d8, d9, [sp, #.LMlasConvSymDepthwiseKernelSize9_d8_d9]
        stp     d10, d11, [sp, #.LMlasConvSymDepthwiseKernelSize9_d10_d11]
        stp     d12, d13, [sp, #.LMlasConvSymDepthwiseKernelSize9_d12_d13]
        stp     d14, d15, [sp, #.LMlasConvSymDepthwiseKernelSize9_d14_d15]

        tst     x6, #.LMLAS_CONV_SYM_FLAG_FIXED_POINT_SCALE
        ldr     x11, [x5, #.LConvSymPostProcessParams_Multiplier]
        ldr     x12, [x5, #.LConvSymPostProcessParams_PostShift]
        ldr     x10, [x5, #.LConvSymPostProcessParams_PreShift]
        ldr     x9, [x5, #.LConvSymPostProcessParams_Bias]
        ldr     x8, [x5, #.LConvSymPostProcessParams_Scale]
        add     x5, x5, #.LConvSymPostProcessParams_ZeroPoint
        stp     x11, x12,[sp, #.LMlasConvSymDepthwiseKernelSize9_multiplier_postshift]
        csel    x8, x8, x10, eq
        ins     v12.d[0], x1                // Filter
        ins     v13.d[0], x9                // Bias
        ins     v13.d[1], x8                // Scale/PreShift
        ld1r    {v0.8h}, [x5]               // zero point
        movi    v5.16b, #0x80                // flip 0x80

        tbnz    x6, #.LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE_BIT_INDEX, .LMlasConvSymDepthwiseKernelSize9_SkipPerTensorScaleInit
        ld1r    {v1.4s}, [x8]               // load scale/PreShift value
        mov     v2.16b, v1.16b
        mov     v3.16b, v1.16b
        mov     v4.16b, v1.16b

.LMlasConvSymDepthwiseKernelSize9_SkipPerTensorScaleInit:

        add     x9, x3, x2                  // x9 <---- Ouput1
        cbz     x4, .LMlasConvSymDepthwiseKernelSize9_Exit

.LMlasConvSymDepthwiseKernelSize9_OutputLoop:
        ldp     x20, x21, [x0], #72         // input ptrs for Output0
        ldp     x22, x23, [x0, #-56]
        sub     x4, x4, #1
        ldp     x24, x25, [x0, #-40]
        ldp     x26, x27, [x0, #-24]
        ldur    x28, [x0, #-8]

        cbz     x4, .LMlasConvSymDepthwiseKernelSize9_Dup_Inputs
        ldp     x10, x11, [x0], #72         // input ptrs for Output1
        ldp     x12, x13, [x0, #-56]
        sub     x4, x4, #1
        ldp     x14, x15, [x0, #-40]
        ldp     x16, x17, [x0, #-24]
        ldur    x19, [x0, #-8]
        b       .LMlasConvSymDepthwiseKernelSize9_Loaded_Input

.LMlasConvSymDepthwiseKernelSize9_Dup_Inputs:
        mov     x9, x3                      // Output1 <-- Output0
        mov     x10, x20                    // Input1 <-- Input0
        mov     x11, x21
        mov     x12, x22
        mov     x13, x23
        mov     x14, x24
        mov     x15, x25
        mov     x16, x26
        mov     x17, x27
        mov     x19, x28

.LMlasConvSymDepthwiseKernelSize9_Loaded_Input:

        stp     x0, x4, [sp, #.LMlasConvSymDepthwiseKernelSize9_x0_x4]
        eor     x8, x8, x8                  // Processed channels
        umov    x1, v12.D[0]                // filter
        umov    x5, v13.D[0]                // bias
        umov    x7, v13.D[1]                // scale/PreShift
        ldp     x0, x4, [sp, #.LMlasConvSymDepthwiseKernelSize9_multiplier_postshift]

        cmp     x8, x2                      // Save one register by not using count down to zero here
        bhs     .LMlasConvSymDepthwiseKernelSize9_Finish_Channels16_Loop

.LMlasConvSymDepthwiseKernelSize9_Channels16_Loop:
        ld1     {v10.16b}, [x1], x2         // vk0
        ldr     q16, [x20, x8]              // out0 vi0
        ldr     q17, [x10, x8]              // out1 vi0
        ld1     {v6.4s, v7.4s, v8.4s, v9.4s}, [x5], #64       // bias vacc 0-15 for outs
        ld1     {v11.16b}, [x1], x2         // vk1
        ldr     q18, [x21, x8]              // out0 vi1
        ldr     q19, [x11, x8]              // out1 vi1

        eor     v16.16b, v16.16b, v5.16b    // -128 to signed int8
        eor     v17.16b, v17.16b, v5.16b
        ld1     {v14.16b}, [x1], x2         // vk2
        eor     v18.16b, v18.16b, v5.16b
        eor     v19.16b, v19.16b, v5.16b

        ldr     q20, [x22, x8]              // out0 vi2
        smull   v24.8h, v10.8b, v16.8b
        smull2  v25.8h, v10.16b, v16.16b
        ldr     q21, [x12, x8]              // out1 vi2
        smull   v26.8h, v10.8b, v17.8b
        ld1     {v15.16b}, [x1], x2         // vk3
        smull2  v27.8h, v10.16b, v17.16b
        ldr     q22, [x23, x8]              // out0 vi3
        smull   v28.8h, v11.8b, v18.8b
        smull2  v29.8h, v11.16b, v18.16b
        ldr     q23, [x13, x8]              // out1 vi3
        smull   v30.8h, v11.8b, v19.8b
        smull2  v31.8h, v11.16b, v19.16b

        eor     v20.16b, v20.16b, v5.16b
        eor     v21.16b, v21.16b, v5.16b
        eor     v22.16b, v22.16b, v5.16b
        eor     v23.16b, v23.16b, v5.16b
        ld1     {v10.16b}, [x1], x2         // vk4

        smlal   v24.8h, v14.8b, v20.8b
        smlal2  v25.8h, v14.16b, v20.16b
        smlal   v26.8h, v14.8b, v21.8b
        smlal2  v27.8h, v14.16b, v21.16b
        smlal   v28.8h, v15.8b, v22.8b
        smlal2  v29.8h, v15.16b, v22.16b
        smlal   v30.8h, v15.8b, v23.8b
        smlal2  v31.8h, v15.16b, v23.16b
        ld1     {v11.16b}, [x1], x2         // vk5

        saddw   v16.4s, v6.4s, v24.4h       // dup acc for out1
        saddw2  v17.4s, v7.4s, v24.8h
        saddw   v18.4s, v8.4s, v25.4h
        saddw2  v19.4s, v9.4s, v25.8h

        ldr     q20, [x24, x8]              // out0 vi4
        saddw   v6.4s, v6.4s, v26.4h
        saddw2  v7.4s, v7.4s, v26.8h
        ldr     q21, [x14, x8]              // out1 vi4
        saddw   v8.4s, v8.4s, v27.4h
        saddw2  v9.4s, v9.4s, v27.8h
        ldr     q22, [x25, x8]              // out0 vi5
        saddw   v16.4s, v16.4s, v28.4h
        saddw2  v17.4s, v17.4s, v28.8h
        ldr     q23, [x15, x8]              // out1 vi5
        saddw   v18.4s, v18.4s, v29.4h
        saddw2  v19.4s, v19.4s, v29.8h
        ld1     {v14.16b}, [x1], x2         // vk6

        saddw   v6.4s, v6.4s, v30.4h
        saddw2  v7.4s, v7.4s, v30.8h
        eor     v20.16b, v20.16b, v5.16b
        eor     v21.16b, v21.16b, v5.16b
        eor     v22.16b, v22.16b, v5.16b
        eor     v23.16b, v23.16b, v5.16b
        ld1     {v15.16b}, [x1], x2         // vk7
        saddw   v8.4s, v8.4s, v31.4h
        saddw2  v9.4s, v9.4s, v31.8h

        smull   v24.8h, v10.8b, v20.8b
        smull2  v25.8h, v10.16b, v20.16b
        smull   v26.8h, v10.8b, v21.8b
        smull2  v27.8h, v10.16b, v21.16b
        smull   v28.8h, v11.8b, v22.8b
        smull2  v29.8h, v11.16b, v22.16b
        smull   v30.8h, v11.8b, v23.8b
        smull2  v31.8h, v11.16b, v23.16b

        ldr     q20, [x26, x8]              // out0 vi6
        ldr     q21, [x16, x8]              // out1 vi6
        ldr     q22, [x27, x8]              // out0 vi7
        ldr     q23, [x17, x8]              // out1 vi7

        tbz     x6, #.LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE_BIT_INDEX, .LDonePerChannelScaleLoad_MlasConvSymDepthwiseKernelSize9
        ld1     {v1.4s, v2.4s, v3.4s, v4.4s}, [x7], #64     // scales/preshift 0-15 for outs

.LDonePerChannelScaleLoad_MlasConvSymDepthwiseKernelSize9:
        eor     v20.16b, v20.16b, v5.16b
        eor     v21.16b, v21.16b, v5.16b
        eor     v22.16b, v22.16b, v5.16b
        eor     v23.16b, v23.16b, v5.16b
        ldr     q10, [x1]                   // vk8

        smlal   v24.8h, v14.8b, v20.8b
        smlal2  v25.8h, v14.16b, v20.16b
        smlal   v26.8h, v14.8b, v21.8b
        smlal2  v27.8h, v14.16b, v21.16b
        smlal   v28.8h, v15.8b, v22.8b
        smlal2  v29.8h, v15.16b, v22.16b
        smlal   v30.8h, v15.8b, v23.8b
        smlal2  v31.8h, v15.16b, v23.16b

        saddw   v16.4s, v16.4s, v24.4h
        saddw2  v17.4s, v17.4s, v24.8h
        saddw   v18.4s, v18.4s, v25.4h
        saddw2  v19.4s, v19.4s, v25.8h
        ldr     q20, [x28, x8]              // out0 vi8
        saddw   v6.4s, v6.4s, v26.4h
        saddw2  v7.4s, v7.4s, v26.8h
        ldr     q21, [x19, x8]              // out1 vi8
        saddw   v8.4s, v8.4s, v27.4h
        saddw2  v9.4s, v9.4s, v27.8h

        saddw   v16.4s, v16.4s, v28.4h
        saddw2  v17.4s, v17.4s, v28.8h
        eor     v20.16b, v20.16b, v5.16b
        eor     v21.16b, v21.16b, v5.16b
        saddw   v18.4s, v18.4s, v29.4h
        saddw2  v19.4s, v19.4s, v29.8h

        saddw   v6.4s, v6.4s, v30.4h
        saddw2  v7.4s, v7.4s, v30.8h
        saddw   v8.4s, v8.4s, v31.4h
        saddw2  v9.4s, v9.4s, v31.8h

        smull   v24.8h, v10.8b, v20.8b
        smull2  v25.8h, v10.16b, v20.16b
        smull   v26.8h, v10.8b, v21.8b
        smull2  v27.8h, v10.16b, v21.16b

        saddw   v16.4s, v16.4s, v24.4h
        saddw2  v17.4s, v17.4s, v24.8h
        saddw   v18.4s, v18.4s, v25.4h
        saddw2  v19.4s, v19.4s, v25.8h

        saddw   v6.4s, v6.4s, v26.4h
        saddw2  v7.4s, v7.4s, v26.8h
        saddw   v8.4s, v8.4s, v27.4h
        saddw2  v9.4s, v9.4s, v27.8h

        tst     x6, #.LMLAS_CONV_SYM_FLAG_FIXED_POINT_SCALE
        bne     .LFixedPointScale_U8S8
.LFloatPointScale_U8S8:
        scvtf   v16.4s, v16.4s              // Requantize
        scvtf   v17.4s, v17.4s
        scvtf   v18.4s, v18.4s
        scvtf   v19.4s, v19.4s
        scvtf   v6.4s, v6.4s
        scvtf   v7.4s, v7.4s
        scvtf   v8.4s, v8.4s
        scvtf   v9.4s, v9.4s

        fmul    v16.4s, v16.4s, v1.4s
        fmul    v17.4s, v17.4s, v2.4s
        fmul    v18.4s, v18.4s, v3.4s
        fmul    v19.4s, v19.4s, v4.4s
        fmul    v6.4s, v6.4s, v1.4s
        fmul    v7.4s, v7.4s, v2.4s
        fmul    v8.4s, v8.4s, v3.4s
        fmul    v9.4s, v9.4s, v4.4s

        fcvtns  v16.4s, v16.4s
        fcvtns  v17.4s, v17.4s
        fcvtns  v18.4s, v18.4s
        fcvtns  v19.4s, v19.4s
        fcvtns  v6.4s, v6.4s
        fcvtns  v7.4s, v7.4s
        fcvtns  v8.4s, v8.4s
        fcvtns  v9.4s, v9.4s
        b .LQuantize_U8S8

.LFixedPointScale_U8S8:
        tbz     x6,#.LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE_BIT_INDEX, .LBroadcastFixedPointValue_U8S8
        ldp     q20,q21,[x0],32          // load multiplier vector
        ldp     q22,q23,[x0],32
        ldp     q24,q25,[x4],32          // load postshift vector
        ldp     q26,q27,[x4],32
        b       .LScaleWithFixedPoint_U8S8

.LBroadcastFixedPointValue_U8S8:
        ld1r    {v20.4s},[x0]           // load multiplier Value
        mov     v21.16b, v20.16b
        mov     v22.16b, v20.16b
        mov     v23.16b, v20.16b

        ld1r    {v24.4s},[x4]           // load postshift Value
        mov     v25.16b, v24.16b
        mov     v26.16b, v24.16b
        mov     v27.16b, v24.16b
.LScaleWithFixedPoint_U8S8:
        sqshl   v6.4s, v6.4s, v1.4s          // preshift
        sqshl   v7.4s, v7.4s, v2.4s
        sqshl   v8.4s, v8.4s, v3.4s
        sqshl   v9.4s, v9.4s, v4.4s
        sqshl   v16.4s, v16.4s, v1.4s
        sqshl   v17.4s, v17.4s, v2.4s
        sqshl   v18.4s, v18.4s, v3.4s
        sqshl   v19.4s, v19.4s, v4.4s

        sqdmulh v6.4s, v6.4s, v20.4s         // multiply by multiplier
        sqdmulh v7.4s, v7.4s, v21.4s
        sqdmulh v8.4s, v8.4s, v22.4s
        sqdmulh v9.4s, v9.4s, v23.4s
        sqdmulh v16.4s, v16.4s, v20.4s
        sqdmulh v17.4s, v17.4s, v21.4s
        sqdmulh v18.4s, v18.4s, v22.4s
        sqdmulh v19.4s, v19.4s, v23.4s

        srshl   v6.4s,v6.4s, v24.4s          // post shift
        srshl   v7.4s,v7.4s, v25.4s
        srshl   v8.4s,v8.4s, v26.4s
        srshl   v9.4s,v9.4s, v27.4s
        srshl   v16.4s,v16.4s, v24.4s
        srshl   v17.4s,v17.4s, v25.4s
        srshl   v18.4s,v18.4s, v26.4s
        srshl   v19.4s,v19.4s, v27.4s

.LQuantize_U8S8:
        sqxtn   v16.4h, v16.4s              // +zp, narrow and combine
        sqxtn   v18.4h, v18.4s
        sqxtn   v6.4h, v6.4s
        sqxtn   v8.4h, v8.4s
        sqxtn2  v16.8h, v17.4s
        sqxtn2  v18.8h, v19.4s
        sqxtn2  v6.8h, v7.4s
        sqxtn2  v8.8h, v9.4s
        sqadd   v16.8h, v16.8h, v0.8h
        sqadd   v18.8h, v18.8h, v0.8h
        sqadd   v6.8h, v6.8h, v0.8h
        sqadd   v8.8h, v8.8h, v0.8h
        sqxtun  v16.8b, v16.8h
        sqxtun2 v16.16b, v18.8h
        sqxtun  v6.8b, v6.8h
        sqxtun2 v6.16b, v8.8h

        str     q16, [x3, x8]
        str     q6, [x9, x8]
        add     x8, x8, #16
        umov    x1, v12.D[0]                // filter
        cmp     x8, x2
        add     x1, x1, x8
        blo     .LMlasConvSymDepthwiseKernelSize9_Channels16_Loop

.LMlasConvSymDepthwiseKernelSize9_Finish_Channels16_Loop:
        ldp     x0, x4, [sp, #.LMlasConvSymDepthwiseKernelSize9_x0_x4]
        add     x3, x3, x2, LSL #1
        add     x9, x9, x2, LSL #1
        cbnz    x4, .LMlasConvSymDepthwiseKernelSize9_OutputLoop

.LMlasConvSymDepthwiseKernelSize9_Exit:
        ldp     d14, d15, [sp, #.LMlasConvSymDepthwiseKernelSize9_d14_d15]
        ldp     d12, d13, [sp, #.LMlasConvSymDepthwiseKernelSize9_d12_d13]
        ldp     d10, d11, [sp, #.LMlasConvSymDepthwiseKernelSize9_d10_d11]
        ldp     d8, d9, [sp, #.LMlasConvSymDepthwiseKernelSize9_d8_d9]
        ldp     x27, x28, [sp, #.LMlasConvSymDepthwiseKernelSize9_x27_x28]
        ldp     x25, x26, [sp, #.LMlasConvSymDepthwiseKernelSize9_x25_x26]
        ldp     x23, x24, [sp, #.LMlasConvSymDepthwiseKernelSize9_x23_x24]
        ldp     x21, x22, [sp, #.LMlasConvSymDepthwiseKernelSize9_x21_x22]
        ldp     x19, x20, [sp], #.LMlasConvSymDepthwiseKernelSize9_SavedRegisters
        ret


/*++

Routine Description:

    This routine is the inner kernel to compute a depthwise quantized convolution
    on kernel size 9 for s8s8

Arguments:

    Input (x0) - Supplies the address of the indirection buffer.

    Filter (x1) - Supplies the address of the filter buffer.

    Channels (x2) - Supplies the number of input and output channels.

    Output (x3) - Supplies the address of the output buffer.

    OutputCount (x4)- Supplies the number of image pixels.

    PostProcessParams (x5) - Supplies the address of the post process parameter block.

    KernelFlags (x6) - Supplies additional flags controlling the operation.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvSymDepthwiseKernelSize9Arm64S8S8

        stp     x19, x20, [sp, #.LMlasConvSymDepthwiseKernelSize9_SavedRegisters_Neg]!
        stp     x21, x22, [sp, #.LMlasConvSymDepthwiseKernelSize9_x21_x22]
        stp     x23, x24, [sp, #.LMlasConvSymDepthwiseKernelSize9_x23_x24]
        stp     x25, x26, [sp, #.LMlasConvSymDepthwiseKernelSize9_x25_x26]
        stp     x27, x28, [sp, #.LMlasConvSymDepthwiseKernelSize9_x27_x28]
        stp     d8, d9, [sp, #.LMlasConvSymDepthwiseKernelSize9_d8_d9]
        stp     d10, d11, [sp, #.LMlasConvSymDepthwiseKernelSize9_d10_d11]
        stp     d12, d13, [sp, #.LMlasConvSymDepthwiseKernelSize9_d12_d13]
        stp     d14, d15, [sp, #.LMlasConvSymDepthwiseKernelSize9_d14_d15]

        tst     x6, #.LMLAS_CONV_SYM_FLAG_FIXED_POINT_SCALE
        ldr     x9, [x5, #.LConvSymPostProcessParams_Bias]
        ldr     x8, [x5, #.LConvSymPostProcessParams_Scale]
        ldr     x10, [x5, #.LConvSymPostProcessParams_PreShift]
        ldr     x11, [x5, #.LConvSymPostProcessParams_Multiplier]
        ldr     x12, [x5, #.LConvSymPostProcessParams_PostShift]
        add     x5, x5, #.LConvSymPostProcessParams_ZeroPoint
        csel    x8, x8, x10, eq
        ins     v12.d[0], x1                // Filter
        ins     v13.d[0], x9                // Bias
        ins     v13.d[1], x8                // Scale/PreShift
        ins     v5.d[0], x11                // Multiplier
        ins     v5.d[1], x12                // PostShift
        ld1r    {v0.8h}, [x5]               // zero point

        tbnz    x6, #.LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE_BIT_INDEX, .LMlasConvSymDepthwiseKernelSize9S8S8_SkipPerTensorScaleInit
        ld1r    {v1.4s}, [x8]               // load scale/PreShift value
        mov     v2.16b, v1.16b
        mov     v3.16b, v1.16b
        mov     v4.16b, v1.16b

.LMlasConvSymDepthwiseKernelSize9S8S8_SkipPerTensorScaleInit:

        add     x9, x3, x2                  // x9 <---- Ouput1
        cbz     x4, .LMlasConvSymDepthwiseKernelSize9S8S8_Exit

.LMlasConvSymDepthwiseKernelSize9S8S8_OutputLoop:
        ldp     x20, x21, [x0], #72         // input ptrs for Output0
        ldp     x22, x23, [x0, #-56]
        sub     x4, x4, #1
        ldp     x24, x25, [x0, #-40]
        ldp     x26, x27, [x0, #-24]
        ldur    x28, [x0, #-8]

        cbz     x4, .LMlasConvSymDepthwiseKernelSize9S8S8_Dup_Inputs
        ldp     x10, x11, [x0], #72         // input ptrs for Output0
        ldp     x12, x13, [x0, #-56]
        sub     x4, x4, #1
        ldp     x14, x15, [x0, #-40]
        ldp     x16, x17, [x0, #-24]
        ldur    x19, [x0, #-8]
        b       .LMlasConvSymDepthwiseKernelSize9S8S8_Loaded_Input

.LMlasConvSymDepthwiseKernelSize9S8S8_Dup_Inputs:
        mov     x9, x3                      // Output1 <-- Output0
        mov     x10, x20
        mov     x11, x21
        mov     x12, x22
        mov     x13, x23
        mov     x14, x24
        mov     x15, x25
        mov     x16, x26
        mov     x17, x27
        mov     x19, x28

.LMlasConvSymDepthwiseKernelSize9S8S8_Loaded_Input:

        stp     x0, x4, [sp, #.LMlasConvSymDepthwiseKernelSize9_x0_x4]
        eor     x8, x8, x8                  // Processed channels
        umov    x1, v12.D[0]                // filter
        umov    x5, v13.D[0]                // bias
        umov    x7, v13.D[1]                // scale/PreShift
        umov    x0, v5.D[0]
        umov    x4, v5.D[1]

        cmp     x8, x2                      // Save one register by not using count down to zero here
        bhs     .LMlasConvSymDepthwiseKernelSize9S8S8_Finish_Channels16_Loop

.LMlasConvSymDepthwiseKernelSize9S8S8_Channels16_Loop:
        ld1     {v10.16b}, [x1], x2         // vk0
        ldr     q16, [x20, x8]              // out0 vi0
        ldr     q17, [x10, x8]              // out1 vi0
        ld1     {v6.4s, v7.4s, v8.4s, v9.4s}, [x5], #64       // bias vacc 0-15 for outs
        ld1     {v11.16b}, [x1], x2         // vk1
        ldr     q18, [x21, x8]              // out0 vi1
        ldr     q19, [x11, x8]              // out1 vi1

        ld1     {v14.16b}, [x1], x2         // vk2

        ldr     q20, [x22, x8]              // out0 vi2
        smull   v24.8h, v10.8b, v16.8b
        smull2  v25.8h, v10.16b, v16.16b
        ldr     q21, [x12, x8]              // out1 vi2
        smull   v26.8h, v10.8b, v17.8b
        ld1     {v15.16b}, [x1], x2         // vk3
        smull2  v27.8h, v10.16b, v17.16b
        ldr     q22, [x23, x8]              // out0 vi3
        smull   v28.8h, v11.8b, v18.8b
        smull2  v29.8h, v11.16b, v18.16b
        ldr     q23, [x13, x8]              // out1 vi3
        smull   v30.8h, v11.8b, v19.8b
        smull2  v31.8h, v11.16b, v19.16b

        ld1     {v10.16b}, [x1], x2         // vk4

        smlal   v24.8h, v14.8b, v20.8b
        smlal2  v25.8h, v14.16b, v20.16b
        smlal   v26.8h, v14.8b, v21.8b
        smlal2  v27.8h, v14.16b, v21.16b
        smlal   v28.8h, v15.8b, v22.8b
        smlal2  v29.8h, v15.16b, v22.16b
        smlal   v30.8h, v15.8b, v23.8b
        smlal2  v31.8h, v15.16b, v23.16b
        ld1     {v11.16b}, [x1], x2         // vk5

        saddw   v16.4s, v6.4s, v24.4h       // dup acc for out1
        saddw2  v17.4s, v7.4s, v24.8h
        saddw   v18.4s, v8.4s, v25.4h
        saddw2  v19.4s, v9.4s, v25.8h

        ldr     q20, [x24, x8]              // out0 vi4
        saddw   v6.4s, v6.4s, v26.4h
        saddw2  v7.4s, v7.4s, v26.8h
        ldr     q21, [x14, x8]              // out1 vi4
        saddw   v8.4s, v8.4s, v27.4h
        saddw2  v9.4s, v9.4s, v27.8h
        ldr     q22, [x25, x8]              // out0 vi5
        saddw   v16.4s, v16.4s, v28.4h
        saddw2  v17.4s, v17.4s, v28.8h
        ldr     q23, [x15, x8]              // out1 vi5
        saddw   v18.4s, v18.4s, v29.4h
        saddw2  v19.4s, v19.4s, v29.8h
        ld1     {v14.16b}, [x1], x2         // vk6

        saddw   v6.4s, v6.4s, v30.4h
        saddw2  v7.4s, v7.4s, v30.8h
        ld1     {v15.16b}, [x1], x2         // vk7
        saddw   v8.4s, v8.4s, v31.4h
        saddw2  v9.4s, v9.4s, v31.8h

        smull   v24.8h, v10.8b, v20.8b
        smull2  v25.8h, v10.16b, v20.16b
        smull   v26.8h, v10.8b, v21.8b
        smull2  v27.8h, v10.16b, v21.16b
        smull   v28.8h, v11.8b, v22.8b
        smull2  v29.8h, v11.16b, v22.16b
        smull   v30.8h, v11.8b, v23.8b
        smull2  v31.8h, v11.16b, v23.16b

        ldr     q20, [x26, x8]              // out0 vi6
        ldr     q21, [x16, x8]              // out1 vi6
        ldr     q22, [x27, x8]              // out0 vi7
        ldr     q23, [x17, x8]              // out1 vi7

        tbz     x6, #.LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE_BIT_INDEX, .LDonePerChannelScaleLoad_MlasConvSymDepthwiseKernelSize9S8S8
        ld1     {v1.4s, v2.4s, v3.4s, v4.4s}, [x7], #64     // scales/preshift 0-15 for outs

.LDonePerChannelScaleLoad_MlasConvSymDepthwiseKernelSize9S8S8:
        ldr     q10, [x1]                   // vk8

        smlal   v24.8h, v14.8b, v20.8b
        smlal2  v25.8h, v14.16b, v20.16b
        smlal   v26.8h, v14.8b, v21.8b
        smlal2  v27.8h, v14.16b, v21.16b
        smlal   v28.8h, v15.8b, v22.8b
        smlal2  v29.8h, v15.16b, v22.16b
        smlal   v30.8h, v15.8b, v23.8b
        smlal2  v31.8h, v15.16b, v23.16b

        saddw   v16.4s, v16.4s, v24.4h
        saddw2  v17.4s, v17.4s, v24.8h
        saddw   v18.4s, v18.4s, v25.4h
        saddw2  v19.4s, v19.4s, v25.8h
        ldr     q20, [x28, x8]              // out0 vi8
        saddw   v6.4s, v6.4s, v26.4h
        saddw2  v7.4s, v7.4s, v26.8h
        ldr     q21, [x19, x8]              // out1 vi8
        saddw   v8.4s, v8.4s, v27.4h
        saddw2  v9.4s, v9.4s, v27.8h

        saddw   v16.4s, v16.4s, v28.4h
        saddw2  v17.4s, v17.4s, v28.8h
        saddw   v18.4s, v18.4s, v29.4h
        saddw2  v19.4s, v19.4s, v29.8h

        saddw   v6.4s, v6.4s, v30.4h
        saddw2  v7.4s, v7.4s, v30.8h
        saddw   v8.4s, v8.4s, v31.4h
        saddw2  v9.4s, v9.4s, v31.8h

        smull   v24.8h, v10.8b, v20.8b
        smull2  v25.8h, v10.16b, v20.16b
        smull   v26.8h, v10.8b, v21.8b
        smull2  v27.8h, v10.16b, v21.16b

        saddw   v16.4s, v16.4s, v24.4h
        saddw2  v17.4s, v17.4s, v24.8h
        saddw   v18.4s, v18.4s, v25.4h
        saddw2  v19.4s, v19.4s, v25.8h

        saddw   v6.4s, v6.4s, v26.4h
        saddw2  v7.4s, v7.4s, v26.8h
        saddw   v8.4s, v8.4s, v27.4h
        saddw2  v9.4s, v9.4s, v27.8h

        tst     x6, #.LMLAS_CONV_SYM_FLAG_FIXED_POINT_SCALE
        bne     FixedPointScale
FloatPointScale:
        scvtf   v16.4s, v16.4s              // Requantize
        scvtf   v17.4s, v17.4s
        scvtf   v18.4s, v18.4s
        scvtf   v19.4s, v19.4s
        scvtf   v6.4s, v6.4s
        scvtf   v7.4s, v7.4s
        scvtf   v8.4s, v8.4s
        scvtf   v9.4s, v9.4s

        fmul    v16.4s, v16.4s, v1.4s
        fmul    v17.4s, v17.4s, v2.4s
        fmul    v18.4s, v18.4s, v3.4s
        fmul    v19.4s, v19.4s, v4.4s
        fmul    v6.4s, v6.4s, v1.4s
        fmul    v7.4s, v7.4s, v2.4s
        fmul    v8.4s, v8.4s, v3.4s
        fmul    v9.4s, v9.4s, v4.4s

        fcvtns  v16.4s, v16.4s
        fcvtns  v17.4s, v17.4s
        fcvtns  v18.4s, v18.4s
        fcvtns  v19.4s, v19.4s
        fcvtns  v6.4s, v6.4s
        fcvtns  v7.4s, v7.4s
        fcvtns  v8.4s, v8.4s
        fcvtns  v9.4s, v9.4s
        b Quantize

FixedPointScale:
        tbz     x6,#.LMLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE_BIT_INDEX, .LBroadcastFixedPointValue
        ldp     q20,q21,[x0],32          // load multiplier vector
        ldp     q22,q23,[x0],32
        ldp     q24,q25,[x4],32          // load postshift vector
        ldp     q26,q27,[x4],32
        b       ScaleWithFixedPoint

.LBroadcastFixedPointValue:
        ld1r    {v20.4s},[x0]           // load multiplier Value
        mov     v21.16b, v20.16b
        mov     v22.16b, v20.16b
        mov     v23.16b, v20.16b

        ld1r    {v24.4s},[x4]           // load postshift Value
        mov     v25.16b, v24.16b
        mov     v26.16b, v24.16b
        mov     v27.16b, v24.16b
ScaleWithFixedPoint:
        sqshl   v6.4s, v6.4s, v1.4s          // preshift
        sqshl   v7.4s, v7.4s, v2.4s
        sqshl   v8.4s, v8.4s, v3.4s
        sqshl   v9.4s, v9.4s, v4.4s
        sqshl   v16.4s, v16.4s, v1.4s
        sqshl   v17.4s, v17.4s, v2.4s
        sqshl   v18.4s, v18.4s, v3.4s
        sqshl   v19.4s, v19.4s, v4.4s

        sqdmulh v6.4s, v6.4s, v20.4s         // multiply by multiplier
        sqdmulh v7.4s, v7.4s, v21.4s
        sqdmulh v8.4s, v8.4s, v22.4s
        sqdmulh v9.4s, v9.4s, v23.4s
        sqdmulh v16.4s, v16.4s, v20.4s
        sqdmulh v17.4s, v17.4s, v21.4s
        sqdmulh v18.4s, v18.4s, v22.4s
        sqdmulh v19.4s, v19.4s, v23.4s

        srshl   v6.4s,v6.4s, v24.4s          // post shift
        srshl   v7.4s,v7.4s, v25.4s
        srshl   v8.4s,v8.4s, v26.4s
        srshl   v9.4s,v9.4s, v27.4s
        srshl   v16.4s,v16.4s, v24.4s
        srshl   v17.4s,v17.4s, v25.4s
        srshl   v18.4s,v18.4s, v26.4s
        srshl   v19.4s,v19.4s, v27.4s

Quantize:
        sqxtn   v16.4h, v16.4s              // +zp, narrow and combine
        sqxtn   v18.4h, v18.4s
        sqxtn   v6.4h, v6.4s
        sqxtn   v8.4h, v8.4s
        sqxtn2  v16.8h, v17.4s
        sqxtn2  v18.8h, v19.4s
        sqxtn2  v6.8h, v7.4s
        sqxtn2  v8.8h, v9.4s
        sqadd   v16.8h, v16.8h, v0.8h
        sqadd   v18.8h, v18.8h, v0.8h
        sqadd   v6.8h, v6.8h, v0.8h
        sqadd   v8.8h, v8.8h, v0.8h
        sqxtn   v16.8b, v16.8h
        sqxtn2  v16.16b, v18.8h
        sqxtn   v6.8b, v6.8h
        sqxtn2  v6.16b, v8.8h

        str     q16, [x3, x8]
        str     q6, [x9, x8]
        add     x8, x8, #16
        umov    x1, v12.D[0]                // filter
        cmp     x8, x2
        add     x1, x1, x8
        blo     .LMlasConvSymDepthwiseKernelSize9S8S8_Channels16_Loop

.LMlasConvSymDepthwiseKernelSize9S8S8_Finish_Channels16_Loop:
        ldp     x0, x4, [sp, #.LMlasConvSymDepthwiseKernelSize9_x0_x4]
        add     x3, x3, x2, LSL #1
        add     x9, x9, x2, LSL #1
        cbnz    x4, .LMlasConvSymDepthwiseKernelSize9S8S8_OutputLoop

.LMlasConvSymDepthwiseKernelSize9S8S8_Exit:
        ldp     d14, d15, [sp, #.LMlasConvSymDepthwiseKernelSize9_d14_d15]
        ldp     d12, d13, [sp, #.LMlasConvSymDepthwiseKernelSize9_d12_d13]
        ldp     d10, d11, [sp, #.LMlasConvSymDepthwiseKernelSize9_d10_d11]
        ldp     d8, d9, [sp, #.LMlasConvSymDepthwiseKernelSize9_d8_d9]
        ldp     x27, x28, [sp, #.LMlasConvSymDepthwiseKernelSize9_x27_x28]
        ldp     x25, x26, [sp, #.LMlasConvSymDepthwiseKernelSize9_x25_x26]
        ldp     x23, x24, [sp, #.LMlasConvSymDepthwiseKernelSize9_x23_x24]
        ldp     x21, x22, [sp, #.LMlasConvSymDepthwiseKernelSize9_x21_x22]
        ldp     x19, x20, [sp], #.LMlasConvSymDepthwiseKernelSize9_SavedRegisters
        ret

        .end
