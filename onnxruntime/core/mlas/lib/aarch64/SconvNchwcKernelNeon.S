/*++
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT

Module Name:

    SconvNchwcKernelNeon.S

Abstract:

    This module implements the single precision NCHWC convolution kernel for
    AArch64 processors with NEON support.

--*/

#if defined(__aarch64__) && !defined(_WIN32)

#include "asmmacro.h"

    .text
    .align 2

//
// PROCESS_LANE - Load one 16-float filter row and accumulate it using a
// single input lane.
//

    .macro PROCESS_LANE offset0, offset1, inreg, lane
        ldp     q16, q17, [x7, #\offset0]
        ldp     q18, q19, [x7, #\offset1]
        fmla    v0.4s, v16.4s, \inreg\().s[\lane]
        fmla    v1.4s, v17.4s, \inreg\().s[\lane]
        fmla    v2.4s, v18.4s, \inreg\().s[\lane]
        fmla    v3.4s, v19.4s, \inreg\().s[\lane]
    .endm

//
// PROCESS_LANE_2OUT - Load one 16-float filter row and accumulate it using
// two input lanes for a dual-output path.
//

    .macro PROCESS_LANE_2OUT offset0, offset1, inreg0, inreg1, lane
        ldp     q16, q17, [x7, #\offset0]
        ldp     q18, q19, [x7, #\offset1]
        fmla    v0.4s,  v16.4s, \inreg0\().s[\lane]
        fmla    v1.4s,  v17.4s, \inreg0\().s[\lane]
        fmla    v2.4s,  v18.4s, \inreg0\().s[\lane]
        fmla    v3.4s,  v19.4s, \inreg0\().s[\lane]
        fmla    v20.4s, v16.4s, \inreg1\().s[\lane]
        fmla    v21.4s, v17.4s, \inreg1\().s[\lane]
        fmla    v22.4s, v18.4s, \inreg1\().s[\lane]
        fmla    v23.4s, v19.4s, \inreg1\().s[\lane]
    .endm

//
// PROCESS_LANE_3OUT - Load one 16-float filter row and accumulate it using
// three input lanes for a tri-output path.
//

    .macro PROCESS_LANE_3OUT offset0, offset1, inreg0, inreg1, inreg2, lane
        ldp     q16, q17, [x7, #\offset0]
        ldp     q18, q19, [x7, #\offset1]
        fmla    v0.4s,  v16.4s, \inreg0\().s[\lane]
        fmla    v1.4s,  v17.4s, \inreg0\().s[\lane]
        fmla    v2.4s,  v18.4s, \inreg0\().s[\lane]
        fmla    v3.4s,  v19.4s, \inreg0\().s[\lane]
        fmla    v4.4s,  v16.4s, \inreg1\().s[\lane]
        fmla    v5.4s,  v17.4s, \inreg1\().s[\lane]
        fmla    v6.4s,  v18.4s, \inreg1\().s[\lane]
        fmla    v7.4s,  v19.4s, \inreg1\().s[\lane]
        fmla    v8.4s,  v16.4s, \inreg2\().s[\lane]
        fmla    v9.4s,  v17.4s, \inreg2\().s[\lane]
        fmla    v10.4s, v18.4s, \inreg2\().s[\lane]
        fmla    v11.4s, v19.4s, \inreg2\().s[\lane]
    .endm

//
// PROCESS_LANE_4OUT - Load one 16-float filter row and accumulate it using
// four input lanes for a quad-output path.
//

    .macro PROCESS_LANE_4OUT offset0, offset1, inreg0, inreg1, inreg2, inreg3, lane
        ldp     q16, q17, [x7, #\offset0]
        ldp     q18, q19, [x7, #\offset1]
        fmla    v0.4s,  v16.4s, \inreg0\().s[\lane]
        fmla    v1.4s,  v17.4s, \inreg0\().s[\lane]
        fmla    v2.4s,  v18.4s, \inreg0\().s[\lane]
        fmla    v3.4s,  v19.4s, \inreg0\().s[\lane]
        fmla    v4.4s,  v16.4s, \inreg1\().s[\lane]
        fmla    v5.4s,  v17.4s, \inreg1\().s[\lane]
        fmla    v6.4s,  v18.4s, \inreg1\().s[\lane]
        fmla    v7.4s,  v19.4s, \inreg1\().s[\lane]
        fmla    v8.4s,  v16.4s, \inreg2\().s[\lane]
        fmla    v9.4s,  v17.4s, \inreg2\().s[\lane]
        fmla    v10.4s, v18.4s, \inreg2\().s[\lane]
        fmla    v11.4s, v19.4s, \inreg2\().s[\lane]
        fmla    v12.4s, v16.4s, \inreg3\().s[\lane]
        fmla    v13.4s, v17.4s, \inreg3\().s[\lane]
        fmla    v14.4s, v18.4s, \inreg3\().s[\lane]
        fmla    v15.4s, v19.4s, \inreg3\().s[\lane]
    .endm

//
// CONV_LOOP_PAD - Convolution loop with per-position bounds checks. This
// path is used for the padded output regions on the left and right edges.
//

    .macro CONV_LOOP_PAD
        mov     x7, x4                       // Filter pointer
        mov     x8, x1                       // Input row pointer
        mov     x9, x13                      // Row start pointer
        mov     x10, x27                     // KernelHeight counter

        cbz     x10, 90f
        cbz     x28, 90f

    91:
        mov     x12, x8                      // Input pointer for width
        mov     x16, x28                     // KernelWidth counter

    92:
        // Branch if the input pointer lies outside [row_start, row_start + row_width).
        sub     x11, x12, x9
        cmp     x11, x14
        b.hs    93f

        ldp     q4, q5, [x12, #0]
        ldp     q6, q7, [x12, #32]

        PROCESS_LANE 0,   32,  v4, 0
        PROCESS_LANE 64,  96,  v4, 1
        PROCESS_LANE 128, 160, v4, 2
        PROCESS_LANE 192, 224, v4, 3
        PROCESS_LANE 256, 288, v5, 0
        PROCESS_LANE 320, 352, v5, 1
        PROCESS_LANE 384, 416, v5, 2
        PROCESS_LANE 448, 480, v5, 3
        PROCESS_LANE 512, 544, v6, 0
        PROCESS_LANE 576, 608, v6, 1
        PROCESS_LANE 640, 672, v6, 2
        PROCESS_LANE 704, 736, v6, 3
        PROCESS_LANE 768, 800, v7, 0
        PROCESS_LANE 832, 864, v7, 1
        PROCESS_LANE 896, 928, v7, 2
        PROCESS_LANE 960, 992, v7, 3

    93:
        add     x7, x7, #1024
        add     x12, x12, x23
        subs    x16, x16, #1
        b.ne    92b

        add     x9, x9, x15
        add     x8, x8, x15
        subs    x10, x10, #1
        b.ne    91b

    90:
    .endm

//
// CONV_LOOP_MID - Convolution loop without bounds checks. Output positions in
// the middle region are guaranteed to be fully in-bounds.
//

    .macro CONV_LOOP_MID
        mov     x7, x4                       // Filter pointer
        mov     x8, x1                       // Input row pointer
        mov     x10, x27                     // KernelHeight counter

        cbz     x10, 100f
        cbz     x28, 100f

    101:
        mov     x12, x8                      // Input pointer for width
        mov     x16, x28                     // KernelWidth counter

    102:
        ldp     q4, q5, [x12, #0]
        ldp     q6, q7, [x12, #32]

        PROCESS_LANE 0,   32,  v4, 0
        PROCESS_LANE 64,  96,  v4, 1
        PROCESS_LANE 128, 160, v4, 2
        PROCESS_LANE 192, 224, v4, 3
        PROCESS_LANE 256, 288, v5, 0
        PROCESS_LANE 320, 352, v5, 1
        PROCESS_LANE 384, 416, v5, 2
        PROCESS_LANE 448, 480, v5, 3
        PROCESS_LANE 512, 544, v6, 0
        PROCESS_LANE 576, 608, v6, 1
        PROCESS_LANE 640, 672, v6, 2
        PROCESS_LANE 704, 736, v6, 3
        PROCESS_LANE 768, 800, v7, 0
        PROCESS_LANE 832, 864, v7, 1
        PROCESS_LANE 896, 928, v7, 2
        PROCESS_LANE 960, 992, v7, 3

        add     x7, x7, #1024
        add     x12, x12, x23
        subs    x16, x16, #1
        b.ne    102b

        add     x8, x8, x15
        subs    x10, x10, #1
        b.ne    101b

    100:
    .endm

//
// CONV_LOOP_MID_3OUT - Convolution loop without bounds checks that computes
// three adjacent output points per iteration.
//

    .macro CONV_LOOP_MID_3OUT
        mov     x7, x4                       // Filter pointer
        mov     x8, x1                       // Input row pointer (output 0)
        mov     x9, x27                      // KernelHeight counter

        cbz     x9, 110f
        cbz     x28, 110f

    111:
        mov     x12, x8                      // Input pointer for width (output 0)
        mov     x16, x28                     // KernelWidth counter

    112:
        add     x11, x12, x22                // Output 1 input pointer
        add     x10, x12, x22, lsl #1        // Output 2 input pointer

        ldp     q20, q21, [x12, #0]
        ldp     q22, q23, [x12, #32]
        ldp     q24, q25, [x11, #0]
        ldp     q26, q27, [x11, #32]
        ldp     q12, q13, [x10, #0]
        ldp     q14, q15, [x10, #32]

        PROCESS_LANE_3OUT 0,   32,  v20, v24, v12, 0
        PROCESS_LANE_3OUT 64,  96,  v20, v24, v12, 1
        PROCESS_LANE_3OUT 128, 160, v20, v24, v12, 2
        PROCESS_LANE_3OUT 192, 224, v20, v24, v12, 3
        PROCESS_LANE_3OUT 256, 288, v21, v25, v13, 0
        PROCESS_LANE_3OUT 320, 352, v21, v25, v13, 1
        PROCESS_LANE_3OUT 384, 416, v21, v25, v13, 2
        PROCESS_LANE_3OUT 448, 480, v21, v25, v13, 3
        PROCESS_LANE_3OUT 512, 544, v22, v26, v14, 0
        PROCESS_LANE_3OUT 576, 608, v22, v26, v14, 1
        PROCESS_LANE_3OUT 640, 672, v22, v26, v14, 2
        PROCESS_LANE_3OUT 704, 736, v22, v26, v14, 3
        PROCESS_LANE_3OUT 768, 800, v23, v27, v15, 0
        PROCESS_LANE_3OUT 832, 864, v23, v27, v15, 1
        PROCESS_LANE_3OUT 896, 928, v23, v27, v15, 2
        PROCESS_LANE_3OUT 960, 992, v23, v27, v15, 3

        add     x7, x7, #1024
        add     x12, x12, x23
        subs    x16, x16, #1
        b.ne    112b

        add     x8, x8, x15
        subs    x9, x9, #1
        b.ne    111b

    110:
    .endm

//
// CONV_LOOP_MID_2OUT - Convolution loop without bounds checks that computes
// two adjacent output points per iteration.
//

    .macro CONV_LOOP_MID_2OUT
        mov     x7, x4                       // Filter pointer
        mov     x8, x1                       // Input row pointer (output 0)
        add     x9, x1, x22                  // Input row pointer (output 1)
        mov     x10, x27                     // KernelHeight counter

        cbz     x10, 120f
        cbz     x28, 120f

    121:
        mov     x12, x8                      // Input pointer for width (output 0)
        mov     x11, x9                      // Input pointer for width (output 1)
        mov     x16, x28                     // KernelWidth counter

    122:
        ldp     q4,  q5,  [x12, #0]
        ldp     q6,  q7,  [x12, #32]
        ldp     q24, q25, [x11, #0]
        ldp     q26, q27, [x11, #32]

        PROCESS_LANE_2OUT 0,   32,  v4, v24, 0
        PROCESS_LANE_2OUT 64,  96,  v4, v24, 1
        PROCESS_LANE_2OUT 128, 160, v4, v24, 2
        PROCESS_LANE_2OUT 192, 224, v4, v24, 3
        PROCESS_LANE_2OUT 256, 288, v5, v25, 0
        PROCESS_LANE_2OUT 320, 352, v5, v25, 1
        PROCESS_LANE_2OUT 384, 416, v5, v25, 2
        PROCESS_LANE_2OUT 448, 480, v5, v25, 3
        PROCESS_LANE_2OUT 512, 544, v6, v26, 0
        PROCESS_LANE_2OUT 576, 608, v6, v26, 1
        PROCESS_LANE_2OUT 640, 672, v6, v26, 2
        PROCESS_LANE_2OUT 704, 736, v6, v26, 3
        PROCESS_LANE_2OUT 768, 800, v7, v27, 0
        PROCESS_LANE_2OUT 832, 864, v7, v27, 1
        PROCESS_LANE_2OUT 896, 928, v7, v27, 2
        PROCESS_LANE_2OUT 960, 992, v7, v27, 3

        add     x7, x7, #1024
        add     x12, x12, x23
        add     x11, x11, x23
        subs    x16, x16, #1
        b.ne    122b

        add     x8, x8, x15
        add     x9, x9, x15
        subs    x10, x10, #1
        b.ne    121b

    120:
    .endm

//
// CONV_LOOP_MID_4OUT - Convolution loop without bounds checks that computes
// four adjacent output points per iteration. This path is used by the
// KernelFlags==0 fast path to reduce filter load pressure.
//

    .macro CONV_LOOP_MID_4OUT
        mov     x7, x4                       // Filter pointer
        mov     x8, x1                       // Input row pointer (output 0)
        mov     x6, x27                      // KernelHeight counter (x6 is scratch in this path)

        cbz     x6, 130f
        cbz     x28, 130f

    131:
        mov     x12, x8                      // Input pointer for width (output 0)
        mov     x16, x28                     // KernelWidth counter

    132:
        add     x9,  x12, x22                // Output 1 input pointer
        add     x10, x9,  x22                // Output 2 input pointer
        add     x11, x10, x22                // Output 3 input pointer

        // Load lanes 0..7 for all outputs.
        ldp     q20, q21, [x12, #0]
        ldp     q22, q23, [x9,  #0]
        ldp     q24, q25, [x10, #0]
        ldp     q26, q27, [x11, #0]

        PROCESS_LANE_4OUT 0,   32,  v20, v22, v24, v26, 0
        PROCESS_LANE_4OUT 64,  96,  v20, v22, v24, v26, 1
        PROCESS_LANE_4OUT 128, 160, v20, v22, v24, v26, 2
        PROCESS_LANE_4OUT 192, 224, v20, v22, v24, v26, 3
        PROCESS_LANE_4OUT 256, 288, v21, v23, v25, v27, 0
        PROCESS_LANE_4OUT 320, 352, v21, v23, v25, v27, 1
        PROCESS_LANE_4OUT 384, 416, v21, v23, v25, v27, 2
        PROCESS_LANE_4OUT 448, 480, v21, v23, v25, v27, 3

        // Load lanes 8..15 for all outputs.
        ldp     q20, q21, [x12, #32]
        ldp     q22, q23, [x9,  #32]
        ldp     q24, q25, [x10, #32]
        ldp     q26, q27, [x11, #32]

        PROCESS_LANE_4OUT 512, 544, v20, v22, v24, v26, 0
        PROCESS_LANE_4OUT 576, 608, v20, v22, v24, v26, 1
        PROCESS_LANE_4OUT 640, 672, v20, v22, v24, v26, 2
        PROCESS_LANE_4OUT 704, 736, v20, v22, v24, v26, 3
        PROCESS_LANE_4OUT 768, 800, v21, v23, v25, v27, 0
        PROCESS_LANE_4OUT 832, 864, v21, v23, v25, v27, 1
        PROCESS_LANE_4OUT 896, 928, v21, v23, v25, v27, 2
        PROCESS_LANE_4OUT 960, 992, v21, v23, v25, v27, 3

        add     x7, x7, #1024
        add     x12, x12, x23
        subs    x16, x16, #1
        b.ne    132b

        add     x8, x8, x15
        subs    x6, x6, #1
        b.ne    131b

    130:
    .endm

//
// void
// MlasConvNchwcFloatKernelNeonAsm(
//     const float* Input,
//     const float* Filter,
//     float* Output,
//     size_t StrideWidth,
//     size_t DilationWidth,
//     size_t FilterCount,
//     size_t InputStride,
//     size_t FilterStride,
//     size_t OutputStride,
//     size_t KernelHeight,
//     size_t KernelWidth,
//     const float* InputBase,
//     size_t InputWidth,
//     size_t DilatedInputWidth,
//     size_t OutputCountLeftPad,
//     size_t OutputCount,
//     size_t OutputCountRightPad,
//     const float* Bias,
//     unsigned KernelFlags
//     );
//

    FUNCTION_ENTRY MlasConvNchwcFloatKernelNeonAsm

    // Preserve the incoming stack pointer to access stack-passed arguments.
    mov     x9, sp

    // Prologue and callee-saved register spill.
    // Save callee-saved SIMD registers v8-v15 per AArch64 ABI.
    stp     x29, x30, [sp, #-272]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     q8, q9, [sp, #96]
    stp     q10, q11, [sp, #128]
    stp     q12, q13, [sp, #160]
    stp     q14, q15, [sp, #192]

    // Move register arguments into callee-saved registers.
    mov     x19, x0                      // Input
    mov     x20, x1                      // Filter
    mov     x21, x2                      // Output
    mov     x22, x3                      // StrideWidth (bytes)
    mov     x23, x4                      // DilationWidth (bytes)
    mov     x24, x5                      // FilterCount
    mov     x25, x7                      // FilterStride (bytes)

    // Load stack arguments using the preserved incoming stack pointer.
    ldr     x10, [x9, #0]                // OutputStride (bytes)
    ldr     x11, [x9, #8]                // KernelHeight
    ldr     x12, [x9, #16]               // KernelWidth
    ldr     x13, [x9, #24]               // InputBase
    ldr     x14, [x9, #32]               // InputWidth (bytes)
    ldr     x15, [x9, #40]               // DilatedInputWidth (bytes)
    ldr     x16, [x9, #48]               // OutputCountLeftPad
    ldr     x17, [x9, #56]               // OutputCount
    ldr     x6,  [x9, #72]               // Bias
    ldr     w8,  [x9, #80]               // KernelFlags

    // Early exit when nothing to compute.
    mov     x26, x10                     // OutputStride (bytes)
    ldr     x10, [x9, #64]               // OutputCountRightPad
    add     x0, x16, x17
    add     x0, x0, x10                  // x0 = TotalOutputCount
    cbz     x0, .Lepilogue
    cbz     x24, .Lepilogue

    // Spill the output counts so that x16/x17 can be used as scratch.
    str     x16, [sp, #224]
    str     x17, [sp, #232]
    str     x10, [sp, #240]
    str     w8,  [sp, #256]

    mov     x27, x11                     // KernelHeight
    mov     x28, x12                     // KernelWidth
    mov     x17, x6                      // Bias

    // Set up a zero vector for ReLU.
    movi    v31.4s, #0

    // x1 = current input base for the output index.
    // x2 = output offset in bytes for the output index.
    mov     x1, x19
    mov     x2, xzr

    // Fast path when no post-processing flags are enabled. This removes
    // repeated flag checks and branches from the steady-state loops.
    ldr     w9, [sp, #256]
    tst     w9, #7
    b.eq    .Lkernel_flags0

    // Process the left padded output region with bounds checks.
    ldr     x0, [sp, #224]
    cbz     x0, .Loutput_mid_begin

.Loutput_left_loop:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x6, x17

.Lfilter_left_loop:

    // Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    // Convolution loop with bounds checks.
    CONV_LOOP_PAD

    // Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    // Conditionally accumulate the existing output.
    ldr     w9, [sp, #256]
    tst     w9, #1
    b.eq    .Lskip_accumulate
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

.Lskip_accumulate:

    // Conditionally add bias.
    tst     w9, #2
    b.eq    .Lskip_bias
    ldp     q16, q17, [x6, #0]
    ldp     q18, q19, [x6, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

.Lskip_bias:

    // Conditionally apply ReLU.
    tst     w9, #4
    b.eq    .Lskip_relu
    fmax    v0.4s, v0.4s, v31.4s
    fmax    v1.4s, v1.4s, v31.4s
    fmax    v2.4s, v2.4s, v31.4s
    fmax    v3.4s, v3.4s, v31.4s

.Lskip_relu:

    // Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    // Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x6, x6, #64
    subs    x3, x3, #1
    b.ne    .Lfilter_left_loop

    // Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    .Loutput_left_loop

    // Process the middle output region without bounds checks.
.Loutput_mid_begin:
    ldr     x0, [sp, #232]
    cbz     x0, .Loutput_right_begin

    // Compute the number of output triads and spill the remaining outputs.
    mov     x12, x0
    // Use a reciprocal multiply to divide by 3 and avoid the expensive UDIV.
    movz    x16, #0xaaab
    movk    x16, #0xaaaa, lsl #16
    movk    x16, #0xaaaa, lsl #32
    movk    x16, #0xaaaa, lsl #48
    umulh   x0, x0, x16
    lsr     x0, x0, #1
    // Compute remainder = OutputCount - triads * 3.
    add     x16, x0, x0, lsl #1
    sub     x16, x12, x16
    str     x16, [sp, #248]
    cbz     x0, .Loutput_mid_pair_begin

.Loutput_mid_triad_loop:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x6, x17

.Lfilter_mid_triad_loop:

    // Clear accumulators for three output points.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b
    eor     v4.16b, v4.16b, v4.16b
    eor     v5.16b, v5.16b, v5.16b
    eor     v6.16b, v6.16b, v6.16b
    eor     v7.16b, v7.16b, v7.16b
    eor     v8.16b, v8.16b, v8.16b
    eor     v9.16b, v9.16b, v9.16b
    eor     v10.16b, v10.16b, v10.16b
    eor     v11.16b, v11.16b, v11.16b

    // Convolution loop without bounds checks computing three outputs.
    CONV_LOOP_MID_3OUT

    // Compute the output pointers for the three output points.
    add     x12, x5, x2
    add     x11, x12, #64
    add     x10, x12, #128

    // Conditionally accumulate the existing output.
    ldr     w9, [sp, #256]
    tst     w9, #1
    b.eq    .Lskip_accumulate_triad
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    ldp     q20, q21, [x11, #0]
    ldp     q22, q23, [x11, #32]
    ldp     q24, q25, [x10, #0]
    ldp     q26, q27, [x10, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s
    fadd    v4.4s, v4.4s, v20.4s
    fadd    v5.4s, v5.4s, v21.4s
    fadd    v6.4s, v6.4s, v22.4s
    fadd    v7.4s, v7.4s, v23.4s
    fadd    v8.4s, v8.4s, v24.4s
    fadd    v9.4s, v9.4s, v25.4s
    fadd    v10.4s, v10.4s, v26.4s
    fadd    v11.4s, v11.4s, v27.4s

.Lskip_accumulate_triad:

    // Conditionally add bias.
    tst     w9, #2
    b.eq    .Lskip_bias_triad
    ldp     q16, q17, [x6, #0]
    ldp     q18, q19, [x6, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s
    fadd    v4.4s, v4.4s, v16.4s
    fadd    v5.4s, v5.4s, v17.4s
    fadd    v6.4s, v6.4s, v18.4s
    fadd    v7.4s, v7.4s, v19.4s
    fadd    v8.4s, v8.4s, v16.4s
    fadd    v9.4s, v9.4s, v17.4s
    fadd    v10.4s, v10.4s, v18.4s
    fadd    v11.4s, v11.4s, v19.4s

.Lskip_bias_triad:

    // Conditionally apply ReLU.
    tst     w9, #4
    b.eq    .Lskip_relu_triad
    fmax    v0.4s, v0.4s, v31.4s
    fmax    v1.4s, v1.4s, v31.4s
    fmax    v2.4s, v2.4s, v31.4s
    fmax    v3.4s, v3.4s, v31.4s
    fmax    v4.4s, v4.4s, v31.4s
    fmax    v5.4s, v5.4s, v31.4s
    fmax    v6.4s, v6.4s, v31.4s
    fmax    v7.4s, v7.4s, v31.4s
    fmax    v8.4s, v8.4s, v31.4s
    fmax    v9.4s, v9.4s, v31.4s
    fmax    v10.4s, v10.4s, v31.4s
    fmax    v11.4s, v11.4s, v31.4s

.Lskip_relu_triad:

    // Store the results for the three output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q4, q5, [x11, #0]
    stp     q6, q7, [x11, #32]
    stp     q8, q9, [x10, #0]
    stp     q10, q11, [x10, #32]

    // Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x6, x6, #64
    subs    x3, x3, #1
    b.ne    .Lfilter_mid_triad_loop

    // Advance to the next three output indices.
    add     x1, x1, x22, lsl #1
    add     x1, x1, x22
    add     x2, x2, #192
    subs    x0, x0, #1
    b.ne    .Loutput_mid_triad_loop

.Loutput_mid_pair_begin:
    ldr     x0, [sp, #248]
    and     x16, x0, #1
    str     x16, [sp, #248]
    lsr     x0, x0, #1
    cbz     x0, .Loutput_mid_single_begin

.Loutput_mid_pair_loop:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x6, x17

.Lfilter_mid_pair_loop:

    // Clear accumulators for both output points.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b
    eor     v20.16b, v20.16b, v20.16b
    eor     v21.16b, v21.16b, v21.16b
    eor     v22.16b, v22.16b, v22.16b
    eor     v23.16b, v23.16b, v23.16b

    // Convolution loop without bounds checks computing two outputs.
    CONV_LOOP_MID_2OUT

    // Compute the output pointers for the two output points.
    add     x12, x5, x2
    add     x11, x12, #64

    // Conditionally accumulate the existing output.
    ldr     w9, [sp, #256]
    tst     w9, #1
    b.eq    .Lskip_accumulate_pair
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    ldp     q24, q25, [x11, #0]
    ldp     q26, q27, [x11, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s
    fadd    v20.4s, v20.4s, v24.4s
    fadd    v21.4s, v21.4s, v25.4s
    fadd    v22.4s, v22.4s, v26.4s
    fadd    v23.4s, v23.4s, v27.4s

.Lskip_accumulate_pair:

    // Conditionally add bias.
    tst     w9, #2
    b.eq    .Lskip_bias_pair
    ldp     q16, q17, [x6, #0]
    ldp     q18, q19, [x6, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s
    fadd    v20.4s, v20.4s, v16.4s
    fadd    v21.4s, v21.4s, v17.4s
    fadd    v22.4s, v22.4s, v18.4s
    fadd    v23.4s, v23.4s, v19.4s

.Lskip_bias_pair:

    // Conditionally apply ReLU.
    tst     w9, #4
    b.eq    .Lskip_relu_pair
    fmax    v0.4s, v0.4s, v31.4s
    fmax    v1.4s, v1.4s, v31.4s
    fmax    v2.4s, v2.4s, v31.4s
    fmax    v3.4s, v3.4s, v31.4s
    fmax    v20.4s, v20.4s, v31.4s
    fmax    v21.4s, v21.4s, v31.4s
    fmax    v22.4s, v22.4s, v31.4s
    fmax    v23.4s, v23.4s, v31.4s

.Lskip_relu_pair:

    // Store the results for the two output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q20, q21, [x11, #0]
    stp     q22, q23, [x11, #32]

    // Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x6, x6, #64
    subs    x3, x3, #1
    b.ne    .Lfilter_mid_pair_loop

    // Advance to the next two output indices.
    add     x1, x1, x22, lsl #1
    add     x2, x2, #128
    subs    x0, x0, #1
    b.ne    .Loutput_mid_pair_loop

.Loutput_mid_single_begin:
    ldr     x0, [sp, #248]
    cbz     x0, .Loutput_right_begin

.Loutput_mid_loop:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x6, x17

.Lfilter_mid_loop:

    // Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    // Convolution loop without bounds checks.
    CONV_LOOP_MID

    // Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    // Conditionally accumulate the existing output.
    ldr     w9, [sp, #256]
    tst     w9, #1
    b.eq    .Lskip_accumulate_mid
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

.Lskip_accumulate_mid:

    // Conditionally add bias.
    tst     w9, #2
    b.eq    .Lskip_bias_mid
    ldp     q16, q17, [x6, #0]
    ldp     q18, q19, [x6, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

.Lskip_bias_mid:

    // Conditionally apply ReLU.
    tst     w9, #4
    b.eq    .Lskip_relu_mid
    fmax    v0.4s, v0.4s, v31.4s
    fmax    v1.4s, v1.4s, v31.4s
    fmax    v2.4s, v2.4s, v31.4s
    fmax    v3.4s, v3.4s, v31.4s

.Lskip_relu_mid:

    // Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    // Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x6, x6, #64
    subs    x3, x3, #1
    b.ne    .Lfilter_mid_loop

    // Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    .Loutput_mid_loop

    // Process the right padded output region with bounds checks.
.Loutput_right_begin:
    ldr     x0, [sp, #240]
    cbz     x0, .Lepilogue

.Loutput_right_loop:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21
    mov     x6, x17

.Lfilter_right_loop:

    // Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    // Convolution loop with bounds checks.
    CONV_LOOP_PAD

    // Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    // Conditionally accumulate the existing output.
    ldr     w9, [sp, #256]
    tst     w9, #1
    b.eq    .Lskip_accumulate_right
    ldp     q16, q17, [x12, #0]
    ldp     q18, q19, [x12, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

.Lskip_accumulate_right:

    // Conditionally add bias.
    tst     w9, #2
    b.eq    .Lskip_bias_right
    ldp     q16, q17, [x6, #0]
    ldp     q18, q19, [x6, #32]
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s

.Lskip_bias_right:

    // Conditionally apply ReLU.
    tst     w9, #4
    b.eq    .Lskip_relu_right
    fmax    v0.4s, v0.4s, v31.4s
    fmax    v1.4s, v1.4s, v31.4s
    fmax    v2.4s, v2.4s, v31.4s
    fmax    v3.4s, v3.4s, v31.4s

.Lskip_relu_right:

    // Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    // Advance filter/output/bias pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    add     x6, x6, #64
    subs    x3, x3, #1
    b.ne    .Lfilter_right_loop

    // Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    .Loutput_right_loop

    // Skip the flag-free fast path section.
    b       .Lepilogue

.Lkernel_flags0:

    // KernelFlags == 0 fast path: no accumulation, bias, or activation.

    // Process the left padded output region with bounds checks.
    ldr     x0, [sp, #224]
    cbz     x0, .Loutput_mid_begin_flags0

.Loutput_left_loop_flags0:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

.Lfilter_left_loop_flags0:

    // Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    // Convolution loop with bounds checks.
    CONV_LOOP_PAD

    // Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    // Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    // Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    .Lfilter_left_loop_flags0

    // Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    .Loutput_left_loop_flags0

    // Process the middle output region without bounds checks.
.Loutput_mid_begin_flags0:
    ldr     x0, [sp, #232]
    cbz     x0, .Loutput_right_begin_flags0

    // Process four outputs at a time to amortize the filter loads.
    and     x16, x0, #3                  // Remainder outputs after quads.
    str     x16, [sp, #248]
    lsr     x0, x0, #2                   // Quad output count.
    cbz     x0, .Loutput_mid_remainder_begin_flags0

.Loutput_mid_quad_loop_flags0:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

.Lfilter_mid_quad_loop_flags0:

    // Clear accumulators for four output points.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b
    eor     v4.16b, v4.16b, v4.16b
    eor     v5.16b, v5.16b, v5.16b
    eor     v6.16b, v6.16b, v6.16b
    eor     v7.16b, v7.16b, v7.16b
    eor     v8.16b, v8.16b, v8.16b
    eor     v9.16b, v9.16b, v9.16b
    eor     v10.16b, v10.16b, v10.16b
    eor     v11.16b, v11.16b, v11.16b
    eor     v12.16b, v12.16b, v12.16b
    eor     v13.16b, v13.16b, v13.16b
    eor     v14.16b, v14.16b, v14.16b
    eor     v15.16b, v15.16b, v15.16b

    // Convolution loop without bounds checks computing four outputs.
    CONV_LOOP_MID_4OUT

    // Compute the output pointers for the four output points.
    add     x12, x5, x2
    add     x11, x12, #64
    add     x10, x12, #128
    add     x9,  x12, #192

    // Store the results for the four output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q4, q5, [x11, #0]
    stp     q6, q7, [x11, #32]
    stp     q8, q9, [x10, #0]
    stp     q10, q11, [x10, #32]
    stp     q12, q13, [x9, #0]
    stp     q14, q15, [x9, #32]

    // Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    .Lfilter_mid_quad_loop_flags0

    // Advance to the next four output indices.
    add     x1, x1, x22, lsl #2
    add     x2, x2, #256
    subs    x0, x0, #1
    b.ne    .Loutput_mid_quad_loop_flags0

.Loutput_mid_remainder_begin_flags0:
    ldr     x0, [sp, #248]
    cbz     x0, .Loutput_right_begin_flags0
    cmp     x0, #3
    b.ne    .Loutput_mid_remainder_not3_flags0
    // Exactly three outputs remain.
    str     xzr, [sp, #248]
    mov     x0, #1
    b       .Loutput_mid_triad_loop_flags0

.Loutput_mid_remainder_not3_flags0:
    cmp     x0, #2
    b.ne    .Loutput_mid_remainder_single_flags0
    // Exactly two outputs remain.
    str     xzr, [sp, #248]
    mov     x0, #1
    b       .Loutput_mid_pair_loop_flags0

.Loutput_mid_remainder_single_flags0:
    // Exactly one output remains.
    mov     x0, #1
    b       .Loutput_mid_loop_flags0

.Loutput_mid_triad_loop_flags0:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

.Lfilter_mid_triad_loop_flags0:

    // Clear accumulators for three output points.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b
    eor     v4.16b, v4.16b, v4.16b
    eor     v5.16b, v5.16b, v5.16b
    eor     v6.16b, v6.16b, v6.16b
    eor     v7.16b, v7.16b, v7.16b
    eor     v8.16b, v8.16b, v8.16b
    eor     v9.16b, v9.16b, v9.16b
    eor     v10.16b, v10.16b, v10.16b
    eor     v11.16b, v11.16b, v11.16b

    // Convolution loop without bounds checks computing three outputs.
    CONV_LOOP_MID_3OUT

    // Compute the output pointers for the three output points.
    add     x12, x5, x2
    add     x11, x12, #64
    add     x10, x12, #128

    // Store the results for the three output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q4, q5, [x11, #0]
    stp     q6, q7, [x11, #32]
    stp     q8, q9, [x10, #0]
    stp     q10, q11, [x10, #32]

    // Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    .Lfilter_mid_triad_loop_flags0

    // Advance to the next three output indices.
    add     x1, x1, x22, lsl #1
    add     x1, x1, x22
    add     x2, x2, #192
    subs    x0, x0, #1
    b.ne    .Loutput_mid_triad_loop_flags0

.Loutput_mid_pair_begin_flags0:
    ldr     x0, [sp, #248]
    and     x16, x0, #1
    str     x16, [sp, #248]
    lsr     x0, x0, #1
    cbz     x0, .Loutput_mid_single_begin_flags0

.Loutput_mid_pair_loop_flags0:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

.Lfilter_mid_pair_loop_flags0:

    // Clear accumulators for both output points.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b
    eor     v20.16b, v20.16b, v20.16b
    eor     v21.16b, v21.16b, v21.16b
    eor     v22.16b, v22.16b, v22.16b
    eor     v23.16b, v23.16b, v23.16b

    // Convolution loop without bounds checks computing two outputs.
    CONV_LOOP_MID_2OUT

    // Compute the output pointers for the two output points.
    add     x12, x5, x2
    add     x11, x12, #64

    // Store the results for the two output points.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]
    stp     q20, q21, [x11, #0]
    stp     q22, q23, [x11, #32]

    // Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    .Lfilter_mid_pair_loop_flags0

    // Advance to the next two output indices.
    add     x1, x1, x22, lsl #1
    add     x2, x2, #128
    subs    x0, x0, #1
    b.ne    .Loutput_mid_pair_loop_flags0

.Loutput_mid_single_begin_flags0:
    ldr     x0, [sp, #248]
    cbz     x0, .Loutput_right_begin_flags0

.Loutput_mid_loop_flags0:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

.Lfilter_mid_loop_flags0:

    // Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    // Convolution loop without bounds checks.
    CONV_LOOP_MID

    // Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    // Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    // Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    .Lfilter_mid_loop_flags0

    // Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    .Loutput_mid_loop_flags0

    // Process the right padded output region with bounds checks.
.Loutput_right_begin_flags0:
    ldr     x0, [sp, #240]
    cbz     x0, .Lepilogue

.Loutput_right_loop_flags0:

    // Initialize per-filter-set pointers and loop counter.
    mov     x3, x24
    mov     x4, x20
    mov     x5, x21

.Lfilter_right_loop_flags0:

    // Clear accumulators.
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
    eor     v2.16b, v2.16b, v2.16b
    eor     v3.16b, v3.16b, v3.16b

    // Convolution loop with bounds checks.
    CONV_LOOP_PAD

    // Compute the output pointer for this filter set and output index.
    add     x12, x5, x2

    // Store the result.
    stp     q0, q1, [x12, #0]
    stp     q2, q3, [x12, #32]

    // Advance filter/output pointers for the next filter set block.
    add     x4, x4, x25
    add     x5, x5, x26
    subs    x3, x3, #1
    b.ne    .Lfilter_right_loop_flags0

    // Advance to the next output index.
    add     x1, x1, x22
    add     x2, x2, #64
    subs    x0, x0, #1
    b.ne    .Loutput_right_loop_flags0

    b       .Lepilogue

.Lepilogue:

    // Epilogue and callee-saved register restore.
    ldp     q14, q15, [sp, #192]
    ldp     q12, q13, [sp, #160]
    ldp     q10, q11, [sp, #128]
    ldp     q8, q9, [sp, #96]
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #272
    ret

#endif // defined(__aarch64__) && !defined(_WIN32)
