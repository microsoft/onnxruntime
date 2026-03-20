/*++
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT

Module Name:
    SconvNchwcKernelNeon.S

Abstract:
    Hand written AArch64 vectorised kernel used by the convolution path
    (NCHWc activations and NCHWc weights).  The kernel computes one output
    row and processes up to four 16-wide filter blocks (FilterCount <= 4).
--*/

#include "asmmacro.h"

        // Stack layout for the callee-saved register frame plus the spilled
        // tail of the convolution parameter list. The first eight integer
        // arguments arrive in x0-x7 and everything after that is read back
        // from fixed offsets relative to sp.

        .equ    .LFrame_SavedRegs,         (5*16 + 4*32 + 16 + 32)
        .equ    .LFrame_x19_x20,           0
        .equ    .LFrame_x21_x22,           16
        .equ    .LFrame_x23_x24,           32
        .equ    .LFrame_x25_x26,           48
        .equ    .LFrame_x27_x28,           64
        .equ    .LFrame_q8_q9,             80
        .equ    .LFrame_q10_q11,           112
        .equ    .LFrame_q12_q13,           144
        .equ    .LFrame_q14_q15,           176
        .equ    .LFrame_InputBaseSaved,    208
        .equ    .LFrame_FilterBaseSaved,   216
        .equ    .LFrame_OutputBaseSaved,   224
        .equ    .LFrame_LrSaved,           232

        .equ    .KO_OutputStride,          (0 + .LFrame_SavedRegs)
        .equ    .KO_KernelHeight,          (8 + .LFrame_SavedRegs)
        .equ    .KO_KernelWidth,           (16 + .LFrame_SavedRegs)
        .equ    .KO_InputBase,             (24 + .LFrame_SavedRegs)
        .equ    .KO_InputWidth,            (32 + .LFrame_SavedRegs)
        .equ    .KO_DilatedInputWidth,     (40 + .LFrame_SavedRegs)
        .equ    .KO_OutputCountLeftPad,    (48 + .LFrame_SavedRegs)
        .equ    .KO_OutputCount,           (56 + .LFrame_SavedRegs)
        .equ    .KO_OutputCountRightPad,   (64 + .LFrame_SavedRegs)
        .equ    .KO_Bias,                  (72 + .LFrame_SavedRegs)
        .equ    .KO_Flags,                 (80 + .LFrame_SavedRegs)

        .text

        // ---------------------------------------------------------------------
        // Helper macros that build the computation, postprocess, and store
        // phases used by the NCHWc kernel body.
        // ---------------------------------------------------------------------

        // Zero the accumulator vectors for a 1-4 filter-block microkernel.
        .macro CLEAR_ACCUM N
        eor     v0.16b,v0.16b,v0.16b
        eor     v1.16b,v1.16b,v1.16b
        eor     v2.16b,v2.16b,v2.16b
        eor     v3.16b,v3.16b,v3.16b
    .if \N >= 2
        eor     v4.16b,v4.16b,v4.16b
        eor     v5.16b,v5.16b,v5.16b
        eor     v6.16b,v6.16b,v6.16b
        eor     v7.16b,v7.16b,v7.16b
    .endif
    .if \N >= 3
        eor     v8.16b,v8.16b,v8.16b
        eor     v9.16b,v9.16b,v9.16b
        eor     v10.16b,v10.16b,v10.16b
        eor     v11.16b,v11.16b,v11.16b
    .endif
    .if \N >= 4
        eor     v12.16b,v12.16b,v12.16b
        eor     v13.16b,v13.16b,v13.16b
        eor     v14.16b,v14.16b,v14.16b
        eor     v15.16b,v15.16b,v15.16b
    .endif
        .endm

        // Finalize one output position for N filter blocks by optionally
        // adding the prior output, adding bias, applying ReLU, and storing.
        // x27 points at the first output block, x8 is the output stride,
        // x17 is the bias pointer, v31 is zero, and w19 holds KernelFlags.
        .macro POSTPROCESS_STORE_1OUTPUT_NFILTERS N
        tbz     w19,#0,1f
        ldr     q16,[x27]
        ldr     q17,[x27,#16]
        ldr     q18,[x27,#32]
        ldr     q19,[x27,#48]
        fadd    v0.4s,v0.4s,v16.4s
        fadd    v1.4s,v1.4s,v17.4s
        fadd    v2.4s,v2.4s,v18.4s
        fadd    v3.4s,v3.4s,v19.4s
    .if \N >= 2
        add     x24,x27,x8
        ldr     q16,[x24]
        ldr     q17,[x24,#16]
        ldr     q18,[x24,#32]
        ldr     q19,[x24,#48]
        fadd    v4.4s,v4.4s,v16.4s
        fadd    v5.4s,v5.4s,v17.4s
        fadd    v6.4s,v6.4s,v18.4s
        fadd    v7.4s,v7.4s,v19.4s
    .endif
    .if \N >= 3
        add     x24,x27,x8,lsl #1
        ldr     q16,[x24]
        ldr     q17,[x24,#16]
        ldr     q18,[x24,#32]
        ldr     q19,[x24,#48]
        fadd    v8.4s,v8.4s,v16.4s
        fadd    v9.4s,v9.4s,v17.4s
        fadd    v10.4s,v10.4s,v18.4s
        fadd    v11.4s,v11.4s,v19.4s
    .endif
    .if \N >= 4
        add     x24,x27,x8
        add     x24,x24,x8,lsl #1
        ldr     q16,[x24]
        ldr     q17,[x24,#16]
        ldr     q18,[x24,#32]
        ldr     q19,[x24,#48]
        fadd    v12.4s,v12.4s,v16.4s
        fadd    v13.4s,v13.4s,v17.4s
        fadd    v14.4s,v14.4s,v18.4s
        fadd    v15.4s,v15.4s,v19.4s
    .endif
    1:
        tbz     w19,#1,2f
        ldr     q16,[x17]
        ldr     q17,[x17,#16]
        ldr     q18,[x17,#32]
        ldr     q19,[x17,#48]
        fadd    v0.4s,v0.4s,v16.4s
        fadd    v1.4s,v1.4s,v17.4s
        fadd    v2.4s,v2.4s,v18.4s
        fadd    v3.4s,v3.4s,v19.4s
    .if \N >= 2
        add     x24,x17,#64
        ldr     q16,[x24]
        ldr     q17,[x24,#16]
        ldr     q18,[x24,#32]
        ldr     q19,[x24,#48]
        fadd    v4.4s,v4.4s,v16.4s
        fadd    v5.4s,v5.4s,v17.4s
        fadd    v6.4s,v6.4s,v18.4s
        fadd    v7.4s,v7.4s,v19.4s
    .endif
    .if \N >= 3
        add     x24,x17,#128
        ldr     q16,[x24]
        ldr     q17,[x24,#16]
        ldr     q18,[x24,#32]
        ldr     q19,[x24,#48]
        fadd    v8.4s,v8.4s,v16.4s
        fadd    v9.4s,v9.4s,v17.4s
        fadd    v10.4s,v10.4s,v18.4s
        fadd    v11.4s,v11.4s,v19.4s
    .endif
    .if \N >= 4
        add     x24,x17,#192
        ldr     q16,[x24]
        ldr     q17,[x24,#16]
        ldr     q18,[x24,#32]
        ldr     q19,[x24,#48]
        fadd    v12.4s,v12.4s,v16.4s
        fadd    v13.4s,v13.4s,v17.4s
        fadd    v14.4s,v14.4s,v18.4s
        fadd    v15.4s,v15.4s,v19.4s
    .endif
    2:
        tbz     w19,#2,3f
        fmax    v0.4s,v0.4s,v31.4s
        fmax    v1.4s,v1.4s,v31.4s
        fmax    v2.4s,v2.4s,v31.4s
        fmax    v3.4s,v3.4s,v31.4s
    .if \N >= 2
        fmax    v4.4s,v4.4s,v31.4s
        fmax    v5.4s,v5.4s,v31.4s
        fmax    v6.4s,v6.4s,v31.4s
        fmax    v7.4s,v7.4s,v31.4s
    .endif
    .if \N >= 3
        fmax    v8.4s,v8.4s,v31.4s
        fmax    v9.4s,v9.4s,v31.4s
        fmax    v10.4s,v10.4s,v31.4s
        fmax    v11.4s,v11.4s,v31.4s
    .endif
    .if \N >= 4
        fmax    v12.4s,v12.4s,v31.4s
        fmax    v13.4s,v13.4s,v31.4s
        fmax    v14.4s,v14.4s,v31.4s
        fmax    v15.4s,v15.4s,v31.4s
    .endif
    3:
        str     q0,[x27]
        str     q1,[x27,#16]
        str     q2,[x27,#32]
        str     q3,[x27,#48]
    .if \N >= 2
        add     x24,x27,x8
        str     q4,[x24]
        str     q5,[x24,#16]
        str     q6,[x24,#32]
        str     q7,[x24,#48]
    .endif
    .if \N >= 3
        add     x24,x27,x8,lsl #1
        str     q8,[x24]
        str     q9,[x24,#16]
        str     q10,[x24,#32]
        str     q11,[x24,#48]
    .endif
    .if \N >= 4
        add     x24,x27,x8
        add     x24,x24,x8,lsl #1
        str     q12,[x24]
        str     q13,[x24,#16]
        str     q14,[x24,#32]
        str     q15,[x24,#48]
    .endif
        .endm

        // Zero the wider accumulator set used when computing two outputs at
        // once for the same filter block.
        .macro CLEAR_ACCUM2 N
        eor     v0.16b,v0.16b,v0.16b
        eor     v1.16b,v1.16b,v1.16b
        eor     v2.16b,v2.16b,v2.16b
        eor     v3.16b,v3.16b,v3.16b
        eor     v4.16b,v4.16b,v4.16b
        eor     v5.16b,v5.16b,v5.16b
        eor     v6.16b,v6.16b,v6.16b
        eor     v7.16b,v7.16b,v7.16b
    .if \N >= 2
        eor     v8.16b,v8.16b,v8.16b
        eor     v9.16b,v9.16b,v9.16b
        eor     v10.16b,v10.16b,v10.16b
        eor     v11.16b,v11.16b,v11.16b
        eor     v12.16b,v12.16b,v12.16b
        eor     v13.16b,v13.16b,v13.16b
        eor     v14.16b,v14.16b,v14.16b
        eor     v15.16b,v15.16b,v15.16b
    .endif
    .if \N >= 3
        eor     v16.16b,v16.16b,v16.16b
        eor     v17.16b,v17.16b,v17.16b
        eor     v18.16b,v18.16b,v18.16b
        eor     v19.16b,v19.16b,v19.16b
        eor     v20.16b,v20.16b,v20.16b
        eor     v21.16b,v21.16b,v21.16b
        eor     v22.16b,v22.16b,v22.16b
        eor     v23.16b,v23.16b,v23.16b
    .endif
        .endm

        // Finalize and store two output positions for a single filter block.
        .macro POSTPROCESS_STORE_2OUTPUTS_FC1_NOPAD
        add     x24,x27,#64
        tbz     w19,#0,1f
        ldr     q26,[x27]
        ldr     q27,[x27,#16]
        ldr     q28,[x27,#32]
        ldr     q29,[x27,#48]
        fadd    v0.4s,v0.4s,v26.4s
        fadd    v1.4s,v1.4s,v27.4s
        fadd    v2.4s,v2.4s,v28.4s
        fadd    v3.4s,v3.4s,v29.4s
        ldr     q26,[x24]
        ldr     q27,[x24,#16]
        ldr     q28,[x24,#32]
        ldr     q29,[x24,#48]
        fadd    v4.4s,v4.4s,v26.4s
        fadd    v5.4s,v5.4s,v27.4s
        fadd    v6.4s,v6.4s,v28.4s
        fadd    v7.4s,v7.4s,v29.4s
    1:
        tbz     w19,#1,2f
        ldr     q26,[x17]
        ldr     q27,[x17,#16]
        ldr     q28,[x17,#32]
        ldr     q29,[x17,#48]
        fadd    v0.4s,v0.4s,v26.4s
        fadd    v1.4s,v1.4s,v27.4s
        fadd    v2.4s,v2.4s,v28.4s
        fadd    v3.4s,v3.4s,v29.4s
        fadd    v4.4s,v4.4s,v26.4s
        fadd    v5.4s,v5.4s,v27.4s
        fadd    v6.4s,v6.4s,v28.4s
        fadd    v7.4s,v7.4s,v29.4s
    2:
        tbz     w19,#2,3f
        fmax    v0.4s,v0.4s,v31.4s
        fmax    v1.4s,v1.4s,v31.4s
        fmax    v2.4s,v2.4s,v31.4s
        fmax    v3.4s,v3.4s,v31.4s
        fmax    v4.4s,v4.4s,v31.4s
        fmax    v5.4s,v5.4s,v31.4s
        fmax    v6.4s,v6.4s,v31.4s
        fmax    v7.4s,v7.4s,v31.4s
    3:
        str     q0,[x27]
        str     q1,[x27,#16]
        str     q2,[x27,#32]
        str     q3,[x27,#48]
        str     q4,[x24]
        str     q5,[x24,#16]
        str     q6,[x24,#32]
        str     q7,[x24,#48]
        .endm

        // Finalize and store two output positions for two filter blocks.
        .macro POSTPROCESS_STORE_2OUTPUTS_FC2_NOPAD
        POSTPROCESS_STORE_2OUTPUTS_FC1_NOPAD
        add     x24,x27,x8
        add     x24,x24,#64
        add     x27,x27,x8
        tbz     w19,#0,1f
        ldr     q26,[x27]
        ldr     q27,[x27,#16]
        ldr     q28,[x27,#32]
        ldr     q29,[x27,#48]
        fadd    v8.4s,v8.4s,v26.4s
        fadd    v9.4s,v9.4s,v27.4s
        fadd    v10.4s,v10.4s,v28.4s
        fadd    v11.4s,v11.4s,v29.4s
        ldr     q26,[x24]
        ldr     q27,[x24,#16]
        ldr     q28,[x24,#32]
        ldr     q29,[x24,#48]
        fadd    v12.4s,v12.4s,v26.4s
        fadd    v13.4s,v13.4s,v27.4s
        fadd    v14.4s,v14.4s,v28.4s
        fadd    v15.4s,v15.4s,v29.4s
    1:
        tbz     w19,#1,2f
        add     x23,x17,#64
        ldr     q26,[x23]
        ldr     q27,[x23,#16]
        ldr     q28,[x23,#32]
        ldr     q29,[x23,#48]
        fadd    v8.4s,v8.4s,v26.4s
        fadd    v9.4s,v9.4s,v27.4s
        fadd    v10.4s,v10.4s,v28.4s
        fadd    v11.4s,v11.4s,v29.4s
        fadd    v12.4s,v12.4s,v26.4s
        fadd    v13.4s,v13.4s,v27.4s
        fadd    v14.4s,v14.4s,v28.4s
        fadd    v15.4s,v15.4s,v29.4s
    2:
        tbz     w19,#2,3f
        fmax    v8.4s,v8.4s,v31.4s
        fmax    v9.4s,v9.4s,v31.4s
        fmax    v10.4s,v10.4s,v31.4s
        fmax    v11.4s,v11.4s,v31.4s
        fmax    v12.4s,v12.4s,v31.4s
        fmax    v13.4s,v13.4s,v31.4s
        fmax    v14.4s,v14.4s,v31.4s
        fmax    v15.4s,v15.4s,v31.4s
    3:
        str     q8,[x27]
        str     q9,[x27,#16]
        str     q10,[x27,#32]
        str     q11,[x27,#48]
        str     q12,[x24]
        str     q13,[x24,#16]
        str     q14,[x24,#32]
        str     q15,[x24,#48]
        sub     x27,x27,x8
        .endm

        // Finalize and store two output positions for three filter blocks.
        .macro POSTPROCESS_STORE_2OUTPUTS_FC3_NOPAD
        POSTPROCESS_STORE_2OUTPUTS_FC1_NOPAD
        add     x24,x27,x8
        add     x24,x24,#64
        add     x27,x27,x8
        tbz     w19,#0,1f
        ldr     q26,[x27]
        ldr     q27,[x27,#16]
        ldr     q28,[x27,#32]
        ldr     q29,[x27,#48]
        fadd    v8.4s,v8.4s,v26.4s
        fadd    v9.4s,v9.4s,v27.4s
        fadd    v10.4s,v10.4s,v28.4s
        fadd    v11.4s,v11.4s,v29.4s
        ldr     q26,[x24]
        ldr     q27,[x24,#16]
        ldr     q28,[x24,#32]
        ldr     q29,[x24,#48]
        fadd    v12.4s,v12.4s,v26.4s
        fadd    v13.4s,v13.4s,v27.4s
        fadd    v14.4s,v14.4s,v28.4s
        fadd    v15.4s,v15.4s,v29.4s
    1:
        tbz     w19,#1,2f
        add     x23,x17,#64
        ldr     q26,[x23]
        ldr     q27,[x23,#16]
        ldr     q28,[x23,#32]
        ldr     q29,[x23,#48]
        fadd    v8.4s,v8.4s,v26.4s
        fadd    v9.4s,v9.4s,v27.4s
        fadd    v10.4s,v10.4s,v28.4s
        fadd    v11.4s,v11.4s,v29.4s
        fadd    v12.4s,v12.4s,v26.4s
        fadd    v13.4s,v13.4s,v27.4s
        fadd    v14.4s,v14.4s,v28.4s
        fadd    v15.4s,v15.4s,v29.4s
    2:
        tbz     w19,#2,3f
        fmax    v8.4s,v8.4s,v31.4s
        fmax    v9.4s,v9.4s,v31.4s
        fmax    v10.4s,v10.4s,v31.4s
        fmax    v11.4s,v11.4s,v31.4s
        fmax    v12.4s,v12.4s,v31.4s
        fmax    v13.4s,v13.4s,v31.4s
        fmax    v14.4s,v14.4s,v31.4s
        fmax    v15.4s,v15.4s,v31.4s
    3:
        str     q8,[x27]
        str     q9,[x27,#16]
        str     q10,[x27,#32]
        str     q11,[x27,#48]
        str     q12,[x24]
        str     q13,[x24,#16]
        str     q14,[x24,#32]
        str     q15,[x24,#48]
        add     x27,x27,x8
        add     x24,x27,#64
        tbz     w19,#0,4f
        ldr     q26,[x27]
        ldr     q27,[x27,#16]
        ldr     q28,[x27,#32]
        ldr     q29,[x27,#48]
        fadd    v16.4s,v16.4s,v26.4s
        fadd    v17.4s,v17.4s,v27.4s
        fadd    v18.4s,v18.4s,v28.4s
        fadd    v19.4s,v19.4s,v29.4s
        ldr     q26,[x24]
        ldr     q27,[x24,#16]
        ldr     q28,[x24,#32]
        ldr     q29,[x24,#48]
        fadd    v20.4s,v20.4s,v26.4s
        fadd    v21.4s,v21.4s,v27.4s
        fadd    v22.4s,v22.4s,v28.4s
        fadd    v23.4s,v23.4s,v29.4s
    4:
        tbz     w19,#1,5f
        add     x23,x17,#128
        ldr     q26,[x23]
        ldr     q27,[x23,#16]
        ldr     q28,[x23,#32]
        ldr     q29,[x23,#48]
        fadd    v16.4s,v16.4s,v26.4s
        fadd    v17.4s,v17.4s,v27.4s
        fadd    v18.4s,v18.4s,v28.4s
        fadd    v19.4s,v19.4s,v29.4s
        fadd    v20.4s,v20.4s,v26.4s
        fadd    v21.4s,v21.4s,v27.4s
        fadd    v22.4s,v22.4s,v28.4s
        fadd    v23.4s,v23.4s,v29.4s
    5:
        tbz     w19,#2,6f
        fmax    v16.4s,v16.4s,v31.4s
        fmax    v17.4s,v17.4s,v31.4s
        fmax    v18.4s,v18.4s,v31.4s
        fmax    v19.4s,v19.4s,v31.4s
        fmax    v20.4s,v20.4s,v31.4s
        fmax    v21.4s,v21.4s,v31.4s
        fmax    v22.4s,v22.4s,v31.4s
        fmax    v23.4s,v23.4s,v31.4s
    6:
        str     q16,[x27]
        str     q17,[x27,#16]
        str     q18,[x27,#32]
        str     q19,[x27,#48]
        str     q20,[x24]
        str     q21,[x24,#16]
        str     q22,[x24,#32]
        str     q23,[x24,#48]
        sub     x27,x27,x8
        sub     x27,x27,x8
        .endm

        // Consume one 16x16 weight row and accumulate it into one output
        // position using a selected lane from the loaded activation row.
        .macro FMLA_ROW AReg, Lane
        ld1     {v24.4s,v25.4s,v26.4s,v27.4s},[x28],#64
        fmla    v0.4s,v24.4s,\AReg\().s[\Lane]
        fmla    v1.4s,v25.4s,\AReg\().s[\Lane]
        fmla    v2.4s,v26.4s,\AReg\().s[\Lane]
        fmla    v3.4s,v27.4s,\AReg\().s[\Lane]
        .endm

        // Consume one 16x16 weight row and accumulate it into two output
        // positions, reusing the same filter load for both outputs.
        .macro FMLA_ROW2 AReg0, Lane0, AReg1, Lane1
        ld1     {v24.4s,v25.4s,v26.4s,v27.4s},[x28],#64
        fmla    v0.4s,v24.4s,\AReg0\().s[\Lane0]
        fmla    v1.4s,v25.4s,\AReg0\().s[\Lane0]
        fmla    v2.4s,v26.4s,\AReg0\().s[\Lane0]
        fmla    v3.4s,v27.4s,\AReg0\().s[\Lane0]
        fmla    v4.4s,v24.4s,\AReg1\().s[\Lane1]
        fmla    v5.4s,v25.4s,\AReg1\().s[\Lane1]
        fmla    v6.4s,v26.4s,\AReg1\().s[\Lane1]
        fmla    v7.4s,v27.4s,\AReg1\().s[\Lane1]
        .endm

        // Consume one 16x16 weight row for two filter blocks and accumulate it
        // into one output position.
        .macro FMLA_ROW_FC2 AReg, Lane
        ld1     {v24.4s,v25.4s,v26.4s,v27.4s},[x28],#64
        fmla    v0.4s,v24.4s,\AReg\().s[\Lane]
        fmla    v1.4s,v25.4s,\AReg\().s[\Lane]
        fmla    v2.4s,v26.4s,\AReg\().s[\Lane]
        fmla    v3.4s,v27.4s,\AReg\().s[\Lane]
        ld1     {v24.4s,v25.4s,v26.4s,v27.4s},[x13],#64
        fmla    v4.4s,v24.4s,\AReg\().s[\Lane]
        fmla    v5.4s,v25.4s,\AReg\().s[\Lane]
        fmla    v6.4s,v26.4s,\AReg\().s[\Lane]
        fmla    v7.4s,v27.4s,\AReg\().s[\Lane]
        .endm

        // Consume one 16x16 weight row for two filter blocks and accumulate it
        // into two output positions.
        .macro FMLA_ROW2_FC2 AReg0, Lane0, AReg1, Lane1
        ld1     {v24.4s,v25.4s,v26.4s,v27.4s},[x28],#64
        fmla    v0.4s,v24.4s,\AReg0\().s[\Lane0]
        fmla    v1.4s,v25.4s,\AReg0\().s[\Lane0]
        fmla    v2.4s,v26.4s,\AReg0\().s[\Lane0]
        fmla    v3.4s,v27.4s,\AReg0\().s[\Lane0]
        fmla    v4.4s,v24.4s,\AReg1\().s[\Lane1]
        fmla    v5.4s,v25.4s,\AReg1\().s[\Lane1]
        fmla    v6.4s,v26.4s,\AReg1\().s[\Lane1]
        fmla    v7.4s,v27.4s,\AReg1\().s[\Lane1]
        ld1     {v24.4s,v25.4s,v26.4s,v27.4s},[x13],#64
        fmla    v8.4s,v24.4s,\AReg0\().s[\Lane0]
        fmla    v9.4s,v25.4s,\AReg0\().s[\Lane0]
        fmla    v10.4s,v26.4s,\AReg0\().s[\Lane0]
        fmla    v11.4s,v27.4s,\AReg0\().s[\Lane0]
        fmla    v12.4s,v24.4s,\AReg1\().s[\Lane1]
        fmla    v13.4s,v25.4s,\AReg1\().s[\Lane1]
        fmla    v14.4s,v26.4s,\AReg1\().s[\Lane1]
        fmla    v15.4s,v27.4s,\AReg1\().s[\Lane1]
        .endm

        // Compute one output position for the interior no-padding case. The
        // outer loop walks kernel rows, the inner loop walks kernel columns,
        // and each iteration multiplies one 16-lane input block by one 16x16
        // filter tile before handing off to the single-output postprocess path.
        .macro CONV_NCHWC_FC1_1OUTPUT_NOPAD
        CLEAR_ACCUM 1
        mov     x28,x1
        mov     x21,x0
        ldr     x9,[sp,#.KO_KernelHeight]
        cbz     x9,3f
        ldr     x10,[sp,#.KO_KernelWidth]
    1:      mov     x25,x21
        mov     x11,x10
    2:
        ldp     q20,q21,[x25]
        ldp     q22,q23,[x25,#32]
        FMLA_ROW v20,0
        FMLA_ROW v20,1
        FMLA_ROW v20,2
        FMLA_ROW v20,3
        FMLA_ROW v21,0
        FMLA_ROW v21,1
        FMLA_ROW v21,2
        FMLA_ROW v21,3
        FMLA_ROW v22,0
        FMLA_ROW v22,1
        FMLA_ROW v22,2
        FMLA_ROW v22,3
        FMLA_ROW v23,0
        FMLA_ROW v23,1
        FMLA_ROW v23,2
        FMLA_ROW v23,3
        add     x25,x25,x4
        subs    x11,x11,#1
        b.ne    2b
        add     x21,x25,x6
        subs    x9,x9,#1
        b.ne    1b
    3:      POSTPROCESS_STORE_1OUTPUT_NFILTERS 1
        .endm

        // Compute one output position for two filter blocks in the interior
        // no-padding case.
        .macro CONV_NCHWC_FC2_1OUTPUT_NOPAD
        CLEAR_ACCUM 2
        mov     x28,x1
        add     x13,x1,x7
        mov     x21,x0
        ldr     x9,[sp,#.KO_KernelHeight]
        cbz     x9,3f
        ldr     x10,[sp,#.KO_KernelWidth]
    1:  mov     x25,x21
        mov     x11,x10
    2:
        ldp     q20,q21,[x25]
        ldp     q22,q23,[x25,#32]
        FMLA_ROW_FC2 v20,0
        FMLA_ROW_FC2 v20,1
        FMLA_ROW_FC2 v20,2
        FMLA_ROW_FC2 v20,3
        FMLA_ROW_FC2 v21,0
        FMLA_ROW_FC2 v21,1
        FMLA_ROW_FC2 v21,2
        FMLA_ROW_FC2 v21,3
        FMLA_ROW_FC2 v22,0
        FMLA_ROW_FC2 v22,1
        FMLA_ROW_FC2 v22,2
        FMLA_ROW_FC2 v22,3
        FMLA_ROW_FC2 v23,0
        FMLA_ROW_FC2 v23,1
        FMLA_ROW_FC2 v23,2
        FMLA_ROW_FC2 v23,3
        add     x25,x25,x4
        subs    x11,x11,#1
        b.ne    2b
        add     x21,x25,x6
        subs    x9,x9,#1
        b.ne    1b
    3:  POSTPROCESS_STORE_1OUTPUT_NFILTERS 2
        .endm

        // Compute two adjacent output positions together for the interior
        // no-padding case. The loops are the same as the single-output form,
        // but each filter load is shared across two activation positions to
        // improve throughput before the paired postprocess/store step.
        .macro CONV_NCHWC_FC1_2OUTPUTS_NOPAD
        CLEAR_ACCUM2 1
        mov     x28,x1
        mov     x21,x0
        add     x22,x0,x3
        ldr     x9,[sp,#.KO_KernelHeight]
        cbz     x9,3f
        ldr     x10,[sp,#.KO_KernelWidth]
    1:      mov     x25,x21
        mov     x26,x22
        mov     x11,x10
    2:
        ldp     q16,q17,[x25]
        ldp     q18,q19,[x25,#32]
        ldp     q20,q21,[x26]
        ldp     q22,q23,[x26,#32]
        FMLA_ROW2 v16,0,v20,0
        FMLA_ROW2 v16,1,v20,1
        FMLA_ROW2 v16,2,v20,2
        FMLA_ROW2 v16,3,v20,3
        FMLA_ROW2 v17,0,v21,0
        FMLA_ROW2 v17,1,v21,1
        FMLA_ROW2 v17,2,v21,2
        FMLA_ROW2 v17,3,v21,3
        FMLA_ROW2 v18,0,v22,0
        FMLA_ROW2 v18,1,v22,1
        FMLA_ROW2 v18,2,v22,2
        FMLA_ROW2 v18,3,v22,3
        FMLA_ROW2 v19,0,v23,0
        FMLA_ROW2 v19,1,v23,1
        FMLA_ROW2 v19,2,v23,2
        FMLA_ROW2 v19,3,v23,3
        add     x25,x25,x4
        add     x26,x26,x4
        subs    x11,x11,#1
        b.ne    2b
        add     x21,x25,x6
        add     x22,x26,x6
        subs    x9,x9,#1
        b.ne    1b
    3:      POSTPROCESS_STORE_2OUTPUTS_FC1_NOPAD
        .endm

        // Compute two adjacent output positions together for two filter blocks
        // in the interior no-padding case.
        .macro CONV_NCHWC_FC2_2OUTPUTS_NOPAD
        CLEAR_ACCUM2 2
        mov     x28,x1
        add     x13,x1,x7
        mov     x21,x0
        add     x22,x0,x3
        ldr     x9,[sp,#.KO_KernelHeight]
        cbz     x9,3f
        ldr     x10,[sp,#.KO_KernelWidth]
    1:  mov     x25,x21
        mov     x26,x22
        mov     x11,x10
    2:
        ldp     q16,q17,[x25]
        ldp     q18,q19,[x25,#32]
        ldp     q20,q21,[x26]
        ldp     q22,q23,[x26,#32]
        FMLA_ROW2_FC2 v16,0,v20,0
        FMLA_ROW2_FC2 v16,1,v20,1
        FMLA_ROW2_FC2 v16,2,v20,2
        FMLA_ROW2_FC2 v16,3,v20,3
        FMLA_ROW2_FC2 v17,0,v21,0
        FMLA_ROW2_FC2 v17,1,v21,1
        FMLA_ROW2_FC2 v17,2,v21,2
        FMLA_ROW2_FC2 v17,3,v21,3
        FMLA_ROW2_FC2 v18,0,v22,0
        FMLA_ROW2_FC2 v18,1,v22,1
        FMLA_ROW2_FC2 v18,2,v22,2
        FMLA_ROW2_FC2 v18,3,v22,3
        FMLA_ROW2_FC2 v19,0,v23,0
        FMLA_ROW2_FC2 v19,1,v23,1
        FMLA_ROW2_FC2 v19,2,v23,2
        FMLA_ROW2_FC2 v19,3,v23,3
        add     x25,x25,x4
        add     x26,x26,x4
        subs    x11,x11,#1
        b.ne    2b
        add     x21,x25,x6
        add     x22,x26,x6
        subs    x9,x9,#1
        b.ne    1b
    3:  POSTPROCESS_STORE_2OUTPUTS_FC2_NOPAD
        .endm

        // Restore the nonvolatile general-purpose and vector registers and
        // return to the C++ caller.
        .macro RESTORE_NONVOLATILE_AND_RETURN
        ldp     q14,q15,[sp,#.LFrame_q14_q15]
        ldp     q12,q13,[sp,#.LFrame_q12_q13]
        ldp     q10,q11,[sp,#.LFrame_q10_q11]
        ldp     q8,q9,[sp,#.LFrame_q8_q9]
        ldp     x27,x28,[sp,#.LFrame_x27_x28]
        ldp     x25,x26,[sp,#.LFrame_x25_x26]
        ldp     x23,x24,[sp,#.LFrame_x23_x24]
        ldp     x21,x22,[sp,#.LFrame_x21_x22]
        ldr     x30,[sp,#.LFrame_LrSaved]
        ldp     x19,x20,[sp],#.LFrame_SavedRegs
        ret
        .endm

// -----------------------------------------------------------------------------
// void MlasConvNchwcFloatKernelNeonAsm(...)
// Implements the convolution micro-kernel for the NCHWc input format.
// -----------------------------------------------------------------------------

        FUNCTION_ENTRY MlasConvNchwcFloatKernelNeonAsm

        stp     x19,x20,[sp,#-.LFrame_SavedRegs]!
        stp     x21,x22,[sp,#.LFrame_x21_x22]
        stp     x23,x24,[sp,#.LFrame_x23_x24]
        stp     x25,x26,[sp,#.LFrame_x25_x26]
        stp     x27,x28,[sp,#.LFrame_x27_x28]
        stp     q8,q9,[sp,#.LFrame_q8_q9]
        stp     q10,q11,[sp,#.LFrame_q10_q11]
        stp     q12,q13,[sp,#.LFrame_q12_q13]
        stp     q14,q15,[sp,#.LFrame_q14_q15]
        str     x0,[sp,#.LFrame_InputBaseSaved]
        str     x1,[sp,#.LFrame_FilterBaseSaved]
        str     x2,[sp,#.LFrame_OutputBaseSaved]
        str     x30,[sp,#.LFrame_LrSaved]

        ldr     x1,[sp,#.LFrame_FilterBaseSaved]
        ldr     x2,[sp,#.LFrame_OutputBaseSaved]
        ldr     x8,[sp,#.KO_OutputStride]
        ldr     x14,[sp,#.KO_OutputCountLeftPad]
        ldr     x15,[sp,#.KO_OutputCount]
        ldr     x16,[sp,#.KO_OutputCountRightPad]
        ldr     x17,[sp,#.KO_Bias]
        ldr     w19,[sp,#.KO_Flags]

        eor     v31.16b,v31.16b,v31.16b

        cmp     x5,#1
        b.eq    .LFilterCountSupported
        cmp     x5,#2
        b.eq    .LFilterCountSupported
        cmp     x5,#3
        b.eq    .LFilterCountSupportedFC3
        cmp     x5,#4
        b.ne    .LKernelExit

        cbnz    x14,.LKernelExit
        cbnz    x16,.LKernelExit
        cbz     x15,.LKernelExit

        mov     x20,#0

    .LNoPadLoopFC4Pass1:
        sub     x12,x15,x20
        cmp     x12,#2
        b.lt    .LNoPadTailFC4Pass1
        ldr     x21,[sp,#.LFrame_InputBaseSaved]
        madd    x0,x20,x3,x21
        lsl     x23,x20,#6
        add     x27,x2,x23
        CONV_NCHWC_FC2_2OUTPUTS_NOPAD
        add     x20,x20,#2
        cmp     x20,x15
        blo     .LNoPadLoopFC4Pass1
        b       .LStartFC4Pass2

    .LNoPadTailFC4Pass1:
        ldr     x21,[sp,#.LFrame_InputBaseSaved]
        madd    x0,x20,x3,x21
        lsl     x23,x20,#6
        add     x27,x2,x23
        CONV_NCHWC_FC2_1OUTPUT_NOPAD
        add     x20,x20,#1
        cmp     x20,x15
        blo     .LNoPadLoopFC4Pass1

    .LStartFC4Pass2:
        ldr     x1,[sp,#.LFrame_FilterBaseSaved]
        add     x1,x1,x7
        add     x1,x1,x7
        ldr     x2,[sp,#.LFrame_OutputBaseSaved]
        add     x2,x2,x8
        add     x2,x2,x8
        add     x17,x17,#128
        mov     x20,#0

    .LNoPadLoopFC4Pass2:
        sub     x12,x15,x20
        cmp     x12,#2
        b.lt    .LNoPadTailFC4Pass2
        ldr     x21,[sp,#.LFrame_InputBaseSaved]
        madd    x0,x20,x3,x21
        lsl     x23,x20,#6
        add     x27,x2,x23
        CONV_NCHWC_FC2_2OUTPUTS_NOPAD
        add     x20,x20,#2
        cmp     x20,x15
        blo     .LNoPadLoopFC4Pass2
        b       .LKernelExit

    .LNoPadTailFC4Pass2:
        ldr     x21,[sp,#.LFrame_InputBaseSaved]
        madd    x0,x20,x3,x21
        lsl     x23,x20,#6
        add     x27,x2,x23
        CONV_NCHWC_FC2_1OUTPUT_NOPAD
        add     x20,x20,#1
        cmp     x20,x15
        blo     .LNoPadLoopFC4Pass2
        b       .LKernelExit

    .LFilterCountSupportedFC3:

        cbnz    x14,.LKernelExit
        cbnz    x16,.LKernelExit
        cbz     x15,.LKernelExit

        mov     x20,#0

    .LNoPadLoopFC3Pass1:
        sub     x12,x15,x20
        cmp     x12,#2
        b.lt    .LNoPadTailFC3Pass1
        ldr     x21,[sp,#.LFrame_InputBaseSaved]
        madd    x0,x20,x3,x21
        lsl     x23,x20,#6
        add     x27,x2,x23
        CONV_NCHWC_FC2_2OUTPUTS_NOPAD
        add     x20,x20,#2
        cmp     x20,x15
        blo     .LNoPadLoopFC3Pass1
        b       .LStartFC3Pass2

    .LNoPadTailFC3Pass1:
        ldr     x21,[sp,#.LFrame_InputBaseSaved]
        madd    x0,x20,x3,x21
        lsl     x23,x20,#6
        add     x27,x2,x23
        CONV_NCHWC_FC2_1OUTPUT_NOPAD
        add     x20,x20,#1
        cmp     x20,x15
        blo     .LNoPadLoopFC3Pass1

    .LStartFC3Pass2:
        ldr     x1,[sp,#.LFrame_FilterBaseSaved]
        add     x1,x1,x7
        add     x1,x1,x7
        ldr     x2,[sp,#.LFrame_OutputBaseSaved]
        add     x2,x2,x8
        add     x2,x2,x8
        add     x17,x17,#128
        mov     x20,#0

    .LNoPadLoopFC3Pass2:
        sub     x12,x15,x20
        cmp     x12,#2
        b.lt    .LNoPadTailFC3Pass2
        ldr     x21,[sp,#.LFrame_InputBaseSaved]
        madd    x0,x20,x3,x21
        lsl     x23,x20,#6
        add     x27,x2,x23
        CONV_NCHWC_FC1_2OUTPUTS_NOPAD
        add     x20,x20,#2
        cmp     x20,x15
        blo     .LNoPadLoopFC3Pass2
        b       .LKernelExit

    .LNoPadTailFC3Pass2:
        ldr     x21,[sp,#.LFrame_InputBaseSaved]
        madd    x0,x20,x3,x21
        lsl     x23,x20,#6
        add     x27,x2,x23
        CONV_NCHWC_FC1_1OUTPUT_NOPAD
        add     x20,x20,#1
        cmp     x20,x15
        blo     .LNoPadLoopFC3Pass2
        b       .LKernelExit

    .LFilterCountSupported:
        cbnz    x14,.LKernelExit
        cbnz    x16,.LKernelExit
        cbz     x15,.LKernelExit

        mov     x20,#0

    .LNoPadLoop1:
        sub     x12,x15,x20
        cmp     x12,#2
        b.lt    .LNoPadTail1
        ldr     x21,[sp,#.LFrame_InputBaseSaved]
        madd    x0,x20,x3,x21
        lsl     x23,x20,#6
        add     x27,x2,x23
        cmp     x5,#1
        b.eq    .LNoPad2FC1
        CONV_NCHWC_FC2_2OUTPUTS_NOPAD
        b       .LNoPad2Done

    .LNoPad2FC1:
        CONV_NCHWC_FC1_2OUTPUTS_NOPAD

    .LNoPad2Done:
        add     x20,x20,#2
        cmp     x20,x15
        blo     .LNoPadLoop1
        b       .LKernelExit

    .LNoPadTail1:
        ldr     x21,[sp,#.LFrame_InputBaseSaved]
        madd    x0,x20,x3,x21
        lsl     x23,x20,#6
        add     x27,x2,x23
        cmp     x5,#1
        b.eq    .LNoPad1FC1
        CONV_NCHWC_FC2_1OUTPUT_NOPAD
        b       .LNoPad1Done

    .LNoPad1FC1:
        CONV_NCHWC_FC1_1OUTPUT_NOPAD

    .LNoPad1Done:
        add     x20,x20,#1
        cmp     x20,x15
        blo     .LNoPadLoop1

    .LKernelExit:
        RESTORE_NONVOLATILE_AND_RETURN

        .end
