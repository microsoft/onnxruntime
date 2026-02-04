/*++
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT

Module Name:
    SconvKernelNeon.S

Abstract:
    Hand written AArch64 vectorised kernel used by the convolution path
    (NCHW activations and NCHWc weights).  The kernel computes one output
    row and processes up to four 16-wide filter blocks (FilterCount <= 4).
    The hot interior loop avoids all bounds checks and the two-output path
    amortizes filter loads when possible.
--*/

#include "asmmacro.h"

        // Stack layout for parameters passed on the stack.  The first eight
        // integer parameters follow the AArch64 calling convention and are in
        // x0-x7.  Remaining parameters are spilled by the C++ caller.

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
        // Helper macros.
        // ---------------------------------------------------------------------

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

        // Post processing for a single output element working on N filter
        // blocks.  x27 points at output base for set0, x8 holds OutputStride
        // and x17 contains the Bias pointer for set0.  The zero vector is in
        // v31 and KernelFlags in w6.
        .macro POSTPROCESS_STORE N
        // accumulate from existing output if requested
        tbz     w6,#0,1f
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
        // add bias
        tbz     w6,#1,2f
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
        // optional ReLU
        tbz     w6,#2,3f
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
        // store results
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

        // Two-output helpers --------------------------------------------------
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

        // Post process helpers for the two-output interior kernel.  These are
        // written out-of-line as they are sizable and used in several places.
        .macro POSTPROCESS_STORE2_1
        add     x24,x27,#64
        // accumulate
        tbz     w6,#0,1f
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
        // bias
        tbz     w6,#1,2f
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
        tbz     w6,#2,3f
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

        .macro POSTPROCESS_STORE2_2
        POSTPROCESS_STORE2_1
        add     x24,x27,x8
        add     x24,x24,#64
        add     x27,x27,x8
        tbz     w6,#0,1f
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
        tbz     w6,#1,2f
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
        tbz     w6,#2,3f
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

        .macro POSTPROCESS_STORE2_3
        POSTPROCESS_STORE2_1
        add     x24,x27,x8
        add     x24,x24,#64
        add     x27,x27,x8
        tbz     w6,#0,1f
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
        tbz     w6,#1,2f
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
        tbz     w6,#2,3f
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
        add     x27,x27,x8       // set2 base
        add     x24,x27,#64
        tbz     w6,#0,4f
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
        tbz     w6,#1,5f
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
        tbz     w6,#2,6f
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

        // 1-output compute with padding checks ---------------------------------
        .macro CONV_PAD N
        CLEAR_ACCUM \N
        mov     x28,x1
.if \N >= 2
        add     x19,x1,x7
.endif
.if \N >= 3
        add     x24,x19,x7
.endif
.if \N >= 4
        add     x26,x24,x7
.endif
        mov     x21,x0
        ldr     x9,[sp,#.KO_KernelHeight]
        cbz     x9,5f
        ldr     x10,[sp,#.KO_KernelWidth]
        ldr     x22,[sp,#.KO_InputBase]
        ldr     x12,[sp,#.KO_InputWidth]
        add     x23,x22,x12
1:      mov     x25,x21
        mov     x11,x10
2:      cmp     x25,x22
        blo     3f
        cmp     x25,x23
        bhs     3f
        ld1r    {v24.4s},[x25]
        b       4f
3:      eor     v24.16b,v24.16b,v24.16b
4:      // Small lookahead to keep the miss machinery busy without polluting
        // cache for tiny kernels.
        prfm    pldl1keep,[x28,#192]
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x28],#64
        fmla    v0.4s,v26.4s,v24.4s
        fmla    v1.4s,v27.4s,v24.4s
        fmla    v2.4s,v28.4s,v24.4s
        fmla    v3.4s,v29.4s,v24.4s
.if \N >= 2
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x19],#64
        fmla    v4.4s,v26.4s,v24.4s
        fmla    v5.4s,v27.4s,v24.4s
        fmla    v6.4s,v28.4s,v24.4s
        fmla    v7.4s,v29.4s,v24.4s
.endif
.if \N >= 3
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x24],#64
        fmla    v8.4s,v26.4s,v24.4s
        fmla    v9.4s,v27.4s,v24.4s
        fmla    v10.4s,v28.4s,v24.4s
        fmla    v11.4s,v29.4s,v24.4s
.endif
.if \N >= 4
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x26],#64
        fmla    v12.4s,v26.4s,v24.4s
        fmla    v13.4s,v27.4s,v24.4s
        fmla    v14.4s,v28.4s,v24.4s
        fmla    v15.4s,v29.4s,v24.4s
.endif
        add     x25,x25,x4
        subs    x11,x11,#1
        b.ne    2b
        ldr     x13,[sp,#.KO_DilatedInputWidth]
        add     x21,x21,x13
        add     x22,x22,x13
        add     x23,x23,x13
        subs    x9,x9,#1
        b.ne    1b
5:      POSTPROCESS_STORE \N
        .endm

        // 1-output compute without padding checks ------------------------------
        .macro CONV_NOPAD N
        CLEAR_ACCUM \N
        mov     x28,x1
.if \N >= 2
        add     x19,x1,x7
.endif
.if \N >= 3
        add     x24,x19,x7
.endif
.if \N >= 4
        add     x26,x24,x7
.endif
        mov     x21,x0
        ldr     x9,[sp,#.KO_KernelHeight]
        cbz     x9,3f
        ldr     x10,[sp,#.KO_KernelWidth]
1:      mov     x25,x21
        mov     x11,x10
2:      ld1r    {v24.4s},[x25]
        prfm    pldl1keep,[x28,#192]
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x28],#64
        fmla    v0.4s,v26.4s,v24.4s
        fmla    v1.4s,v27.4s,v24.4s
        fmla    v2.4s,v28.4s,v24.4s
        fmla    v3.4s,v29.4s,v24.4s
.if \N >= 2
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x19],#64
        fmla    v4.4s,v26.4s,v24.4s
        fmla    v5.4s,v27.4s,v24.4s
        fmla    v6.4s,v28.4s,v24.4s
        fmla    v7.4s,v29.4s,v24.4s
.endif
.if \N >= 3
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x24],#64
        fmla    v8.4s,v26.4s,v24.4s
        fmla    v9.4s,v27.4s,v24.4s
        fmla    v10.4s,v28.4s,v24.4s
        fmla    v11.4s,v29.4s,v24.4s
.endif
.if \N >= 4
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x26],#64
        fmla    v12.4s,v26.4s,v24.4s
        fmla    v13.4s,v27.4s,v24.4s
        fmla    v14.4s,v28.4s,v24.4s
        fmla    v15.4s,v29.4s,v24.4s
.endif
        add     x25,x25,x4
        subs    x11,x11,#1
        b.ne    2b
        ldr     x13,[sp,#.KO_DilatedInputWidth]
        add     x21,x21,x13
        subs    x9,x9,#1
        b.ne    1b
3:      POSTPROCESS_STORE \N
        .endm

        // Two-output compute core when no padding is required.  x25/x26 are
        // the two input pointers, x27 the output base.  x28, x19 and x20 are
        // filter pointers for each set.  v24/v25 hold broadcast inputs.
        .macro CONV2_NOPAD N
        CLEAR_ACCUM2 \N
        mov     x28,x1
.if \N >= 2
        add     x19,x1,x7
.endif
.if \N >= 3
        add     x24,x19,x7
.endif
        ldr     x11,[sp,#.KO_KernelWidth]
        ldr     x9,[sp,#.KO_KernelHeight]
        ldr     x12,[sp,#.KO_DilatedInputWidth]
        mul     x13,x11,x4          // kernel_width * dilation_width
        sub     x12,x12,x13         // input stride between kernel rows
1:      mov     x10,x11
2:      prfm    pldl1keep,[x28,#192]
        ld1r    {v24.4s},[x25]
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x28],#64
        fmla    v0.4s,v26.4s,v24.4s
        fmla    v1.4s,v27.4s,v24.4s
        fmla    v2.4s,v28.4s,v24.4s
        fmla    v3.4s,v29.4s,v24.4s
        ld1r    {v25.4s},[x26]
        fmla    v4.4s,v26.4s,v25.4s
        fmla    v5.4s,v27.4s,v25.4s
        fmla    v6.4s,v28.4s,v25.4s
        fmla    v7.4s,v29.4s,v25.4s
.if \N >= 2
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x19],#64
        fmla    v8.4s,v26.4s,v24.4s
        fmla    v9.4s,v27.4s,v24.4s
        fmla    v10.4s,v28.4s,v24.4s
        fmla    v11.4s,v29.4s,v24.4s
        fmla    v12.4s,v26.4s,v25.4s
        fmla    v13.4s,v27.4s,v25.4s
        fmla    v14.4s,v28.4s,v25.4s
        fmla    v15.4s,v29.4s,v25.4s
.endif
.if \N >= 3
        ld1     {v26.4s,v27.4s,v28.4s,v29.4s},[x24],#64
        fmla    v16.4s,v26.4s,v24.4s
        fmla    v17.4s,v27.4s,v24.4s
        fmla    v18.4s,v28.4s,v24.4s
        fmla    v19.4s,v29.4s,v24.4s
        fmla    v20.4s,v26.4s,v25.4s
        fmla    v21.4s,v27.4s,v25.4s
        fmla    v22.4s,v28.4s,v25.4s
        fmla    v23.4s,v29.4s,v25.4s
.endif
        add     x25,x25,x4
        add     x26,x26,x4
        subs    x10,x10,#1
        b.ne    2b
        add     x25,x25,x12
        add     x26,x26,x12
        subs    x9,x9,#1
        b.ne    1b
        .endm

// -----------------------------------------------------------------------------
// void MlasConvNchwFloatKernelNeonAsm(...)
// Implements the convolution micro-kernel for the NCHW input format.
// -----------------------------------------------------------------------------

        FUNCTION_ENTRY MlasConvNchwFloatKernelNeonAsm

        // prologue -------------------------------------------------------------
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

        cmp     x5,#4
        b.ne    .LRunKernelSingle

        // Split the 4-filter case into two 2-filter passes to enable the
        // two-output inner loop without inflating register pressure.
        mov     x5,#2

        ldr     x1,[sp,#.LFrame_FilterBaseSaved]
        ldr     x2,[sp,#.LFrame_OutputBaseSaved]
        ldr     x17,[sp,#.KO_Bias]
        bl      .LKernelBody

        ldr     x1,[sp,#.LFrame_FilterBaseSaved]
        add     x1,x1,x7,lsl #1
        ldr     x2,[sp,#.LFrame_OutputBaseSaved]
        ldr     x8,[sp,#.KO_OutputStride]
        add     x2,x2,x8,lsl #1
        ldr     x17,[sp,#.KO_Bias]
        cbz     x17,1f
        add     x17,x17,#128
1:
        mov     x5,#2
        bl      .LKernelBody

        b       .LDoneEntry

.LRunKernelSingle:
        ldr     x17,[sp,#.KO_Bias]
        bl      .LKernelBody
        b       .LDoneEntry

.LKernelBody:
        // frequently used parameters
        ldr     x8,[sp,#.KO_OutputStride]
        ldr     x14,[sp,#.KO_OutputCountLeftPad]
        ldr     x15,[sp,#.KO_OutputCount]
        ldr     x16,[sp,#.KO_OutputCountRightPad]
        ldr     w6,[sp,#.KO_Flags]          // kernel flags

        eor     v31.16b,v31.16b,v31.16b

        // left padded region --------------------------------------------------
        cbz     x14,.LInterior
        mov     x20,#0
.LLeftLoop:
        ldr     x0,[sp,#.LFrame_InputBaseSaved]
        madd    x21,x20,x3,x0       // input pointer for this output
        lsl     x22,x20,#6          // (index*64)
        add     x27,x2,x22
        mov     x0,x21
        cmp     x5,#1
        b.eq    .LLeftFC1
        cmp     x5,#2
        b.eq    .LLeftFC2
        cmp     x5,#3
        b.eq    .LLeftFC3
.LLeftFC4:
        CONV_PAD 4
        b       .LLeftDoneOne
.LLeftFC3:
        CONV_PAD 3
        b       .LLeftDoneOne
.LLeftFC2:
        CONV_PAD 2
        b       .LLeftDoneOne
.LLeftFC1:
        CONV_PAD 1
.LLeftDoneOne:
        add     x20,x20,#1
        cmp     x20,x14
        blo     .LLeftLoop

        // interior region -----------------------------------------------------
.LInterior:
        cbz     x15,.LRight

        ldr     x0,[sp,#.LFrame_InputBaseSaved]
        mul     x24,x14,x3          // input byte offset after left pad
        add     x21,x0,x24          // base input for interior
        lsl     x23,x14,#6          // output offset (bytes)

        mov     x20,#0
.LIntLoop:
        sub     x12,x15,x20
        cmp     x12,#2
        b.lt    .LIntProcessOne
        // two-output path
        madd    x25,x20,x3,x21       // input for first output
        add     x26,x25,x3           // input for second output
        add     x22,x14,x20          // output index
        lsl     x22,x22,#6           // output offset (bytes)
        add     x27,x2,x22
        cmp     x5,#1
        b.eq    .LInt2FC1
        cmp     x5,#2
        b.eq    .LInt2FC2
        cmp     x5,#3
        b.eq    .LInt2FC3
        // FilterCount==4 falls back to single-output implementation due to
        // register pressure.
        b       .LIntProcessOne
.LInt2FC3:
        CONV2_NOPAD 3
        POSTPROCESS_STORE2_3
        b       .LIntNext2
.LInt2FC2:
        CONV2_NOPAD 2
        POSTPROCESS_STORE2_2
        b       .LIntNext2
.LInt2FC1:
        CONV2_NOPAD 1
        POSTPROCESS_STORE2_1
.LIntNext2:
        add     x20,x20,#2
        cmp     x20,x15
        blo     .LIntLoop
        b       .LRight

.LIntProcessOne:
        madd    x25,x20,x3,x21
        add     x22,x14,x20          // output index
        lsl     x22,x22,#6           // output offset (bytes)
        add     x27,x2,x22
        mov     x0,x25
        cmp     x5,#1
        b.eq    .LIntFC1
        cmp     x5,#2
        b.eq    .LIntFC2
        cmp     x5,#3
        b.eq    .LIntFC3
        CONV_NOPAD 4
        b       .LIntNext1
.LIntFC3:
        CONV_NOPAD 3
        b       .LIntNext1
.LIntFC2:
        CONV_NOPAD 2
        b       .LIntNext1
.LIntFC1:
        CONV_NOPAD 1
.LIntNext1:
        add     x20,x20,#1
        cmp     x20,x15
        blo     .LIntLoop

        // right padded region -------------------------------------------------
.LRight:
        cbz     x16,.LKernelReturn
        ldr     x0,[sp,#.LFrame_InputBaseSaved]
        mov     x20,#0
.LRightLoop:
        ldr     x0,[sp,#.LFrame_InputBaseSaved]
        add     x24,x20,x14
        add     x24,x24,x15
        madd    x21,x24,x3,x0
        lsl     x22,x24,#6
        add     x27,x2,x22
        mov     x0,x21
        cmp     x5,#1
        b.eq    .LRFC1
        cmp     x5,#2
        b.eq    .LRFC2
        cmp     x5,#3
        b.eq    .LRFC3
        CONV_PAD 4
        b       .LRightNext
.LRFC3:
        CONV_PAD 3
        b       .LRightNext
.LRFC2:
        CONV_PAD 2
        b       .LRightNext
.LRFC1:
        CONV_PAD 1
.LRightNext:
        add     x20,x20,#1
        cmp     x20,x16
        blo     .LRightLoop

.LKernelReturn:
        ret

.LDoneEntry:
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

        .end
