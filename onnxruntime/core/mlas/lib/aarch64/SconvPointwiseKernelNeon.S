/*++
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: MIT

Module Name:

    SconvPointwiseKernelNeon.S

Abstract:

    A hand written AArch64 vectorised micro-kernel for pointwise (1x1) convolution
    operating on tensors formatted in the NCHWc layout.  The kernel computes
    up to four output positions in parallel which allows the filter weights to
    be re-used across several outputs, greatly reducing memory bandwidth.

--*/

#include "asmmacro.h"

        .text

// Stack layout for arguments passed on the stack.  The first eight arguments
// are in x0-x7, the remaining four are placed on the stack by the caller.

        .equ    .LPW_OutputStride, 0
        .equ    .LPW_OutputCount,  8
        .equ    .LPW_Bias,         16
        .equ    .LPW_Flags,        24

// Kernel flag bits.  Keep these in sync with sconv.h.
        .equ    .LPWFlag_Accumulate, 1
        .equ    .LPWFlag_Bias,       2
        .equ    .LPWFlag_Relu,       4

// Size in bytes of one NCHWc block (16 FP32 values).
        .equ    .LPW_BlockBytes,     64

//-------------------------------------------------------------------------
//  Helper macros
//-------------------------------------------------------------------------

// Compute four outputs for a single input channel block.  The accumulators
// for the four outputs are held in v16-v31.
        .macro  CPK4_FmlaStep
        ldp     q0,q1,[x24],#32
        ldp     q2,q3,[x24],#32
        ld1r    {v4.4s},[x25],#4
        ld1r    {v5.4s},[x26],#4
        ld1r    {v6.4s},[x27],#4
        ld1r    {v7.4s},[x28],#4
        fmla    v16.4s,v0.4s,v4.4s
        fmla    v17.4s,v1.4s,v4.4s
        fmla    v18.4s,v2.4s,v4.4s
        fmla    v19.4s,v3.4s,v4.4s
        fmla    v20.4s,v0.4s,v5.4s
        fmla    v21.4s,v1.4s,v5.4s
        fmla    v22.4s,v2.4s,v5.4s
        fmla    v23.4s,v3.4s,v5.4s
        fmla    v24.4s,v0.4s,v6.4s
        fmla    v25.4s,v1.4s,v6.4s
        fmla    v26.4s,v2.4s,v6.4s
        fmla    v27.4s,v3.4s,v6.4s
        fmla    v28.4s,v0.4s,v7.4s
        fmla    v29.4s,v1.4s,v7.4s
        fmla    v30.4s,v2.4s,v7.4s
        fmla    v31.4s,v3.4s,v7.4s
        .endm

// Accumulate helper for the single output path.  The input values for the
// output are loaded to v0-v3 and each lane is multiplied with a block of
// filter coefficients.
        .macro  CPK1_FmlaWithLane lane, AReg
        ldp     q4,q5,[x24],#32
        ldp     q6,q7,[x24],#32
        fmla    v16.4s,v4.4s,\AReg\().s[\lane]
        fmla    v17.4s,v5.4s,\AReg\().s[\lane]
        fmla    v18.4s,v6.4s,\AReg\().s[\lane]
        fmla    v19.4s,v7.4s,\AReg\().s[\lane]
        .endm

// Compute a single output position.  Results are returned in v16-v19.
        .macro  CPK_ComputeOneOutput
        mov     x22,#0
        eor     v16.16b,v16.16b,v16.16b
        eor     v17.16b,v17.16b,v17.16b
        eor     v18.16b,v18.16b,v18.16b
        eor     v19.16b,v19.16b,v19.16b
.Lpw_ic_loop1:
        madd    x23,x22,x6,x15
        ldp     q0,q1,[x23]
        ldp     q2,q3,[x23,#32]
        add     x24,x17,x22,lsl #10
        CPK1_FmlaWithLane 0, v0
        CPK1_FmlaWithLane 1, v0
        CPK1_FmlaWithLane 2, v0
        CPK1_FmlaWithLane 3, v0
        CPK1_FmlaWithLane 0, v1
        CPK1_FmlaWithLane 1, v1
        CPK1_FmlaWithLane 2, v1
        CPK1_FmlaWithLane 3, v1
        CPK1_FmlaWithLane 0, v2
        CPK1_FmlaWithLane 1, v2
        CPK1_FmlaWithLane 2, v2
        CPK1_FmlaWithLane 3, v2
        CPK1_FmlaWithLane 0, v3
        CPK1_FmlaWithLane 1, v3
        CPK1_FmlaWithLane 2, v3
        CPK1_FmlaWithLane 3, v3
        add     x22,x22,#1
        cmp     x22,x4
        blt     .Lpw_ic_loop1
        .endm

//-------------------------------------------------------------------------
//  Entry point
//-------------------------------------------------------------------------

        FUNCTION_ENTRY MlasConvPointwiseFloatKernelNeonAsm

        // Load the arguments passed on the stack.
        ldr     x8,[sp,#.LPW_OutputStride]
        ldr     x9,[sp,#.LPW_OutputCount]
        ldr     x10,[sp,#.LPW_Bias]
        ldr     w11,[sp,#.LPW_Flags]

        // Preserve callee saved registers that are used by this routine.
        stp     x22,x23,[sp,#-16]!
        stp     x24,x25,[sp,#-16]!
        stp     x26,x27,[sp,#-16]!
        stp     x28,x19,[sp,#-16]!

        mov     x14,#0                     // current filter set
        cbz     x5,.Lpw_exit               // nothing to do

.Lpw_filter_loop:
        // Compute the base pointers for this filter block.
        madd    x16,x14,x8,x2              // output pointer for this filter
        madd    x17,x14,x7,x1              // filter pointer for this filter
        add     x18,x10,x14,lsl #6         // bias pointer (if used)

        mov     x15,x0                     // input base for this iteration
        mov     x20,x16                    // running output pointer
        mov     x12,x9                     // output count this iteration

        lsr     x13,x12,#2                 // number of groups of four outputs
        cbz     x13,.Lpw_process_remainder

// ------------------------------------------------------------------
//  Main loop processing 4 outputs at a time.
// ------------------------------------------------------------------
.Lpw_groups:
        // Clear accumulators for 4 outputs (16 vectors total).
        eor     v16.16b,v16.16b,v16.16b
        eor     v17.16b,v17.16b,v17.16b
        eor     v18.16b,v18.16b,v18.16b
        eor     v19.16b,v19.16b,v19.16b
        eor     v20.16b,v20.16b,v20.16b
        eor     v21.16b,v21.16b,v21.16b
        eor     v22.16b,v22.16b,v22.16b
        eor     v23.16b,v23.16b,v23.16b
        eor     v24.16b,v24.16b,v24.16b
        eor     v25.16b,v25.16b,v25.16b
        eor     v26.16b,v26.16b,v26.16b
        eor     v27.16b,v27.16b,v27.16b
        eor     v28.16b,v28.16b,v28.16b
        eor     v29.16b,v29.16b,v29.16b
        eor     v30.16b,v30.16b,v30.16b
        eor     v31.16b,v31.16b,v31.16b

        mov     x22,#0                     // current input channel block
.Lpw_ic_loop4:
        madd    x23,x22,x6,x15             // input for this block
        mov     x25,x23                    // four rows starting positions
        add     x26,x23,x3
        add     x27,x26,x3
        add     x28,x27,x3
        add     x24,x17,x22,lsl #10        // filter for this block

        // The block size is 16 so unroll 16 steps.
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep
        CPK4_FmlaStep

        add     x22,x22,#1
        cmp     x22,x4
        blt     .Lpw_ic_loop4

        // -----------------------------------------------------------------
        // Store the four outputs computed above.  There are several cases to
        // handle based on accumulation, bias and ReLU flags.
        // -----------------------------------------------------------------

        // Test if the kernel should accumulate into the existing output.
        tbz     w11,#0,.Lpw_store_nacc

        // Accumulation path.  Load bias once as it is re-used for all four
        // stores when present.
        tbz     w11,#1,1f
        ldp     q4,q5,[x18]
        ldp     q6,q7,[x18,#32]
1:
        // ---- output 0 ----
        ldp     q0,q1,[x20]
        ldp     q2,q3,[x20,#32]
        tbz     w11,#1,2f
        fadd    v0.4s,v0.4s,v4.4s
        fadd    v1.4s,v1.4s,v5.4s
        fadd    v2.4s,v2.4s,v6.4s
        fadd    v3.4s,v3.4s,v7.4s
2:
        fadd    v16.4s,v16.4s,v0.4s
        fadd    v17.4s,v17.4s,v1.4s
        fadd    v18.4s,v18.4s,v2.4s
        fadd    v19.4s,v19.4s,v3.4s
        tbz     w11,#2,3f
        eor     v0.16b,v0.16b,v0.16b
        fmax    v16.4s,v16.4s,v0.4s
        fmax    v17.4s,v17.4s,v0.4s
        fmax    v18.4s,v18.4s,v0.4s
        fmax    v19.4s,v19.4s,v0.4s
3:
        stp     q16,q17,[x20]
        stp     q18,q19,[x20,#32]

        // ---- output 1 ----
        add     x22,x20,#.LPW_BlockBytes
        ldp     q0,q1,[x22]
        ldp     q2,q3,[x22,#32]
        tbz     w11,#1,4f
        fadd    v0.4s,v0.4s,v4.4s
        fadd    v1.4s,v1.4s,v5.4s
        fadd    v2.4s,v2.4s,v6.4s
        fadd    v3.4s,v3.4s,v7.4s
4:
        fadd    v20.4s,v20.4s,v0.4s
        fadd    v21.4s,v21.4s,v1.4s
        fadd    v22.4s,v22.4s,v2.4s
        fadd    v23.4s,v23.4s,v3.4s
        tbz     w11,#2,5f
        eor     v0.16b,v0.16b,v0.16b
        fmax    v20.4s,v20.4s,v0.4s
        fmax    v21.4s,v21.4s,v0.4s
        fmax    v22.4s,v22.4s,v0.4s
        fmax    v23.4s,v23.4s,v0.4s
5:
        stp     q20,q21,[x22]
        stp     q22,q23,[x22,#32]

        // ---- output 2 ----
        add     x22,x22,#.LPW_BlockBytes
        ldp     q0,q1,[x22]
        ldp     q2,q3,[x22,#32]
        tbz     w11,#1,6f
        fadd    v0.4s,v0.4s,v4.4s
        fadd    v1.4s,v1.4s,v5.4s
        fadd    v2.4s,v2.4s,v6.4s
        fadd    v3.4s,v3.4s,v7.4s
6:
        fadd    v24.4s,v24.4s,v0.4s
        fadd    v25.4s,v25.4s,v1.4s
        fadd    v26.4s,v26.4s,v2.4s
        fadd    v27.4s,v27.4s,v3.4s
        tbz     w11,#2,7f
        eor     v0.16b,v0.16b,v0.16b
        fmax    v24.4s,v24.4s,v0.4s
        fmax    v25.4s,v25.4s,v0.4s
        fmax    v26.4s,v26.4s,v0.4s
        fmax    v27.4s,v27.4s,v0.4s
7:
        stp     q24,q25,[x22]
        stp     q26,q27,[x22,#32]

        // ---- output 3 ----
        add     x22,x22,#.LPW_BlockBytes
        ldp     q0,q1,[x22]
        ldp     q2,q3,[x22,#32]
        tbz     w11,#1,8f
        fadd    v0.4s,v0.4s,v4.4s
        fadd    v1.4s,v1.4s,v5.4s
        fadd    v2.4s,v2.4s,v6.4s
        fadd    v3.4s,v3.4s,v7.4s
8:
        fadd    v28.4s,v28.4s,v0.4s
        fadd    v29.4s,v29.4s,v1.4s
        fadd    v30.4s,v30.4s,v2.4s
        fadd    v31.4s,v31.4s,v3.4s
        tbz     w11,#2,9f
        eor     v0.16b,v0.16b,v0.16b
        fmax    v28.4s,v28.4s,v0.4s
        fmax    v29.4s,v29.4s,v0.4s
        fmax    v30.4s,v30.4s,v0.4s
        fmax    v31.4s,v31.4s,v0.4s
9:
        stp     q28,q29,[x22]
        stp     q30,q31,[x22,#32]
        b       .Lpw_advance_group

// Non-accumulating path: add bias directly to the results if requested
.Lpw_store_nacc:
        tbz     w11,#1,10f
        ldp     q4,q5,[x18]
        ldp     q6,q7,[x18,#32]
        fadd    v16.4s,v16.4s,v4.4s
        fadd    v17.4s,v17.4s,v5.4s
        fadd    v18.4s,v18.4s,v6.4s
        fadd    v19.4s,v19.4s,v7.4s
        fadd    v20.4s,v20.4s,v4.4s
        fadd    v21.4s,v21.4s,v5.4s
        fadd    v22.4s,v22.4s,v6.4s
        fadd    v23.4s,v23.4s,v7.4s
        fadd    v24.4s,v24.4s,v4.4s
        fadd    v25.4s,v25.4s,v5.4s
        fadd    v26.4s,v26.4s,v6.4s
        fadd    v27.4s,v27.4s,v7.4s
        fadd    v28.4s,v28.4s,v4.4s
        fadd    v29.4s,v29.4s,v5.4s
        fadd    v30.4s,v30.4s,v6.4s
        fadd    v31.4s,v31.4s,v7.4s
10:
        tbz     w11,#2,11f
        eor     v0.16b,v0.16b,v0.16b
        fmax    v16.4s,v16.4s,v0.4s
        fmax    v17.4s,v17.4s,v0.4s
        fmax    v18.4s,v18.4s,v0.4s
        fmax    v19.4s,v19.4s,v0.4s
        fmax    v20.4s,v20.4s,v0.4s
        fmax    v21.4s,v21.4s,v0.4s
        fmax    v22.4s,v22.4s,v0.4s
        fmax    v23.4s,v23.4s,v0.4s
        fmax    v24.4s,v24.4s,v0.4s
        fmax    v25.4s,v25.4s,v0.4s
        fmax    v26.4s,v26.4s,v0.4s
        fmax    v27.4s,v27.4s,v0.4s
        fmax    v28.4s,v28.4s,v0.4s
        fmax    v29.4s,v29.4s,v0.4s
        fmax    v30.4s,v30.4s,v0.4s
        fmax    v31.4s,v31.4s,v0.4s
11:
        stp     q16,q17,[x20]
        stp     q18,q19,[x20,#32]
        add     x22,x20,#.LPW_BlockBytes
        stp     q20,q21,[x22]
        stp     q22,q23,[x22,#32]
        add     x22,x22,#.LPW_BlockBytes
        stp     q24,q25,[x22]
        stp     q26,q27,[x22,#32]
        add     x22,x22,#.LPW_BlockBytes
        stp     q28,q29,[x22]
        stp     q30,q31,[x22,#32]

.Lpw_advance_group:
        add     x15,x15,x3,lsl #2
        add     x20,x20,#(.LPW_BlockBytes*4)
        subs    x13,x13,#1
        b.ne    .Lpw_groups

// ------------------------------------------------------------------
//  Handle the leftover (0..3) output positions.
// ------------------------------------------------------------------
.Lpw_process_remainder:
        ands    x12,x12,#3
        beq     .Lpw_after_filter
.Lpw_left_loop:
        CPK_ComputeOneOutput

        // Accumulate?
        tbz     w11,#0,.Lpw_left_noacc
        ldp     q0,q1,[x20]
        ldp     q2,q3,[x20,#32]
        tbz     w11,#1,12f
        ldp     q4,q5,[x18]
        ldp     q6,q7,[x18,#32]
        fadd    v0.4s,v0.4s,v4.4s
        fadd    v1.4s,v1.4s,v5.4s
        fadd    v2.4s,v2.4s,v6.4s
        fadd    v3.4s,v3.4s,v7.4s
12:
        fadd    v16.4s,v16.4s,v0.4s
        fadd    v17.4s,v17.4s,v1.4s
        fadd    v18.4s,v18.4s,v2.4s
        fadd    v19.4s,v19.4s,v3.4s
        tbz     w11,#2,13f
        eor     v0.16b,v0.16b,v0.16b
        fmax    v16.4s,v16.4s,v0.4s
        fmax    v17.4s,v17.4s,v0.4s
        fmax    v18.4s,v18.4s,v0.4s
        fmax    v19.4s,v19.4s,v0.4s
13:
        stp     q16,q17,[x20]
        stp     q18,q19,[x20,#32]
        b       14f

.Lpw_left_noacc:
        tbz     w11,#1,15f
        ldp     q4,q5,[x18]
        ldp     q6,q7,[x18,#32]
        fadd    v16.4s,v16.4s,v4.4s
        fadd    v17.4s,v17.4s,v5.4s
        fadd    v18.4s,v18.4s,v6.4s
        fadd    v19.4s,v19.4s,v7.4s
15:
        tbz     w11,#2,16f
        eor     v0.16b,v0.16b,v0.16b
        fmax    v16.4s,v16.4s,v0.4s
        fmax    v17.4s,v17.4s,v0.4s
        fmax    v18.4s,v18.4s,v0.4s
        fmax    v19.4s,v19.4s,v0.4s
16:
        stp     q16,q17,[x20]
        stp     q18,q19,[x20,#32]
14:
        add     x15,x15,x3
        add     x20,x20,#.LPW_BlockBytes
        subs    x12,x12,#1
        b.ne    .Lpw_left_loop

.Lpw_after_filter:
        add     x14,x14,#1
        cmp     x14,x5
        blt     .Lpw_filter_loop
.Lpw_exit:
        ldp     x28,x19,[sp],#16
        ldp     x26,x27,[sp],#16
        ldp     x24,x25,[sp],#16
        ldp     x22,x23,[sp],#16
        ret

        .end
