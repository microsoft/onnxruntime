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

// Kernel flag bits.  Keep these in sync with sconv_nchwc_kernel_neon.h.
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
        ldp     q0,q1,[x0],#32
        ldp     q2,q3,[x0],#32
        ld1r    {v4.4s},[x1],#4
        ld1r    {v5.4s},[x2],#4
        ld1r    {v6.4s},[x3],#4
        ld1r    {v7.4s},[x4],#4
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
        ldp     q4,q5,[x0],#32
        ldp     q6,q7,[x0],#32
        fmla    v16.4s,v4.4s,\AReg\().s[\lane]
        fmla    v17.4s,v5.4s,\AReg\().s[\lane]
        fmla    v18.4s,v6.4s,\AReg\().s[\lane]
        fmla    v19.4s,v7.4s,\AReg\().s[\lane]
        .endm

// Compute a single output position.  Results are returned in v16-v19.
        .macro  CPK_ComputeOneOutput
        mov     x5,#0
        eor     v16.16b,v16.16b,v16.16b
        eor     v17.16b,v17.16b,v17.16b
        eor     v18.16b,v18.16b,v18.16b
        eor     v19.16b,v19.16b,v19.16b
.Lpw_ic_loop1:
        madd    x1,x5,x7,x15
        ldp     q0,q1,[x1]
        ldp     q2,q3,[x1,#32]
        add     x0,x17,x5,lsl #10
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
        add     x5,x5,#1
        cmp     x5,x9
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

        // Spill base arguments so caller-saved registers can be reused freely.
        sub     sp,sp,#96
        stp     x0,x1,[sp,#0]
        stp     x2,x3,[sp,#16]
        stp     x4,x5,[sp,#32]
        stp     x6,x7,[sp,#48]
        str     x10,[sp,#64]               // bias base
        str     x9,[sp,#72]                // output count

        mov     x12,#0                     // current filter set
        cbz     x5,.Lpw_exit               // nothing to do

.Lpw_filter_loop:
        // Compute the base pointers for this filter block.
        ldr     x15,[sp,#0]                // input base
        ldr     x16,[sp,#16]               // output base
        madd    x16,x12,x8,x16             // output pointer for this filter
        ldr     x17,[sp,#8]                // filter base
        ldr     x0,[sp,#56]                // filter set stride
        madd    x17,x12,x0,x17             // filter pointer for this filter
        ldr     x10,[sp,#64]               // bias base
        add     x10,x10,x12,lsl #6         // bias pointer (if used)
        ldr     x6,[sp,#24]                // input row stride
        ldr     x7,[sp,#48]                // input channel stride
        ldr     x13,[sp,#72]               // output count
        lsr     x14,x13,#2                 // number of groups of four outputs
        and     x13,x13,#3                 // remaining outputs
        ldr     x9,[sp,#32]                // input channel blocks
        cbz     x14,.Lpw_process_remainder

// ------------------------------------------------------------------
//  Main loop processing 4 outputs at a time.
// ------------------------------------------------------------------
        .p2align 4
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

        mov     x5,#0                      // current input channel block
.Lpw_ic_loop4:
        madd    x1,x5,x7,x15               // input for this block
        add     x2,x1,x6                   // four rows starting positions
        add     x3,x2,x6
        add     x4,x3,x6
        add     x0,x17,x5,lsl #10          // filter for this block

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

        add     x5,x5,#1
        cmp     x5,x9
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
        ldp     q4,q5,[x10]
        ldp     q6,q7,[x10,#32]
1:
        // ---- output 0 ----
        ldp     q0,q1,[x16]
        ldp     q2,q3,[x16,#32]
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
        stp     q16,q17,[x16]
        stp     q18,q19,[x16,#32]

        // ---- output 1 ----
        add     x0,x16,#.LPW_BlockBytes
        ldp     q0,q1,[x0]
        ldp     q2,q3,[x0,#32]
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
        stp     q20,q21,[x0]
        stp     q22,q23,[x0,#32]

        // ---- output 2 ----
        add     x0,x0,#.LPW_BlockBytes
        ldp     q0,q1,[x0]
        ldp     q2,q3,[x0,#32]
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
        stp     q24,q25,[x0]
        stp     q26,q27,[x0,#32]

        // ---- output 3 ----
        add     x0,x0,#.LPW_BlockBytes
        ldp     q0,q1,[x0]
        ldp     q2,q3,[x0,#32]
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
        stp     q28,q29,[x0]
        stp     q30,q31,[x0,#32]
        b       .Lpw_advance_group

// Non-accumulating path: add bias directly to the results if requested
.Lpw_store_nacc:
        tbz     w11,#1,10f
        ldp     q4,q5,[x10]
        ldp     q6,q7,[x10,#32]
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
        stp     q16,q17,[x16]
        stp     q18,q19,[x16,#32]
        add     x0,x16,#.LPW_BlockBytes
        stp     q20,q21,[x0]
        stp     q22,q23,[x0,#32]
        add     x0,x0,#.LPW_BlockBytes
        stp     q24,q25,[x0]
        stp     q26,q27,[x0,#32]
        add     x0,x0,#.LPW_BlockBytes
        stp     q28,q29,[x0]
        stp     q30,q31,[x0,#32]

.Lpw_advance_group:
        add     x15,x15,x6,lsl #2
        add     x16,x16,#(.LPW_BlockBytes*4)
        subs    x14,x14,#1
        b.ne    .Lpw_groups

// ------------------------------------------------------------------
//  Handle the leftover (0..3) output positions.
// ------------------------------------------------------------------
.Lpw_process_remainder:
        cbz     x13,.Lpw_after_filter
.Lpw_left_loop:
        CPK_ComputeOneOutput

        // Accumulate?
        tbz     w11,#0,.Lpw_left_noacc
        ldp     q0,q1,[x16]
        ldp     q2,q3,[x16,#32]
        tbz     w11,#1,12f
        ldp     q4,q5,[x10]
        ldp     q6,q7,[x10,#32]
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
        stp     q16,q17,[x16]
        stp     q18,q19,[x16,#32]
        b       14f

.Lpw_left_noacc:
        tbz     w11,#1,15f
        ldp     q4,q5,[x10]
        ldp     q6,q7,[x10,#32]
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
        stp     q16,q17,[x16]
        stp     q18,q19,[x16,#32]
14:
        add     x15,x15,x6
        add     x16,x16,#.LPW_BlockBytes
        subs    x13,x13,#1
        b.ne    .Lpw_left_loop

.Lpw_after_filter:
        add     x12,x12,#1
        ldr     x0,[sp,#40]                // output channel blocks
        cmp     x12,x0
        blt     .Lpw_filter_loop
.Lpw_exit:
        add     sp,sp,#96
        ret

        .end
